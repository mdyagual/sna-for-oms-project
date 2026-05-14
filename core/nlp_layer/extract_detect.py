import json
import re
import asyncio
import logging

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.utils import HfHubHTTPError
from core.config import token, model, prompt, demo
from core.schemas import LogResponse
from data.db_storage import check_cache, save_in_cache, get_all_cache
from core.schemas import Taxonomy

logger = logging.getLogger(__name__)
client = AsyncInferenceClient(token=token.HF_TOKEN)


def safe_json_load(content):
    content_str = str(content).strip()
    d_log = {}
    try:
        return json.loads(content_str)
    except json.JSONDecodeError as err:
        d_log = {"error": str(err), "raw_output": content_str[:500]}

    # Regex fallback to find the first JSON block
    match = re.search(r"(\{.*\}|\[.*\])", content_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as err:
            d_log = {"error": str(err), "raw_output": content_str[:500]}

    return d_log


async def async_retry_call(fn, retries=5):
    base_wait_time = 5
    for i in range(retries):
        try:
            return await fn()
        except HfHubHTTPError as e:
            # Check specifically for a 429 Rate Limit error
            if e.response.status_code == 429:
                wait_time = base_wait_time * (2 ** i) # 10s, 20s, 40s, 80s...
                logger.warning("Rate limit hit. Waiting %d  before retry...", (wait_time))
                await asyncio.sleep(wait_time)
            else:
                # If it's a 500 or 503 error (Server overloaded), also wait
                logger.warning("Server error %d . Retrying in 5s...", (e.response.status_code))
                await asyncio.sleep(5) 
                
            if i == retries - 1:
                logger.error("[Failed] Maximum retries reached.")
                raise e
        except Exception as e:
            logger.warning("[Network Error] %s... Retrying...", (str(e)[:50]))
            await asyncio.sleep(5)
            if i == retries - 1:
                raise e


async def call_classif(or_text, sel_model):
    return (await client.text_classification(text=or_text, model=sel_model))[0]


async def call_complet(sel_model, l_info=None,  max_tokens=150, temperature=0.1):
    if l_info is None:
        l_info = []
    response = await client.chat_completion(
        model=sel_model,
        messages=l_info,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# 4. STAGE 2: ROUTING & CLASSIFICATION - TO CHECK
async def get_master_taxonomy(original_text):
    prompt_c = f"""Analyze the following text: '{original_text}'. 
        Classify it strictly into EXACTLY ONE of these categories: 
        {Taxonomy.EXTREMIST.name}, {Taxonomy.RADICALISM.name}, {Taxonomy.THREAT.name}, {Taxonomy.VIOLATED_SCALED.name}, or {Taxonomy.NEUTRAL.name}.
        
        OUTPUT ONLY VALID JSON:
        {{
        "taxonomy_classification": "CATEGORY",
        "confidence_score": 0.85,
        "confidence_reasoning": "Brief reason why"
        }}""" 
    
    try:
        res_b, res_a, res_text_c = await asyncio.gather(
            async_retry_call(lambda: call_classif(original_text, model.GROUP_B)),
            async_retry_call(lambda: call_classif(original_text, model.GROUP_A)),
            async_retry_call(lambda: call_complet(model.GROUP_C, [{"role": "user", "content": prompt_c}]))
        )
        
        score_b = res_b.score
        label_b = res_b.label # Usually 'hate' or 'nothate'
        
        
        score_a = res_a.score
        label_a = res_a.label 

        data_c = safe_json_load(res_text_c)       
        label_c = data_c.get("taxonomy_classification", Taxonomy.NEUTRAL.name)

        try:
            score_c = float(data_c.get("confidence_score", 0.0))
        except (ValueError, TypeError):
            score_c = 0.0
        # ==========================================
        # THE ARBITRATOR HIERARCHY (Highest to Lowest)
        # ==========================================
        
        # TIER 1: Critical Threats (Llama C output)
        if label_c in [Taxonomy.EXTREMIST.name, Taxonomy.RADICALISM.name, Taxonomy.THREAT.name, Taxonomy.VIOLATED_SCALED.name] and score_c >= 0.70:
            return {
                "taxonomy_category": label_c,
                "reason": f"Group C detected {label_c} with {round(score_c*100, 1)}% confidence.",
                "confidence_score": score_c 
            }
            
        # TIER 2: Hate Speech (RoBERTa B output)
        # Only accept if the model is highly confident it IS hate (> 75%)
        if label_b == "hate" and score_b > 0.75:
            return {
                "taxonomy_category": Taxonomy.HATE.name,
                "reason": f"Group B detected hate with {round(score_b*100, 1)}% confidence.",
                "confidence_score": score_b 
            }
            
        # TIER 3: Emotional/Sentimental (RoBERTa A output)
        if label_a == "negative" and score_a > 0.60:
            return {
                "taxonomy_category": Taxonomy.EMOTIONAL.name,
                "reason": f"Group A detected negative emotion with {round(score_a*100, 1)}% confidence.",
                "confidence_score": score_a 
            }
        elif label_a in ["positive", "neutral"]:
            return {
                "taxonomy_category": Taxonomy.SENTIMENTAL.name,
                "reason": f"Group A detected {label_a} sentiment with {round(score_a*100, 1)}% confidence.",
                "confidence_score": score_a 
            }
            
        # Fallback
        return {
            "taxonomy_category": Taxonomy.SENTIMENTAL.name,
            "reason": "Fallback: Did not trigger higher threat thresholds.",
            "confidence_score": -1 
        }

    except Exception as e:
        return {"taxonomy_category": "ERROR", "reason": f"API Error: {str(e)}"}

# MAIN BATCH PROCESSOR
async def process_entry(text):
    logger.info("Stage 1: Extraction of entities and relationships...")
    s1_res = await async_retry_call(lambda: call_complet(model.EXTR_ENT_REL, [{"role": "system", "content": prompt.EXTR_ENT_REL},{"role": "user", "content": text}], max_tokens=1000))
    graph = safe_json_load(s1_res)

    logger.info("Stage 2: Taxonomy Arbitration...")
    taxonomy_result = await get_master_taxonomy(text)

    # Safely extract the arrays from Llama's JSON, defaulting to empty lists if it failed
    entities_list = graph.get("entities", [])
    relations_list = graph.get("relations", [])

    return {
      
        "log_text": text,
        "taxonomy_category": taxonomy_result.get("taxonomy_category", "ERROR"),
        "reasoning": taxonomy_result.get("reason", ""),
        "confidence": taxonomy_result.get("confidence_score", -1),
        "entities": entities_list,
        "relations": relations_list

    }

# TEST BATCH PROCESSOR
async def process_entry_file(): 
    f = open(demo.FILE_TEST, 'r', encoding='utf-8')
    lines = [line.strip() for line in f if line != '\n']
    f.close()
    
    for i, text in enumerate(lines):
        logger.info("Stage 1: Extraction of entities and relationships from report #%d...", (i+1))
        if not await check_cache(text):
            s1_res = await async_retry_call(lambda: call_complet(model.EXTR_ENT_REL, [{"role": "system", "content": prompt.EXTR_ENT_REL},{"role": "user", "content": text}], max_tokens=1000))
            graph = safe_json_load(s1_res)

            logger.info("Stage 2: Taxonomy Arbitration from report #%d...", (i+1))
            taxonomy_result = await get_master_taxonomy(text)

        # Safely extract the arrays from Llama's JSON, defaulting to empty lists if it failed
            entities_list = graph.get("entities", [])
            relations_list = graph.get("relations", [])

            r = {                
                "log_text": text,
                "taxonomy_category": taxonomy_result.get("taxonomy_category", "ERROR"),
                "reasoning": taxonomy_result.get("reason", ""),
                "confidence": taxonomy_result.get("confidence_score", -1),
                "entities": entities_list,
                "relations": relations_list
                }
            
            parsed_result = LogResponse(**r).model_dump()

            await save_in_cache(parsed_result)
    
    return await get_all_cache()
       
   