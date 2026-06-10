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

MAX_CONCURRENT_LOGS = 3 
semaphore = asyncio.Semaphore(MAX_CONCURRENT_LOGS)

def safe_json_load(content):
    content_str = str(content).strip()
    content_str = re.sub(r'```json', '', content_str)
    content_str = re.sub(r'```', '', content_str)
    content_str = content_str.strip()
    d_log = {}
    try:
        return json.loads(content_str)
    except json.JSONDecodeError:
        # 2. Aggressive Regex: Find everything from the FIRST { to the LAST }
        match = re.search(r"\{.*\}", content_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as err:
                d_log = {"error": str(err), "raw_output": content_str[:500]}

    return d_log

def heal_and_deduplicate_graph(entities_list, relations_list):
    """Prunes bad nodes, removes duplicate edges, and fixes missing orphans."""
    
    # 1. Prune Generic Nodes (Flagged by Pydantic)
    valid_entities = [e for e in entities_list if e.get("entity") != "DROP_ME_GENERIC"]
    
    # Extract all cleanly declared node names
    declared_entities = {e.get("entity") for e in valid_entities}
    
    seen_edges = set()
    unique_relations = []
    
    # 2. Prune bad edges and deduplicate
    for r in relations_list:
        subj = r.get("subject", "")
        pred = r.get("predicate", "")
        obj = r.get("object", "")
        
        # SILENT PRUNER: Drop edge if it connects to a generic/garbage node
        if subj == "DROP_ME_GENERIC" or obj == "DROP_ME_GENERIC":
            continue
            
        # Create a unique signature for the edge
        edge_sig = (subj, pred, obj)
        
        if edge_sig not in seen_edges:
            seen_edges.add(edge_sig)
            unique_relations.append(r)
            
            # 3. Heal Orphans (Nodes in relations but missing from entities)
            if subj and subj not in declared_entities:
                valid_entities.append({"entity": subj, "label": "Unknown"})
                declared_entities.add(subj)
                
            if obj and obj not in declared_entities:
                valid_entities.append({"entity": obj, "label": "Unknown"})
                declared_entities.add(obj)
                
    return valid_entities, unique_relations
            

def normalize_confidence(score: float) -> float:
    """Ensure confidence is always 0-1. Handle Llama's percentage returns."""
    try:
        score = float(score)
        # If it looks like a percentage (>1), treat as out-of-range
        if score > 1.0:
            # Option A: Assume it's a percentage and divide by 100
            if score <= 100:
                return score / 100.0
            # Option B: Clamp to 1.0 (treat as "super confident")
            else:
                return 1.0
        # Option C: Clamp negative to 0
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0  # Invalid → assume no confidence
    
async def process_single_log_safely(i, text):
    """Wrapper to process a log while respecting the rate-limit bouncer."""
    async with semaphore:
        max_retries = 3
        attempt = 0
        success = False
        parsed_result = None

        while attempt < max_retries and not success:
            attempt += 1
            logger.info("Processing report #%d (Attempt %d/%d)...", (i+1), attempt, max_retries)
            
            try:
                # Stage 1: Extraction
                logger.info("Stage 1: Extraction of entities and relationships from report #%d...", (i+1))
                s1_res = await async_retry_call(lambda: get_nodes_edges(model.EXTR_ENT_REL, [{"role": "system", "content": prompt.EXTR_ENT_REL},{"role": "user", "content": text}], max_tokens=1000))
                graph = safe_json_load(s1_res)

                # Stage 2: Taxonomy
                logger.info("Stage 2: Taxonomy Arbitration from report #%d...", (i+1))
                taxonomy_result = await get_master_taxonomy(text)

                # Safety Check: Did the API fail and return SYSTEM_ERROR?
                if taxonomy_result.get("taxonomy_category") == Taxonomy.SYSTEM_ERROR.name:
                    logger.warning("Taxonomy API failed for report #%d. Reason: %s", (i+1), taxonomy_result.get('reason'))
                    
                    # If it's a 403 error, do NOT retry. Break immediately to avoid getting banned.
                    if "403" in str(taxonomy_result.get("reason")):
                        logger.error("FATAL 403 ERROR: Check Hugging Face Token Permissions. Halting retries for this log.")
                        break 

                    if attempt < max_retries:
                        logger.info("Waiting 10 seconds before retrying entire block...")
                        await asyncio.sleep(10)
                        continue # Send it back to the top of the while loop!
                    else:
                        logger.error("Maximum retries reached for report #%d. Saving as SYSTEM_ERROR.", (i+1))
                else:
                    # We got a valid taxonomy!
                    success = True 

                # Extract arrays safely
                entities_list = graph.get("entities", [])
                relations_list = graph.get("relations", [])
                
                # Heal the graph (Remove duplicates, fix orphans)
                entities_list, relations_list = heal_and_deduplicate_graph(entities_list, relations_list)

                r = {                
                    "log_text": text,
                    "taxonomy_category": taxonomy_result.get("taxonomy_category", Taxonomy.SYSTEM_ERROR.name),
                    "reasoning": taxonomy_result.get("reason", ""),
                    "confidence": taxonomy_result.get("confidence_score", -1),
                    "entities": entities_list,
                    "relations": relations_list
                }
                
                # STEP 1: Pydantic Validation 
                # (Standardizes names, strips adjectives, flags generics, enforces Ontology)
                cleaned_result = LogResponse(**r).model_dump(mode='json')
                
                # STEP 2: Topology Healing 
                # (Prunes the flagged generics, deduplicates edges, fixes missing orphans)
                final_ents, final_rels = heal_and_deduplicate_graph(
                    cleaned_result["entities"], 
                    cleaned_result["relations"]
                )
                
                # Re-assign the perfected topology back to the dictionary
                cleaned_result["entities"] = final_ents
                cleaned_result["relations"] = final_rels
                
                parsed_result = cleaned_result

                parsed_result = LogResponse(**r).model_dump(mode='json')
                
                # If we successfully parsed everything, break out of the retry loop
                if success:
                    break

            except Exception as e:
                logger.error("Unexpected error processing report #%d: %s", (i+1), str(e))
                if attempt < max_retries:
                    await asyncio.sleep(10)
                else:
                    break
        
        # Save whatever we ended up with (success or exhausted errors)
        if parsed_result:
            await save_in_cache(parsed_result)

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


async def get_nodes_edges(sel_model, l_info=None,  max_tokens=150, temperature=0.1):
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
            async_retry_call(lambda: get_nodes_edges(model.GROUP_C, [{"role": "user", "content": prompt_c}]))
        )
        
        score_b = res_b.score
        label_b = res_b.label 
        score_a = res_a.score
        label_a = res_a.label 

        data_c = safe_json_load(res_text_c)       
        label_c = data_c.get("taxonomy_classification", Taxonomy.NEUTRAL.name)

        try:
            score_c = normalize_confidence(float(data_c.get("confidence_score", 0.0)))
        except (ValueError, TypeError):
            score_c = 0.0

        # ==========================================
        # THE ARBITRATOR HIERARCHY
        # ==========================================
        
        # TIER 1: Extreme Physical Threats (Llama C)
        if score_c >= 0.70 and label_c in [Taxonomy.EXTREMIST.name, Taxonomy.RADICALISM.name, Taxonomy.THREAT.name, Taxonomy.VIOLATED_SCALED.name]:
            return {"taxonomy_category": label_c, "reason": f"Group C detected {label_c}.", "confidence_score": score_c}
            
        # TIER 2: Hate Speech (RoBERTa B)
        # This now catches hate speech even if Llama-3 was too "polite" to flag it!
        if label_b == "hate" and score_b > 0.75:
            return {"taxonomy_category": Taxonomy.HATE.name, "reason": "Group B detected hate.", "confidence_score": score_b}

        # TIER 3: The Neutral Shield (Llama C)
        # If it's not a threat and not hate, let's trust Llama if it says the log is boring.
        # This stops the Sentiment model below from turning everything into an "Emotional" alert.
        if score_c >= 0.70 and label_c == Taxonomy.NEUTRAL.name:
            return {"taxonomy_category": Taxonomy.NEUTRAL.name, "reason": "Group C confirmed Neutral.", "confidence_score": score_c}
            
        # TIER 4: Emotional/Sentimental (RoBERTa A)
        if label_a == "negative" and score_a > 0.60:
            return {"taxonomy_category": Taxonomy.EMOTIONAL.name, "reason": "Group A detected negative emotion.", "confidence_score": score_a}
        elif label_a in ["positive", "neutral"]:
            return {"taxonomy_category": Taxonomy.SENTIMENTAL.name, "reason": "Group A detected sentiment.", "confidence_score": score_a}
            
        return {"taxonomy_category": Taxonomy.SENTIMENTAL.name, "reason": "Fallback.", "confidence_score": -1}

    except Exception as e:
        return {"taxonomy_category": Taxonomy.SYSTEM_ERROR.name, "reason": f"API Error: {str(e)}"}

# MAIN BATCH PROCESSOR
async def process_entry(text):
    logger.info("Stage 1: Extraction of entities and relationships...")
    s1_res = await async_retry_call(lambda: get_nodes_edges(model.EXTR_ENT_REL, [{"role": "system", "content": prompt.EXTR_ENT_REL},{"role": "user", "content": text}], max_tokens=1000))
    graph = safe_json_load(s1_res)

    logger.info("Stage 2: Taxonomy Arbitration...")
    taxonomy_result = await get_master_taxonomy(text)

    # Safely extract the arrays from Llama's JSON, defaulting to empty lists if it failed
    entities_list = graph.get("entities", [])
    relations_list = graph.get("relations", [])

    r = {                
                "log_text": text,
                "taxonomy_category": taxonomy_result.get("taxonomy_category", Taxonomy.SYSTEM_ERROR.name),
                "reasoning": taxonomy_result.get("reason", ""),
                "confidence": taxonomy_result.get("confidence_score", -1),
                "entities": entities_list,
                "relations": relations_list
                }
    
    # 1. Keep only entities that are NOT flagged for dropping
    valid_entities = [e for e in r["entities"] if e.get("entity") != "Generic"]
    
    # 2. Get a list of surviving names
    surviving_entity_names = {e.get("entity") for e in valid_entities}
    
    # 3. Keep relations only if BOTH subject and object survived
    valid_relations = [
        rel for rel in r["relations"] 
        if rel.get("subject") in surviving_entity_names 
        and rel.get("object") in surviving_entity_names
    ]
    
    r["entities"] = valid_entities
    r["relations"] = valid_relations
            
    parsed_result = LogResponse(**r).model_dump(mode='json')

    return parsed_result

# TEST BATCH PROCESSOR
async def process_entry_file(): 
    f = open(demo.FILE_TEST, 'r', encoding='utf-8')
    lines = [line.strip() for line in f if line != '\n']
    f.close()
    
    tasks = []
    for i, text in enumerate(lines):
        # Await the cache check first so we don't spawn tasks for already-completed logs
        if not await check_cache(text):
            # Append the safe, throttled function to our task list
            tasks.append(process_single_log_safely(i, text))
            
    # Run all tasks concurrently! (The semaphore will ensure only 3 run at a exact same time)
    if tasks:
        logger.info("Starting concurrent processing for %d new logs...", len(tasks))
        await asyncio.gather(*tasks)
    else:
        logger.info("All logs are already processed and cached.")
    
    return await get_all_cache()
       
   