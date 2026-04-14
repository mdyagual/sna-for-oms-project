import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

## Extractor model
MODEL_EXTR = "meta-llama/Llama-3.1-8B-Instruct"

## DETECTION TAXONOMY
# Sentimental and Emotional
MODEL_GROUP_A = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Hate and Threat
MODEL_GROUP_B = "facebook/roberta-hate-speech-dynabench-r4-target"
# Extremist, Radicalism and Violated-scaled
MODEL_GROUP_C = "meta-llama/Meta-Llama-3-8B-Instruct"

# 2. MAIN PROMPT (Base Graph Extraction)
MAIN_PROMPT = """
You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA).
Your task is to extract a base graph (Entities and Relationships) from the text.

RULES:
- Extract ONLY socially relevant entities (Person, Group, Institution, Location, Target).
- Resolve pronouns to their actual entity names based on the context. (DO NOT use generic terms like "Author").
- Extract relationships ONLY if there is meaningful interaction or implied action.
- The 'predicate' must be a specific, normalized verb (e.g., "instructs", "relocates", "threatens").

CRITICAL: OUTPUT ONLY VALID JSON. START DIRECTLY WITH {:
{
  "entities": [
    {"entity": "Name", "label": "Type"}
  ],
  "relations": [
    {"subject": "Name", "predicate": "Verb", "object": "Name"}
  ]
}
"""


# 3. JSON Parser
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


def retry_call(fn, retries=3):
    for i in range(retries):
        try:
            return fn()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2.5 * (i + 1))


def call_classif(or_text, sel_model):
    return client.text_classification(text=or_text, model=sel_model)[0]


def call_complet(sel_model, l_info=[],  max_tokens=150, temperature=0.1):
    response = client.chat_completion(
        model=sel_model,
        messages=l_info,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# 4. STAGE 2: ROUTING & CLASSIFICATION
def get_master_taxonomy(original_text):
    """
    Arbitrator function: Runs the text through the models and applies a hierarchy
    to select EXACTLY ONE taxonomy category for the entire log.
    """
    try:
        # --- 1. Query Group B (Hate Speech) ---
        res_b = retry_call(lambda: call_classif(original_text, MODEL_GROUP_B))
        score_b = res_b.score
        label_b = res_b.label # Usually 'hate' or 'nothate'
        
        # --- 2. Query Group A (Sentiment) ---
        res_a = retry_call(lambda: call_classif(original_text, MODEL_GROUP_A))
        score_a = res_a.score
        label_a = res_a.label # Usually 'positive', 'neutral', 'negative'

        # --- 3. Query Group C (Llama-3 Reasoner for High-Tier Threats) ---
        prompt_c = f"""Analyze the following text: '{original_text}'. 
        Classify it strictly into EXACTLY ONE of these categories: 
        EXTREMIST, RADICALISM, THREAT, VIOLATED-SCALED, or NEUTRAL.
        
        OUTPUT ONLY VALID JSON:
        {{
          "taxonomy_classification": "CATEGORY",
          "confidence_reasoning": "Brief reason why"
        }}"""
        res_text_c = retry_call(lambda: call_complet(MODEL_GROUP_C, [{"role": "user", "content": prompt_c}]))
        data_c = safe_json_load(res_text_c)
        label_c = data_c.get("taxonomy_classification", "NEUTRAL")

        # ==========================================
        # THE ARBITRATOR HIERARCHY (Highest to Lowest)
        # ==========================================
        
        # TIER 1: Critical Threats (Llama C output)
        if label_c in ["EXTREMIST", "RADICALISM", "THREAT", "VIOLATED-SCALED"]:
            return {
                "taxonomy_category": label_c,
                "reason": data_c.get("confidence_reasoning", "Detected by Group C")
            }
            
        # TIER 2: Hate Speech (RoBERTa B output)
        # Only accept if the model is highly confident it IS hate (> 75%)
        if label_b == "hate" and score_b > 0.75:
            return {
                "taxonomy_category": "HATE",
                "reason": f"Group B detected hate with {round(score_b*100, 1)}% confidence"
            }
            
        # TIER 3: Emotional/Sentimental (RoBERTa A output)
        if label_a == "negative" and score_a > 0.60:
            return {
                "taxonomy_category": "EMOTIONAL",
                "reason": f"Group A detected negative emotion with {round(score_a*100, 1)}% confidence"
            }
        elif label_a in ["positive", "neutral"]:
            return {
                "taxonomy_category": "SENTIMENTAL",
                "reason": f"Group A detected {label_a} sentiment with {round(score_a*100, 1)}% confidence"
            }
            
        # Fallback
        return {
            "taxonomy_category": "SENTIMENTAL",
            "reason": "Fallback: Did not trigger higher threat thresholds."
        }

    except Exception as e:
        return {"taxonomy_category": "ERROR", "reason": f"API Error: {str(e)}"}

# 5. MAIN BATCH PROCESSOR
def process_entry(idx, text):
    print(f"[{idx}] Stage 1: Extraction (Graph)...")
    s1_res = retry_call(lambda: call_complet(MODEL_EXTR, [{"role": "system", "content": MAIN_PROMPT},{"role": "user", "content": text}], max_tokens=1000))
    graph = safe_json_load(s1_res)

    print(f"[{idx}] Stage 2: Taxonomy Arbitration...")
    taxonomy_result = get_master_taxonomy(text)

    # Safely extract the arrays from Llama's JSON, defaulting to empty lists if it failed
    entities_list = graph.get("entities", [])
    relations_list = graph.get("relations", [])

    # Build the flattened, final JSON structure
    final_output = {
        "id": idx,  # Highly recommended to keep for database tracking
        "log_text": text,
        "taxonomy_category": taxonomy_result.get("taxonomy_category", "ERROR"),
        "reasoning": taxonomy_result.get("reason", ""),
        "entities": entities_list,
        "relations": relations_list
    }

    return final_output


### THE MAIN FUNCTION

def run_batch_test():
    input_file = "./data/logs/examples.txt"
    output_file = "./data/jsons/extraction_results.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if len(l.strip()) > 5]

        print(f"Starting Two-Stage Pipeline for {len(lines)} examples...\n")
        final_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(process_entry, i + 1, text)
                for i, text in enumerate(lines)
            ]
            for future in as_completed(futures):
                final_results.append(future.result())

        final_results.sort(key=lambda x: x["id"])

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"\nSuccess! Results saved to {output_file}")

    except FileNotFoundError:
        print("Error: examples.txt not found.")


###### MAIN

if __name__ == "__main__":
    run_batch_test()
