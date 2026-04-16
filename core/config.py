import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass(frozen=True)
class Token:
    HF_TOKEN: str = os.getenv("HF_TOKEN","")

@dataclass(frozen=True)
class LLMConfig:   

    ## Extractor model
    EXTR: str = "meta-llama/Llama-3.1-8B-Instruct"

    ## DETECTION TAXONOMY
    # Sentimental and Emotional
    GROUP_A: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # Hate and Threat
    GROUP_B: str = "facebook/roberta-hate-speech-dynabench-r4-target"
    # Extremist, Radicalism and Violated-scaled
    GROUP_C: str = "meta-llama/Meta-Llama-3-8B-Instruct"

@dataclass
class Prompt:
    EXTR = """
    You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA).
    Your task is to extract a base graph (Entities and Relationships) from the text.

    RULES:
    - Extract ONLY socially relevant entities (Person, Group, Institution, Location, Target).
    - EXCLUDE the reporting officer or system administrator from the extracted network. Focus only on the relationships between the subjects of the report (e.g., inmates, visitors, external contacts)
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
    }"""

token = Token()
model = LLMConfig()
prompt = Prompt()
