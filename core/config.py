import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass(frozen=True)
class Token:
    HF_TOKEN: str = os.getenv("HF_TOKEN","")

@dataclass(frozen=True)
class Neo4J:
    USER: str = os.getenv("NEO4J_USER","")
    PASS: str = os.getenv("NEO4J_PASS","")
    URL: str = os.getenv("NEO4J_URL","")

@dataclass(frozen=True)
class LLMConfig:   

    ## Extractor model
    EXTR_ENT_REL: str = "meta-llama/Llama-3.1-8B-Instruct"

    ## DETECTION TAXONOMY
    # Sentimental and Emotional
    GROUP_A: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # Hate and Threat
    GROUP_B: str = "facebook/roberta-hate-speech-dynabench-r4-target"
    # Extremist, Radicalism and Violated-scaled
    GROUP_C: str = "meta-llama/Meta-Llama-3-8B-Instruct"

@dataclass(frozen=True)
class Prompt:
    EXTR_ENT_REL = """
    You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA).
    Your task is to extract a base graph (Entities and Relationships) from the text.

    RULES:
    - Extract ONLY socially relevant entities (Person, Group, Instruction, Location, Target).
    - Ensure entity are capitalized and stripped of titles or adjectives (e.g., use 'Miller' instead of 'Inmate Miller')
    - EXCLUDE the reporting officer or system administrator from the extracted network. Focus only on the relationships between the subjects of the report (e.g., inmates, visitors, external contacts)
    - Resolve pronouns to their actual entity names based on the context. (DO NOT use generic terms like "Author").
    - The entity 'label' MUST be exactly one of the following, strictly capitalized: 
  [Inmate, Staff, ExternalContact, Group, Contraband, Weapon, Currency, InternalLocation, ExternalLocation, Event, DigitalAsset]. 
  DO NOT invent new labels.
    - Always extract relationships when there is meaningful interaction or implied action.
    - The 'predicate' must be a specific, normalized present-tense verb (e.g., "instructs", "relocates", "threatens").
    - Connect people to objects and locations using normalized present-tense verbs (e.g., {"subject": "Miller", "predicate": "possesses", "object": "shank"}).
    CRITICAL: Every entity you extract MUST be involved in at least one relationship. Do not leave "floating" entities.

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
graph_base = Neo4J()
model = LLMConfig()
prompt = Prompt()
