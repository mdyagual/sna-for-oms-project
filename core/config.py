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
    URL: str = os.getenv("NEO4J_URL_PAPER","")

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
    GROUP_C: str = "meta-llama/Llama-3.1-8B-Instruct"

@dataclass(frozen=True)
class Prompt:
    EXTR_ENT_REL = """
    You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA) and Offender Management Systems (OMS).
    Your task is to extract a structured Knowledge Graph (Entities and Relationships) from the text.

   CRITICAL RULE 1: STRICT ENTITY TYPING AND NAMING (No Abstracts, No Duplicates)
    - You MUST use EXACTLY one of the following labels in PascalCase: [Inmate, Staff, ExternalContact, Group, Contraband, Weapon, Currency, InternalLocation, ExternalLocation, Event, DigitalAsset].
    - NEGATIVE CONSTRAINT: DO NOT use underscores or snake_case for entity names. Use spaces (e.g., "Gang Leader" NOT "Gang_Leader").
    - DO NOT use ALL CAPS.
    - NEGATIVE CONSTRAINT: DO NOT extract abstract concepts, mental states, emotions, or structural concepts (e.g., NEVER extract 'Trauma', 'Paranoia', 'Depression', 'Gang Factions', or 'Gang Structure'). Only extract tangible, physical entities or specific named groups.
    - NEGATIVE CONSTRAINT: DO NOT extract generic body parts, dates, times, or the reporting system administrator.
    - PLACEHOLDER RULE: If a person's exact name is missing, use "Unknown Inmate" or "Unknown Staff". If the text mentions MULTIPLE unnamed people (e.g., "two inmates"), group them into a single node called "Unknown Group" with the label "Group". DO NOT use plurals like "Inmates". 
    - DO NOT invent variations like "Inmate1", "Inmate2", "Officers", "Corrupt Officer", or "External".
    
    CRITICAL RULE 2: STRICT PREDICATE TRANSLATION (No Synonyms Allowed)
    - You are FORBIDDEN from inventing your own verbs OR USING the exact verbs from the text (e.g., NEVER output "TARGETS" or "PLANS").
    - You MUST TRANSLATE the action into EXACTLY ONE of these 10 standardized buckets (UPPERCASE_WITH_UNDERSCORES):
        * ASSAULTS (Translate: physical violence, attacks, stabs, fights, injures)
        * THREATENS (Translate: verbal/implied aggression, intimidates, confronts, targets)
        * COMMUNICATES_WITH (Translate: talking, speaks to, argues, discusses, meets, reporting to staff, reports)
        * AFFILIATED_WITH (Translate: loyalty, gang ties, joins, leads, conspires, part of, plans)
        * POSSESSES (Translate: holding/making, acquires, hides, smuggles, manufactures)
        * EXCHANGES (Translate: 1-to-1 peer transactions, pays, delivers, receives, swapping items, trades with, barters)
        * DISTRIBUTES (Translate: 1-to-many supply chain, acts as supplier, kingpin, handing out contraband to multiple people, traffics, circulates, peddles)
        * LOCATED_IN (Translate: movement/presence, enters, visits, resides in, transferred to)
        * CONTAINS (Translate: an inanimate object or location holding another object. e.g., "Package contains Drugs")
        * ASSOCIATED_WITH (Translate: generic, non-violent interactions: analyzes, reviews, investigates, witnesses, knows)
        * SELF_HARM (Translate: any self-directed violence or injury. Examples: attempted suicide, self-mutilation, harming oneself.)

    - If the text uses passive voice (e.g., "Jackson was attacked by Miller"), you MUST REVERSE the subject/object to keep the active predicate: {"subject": "Miller", "predicate": "ASSAULTS", "object": "Jackson"}.
    
    ### EXAMPLE INPUT ###
    "Officer Smith reports that the senior inmate Jackson hid a knife and was arguing with an unidentified suspect. Jackson targets the rival Iron Syndicate and plans an attack."
    
    ### EXAMPLE OUTPUT ###
    {
      "entities": [
        {"entity": "Smith", "label": "Staff"},
        {"entity": "Jackson", "label": "Inmate"},
        {"entity": "Knife", "label": "Weapon"},
        {"entity": "Unknown Inmate", "label": "Inmate"},
        {"entity": "Iron Syndicate", "label": "Group"}
      ],
      "relations": [
        {"subject": "Smith", "predicate": "ASSOCIATED_WITH", "object": "Jackson"},
        {"subject": "Jackson", "predicate": "POSSESSES", "object": "Knife"},
        {"subject": "Jackson", "predicate": "COMMUNICATES_WITH", "object": "Unknown Inmate"},
        {"subject": "Jackson", "predicate": "THREATENS", "object": "Iron Syndicate"}
      ]
    }
    CRITICAL RULE 3: GRAPH INTEGRITY
    - Ensure entity names are capitalized and stripped of titles (e.g., use 'Miller' instead of 'Inmate Miller').
    - Resolve pronouns to their actual entity names based on the context.
    - Every entity you extract MUST BE CONNECTED to at least one relationship. No floating nodes.

    CRITICAL RULE 4: JSON OUTPUT ONLY
    - OUTPUT ONLY VALID JSON. START DIRECTLY WITH {:
    {
    "entities": [
        {"entity": "Name", "label": "Type"}
    ],
    "relations": [
        {"subject": "Name", "predicate": "MUST BE EXACTLY ONE OF: ASSAULTS, THREATENS, COMMUNICATES_WITH, AFFILIATED_WITH, POSSESSES, EXCHANGES, DISTRIBUTES, LOCATED_IN, CONTAINS, ASSOCIATED_WITH", "object": "Name"}
    ]
    }"""
    
@dataclass(frozen=True)
class Demo:
    FILE_TEST = "./data/files/data_for_test.txt"
    FILE_PAPER = "./data/files/data_for_paper.txt"
    DB_CACHE = "./data/extract_detect_cache.json"
    DB_PAPER ="./data/extract_detect_paper.json"


token = Token()
graph_base = Neo4J()
model = LLMConfig()
prompt = Prompt()
demo = Demo()
