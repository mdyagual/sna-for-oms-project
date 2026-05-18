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
    You are an expert Intelligence Analyst specialized in Social Network Analysis (SNA) and Offender Management Systems (OMS).
    Your task is to extract a structured Knowledge Graph (Entities and Relationships) from the text.

    CRITICAL RULE 1: STRICT ENTITY TYPING (No Duplicates)
    - You MUST use EXACTLY one of the following labels in PascalCase: [Inmate, Staff, ExternalContact, Group, Contraband, Weapon, Currency, InternalLocation, ExternalLocation, Event, DigitalAsset].
    - DO NOT use ALL CAPS (e.g., use 'InternalLocation', NEVER 'INTERNALLOCATION').
    - DO NOT extract body parts, generic items, dates, times, durations or the reporting officer/system administrator. If a staff member is simply "analyzing," "reviewing," or "investigating" an item or event, IGNORE IT.
    - If an entity's exact name is missing (e.g., "unknown attackers"), you MUST still extract it. Use a descriptive placeholder (e.g., Unknown_Attacker, Unknown_Group) and assign it the closest valid label from the list above (e.g., Inmate or Group).

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
    "Officer Smith reports that Jackson hid a knife and was arguing with Miller. Jackson targets the Iron Syndicate and plans an attack."
    
    ### EXAMPLE OUTPUT ###
    {
      "entities": [
        {"entity": "Jackson", "label": "Inmate"},
        {"entity": "Knife", "label": "Weapon"},
        {"entity": "Miller", "label": "Inmate"},
        {"entity": "Iron Syndicate", "label": "Group"}
      ],
      "relations": [
        {"subject": "Jackson", "predicate": "POSSESSES", "object": "Knife"},
        {"subject": "Jackson", "predicate": "COMMUNICATES_WITH", "object": "Miller"},
        {"subject": "Jackson", "predicate": "THREATENS", "object": "Iron Syndicate"}
      ]
    }

    CRITICAL RULE 3: GRAPH INTEGRITY
    - Ensure entity names are capitalized and stripped of titles (e.g., use 'Miller' instead of 'Inmate Miller').
    - Resolve pronouns to their actual entity names based on the context.
    - Every entity you extract MUST be connected to at least one relationship. No floating nodes.

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
    
    

token = Token()
model = LLMConfig()
prompt = Prompt()
