from enum import Enum
from typing import List
from datetime import datetime, timezone
from dataclasses import dataclass
import uuid
import re
from pydantic import BaseModel, Field, field_validator, model_validator

def generate_uuid():
    return str(uuid.uuid4())

def generate_timestamp():
    # Always use UTC timezone for server logs!
    return datetime.now(timezone.utc).isoformat()

@dataclass(frozen=True)
class PredicateMap:
    RAW_HEALING_GROUPS = {
        "ASSAULTS": ["ASSAULTED_BY"],
        "AFFILIATED_WITH": ["PLANS", "PLANNED", "DEFENDS", "DEFENDED", "FACILITATES", "FACILITATED", "HELPS", "HELPED", "PARTICIPATES_IN", "PARTICIPATED_IN", "COORDINATES", "COORDINATED", "SUPPORTS", "SUPPORTED"],
        "COMMUNICATES_WITH": ["REPORTS","REPORTED", "REPORTED_TO", "REPORTS_TO", "NOTIFIED", "NOTIFIES", "DISCUSSED", "DISCUSSES", "REQUESTS", "REQUESTED", "EXPRESSES", "EXPRESSED", "COMMUNICATE_WITH", "CONSULTS", "CONSULTED"],
        "ASSOCIATED_WITH": ["WITNESSED", "ANALYZES", "INVESTIGATES", "DESTROYED_IN", "TREATED_WITH", "REQUIRES", "WANTS", "WANTED", "UNDERGOES", "UNDERWENT", "ISOLATED_FROM", "SEPARATED_FROM", "OPERATED_BY", "CONTROLLED_BY", 
                            "INITIATES", "INITIATED", "CONSIDERS", "CONSIDERED", "SUFFERED", "SUFFERS", "DIAGNOSED_WITH", "DIAGNOSED", "DOCUMENTS", "DOCUMENTED", "WILLING_TO", "EXPERIENCES", "NEUTRALIZES", "NEUTRALIZED", 
                            "ACTIVATES", "ACTIVATED", "FILES", "FILED", "FILED_CHARGES", "SUSTAINS", "SUSTAINED" ],
        "EXCHANGES": ["RECEIVES", "RECEIVED", "FUNDS", "FUNDED", "FINANCES", "FINANCED", "PAYS", "PAID"],
        "THREATENS": ["TARGETS", "TARGETED_BY", "ASSIGNED_TO_ELIMINATE", "ORDERED_TO_ELIMINATE", "ASSIGNED_TO_KILL", "ORDERED_TO_KILL"],
        "SELF_HARM": ["ATTEMPTED_SUICIDE", "SUICIDE_ATTEMPT", "CUTS_SELF"],
        "LOCATED_IN": ["ASSIGNED_TO", "TRANSFERRED_TO", "MOVED_TO", "HOUSED_IN", "MANUFACTURED_IN", "MADE_IN", "ASSIGNED"],
        "POSSESSES": ["POSESSES", "POSSESES", "OWNS", "OWES", "ACQUIRES", "USES", "USED", "UTILIZES", "PRODUCES", "PRODUCED", "MAKES", "MADE", "CREATES", "CREATED"]
    }

    COMPILED_HEALING_MAP = {
        word: target 
        for target, words in RAW_HEALING_GROUPS.items() 
        for word in words
    }

@dataclass(frozen=True)
class EntityMap:
    RAW_LABEL_GROUPS = {
        "Inmate": ["Person", "Prisoner", "Convict", "Individual", "Offender"],
        "Group": ["Gang", "Organization", "Syndicate", "Faction", "Cartel"],
        "Event": ["Incident", "Meeting", "Altercation", "Riot", "Trial"],
        "Staff": ["Guard", "Employee", "Warden", "Personnel"],
        "Contraband": ["Drugs", "Narcotics", "Phone", "Cell Phone", "Contraband", "Prescription Narcotics", "Commissary", "Commissary Items", "Commissary Goods"],
        "Weapon": ["SharpObject", "Shiv", "Shank", "Blade"],
        "InternalLocation": ["Location", "Cell", "Area", "Yard", "Basketball Court", "Block", "Wing", "Tier", "Unit", "Segregation", "Medical", "Kitchen"]    
        }

    COMPILED_LABEL_MAP = {
        word: target 
        for target, words in RAW_LABEL_GROUPS.items() 
        for word in words
    }

@dataclass(frozen=True)
class NameResolutionMap:
    """Consolidates messy LLM entity names into clean, universal graph nodes."""
    RAW_NAME_GROUPS = {
        "Unknown": [
            # Base singular actors
            "Inmate", "Officer", "Guard", "Psychiatric", "Psychiatric Patient",
            "Gang Leader", "Leader", "Suspect", "Victim", "Attacker", "Target",
            
            # Numeric / Formatted LLM Hallucinations
            "Inmate1", "Inmate2", "Inmate 1", "Inmate 2", "UnknownInmate", 
            "UnknownAttacker", "UnknownTarget", "UnknownStaff", "Gang Lieutenant", "Lieutenant"
            
            # Descriptive Hallucinations
            "Hispanic Inmate", "Black Inmate", "White Inmate",
            "Unknown Attacker", "Unknown Target", "Unknown Inmate", 
            "Corrupt Officer", "Corrupt Staff Member", "Unknown Staff Member", 
            
            # Unknown Groups / Specialty Staff
            "Unknown Group", "Unidentified Group", "Medical Staff", "Medical"
        ],
        "DROP_ME_GENERIC": [
            # Base Generic Drops
            "Generic", "Gang", "Group", "Coordinated Gang", "Cartel", "Syndicate", 
            "Inmates", "Officers", "Staff", "Staff Member", "Guards", "Factions", 
            "Targets", "Minority", "Unknowns", "Individuals", "Security Presence", "Internallocation",
            
            # Category Words (Too broad)
            "Weapon", "Game", "Contraband", "Contraband Weapons", "Two Guards", "Prison Administration",
            "Ringleader", "Informant", "Patient", "Sleeper Agent", "Assailant", 
            "Weaker Inmate", "Younger Inmate",
            # Abstract Institutional Concepts
            "Violation", "Violations", "Code Of Conduct", "Inmate Code Of Conduct Violations", 
            "Violation Report", "Protective Custody", "Medical Observation", "Segregation", 
            "Escalating Violent Institutional Behavior", "Negative Institutional Contribution",
            "Investigation", "Formal Discipline", "Witness", "Witnesses","Trauma", "Anti-Extremism Philosophy"
            
            # Physical Abstractions
            "Bruising", "Intelligence", "Multiple Sources", "Ideology", "Blood Evidence",
            "Targeted Harassment Campaign", "Territorial Boundaries", 
            "Gambling", "Gambling Operations", "Card Games", "Extremist Group"
        ],
        "External Contact": ["External", "Civilian", "Outside Contact"],
        "Vipers": ["The Vipers", "Viper Gang", "Vipers Gang"],
        "Ghost Cartel": ["The Ghost Cartel", "Ghost Cartel Gang"],
        "Federal Bureau Of Investigation": ["FBI", "Carver Federal Bureau Of Investigation"]
      
    }

    COMPILED_NAME_MAP = {
        word.lower(): target 
        for target, words in RAW_NAME_GROUPS.items() 
        for word in words
    }


#The taxonomy defined for the project with their severity to consider
class Taxonomy(str, Enum):
    HATE = "HATE"
    EMOTIONAL = "EMOTIONAL"
    SENTIMENTAL = "SENTIMENTAL"
    EXTREMIST = "EXTREMIST"
    RADICALISM = "RADICALISM"
    THREAT = "THREAT"
    VIOLATED_SCALED = "VIOLATED_SCALED"
    NEUTRAL = "NEUTRAL"
    SYSTEM_ERROR = "SYSTEM_ERROR"

    @property
    def severity(self) -> int:
        """Dynamically returns the integer severity score for math calculations."""
        severity_map = {
            self.HATE: 3,
            self.EMOTIONAL: 1,
            self.SENTIMENTAL: 1,
            self.EXTREMIST: 5,
            self.RADICALISM: 5,
            self.THREAT: 4,
            self.VIOLATED_SCALED: 2,
            self.NEUTRAL: 1,
            self.SYSTEM_ERROR:0
        }
        return severity_map[self]

class EntityOntology(str, Enum):
    INMATE = "Inmate"
    STAFF = "Staff"
    EXTERNAL_CONTACT = "ExternalContact"
    GROUP = "Group"
    CONTRABAND = "Contraband"
    WEAPON = "Weapon"
    CURRENCY = "Currency"
    INTERNAL_LOCATION = "InternalLocation"
    EXTERNAL_LOCATION = "ExternalLocation"
    EVENT = "Event"
    DIGITAL_ASSET = "DigitalAsset"
    UNKNOWN = "Unknown"

class PredicateOntology(str, Enum):
    ASSAULTS = "ASSAULTS"
    THREATENS = "THREATENS"
    COMMUNICATES_WITH = "COMMUNICATES_WITH"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    POSSESSES = "POSSESSES"
    EXCHANGES = "EXCHANGES"
    DISTRIBUTES = "DISTRIBUTES"
    LOCATED_IN = "LOCATED_IN"
    CONTAINS = "CONTAINS"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    SELF_HARM = "SELF_HARM"


###########
def _strip_llm_noise(text: str) -> str:
    """Recursively removes descriptive fluff (e.g., 'Fellow Senior Inmate' -> 'Inmate')"""
    pattern = r"^(the|a|an|fellow|senior|rival|suspected|known|former|unidentified|corrupt|other|generic|inmate|officer|guard|staff|minority|hispanic)\b\s*"
    while True:
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        if cleaned == text:
            return cleaned
        text = cleaned

###########

class Entity(BaseModel):
    entity: str = Field(..., description="Entity name")
    label: EntityOntology = Field(..., description="Entity type defined in EntityOntology")

    @field_validator('entity', mode='before')
    @classmethod
    def normalize_entity_name(cls, value):        
        if not value or not isinstance(value, str):
            return "Unknown"
            
        val_clean = value.replace("_", " ").strip()
        val_clean = _strip_llm_noise(val_clean)
        if not val_clean:
            return "Unknown"

        # ==========================================
        # 1. THE SMART INCINERATOR (Dynamic Concept Dropper)
        # ==========================================
        abstract_keywords = [
            # Base Concepts
            "programming", "protocol", "network", "activity", "measures", 
            "investigation", "information", "records", "documentation", 
            "methodology", "charges", "therapy", "care", "assessment", 
            "evaluation", "stability", "trauma", "infrastructure", "procedure", 
            "management", "assignment", "administration", "conspiracy", 
            "breach", "threats", "materials", "services", "schedules", 
            "analysis", "enhancement", "counseling", "program", "illness", 
            "education", "timing", "enterprises", "departments", "resource", 
            "injuries", "payments", "fragility", "behavior", "rehabilitation", 
            "degree", "currency", "identification", "dates", "meetings", 
            "capture", "addresses", "coordination", "authorities", "language", 
            "descriptions", "timeline", "response", "dollars", "evidence", 
            "ideology", "philosophy", "population", "components", "opponents",
            "attendees", "extortion", "racketeering", "operations",
            # --- NEW ADDITIONS FROM THIS RUN ---
            "conflict", "ideation", "trial", "prosecution", "communications", 
            "plans", "methods", "fracture", "wounds", "notification", "stages", 
            "plot", "list", "torture", "objectives", "uprising", "alarms", 
            "violations", "violence", "debts", "elements", "organization", 
            "factions", "structure", "activities", "schematics", "specifications", 
            "instructions", "retribution", "corruption", "privileges", "opportunities", 
            "status", "drawings", "windows", "period", "hours", "restraints", 
            "connections", "oversight", "details", "assaults", "incident"
        ]
        
        # If the string contains any of these abstract concept words, instantly incinerate it
        if any(kw in val_clean.lower() for kw in abstract_keywords):
            return "DROP_ME_GENERIC"

        # 2. Check the Routing Maps for exact matches
        lower_val = val_clean.lower()
        resolved_name = NameResolutionMap.COMPILED_NAME_MAP.get(lower_val)
        if resolved_name:
            return resolved_name
            
        # 3. Fallback: Return clean Title Case
        title_cased = val_clean.title()
        return title_cased.replace("'S", "'s")

    @field_validator('label', mode='before')
    @classmethod
    def heal_label_hallucinations(cls, value: str) -> str:
        if isinstance(value, str):
            val_clean = value.strip()
            lower_val = val_clean.lower()
            entity_cat = EntityMap.COMPILED_LABEL_MAP.get(lower_val, val_clean) 
            if entity_cat not in [e.value for e in EntityOntology]:
                return EntityOntology.UNKNOWN.value          
            return entity_cat
        return value
    
    @model_validator(mode='after')
    def cross_check_entity_and_label(self) -> 'Entity':
        
        current_entity_name = str(self.entity)
        name_title = current_entity_name.title() # For exact name matching
        name_lower = current_entity_name.lower()

        KNOWN_INMATES = [
            "Viper", "Runner", "Miller", "Jackson", "O'Reilly", "Chen", 
            "Batista", "Smalls", "Kincaid", "Torres", "Washington", 
            "Diaz", "Petrov", "Rossi", "Carver"
        ]

        if name_title in KNOWN_INMATES:
            self.label = EntityOntology.INMATE # Adjust to your Enum (e.g., INMATE)
            return self
        
        KNOWN_GROUPS = [
            "Iron Syndicate", "Ghost Cartel", "Vipers"
        ]
        if name_title in KNOWN_GROUPS:
            self.label = EntityOntology.GROUP 
            return self

        KNOWN_STAFF = [
            "Williams", "Thompson", "Martinez", "Davis", "Mitchell" 
        ]

        if name_title in KNOWN_STAFF:
            self.label = EntityOntology.STAFF # Adjust to your Enum (e.g., STAFF)
            return self

        vulnerable_labels = ["Unknown", "UNKNOWN", "DigitalAsset", "DIGITAL_ASSET", "Inmate", "INMATE", "InternalLocation", "INTERNAL_LOCATION", "Group", "GROUP"]    

        if self.label.value in vulnerable_labels:
            current_entity_name = str(self.entity)
            name_lower = current_entity_name.lower()
            
            # 1. Contraband Heuristics (Runs FIRST to catch "Cell Phone")
            contra_keywords = ["drug", "narcotic", "phone", "tobacco", "pipe", "device", "bedsheet", "radio", "alcohol", "lock", "machinery", "substance", "ivs", "oil"]
            if any(kw in name_lower for kw in contra_keywords):
                self.label = EntityOntology.CONTRABAND
                return self

            # 2. Weapon Heuristics
            weapon_keywords = ["shank", "shiv", "blade", "razor", "tray", "club", "spear", "knife", "weapon", "metal", "firearm", "scalpel", "rod", "projectile", "instrument"]
            if any(kw in name_lower for kw in weapon_keywords):
                self.label = EntityOntology.WEAPON
                return self
            
            # 3. Location Heuristics (Runs AFTER items are filtered out)
            loc_keywords = ["yard", "court", "cell", "block", "area", "library", "hall", "room", "commissary", "garden", "laundry", "door", "wing", "unit", "facility", "kitchen"]
            if any(kw in name_lower for kw in loc_keywords):
                self.label = EntityOntology.INTERNAL_LOCATION 
                return self
                
            # 4. Group Heuristics
            group_keywords = ["gang", "cartel", "syndicate", "family", "vipers", "team", "squad", "crew", "faction"]
            if any(kw in name_lower for kw in group_keywords):
                self.label = EntityOntology.GROUP
                return self
                
            # 5. Staff Heuristics
            staff_keywords = ["officer", "staff", "guard", "forensic", "instructor", "warden", "nurse"]
            if any(kw in name_lower for kw in staff_keywords):
                self.label = EntityOntology.STAFF
                return self
                
        return self

class Relation(BaseModel):
    subject: str = Field(..., description="Source entity")
    predicate: PredicateOntology = Field(..., description="Relationship allowed defined in PredicateOntology")
    object: str = Field(..., description="Target entity")

    @field_validator('subject', 'object', mode='before')
    @classmethod
    def heal_missing_nodes(cls, value) -> str:
        # 1. Null / Type Check
        if not value or not isinstance(value, str):
            return "Unknown"
            
        # 2. Base Formatting
        val_clean = value.replace("_", " ").strip()
        
        # 3. Strip Noisy Adjectives (Using our helper function!)
        val_clean = _strip_llm_noise(val_clean)
        if not val_clean:
            return "Unknown"

        # 4. Routing Map Check
        lower_val = val_clean.lower()
        resolved_name = NameResolutionMap.COMPILED_NAME_MAP.get(lower_val)
        if resolved_name:
            return resolved_name
            
        # 5. Fallback: Clean Title Case (with 's fix)
        title_cased = val_clean.title()
        return title_cased.replace("'S", "'s")

    @field_validator('predicate', mode='before')
    @classmethod
    def heal_llm_hallucinations(cls, value: str) -> str:
        if isinstance(value, str):
            val_clean = value.upper().strip()
            rel_cat = PredicateMap.COMPILED_HEALING_MAP.get(val_clean, val_clean) 
            if rel_cat not in [p.value for p in PredicateOntology]:
                return PredicateOntology.ASSOCIATED_WITH.value           
            return rel_cat                            
        return value


class LogRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Original log text to process")
    @field_validator('text')
    @classmethod
    def text_not_empty(cls,v):
        if not v.strip():
            raise ValueError('Text cannot be blank')
        return v.strip()


class LogResponse(BaseModel):
    id: str = Field(default_factory=generate_uuid, description="Unique response ID")
    log_text: str
    processed_at: str = Field(default_factory=generate_timestamp, description="UTC processing timestamp")
    taxonomy_category: Taxonomy = Field(..., description="Classification category")
    reasoning: str
    confidence: float
    entities: List[Entity]
    relations: List[Relation]

