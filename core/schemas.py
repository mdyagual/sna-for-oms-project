from enum import Enum
from typing import List
from datetime import datetime, timezone
from dataclasses import dataclass
import uuid
from pydantic import BaseModel, Field, field_validator

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
        "ASSOCIATED_WITH": ["WITNESSED", "ANALYZES", "INVESTIGATES", "DESTROYED_IN", "TREATED_WITH", "REQUIRES", "WANTS", "WANTED", "UNDERGOES", "UNDERWENT", "ISOLATED_FROM", "SEPARATED_FROM", "OPERATED_BY", "CONTROLLED_BY", "INITIATES", "INITIATED", "CONSIDERS", "CONSIDERED", "SUFFERED", "SUFFERS", "DIAGNOSED_WITH", "DIAGNOSED", "DOCUMENTS", "DOCUMENTED", "WILLING_TO", "EXPERIENCES", "NEUTRALIZES", "NEUTRALIZED", "ACTIVATES", "ACTIVATED", "FILES", "FILED", "FILED_CHARGES", "SUSTAINS", "SUSTAINED" ],
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
        "Inmate": ["Unknown_Attacker", "Unknown_Inmate", "Victim", "Suspect", "Unknown_Targets", "Unknown_Target", "Target", "Targets"],
        "Group": ["Unknown_Group", "Unidentified_Group" ],
        "Event": ["Unknown_Event", "TimePeriod", "Time", "Date"],
        "Staff": ["Unknown_Staff"]
    }

    COMPILED_LABEL_MAP = {
        word: target 
        for target, words in RAW_LABEL_GROUPS.items() 
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
            self.NEUTRAL: 1
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

class Entity(BaseModel):
    entity: str = Field(..., description="Entity name")
    label: EntityOntology = Field(..., description="Entity type defined in EntityOntology")

    @field_validator('label', mode='before')
    @classmethod
    def heal_label_hallucinations(cls, value: str) -> str:
        if isinstance(value, str):
            val_clean = value.strip()            
            return EntityMap.COMPILED_LABEL_MAP.get(val_clean, val_clean)
        return value

class Relation(BaseModel):
    subject: str = Field(..., description="Source entity")
    predicate: PredicateOntology = Field(..., description="Relationship allowed defined in PredicateOntology")
    object: str = Field(..., description="Target entity")

    @field_validator('subject', 'object', mode='before')
    @classmethod
    def heal_missing_nodes(cls, value) -> str:
        # If the LLM output null (None) or an empty string ""
        if not value:
            return "Unknown_Entity"
            
        # Ensure it's a string just in case the LLM outputs an integer ID
        return str(value)

    @field_validator('predicate', mode='before')
    @classmethod
    def heal_llm_hallucinations(cls, value: str) -> str:
        if isinstance(value, str):
            val_clean = value.upper().strip()            
            return PredicateMap.COMPILED_HEALING_MAP.get(val_clean, val_clean)                            
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

