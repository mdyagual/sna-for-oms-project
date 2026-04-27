from enum import Enum
from typing import List
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field, field_validator

def generate_uuid():
    return str(uuid.uuid4())

def generate_timestamp():
    # Always use UTC timezone for server logs!
    return datetime.now(timezone.utc).isoformat()

class Entity(BaseModel):
    entity: str = Field(..., description="Entity name")
    label: str = Field(..., description="Entity type (Person, Group, Location, Target)")

class Relation(BaseModel):
    subject: str = Field(..., description="Source entity")
    predicate: str = Field(..., description="Relationship verb")
    object: str = Field(..., description="Target entity")


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
    taxonomy_category: str = Field(..., description="Classification category")
    reasoning: str
    confidence: float
    entities: List[Entity]
    relations: List[Relation]

#The taxonomy defined for the project with their severity to consider
class Taxonomy(Enum):
    HATE = 3
    EMOTIONAL = 1
    SENTIMENTAL = 1
    EXTREMIST = 5
    RADICALISM = 5
    THREAT = 4
    VIOLATED_SCALED = 2
    NEUTRAL = 1