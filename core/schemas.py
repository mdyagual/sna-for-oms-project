from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime, timezone
import uuid

def generate_uuid():
    return str(uuid.uuid4())

def generate_timestamp():
    # Always use UTC timezone for server logs!
    return datetime.now(timezone.utc).isoformat()

# 1. Define the Expected Input
class LogRequest(BaseModel):
    text: str

# 2. Define the Expected Output
class LogResponse(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    log_text: str
    processed_at: str = Field(default_factory=generate_timestamp)
    taxonomy_category: str
    reasoning: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]