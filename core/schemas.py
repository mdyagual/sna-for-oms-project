from pydantic import BaseModel
from typing import List, Dict, Any

# 1. Define the Expected Input
class LogRequest(BaseModel):
    text: str

# 2. Define the Expected Output
class LogResponse(BaseModel):
    id: str
    log_text: str
    date: str
    taxonomy_category: str
    reasoning: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]