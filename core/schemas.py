from pydantic import BaseModel

# 1. Define the Expected Input (The Contract)
class OMSLog(BaseModel):
    log_id: str
    date: str
    log_text: str

