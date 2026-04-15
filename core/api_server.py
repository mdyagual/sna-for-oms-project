from fastapi import FastAPI, HTTPException
from core.nlp_layer.extract_detect import process_entry
from core.schemas import LogRequest, LogResponse

app = FastAPI(
    title="SNA for OMS - API",
    description="NLP Extraction and Taxonomy Detection for Offender Management Logs",
    version="1.0.0"
)

# Notice we added response_model=ResponseLog here!
@app.post("/api/v1/extract", response_model=LogResponse, summary="Extract Graph and Taxonomy from Text")
def extract_knowledge_graph(request: LogRequest):
    try:
        # Fixed the variable mappings to match your LogRequest schema
        result = process_entry(idx=request.id, text=request.text)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))