import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from core.nlp_layer.extract_detect import process_entry
from core.schemas import LogRequest, LogResponse
from core.kg_sna_layer.graph_builder import AsyncKnowledgeGraphManager
from core.config import graph_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kg_manager = AsyncKnowledgeGraphManager(graph_base.URL, graph_base.USER, graph_base.PASS)

app = FastAPI(
    title="SNA for OMS - API",
    description="NLP Extraction and Taxonomy Detection for Offender Management Logs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before the 'yield' happens on Server Startup
    logger.info("Connecting to Neo4j Database...")
    yield
    # Everything after the 'yield' happens on Server Shutdown
    logger.info("Closing Neo4j Database connection...")
    await kg_manager.close()

@app.get("/", include_in_schema=False)
async def load_swagger():
    return RedirectResponse(url="/docs")

# Notice we added response_model=ResponseLog here!
@app.post("/api/v1/extract", 
        response_model=LogResponse, 
        status_code=status.HTTP_200_OK,
        summary="Extract entities, relationships and taxonomy from Text",
        responses={
            500: {"description": "Internal server error during processing"},
            422: {"description": "Validation error in request body"}
        }
)
async def extract_knowledge_graph(request: LogRequest):
    try:
        logger.info("Processing text of length: %d" , (len(request.text)))
        result = await process_entry(request.text)
        parsed_result = LogResponse(**result)
        logger.info("Successfully processed entry with ID: %s", (parsed_result.id))

        await kg_manager.ingest_log_data(parsed_result.model_dump())

        return parsed_result
        
    except ValueError as e:
        logger.error("Validation error: %s", (str(e)))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Processing error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error("Unexpected error: %s", (str(e)), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during processing"
        ) from e
    
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}