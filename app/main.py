from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
from app.rag_service import rag_service
from app.config import settings
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pinecone API",
    description="Retrieval-Augmented Generation API using Pinecone and OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Request/Response models
class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    source_documents: List[dict]


class DocumentResponse(BaseModel):
    status: str
    chunks_added: int


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        logger.info("Initializing RAG service...")
        rag_service.initialize()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Don't raise - allow server to start even if keys are missing
        logger.warning("Server starting without RAG initialization. Please check API keys.")


@app.get("/")
async def root():
    """Serve the web UI."""
    static_path = os.path.join(static_dir, "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    return {
        "message": "RAG Pinecone API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/documents", response_model=DocumentResponse)
async def add_document(request: DocumentRequest):
    """
    Add a document to the vector store.
    
    - **text**: The text content to add
    - **metadata**: Optional metadata dictionary
    """
    try:
        result = rag_service.add_documents(request.text, request.metadata)
        return DocumentResponse(**result)
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    - **question**: The user's question
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag_service.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)
