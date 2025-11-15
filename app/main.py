from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
from app.rag_service import rag_service
from app.config import settings

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
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "RAG Pinecone API is running",
        "version": "1.0.0"
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
