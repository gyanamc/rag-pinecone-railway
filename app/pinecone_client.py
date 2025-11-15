from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class PineconeManager:
    """Manages Pinecone index initialization and operations."""
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        self.vector_store = None
    
    def initialize_index(self):
        """Initialize or connect to Pinecone index."""
        try:
            # Check if index exists
            if settings.pinecone_index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating index: {settings.pinecone_index_name}")
                self.pc.create_index(
                    name=settings.pinecone_index_name,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            else:
                logger.info(f"Index {settings.pinecone_index_name} already exists")
            
            # Connect to the index
            index = self.pc.Index(settings.pinecone_index_name)
            
            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=self.embeddings
            )
            
            logger.info("Pinecone initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def get_vector_store(self):
        """Get the vector store instance."""
        if self.vector_store is None:
            self.initialize_index()
        return self.vector_store


# Global instance
pinecone_manager = PineconeManager()
