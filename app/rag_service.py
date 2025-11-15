from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.pinecone_client import pinecone_manager
from app.document_processor import DocumentProcessor
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """Handles RAG operations: document ingestion and querying."""
    
    def __init__(self):
        self.vector_store = None
        self.document_processor = DocumentProcessor()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        self.retriever = None
    
    def initialize(self):
        """Initialize the RAG service with vector store."""
        try:
            self.vector_store = pinecone_manager.get_vector_store()
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            logger.info("RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}")
            raise
    
    def add_documents(self, text: str, metadata: dict = None):
        """
        Add documents to the vector store.
        
        Args:
            text: The text content to add
            metadata: Optional metadata dictionary
        """
        try:
            if self.vector_store is None:
                self.vector_store = pinecone_manager.get_vector_store()
            
            # Process text into chunks
            documents = self.document_processor.process_text(text, metadata or {})
            
            if not documents or len(documents) == 0:
                logger.warning("No documents to add after processing")
                return {"status": "error", "chunks_added": 0, "message": "No valid text to process"}
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Re-initialize retriever after adding documents to ensure it's up to date
            if self.vector_store:
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            logger.info(f"Added {len(documents)} document chunks to vector store")
            return {"status": "success", "chunks_added": len(documents)}
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(self, question: str):
        """
        Query the RAG system.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            if self.retriever is None:
                self.initialize()
            
            # Get retrieved documents for source tracking
            retrieved_docs = self.retriever.invoke(question)
            
            # Check if we have any documents
            if not retrieved_docs or len(retrieved_docs) == 0:
                logger.warning(f"No documents found for query: {question}")
                return {
                    "answer": "I don't have any documents in my knowledge base yet. Please upload some documents first using the 'Upload Documents' tab.",
                    "source_documents": []
                }
            
            # Format documents for context
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            context = format_docs(retrieved_docs)
            
            # Create prompt with context
            prompt = ChatPromptTemplate.from_template("""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:""")
            
            # Get response from LLM
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            
            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            raise


# Global instance
rag_service = RAGService()

