from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        self.qa_chain = None
    
    def initialize(self):
        """Initialize the RAG service with vector store."""
        try:
            self.vector_store = pinecone_manager.get_vector_store()
            
            # Create custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
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
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
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
            if self.qa_chain is None:
                self.initialize()
            
            # Get response
            result = self.qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            raise


# Global instance
rag_service = RAGService()

