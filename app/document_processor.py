from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document chunking and processing."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Process text into chunks.
        
        Args:
            text: The text content to process
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        # Split text into chunks
        chunks = self.text_splitter.create_documents([text], [metadata])
        
        logger.info(f"Processed text into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, documents: List[str], metadatas: List[dict] = None) -> List[Document]:
        """
        Process multiple documents into chunks.
        
        Args:
            documents: List of text contents
            metadatas: Optional list of metadata dicts for each document
            
        Returns:
            List of Document objects
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        all_chunks = []
        for doc_text, metadata in zip(documents, metadatas):
            chunks = self.process_text(doc_text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks

