from openai import OpenAI
from typing import List, Dict, Any
from app.config import settings


class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.embedding_model = settings.embedding_model
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response using the chat completion API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content


# Global instance
openai_client = OpenAIClient()

