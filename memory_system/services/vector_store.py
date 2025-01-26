"""Vector store service for semantic search and retrieval."""

from typing import Dict, List, Optional, Any
import os

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from pymongo import MongoClient

from memory_system.config import config


class VectorStoreService:
    """Vector store service for semantic search using MongoDB Atlas Vector Search."""

    def __init__(
        self, 
        collection_name: str = "vector_memories",
        embedding_model: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        database_name: Optional[str] = None
    ):
        """Initialize vector store service.
        
        Args:
            collection_name: MongoDB collection name
            embedding_model: Model to use for embeddings
                (e.g. 'openai', 'huggingface/all-MiniLM-L6-v2')
            mongo_uri: MongoDB connection URI (if not provided, uses config)
            database_name: Database name (if not provided, uses config)
        """
        self.collection_name = collection_name
        self.mongo_uri = mongo_uri or config.mongodb.uri
        self.database_name = database_name or config.mongodb.database
        
        # Initialize embeddings model
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "huggingface/all-MiniLM-L6-v2")
        self.embeddings = self._get_embeddings(self.embedding_model)
        
        # Vector store is initialized in connect()
        self.vector_store = None
        
    def _get_embeddings(self, model_name: str) -> Embeddings:
        """Get embeddings model based on name.
        
        Args:
            model_name: Name of the embeddings model
            
        Returns:
            Embeddings instance
        """
        if model_name.startswith("openai"):
            return OpenAIEmbeddings(model=model_name)
        else:
            # Default to HuggingFace embeddings
            return HuggingFaceEmbeddings(model_name=model_name)
    
    def connect(self) -> MongoDBAtlasVectorSearch:
        """Connect to vector store.
        
        Returns:
            Vector store instance
        """
        if self.vector_store is None:
            # Create MongoDB client
            client = MongoClient(self.mongo_uri)
            db = client[self.database_name]
            collection = db[self.collection_name]
            
            # Initialize vector store
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embeddings,
                index_name="vector_index",
                text_key="content"
            )
            
            # Check if index exists and create it if not
            index_info = collection.index_information()
            if "vector_index" not in index_info:
                self._create_vector_index(collection)
                
        return self.vector_store
    
    def _create_vector_index(self, collection) -> None:
        """Create vector index in MongoDB collection.
        
        Args:
            collection: MongoDB collection
        """
        # Create vector search index
        collection.create_index(
            [("vector", "vector")],
            name="vector_index",
            vectorOptions={
                "type": "cosine",
                "numDimensions": self.embeddings.dimension,
            }
        )
    
    async def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add texts to vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        vector_store = self.connect()
        return vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results
            filter: Optional metadata filter
            
        Returns:
            List of documents
        """
        vector_store = self.connect()
        return vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    async def delete(self, ids: List[str]) -> None:
        """Delete documents from vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        vector_store = self.connect()
        # Use the MongoDB collection directly since langchain might not expose delete
        client = MongoClient(self.mongo_uri)
        db = client[self.database_name]
        collection = db[self.collection_name]
        
        collection.delete_many({"_id": {"$in": ids}})


# Global vector store service instance
vector_store_service = VectorStoreService() 
