"""MongoDB database service for long-term memory persistence."""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Type

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pydantic import BaseModel

from memory_system.config import config


T = TypeVar('T', bound=BaseModel)


class MongoDBService(Generic[T]):
    """MongoDB service for document storage and retrieval."""

    def __init__(
        self, 
        mongo_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        model_class: Optional[Type[T]] = None
    ):
        """Initialize MongoDB service.
        
        Args:
            mongo_uri: MongoDB connection URI (if not provided, uses config)
            database_name: Database name (if not provided, uses config)
            model_class: Pydantic model class for document validation
        """
        self.mongo_uri = mongo_uri or config.mongodb.uri
        self.database_name = database_name or config.mongodb.database
        self.model_class = model_class
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> AsyncIOMotorDatabase:
        """Connect to MongoDB.
        
        Returns:
            Motor database instance
        """
        if self.client is None:
            self.client = AsyncIOMotorClient(self.mongo_uri)
            self.db = self.client[self.database_name]
        return self.db
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
    
    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Motor collection instance
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db[collection_name]
    
    async def create_indexes(self, collection_name: str, indexes: List[Dict[str, Any]]) -> None:
        """Create indexes on a collection.
        
        Args:
            collection_name: Name of the collection
            indexes: List of index specifications
        """
        db = await self.connect()
        collection = db[collection_name]
        for index in indexes:
            await collection.create_index(**index)
    
    async def insert_one(
        self, collection_name: str, document: Dict[str, Any]
    ) -> str:
        """Insert a document into a collection.
        
        Args:
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            ID of the inserted document
        """
        if self.model_class:
            # Validate with pydantic model if provided
            validated = self.model_class.parse_obj(document)
            document = validated.dict()
            
        db = await self.connect()
        collection = db[collection_name]
        result = await collection.insert_one(document)
        return str(result.inserted_id)
    
    async def find_one(
        self, collection_name: str, query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find a document in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            
        Returns:
            Document or None if not found
        """
        db = await self.connect()
        collection = db[collection_name]
        result = await collection.find_one(query)
        
        if result is None:
            return None
            
        if result and self.model_class:
            # Convert ObjectId to string for _id
            if "_id" in result:
                result["_id"] = str(result["_id"])
            # Could convert to pydantic model here if needed
                
        return result
    
    async def find_many(
        self, 
        collection_name: str, 
        query: Dict[str, Any],
        sort: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find multiple documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            sort: Optional sort specification
            limit: Optional result limit
            skip: Optional number of documents to skip
            
        Returns:
            List of documents
        """
        db = await self.connect()
        collection = db[collection_name]
        
        cursor = collection.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        if skip:
            cursor = cursor.skip(skip)
            
        if limit:
            cursor = cursor.limit(limit)
            
        results = await cursor.to_list(length=limit or 100)
        
        # Convert ObjectId to string for _id
        for result in results:
            if "_id" in result:
                result["_id"] = str(result["_id"])
                
        return results
    
    async def update_one(
        self, 
        collection_name: str, 
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        """Update a document in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            update: Update specification
            upsert: Whether to insert if document doesn't exist
            
        Returns:
            True if a document was modified
        """
        db = await self.connect()
        collection = db[collection_name]
        result = await collection.update_one(query, update, upsert=upsert)
        return result.modified_count > 0
    
    async def delete_one(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """Delete a document from a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query filter
            
        Returns:
            True if a document was deleted
        """
        db = await self.connect()
        collection = db[collection_name]
        result = await collection.delete_one(query)
        return result.deleted_count > 0


# Global MongoDB service instance
mongodb_service = MongoDBService() 
