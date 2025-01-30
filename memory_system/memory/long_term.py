"""Long-term memory implementation using MongoDB and vector search."""

import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
import pymongo
from pymongo import MongoClient
from pydantic import BaseModel, Field

from memory_system.config import config
from memory_system.services.database import MongoDBService
from memory_system.services.vector_store import VectorStoreService
from memory_system.services.event_bus import event_bus, MemoryEvent
from memory_system.memory import MEMORY_CREATED, MEMORY_RETRIEVED, LONG_TERM


class MemoryEntry(BaseModel):
    """Representation of a long-term memory entry."""
    
    conversation_id: str
    content: str
    memory_type: str = "message"  # message, summary, fact, etc.
    source_message_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    importance: float = 0.0  # 0-1 score indicating importance


class LongTermMemory:
    """Long-term memory implementation using MongoDB and vector search."""
    
    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        collection_name: str = "long_term_memories"
    ):
        """Initialize long-term memory.
        
        Args:
            mongo_uri: MongoDB connection URI (if not provided, uses config)
            collection_name: Collection name for memories
        """
        self.mongo_uri = mongo_uri or config.mongodb.uri
        self.collection_name = collection_name
        
        # Initialize services
        self.db = MongoDBService[MemoryEntry](
            mongo_uri=self.mongo_uri,
            model_class=MemoryEntry
        )
        self.vector_store = VectorStoreService(
            collection_name=f"{collection_name}_vectors"
        )
    
    async def connect(self) -> None:
        """Connect to database and vector store."""
        await self.db.connect()
        self.vector_store.connect()
        
        # Create indexes
        await self.db.create_indexes(
            self.collection_name,
            [
                {"keys": [("conversation_id", pymongo.ASCENDING)]},
                {"keys": [("timestamp", pymongo.DESCENDING)]},
                {"keys": [("importance", pymongo.DESCENDING)]},
            ]
        )
    
    async def disconnect(self) -> None:
        """Disconnect from database."""
        await self.db.disconnect()
    
    async def store(
        self,
        conversation_id: str,
        content: str,
        memory_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.0
    ) -> str:
        """Store a memory in long-term storage.
        
        Args:
            conversation_id: Conversation ID
            content: Memory content
            memory_type: Type of memory (message, summary, fact, etc.)
            metadata: Additional metadata
            importance: Importance score (0-1)
            
        Returns:
            ID of the stored memory
        """
        metadata = metadata or {}
        
        # Create memory entry
        memory = MemoryEntry(
            conversation_id=conversation_id,
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            importance=importance
        )
        
        # Store in MongoDB
        memory_id = await self.db.insert_one(
            self.collection_name,
            memory.dict()
        )
        
        # Store in vector store for semantic search
        await self.vector_store.add_texts(
            texts=[content],
            metadatas=[{
                "memory_id": memory_id,
                "conversation_id": conversation_id,
                "memory_type": memory_type,
                "importance": importance,
                **metadata
            }]
        )
        
        # Publish memory event
        await event_bus.publish_event(
            MemoryEvent(
                event_type=MEMORY_CREATED,
                conversation_id=conversation_id,
                payload={
                    "memory_type": LONG_TERM,
                    "memory_id": memory_id,
                    "content": content
                }
            )
        )
        
        return memory_id
    
    async def store_message(
        self,
        conversation_id: str,
        message: BaseMessage,
        importance: float = 0.0
    ) -> str:
        """Store a message in long-term memory.
        
        Args:
            conversation_id: Conversation ID
            message: Message to store
            importance: Importance score (0-1)
            
        Returns:
            ID of the stored memory
        """
        # Determine message type
        if isinstance(message, AIMessage):
            sender = "ai"
        elif isinstance(message, SystemMessage):
            sender = "system"
        else:
            sender = "human"
            
        # Store message with metadata
        return await self.store(
            conversation_id=conversation_id,
            content=message.content,
            memory_type="message",
            metadata={"sender": sender},
            importance=importance
        )
    
    async def retrieve_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory entry or None if not found
        """
        result = await self.db.find_one(
            self.collection_name,
            {"_id": memory_id}
        )
        
        if result:
            return MemoryEntry(**result)
        return None
    
    async def search(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None
    ) -> List[Document]:
        """Search for memories semantically related to query.
        
        Args:
            query: Search query
            conversation_id: Optional filter by conversation
            limit: Maximum number of results
            memory_type: Optional filter by memory type
            min_importance: Optional minimum importance score
            
        Returns:
            List of matching documents
        """
        # Build filter
        filter_dict = {}
        if conversation_id:
            filter_dict["conversation_id"] = conversation_id
        if memory_type:
            filter_dict["memory_type"] = memory_type
        if min_importance is not None:
            filter_dict["importance"] = {"$gte": min_importance}
        
        # Perform vector search
        results = await self.vector_store.similarity_search(
            query=query,
            k=limit,
            filter=filter_dict if filter_dict else None
        )
        
        # Publish memory event
        await event_bus.publish_event(
            MemoryEvent(
                event_type=MEMORY_RETRIEVED,
                conversation_id=conversation_id or "global",
                payload={
                    "memory_type": LONG_TERM,
                    "query": query,
                    "result_count": len(results)
                }
            )
        )
        
        return results
    
    async def get_conversation_memories(
        self,
        conversation_id: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None
    ) -> List[MemoryEntry]:
        """Get memories for a specific conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of memories to retrieve
            memory_type: Optional filter by memory type
            min_importance: Optional minimum importance score
            
        Returns:
            List of memory entries
        """
        # Build query
        query = {"conversation_id": conversation_id}
        if memory_type:
            query["memory_type"] = memory_type
        if min_importance is not None:
            query["importance"] = {"$gte": min_importance}
        
        # Query MongoDB
        results = await self.db.find_many(
            self.collection_name,
            query=query,
            sort=[("timestamp", -1)],
            limit=limit
        )
        
        # Convert to MemoryEntry objects
        return [MemoryEntry(**result) for result in results]
    
    async def summarize_conversation(self, conversation_id: str) -> str:
        """Create a summary of a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Summary ID
        """
        # This would typically use an LLM to generate a summary
        # For now we'll implement a placeholder that would be replaced
        # with actual summarization logic
        
        # Get recent messages
        messages = await self.get_conversation_memories(
            conversation_id=conversation_id,
            limit=20,
            memory_type="message"
        )
        
        if not messages:
            return ""
            
        # Create a basic summary (in real implementation, use LLM)
        summary = f"Conversation with {len(messages)} messages"
        
        # Store the summary
        summary_id = await self.store(
            conversation_id=conversation_id,
            content=summary,
            memory_type="summary",
            importance=1.0  # Summaries are important
        )
        
        return summary_id


# Global long-term memory instance
long_term_memory = LongTermMemory() 
