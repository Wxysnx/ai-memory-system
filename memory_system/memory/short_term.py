"""Short-term memory implementation using Redis."""

import json
import time
from typing import Dict, List, Optional, Any

from langchain_core.messages import (
    AIMessage, 
    BaseMessage, 
    HumanMessage, 
    SystemMessage
)
from redis.asyncio.client import Redis
import redis.asyncio as redis

from memory_system.config import config
from memory_system.services.cache import RedisCache
from memory_system.services.event_bus import event_bus, MemoryEvent
from memory_system.memory import MEMORY_CREATED, MEMORY_UPDATED, SHORT_TERM


class ShortTermMemory:
    """Short-term memory implementation using Redis."""
    
    def __init__(
        self, 
        redis_url: Optional[str] = None,
        ttl: int = 3600,  # 1 hour default TTL
        max_messages: int = 20
    ):
        """Initialize short-term memory.
        
        Args:
            redis_url: Redis connection URL (if not provided, uses config)
            ttl: Time to live in seconds for memory entries
            max_messages: Maximum number of messages to store per conversation
        """
        self.redis_url = redis_url or config.redis.url
        self.ttl = ttl
        self.max_messages = max_messages
        self.redis_client: Optional[Redis] = None
        self.cache = RedisCache(redis_url=redis_url)
    
    async def connect(self) -> Redis:
        """Connect to Redis.
        
        Returns:
            Redis client
        """
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.redis_url, 
                decode_responses=True
            )
        return self.redis_client
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
    
    def _get_conversation_key(self, conversation_id: str) -> str:
        """Get Redis key for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Redis key
        """
        return f"conversation:{conversation_id}:messages"
    
    def _parse_message(self, message_data: str) -> BaseMessage:
        """Parse message data from Redis.
        
        Args:
            message_data: JSON string of message data
            
        Returns:
            BaseMessage instance
        """
        data = json.loads(message_data)
        message_type = data.get("type", "human")
        content = data.get("content", "")
        
        if message_type == "ai":
            return AIMessage(content=content)
        elif message_type == "system":
            return SystemMessage(content=content)
        else:
            return HumanMessage(content=content)
    
    def _serialize_message(self, message: BaseMessage) -> str:
        """Serialize a message for Redis storage.
        
        Args:
            message: BaseMessage instance
            
        Returns:
            JSON string
        """
        if isinstance(message, AIMessage):
            message_type = "ai"
        elif isinstance(message, SystemMessage):
            message_type = "system"
        else:
            message_type = "human"
            
        data = {
            "type": message_type,
            "content": message.content,
            "timestamp": time.time()
        }
        
        return json.dumps(data)
    
    async def add_message(self, conversation_id: str, message: BaseMessage) -> None:
        """Add a message to short-term memory.
        
        Args:
            conversation_id: Conversation ID
            message: BaseMessage to add
        """
        await self.connect()
        key = self._get_conversation_key(conversation_id)
        serialized = self._serialize_message(message)
        
        # Add message to Redis list
        await self.redis_client.lpush(key, serialized)
        
        # Trim to max length
        await self.redis_client.ltrim(key, 0, self.max_messages - 1)
        
        # Set expiration on the key
        await self.redis_client.expire(key, self.ttl)
        
        # Publish memory event
        await event_bus.publish_event(
            MemoryEvent(
                event_type=MEMORY_CREATED,
                conversation_id=conversation_id,
                payload={
                    "memory_type": SHORT_TERM,
                    "message_type": type(message).__name__,
                    "content": message.content
                }
            )
        )
    
    async def get_messages(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """Get messages from short-term memory.
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages to retrieve
            
        Returns:
            List of messages, newest first
        """
        await self.connect()
        key = self._get_conversation_key(conversation_id)
        max_msgs = limit or self.max_messages
        
        # Get messages from Redis
        message_data = await self.redis_client.lrange(key, 0, max_msgs - 1)
        
        # Parse messages
        messages = [self._parse_message(msg) for msg in message_data]
        
        # Reverse to get chronological order
        messages.reverse()
        
        # Publish memory event
        await event_bus.publish_event(
            MemoryEvent(
                event_type=MEMORY_UPDATED,
                conversation_id=conversation_id,
                payload={
                    "memory_type": SHORT_TERM,
                    "message_count": len(messages)
                }
            )
        )
        
        return messages
    
    async def clear(self, conversation_id: str) -> None:
        """Clear short-term memory for a conversation.
        
        Args:
            conversation_id: Conversation ID
        """
        await self.connect()
        key = self._get_conversation_key(conversation_id)
        await self.redis_client.delete(key)


# Global short-term memory instance
short_term_memory = ShortTermMemory() 
