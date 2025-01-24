"""Redis cache service for short-term memory and caching needs."""

import json
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from redis.asyncio.client import Redis

from memory_system.config import config


class RedisCache:
    """Redis cache service implementation."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the Redis cache service.
        
        Args:
            redis_url: Redis connection URL. If not provided, uses the configuration.
        """
        self.redis_url = redis_url or config.redis.url
        self.client: Optional[Redis] = None
    
    async def connect(self) -> Redis:
        """Connect to Redis server.
        
        Returns:
            Redis client instance.
        """
        if self.client is None or not self.client.ping():
            self.client = redis.from_url(self.redis_url, decode_responses=True)
        return self.client
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client is not None:
            await self.client.close()
            self.client = None
    
    async def set(
        self, key: str, value: Any, expiration: Optional[int] = None
    ) -> bool:
        """Set a key-value pair in the cache.
        
        Args:
            key: Cache key
            value: Value to cache (will be serialized to JSON)
            expiration: Optional expiration time in seconds
            
        Returns:
            True if successful
        """
        client = await self.connect()
        serialized = json.dumps(value) if not isinstance(value, str) else value
        return await client.set(key, serialized, ex=expiration)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Deserialized value or None if key doesn't exist
        """
        client = await self.connect()
        value = await client.get(key)
        
        if value is None:
            return None
            
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    
    async def delete(self, key: str) -> int:
        """Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Number of keys deleted
        """
        client = await self.connect()
        return await client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        client = await self.connect()
        result = await client.exists(key)
        return bool(result)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time on a key.
        
        Args:
            key: Cache key
            seconds: Time in seconds
            
        Returns:
            True if successful
        """
        client = await self.connect()
        return await client.expire(key, seconds)
    
    async def lpush(self, key: str, *values: Any) -> int:
        """Append values to a list.
        
        Args:
            key: List key
            values: Values to append
            
        Returns:
            Length of the list after operation
        """
        client = await self.connect()
        serialized_values = [
            json.dumps(v) if not isinstance(v, str) else v for v in values
        ]
        return await client.lpush(key, *serialized_values)
    
    async def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """Get a range of elements from a list.
        
        Args:
            key: List key
            start: Start index
            end: End index
            
        Returns:
            List of values
        """
        client = await self.connect()
        values = await client.lrange(key, start, end)
        
        result = []
        for v in values:
            try:
                result.append(json.loads(v))
            except json.JSONDecodeError:
                result.append(v)
                
        return result
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim a list to specified range.
        
        Args:
            key: List key
            start: Start index
            end: End index
            
        Returns:
            True if successful
        """
        client = await self.connect()
        return await client.ltrim(key, start, end) == "OK"


# Global Redis cache instance
redis_cache = RedisCache() 
