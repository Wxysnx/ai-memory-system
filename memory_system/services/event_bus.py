"""Kafka event bus for memory events."""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, Union

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import BaseModel

from memory_system.config import config


class MemoryEvent(BaseModel):
    """Base event model for memory events."""
    
    event_type: str
    conversation_id: str
    payload: Dict[str, Any]
    timestamp: Optional[float] = None


class EventBus:
    """Kafka event bus for publishing and subscribing to memory events."""
    
    def __init__(
        self, 
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        group_id: str = "memory_service"
    ):
        """Initialize the event bus.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic for memory events
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers or config.kafka.bootstrap_servers
        self.topic = topic or config.kafka.memory_topic
        self.group_id = group_id
        
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[AIOKafkaConsumer] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_producer(self) -> None:
        """Start the Kafka producer."""
        if self.producer is None:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self.producer.start()
    
    async def stop_producer(self) -> None:
        """Stop the Kafka producer."""
        if self.producer is not None:
            await self.producer.stop()
            self.producer = None
    
    async def publish_event(self, event: Union[MemoryEvent, Dict[str, Any]]) -> None:
        """Publish a memory event.
        
        Args:
            event: Event to publish (MemoryEvent or dict)
        """
        await self.start_producer()
        
        if isinstance(event, MemoryEvent):
            event_data = event.dict()
        else:
            event_data = event
            
        await self.producer.send_and_wait(self.topic, event_data)
    
    async def start_consumer(
        self, 
        event_handler: Callable[[MemoryEvent], Any]
    ) -> None:
        """Start the Kafka consumer with an event handler.
        
        Args:
            event_handler: Callback function to handle events
        """
        if self._running:
            return
            
        self._running = True
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest"
        )
        
        await self.consumer.start()
        
        # Start consumer task
        self._consumer_task = asyncio.create_task(
            self._consume_events(event_handler)
        )
    
    async def _consume_events(
        self, 
        event_handler: Callable[[MemoryEvent], Any]
    ) -> None:
        """Consume events from Kafka and process them.
        
        Args:
            event_handler: Callback function to handle events
        """
        try:
            async for message in self.consumer:
                try:
                    event_data = message.value
                    event = MemoryEvent(**event_data)
                    await asyncio.create_task(event_handler(event))
                except Exception as e:
                    print(f"Error processing event: {e}")
        finally:
            await self.consumer.stop()
            self._running = False
    
    async def stop_consumer(self) -> None:
        """Stop the Kafka consumer."""
        if self.consumer is not None:
            self._running = False
            if self._consumer_task:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass
            await self.consumer.stop()
            self.consumer = None


# Global event bus instance
event_bus = EventBus() 
