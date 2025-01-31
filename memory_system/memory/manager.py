"""Memory manager for coordinating between memory layers."""

import asyncio
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel

from memory_system.memory.short_term import ShortTermMemory, short_term_memory
from memory_system.memory.long_term import LongTermMemory, long_term_memory


class RetrievedMemory(BaseModel):
    """Container for retrieved memory content."""
    
    content: str
    source: str  # short_term, long_term, etc.
    relevance: float = 1.0
    metadata: Dict[str, Any] = {}


class MemoryManager:
    """Memory manager for coordinating different memory layers."""
    
    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        memory_importance_threshold: float = 0.5
    ):
        """Initialize memory manager.
        
        Args:
            short_term: Short-term memory instance
            long_term: Long-term memory instance
            memory_importance_threshold: Threshold for promoting to long-term memory
        """
        self.short_term = short_term or short_term_memory
        self.long_term = long_term or long_term_memory
        self.memory_importance_threshold = memory_importance_threshold
        
    async def initialize(self) -> None:
        """Initialize memory connections."""
        await self.short_term.connect()
        await self.long_term.connect()
        
    async def add_message(
        self,
        conversation_id: str,
        message: BaseMessage,
        importance: Optional[float] = None
    ) -> None:
        """Add a message to memory.
        
        Args:
            conversation_id: Conversation ID
            message: Message to store
            importance: Optional importance score
        """
        # Always add to short-term memory
        await self.short_term.add_message(conversation_id, message)
        
        # Add to long-term memory if it meets the threshold
        if importance is not None and importance >= self.memory_importance_threshold:
            await self.long_term.store_message(
                conversation_id=conversation_id,
                message=message,
                importance=importance
            )
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        current_input: str,
        max_short_term: int = 10,
        max_long_term: int = 3
    ) -> Dict[str, Any]:
        """Get the full context for a conversation.
        
        Args:
            conversation_id: Conversation ID
            current_input: Current user input
            max_short_term: Maximum short-term messages to include
            max_long_term: Maximum long-term memories to include
            
        Returns:
            Context dict with recent messages and relevant memories
        """
        # Get recent messages
        recent_task = asyncio.create_task(
            self.short_term.get_messages(
                conversation_id=conversation_id,
                limit=max_short_term
            )
        )
        
        # Get relevant long-term memories
        memories_task = asyncio.create_task(
            self.long_term.search(
                query=current_input,
                conversation_id=conversation_id,
                limit=max_long_term
            )
        )
        
        # Wait for both tasks to complete
        recent_messages, relevant_memories = await asyncio.gather(
            recent_task, memories_task
        )
        
        # Format for return
        return {
            "recent_messages": recent_messages,
            "relevant_memories": [
                RetrievedMemory(
                    content=doc.page_content,
                    source="long_term",
                    metadata=doc.metadata
                )
                for doc in relevant_memories
            ],
            "current_input": current_input,
            "conversation_id": conversation_id
        }
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = 5
    ) -> List[RetrievedMemory]:
        """Retrieve memories relevant to a query.
        
        Args:
            query: Search query
            conversation_id: Optional conversation ID filter
            limit: Maximum number of results
            
        Returns:
            List of relevant memories
        """
        documents = await self.long_term.search(
            query=query,
            conversation_id=conversation_id,
            limit=limit
        )
        
        return [
            RetrievedMemory(
                content=doc.page_content,
                source="long_term",
                metadata=doc.metadata
            )
            for doc in documents
        ]
    
    async def analyze_importance(
        self,
        message_content: str,
        conversation_context: Dict[str, Any]
    ) -> float:
        """Analyze the importance of a message.
        
        This would typically use an LLM to score importance.
        For now, we use a simple placeholder implementation.
        
        Args:
            message_content: Message content to analyze
            conversation_context: Context of the conversation
            
        Returns:
            Importance score between 0 and 1
        """
        # This is a placeholder. In a real implementation, this would use an LLM
        # to score the importance of the message for long-term memory.
        
        # Basic heuristic: longer messages might be more important
        length_factor = min(len(message_content) / 100, 1.0)
        
        # More sophisticated implementation would consider:
        # - Named entities present
        # - Sentiment
        # - Relation to previous context
        # - Novel information
        
        return length_factor * 0.7  # Scale down a bit
    
    async def create_summary(self, conversation_id: str) -> str:
        """Create a summary of the conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Summary ID
        """
        return await self.long_term.summarize_conversation(conversation_id)
    
    async def format_context_for_llm(
        self,
        context: Dict[str, Any]
    ) -> str:
        """Format context for LLM input.
        
        Args:
            context: Context dictionary from get_conversation_context
            
        Returns:
            Formatted context string
        """
        parts = []
        
        # Add relevant memories
        if context.get("relevant_memories"):
            parts.append("Relevant information from memory:")
            for i, memory in enumerate(context["relevant_memories"], 1):
                parts.append(f"{i}. {memory.content}")
            parts.append("")
        
        # Add conversation history
        if context.get("recent_messages"):
            parts.append("Conversation history:")
            for msg in context["recent_messages"]:
                if isinstance(msg, HumanMessage):
                    parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    parts.append(f"Assistant: {msg.content}")
                elif isinstance(msg, SystemMessage):
                    parts.append(f"System: {msg.content}")
                else:
                    parts.append(f"Message: {msg.content}")
            parts.append("")
        
        # Add current input
        if context.get("current_input"):
            parts.append(f"Current user input: {context['current_input']}")
        
        return "\n".join(parts)


# Global memory manager instance
memory_manager = MemoryManager() 
