"""LangGraph workflow definition for memory management."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from memory_system.memory.manager import MemoryManager, memory_manager
from memory_system.config import config


class Memory:
    """Memory workflow state."""
    
    def __init__(
        self,
        conversation_id: str,
        short_term_messages: List[BaseMessage] = None,
        relevant_memories: List[Dict[str, Any]] = None,
        current_input: str = ""
    ):
        self.conversation_id = conversation_id
        self.short_term_messages = short_term_messages or []
        self.relevant_memories = relevant_memories or []
        self.current_input = current_input


class ConversationState(TypedDict):
    """Conversation state dictionary."""
    
    conversation_id: str
    current_input: str
    context: Dict[str, Any]
    messages: List[BaseMessage]
    response: Optional[str]
    relevant_memories: List[Dict[str, Any]]
    importance_score: Optional[float]


async def retrieve_memories(
    state: ConversationState, 
    memory_mgr: Optional[MemoryManager] = None
) -> ConversationState:
    """Retrieve memories for the conversation.
    
    Args:
        state: Current conversation state
        memory_mgr: Optional memory manager instance
        
    Returns:
        Updated state with retrieved memories
    """
    # Use provided or global memory manager
    mgr = memory_mgr or memory_manager
    
    # Get conversation context
    context = await mgr.get_conversation_context(
        conversation_id=state["conversation_id"],
        current_input=state["current_input"]
    )
    
    # Update state
    return {
        **state,
        "context": context,
        "messages": context.get("recent_messages", []),
        "relevant_memories": [m.dict() for m in context.get("relevant_memories", [])]
    }


async def generate_response(
    state: ConversationState,
    llm: Optional[BaseLanguageModel] = None,
    memory_mgr: Optional[MemoryManager] = None
) -> ConversationState:
    """Generate a response using the language model.
    
    Args:
        state: Current conversation state
        llm: Optional language model
        memory_mgr: Optional memory manager
        
    Returns:
        Updated state with generated response
    """
    mgr = memory_mgr or memory_manager
    
    # Format context for LLM
    formatted_context = await mgr.format_context_for_llm(state["context"])
    
    # In a real implementation, this would use the provided LLM
    # For now, we'll create a simple mock response
    response = f"This is a response to: {state['current_input']}"
    
    if state["relevant_memories"]:
        response += f"\n\nI remember: {state['relevant_memories'][0]['content']}"
    
    # Update state
    return {
        **state,
        "response": response
    }


async def update_memory(
    state: ConversationState,
    memory_mgr: Optional[MemoryManager] = None
) -> ConversationState:
    """Update memory with the conversation.
    
    Args:
        state: Current conversation state
        memory_mgr: Optional memory manager
        
    Returns:
        Updated state
    """
    mgr = memory_mgr or memory_manager
    
    # Create messages
    user_message = HumanMessage(content=state["current_input"])
    ai_message = AIMessage(content=state["response"])
    
    # Analyze importance of the exchange
    importance = await mgr.analyze_importance(
        message_content=state["current_input"],
        conversation_context=state["context"]
    )
    
    # Add messages to memory
    await mgr.add_message(
        conversation_id=state["conversation_id"],
        message=user_message
    )
    
    await mgr.add_message(
        conversation_id=state["conversation_id"],
        message=ai_message,
        importance=importance
    )
    
    # Update state
    return {
        **state,
        "importance_score": importance
    }


def build_memory_graph(
    memory_mgr: Optional[MemoryManager] = None
) -> StateGraph:
    """Build the memory workflow graph.
    
    Args:
        memory_mgr: Optional memory manager
        
    Returns:
        Compiled workflow graph
    """
    # Initialize the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("retrieve_memories", retrieve_memories)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("update_memory", update_memory)
    
    # Define the edges
    workflow.add_edge("retrieve_memories", "generate_response")
    workflow.add_edge("generate_response", "update_memory")
    workflow.add_edge("update_memory", END)
    
    # Set the entry point
    workflow.set_entry_point("retrieve_memories")
    
    # Compile the graph
    return workflow.compile()


# Create a global memory graph instance
memory_graph = build_memory_graph() 
