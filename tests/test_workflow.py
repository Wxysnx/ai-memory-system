"""Test cases for the LangGraph workflow."""

import os
import uuid
from typing import Dict, Any

import pytest
import pytest_asyncio
from langchain_core.messages import HumanMessage

from memory_system.graph import memory_graph, ConversationState
from memory_system.memory.manager import MemoryManager
from memory_system.memory.short_term import ShortTermMemory
from memory_system.memory.long_term import LongTermMemory


@pytest.fixture
def conversation_id():
    """Generate a unique conversation ID for tests."""
    return f"test-{uuid.uuid4()}"


@pytest.fixture
def redis_url():
    """Get Redis URL for tests."""
    return os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture
def mongo_uri():
    """Get MongoDB URI for tests."""
    return os.environ.get("TEST_MONGO_URI", "mongodb://localhost:27017/memory_test")


@pytest_asyncio.fixture
async def test_memory_manager(redis_url, mongo_uri):
    """Set up memory manager for tests."""
    short_term = ShortTermMemory(redis_url=redis_url)
    long_term = LongTermMemory(mongo_uri=mongo_uri)
    
    memory_mgr = MemoryManager(
        short_term=short_term,
        long_term=long_term
    )
    
    await memory_mgr.initialize()
    
    yield memory_mgr
    
    # Clean up
    await short_term.disconnect()
    await long_term.disconnect()


@pytest.fixture
def initial_state(conversation_id):
    """Create initial conversation state for tests."""
    return {
        "conversation_id": conversation_id,
        "current_input": "Hello, how are you?",
        "context": {},
        "messages": [],
        "response": None,
        "relevant_memories": [],
        "importance_score": None
    }


@pytest.mark.asyncio
async def test_memory_graph_end_to_end(test_memory_manager, initial_state):
    """Test the complete memory graph workflow."""
    # Run the workflow
    result = await memory_graph.ainvoke(initial_state)
    
    # Verify that each node executed
    assert "response" in result
    assert result["response"] is not None
    assert "importance_score" in result
    
    # Now test with a second turn in the same conversation
    second_state = {
        "conversation_id": initial_state["conversation_id"],
        "current_input": "What did I just ask you?",
        "context": {},
        "messages": [],
        "response": None,
        "relevant_memories": [],
        "importance_score": None
    }
    
    second_result = await memory_graph.ainvoke(second_state)
    
    # Verify that the second response includes context from the first
    assert second_result["response"] is not None
    assert len(second_result["context"]["recent_messages"]) >= 2
    

@pytest.mark.asyncio
async def test_memory_graph_individual_nodes(test_memory_manager, initial_state):
    """Test individual nodes of the memory graph."""
    from memory_system.graph import retrieve_memories, generate_response, update_memory
    
    # Test retrieve_memories
    retrieve_state = await retrieve_memories(
        initial_state, 
        memory_mgr=test_memory_manager
    )
    assert "context" in retrieve_state
    assert retrieve_state["context"]["conversation_id"] == initial_state["conversation_id"]
    
    # Test generate_response
    generate_state = await generate_response(
        retrieve_state,
        memory_mgr=test_memory_manager
    )
    assert "response" in generate_state
    assert generate_state["response"] is not None
    
    # Test update_memory
    update_state = await update_memory(
        generate_state,
        memory_mgr=test_memory_manager
    )
    assert "importance_score" in update_state
    assert update_state["importance_score"] is not None
    
    # Verify that messages were stored
    messages = await test_memory_manager.short_term.get_messages(
        initial_state["conversation_id"]
    )
    assert len(messages) >= 2  # User message and AI response


@pytest.mark.asyncio
async def test_workflow_with_existing_memories(test_memory_manager, conversation_id):
    """Test workflow with pre-existing memories."""
    # Add some existing memories
    await test_memory_manager.long_term.store(
        conversation_id=conversation_id,
        content="User's name is Alice",
        memory_type="fact",
        importance=0.9
    )
    
    await test_memory_manager.long_term.store(
        conversation_id=conversation_id,
        content="User likes machine learning",
        memory_type="preference",
        importance=0.8
    )
    
    # Create state referencing these memories
    state = {
        "conversation_id": conversation_id,
        "current_input": "What topics do I like?",
        "context": {},
        "messages": [],
        "response": None,
        "relevant_memories": [],
        "importance_score": None
    }
    
    # Run workflow
    result = await memory_graph.ainvoke(state)
    
    # Verify that relevant memories were retrieved
    assert len(result["relevant_memories"]) > 0
    assert any("like" in memory["content"].lower() for memory in result["relevant_memories"]) 
