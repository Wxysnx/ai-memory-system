"""Test cases for the memory modules."""

import asyncio
import os
import time
import uuid
from typing import Dict, Any

import pytest
import pytest_asyncio
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from memory_system.memory.short_term import ShortTermMemory
from memory_system.memory.long_term import LongTermMemory
from memory_system.memory.manager import MemoryManager


# Use a test-specific Redis DB
@pytest.fixture
def redis_url():
    """Get Redis URL for tests."""
    return os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/1")


# Use a test-specific MongoDB database
@pytest.fixture
def mongo_uri():
    """Get MongoDB URI for tests."""
    return os.environ.get("TEST_MONGO_URI", "mongodb://localhost:27017/memory_test")


@pytest_asyncio.fixture
async def short_term_memory(redis_url):
    """Set up short-term memory for tests."""
    memory = ShortTermMemory(redis_url=redis_url)
    await memory.connect()
    yield memory
    # Clean up after tests
    await memory.disconnect()


@pytest_asyncio.fixture
async def long_term_memory(mongo_uri):
    """Set up long-term memory for tests."""
    memory = LongTermMemory(mongo_uri=mongo_uri)
    await memory.connect()
    yield memory
    # Clean up after tests
    await memory.disconnect()


@pytest_asyncio.fixture
async def memory_manager(short_term_memory, long_term_memory):
    """Set up memory manager for tests."""
    manager = MemoryManager(
        short_term=short_term_memory,
        long_term=long_term_memory
    )
    await manager.initialize()
    return manager


@pytest.fixture
def conversation_id():
    """Generate a unique conversation ID for tests."""
    return f"test-{uuid.uuid4()}"


@pytest.mark.asyncio
async def test_short_term_memory_add_retrieve(short_term_memory, conversation_id):
    """Test adding and retrieving messages from short-term memory."""
    # Add messages
    human_msg = HumanMessage(content="Hello, AI!")
    ai_msg = AIMessage(content="Hello, human!")
    
    await short_term_memory.add_message(conversation_id, human_msg)
    await short_term_memory.add_message(conversation_id, ai_msg)
    
    # Retrieve messages
    messages = await short_term_memory.get_messages(conversation_id)
    
    # Verify
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello, AI!"
    assert messages[1].content == "Hello, human!"
    
    # Clean up
    await short_term_memory.clear(conversation_id)


@pytest.mark.asyncio
async def test_long_term_memory_store_search(long_term_memory, conversation_id):
    """Test storing and searching memories in long-term memory."""
    # Store memories
    memory_id1 = await long_term_memory.store(
        conversation_id=conversation_id,
        content="Python is a programming language",
        memory_type="fact",
        importance=0.8
    )
    
    memory_id2 = await long_term_memory.store(
        conversation_id=conversation_id,
        content="JavaScript is used for web development",
        memory_type="fact",
        importance=0.7
    )
    
    # Give the vector index time to update
    await asyncio.sleep(1)
    
    # Search for memories
    results = await long_term_memory.search(
        query="programming languages",
        conversation_id=conversation_id,
        limit=5
    )
    
    # Verify
    assert len(results) > 0
    assert any("Python" in doc.page_content for doc in results)


@pytest.mark.asyncio
async def test_memory_manager_context(memory_manager, conversation_id):
    """Test memory manager's context retrieval."""
    # Add messages
    human_msg = HumanMessage(content="What is Python?")
    ai_msg = AIMessage(content="Python is a programming language.")
    
    await memory_manager.add_message(conversation_id, human_msg)
    await memory_manager.add_message(
        conversation_id, 
        ai_msg, 
        importance=0.8
    )
    
    # Store a fact in long-term memory
    await memory_manager.long_term.store(
        conversation_id=conversation_id,
        content="Python was created by Guido van Rossum",
        memory_type="fact",
        importance=0.9
    )
    
    # Give the vector index time to update
    await asyncio.sleep(1)
    
    # Get context
    context = await memory_manager.get_conversation_context(
        conversation_id=conversation_id,
        current_input="Tell me more about Python"
    )
    
    # Verify
    assert len(context["recent_messages"]) == 2
    assert len(context["relevant_memories"]) > 0
    assert context["current_input"] == "Tell me more about Python"


@pytest.mark.asyncio
async def test_memory_manager_importance_analysis(memory_manager):
    """Test memory manager's importance analysis."""
    # Test with different messages
    importance1 = await memory_manager.analyze_importance(
        message_content="The weather is nice today",
        conversation_context={}
    )
    
    importance2 = await memory_manager.analyze_importance(
        message_content="My password is 123456 and my address is 123 Main St",
        conversation_context={}
    )
    
    # Verify that longer, potentially more important message gets higher score
    assert importance2 > importance1 
