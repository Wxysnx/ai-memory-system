"""FastAPI server for the memory system."""

import asyncio
import time
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from memory_system.config import config
from memory_system.graph import memory_graph, ConversationState
from memory_system.memory.manager import memory_manager
from memory_system.inference import inference_service


app = FastAPI(
    title="AI Memory System",
    description="LangGraph-based memory management system for AI agents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationRequest(BaseModel):
    """Request model for conversation endpoint."""
    
    conversation_id: str
    user_input: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """Response model for conversation endpoint."""
    
    conversation_id: str
    response: str
    relevant_memories: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    """Request model for memory search endpoint."""
    
    query: str
    conversation_id: Optional[str] = None
    limit: int = 5


class MemoryEntry(BaseModel):
    """Memory entry response model."""
    
    id: str
    content: str
    memory_type: str
    conversation_id: str
    importance: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float


class MemorySearchResponse(BaseModel):
    """Response model for memory search endpoint."""
    
    query: str
    results: List[MemoryEntry] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await memory_manager.initialize()
    await inference_service.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    await inference_service.shutdown()


async def get_memory_manager():
    """Dependency for memory manager."""
    return memory_manager


@app.post("/api/conversation", response_model=ConversationResponse)
async def process_conversation(
    request: ConversationRequest,
    memory_mgr: MemoryManager = Depends(get_memory_manager)
):
    """Process a conversation turn.
    
    Args:
        request: Conversation request
        memory_mgr: Memory manager instance
        
    Returns:
        Response with AI response and relevant memories
    """
    try:
        # Initialize state
        state: ConversationState = {
            "conversation_id": request.conversation_id,
            "current_input": request.user_input,
            "context": {},
            "messages": [],
            "response": None,
            "relevant_memories": [],
            "importance_score": None
        }
        
        # Invoke memory graph
        result = await memory_graph.ainvoke(state)
        
        # Format response
        return ConversationResponse(
            conversation_id=request.conversation_id,
            response=result["response"] or "No response generated",
            relevant_memories=result.get("relevant_memories", []),
            metadata={
                "importance_score": result.get("importance_score"),
                "timestamp": time.time()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing conversation: {str(e)}"
        )


@app.post("/api/memories/search", response_model=MemorySearchResponse)
async def search_memories(
    request: MemorySearchRequest,
    memory_mgr: MemoryManager = Depends(get_memory_manager)
):
    """Search for relevant memories.
    
    Args:
        request: Memory search request
        memory_mgr: Memory manager instance
        
    Returns:
        Search results
    """
    try:
        # Retrieve relevant memories
        memories = await memory_mgr.retrieve_relevant_memories(
            query=request.query,
            conversation_id=request.conversation_id,
            limit=request.limit
        )
        
        # Format as response entries
        results = []
        for memory in memories:
            results.append(MemoryEntry(
                id=memory.metadata.get("memory_id", "unknown"),
                content=memory.content,
                memory_type=memory.metadata.get("memory_type", "unknown"),
                conversation_id=memory.metadata.get("conversation_id", "unknown"),
                importance=memory.metadata.get("importance", 0.0),
                metadata=memory.metadata,
                timestamp=memory.metadata.get("timestamp", time.time())
            ))
        
        return MemorySearchResponse(
            query=request.query,
            results=results,
            metadata={
                "count": len(results),
                "timestamp": time.time()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error searching memories: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "memory_system.api:app",
        host=config.api.host,
        port=config.api.port,
        reload=True
    ) 
