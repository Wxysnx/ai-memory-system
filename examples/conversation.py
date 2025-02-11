"""Example client for the AI Memory System API."""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional

import httpx


class MemorySystemClient:
    """Client for the AI Memory System API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.conversation_id = str(uuid.uuid4())
        self.history: List[Dict[str, Any]] = []
    
    async def send_message(self, user_input: str) -> Dict[str, Any]:
        """Send a message to the API.
        
        Args:
            user_input: User input message
            
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/api/conversation"
        payload = {
            "conversation_id": self.conversation_id,
            "user_input": user_input,
            "metadata": {}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
                
            result = response.json()
            
            # Save to history
            self.history.append({
                "user": user_input,
                "assistant": result["response"],
                "relevant_memories": result.get("relevant_memories", [])
            })
            
            return result
    
    async def search_memories(
        self, 
        query: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """Search for relevant memories.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Search results
        """
        url = f"{self.base_url}/api/memories/search"
        payload = {
            "query": query,
            "conversation_id": self.conversation_id,
            "limit": limit
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
                
            return response.json()
    
    def print_history(self) -> None:
        """Print conversation history."""
        for i, turn in enumerate(self.history, 1):
            print(f"Turn {i}:")
            print(f"User: {turn['user']}")
            print(f"Assistant: {turn['assistant']}")
            if turn.get("relevant_memories"):
                print("Relevant memories:")
                for memory in turn["relevant_memories"]:
                    print(f"  - {memory['content']}")
            print()


async def run_conversation_demo() -> None:
    """Run a conversation demo."""
    client = MemorySystemClient()
    
    print("AI Memory System Demo")
    print("--------------------")
    print("Type 'exit' to quit, 'history' to show conversation history\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
            
        if user_input.lower() == "history":
            client.print_history()
            continue
            
        if user_input.lower().startswith("search:"):
            query = user_input[7:].strip()
            result = await client.search_memories(query)
            print("\nMemory search results:")
            for memory in result["results"]:
                print(f"- {memory['content']}")
            print()
            continue
            
        try:
            result = await client.send_message(user_input)
            print(f"\nAI: {result['response']}\n")
            
            if result.get("relevant_memories"):
                print("(System used these memories to generate the response:)")
                for memory in result["relevant_memories"]:
                    print(f"- {memory['content']}")
                print()
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_conversation_demo()) 
