#!/usr/bin/env python3
"""
MCP Character Simulation Server
Integrates RAG-based character simulation with Discord bot capabilities
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from deepseek_integration import DeepSeekClient, DeepSeekConfig, create_deepseek_config


# Configuration Models
class CharacterProfile(BaseModel):
    """Character configuration and personality definition"""
    name: str = Field(..., description="Character name")
    personality: str = Field(..., description="Character personality description")
    speech_patterns: List[str] = Field(default=[], description="Typical speech patterns and phrases")
    background: str = Field("", description="Character background and context")
    conversation_style: str = Field("casual", description="Conversation style (casual/formal/playful)")
    memory_limit: int = Field(10, description="Number of recent conversations to remember")
    temperature: float = Field(0.7, description="LLM temperature for this character")


class ConversationMemory(BaseModel):
    """Conversation history and context tracking"""
    character_name: str
    channel_id: str
    messages: List[Dict[str, Any]] = Field(default=[])
    last_updated: datetime = Field(default_factory=datetime.now)


# Initialize MCP Server
mcp = FastMCP("Character-Simulation-Server")

# Global state management
characters: Dict[str, CharacterProfile] = {}
conversation_memories: Dict[str, ConversationMemory] = {}
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None
deepseek_client: Optional[DeepSeekClient] = None


# Initialize RAG Components
async def initialize_rag_system():
    """Initialize Qdrant client, embedding model, and DeepSeek client"""
    global qdrant_client, embedding_model, deepseek_client
    
    # Initialize Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_client = QdrantClient(qdrant_url)
    
    # Initialize embedding model
    embedding_model = SentenceTransformer("moka-ai/m3e-small")
    
    # Initialize DeepSeek client
    deepseek_config = create_deepseek_config()
    deepseek_client = DeepSeekClient(deepseek_config)
    await deepseek_client.initialize()


# MCP Tools Implementation

@mcp.tool()
def create_character(
    name: str,
    personality: str, 
    speech_patterns: List[str] = [],
    background: str = "",
    conversation_style: str = "casual",
    memory_limit: int = 10,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Create a new character profile for simulation
    
    Args:
        name: Character name
        personality: Character personality description
        speech_patterns: List of typical speech patterns
        background: Character background story
        conversation_style: casual/formal/playful
        memory_limit: Number of recent conversations to remember
        temperature: LLM temperature setting
    """
    character = CharacterProfile(
        name=name,
        personality=personality,
        speech_patterns=speech_patterns,
        background=background,
        conversation_style=conversation_style,
        memory_limit=memory_limit,
        temperature=temperature
    )
    
    characters[name] = character
    
    return {
        "status": "success",
        "message": f"Character '{name}' created successfully",
        "character": character.dict()
    }


@mcp.tool()
def retrieve_character_examples(
    character_name: str,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve relevant conversation examples for character simulation using RAG
    
    Args:
        character_name: Name of the character
        query: Query text to find similar conversations
        top_k: Number of examples to retrieve
    """
    if not qdrant_client or not embedding_model:
        return {"error": "RAG system not initialized"}
    
    try:
        # Generate embedding for query
        query_embedding = embedding_model.encode(query, convert_to_tensor=False).tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name="test_novel2",  # Your existing collection
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.7  # Only return relevant matches
        )
        
        # Format results
        examples = []
        for result in search_results:
            examples.append({
                "score": result.score,
                "text": result.payload.get("chunk", ""),
                "coordinate": result.payload.get("coordinate", []),
                "relevance": "high" if result.score > 0.8 else "medium"
            })
        
        return {
            "character_name": character_name,
            "query": query,
            "examples": examples,
            "count": len(examples)
        }
        
    except Exception as e:
        return {"error": f"Failed to retrieve examples: {str(e)}"}


@mcp.tool()
def simulate_character_response(
    character_name: str,
    user_message: str,
    channel_id: str = "default",
    context_messages: List[Dict[str, str]] = []
) -> Dict[str, Any]:
    """
    Generate character response using RAG-enhanced simulation
    
    Args:
        character_name: Name of character to simulate
        user_message: User's message to respond to
        channel_id: Discord channel ID for context
        context_messages: Recent conversation context
    """
    if character_name not in characters:
        return {"error": f"Character '{character_name}' not found"}
    
    character = characters[character_name]
    
    try:
        # Retrieve relevant examples
        examples_result = retrieve_character_examples(character_name, user_message, top_k=3)
        examples = examples_result.get("examples", [])
        
        # Build context for LLM
        system_prompt = f"""You are roleplaying as {character.name}.

Character Profile:
- Personality: {character.personality}
- Background: {character.background}
- Conversation Style: {character.conversation_style}
- Speech Patterns: {', '.join(character.speech_patterns)}

Based on these example conversations, maintain character consistency:
{chr(10).join([f"Example {i+1}: {ex['text'][:200]}..." for i, ex in enumerate(examples)])}

Respond to the user's message while staying in character. Keep the response natural and conversational."""
        
        # Get conversation memory
        memory_key = f"{character_name}_{channel_id}"
        if memory_key not in conversation_memories:
            conversation_memories[memory_key] = ConversationMemory(
                character_name=character_name,
                channel_id=channel_id
            )
        
        memory = conversation_memories[memory_key]
        
        # Build conversation history
        conversation_context = []
        for msg in memory.messages[-character.memory_limit:]:
            conversation_context.append(msg)
        
        # Add current message to context
        for ctx_msg in context_messages:
            conversation_context.append(ctx_msg)
        
        # Generate response using DeepSeek
        if deepseek_client:
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for ctx in conversation_context:
                messages.append({
                    "role": ctx.get("role", "user"),
                    "content": ctx.get("content", "")
                })
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            result = await deepseek_client.generate_character_response(
                messages=messages,
                temperature=character.temperature,
                max_tokens=300
            )
            
            if result["success"]:
                character_response = result["response"].strip()
            else:
                character_response = f"*{character.name} is thinking...*"
        else:
            # Fallback response if no LLM available
            character_response = f"*{character.name} responds based on their personality: {character.personality[:100]}...*"
        
        # Update conversation memory
        memory.messages.extend([
            {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": character_response, "timestamp": datetime.now().isoformat()}
        ])
        memory.last_updated = datetime.now()
        
        # Trim memory if needed
        if len(memory.messages) > character.memory_limit * 2:
            memory.messages = memory.messages[-character.memory_limit * 2:]
        
        return {
            "character_name": character_name,
            "response": character_response,
            "examples_used": len(examples),
            "context_length": len(conversation_context),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Failed to generate response: {str(e)}"}


@mcp.tool()
def get_character_memory(character_name: str, channel_id: str = "default") -> Dict[str, Any]:
    """
    Retrieve conversation memory for a character in a specific channel
    
    Args:
        character_name: Name of the character
        channel_id: Discord channel ID
    """
    memory_key = f"{character_name}_{channel_id}"
    
    if memory_key not in conversation_memories:
        return {
            "character_name": character_name,
            "channel_id": channel_id,
            "messages": [],
            "message": "No conversation memory found"
        }
    
    memory = conversation_memories[memory_key]
    return {
        "character_name": character_name,
        "channel_id": channel_id,
        "messages": memory.messages,
        "last_updated": memory.last_updated.isoformat(),
        "message_count": len(memory.messages)
    }


@mcp.tool()
def list_characters() -> Dict[str, Any]:
    """List all available characters"""
    return {
        "characters": [
            {
                "name": char.name,
                "personality": char.personality[:100] + "..." if len(char.personality) > 100 else char.personality,
                "conversation_style": char.conversation_style,
                "speech_patterns_count": len(char.speech_patterns)
            }
            for char in characters.values()
        ],
        "count": len(characters)
    }


@mcp.tool()
def clear_character_memory(character_name: str, channel_id: str = "default") -> Dict[str, Any]:
    """
    Clear conversation memory for a character in a specific channel
    
    Args:
        character_name: Name of the character
        channel_id: Discord channel ID  
    """
    memory_key = f"{character_name}_{channel_id}"
    
    if memory_key in conversation_memories:
        del conversation_memories[memory_key]
        return {
            "status": "success", 
            "message": f"Memory cleared for {character_name} in channel {channel_id}"
        }
    else:
        return {
            "status": "info",
            "message": f"No memory found for {character_name} in channel {channel_id}"
        }


# MCP Resources

@mcp.resource("character://{character_name}")
def get_character_profile(character_name: str) -> str:
    """Get detailed character profile information"""
    if character_name not in characters:
        return f"Character '{character_name}' not found"
    
    character = characters[character_name]
    return json.dumps(character.dict(), indent=2)


@mcp.resource("conversations://{character_name}/{channel_id}")
def get_conversation_history(character_name: str, channel_id: str) -> str:
    """Get conversation history for character in specific channel"""
    memory_key = f"{character_name}_{channel_id}"
    
    if memory_key not in conversation_memories:
        return f"No conversation history found for {character_name} in channel {channel_id}"
    
    memory = conversation_memories[memory_key]
    return json.dumps({
        "character_name": character_name,
        "channel_id": channel_id,
        "messages": memory.messages,
        "last_updated": memory.last_updated.isoformat()
    }, indent=2)


# MCP Prompts

@mcp.prompt()
def character_creation_prompt(
    character_type: str = "anime",
    personality_traits: str = "friendly, intelligent",
    setting: str = "modern fantasy"
) -> str:
    """Generate a character creation prompt for LLM-assisted character design"""
    return f"""Create a detailed character profile for a {character_type} character with the following traits: {personality_traits}, set in a {setting} world.

Please provide:
1. Character name
2. Detailed personality description
3. Background story
4. 5-7 typical speech patterns or phrases they would use
5. Their conversation style (casual/formal/playful)
6. Any special quirks or mannerisms

Format the response as a character profile that could be used for roleplaying or character simulation."""


@mcp.prompt()
def conversation_analysis_prompt(conversation_history: str) -> str:
    """Generate a prompt for analyzing conversation patterns"""
    return f"""Analyze the following conversation history to identify character consistency and improvement opportunities:

{conversation_history}

Please evaluate:
1. Character voice consistency
2. Personality trait adherence
3. Speech pattern maintenance
4. Areas for improvement
5. Suggestions for better character simulation

Provide actionable feedback for improving character simulation accuracy."""


# Main entry point
if __name__ == "__main__":
    # Initialize RAG system
    asyncio.run(initialize_rag_system())
    
    # Start MCP server
    mcp.run()