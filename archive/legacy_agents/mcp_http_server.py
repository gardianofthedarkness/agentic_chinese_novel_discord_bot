#!/usr/bin/env python3
"""
HTTP Wrapper for MCP Character Simulation Server
Provides HTTP endpoints that the Discord bot can call
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our simplified character system
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from deepseek_integration import DeepSeekClient, DeepSeekConfig, create_deepseek_config
    DEEPSEEK_AVAILABLE = True
except ImportError:
    print("Warning: DeepSeek integration not available")
    DEEPSEEK_AVAILABLE = False

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

# Request/Response Models
class CreateCharacterRequest(BaseModel):
    name: str
    personality: str
    speech_patterns: List[str] = []
    background: str = ""
    conversation_style: str = "casual"
    memory_limit: int = 10
    temperature: float = 0.7

class SimulateResponseRequest(BaseModel):
    character_name: str
    user_message: str
    channel_id: str = "default"
    context_messages: List[Dict[str, str]] = []

class ClearMemoryRequest(BaseModel):
    character_name: str
    channel_id: str = "default"

# Initialize FastAPI
app = FastAPI(title="MCP Character Simulation Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
characters: Dict[str, CharacterProfile] = {}
conversation_memories: Dict[str, ConversationMemory] = {}
deepseek_client: Optional[DeepSeekClient] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup"""
    global deepseek_client
    
    print("Initializing MCP Character Simulation Server...")
    
    # Initialize DeepSeek client if available
    if DEEPSEEK_AVAILABLE:
        try:
            deepseek_config = create_deepseek_config()
            deepseek_client = DeepSeekClient(deepseek_config)
            await deepseek_client.initialize()
            print("✓ DeepSeek client initialized")
        except Exception as e:
            print(f"Warning: DeepSeek initialization failed: {e}")
    else:
        print("ℹ DeepSeek client not available - using fallback responses")
    
    # Create some default characters for testing
    create_character_internal(
        name="神裂火织",
        personality="严肃、强大、责任感强的女魔法师，致力于保护无辜的人",
        speech_patterns=["以坚定的语气说道", "握紧七天七刀", "为了保护重要的人"],
        background="伦敦清教的女教皇，拥有强大的剑术和魔法能力",
        conversation_style="formal"
    )
    
    create_character_internal(
        name="测试助手",
        personality="友善、乐于助人、充满好奇心的AI助手",
        speech_patterns=["我来帮助你", "这很有趣", "让我想想"],
        background="专门为用户提供帮助和建议的智能助手",
        conversation_style="casual"
    )
    
    print(f"Created {len(characters)} default characters")
    print("MCP HTTP Server initialization complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global deepseek_client
    if deepseek_client:
        await deepseek_client.close()
    print("MCP HTTP Server shutdown complete")

# Internal functions
def create_character_internal(
    name: str,
    personality: str, 
    speech_patterns: List[str] = [],
    background: str = "",
    conversation_style: str = "casual",
    memory_limit: int = 10,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Create a character internally"""
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
        "character": character.model_dump()
    }

def retrieve_character_examples_internal(
    character_name: str,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Simulate retrieving relevant conversation examples"""
    # For now, return mock examples
    mock_examples = [
        {
            "score": 0.85,
            "text": f"Example conversation for {character_name} about similar topics",
            "coordinate": [1, 2],
            "relevance": "high"
        },
        {
            "score": 0.75,
            "text": f"Another relevant example for {character_name}",
            "coordinate": [3, 4], 
            "relevance": "medium"
        }
    ]
    
    return {
        "character_name": character_name,
        "query": query,
        "examples": mock_examples[:top_k],
        "count": len(mock_examples[:top_k])
    }

# HTTP Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "characters_count": len(characters),
        "deepseek_available": deepseek_client is not None,
        "version": "1.0.0"
    }

@app.post("/tools/health_check")
async def health_check_tool():
    """Health check as MCP tool format"""
    return await health_check()

@app.post("/tools/create_character")
async def create_character_endpoint(request: CreateCharacterRequest):
    """Create a new character"""
    try:
        result = create_character_internal(
            name=request.name,
            personality=request.personality,
            speech_patterns=request.speech_patterns,
            background=request.background,
            conversation_style=request.conversation_style,
            memory_limit=request.memory_limit,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/list_characters")
async def list_characters_endpoint():
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

@app.post("/tools/simulate_character_response") 
async def simulate_character_response_endpoint(request: SimulateResponseRequest):
    """Generate character response"""
    character_name = request.character_name
    user_message = request.user_message
    channel_id = request.channel_id
    context_messages = request.context_messages
    
    if character_name not in characters:
        return {"error": f"Character '{character_name}' not found"}
    
    character = characters[character_name]
    
    try:
        # Retrieve examples (simplified)
        examples_result = retrieve_character_examples_internal(character_name, user_message, top_k=3)
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
        
        # Generate response using DeepSeek or fallback
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
            character_response = f"*{character.name} responds: I understand you said '{user_message[:50]}...' Let me think about that based on my personality: {character.personality[:100]}...*"
        
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

@app.post("/tools/clear_character_memory")
async def clear_character_memory_endpoint(request: ClearMemoryRequest):
    """Clear conversation memory for a character in a specific channel"""
    memory_key = f"{request.character_name}_{request.channel_id}"
    
    if memory_key in conversation_memories:
        del conversation_memories[memory_key]
        return {
            "status": "success", 
            "message": f"Memory cleared for {request.character_name} in channel {request.channel_id}"
        }
    else:
        return {
            "status": "info",
            "message": f"No memory found for {request.character_name} in channel {request.channel_id}"
        }

@app.get("/resources/character/{character_name}")
async def get_character_profile_endpoint(character_name: str):
    """Get detailed character profile information"""
    if character_name not in characters:
        raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")
    
    character = characters[character_name]
    return character.model_dump()

if __name__ == "__main__":
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get configuration
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    
    print(f"Starting MCP HTTP Server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )