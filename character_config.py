#!/usr/bin/env python3
"""
Character Configuration and Memory Management System
Handles character profiles, conversation memory, and persistence
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles

from pydantic import BaseModel, Field, validator
import sqlite3
from contextlib import asynccontextmanager


# Configuration Models
class CharacterProfile(BaseModel):
    """Complete character profile with all configuration options"""
    name: str = Field(..., description="Character name (unique identifier)")
    display_name: str = Field("", description="Display name (can be different from ID)")
    
    # Core personality
    personality: str = Field(..., description="Detailed personality description")
    background: str = Field("", description="Character background and history")
    motivation: str = Field("", description="Character motivations and goals")
    
    # Communication style
    speech_patterns: List[str] = Field(default=[], description="Typical phrases and expressions")
    conversation_style: str = Field("casual", description="casual/formal/playful/dramatic")
    language_quirks: List[str] = Field(default=[], description="Unique language habits")
    
    # Behavioral parameters
    response_probability: float = Field(0.8, description="Probability of responding to messages")
    memory_limit: int = Field(20, description="Number of conversations to remember")
    temperature: float = Field(0.7, description="LLM temperature for responses")
    max_response_length: int = Field(300, description="Maximum tokens in response")
    
    # Emotional settings
    base_mood: str = Field("neutral", description="Default emotional state")
    mood_volatility: float = Field(0.5, description="How quickly mood changes (0-1)")
    emotion_keywords: Dict[str, List[str]] = Field(default={}, description="Keywords that trigger emotions")
    
    # Relationship settings
    relationships: Dict[str, str] = Field(default={}, description="Relationships with other characters/users")
    relationship_memory: bool = Field(True, description="Remember user relationships")
    
    # Activity settings
    active_hours: List[int] = Field(default=list(range(24)), description="Hours when character is active")
    timezone: str = Field("UTC", description="Character's timezone")
    
    # Advanced features
    knowledge_domains: List[str] = Field(default=[], description="Areas of expertise")
    forbidden_topics: List[str] = Field(default=[], description="Topics to avoid")
    custom_prompts: Dict[str, str] = Field(default={}, description="Custom system prompts")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: str = Field("1.0", description="Character profile version")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Character name cannot be empty")
        return v.strip()
    
    @validator('response_probability', 'mood_volatility', 'temperature')
    def validate_probability(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Value must be between 0 and 1")
        return v
    
    @validator('conversation_style')
    def validate_style(cls, v):
        valid_styles = ["casual", "formal", "playful", "dramatic", "mysterious", "friendly"]
        if v not in valid_styles:
            raise ValueError(f"Style must be one of: {', '.join(valid_styles)}")
        return v


class ConversationMessage(BaseModel):
    """Individual conversation message"""
    id: str = Field(..., description="Unique message ID")
    channel_id: str = Field(..., description="Discord channel ID")
    user_id: str = Field(..., description="User Discord ID")
    user_name: str = Field(..., description="User display name")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    message_type: str = Field("user", description="user/character/system")
    emotion_detected: Optional[str] = Field(None, description="Detected emotion")
    context_tags: List[str] = Field(default=[], description="Context classification tags")


class ConversationMemory(BaseModel):
    """Conversation memory for a character in a specific channel"""
    character_name: str
    channel_id: str
    messages: List[ConversationMessage] = Field(default=[])
    
    # Context tracking
    current_topic: Optional[str] = Field(None)
    mood_state: str = Field("neutral")
    relationship_context: Dict[str, str] = Field(default={})
    
    # Statistics
    message_count: int = Field(0)
    last_activity: datetime = Field(default_factory=datetime.now)
    average_response_time: float = Field(0.0, description="Average response time in seconds")
    
    # Memory management
    important_messages: List[str] = Field(default=[], description="IDs of important messages to preserve")
    summary: str = Field("", description="Conversation summary for long-term memory")


class CharacterMemoryManager:
    """Manages character profiles and conversation memory with persistence"""
    
    def __init__(self, data_dir: str = "character_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.profiles_file = self.data_dir / "character_profiles.json"
        self.memory_db = self.data_dir / "conversation_memory.db"
        
        # In-memory caches
        self.character_profiles: Dict[str, CharacterProfile] = {}
        self.conversation_memories: Dict[str, ConversationMemory] = {}
        
        # Database connection
        self.db_initialized = False
    
    async def initialize(self):
        """Initialize the memory manager"""
        await self._init_database()
        await self._load_character_profiles()
        await self._load_conversation_memories()
        print(f"✅ Character Memory Manager initialized with {len(self.character_profiles)} characters")
    
    async def _init_database(self):
        """Initialize SQLite database for conversation storage"""
        if self.db_initialized:
            return
        
        def init_db():
            conn = sqlite3.connect(str(self.memory_db))
            cursor = conn.cursor()
            
            # Conversation messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    character_name TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    message_type TEXT NOT NULL,
                    emotion_detected TEXT,
                    context_tags TEXT,
                    FOREIGN KEY (character_name) REFERENCES characters (name)
                )
            ''')
            
            # Conversation memory metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id TEXT PRIMARY KEY,
                    character_name TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    current_topic TEXT,
                    mood_state TEXT NOT NULL DEFAULT 'neutral',
                    relationship_context TEXT,
                    message_count INTEGER DEFAULT 0,
                    last_activity DATETIME NOT NULL,
                    average_response_time REAL DEFAULT 0.0,
                    important_messages TEXT,
                    summary TEXT,
                    UNIQUE(character_name, channel_id)
                )
            ''')
            
            # Indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_character_channel ON conversation_messages(character_name, channel_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON conversation_messages(timestamp)')
            
            conn.commit()
            conn.close()
        
        # Run database initialization in thread pool
        await asyncio.get_event_loop().run_in_executor(None, init_db)
        self.db_initialized = True
    
    async def _load_character_profiles(self):
        """Load character profiles from JSON file"""
        if not self.profiles_file.exists():
            return
        
        try:
            async with aiofiles.open(self.profiles_file, 'r', encoding='utf-8') as f:
                data = json.loads(await f.read())
            
            for name, profile_data in data.items():
                # Convert string dates back to datetime
                if 'created_at' in profile_data:
                    profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                if 'updated_at' in profile_data:
                    profile_data['updated_at'] = datetime.fromisoformat(profile_data['updated_at'])
                
                self.character_profiles[name] = CharacterProfile(**profile_data)
            
            print(f"📚 Loaded {len(self.character_profiles)} character profiles")
            
        except Exception as e:
            print(f"❌ Error loading character profiles: {e}")
    
    async def _save_character_profiles(self):
        """Save character profiles to JSON file"""
        try:
            # Convert to serializable format
            data = {}
            for name, profile in self.character_profiles.items():
                profile_dict = profile.dict()
                # Convert datetime to string
                profile_dict['created_at'] = profile_dict['created_at'].isoformat()
                profile_dict['updated_at'] = profile_dict['updated_at'].isoformat()
                data[name] = profile_dict
            
            async with aiofiles.open(self.profiles_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
            
            print(f"💾 Saved {len(data)} character profiles")
            
        except Exception as e:
            print(f"❌ Error saving character profiles: {e}")
    
    async def _load_conversation_memories(self):
        """Load conversation memories from database"""
        def load_memories():
            conn = sqlite3.connect(str(self.memory_db))
            cursor = conn.cursor()
            
            # Load memory metadata
            cursor.execute('SELECT * FROM conversation_memory')
            memory_rows = cursor.fetchall()
            
            memories = {}
            
            for row in memory_rows:
                memory_id = row[0]
                character_name = row[1]
                channel_id = row[2]
                
                # Load messages for this memory
                cursor.execute('''
                    SELECT * FROM conversation_messages 
                    WHERE character_name = ? AND channel_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                ''', (character_name, channel_id))
                
                message_rows = cursor.fetchall()
                messages = []
                
                for msg_row in message_rows:
                    messages.append(ConversationMessage(
                        id=msg_row[0],
                        channel_id=msg_row[2],
                        user_id=msg_row[3],
                        user_name=msg_row[4],
                        content=msg_row[5],
                        timestamp=datetime.fromisoformat(msg_row[6]),
                        message_type=msg_row[7],
                        emotion_detected=msg_row[8],
                        context_tags=json.loads(msg_row[9]) if msg_row[9] else []
                    ))
                
                # Create memory object
                memory = ConversationMemory(
                    character_name=character_name,
                    channel_id=channel_id,
                    messages=messages,
                    current_topic=row[3],
                    mood_state=row[4],
                    relationship_context=json.loads(row[5]) if row[5] else {},
                    message_count=row[6],
                    last_activity=datetime.fromisoformat(row[7]),
                    average_response_time=row[8],
                    important_messages=json.loads(row[9]) if row[9] else [],
                    summary=row[10] or ""
                )
                
                memories[memory_id] = memory
            
            conn.close()
            return memories
        
        try:
            memories = await asyncio.get_event_loop().run_in_executor(None, load_memories)
            self.conversation_memories = memories
            print(f"🧠 Loaded {len(memories)} conversation memories")
        except Exception as e:
            print(f"❌ Error loading conversation memories: {e}")
    
    # Character Profile Management
    
    async def create_character(self, profile: CharacterProfile) -> bool:
        """Create a new character profile"""
        try:
            if profile.name in self.character_profiles:
                raise ValueError(f"Character '{profile.name}' already exists")
            
            profile.created_at = datetime.now()
            profile.updated_at = datetime.now()
            
            self.character_profiles[profile.name] = profile
            await self._save_character_profiles()
            
            print(f"✅ Created character: {profile.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating character: {e}")
            return False
    
    async def update_character(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing character profile"""
        try:
            if name not in self.character_profiles:
                raise ValueError(f"Character '{name}' not found")
            
            character = self.character_profiles[name]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(character, field):
                    setattr(character, field, value)
            
            character.updated_at = datetime.now()
            
            await self._save_character_profiles()
            print(f"✅ Updated character: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating character: {e}")
            return False
    
    async def delete_character(self, name: str) -> bool:
        """Delete a character and all associated memories"""
        try:
            if name not in self.character_profiles:
                raise ValueError(f"Character '{name}' not found")
            
            # Delete from profiles
            del self.character_profiles[name]
            await self._save_character_profiles()
            
            # Delete from database
            def delete_from_db():
                conn = sqlite3.connect(str(self.memory_db))
                cursor = conn.cursor()
                cursor.execute('DELETE FROM conversation_messages WHERE character_name = ?', (name,))
                cursor.execute('DELETE FROM conversation_memory WHERE character_name = ?', (name,))
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(None, delete_from_db)
            
            # Remove from memory cache
            to_remove = [k for k in self.conversation_memories.keys() if k.startswith(f"{name}_")]
            for k in to_remove:
                del self.conversation_memories[k]
            
            print(f"✅ Deleted character: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting character: {e}")
            return False
    
    def get_character(self, name: str) -> Optional[CharacterProfile]:
        """Get character profile by name"""
        return self.character_profiles.get(name)
    
    def list_characters(self) -> List[CharacterProfile]:
        """Get all character profiles"""
        return list(self.character_profiles.values())
    
    # Conversation Memory Management
    
    async def add_message(
        self, 
        character_name: str, 
        channel_id: str, 
        message: ConversationMessage
    ) -> bool:
        """Add a message to conversation memory"""
        try:
            memory_key = f"{character_name}_{channel_id}"
            
            # Get or create memory
            if memory_key not in self.conversation_memories:
                self.conversation_memories[memory_key] = ConversationMemory(
                    character_name=character_name,
                    channel_id=channel_id
                )
            
            memory = self.conversation_memories[memory_key]
            
            # Add message
            memory.messages.append(message)
            memory.message_count += 1
            memory.last_activity = datetime.now()
            
            # Maintain memory limit
            character = self.get_character(character_name)
            if character:
                memory_limit = character.memory_limit * 2  # Keep user + character messages
                if len(memory.messages) > memory_limit:
                    # Keep important messages and recent messages
                    important_msgs = [msg for msg in memory.messages if msg.id in memory.important_messages]
                    recent_msgs = memory.messages[-memory_limit//2:]
                    memory.messages = important_msgs + recent_msgs
            
            # Save to database
            await self._save_message_to_db(message)
            await self._save_memory_to_db(memory)
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding message: {e}")
            return False
    
    async def _save_message_to_db(self, message: ConversationMessage):
        """Save message to database"""
        def save_msg():
            conn = sqlite3.connect(str(self.memory_db))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO conversation_messages 
                (id, character_name, channel_id, user_id, user_name, content, timestamp, message_type, emotion_detected, context_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.id,
                message.channel_id.split('_')[0],  # Extract character name from channel context
                message.channel_id,
                message.user_id,
                message.user_name,
                message.content,
                message.timestamp.isoformat(),
                message.message_type,
                message.emotion_detected,
                json.dumps(message.context_tags)
            ))
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(None, save_msg)
    
    async def _save_memory_to_db(self, memory: ConversationMemory):
        """Save memory metadata to database"""
        def save_memory():
            conn = sqlite3.connect(str(self.memory_db))
            cursor = conn.cursor()
            
            memory_id = f"{memory.character_name}_{memory.channel_id}"
            
            cursor.execute('''
                INSERT OR REPLACE INTO conversation_memory 
                (id, character_name, channel_id, current_topic, mood_state, relationship_context, 
                 message_count, last_activity, average_response_time, important_messages, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_id,
                memory.character_name,
                memory.channel_id,
                memory.current_topic,
                memory.mood_state,
                json.dumps(memory.relationship_context),
                memory.message_count,
                memory.last_activity.isoformat(),
                memory.average_response_time,
                json.dumps(memory.important_messages),
                memory.summary
            ))
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(None, save_memory)
    
    def get_conversation_memory(self, character_name: str, channel_id: str) -> Optional[ConversationMemory]:
        """Get conversation memory for character in channel"""
        memory_key = f"{character_name}_{channel_id}"
        return self.conversation_memories.get(memory_key)
    
    async def clear_conversation_memory(self, character_name: str, channel_id: str) -> bool:
        """Clear conversation memory for character in channel"""
        try:
            memory_key = f"{character_name}_{channel_id}"
            
            if memory_key in self.conversation_memories:
                del self.conversation_memories[memory_key]
            
            # Clear from database
            def clear_from_db():
                conn = sqlite3.connect(str(self.memory_db))
                cursor = conn.cursor()
                cursor.execute('DELETE FROM conversation_messages WHERE character_name = ? AND channel_id = ?', 
                             (character_name, channel_id))
                cursor.execute('DELETE FROM conversation_memory WHERE character_name = ? AND channel_id = ?', 
                             (character_name, channel_id))
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(None, clear_from_db)
            
            print(f"✅ Cleared memory for {character_name} in {channel_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")
            return False
    
    # Analytics and Insights
    
    def get_character_statistics(self, character_name: str) -> Dict[str, Any]:
        """Get statistics for a character across all channels"""
        character_memories = [
            memory for key, memory in self.conversation_memories.items() 
            if key.startswith(f"{character_name}_")
        ]
        
        if not character_memories:
            return {"error": "No conversation data found"}
        
        total_messages = sum(memory.message_count for memory in character_memories)
        active_channels = len(character_memories)
        
        # Recent activity
        recent_activity = max(
            (memory.last_activity for memory in character_memories),
            default=datetime.min
        )
        
        # Mood distribution
        moods = [memory.mood_state for memory in character_memories]
        mood_counts = {}
        for mood in moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        return {
            "character_name": character_name,
            "total_messages": total_messages,
            "active_channels": active_channels,
            "recent_activity": recent_activity.isoformat(),
            "mood_distribution": mood_counts,
            "average_response_time": sum(memory.average_response_time for memory in character_memories) / len(character_memories)
        }


# Utility functions

def create_default_character(name: str, personality: str) -> CharacterProfile:
    """Create a character profile with default settings"""
    return CharacterProfile(
        name=name,
        display_name=name,
        personality=personality,
        conversation_style="casual",
        response_probability=0.8,
        memory_limit=20,
        temperature=0.7
    )


async def export_character_data(manager: CharacterMemoryManager, character_name: str, output_path: str):
    """Export all character data to JSON file"""
    character = manager.get_character(character_name)
    if not character:
        raise ValueError(f"Character '{character_name}' not found")
    
    # Collect all memories
    memories = {}
    for key, memory in manager.conversation_memories.items():
        if key.startswith(f"{character_name}_"):
            memories[key] = memory.dict()
    
    export_data = {
        "character_profile": character.dict(),
        "conversation_memories": memories,
        "statistics": manager.get_character_statistics(character_name),
        "export_timestamp": datetime.now().isoformat()
    }
    
    async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(export_data, indent=2, ensure_ascii=False, default=str))
    
    print(f"✅ Exported character data to {output_path}")


# Example usage
async def test_character_config():
    """Test the character configuration system"""
    manager = CharacterMemoryManager("test_character_data")
    await manager.initialize()
    
    # Create test character
    test_char = CharacterProfile(
        name="test_character",
        personality="A friendly and helpful AI assistant",
        speech_patterns=["Hello there!", "How can I help?", "That's interesting!"],
        conversation_style="friendly"
    )
    
    await manager.create_character(test_char)
    
    # Test message handling
    test_message = ConversationMessage(
        id="msg_001",
        channel_id="test_channel",
        user_id="user_123",
        user_name="TestUser",
        content="Hello, how are you?",
        message_type="user"
    )
    
    await manager.add_message("test_character", "test_channel", test_message)
    
    # Get statistics
    stats = manager.get_character_statistics("test_character")
    print(f"📊 Character statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(test_character_config())