#!/usr/bin/env python3
"""
Timeline Database Management
SQLite database schema and operations for timeline storage and retrieval
"""

import sqlite3
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from timeline_models import TimelineNode, ChapterContext, CharacterState, StoryEvent, UserCharacter, UniverseFingerprint, EventType, CharacterType

logger = logging.getLogger(__name__)


class TimelineDatabase:
    """Database manager for timeline system"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialized = False
    
    async def initialize(self):
        """Initialize database with required tables"""
        if self.initialized:
            return
        
        def init_db():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create timeline nodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS timeline_nodes (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    channel_id TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    story_timestamp DATETIME,
                    
                    -- Chapter context
                    chapter_id TEXT,
                    chapter_title TEXT,
                    setting TEXT,
                    time_period TEXT,
                    main_plot_threads TEXT,  -- JSON array
                    sub_plot_threads TEXT,   -- JSON array
                    recent_events TEXT,      -- JSON array
                    world_state TEXT,        -- JSON object
                    available_locations TEXT, -- JSON array
                    active_npcs TEXT,        -- JSON array
                    story_themes TEXT,       -- JSON array
                    narrative_tone TEXT,
                    genre_tags TEXT,         -- JSON array
                    story_rules TEXT,        -- JSON array
                    character_limitations TEXT, -- JSON object
                    
                    -- Universe fingerprint
                    universe_id TEXT,
                    main_characters TEXT,     -- JSON array (set)
                    current_location TEXT,
                    story_phase TEXT,
                    relationship_summary TEXT, -- JSON object
                    completed_plot_points TEXT, -- JSON array (set)
                    active_conflicts TEXT,     -- JSON array (set)
                    world_state_hash TEXT,
                    fingerprint_updated DATETIME,
                    
                    -- Node metadata
                    conversation_summary TEXT,
                    message_count INTEGER DEFAULT 0,
                    participants TEXT,        -- JSON array
                    is_milestone BOOLEAN DEFAULT FALSE,
                    tags TEXT,               -- JSON array
                    notes TEXT,
                    branch_reason TEXT,
                    branch_weight REAL DEFAULT 1.0,
                    
                    FOREIGN KEY (parent_id) REFERENCES timeline_nodes(id)
                )
            ''')
            
            # Create character states table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS character_states (
                    id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    character_type TEXT NOT NULL, -- 'npc', 'user', 'narrator', 'observer'
                    user_id TEXT,                  -- Discord user ID if user character
                    
                    -- Emotional state
                    emotional_state TEXT DEFAULT 'neutral',
                    energy_level REAL DEFAULT 0.7,
                    mood_stability REAL DEFAULT 0.5,
                    
                    -- Location and context
                    current_location TEXT,
                    recent_actions TEXT,          -- JSON array
                    
                    -- Knowledge and relationships
                    knowledge_items TEXT,         -- JSON array
                    relationships TEXT,           -- JSON object (character_name -> score)
                    secrets_known TEXT,           -- JSON array
                    
                    -- Goals and motivations
                    current_goals TEXT,           -- JSON array
                    motivations TEXT,             -- JSON array
                    
                    -- Story integration
                    story_role TEXT DEFAULT 'participant',
                    plot_importance REAL DEFAULT 0.5,
                    
                    -- User character specific
                    abilities TEXT,               -- JSON array
                    inventory TEXT,               -- JSON array
                    experience_points INTEGER DEFAULT 0,
                    
                    FOREIGN KEY (node_id) REFERENCES timeline_nodes(id) ON DELETE CASCADE,
                    UNIQUE(node_id, character_name)
                )
            ''')
            
            # Create story events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS story_events (
                    id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    summary TEXT NOT NULL,
                    characters_involved TEXT,      -- JSON array
                    user_ids_involved TEXT,        -- JSON array
                    impact_score REAL DEFAULT 1.0,
                    consequences TEXT,             -- JSON array
                    emotional_impact TEXT,         -- JSON object
                    
                    FOREIGN KEY (node_id) REFERENCES timeline_nodes(id) ON DELETE CASCADE
                )
            ''')
            
            # Create user characters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_characters (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    
                    -- Basic character info
                    personality TEXT NOT NULL,
                    background TEXT,
                    appearance TEXT,
                    
                    -- Roleplay mechanics
                    abilities TEXT,               -- JSON array
                    limitations TEXT,             -- JSON array
                    goals TEXT,                   -- JSON array
                    motivations TEXT,             -- JSON array
                    
                    -- Story integration
                    integration_method TEXT DEFAULT 'natural',
                    story_role TEXT DEFAULT 'participant',
                    plot_hooks TEXT,              -- JSON array
                    
                    -- Relationships
                    initial_relationships TEXT,   -- JSON object
                    relationship_preferences TEXT, -- JSON object
                    
                    -- Character progression
                    experience_gained INTEGER DEFAULT 0,
                    character_growth TEXT,        -- JSON array
                    major_achievements TEXT,      -- JSON array
                    
                    -- Meta information
                    created_at DATETIME NOT NULL,
                    last_active DATETIME NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    
                    -- User preferences
                    roleplay_style TEXT DEFAULT 'immersive',
                    communication_style TEXT DEFAULT 'first_person',
                    comfort_level TEXT DEFAULT 'moderate',
                    
                    UNIQUE(user_id, character_name)
                )
            ''')
            
            # Create timeline branches table for easier querying
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS timeline_branches (
                    id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    root_node_id TEXT NOT NULL,
                    current_node_id TEXT NOT NULL,
                    branch_name TEXT,
                    created_at DATETIME NOT NULL,
                    last_active DATETIME NOT NULL,
                    participant_count INTEGER DEFAULT 0,
                    message_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE,
                    
                    FOREIGN KEY (root_node_id) REFERENCES timeline_nodes(id),
                    FOREIGN KEY (current_node_id) REFERENCES timeline_nodes(id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_channel ON timeline_nodes(channel_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_parent ON timeline_nodes(parent_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_created ON timeline_nodes(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_universe ON timeline_nodes(universe_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_location ON timeline_nodes(current_location)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_characters_node ON character_states(node_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_characters_name ON character_states(character_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_characters_user ON character_states(user_id)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_node ON story_events(node_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON story_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON story_events(timestamp)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_userchars_user ON user_characters(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_userchars_active ON user_characters(is_active)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_branches_channel ON timeline_branches(channel_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_branches_active ON timeline_branches(is_active)')
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(None, init_db)
        self.initialized = True
        logger.info(f"✅ Timeline database initialized at {self.db_path}")
    
    # Timeline Node Operations
    
    async def save_timeline_node(self, node: TimelineNode) -> bool:
        """Save a timeline node to database"""
        try:
            def save_node():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Prepare data
                cursor.execute('''
                    INSERT OR REPLACE INTO timeline_nodes (
                        id, parent_id, channel_id, created_at, story_timestamp,
                        chapter_id, chapter_title, setting, time_period,
                        main_plot_threads, sub_plot_threads, recent_events,
                        world_state, available_locations, active_npcs,
                        story_themes, narrative_tone, genre_tags,
                        story_rules, character_limitations,
                        universe_id, main_characters, current_location, story_phase,
                        relationship_summary, completed_plot_points, active_conflicts,
                        world_state_hash, fingerprint_updated,
                        conversation_summary, message_count, participants,
                        is_milestone, tags, notes, branch_reason, branch_weight
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    node.id, node.parent_id, node.channel_id,
                    node.created_at, node.story_timestamp,
                    node.chapter_context.chapter_id, node.chapter_context.title,
                    node.chapter_context.setting, node.chapter_context.time_period,
                    json.dumps(node.chapter_context.main_plot_threads),
                    json.dumps(node.chapter_context.sub_plot_threads),
                    json.dumps(node.chapter_context.recent_events),
                    json.dumps(node.chapter_context.world_state),
                    json.dumps(node.chapter_context.available_locations),
                    json.dumps(node.chapter_context.active_npcs),
                    json.dumps(node.chapter_context.story_themes),
                    node.chapter_context.narrative_tone,
                    json.dumps(node.chapter_context.genre_tags),
                    json.dumps(node.chapter_context.story_rules),
                    json.dumps(node.chapter_context.character_limitations),
                    node.universe_fingerprint.fingerprint_id,
                    json.dumps(list(node.universe_fingerprint.main_characters)),
                    node.universe_fingerprint.current_location,
                    node.universe_fingerprint.story_phase,
                    json.dumps(node.universe_fingerprint.relationship_summary),
                    json.dumps(list(node.universe_fingerprint.completed_plot_points)),
                    json.dumps(list(node.universe_fingerprint.active_conflicts)),
                    node.universe_fingerprint.world_state_hash,
                    node.universe_fingerprint.last_updated,
                    node.conversation_summary, node.message_count,
                    json.dumps(node.participants),
                    node.is_milestone, json.dumps(node.tags),
                    node.notes, node.branch_reason, node.branch_weight
                ))
                
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(None, save_node)
            
            # Save character states
            for char_name, char_state in node.character_states.items():
                await self.save_character_state(node.id, char_state)
            
            # Save events
            for event in node.events:
                await self.save_story_event(node.id, event)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving timeline node {node.id}: {e}")
            return False
    
    async def load_timeline_node(self, node_id: str) -> Optional[TimelineNode]:
        """Load a timeline node from database"""
        try:
            def load_node():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM timeline_nodes WHERE id = ?', (node_id,))
                row = cursor.fetchone()
                conn.close()
                return row
            
            row = await asyncio.get_event_loop().run_in_executor(None, load_node)
            
            if not row:
                return None
            
            # Reconstruct node from database row
            columns = [desc[0] for desc in cursor.description] if 'cursor' in locals() else []
            
            # Create ChapterContext
            chapter_context = ChapterContext(
                chapter_id=row[5] if row[5] else "",
                title=row[6] if row[6] else "",
                setting=row[7] if row[7] else "",
                time_period=row[8] if row[8] else "",
                main_plot_threads=json.loads(row[9]) if row[9] else [],
                sub_plot_threads=json.loads(row[10]) if row[10] else [],
                recent_events=json.loads(row[11]) if row[11] else [],
                world_state=json.loads(row[12]) if row[12] else {},
                available_locations=json.loads(row[13]) if row[13] else [],
                active_npcs=json.loads(row[14]) if row[14] else [],
                story_themes=json.loads(row[15]) if row[15] else [],
                narrative_tone=row[16] if row[16] else "neutral",
                genre_tags=json.loads(row[17]) if row[17] else [],
                story_rules=json.loads(row[18]) if row[18] else [],
                character_limitations=json.loads(row[19]) if row[19] else {}
            )
            
            # Create UniverseFingerprint
            universe_fingerprint = UniverseFingerprint(
                fingerprint_id=row[20] if row[20] else "",
                main_characters=set(json.loads(row[21])) if row[21] else set(),
                current_location=row[22] if row[22] else "",
                story_phase=row[23] if row[23] else "",
                relationship_summary=json.loads(row[24]) if row[24] else {},
                completed_plot_points=set(json.loads(row[25])) if row[25] else set(),
                active_conflicts=set(json.loads(row[26])) if row[26] else set(),
                world_state_hash=row[27] if row[27] else "",
                last_updated=datetime.fromisoformat(row[28]) if row[28] else datetime.now()
            )
            
            # Create TimelineNode
            node = TimelineNode(
                id=row[0],
                parent_id=row[1],
                channel_id=row[2],
                created_at=datetime.fromisoformat(row[3]),
                story_timestamp=datetime.fromisoformat(row[4]) if row[4] else None,
                chapter_context=chapter_context,
                universe_fingerprint=universe_fingerprint,
                conversation_summary=row[29] if row[29] else "",
                message_count=row[30] if row[30] else 0,
                participants=json.loads(row[31]) if row[31] else [],
                is_milestone=bool(row[32]) if row[32] is not None else False,
                tags=json.loads(row[33]) if row[33] else [],
                notes=row[34] if row[34] else "",
                branch_reason=row[35],
                branch_weight=row[36] if row[36] else 1.0
            )
            
            # Load character states
            character_states = await self.load_character_states(node_id)
            node.character_states = {state.character_name: state for state in character_states}
            
            # Load events
            events = await self.load_story_events(node_id)
            node.events = events
            
            # Load children
            children = await self.get_child_node_ids(node_id)
            node.children = children
            
            return node
            
        except Exception as e:
            logger.error(f"❌ Error loading timeline node {node_id}: {e}")
            return None
    
    async def save_character_state(self, node_id: str, character_state: CharacterState) -> bool:
        """Save character state to database"""
        try:
            def save_state():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                state_id = f"{node_id}_{character_state.character_name}"
                
                cursor.execute('''
                    INSERT OR REPLACE INTO character_states (
                        id, node_id, character_name, character_type, user_id,
                        emotional_state, energy_level, mood_stability,
                        current_location, recent_actions,
                        knowledge_items, relationships, secrets_known,
                        current_goals, motivations,
                        story_role, plot_importance,
                        abilities, inventory, experience_points
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    state_id, node_id, character_state.character_name,
                    character_state.character_type.value, character_state.user_id,
                    character_state.emotional_state, character_state.energy_level,
                    character_state.mood_stability, character_state.current_location,
                    json.dumps(character_state.recent_actions),
                    json.dumps(character_state.knowledge_items),
                    json.dumps(character_state.relationships),
                    json.dumps(character_state.secrets_known),
                    json.dumps(character_state.current_goals),
                    json.dumps(character_state.motivations),
                    character_state.story_role, character_state.plot_importance,
                    json.dumps(character_state.abilities),
                    json.dumps(character_state.inventory),
                    character_state.experience_points
                ))
                
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(None, save_state)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving character state: {e}")
            return False
    
    async def load_character_states(self, node_id: str) -> List[CharacterState]:
        """Load all character states for a node"""
        try:
            def load_states():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM character_states WHERE node_id = ?', (node_id,))
                rows = cursor.fetchall()
                conn.close()
                return rows
            
            rows = await asyncio.get_event_loop().run_in_executor(None, load_states)
            
            states = []
            for row in rows:
                state = CharacterState(
                    character_name=row[2],
                    character_type=CharacterType(row[3]),
                    user_id=row[4],
                    emotional_state=row[5] if row[5] else "neutral",
                    energy_level=row[6] if row[6] is not None else 0.7,
                    mood_stability=row[7] if row[7] is not None else 0.5,
                    current_location=row[8] if row[8] else "",
                    recent_actions=json.loads(row[9]) if row[9] else [],
                    knowledge_items=json.loads(row[10]) if row[10] else [],
                    relationships=json.loads(row[11]) if row[11] else {},
                    secrets_known=json.loads(row[12]) if row[12] else [],
                    current_goals=json.loads(row[13]) if row[13] else [],
                    motivations=json.loads(row[14]) if row[14] else [],
                    story_role=row[15] if row[15] else "participant",
                    plot_importance=row[16] if row[16] is not None else 0.5,
                    abilities=json.loads(row[17]) if row[17] else [],
                    inventory=json.loads(row[18]) if row[18] else [],
                    experience_points=row[19] if row[19] is not None else 0
                )
                states.append(state)
            
            return states
            
        except Exception as e:
            logger.error(f"❌ Error loading character states for node {node_id}: {e}")
            return []
    
    async def save_story_event(self, node_id: str, event: StoryEvent) -> bool:
        """Save story event to database"""
        try:
            def save_event():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO story_events (
                        id, node_id, event_type, timestamp, summary,
                        characters_involved, user_ids_involved, impact_score,
                        consequences, emotional_impact
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id, node_id, event.event_type.value, event.timestamp,
                    event.summary, json.dumps(event.characters_involved),
                    json.dumps(event.user_ids_involved), event.impact_score,
                    json.dumps(event.consequences), json.dumps(event.emotional_impact)
                ))
                
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(None, save_event)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving story event: {e}")
            return False
    
    async def load_story_events(self, node_id: str) -> List[StoryEvent]:
        """Load all story events for a node"""
        try:
            def load_events():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM story_events WHERE node_id = ? ORDER BY timestamp', (node_id,))
                rows = cursor.fetchall()
                conn.close()
                return rows
            
            rows = await asyncio.get_event_loop().run_in_executor(None, load_events)
            
            events = []
            for row in rows:
                event = StoryEvent(
                    id=row[0],
                    event_type=EventType(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    summary=row[4],
                    characters_involved=json.loads(row[5]) if row[5] else [],
                    user_ids_involved=json.loads(row[6]) if row[6] else [],
                    impact_score=row[7] if row[7] is not None else 1.0,
                    consequences=json.loads(row[8]) if row[8] else [],
                    emotional_impact=json.loads(row[9]) if row[9] else {}
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error loading story events for node {node_id}: {e}")
            return []
    
    # User Character Operations
    
    async def save_user_character(self, user_character: UserCharacter) -> bool:
        """Save user character to database"""
        try:
            def save_char():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_characters (
                        id, user_id, character_name, personality, background, appearance,
                        abilities, limitations, goals, motivations,
                        integration_method, story_role, plot_hooks,
                        initial_relationships, relationship_preferences,
                        experience_gained, character_growth, major_achievements,
                        created_at, last_active, is_active,
                        roleplay_style, communication_style, comfort_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_character.id, user_character.user_id, user_character.character_name,
                    user_character.personality, user_character.background, user_character.appearance,
                    json.dumps(user_character.abilities), json.dumps(user_character.limitations),
                    json.dumps(user_character.goals), json.dumps(user_character.motivations),
                    user_character.integration_method, user_character.story_role,
                    json.dumps(user_character.plot_hooks),
                    json.dumps(user_character.initial_relationships),
                    json.dumps(user_character.relationship_preferences),
                    user_character.experience_gained, json.dumps(user_character.character_growth),
                    json.dumps(user_character.major_achievements),
                    user_character.created_at, user_character.last_active, user_character.is_active,
                    user_character.roleplay_style, user_character.communication_style,
                    user_character.comfort_level
                ))
                
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(None, save_char)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving user character: {e}")
            return False
    
    async def load_user_character(self, user_id: str, character_name: str) -> Optional[UserCharacter]:
        """Load user character by user ID and character name"""
        try:
            def load_char():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute(
                    'SELECT * FROM user_characters WHERE user_id = ? AND character_name = ?',
                    (user_id, character_name)
                )
                row = cursor.fetchone()
                conn.close()
                return row
            
            row = await asyncio.get_event_loop().run_in_executor(None, load_char)
            
            if not row:
                return None
            
            return UserCharacter(
                id=row[0],
                user_id=row[1],
                character_name=row[2],
                personality=row[3],
                background=row[4] if row[4] else "",
                appearance=row[5] if row[5] else "",
                abilities=json.loads(row[6]) if row[6] else [],
                limitations=json.loads(row[7]) if row[7] else [],
                goals=json.loads(row[8]) if row[8] else [],
                motivations=json.loads(row[9]) if row[9] else [],
                integration_method=row[10] if row[10] else "natural",
                story_role=row[11] if row[11] else "participant",
                plot_hooks=json.loads(row[12]) if row[12] else [],
                initial_relationships=json.loads(row[13]) if row[13] else {},
                relationship_preferences=json.loads(row[14]) if row[14] else {},
                experience_gained=row[15] if row[15] is not None else 0,
                character_growth=json.loads(row[16]) if row[16] else [],
                major_achievements=json.loads(row[17]) if row[17] else [],
                created_at=datetime.fromisoformat(row[18]),
                last_active=datetime.fromisoformat(row[19]),
                is_active=bool(row[20]) if row[20] is not None else True,
                roleplay_style=row[21] if row[21] else "immersive",
                communication_style=row[22] if row[22] else "first_person",
                comfort_level=row[23] if row[23] else "moderate"
            )
            
        except Exception as e:
            logger.error(f"❌ Error loading user character: {e}")
            return None
    
    # Query Operations
    
    async def get_child_node_ids(self, parent_id: str) -> List[str]:
        """Get child node IDs for a parent node"""
        try:
            def get_children():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('SELECT id FROM timeline_nodes WHERE parent_id = ?', (parent_id,))
                rows = cursor.fetchall()
                conn.close()
                return [row[0] for row in rows]
            
            return await asyncio.get_event_loop().run_in_executor(None, get_children)
            
        except Exception as e:
            logger.error(f"❌ Error getting child nodes for {parent_id}: {e}")
            return []
    
    async def find_similar_timelines(self, channel_id: str, universe_hash: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar timeline nodes for parallel universe matching"""
        try:
            def find_similar():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Find nodes with similar universe fingerprints
                cursor.execute('''
                    SELECT id, universe_id, main_characters, current_location, 
                           story_phase, completed_plot_points, active_conflicts,
                           world_state_hash, created_at, message_count
                    FROM timeline_nodes 
                    WHERE channel_id = ? AND world_state_hash != ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (channel_id, universe_hash, limit))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [
                    {
                        "node_id": row[0],
                        "universe_id": row[1],
                        "main_characters": json.loads(row[2]) if row[2] else [],
                        "current_location": row[3],
                        "story_phase": row[4],
                        "completed_plot_points": json.loads(row[5]) if row[5] else [],
                        "active_conflicts": json.loads(row[6]) if row[6] else [],
                        "world_state_hash": row[7],
                        "created_at": row[8],
                        "message_count": row[9]
                    }
                    for row in rows
                ]
            
            return await asyncio.get_event_loop().run_in_executor(None, find_similar)
            
        except Exception as e:
            logger.error(f"❌ Error finding similar timelines: {e}")
            return []
    
    async def get_channel_timelines(self, channel_id: str) -> List[Dict[str, Any]]:
        """Get all timeline roots for a channel"""
        try:
            def get_timelines():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, chapter_title, current_location, created_at, 
                           message_count, is_milestone, tags
                    FROM timeline_nodes 
                    WHERE channel_id = ? AND parent_id IS NULL
                    ORDER BY created_at DESC
                ''', (channel_id,))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [
                    {
                        "node_id": row[0],
                        "chapter_title": row[1],
                        "location": row[2],
                        "created_at": row[3],
                        "message_count": row[4],
                        "is_milestone": bool(row[5]),
                        "tags": json.loads(row[6]) if row[6] else []
                    }
                    for row in rows
                ]
            
            return await asyncio.get_event_loop().run_in_executor(None, get_timelines)
            
        except Exception as e:
            logger.error(f"❌ Error getting channel timelines: {e}")
            return []


# Utility functions for database management

async def initialize_timeline_database(db_path: str) -> TimelineDatabase:
    """Initialize and return timeline database"""
    db = TimelineDatabase(db_path)
    await db.initialize()
    return db


# Example usage and testing
async def test_database():
    """Test database operations"""
    from timeline_models import create_example_timeline
    
    # Initialize database
    db = await initialize_timeline_database("test_timeline.db")
    
    # Create and save example timeline
    example_node = create_example_timeline()
    success = await db.save_timeline_node(example_node)
    print(f"Save success: {success}")
    
    # Load the node back
    loaded_node = await db.load_timeline_node(example_node.id)
    if loaded_node:
        print(f"Loaded node: {loaded_node.id}")
        print(f"Character count: {len(loaded_node.character_states)}")
        print(f"Event count: {len(loaded_node.events)}")
    else:
        print("Failed to load node")


if __name__ == "__main__":
    asyncio.run(test_database())