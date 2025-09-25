#!/usr/bin/env python3
"""
Enhanced Intelligent Literary Agent System
Uses pre-processed PostgreSQL data + RAG + DeepSeek for real-time chat responses
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

logger = logging.getLogger(__name__)

@dataclass
class ProcessedCharacter:
    name: str
    character_type: str
    personality_traits: List[str]
    relationships: Dict[str, str]
    psychological_profile: str
    confidence_score: float
    first_appearance_chapter: int

@dataclass
class ProcessedStoryline:
    title: str
    description: str
    storyline_type: str
    chapters_involved: List[int]
    key_events: List[str]
    importance_score: float

@dataclass 
class ProcessedTimeline:
    event_description: str
    chapter_index: int
    event_type: str
    characters_involved: List[str]
    chronological_order: int
    importance_score: float

class EnhancedIntelligentAgent:
    """Enhanced agent using pre-processed PostgreSQL data + RAG + DeepSeek"""
    
    def __init__(self, 
                 db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
                 qdrant_url: str = "http://localhost:32768", 
                 collection: str = "test_novel2"):
        
        # Database connection with proper encoding
        self.engine = create_engine(db_url,
                                   connect_args={"client_encoding": "utf8"})
        
        # RAG and AI clients  
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        self.collection = collection
        
        # Cache for processed data
        self._characters_cache = None
        self._storylines_cache = None
        self._timeline_cache = None
        self._last_cache_update = None
        
        logger.info("Enhanced Intelligent Agent initialized")
    
    async def get_processed_characters(self) -> List[ProcessedCharacter]:
        """Get characters from processed PostgreSQL data"""
        if self._should_refresh_cache():
            await self._refresh_cache()
        
        return self._characters_cache or []
    
    async def get_processed_storylines(self) -> List[ProcessedStoryline]:
        """Get storylines from processed PostgreSQL data"""
        if self._should_refresh_cache():
            await self._refresh_cache()
            
        return self._storylines_cache or []
    
    async def get_processed_timeline(self) -> List[ProcessedTimeline]:
        """Get timeline events from processed PostgreSQL data"""  
        if self._should_refresh_cache():
            await self._refresh_cache()
            
        return self._timeline_cache or []
    
    async def enhanced_character_chat(self, user_message: str, character_name: str = None) -> Dict[str, Any]:
        """Chat as a specific character using processed data + RAG"""
        
        characters = await self.get_processed_characters()
        
        # Find the character or pick the main character
        target_character = None
        if character_name:
            target_character = next((c for c in characters if character_name in c.name), None)
        
        if not target_character and characters:
            # Pick protagonist or first character
            target_character = next((c for c in characters if c.character_type == 'protagonist'), characters[0])
        
        if not target_character:
            return {
                "response": "抱歉，目前没有发现可以对话的角色。请先进行小说分析。",
                "character": None,
                "context": {}
            }
        
        # Get RAG context
        rag_results = self.rag_client.search_text(user_message, collection=self.collection, limit=3)
        rag_context = "\n\n".join([r['content'][:200] for r in rag_results])
        
        # Build character context from processed data
        character_context = f"""
角色信息：
- 姓名：{target_character.name}
- 类型：{target_character.character_type}  
- 性格特征：{', '.join(target_character.personality_traits)}
- 心理描述：{target_character.psychological_profile}
- 人物关系：{json.dumps(target_character.relationships, ensure_ascii=False)}
"""
        
        # Get storyline context
        storylines = await self.get_processed_storylines()
        storyline_context = ""
        if storylines:
            relevant_storylines = [s for s in storylines[:3]]  # Top 3 storylines
            storyline_context = "\n".join([f"- {s.title}: {s.description[:100]}" for s in relevant_storylines])
        
        # Build roleplay prompt
        roleplay_prompt = f"""你现在要完全以'{target_character.name}'的身份进行对话。

{character_context}

相关故事背景：
{storyline_context}

相关文本内容：
{rag_context}

用户问题：{user_message}

请完全以{target_character.name}的身份、语气和性格特点来回应，表现出角色的独特个性和说话风格。不要提及你是AI，完全沉浸在角色中。"""
        
        try:
            # Create a proper session for async calls
            if not self.deepseek_client.session:
                await self.deepseek_client.initialize()
            
            messages = [{"role": "user", "content": roleplay_prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1000,
                temperature=0.8  # Higher temperature for more creative character responses
            )
            
            if response.get("success"):
                return {
                    "response": response["response"],
                    "character": {
                        "name": target_character.name,
                        "type": target_character.character_type,
                        "traits": target_character.personality_traits[:3]
                    },
                    "context": {
                        "storylines_available": len(storylines),
                        "rag_sources": len(rag_results),
                        "character_confidence": target_character.confidence_score
                    }
                }
            else:
                return {
                    "response": f"抱歉，{target_character.name}现在无法回应您的问题。",
                    "character": {"name": target_character.name},
                    "context": {"error": response.get("error")}
                }
                
        except Exception as e:
            logger.error(f"Character chat error: {e}")
            return {
                "response": "抱歉，对话过程中出现了问题。",
                "character": {"name": target_character.name if target_character else "未知"},
                "context": {"error": str(e)}
            }
    
    async def enhanced_contextual_chat(self, user_message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Enhanced contextual chat using all processed data"""
        
        # Get all processed data
        characters = await self.get_processed_characters()
        storylines = await self.get_processed_storylines()
        timeline = await self.get_processed_timeline()
        
        # Get RAG context
        rag_results = self.rag_client.search_text(user_message, collection=self.collection, limit=4)
        rag_context = "\n\n".join([r['content'][:250] for r in rag_results])
        
        # Build comprehensive context
        context_parts = []
        
        if characters:
            char_info = []
            for char in characters[:5]:  # Top 5 characters
                char_info.append(f"- {char.name} ({char.character_type}): {', '.join(char.personality_traits[:2])}")
            char_info_text = "\n".join(char_info)
            context_parts.append(f"主要角色：\n{char_info_text}")
        
        if storylines:
            story_info = []
            for story in storylines[:3]:  # Top 3 storylines
                story_info.append(f"- {story.title}: {story.description[:80]}")
            story_info_text = "\n".join(story_info)
            context_parts.append(f"故事线索：\n{story_info_text}")
        
        if timeline:
            recent_events = []
            for event in timeline[:3]:  # Recent events
                recent_events.append(f"- {event.event_description}")
            events_text = "\n".join(recent_events)
            context_parts.append(f"重要事件：\n{events_text}")
        
        comprehensive_context = "\n\n".join(context_parts)
        
        # Build intelligent prompt
        chat_prompt = f"""你是一位深谙中文文学的智能助手，对当前讨论的小说有深入了解。

当前小说背景信息：
{comprehensive_context}

相关文本内容：
{rag_context}

用户问题：{user_message}

请结合小说的背景信息、人物关系、故事情节等，提供深入、准确、有见地的回应。如果用户询问特定角色，可以详细介绍其性格特征和在故事中的作用。"""
        
        try:
            # Ensure session is initialized
            if not self.deepseek_client.session:
                await self.deepseek_client.initialize()
                
            messages = [{"role": "user", "content": chat_prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )
            
            if response.get("success"):
                return {
                    "response": response["response"],
                    "context": {
                        "characters_available": [c.name for c in characters],
                        "storylines_tracked": len(storylines),
                        "timeline_events": len(timeline),
                        "rag_sources": len(rag_results),
                        "last_updated": self._last_cache_update.isoformat() if self._last_cache_update else None
                    }
                }
            else:
                return {
                    "response": "抱歉，处理您的问题时遇到了困难。",
                    "context": {"error": response.get("error")}
                }
                
        except Exception as e:
            logger.error(f"Enhanced contextual chat error: {e}")
            return {
                "response": "抱歉，系统暂时无法处理您的请求。",
                "context": {"error": str(e)}
            }
    
    async def get_novel_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary from processed data"""
        
        characters = await self.get_processed_characters()
        storylines = await self.get_processed_storylines()
        timeline = await self.get_processed_timeline()
        
        # Categorize characters
        protagonists = [c for c in characters if c.character_type == 'protagonist']
        antagonists = [c for c in characters if c.character_type == 'antagonist']
        supporting = [c for c in characters if c.character_type == 'supporting']
        
        # Get chapter progress from database
        with self.engine.connect() as conn:
            chapter_count = conn.execute(text("SELECT COUNT(*) FROM chapter_summaries")).scalar() or 0
            
            recent_chapter = None
            if chapter_count > 0:
                recent_result = conn.execute(
                    text("SELECT chapter_title, chapter_index FROM chapter_summaries ORDER BY chapter_index DESC LIMIT 1")
                ).fetchone()
                if recent_result:
                    recent_chapter = {
                        "title": recent_result[0],
                        "index": recent_result[1]
                    }
        
        return {
            "analysis_summary": {
                "chapters_processed": chapter_count,
                "characters_discovered": len(characters),
                "storylines_identified": len(storylines),
                "timeline_events": len(timeline),
                "recent_chapter": recent_chapter
            },
            "character_breakdown": {
                "protagonists": [{"name": c.name, "traits": c.personality_traits[:2]} for c in protagonists],
                "antagonists": [{"name": c.name, "traits": c.personality_traits[:2]} for c in antagonists],
                "supporting": [{"name": c.name, "traits": c.personality_traits[:2]} for c in supporting]
            },
            "storyline_overview": [
                {
                    "title": s.title,
                    "type": s.storyline_type,
                    "importance": s.importance_score,
                    "chapters": s.chapters_involved
                } for s in storylines
            ],
            "system_status": {
                "data_source": "postgresql_processed",
                "last_updated": self._last_cache_update.isoformat() if self._last_cache_update else None,
                "rag_enabled": True,
                "ai_model": "deepseek"
            }
        }
    
    async def explore_topic(self, topic: str, depth: str = 'medium') -> Dict[str, Any]:
        """Explore topic using processed data + RAG + AI"""
        
        # Get relevant processed data
        characters = await self.get_processed_characters()
        storylines = await self.get_processed_storylines()
        
        # Get RAG results for topic
        rag_results = self.rag_client.search_text(topic, collection=self.collection, limit=5)
        rag_content = "\n\n".join([r['content'][:300] for r in rag_results])
        
        # Build exploration context
        context_info = []
        if characters:
            relevant_chars = [c for c in characters if any(trait in topic for trait in c.personality_traits)][:3]
            if relevant_chars:
                char_lines = [f"- {c.name}: {c.psychological_profile[:100]}" for c in relevant_chars]
                char_context = "相关角色：\n" + "\n".join(char_lines)
                context_info.append(char_context)
        
        if storylines:
            relevant_stories = [s for s in storylines if topic.lower() in s.description.lower()][:2]
            if relevant_stories:
                story_lines = [f"- {s.title}: {s.description[:150]}" for s in relevant_stories]
                story_context = "相关情节：\n" + "\n".join(story_lines)
                context_info.append(story_context)
        
        context = "\n\n".join(context_info) if context_info else "当前分析的小说内容"
        
        # Build depth-appropriate prompt
        depth_instructions = {
            'shallow': "请提供简明的概述和基本分析",
            'medium': "请进行中等深度的分析，包含具体例证和解释",
            'deep': "请进行深入分析，包含多角度的详细解读和深层含义"
        }
        
        exploration_prompt = f"""请深入探讨和分析主题：{topic}

背景信息：
{context}

相关文本内容：
{rag_content}

分析要求：{depth_instructions.get(depth, depth_instructions['medium'])}

请结合文学理论、角色心理、情节发展等多个维度进行分析。"""
        
        try:
            # Ensure session is initialized
            if not self.deepseek_client.session:
                await self.deepseek_client.initialize()
                
            messages = [{"role": "user", "content": exploration_prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1500 if depth == 'deep' else 1000,
                temperature=0.6
            )
            
            if response.get("success"):
                return {
                    "topic": topic,
                    "depth": depth,
                    "analysis": response["response"],
                    "sources": {
                        "characters_consulted": len([c for c in characters if any(trait in topic for trait in c.personality_traits)]),
                        "storylines_referenced": len([s for s in storylines if topic.lower() in s.description.lower()]),
                        "rag_sources": len(rag_results)
                    },
                    "context_available": bool(context_info)
                }
            else:
                return {
                    "topic": topic,
                    "analysis": f"抱歉，无法深入分析主题'{topic}'。",
                    "error": response.get("error")
                }
                
        except Exception as e:
            logger.error(f"Topic exploration error: {e}")
            return {
                "topic": topic,
                "analysis": "抱歉，主题探讨过程中出现了问题。",
                "error": str(e)
            }
    
    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed (5 minute TTL)"""
        if not self._last_cache_update:
            return True
        
        time_since_update = datetime.now() - self._last_cache_update
        return time_since_update.total_seconds() > 300  # 5 minutes
    
    async def _refresh_cache(self):
        """Refresh cached data from PostgreSQL"""
        try:
            with self.engine.connect() as conn:
                # Load characters
                chars_result = conn.execute(text("""
                    SELECT name, character_type, personality_traits, relationships, 
                           psychological_profile, confidence_score, first_appearance_chapter
                    FROM character_profiles
                    ORDER BY confidence_score DESC
                """)).fetchall()
                
                self._characters_cache = []
                for row in chars_result:
                    try:
                        traits = json.loads(row[2]) if row[2] else []
                        relations = json.loads(row[3]) if row[3] else {}
                    except:
                        traits, relations = [], {}
                        
                    char = ProcessedCharacter(
                        name=row[0],
                        character_type=row[1],
                        personality_traits=traits,
                        relationships=relations,
                        psychological_profile=row[4] or "",
                        confidence_score=row[5] or 0.0,
                        first_appearance_chapter=row[6] or 1
                    )
                    self._characters_cache.append(char)
                
                # Load storylines - check if table has data
                storyline_count = conn.execute(text("SELECT COUNT(*) FROM storyline_threads")).scalar()
                print(f"Storyline threads count: {storyline_count}")
                
                if storyline_count > 0:
                    stories_result = conn.execute(text("""
                        SELECT title, description, thread_type, chapters_involved, 
                               key_events, importance_score
                        FROM storyline_threads
                        ORDER BY importance_score DESC
                    """)).fetchall()
                else:
                    stories_result = []
                
                self._storylines_cache = []
                for row in stories_result:
                    try:
                        chapters = json.loads(row[3]) if row[3] else []
                        events = json.loads(row[4]) if row[4] else []
                    except:
                        chapters, events = [], []
                        
                    story = ProcessedStoryline(
                        title=row[0] or "未知故事线",
                        description=row[1] or "",
                        storyline_type=row[2] or "unknown",
                        chapters_involved=chapters,
                        key_events=events,
                        importance_score=row[5] or 0.0
                    )
                    self._storylines_cache.append(story)
                
                # Load timeline
                timeline_result = conn.execute(text("""
                    SELECT event_description, chapter_index, event_type, 
                           characters_involved, chronological_order, importance_score
                    FROM timeline_events  
                    ORDER BY chronological_order ASC
                """)).fetchall()
                
                self._timeline_cache = []
                for row in timeline_result:
                    try:
                        chars_involved = json.loads(row[3]) if row[3] else []
                    except:
                        chars_involved = []
                        
                    event = ProcessedTimeline(
                        event_description=row[0] or "",
                        chapter_index=row[1] or 1,
                        event_type=row[2] or "unknown",
                        characters_involved=chars_involved,
                        chronological_order=row[4] or 0,
                        importance_score=row[5] or 0.0
                    )
                    self._timeline_cache.append(event)
                
                self._last_cache_update = datetime.now()
                logger.info(f"Cache refreshed: {len(self._characters_cache)} characters, "
                          f"{len(self._storylines_cache)} storylines, {len(self._timeline_cache)} timeline events")
                
        except Exception as e:
            logger.error(f"Cache refresh error: {e}")
            # Keep existing cache if refresh fails


# Factory function
def create_enhanced_agent(
    db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
    qdrant_url: str = "http://localhost:32768",
    collection: str = "test_novel2"
) -> EnhancedIntelligentAgent:
    """Create enhanced intelligent agent with PostgreSQL integration"""
    return EnhancedIntelligentAgent(db_url, qdrant_url, collection)


# Test the enhanced agent
if __name__ == "__main__":
    async def test_enhanced_agent():
        agent = create_enhanced_agent()
        
        print("Testing enhanced agent with processed data...")
        
        # Test analysis summary
        print("\\n=== Analysis Summary ===")
        summary = await agent.get_novel_analysis_summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        
        # Test character chat
        print("\\n=== Character Chat Test ===")
        char_response = await agent.enhanced_character_chat("你好，你是谁？")
        print(f"Character: {char_response.get('character', {}).get('name', 'Unknown')}")
        print(f"Response: {char_response['response']}")
        
        # Test contextual chat  
        print("\\n=== Contextual Chat Test ===")
        context_response = await agent.enhanced_contextual_chat("请介绍一下这个故事的主要角色")
        print(f"Response: {context_response['response'][:200]}...")
        print(f"Context: {context_response['context']}")
        
    asyncio.run(test_enhanced_agent())