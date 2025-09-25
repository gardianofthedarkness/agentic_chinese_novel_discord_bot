#!/usr/bin/env python3
"""
Simple Chat Agent - Non-async version for Flask compatibility
Works around asyncio timeout issues in Flask environment
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Database imports
from sqlalchemy import create_engine, text

from simple_rag import create_rag_client

logger = logging.getLogger(__name__)

class SimpleChatAgent:
    """Simple chat agent that works with Flask without asyncio issues"""
    
    def __init__(self, 
                 db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
                 qdrant_url: str = "http://localhost:32768", 
                 collection: str = "test_novel2"):
        
        # Database connection with proper encoding for Chinese characters
        self.engine = create_engine(db_url, 
                                   connect_args={
                                       "client_encoding": "utf8"
                                   },
                                   pool_pre_ping=True,
                                   echo=False)
        
        # RAG client  
        self.rag_client = create_rag_client(qdrant_url)
        self.collection = collection
        
        # Cache for processed data
        self._characters_cache = None
        self._timeline_cache = None
        self._last_cache_update = None
        
        logger.info("Simple Chat Agent initialized")
    
    def get_processed_characters(self) -> List[Dict]:
        """Get characters from processed PostgreSQL data"""
        if self._should_refresh_cache():
            self._refresh_cache()
        
        return self._characters_cache or []
    
    def get_processed_timeline(self) -> List[Dict]:
        """Get timeline events from processed PostgreSQL data"""  
        if self._should_refresh_cache():
            self._refresh_cache()
            
        return self._timeline_cache or []
    
    def simple_chat(self, user_message: str) -> Dict[str, Any]:
        """Simple chat using RAG + processed data without DeepSeek (for testing)"""
        
        logger.info(f"=== CHAT DEBUG SESSION START ===")
        logger.info(f"USER INPUT: '{user_message}'")
        
        # Get processed data
        characters = self.get_processed_characters()
        timeline = self.get_processed_timeline()
        
        logger.info(f"SQL DATABASE RESULTS:")
        logger.info(f"  Characters found: {len(characters)}")
        for i, char in enumerate(characters[:3]):  # Log first 3 characters
            logger.info(f"    [{i+1}] Name: {char.get('name', 'N/A')}")
            logger.info(f"        Type: {char.get('character_type', 'N/A')}")
            logger.info(f"        Traits: {char.get('personality_traits', [])}")
            logger.info(f"        Profile: {char.get('psychological_profile', 'N/A')[:100]}...")
        
        logger.info(f"  Timeline events found: {len(timeline)}")
        for i, event in enumerate(timeline[:3]):  # Log first 3 events
            logger.info(f"    [{i+1}] Event: {event.get('event_description', 'N/A')[:80]}...")
            logger.info(f"        Chapter: {event.get('chapter_index', 'N/A')}")
            logger.info(f"        Type: {event.get('event_type', 'N/A')}")
            logger.info(f"        Characters involved: {event.get('characters_involved', [])}")
        
        # Get RAG context
        logger.info(f"QUERYING RAG SYSTEM with: '{user_message}'")
        rag_results = self.rag_client.search_text(user_message, collection=self.collection, limit=3)
        
        logger.info(f"RAG DATABASE RESULTS:")
        logger.info(f"  RAG results found: {len(rag_results)}")
        for i, result in enumerate(rag_results):
            logger.info(f"    [{i+1}] Score: {result.get('score', 'N/A')}")
            logger.info(f"        Content: {result.get('content', 'N/A')[:100]}...")
            logger.info(f"        Metadata: {result.get('metadata', {})}")
        
        # Build response based on available data
        logger.info(f"RESPONSE GENERATION LOGIC:")
        
        if not characters and not timeline:
            response = "系统正在加载小说数据，请稍后再试。"
            logger.info(f"  Logic: NO DATA AVAILABLE - using fallback response")
        else:
            logger.info(f"  Logic: Analyzing message for response pattern...")
            logger.info(f"  Available data: {len(characters)} chars, {len(timeline)} events, {len(rag_results)} RAG results")
            # Simple rule-based responses
            if "角色" in user_message or "人物" in user_message:
                logger.info(f"  PATTERN MATCHED: Character query ('角色' or '人物')")
                if characters:
                    char_names = [c.get('name', '未知') for c in characters[:5]]
                    response = f"目前发现的主要角色有：{', '.join(char_names)}。这些角色都来自我们分析的小说章节。"
                    logger.info(f"  Generated character list response with {len(char_names)} characters")
                else:
                    response = "角色数据正在处理中，请使用 /memory 命令查看最新状态。"
                    logger.info(f"  No character data available, using processing message")
            
            elif "故事" in user_message or "情节" in user_message:
                logger.info(f"  PATTERN MATCHED: Story query ('故事' or '情节')")
                if timeline:
                    events = [e.get('event_description', '未知事件')[:50] for e in timeline[:3]]
                    response = f"故事的主要事件包括：{'; '.join(events)}。这些都是从小说章节中提取的重要情节点。"
                    logger.info(f"  Generated timeline response with {len(events)} events")
                else:
                    response = "故事情节数据正在处理中。"
                    logger.info(f"  No timeline data available, using processing message")
            
            else:
                logger.info(f"  PATTERN MATCHED: General query - analyzing sub-patterns...")
                # Provide varied responses based on message content and available data
                message_lower = user_message.lower()
                
                # Greeting responses
                if any(word in user_message for word in ["你好", "hello", "hi"]):
                    logger.info(f"  SUB-PATTERN: Greeting detected")
                    response = f"你好！我是这部小说的智能分析助手。我已经分析了 {len(characters)} 个角色和 {len(timeline)} 个重要事件。你想了解什么呢？"
                
                # Questions about specific characters
                elif any(name in user_message for name in [c.get('name', '') for c in characters] if name):
                    logger.info(f"  SUB-PATTERN: Specific character mentioned")
                    # Find mentioned character
                    mentioned_char = None
                    for c in characters:
                        char_name = c.get('name', '')
                        if char_name and char_name in user_message:
                            mentioned_char = c
                            break
                    
                    if mentioned_char:
                        name = mentioned_char.get('name', '未知角色')
                        char_type = mentioned_char.get('character_type', '未知类型')
                        traits = mentioned_char.get('personality_traits', [])
                        traits_text = "、".join(traits[:3]) if traits else "性格分析中"
                        response = f"{name}是小说中的{char_type}角色。主要特征：{traits_text}。这些信息来自我们对小说章节的详细分析。"
                    else:
                        response = "我在角色数据中找到了相关信息，但需要更具体的问题。请尝试询问具体的角色名字。"
                
                # General story questions
                elif any(word in user_message for word in ["什么", "讲", "内容", "关于"]):
                    logger.info(f"  SUB-PATTERN: General story question detected")
                    if rag_results and len(rag_results) > 1:
                        # Use different RAG results for variety
                        import random
                        selected_result = random.choice(rag_results[:2])
                        rag_content = selected_result['content'][:180]
                        response = f"这部小说主要讲述：{rag_content}...\n\n我已经分析了小说的{len(characters)}个主要角色和{len(timeline)}个关键情节点。"
                    else:
                        response = f"这是一部精彩的小说，包含了丰富的角色设定和情节发展。我已经识别了{len(characters)}个角色和{len(timeline)}个重要事件。你可以问我关于角色或情节的具体问题。"
                
                # Help or unclear queries
                elif any(word in user_message for word in ["帮助", "能做什么", "功能"]):
                    response = f"我可以帮你了解这部小说！我已经分析了：\n- {len(characters)}个角色的性格和关系\n- {len(timeline)}个重要的故事事件\n- 小说的内容和情节\n\n你可以问我关于角色、情节或者任何小说相关的问题。"
                
                else:
                    logger.info(f"  SUB-PATTERN: Fallback RAG-based response")
                    # Use RAG for other general queries with some variety
                    if rag_results:
                        # Rotate through different responses to avoid repetition
                        import hashlib
                        hash_val = int(hashlib.md5(user_message.encode()).hexdigest(), 16)
                        result_index = hash_val % len(rag_results)
                        selected_result = rag_results[result_index]
                        rag_content = selected_result['content'][:200]
                        
                        response = f"根据小说内容：{rag_content}...\n\n这是从原文中提取的相关信息。如果你想了解更多，可以问我关于具体角色或情节的问题。"
                    else:
                        response = f"我已经从小说中识别了 {len(characters)} 个角色和 {len(timeline)} 个重要事件。你想了解哪个方面的内容？可以问我关于角色介绍、故事情节或其他相关问题。"
        
        # Final logging
        logger.info(f"FINAL RESPONSE GENERATED:")
        logger.info(f"  Response length: {len(response)} characters")
        logger.info(f"  Response preview: {response[:150]}...")
        logger.info(f"  Context data: {len(characters)} chars, {len(timeline)} events, {len(rag_results)} RAG")
        logger.info(f"=== CHAT DEBUG SESSION END ===")
        
        return {
            "response": response,
            "context": {
                "characters_available": [c.get('name', '未知') for c in characters],
                "timeline_events": len(timeline),
                "rag_sources": len(rag_results),
                "data_status": "loaded" if characters or timeline else "loading"
            }
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary from processed data"""
        
        characters = self.get_processed_characters()
        timeline = self.get_processed_timeline()
        
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
        
        # Categorize characters
        protagonists = [c for c in characters if c.get('character_type') == 'protagonist']
        antagonists = [c for c in characters if c.get('character_type') == 'antagonist']
        supporting = [c for c in characters if c.get('character_type') == 'supporting']
        
        return {
            "analysis_summary": {
                "chapters_processed": chapter_count,
                "characters_discovered": len(characters),
                "storylines_identified": 0,  # Will implement when storylines are fixed
                "timeline_events": len(timeline),
                "recent_chapter": recent_chapter
            },
            "character_breakdown": {
                "protagonists": [{"name": c.get('name', '未知'), "traits": c.get('personality_traits', [])[:2]} for c in protagonists],
                "antagonists": [{"name": c.get('name', '未知'), "traits": c.get('personality_traits', [])[:2]} for c in antagonists],
                "supporting": [{"name": c.get('name', '未知'), "traits": c.get('personality_traits', [])[:2]} for c in supporting]
            },
            "system_status": {
                "data_source": "postgresql_processed_simple",
                "last_updated": self._last_cache_update.isoformat() if self._last_cache_update else None,
                "rag_enabled": True,
                "ai_model": "rule_based_with_rag"
            }
        }
    
    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed (5 minute TTL)"""
        if not self._last_cache_update:
            return True
        
        time_since_update = datetime.now() - self._last_cache_update
        return time_since_update.total_seconds() > 300  # 5 minutes
    
    def _refresh_cache(self):
        """Refresh cached data from PostgreSQL"""
        try:
            with self.engine.connect() as conn:
                # Load characters
                chars_result = conn.execute(text("""
                    SELECT name, character_type, personality_traits, psychological_profile, confidence_score
                    FROM character_profiles
                    ORDER BY confidence_score DESC
                """)).fetchall()
                
                self._characters_cache = []
                for row in chars_result:
                    try:
                        traits = json.loads(row[2]) if row[2] else []
                    except:
                        traits = []
                        
                    char = {
                        'name': row[0],
                        'character_type': row[1],
                        'personality_traits': traits,
                        'psychological_profile': row[3] or "",
                        'confidence_score': row[4] or 0.0
                    }
                    self._characters_cache.append(char)
                
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
                        
                    event = {
                        'event_description': row[0] or "",
                        'chapter_index': row[1] or 1,
                        'event_type': row[2] or "unknown",
                        'characters_involved': chars_involved,
                        'chronological_order': row[4] or 0,
                        'importance_score': row[5] or 0.0
                    }
                    self._timeline_cache.append(event)
                
                self._last_cache_update = datetime.now()
                logger.info(f"Simple cache refreshed: {len(self._characters_cache)} characters, "
                          f"{len(self._timeline_cache)} timeline events")
                
        except Exception as e:
            logger.error(f"Simple cache refresh error: {e}")


# Factory function
def create_simple_agent(
    db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
    qdrant_url: str = "http://localhost:32768",
    collection: str = "test_novel2"
) -> SimpleChatAgent:
    """Create simple chat agent without asyncio issues"""
    return SimpleChatAgent(db_url, qdrant_url, collection)