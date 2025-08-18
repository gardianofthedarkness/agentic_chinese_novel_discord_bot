#!/usr/bin/env python3
"""
Enhanced Hybrid Agent - SQL + RAG + DeepSeek with HyDE
Uses SQL for big picture, RAG for detailed chunks, stores all DeepSeek responses
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Database imports
from sqlalchemy import create_engine, text

from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

logger = logging.getLogger(__name__)

@dataclass
class DeepSeekResponse:
    """Store all DeepSeek responses for analysis and improvement"""
    session_id: str
    stage: str  # 'character_analysis', 'storyline_analysis', 'roleplay', 'hyde_query', 'final_response'
    prompt: str
    response: str
    timestamp: datetime
    metadata: Dict[str, Any]

class EnhancedHybridAgent:
    """Advanced agent combining SQL (big picture) + RAG (details) + DeepSeek + HyDE"""
    
    def __init__(self, 
                 db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
                 qdrant_url: str = "http://localhost:32768", 
                 collection: str = "test_novel2"):
        
        # Database connection with proper encoding for PostgreSQL
        self.engine = create_engine(
            db_url, 
            connect_args={
                "client_encoding": "utf8"
            },
            pool_pre_ping=True
        )
        
        # AI and RAG clients
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        self.collection = collection
        
        # Response storage
        self.deepseek_responses: List[DeepSeekResponse] = []
        
        # Cache for SQL data
        self._sql_cache = {}
        self._cache_timestamp = None
        
        logger.info("Enhanced Hybrid Agent initialized with SQL+RAG+DeepSeek+HyDE")
    
    async def get_sql_big_picture(self) -> Dict[str, Any]:
        """Get the big picture from SQL: characters, relationships, timeline, story progress"""
        
        logger.info("=== SQL BIG PICTURE QUERY ===")
        
        # Check cache (5 minute TTL)
        if (self._cache_timestamp and 
            (datetime.now() - self._cache_timestamp).total_seconds() < 300 and
            self._sql_cache):
            logger.info("Using cached SQL data")
            return self._sql_cache
        
        try:
            with self.engine.connect() as conn:
                # Get character overview with proper encoding
                logger.info("Querying character profiles...")
                char_result = conn.execute(text("""
                    SELECT name, character_type, personality_traits, psychological_profile, 
                           relationships, confidence_score, first_appearance_chapter
                    FROM character_profiles 
                    ORDER BY confidence_score DESC
                """)).fetchall()
                
                characters = []
                for row in char_result:
                    try:
                        char_data = {
                            'name': row[0] or 'Unknown',
                            'type': row[1] or 'unknown',
                            'traits': json.loads(row[2]) if row[2] else [],
                            'profile': row[3] or '',
                            'relationships': json.loads(row[4]) if row[4] else {},
                            'confidence': row[5] or 0.0,
                            'first_chapter': row[6] or 1
                        }
                        characters.append(char_data)
                        logger.info(f"  Character: {char_data['name']} ({char_data['type']}) - Traits: {char_data['traits'][:2]}")
                    except Exception as e:
                        logger.warning(f"Error parsing character row: {e}")
                
                # Get timeline overview
                logger.info("Querying timeline events...")
                timeline_result = conn.execute(text("""
                    SELECT event_description, chapter_index, event_type, 
                           characters_involved, chronological_order, importance_score
                    FROM timeline_events 
                    ORDER BY chronological_order ASC
                """)).fetchall()
                
                timeline = []
                for row in timeline_result:
                    try:
                        event_data = {
                            'description': row[0] or '',
                            'chapter': row[1] or 1,
                            'type': row[2] or 'unknown',
                            'characters': json.loads(row[3]) if row[3] else [],
                            'order': row[4] or 0,
                            'importance': row[5] or 0.0
                        }
                        timeline.append(event_data)
                        logger.info(f"  Event: {event_data['description'][:50]}... (Ch.{event_data['chapter']})")
                    except Exception as e:
                        logger.warning(f"Error parsing timeline row: {e}")
                
                # Get story progress
                logger.info("Querying story progress...")
                chapter_result = conn.execute(text("""
                    SELECT chapter_title, chapter_index, content_summary 
                    FROM chapter_summaries 
                    ORDER BY chapter_index DESC LIMIT 1
                """)).fetchone()
                
                story_progress = {}
                if chapter_result:
                    story_progress = {
                        'current_chapter': chapter_result[1],
                        'chapter_title': chapter_result[0],
                        'summary': chapter_result[2] or ''
                    }
                    logger.info(f"  Story progress: Chapter {story_progress['current_chapter']} - {story_progress['chapter_title']}")
                
                self._sql_cache = {
                    'characters': characters,
                    'timeline': timeline,
                    'story_progress': story_progress,
                    'stats': {
                        'character_count': len(characters),
                        'event_count': len(timeline),
                        'chapters_processed': story_progress.get('current_chapter', 0)
                    }
                }
                self._cache_timestamp = datetime.now()
                
                logger.info(f"SQL big picture loaded: {len(characters)} chars, {len(timeline)} events")
                logger.info(f"=== RAW SQL DATA DUMP ===")
                logger.info(f"Characters: {json.dumps(characters, ensure_ascii=False, indent=2)}")
                logger.info(f"Timeline: {json.dumps(timeline, ensure_ascii=False, indent=2)}")
                logger.info(f"Story Progress: {json.dumps(story_progress, ensure_ascii=False, indent=2)}")
                logger.info(f"=== END SQL DATA DUMP ===")
                return self._sql_cache
                
        except Exception as e:
            logger.error(f"SQL big picture query failed: {e}")
            return {
                'characters': [], 'timeline': [], 'story_progress': {}, 
                'stats': {'character_count': 0, 'event_count': 0, 'chapters_processed': 0}
            }
    
    async def hyde_enhanced_rag(self, user_query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use HyDE approach to enhance RAG retrieval with hypothetical document generation"""
        
        logger.info(f"=== HyDE ENHANCED RAG ===")
        logger.info(f"Original query: {user_query}")
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Ensure fresh DeepSeek session for HyDE to avoid event loop conflicts
            if self.deepseek_client.session and not self.deepseek_client.session.closed:
                await self.deepseek_client.close()
            await self.deepseek_client.initialize()
            
            # Step 1: Generate hypothetical document/answer using DeepSeek
            hyde_prompt = f"""基于用户查询，生成一个假设的详细回答，这个回答应该包含用户想要了解的具体信息。

用户查询：{user_query}

故事背景：
- 角色数量：{len(context.get('characters', []))}
- 时间线事件：{len(context.get('timeline', []))}
- 当前进度：第{context.get('story_progress', {}).get('current_chapter', 1)}章

请生成一个详细的假设回答，用于改进文档检索："""
            
            logger.info("Generating HyDE hypothetical document...")
            messages = [{"role": "user", "content": hyde_prompt}]
            hyde_response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            # Store DeepSeek response
            if hyde_response.get("success"):
                hyde_doc = hyde_response["response"]
                logger.info(f"HyDE document generated: {hyde_doc[:100]}...")
                
                self.deepseek_responses.append(DeepSeekResponse(
                    session_id=session_id,
                    stage="hyde_query",
                    prompt=hyde_prompt,
                    response=hyde_doc,
                    timestamp=datetime.now(),
                    metadata={"original_query": user_query}
                ))
                
                # Step 2: Use both original query and hypothetical document for RAG
                combined_query = f"{user_query} {hyde_doc}"
                logger.info("Using combined query for RAG retrieval...")
                logger.info(f"=== RAW HyDE DATA DUMP ===")
                logger.info(f"Original Query: {user_query}")
                logger.info(f"HyDE Prompt: {hyde_prompt}")
                logger.info(f"HyDE Generated Document: {hyde_doc}")
                logger.info(f"Combined Query: {combined_query}")
                logger.info(f"=== END HyDE DATA DUMP ===")
            else:
                logger.warning("HyDE generation failed, using original query")
                combined_query = user_query
            
            # Step 3: Retrieve relevant chunks
            rag_results = self.rag_client.search_text(combined_query, collection=self.collection, limit=5)
            
            logger.info(f"RAG results retrieved: {len(rag_results)}")
            for i, result in enumerate(rag_results):
                logger.info(f"  [{i+1}] Score: {result.get('score', 'N/A')} - {result.get('content', '')[:80]}...")
            
            # Comprehensive RAG results logging
            logger.info(f"=== RAW RAG RESULTS DUMP ===")
            logger.info(f"Combined Query Used: {combined_query}")
            logger.info(f"Number of Results: {len(rag_results)}")
            for i, result in enumerate(rag_results):
                logger.info(f"RAG Result [{i+1}]:")
                logger.info(f"  Score: {result.get('score', 'N/A')}")
                logger.info(f"  Content: {result.get('content', 'No content')}")
                logger.info(f"  Source: {result.get('source', 'Unknown')}")
                logger.info(f"  Metadata: {result.get('metadata', {})}")
            logger.info(f"=== END RAG RESULTS DUMP ===")
            
            return rag_results
            
        except Exception as e:
            logger.error(f"HyDE enhanced RAG failed: {e}")
            # Fallback to regular RAG
            return self.rag_client.search_text(user_query, collection=self.collection, limit=3)
    
    async def intelligent_chat(self, user_message: str, character_name: str = None) -> Dict[str, Any]:
        """Main chat function combining SQL big picture + HyDE RAG + DeepSeek"""
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger.info(f"=== ENHANCED HYBRID CHAT SESSION {session_id} ===")
        logger.info(f"User input: {user_message}")
        logger.info(f"Character roleplay: {character_name if character_name else 'General chat'}")
        
        try:
            # Always ensure fresh DeepSeek session for each request to avoid event loop conflicts
            if self.deepseek_client.session and not self.deepseek_client.session.closed:
                logger.info("Closing existing DeepSeek session to avoid event loop conflicts")
                await self.deepseek_client.close()
            
            # Initialize fresh session for this request
            await self.deepseek_client.initialize()
            logger.info("Fresh DeepSeek session initialized for this request")
            # Step 1: Get SQL big picture
            sql_context = await self.get_sql_big_picture()
            
            # Step 2: Use HyDE enhanced RAG for detailed content
            rag_details = await self.hyde_enhanced_rag(user_message, sql_context)
            
            # Step 3: Build comprehensive context
            if character_name:
                # Character roleplay mode
                target_char = None
                for char in sql_context['characters']:
                    if character_name.lower() in char['name'].lower():
                        target_char = char
                        break
                
                if target_char:
                    logger.info(f"Character roleplay mode: {target_char['name']}")
                    
                    # Build character context
                    char_context = f"""
角色档案：
- 姓名：{target_char['name']}
- 类型：{target_char['type']}
- 性格特征：{', '.join(target_char['traits'])}
- 心理描述：{target_char['profile']}
- 人际关系：{json.dumps(target_char['relationships'], ensure_ascii=False)}
- 首次出现：第{target_char['first_chapter']}章

故事进度：
- 当前章节：第{sql_context['story_progress'].get('current_chapter', 1)}章
- 已知事件：{len(sql_context['timeline'])}个
- 相关角色：{len(sql_context['characters'])}个

相关文本内容：
{chr(10).join([f"- {r['content'][:150]}..." for r in rag_details[:3]])}
"""
                    
                    roleplay_prompt = f"""你现在要完全以'{target_char['name']}'的身份进行对话。

{char_context}

用户问题：{user_message}

请完全以{target_char['name']}的身份、语气和性格特点来回应。基于你的性格特征（{', '.join(target_char['traits'])}）和心理描述来展现角色的独特个性。不要提及你是AI，完全沉浸在角色中。"""
                    
                    # DeepSeek session already initialized at start of intelligent_chat
                    
                    logger.info("Generating character roleplay response...")
                    messages = [{"role": "user", "content": roleplay_prompt}]
                    final_response = await self.deepseek_client.generate_character_response(
                        messages=messages,
                        max_tokens=1200,
                        temperature=0.8
                    )
                    
                    # Store response
                    if final_response.get("success"):
                        # Comprehensive final response logging for character roleplay
                        logger.info(f"=== FINAL DEEPSEEK RESPONSE DUMP (CHARACTER ROLEPLAY) ===")
                        logger.info(f"Session ID: {session_id}")
                        logger.info(f"Character: {target_char['name']}")
                        logger.info(f"User Message: {user_message}")
                        logger.info(f"Full Roleplay Prompt: {roleplay_prompt}")
                        logger.info(f"DeepSeek Raw Response: {final_response['response']}")
                        logger.info(f"DeepSeek Usage Stats: {final_response.get('usage', {})}")
                        logger.info(f"DeepSeek Model: {final_response.get('model', 'unknown')}")
                        logger.info(f"=== END FINAL RESPONSE DUMP ===")
                        
                        self.deepseek_responses.append(DeepSeekResponse(
                            session_id=session_id,
                            stage="character_roleplay",
                            prompt=roleplay_prompt,
                            response=final_response["response"],
                            timestamp=datetime.now(),
                            metadata={
                                "character_name": target_char['name'],
                                "character_type": target_char['type'],
                                "user_message": user_message
                            }
                        ))
                        
                        # Keep session open for future requests
                        # Note: Session will be closed when agent is destroyed
                        
                        return {
                            "response": final_response["response"],
                            "character": {
                                "name": target_char['name'],
                                "type": target_char['type'], 
                                "traits": target_char['traits'][:3]
                            },
                            "context": {
                                "sql_data": sql_context['stats'],
                                "rag_sources": len(rag_details),
                                "session_id": session_id,
                                "mode": "character_roleplay"
                            }
                        }
                else:
                    logger.warning(f"Character '{character_name}' not found")
            
            # Step 4: General contextual chat mode
            logger.info("General contextual chat mode")
            
            # Build comprehensive context from SQL + RAG
            contextual_prompt = f"""你是一位专业的中文文学分析助手，对这部小说有深入了解。

小说整体情况（来自数据库分析）：
- 已分析角色：{sql_context['stats']['character_count']}个
- 重要事件：{sql_context['stats']['event_count']}个  
- 处理进度：第{sql_context['story_progress'].get('current_chapter', 1)}章

主要角色：
{chr(10).join([f"- {char['name']}（{char['type']}）：{', '.join(char['traits'][:2])}" for char in sql_context['characters'][:5]])}

重要事件：
{chr(10).join([f"- 第{event['chapter']}章：{event['description'][:80]}..." for event in sql_context['timeline'][:3]])}

相关文本详情（来自原文检索）：
{chr(10).join([f"- {r['content'][:200]}..." for r in rag_details[:3]])}

用户问题：{user_message}

请结合数据库中的整体分析和原文的具体内容，提供深入、准确且引人入胜的回应。"""
            
            # DeepSeek session already initialized at start of intelligent_chat
            
            logger.info("Generating contextual response...")
            messages = [{"role": "user", "content": contextual_prompt}]
            final_response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )
            
            # Store response
            if final_response.get("success"):
                # Comprehensive final response logging for contextual chat
                logger.info(f"=== FINAL DEEPSEEK RESPONSE DUMP (CONTEXTUAL CHAT) ===")
                logger.info(f"Session ID: {session_id}")
                logger.info(f"User Message: {user_message}")
                logger.info(f"Full Contextual Prompt: {contextual_prompt}")
                logger.info(f"DeepSeek Raw Response: {final_response['response']}")
                logger.info(f"DeepSeek Usage Stats: {final_response.get('usage', {})}")
                logger.info(f"DeepSeek Model: {final_response.get('model', 'unknown')}")
                logger.info(f"SQL Context Stats: {sql_context['stats']}")
                logger.info(f"RAG Sources Count: {len(rag_details)}")
                logger.info(f"=== END FINAL RESPONSE DUMP ===")
                
                self.deepseek_responses.append(DeepSeekResponse(
                    session_id=session_id,
                    stage="contextual_chat",
                    prompt=contextual_prompt,
                    response=final_response["response"],
                    timestamp=datetime.now(),
                    metadata={"user_message": user_message}
                ))
                
                # Keep session open for future requests
                # Note: Session will be closed when agent is destroyed
                
                return {
                    "response": final_response["response"],
                    "context": {
                        "sql_data": sql_context['stats'],
                        "rag_sources": len(rag_details),
                        "session_id": session_id,
                        "mode": "contextual_chat",
                        "hyde_enhanced": True
                    }
                }
            else:
                logger.error(f"DeepSeek generation failed: {final_response.get('error')}")
                logger.info(f"=== DEEPSEEK ERROR DUMP ===")
                logger.info(f"Session ID: {session_id}")
                logger.info(f"User Message: {user_message}")
                logger.info(f"Error: {final_response.get('error')}")
                logger.info(f"Full Error Response: {final_response}")
                logger.info(f"=== END ERROR DUMP ===")
                return {
                    "response": "抱歉，生成回应时遇到了困难。",
                    "context": {"error": final_response.get("error"), "session_id": session_id}
                }
                
        except Exception as e:
            logger.error(f"Enhanced hybrid chat failed: {e}")
            # Keep session open even on error for future retry attempts
            # Session cleanup will be handled by agent destruction
            return {
                "response": f"抱歉，处理您的请求时出现了错误：{str(e)}",
                "context": {"error": str(e), "session_id": session_id}
            }
    
    def get_stored_responses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get stored DeepSeek responses for analysis"""
        recent_responses = self.deepseek_responses[-limit:] if self.deepseek_responses else []
        return [
            {
                "session_id": r.session_id,
                "stage": r.stage,
                "timestamp": r.timestamp.isoformat(),
                "prompt_preview": r.prompt[:100] + "...",
                "response_preview": r.response[:100] + "...",
                "metadata": r.metadata
            }
            for r in recent_responses
        ]


# Factory function
def create_enhanced_hybrid_agent(
    db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
    qdrant_url: str = "http://localhost:32768",
    collection: str = "test_novel2"
) -> EnhancedHybridAgent:
    """Create enhanced hybrid agent"""
    return EnhancedHybridAgent(db_url, qdrant_url, collection)