#!/usr/bin/env python3
"""
Enhanced Agentic Novel Processor
Offline processing system that builds upon existing notebook approach
with modern agentic AI and MCP technologies
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re
from pathlib import Path

# Database imports
from sqlalchemy import create_engine, Table, Column, String, Integer, MetaData, TEXT, DateTime, Float, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import sessionmaker
import uuid

# AI and RAG imports  
from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

logger = logging.getLogger(__name__)

@dataclass
class ChapterSummary:
    chapter_index: int
    chapter_title: str
    content_summary: str
    key_events: List[str]
    characters_introduced: List[str]
    characters_developed: List[str]
    plot_threads: List[str]
    timeline_markers: List[str]
    emotional_arc: str
    created_at: datetime

@dataclass
class CharacterProfile:
    character_id: str
    name: str
    first_appearance_chapter: int
    character_type: str  # protagonist, antagonist, supporting, minor
    personality_traits: List[str]
    relationships: Dict[str, str]
    development_arc: List[Dict[str, Any]]  # Chapter-by-chapter development
    psychological_profile: str
    key_scenes: List[str]
    confidence_score: float
    created_at: datetime

@dataclass 
class StorylineThread:
    thread_id: str
    title: str
    description: str
    thread_type: str  # main_plot, subplot, character_arc, theme
    chapters_involved: List[int]
    key_events: List[Dict[str, Any]]
    resolution_status: str  # ongoing, resolved, abandoned
    importance_score: float
    created_at: datetime

class AgenticNovelProcessor:
    """Enhanced processor based on existing notebook approach"""
    
    def __init__(self, 
                 db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
                 qdrant_url: str = "http://localhost:32768",
                 collection: str = "test_novel2"):
        
        # Initialize database
        self.engine = create_engine(db_url)
        self.metadata = MetaData()
        self._setup_enhanced_tables()
        
        # Initialize AI clients
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        self.collection = collection
        
        # Enhanced prompts based on your notebook approach
        self.prompts = self._setup_enhanced_prompts()
        
        # Processing state
        self.processing_state = {
            'chapters_processed': 0,
            'characters_discovered': 0,
            'storylines_tracked': 0,
            'last_processed': None
        }
    
    def _setup_enhanced_tables(self):
        """Setup enhanced database schema based on existing approach"""
        
        # Enhanced scene_records table (based on your existing schema)
        self.scene_records = Table(
            "enhanced_scene_records",
            self.metadata,
            Column("session_id", String, primary_key=True),
            Column("chapter_index", Integer, index=True),
            Column("chapter_title", String),
            Column("scene_order", Integer),
            Column("scene_type", String),  # dialogue, action, narration, flashback
            Column("content_summary", TEXT),
            Column("characters_involved", JSONB),
            Column("referenced_characters", JSONB),
            Column("emotional_tone", String),
            Column("plot_significance", Float),  # 0-1 importance score
            Column("timeline_markers", JSONB),
            Column("full_content", TEXT),
            Column("ai_analysis", JSONB),  # DeepSeek analysis results
            Column("created_at", DateTime, default=datetime.utcnow),
            Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
        
        # Chapter summaries table (new enhancement)
        self.chapter_summaries = Table(
            "chapter_summaries", 
            self.metadata,
            Column("chapter_id", String, primary_key=True),
            Column("chapter_index", Integer, unique=True, index=True),
            Column("chapter_title", String),
            Column("content_summary", TEXT),
            Column("key_events", JSONB),
            Column("characters_introduced", JSONB),
            Column("characters_developed", JSONB),
            Column("plot_threads", JSONB),
            Column("timeline_markers", JSONB),
            Column("emotional_arc", TEXT),
            Column("word_count", Integer),
            Column("ai_confidence", Float),
            Column("created_at", DateTime, default=datetime.utcnow)
        )
        
        # Character profiles table (enhanced from psychological_log)
        self.character_profiles = Table(
            "character_profiles",
            self.metadata,
            Column("character_id", String, primary_key=True),
            Column("name", String, index=True),
            Column("first_appearance_chapter", Integer),
            Column("character_type", String),  # protagonist, antagonist, etc.
            Column("personality_traits", JSONB),
            Column("relationships", JSONB),
            Column("development_arc", JSONB),  # Chapter-by-chapter changes
            Column("psychological_profile", TEXT),
            Column("key_scenes", JSONB),
            Column("confidence_score", Float),
            Column("created_at", DateTime, default=datetime.utcnow),
            Column("updated_at", DateTime, default=datetime.utcnow)
        )
        
        # Storyline threads table (new)
        self.storyline_threads = Table(
            "storyline_threads",
            self.metadata,
            Column("thread_id", String, primary_key=True),
            Column("title", String),
            Column("description", TEXT),
            Column("thread_type", String),
            Column("chapters_involved", JSONB),
            Column("key_events", JSONB),
            Column("resolution_status", String),
            Column("importance_score", Float),
            Column("created_at", DateTime, default=datetime.utcnow)
        )
        
        # Timeline events table (new)
        self.timeline_events = Table(
            "timeline_events",
            self.metadata,
            Column("event_id", String, primary_key=True),
            Column("chapter_index", Integer, index=True),
            Column("scene_id", String),
            Column("event_type", String),  # character_introduction, plot_point, conflict, resolution
            Column("event_description", TEXT),
            Column("characters_involved", JSONB),
            Column("chronological_order", Integer),  # Overall story timeline order
            Column("importance_score", Float),
            Column("created_at", DateTime, default=datetime.utcnow)
        )
        
        # Create all tables
        self.metadata.create_all(self.engine)
        logger.info("Enhanced database schema created")
    
    def _setup_enhanced_prompts(self) -> Dict[str, str]:
        """Setup enhanced prompts based on notebook approach"""
        
        return {
            'chapter_analysis': """
你是一位专业的中文小说分析专家。请对以下章节内容进行全面分析，提取关键信息。

请按以下JSON格式返回分析结果：
{
  "chapter_summary": {
    "title": "章节标题",
    "content_summary": "简要内容概述（100-200字）",
    "key_events": ["关键事件1", "关键事件2"],
    "emotional_arc": "情感发展轨迹描述",
    "plot_significance": 0.8
  },
  "characters_analysis": [
    {
      "name": "人物姓名", 
      "role_in_chapter": "在本章中的作用",
      "character_development": "性格或关系的变化",
      "psychological_state": "心理状态描述"
    }
  ],
  "storyline_threads": [
    {
      "thread_title": "故事线索标题",
      "thread_type": "main_plot/subplot/character_arc",
      "development": "在本章的发展情况"
    }
  ],
  "timeline_markers": [
    {
      "event": "重要时间节点",
      "chronological_significance": "时间线意义"
    }
  ]
}

章节内容：
{content}

请提供详细准确的分析：
""",

            'character_discovery': """
作为中文文学专家，请分析以下文本，识别和分析其中的人物角色。

重点关注：
1. 人物姓名和身份
2. 性格特征和心理状态
3. 人物关系网络
4. 在故事中的重要性
5. 发展轨迹和变化

请以结构化JSON格式返回：
{
  "characters": [
    {
      "name": "姓名",
      "character_type": "protagonist/antagonist/supporting/minor",
      "personality_traits": ["特征1", "特征2"],
      "relationships": {"人物A": "关系描述"},
      "importance_score": 0.9,
      "psychological_profile": "心理描述"
    }
  ]
}

文本内容：
{content}
""",

            'storyline_extraction': """
请分析以下内容，提取和跟踪故事线索。

分析要点：
1. 主要故事线索
2. 子情节发展  
3. 人物成长线
4. 主题线索
5. 冲突和解决

请返回JSON格式：
{
  "storylines": [
    {
      "title": "故事线标题",
      "type": "main_plot/subplot/character_arc/theme",
      "description": "详细描述",
      "key_events": ["事件1", "事件2"],
      "characters_involved": ["人物1", "人物2"],
      "resolution_status": "ongoing/resolved/abandoned",
      "importance": 0.8
    }
  ]
}

内容：
{content}
""",

            'timeline_construction': """
请分析以下内容，构建时间线和事件顺序。

关注：
1. 时间标记和线索
2. 事件的因果关系
3. 叙述顺序vs时间顺序
4. 回忆和闪回
5. 关键转折点

请返回：
{
  "timeline_events": [
    {
      "event": "事件描述",
      "chronological_order": 1,
      "narrative_order": 1,
      "event_type": "introduction/conflict/climax/resolution",
      "characters": ["相关人物"],
      "significance": 0.9
    }
  ]
}

内容：
{content}
"""
        }
    
    async def process_novel_systematically(self, max_chapters: int = None) -> Dict[str, Any]:
        """
        Main processing loop - enhanced version of notebook approach
        Processes novel chapter by chapter with full analysis
        """
        logger.info("Starting systematic novel processing...")
        
        try:
            # Get novel content from RAG
            chapters_content = await self._get_chapters_content(max_chapters)
            
            results = {
                'chapters_processed': 0,
                'characters_discovered': {},
                'storylines_tracked': {},
                'timeline_constructed': [],
                'processing_errors': []
            }
            
            # Process each chapter
            for chapter_idx, chapter_content in chapters_content:
                try:
                    logger.info(f"Processing chapter {chapter_idx}...")
                    
                    # Analyze chapter comprehensively
                    chapter_analysis = await self._analyze_chapter_comprehensively(
                        chapter_idx, chapter_content
                    )
                    
                    # Store results in database
                    await self._store_chapter_analysis(chapter_idx, chapter_analysis)
                    
                    # Update tracking
                    results['chapters_processed'] += 1
                    
                    # Merge characters
                    if 'characters' in chapter_analysis:
                        for char in chapter_analysis['characters']:
                            char_name = char['name']
                            if char_name not in results['characters_discovered']:
                                results['characters_discovered'][char_name] = char
                            else:
                                # Merge character development across chapters
                                existing = results['characters_discovered'][char_name]
                                existing = self._merge_character_development(existing, char, chapter_idx)
                    
                    # Track storylines
                    if 'storylines' in chapter_analysis:
                        for storyline in chapter_analysis['storylines']:
                            thread_id = storyline.get('title', f'thread_{len(results["storylines_tracked"])}')
                            if thread_id not in results['storylines_tracked']:
                                results['storylines_tracked'][thread_id] = storyline
                                storyline['chapters_involved'] = [chapter_idx]
                            else:
                                results['storylines_tracked'][thread_id]['chapters_involved'].append(chapter_idx)
                    
                    # Build timeline
                    if 'timeline_events' in chapter_analysis:
                        for event in chapter_analysis['timeline_events']:
                            event['chapter_index'] = chapter_idx
                            results['timeline_constructed'].append(event)
                    
                    logger.info(f"Chapter {chapter_idx} processed successfully")
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Error processing chapter {chapter_idx}: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    results['processing_errors'].append({
                        'chapter': chapter_idx,
                        'error': str(e)
                    })
            
            # Post-process: build unified timeline
            await self._build_unified_timeline(results['timeline_constructed'])
            
            # Update processing state
            self.processing_state.update({
                'chapters_processed': results['chapters_processed'],
                'characters_discovered': len(results['characters_discovered']),
                'storylines_tracked': len(results['storylines_tracked']),
                'last_processed': datetime.now().isoformat()
            })
            
            logger.info(f"Novel processing complete: {results['chapters_processed']} chapters processed")
            return results
            
        except Exception as e:
            logger.error(f"Novel processing error: {e}")
            return {'error': str(e)}
    
    async def _get_chapters_content(self, max_chapters: int = None) -> List[Tuple[int, str]]:
        """Get chapter content from RAG, organized by chapters"""
        
        # Search for chapter markers
        chapter_results = self.rag_client.search_text(
            query="章节 第一章 第二章", 
            collection=self.collection, 
            limit=max_chapters or 50
        )
        
        chapters = []
        for i, result in enumerate(chapter_results):
            content = result['content']
            # Simple chapter detection - can be enhanced
            if any(marker in content for marker in ['第', '章', '节']):
                chapters.append((i+1, content))
        
        return chapters[:max_chapters] if max_chapters else chapters
    
    async def _analyze_chapter_comprehensively(self, chapter_idx: int, content: str) -> Dict[str, Any]:
        """Enhanced chapter analysis using DeepSeek AI"""
        
        analysis_results = {}
        
        # Chapter-level analysis  
        # Use safe string replacement to avoid KeyError from braces in content
        chapter_prompt = self.prompts['chapter_analysis'].replace('{content}', content)
        logger.info(f"Sending chapter analysis prompt...")
        chapter_response = await self._query_deepseek(chapter_prompt)
        
        logger.info(f"Chapter response received: {bool(chapter_response)}")
        if chapter_response:
            logger.info(f"Raw chapter response: {chapter_response[:100]}")
            try:
                # Clean up markdown code blocks if present
                cleaned_response = self._extract_json_from_markdown(chapter_response)
                logger.info(f"Cleaned chapter response: {cleaned_response[:200]}")
                chapter_data = json.loads(cleaned_response)
                analysis_results.update(chapter_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chapter analysis JSON for chapter {chapter_idx}: {e}")
                analysis_results['chapter_summary'] = {
                    'title': f'第{chapter_idx}章',
                    'content_summary': chapter_response[:200],
                    'key_events': [],
                    'emotional_arc': '',
                    'plot_significance': 0.5
                }
        else:
            logger.warning(f"No chapter response received")
        
        # Character analysis
        char_prompt = self.prompts['character_discovery'].replace('{content}', content)
        char_response = await self._query_deepseek(char_prompt)
        
        if char_response:
            try:
                cleaned_char_response = self._extract_json_from_markdown(char_response)
                char_data = json.loads(cleaned_char_response)
                analysis_results['characters'] = char_data.get('characters', [])
            except json.JSONDecodeError:
                analysis_results['characters'] = []
        
        # Storyline analysis
        story_prompt = self.prompts['storyline_extraction'].replace('{content}', content)
        story_response = await self._query_deepseek(story_prompt)
        
        if story_response:
            try:
                cleaned_story_response = self._extract_json_from_markdown(story_response)
                story_data = json.loads(cleaned_story_response)
                analysis_results['storylines'] = story_data.get('storylines', [])
            except json.JSONDecodeError:
                analysis_results['storylines'] = []
        
        # Timeline analysis
        timeline_prompt = self.prompts['timeline_construction'].replace('{content}', content)
        timeline_response = await self._query_deepseek(timeline_prompt)
        
        if timeline_response:
            try:
                cleaned_timeline_response = self._extract_json_from_markdown(timeline_response)
                timeline_data = json.loads(cleaned_timeline_response)
                analysis_results['timeline_events'] = timeline_data.get('timeline_events', [])
            except json.JSONDecodeError:
                analysis_results['timeline_events'] = []
        
        return analysis_results
    
    async def _query_deepseek(self, prompt: str) -> Optional[str]:
        """Query DeepSeek AI with error handling"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=2000,
                temperature=0.2
            )
            
            if response.get("success"):
                return response["response"]
            else:
                logger.error(f"DeepSeek query failed: {response.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek query error: {e}")
            return None
    
    async def _store_chapter_analysis(self, chapter_idx: int, analysis: Dict[str, Any]):
        """Store chapter analysis results in database"""
        
        with self.engine.connect() as conn:
            # Store chapter summary
            if 'chapter_summary' in analysis:
                chapter_data = analysis['chapter_summary']
                chapter_id = f"chapter_{chapter_idx}"
                
                conn.execute(self.chapter_summaries.insert().values(
                    chapter_id=chapter_id,
                    chapter_index=chapter_idx,
                    chapter_title=chapter_data.get('title', f'第{chapter_idx}章'),
                    content_summary=chapter_data.get('content_summary', ''),
                    key_events=json.dumps(chapter_data.get('key_events', []), ensure_ascii=False),
                    emotional_arc=chapter_data.get('emotional_arc', ''),
                    ai_confidence=chapter_data.get('plot_significance', 0.5)
                ))
            
            # Store character profiles
            if 'characters' in analysis:
                for char in analysis['characters']:
                    char_id = f"{char['name']}_{chapter_idx}"
                    
                    conn.execute(self.character_profiles.insert().values(
                        character_id=char_id,
                        name=char['name'],
                        first_appearance_chapter=chapter_idx,
                        character_type=char.get('character_type', 'unknown'),
                        personality_traits=json.dumps(char.get('personality_traits', []), ensure_ascii=False),
                        relationships=json.dumps(char.get('relationships', {}), ensure_ascii=False),
                        psychological_profile=char.get('psychological_profile', ''),
                        confidence_score=char.get('importance_score', 0.5)
                    ))
            
            conn.commit()
            
    def _extract_json_from_markdown(self, response_text: str) -> str:
        """Extract JSON from markdown code blocks if present"""
        import re
        
        logger.info(f"Extracting JSON from response: {response_text[:100]}")
        
        # Pattern for markdown code blocks with optional json language identifier
        # More flexible pattern that handles different markdown formats
        markdown_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        match = re.search(markdown_pattern, response_text, re.DOTALL | re.MULTILINE)
        
        if match:
            # Extract the JSON content from the code block
            json_content = match.group(1).strip()
            logger.info(f"Found markdown JSON: {json_content[:100]}")
            return json_content
        else:
            # Try to find JSON-like patterns in the text
            # Look for balanced braces that could contain JSON
            import re
            
            # Find the first { and the matching }
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            json_content = response_text[start_idx:i+1]
                            logger.info(f"Found JSON by brace matching: {json_content[:100]}")
                            return json_content
                            
            # If no balanced braces found, return original text
            logger.warning(f"No JSON pattern found, returning original")
            return response_text.strip()
    
    def _merge_character_development(self, existing: Dict, new: Dict, chapter_idx: int) -> Dict:
        """Merge character development across chapters"""
        
        # Merge personality traits
        existing_traits = set(existing.get('personality_traits', []))
        new_traits = set(new.get('personality_traits', []))
        merged_traits = list(existing_traits.union(new_traits))
        
        # Merge relationships
        existing_relations = existing.get('relationships', {})
        new_relations = new.get('relationships', {})
        merged_relations = {**existing_relations, **new_relations}
        
        # Add development arc entry
        if 'development_arc' not in existing:
            existing['development_arc'] = []
        
        existing['development_arc'].append({
            'chapter': chapter_idx,
            'development': new.get('psychological_profile', ''),
            'role': new.get('role_in_chapter', '')
        })
        
        # Update merged data
        existing.update({
            'personality_traits': merged_traits,
            'relationships': merged_relations,
            'importance_score': max(existing.get('importance_score', 0), new.get('importance_score', 0))
        })
        
        return existing
    
    async def _build_unified_timeline(self, timeline_events: List[Dict]):
        """Build unified chronological timeline"""
        
        # Sort events by chronological order if available, otherwise by chapter order
        sorted_events = sorted(timeline_events, key=lambda x: (
            x.get('chronological_order', x.get('chapter_index', 0)),
            x.get('chapter_index', 0)
        ))
        
        # Store in timeline_events table
        with self.engine.connect() as conn:
            for i, event in enumerate(sorted_events):
                event_id = str(uuid.uuid4())
                
                conn.execute(self.timeline_events.insert().values(
                    event_id=event_id,
                    chapter_index=event.get('chapter_index', 0),
                    event_type=event.get('event_type', 'unknown'),
                    event_description=event.get('event', ''),
                    characters_involved=json.dumps(event.get('characters', []), ensure_ascii=False),
                    chronological_order=i + 1,
                    importance_score=event.get('significance', 0.5)
                ))
            
            conn.commit()
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return self.processing_state
    
    async def get_stored_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of stored analysis results"""
        
        with self.engine.connect() as conn:
            # Count chapters
            chapter_count = conn.execute(
                text("SELECT COUNT(*) FROM chapter_summaries")
            ).scalar()
            
            # Count characters  
            char_count = conn.execute(
                text("SELECT COUNT(DISTINCT name) FROM character_profiles")
            ).scalar()
            
            # Count storylines
            thread_count = conn.execute(
                text("SELECT COUNT(*) FROM storyline_threads")
            ).scalar()
            
            # Count timeline events
            timeline_count = conn.execute(
                text("SELECT COUNT(*) FROM timeline_events")
            ).scalar()
            
            # Get recent chapters
            recent_chapters = conn.execute(
                text("SELECT chapter_title, chapter_index FROM chapter_summaries ORDER BY chapter_index DESC LIMIT 5")
            ).fetchall()
            
            # Convert SQLAlchemy rows to dictionaries properly
            recent_chapters_dict = []
            for row in recent_chapters:
                recent_chapters_dict.append({
                    'chapter_title': row[0],
                    'chapter_index': row[1]
                })
            
            return {
                'chapters_analyzed': chapter_count,
                'characters_discovered': char_count,
                'storylines_tracked': thread_count,
                'timeline_events': timeline_count,
                'recent_chapters': recent_chapters_dict,
                'processing_state': self.processing_state
            }

# Factory function
def create_agentic_processor(
    db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim",
    qdrant_url: str = "http://localhost:32768"
) -> AgenticNovelProcessor:
    """Create agentic novel processor"""
    return AgenticNovelProcessor(db_url, qdrant_url)

# Main processing script
if __name__ == "__main__":
    async def run_processing():
        processor = create_agentic_processor()
        
        print("Starting comprehensive novel analysis...")
        results = await processor.process_novel_systematically(max_chapters=3)  # Process first 3 big chapters
        
        print(f"Processing Results:")
        print(f"- Chapters processed: {results.get('chapters_processed', 0)}")
        print(f"- Characters discovered: {len(results.get('characters_discovered', {}))}")
        print(f"- Storylines tracked: {len(results.get('storylines_tracked', {}))}")
        print(f"- Timeline events: {len(results.get('timeline_constructed', []))}")
        
        if results.get('processing_errors'):
            print(f"- Errors encountered: {len(results['processing_errors'])}")
        
        # Get final summary
        summary = await processor.get_stored_analysis_summary()
        print(f"\nStored Analysis Summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    asyncio.run(run_processing())