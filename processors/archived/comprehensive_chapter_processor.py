#!/usr/bin/env python3
"""
Comprehensive Chapter Processor
Integrates Enhanced Hybrid Agent with Dynamic Character Tracking
Processes all 3 big chapters with storylines, timeline, character evolution, and recursive recap
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config
from hybrid_agent import HybridAgent, HybridQueryResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AmbiguousCharacter:
    """Temporary character reference before identity revealed"""
    identifier: str  
    descriptions: List[str]
    actions: List[str] 
    locations: List[str]
    relationships: List[str]
    big_chapter_appearances: List[int]
    dialogue_samples: List[str]
    confidence_score: float
    first_appearance: int

@dataclass
class StorylineEvent:
    """Individual storyline event"""
    big_chapter: int
    event_id: str
    description: str
    characters_involved: List[str]
    location: str
    timestamp: str
    importance_score: float
    event_type: str  # 'action', 'dialogue', 'description', 'conflict'

@dataclass
class TimelineNode:
    """Timeline node with causality"""
    big_chapter: int
    sequence_id: int
    event: str
    cause: Optional[str]
    effect: Optional[str]
    characters: List[str]
    significance: float

@dataclass
class BigChapterSummary:
    """Complete summary of a big chapter"""
    chapter_num: int
    title: str
    main_events: List[StorylineEvent]
    character_introductions: List[str]
    character_developments: List[str]
    timeline_nodes: List[TimelineNode]
    themes: List[str]
    conflicts: List[str]
    word_count: int
    key_quotes: List[str]

class ComprehensiveChapterProcessor:
    """Integrated system for comprehensive chapter analysis"""
    
    def __init__(self, qdrant_url: str = "http://localhost:32768"):
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Initialize hybrid agent
        self.hybrid_agent = HybridAgent(
            qdrant_url=qdrant_url,
            deepseek_config=self.deepseek_config
        )
        
        # Character tracking
        self.ambiguous_characters: Dict[str, AmbiguousCharacter] = {}
        self.named_characters: Dict[str, Dict[str, Any]] = {}
        
        # Story analysis
        self.big_chapter_summaries: Dict[int, BigChapterSummary] = {}
        self.global_timeline: List[TimelineNode] = []
        self.character_evolution_log: List[Dict[str, Any]] = []
        
        # Processing stats
        self.processing_stats = {
            'total_tokens_used': 0,
            'deepseek_calls': 0,
            'character_matches_found': 0,
            'storyline_events_extracted': 0
        }
        
        logger.info("Comprehensive Chapter Processor initialized")
    
    async def get_big_chapters(self, n: int = 3) -> Dict[int, List[Dict[str, Any]]]:
        """Get first N big chapters with all content"""
        logger.info(f"Fetching first {n} big chapters from Qdrant...")
        
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            points = client.scroll(
                collection_name="test_novel2",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            big_chapters = defaultdict(list)
            
            for point in points[0]:
                payload = point.payload
                if 'chunk' in payload:
                    content = payload['chunk']
                    big_chapter_num = self._extract_big_chapter_number(content)
                    
                    if big_chapter_num <= n and big_chapter_num > 0:
                        big_chapters[big_chapter_num].append({
                            'point_id': point.id,
                            'content': content,
                            'metadata': payload,
                            'chapter_num': big_chapter_num
                        })
            
            final_chapters = {}
            for chap_num in sorted(big_chapters.keys())[:n]:
                final_chapters[chap_num] = sorted(big_chapters[chap_num], key=lambda x: x['point_id'])
            
            logger.info(f"Retrieved {len(final_chapters)} big chapters:")
            for chap_num, points in final_chapters.items():
                total_chars = sum(len(p['content']) for p in points)
                logger.info(f"  Big Chapter {chap_num}: {len(points)} sections, {total_chars} total characters")
            
            return final_chapters
            
        except Exception as e:
            logger.error(f"Failed to get big chapters: {e}")
            return {}
    
    def _extract_big_chapter_number(self, content: str) -> int:
        """Extract big chapter number from content"""
        patterns = [
            r'第([一二三四五六七八九十]+)章',
            r'第(\d+)章',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    try:
                        chinese_numerals = {
                            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
                        }
                        
                        if match in chinese_numerals:
                            return chinese_numerals[match]
                        elif match.isdigit():
                            return int(match)
                    except:
                        continue
        
        return 0
    
    async def process_big_chapter_comprehensive(self, big_chapter_num: int, content_sections: List[Dict[str, Any]]) -> BigChapterSummary:
        """Comprehensive processing of a single big chapter"""
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPREHENSIVE PROCESSING: BIG CHAPTER {big_chapter_num}")
        logger.info(f"{'='*60}")
        
        # Combine all content from the big chapter
        combined_content = ""
        for section in content_sections:
            combined_content += section['content'] + "\n\n"
        
        word_count = len(combined_content)
        logger.info(f"Total content length: {word_count} characters")
        
        # 1. CHARACTER EXTRACTION AND TRACKING
        logger.info("\n1. EXTRACTING CHARACTERS...")
        character_refs = await self._extract_and_track_characters(big_chapter_num, content_sections)
        
        # 2. STORYLINE ANALYSIS USING HYBRID AGENT
        logger.info("\n2. ANALYZING STORYLINES...")
        storyline_events = await self._analyze_storylines(big_chapter_num, combined_content)
        
        # 3. TIMELINE CONSTRUCTION
        logger.info("\n3. CONSTRUCTING TIMELINE...")
        timeline_nodes = await self._construct_timeline(big_chapter_num, storyline_events)
        
        # 4. CHARACTER DEVELOPMENT ANALYSIS
        logger.info("\n4. ANALYZING CHARACTER DEVELOPMENT...")
        char_introductions, char_developments = await self._analyze_character_development(big_chapter_num, combined_content)
        
        # 5. THEMATIC ANALYSIS
        logger.info("\n5. EXTRACTING THEMES AND CONFLICTS...")
        themes, conflicts = await self._extract_themes_and_conflicts(big_chapter_num, combined_content)
        
        # 6. KEY QUOTES EXTRACTION
        logger.info("\n6. EXTRACTING KEY QUOTES...")
        key_quotes = await self._extract_key_quotes(big_chapter_num, combined_content)
        
        # Create comprehensive summary
        chapter_summary = BigChapterSummary(
            chapter_num=big_chapter_num,
            title=f"Big Chapter {big_chapter_num}",
            main_events=storyline_events,
            character_introductions=char_introductions,
            character_developments=char_developments,
            timeline_nodes=timeline_nodes,
            themes=themes,
            conflicts=conflicts,
            word_count=word_count,
            key_quotes=key_quotes
        )
        
        self.big_chapter_summaries[big_chapter_num] = chapter_summary
        return chapter_summary
    
    async def _extract_and_track_characters(self, big_chapter_num: int, content_sections: List[Dict[str, Any]]) -> List[str]:
        """Extract and track character evolution"""
        combined_content = ""
        for section in content_sections:
            combined_content += section['content'] + " "
        
        # Character patterns for garbled text
        character_patterns = [
            r'([^\\s]{2,6}(?=说|道|想|看|听|走|去|来|的|是|在|有|要|会|能|可|把|被|让|使|给|对|向|从|到|为|因|由))',
            r'([^\\s]{2,5}(?=��|��|��|��|��|��|��|��))',
            r'([^\\s]{2,4})',
        ]
        
        char_frequencies = defaultdict(int)
        for pattern in character_patterns:
            matches = re.findall(pattern, combined_content)
            for match in matches:
                if 2 <= len(match) <= 6:
                    char_frequencies[match] += 1
        
        # Filter for frequently mentioned characters
        character_refs = []
        for char_seq, frequency in char_frequencies.items():
            if frequency >= 3:
                character_refs.append(char_seq)
                
                # Track in our character system
                if char_seq not in self.ambiguous_characters and char_seq not in self.named_characters:
                    self.ambiguous_characters[char_seq] = AmbiguousCharacter(
                        identifier=char_seq,
                        descriptions=[],
                        actions=[],
                        locations=[],
                        relationships=[],
                        big_chapter_appearances=[big_chapter_num],
                        dialogue_samples=[],
                        confidence_score=min(0.3 + (frequency * 0.1), 0.9),
                        first_appearance=big_chapter_num
                    )
                    logger.info(f"   NEW CHARACTER: '{char_seq}' (frequency: {frequency})")
                else:
                    # Update existing character
                    if char_seq in self.ambiguous_characters:
                        char = self.ambiguous_characters[char_seq]
                        if big_chapter_num not in char.big_chapter_appearances:
                            char.big_chapter_appearances.append(big_chapter_num)
                            char.confidence_score = min(char.confidence_score + 0.2, 0.95)
                            logger.info(f"   UPDATED CHARACTER: '{char_seq}' (chapters: {char.big_chapter_appearances})")
        
        return character_refs
    
    async def _analyze_storylines(self, big_chapter_num: int, content: str) -> List[StorylineEvent]:
        """Use hybrid agent to analyze storylines"""
        try:
            # Initialize hybrid agent
            await self.hybrid_agent.initialize()
            self.processing_stats['deepseek_calls'] += 1
            
            # Query for storyline analysis
            query = f"""分析第{big_chapter_num}章的主要情节事件。请提取：
1. 主要行动和事件
2. 角色互动
3. 冲突和转折点
4. 重要对话
5. 场景变化

请按时间顺序列出所有重要事件，每个事件包括：
- 事件描述
- 涉及角色
- 发生地点
- 重要程度(1-10分)
- 事件类型(行动/对话/描述/冲突)"""
            
            result = await self.hybrid_agent.process_query(
                query=query,
                use_sql=False,
                use_rag=True,
                max_tokens=2000
            )
            
            if result.success:
                response_text = result.response
                events = self._parse_storyline_events(big_chapter_num, response_text)
                self.processing_stats['storyline_events_extracted'] += len(events)
                self.processing_stats['total_tokens_used'] += result.token_usage.get('total', 0)
                
                logger.info(f"   Extracted {len(events)} storyline events")
                return events
            else:
                logger.error(f"Storyline analysis failed: {result.error}")
                return []
                
        except Exception as e:
            logger.error(f"Error in storyline analysis: {e}")
            return []
    
    def _parse_storyline_events(self, big_chapter_num: int, response_text: str) -> List[StorylineEvent]:
        """Parse storyline events from AI response"""
        events = []
        event_id = 1
        
        # Split response into potential events
        lines = response_text.split('\n')
        current_event = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for event markers
            if '事件' in line or '情节' in line or re.match(r'\d+\.', line):
                if current_event:
                    # Save previous event
                    event = StorylineEvent(
                        big_chapter=big_chapter_num,
                        event_id=f"ch{big_chapter_num}_evt{event_id}",
                        description=current_event.get('description', line),
                        characters_involved=current_event.get('characters', []),
                        location=current_event.get('location', '未知'),
                        timestamp=f"Chapter {big_chapter_num}, Event {event_id}",
                        importance_score=current_event.get('importance', 5.0),
                        event_type=current_event.get('type', 'action')
                    )
                    events.append(event)
                    event_id += 1
                
                # Start new event
                current_event = {'description': line}
            
            # Extract additional info
            elif '角色' in line or '人物' in line:
                chars = re.findall(r'[^，、：:。]+', line)
                current_event['characters'] = [c.strip() for c in chars if len(c.strip()) > 1]
            elif '地点' in line or '场所' in line:
                current_event['location'] = line.split('：')[-1].strip() if '：' in line else line
            elif '重要程度' in line or '分数' in line:
                scores = re.findall(r'\d+', line)
                if scores:
                    current_event['importance'] = float(scores[0])
        
        # Don't forget the last event
        if current_event:
            event = StorylineEvent(
                big_chapter=big_chapter_num,
                event_id=f"ch{big_chapter_num}_evt{event_id}",
                description=current_event.get('description', ''),
                characters_involved=current_event.get('characters', []),
                location=current_event.get('location', '未知'),
                timestamp=f"Chapter {big_chapter_num}, Event {event_id}",
                importance_score=current_event.get('importance', 5.0),
                event_type=current_event.get('type', 'action')
            )
            events.append(event)
        
        return events
    
    async def _construct_timeline(self, big_chapter_num: int, events: List[StorylineEvent]) -> List[TimelineNode]:
        """Construct causal timeline from events"""
        timeline_nodes = []
        
        for i, event in enumerate(events):
            # Determine cause and effect relationships
            cause = events[i-1].description if i > 0 else None
            effect = events[i+1].description if i < len(events)-1 else None
            
            node = TimelineNode(
                big_chapter=big_chapter_num,
                sequence_id=i+1,
                event=event.description,
                cause=cause,
                effect=effect,
                characters=event.characters_involved,
                significance=event.importance_score
            )
            timeline_nodes.append(node)
        
        self.global_timeline.extend(timeline_nodes)
        return timeline_nodes
    
    async def _analyze_character_development(self, big_chapter_num: int, content: str) -> tuple:
        """Analyze character introductions and developments"""
        try:
            await self.hybrid_agent.initialize()
            self.processing_stats['deepseek_calls'] += 1
            
            query = f"""分析第{big_chapter_num}章中的角色发展：
1. 新引入的角色及其特征
2. 现有角色的发展变化
3. 角色关系的变化
4. 角色动机和目标

请分别列出角色介绍和角色发展。"""
            
            result = await self.hybrid_agent.process_query(
                query=query,
                use_sql=False,
                use_rag=True,
                max_tokens=1500
            )
            
            if result.success:
                response = result.response
                self.processing_stats['total_tokens_used'] += result.token_usage.get('total', 0)
                
                # Parse introductions and developments
                introductions = []
                developments = []
                
                lines = response.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if '新引入' in line or '介绍' in line:
                        current_section = 'intro'
                    elif '发展' in line or '变化' in line:
                        current_section = 'dev'
                    elif line and current_section:
                        if current_section == 'intro':
                            introductions.append(line)
                        elif current_section == 'dev':
                            developments.append(line)
                
                logger.info(f"   Character introductions: {len(introductions)}")
                logger.info(f"   Character developments: {len(developments)}")
                
                return introductions, developments
            else:
                return [], []
                
        except Exception as e:
            logger.error(f"Error in character development analysis: {e}")
            return [], []
    
    async def _extract_themes_and_conflicts(self, big_chapter_num: int, content: str) -> tuple:
        """Extract themes and conflicts"""
        try:
            await self.hybrid_agent.initialize()
            self.processing_stats['deepseek_calls'] += 1
            
            query = f"""分析第{big_chapter_num}章的主题和冲突：
1. 主要主题（友情、成长、正义等）
2. 内在冲突（角色内心斗争）
3. 外在冲突（角色间、环境冲突）
4. 道德或哲学冲突

请分别列出主题和冲突。"""
            
            result = await self.hybrid_agent.process_query(
                query=query,
                use_sql=False,
                use_rag=True,
                max_tokens=1000
            )
            
            if result.success:
                response = result.response
                self.processing_stats['total_tokens_used'] += result.token_usage.get('total', 0)
                
                themes = []
                conflicts = []
                
                lines = response.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if '主题' in line:
                        current_section = 'theme'
                    elif '冲突' in line:
                        current_section = 'conflict'
                    elif line and current_section:
                        if current_section == 'theme':
                            themes.append(line)
                        elif current_section == 'conflict':
                            conflicts.append(line)
                
                return themes, conflicts
            else:
                return [], []
                
        except Exception as e:
            logger.error(f"Error in themes/conflicts analysis: {e}")
            return [], []
    
    async def _extract_key_quotes(self, big_chapter_num: int, content: str) -> List[str]:
        """Extract key quotes and dialogue"""
        try:
            await self.hybrid_agent.initialize()
            self.processing_stats['deepseek_calls'] += 1
            
            query = f"""从第{big_chapter_num}章中提取最重要的引语和对话：
1. 揭示角色性格的对话
2. 推动情节的关键语句
3. 体现主题的重要话语
4. 令人印象深刻的描述

请提取5-10个最重要的引语。"""
            
            result = await self.hybrid_agent.process_query(
                query=query,
                use_sql=False,
                use_rag=True,
                max_tokens=800
            )
            
            if result.success:
                response = result.response
                self.processing_stats['total_tokens_used'] += result.token_usage.get('total', 0)
                
                quotes = []
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and ('"' in line or '"' in line or '：' in line):
                        quotes.append(line)
                
                return quotes[:10]  # Limit to 10 quotes
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in key quotes extraction: {e}")
            return []
    
    async def generate_recursive_recap(self) -> str:
        """Generate recursive recap of all processed chapters"""
        logger.info("\n" + "="*60)
        logger.info("GENERATING RECURSIVE RECAP")
        logger.info("="*60)
        
        try:
            await self.hybrid_agent.initialize()
            self.processing_stats['deepseek_calls'] += 1
            
            # Collect all chapter summaries
            recap_data = []
            for chapter_num in sorted(self.big_chapter_summaries.keys()):
                summary = self.big_chapter_summaries[chapter_num]
                recap_data.append({
                    'chapter': chapter_num,
                    'events': len(summary.main_events),
                    'characters': len(summary.character_introductions) + len(summary.character_developments),
                    'themes': summary.themes,
                    'key_events': [event.description for event in summary.main_events[:3]]
                })
            
            query = f"""基于以下章节分析，生成递归式总结：
章节数据：{json.dumps(recap_data, ensure_ascii=False, indent=2)}

请生成：
1. 整体故事arc的发展
2. 角色成长轨迹
3. 主题演进
4. 关键转折点
5. 前后章节的因果关系
6. 伏笔和呼应

生成一个连贯的、递归式的故事总结。"""
            
            result = await self.hybrid_agent.process_query(
                query=query,
                use_sql=False,
                use_rag=False,  # Use accumulated knowledge
                max_tokens=2500
            )
            
            if result.success:
                self.processing_stats['total_tokens_used'] += result.token_usage.get('total', 0)
                return result.response
            else:
                return "递归总结生成失败"
                
        except Exception as e:
            logger.error(f"Error generating recursive recap: {e}")
            return f"递归总结生成错误: {e}"
    
    def print_comprehensive_results(self):
        """Print complete analysis results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CHAPTER ANALYSIS RESULTS")
        print("="*80)
        
        # Processing Statistics
        print(f"\nPROCESSING STATISTICS:")
        print(f"   Total DeepSeek API calls: {self.processing_stats['deepseek_calls']}")
        print(f"   Total tokens used: {self.processing_stats['total_tokens_used']}")
        print(f"   Storyline events extracted: {self.processing_stats['storyline_events_extracted']}")
        print(f"   Character matches found: {self.processing_stats['character_matches_found']}")
        
        # Chapter Summaries
        for chapter_num in sorted(self.big_chapter_summaries.keys()):
            summary = self.big_chapter_summaries[chapter_num]
            print(f"\n{'-'*60}")
            print(f"BIG CHAPTER {chapter_num} ANALYSIS")
            print(f"{'-'*60}")
            
            print(f"Word Count: {summary.word_count:,} characters")
            print(f"Main Events: {len(summary.main_events)}")
            print(f"Timeline Nodes: {len(summary.timeline_nodes)}")
            
            print(f"\nMAJOR EVENTS:")
            for event in summary.main_events[:5]:  # Top 5 events
                print(f"   • {event.description}")
                print(f"     Characters: {', '.join(event.characters_involved)}")
                print(f"     Importance: {event.importance_score}/10")
            
            print(f"\nCHARACTER INTRODUCTIONS:")
            for intro in summary.character_introductions[:3]:
                print(f"   • {intro}")
            
            print(f"\nCHARACTER DEVELOPMENTS:")
            for dev in summary.character_developments[:3]:
                print(f"   • {dev}")
            
            print(f"\nTHEMES:")
            for theme in summary.themes[:5]:
                print(f"   • {theme}")
            
            print(f"\nCONFLICTS:")
            for conflict in summary.conflicts[:3]:
                print(f"   • {conflict}")
            
            print(f"\nKEY QUOTES:")
            for quote in summary.key_quotes[:3]:
                print(f"   • {quote}")
        
        # Character Evolution Summary
        print(f"\n{'-'*60}")
        print("CHARACTER EVOLUTION ACROSS CHAPTERS")
        print(f"{'-'*60}")
        
        # Sort characters by importance
        all_characters = []
        for char_id, char_data in self.ambiguous_characters.items():
            all_characters.append({
                'name': char_id,
                'chapters': char_data.big_chapter_appearances,
                'confidence': char_data.confidence_score,
                'type': 'ambiguous'
            })
        
        for char_id, char_data in self.named_characters.items():
            all_characters.append({
                'name': char_id,
                'chapters': char_data.get('chapter_appearances', []),
                'confidence': char_data.get('confidence', 0.8),
                'type': 'named'
            })
        
        all_characters.sort(key=lambda x: (len(x['chapters']), x['confidence']), reverse=True)
        
        for char in all_characters[:15]:  # Top 15 characters
            print(f"\n'{char['name']}' ({char['type']}):")
            print(f"   Appears in chapters: {char['chapters']}")
            print(f"   Confidence: {char['confidence']:.2f}")
            if len(char['chapters']) >= 2:
                print(f"   >>> RECURRING CHARACTER <<<")
        
        # Global Timeline
        print(f"\n{'-'*60}")
        print("GLOBAL TIMELINE")
        print(f"{'-'*60}")
        
        for i, node in enumerate(self.global_timeline):
            print(f"\n{i+1}. Chapter {node.big_chapter}, Event {node.sequence_id}")
            print(f"   Event: {node.event}")
            print(f"   Characters: {', '.join(node.characters)}")
            print(f"   Significance: {node.significance}/10")
            if node.cause:
                print(f"   Caused by: {node.cause[:100]}...")
            if node.effect:
                print(f"   Leads to: {node.effect[:100]}...")

async def main():
    """Run comprehensive chapter processing"""
    print("COMPREHENSIVE CHAPTER PROCESSING")
    print("="*80)
    print("Processing all 3 big chapters with:")
    print("  • Enhanced Hybrid Agent storyline analysis")
    print("  • Dynamic character tracking and evolution")
    print("  • Timeline construction with causality")
    print("  • Thematic analysis")
    print("  • Recursive recap generation")
    print("="*80)
    
    processor = ComprehensiveChapterProcessor()
    
    # Get big chapters
    big_chapters = await processor.get_big_chapters(3)
    
    if not big_chapters:
        print("No big chapters found in Qdrant")
        return
    
    # Process each big chapter comprehensively
    for chapter_num in sorted(big_chapters.keys()):
        content_sections = big_chapters[chapter_num]
        await processor.process_big_chapter_comprehensive(chapter_num, content_sections)
    
    # Generate recursive recap
    print("\n" + "="*60)
    print("GENERATING RECURSIVE RECAP...")
    print("="*60)
    
    recap = await processor.generate_recursive_recap()
    
    # Print all results
    processor.print_comprehensive_results()
    
    # Print recursive recap
    print(f"\n{'='*80}")
    print("RECURSIVE RECAP")
    print("="*80)
    print(recap)
    
    # Close connections
    if processor.deepseek_client.session:
        await processor.deepseek_client.close()
    if processor.hybrid_agent.deepseek_client.session:
        await processor.hybrid_agent.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())