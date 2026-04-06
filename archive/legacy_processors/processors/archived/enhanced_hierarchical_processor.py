#!/usr/bin/env python3
"""
Enhanced Hierarchical Novel Processor
Integrates hierarchical chapter parsing with existing agentic analysis

This processor properly handles:
1. 卷 (volumes/big chapters) vs 章 (small chapters) distinction
2. Hierarchical analysis from volume level down to chapter level
3. Cross-volume character tracking and storyline continuity
4. Volume-specific thematic analysis
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from hierarchical_chapter_parser import HierarchicalChapterParser, ChapterNode, VolumeStructure
from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

logger = logging.getLogger(__name__)

@dataclass
class VolumeAnalysis:
    """Complete analysis of a volume"""
    volume_id: int
    volume_title: str
    chapter_analyses: List[Dict[str, Any]]
    volume_themes: List[str]
    character_introductions: List[str]
    character_developments: List[str]
    storyline_progression: List[str]
    volume_summary: str
    cross_chapter_connections: List[Dict[str, Any]]
    
@dataclass
class HierarchicalNovelAnalysis:
    """Complete hierarchical analysis of the novel"""
    novel_title: str
    total_volumes: int
    total_chapters: int
    volume_analyses: Dict[int, VolumeAnalysis]
    cross_volume_storylines: List[Dict[str, Any]]
    character_evolution_timeline: List[Dict[str, Any]]
    thematic_progression: List[str]
    overall_narrative_arc: str

class EnhancedHierarchicalProcessor:
    """Enhanced processor with proper 卷/章 hierarchy support"""
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:32768",
                 db_url: str = "postgresql://admin:admin@localhost:5432/novel_sim"):
        
        self.hierarchical_parser = HierarchicalChapterParser()
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Analysis storage
        self.novel_structure: Dict[int, VolumeStructure] = {}
        self.volume_analyses: Dict[int, VolumeAnalysis] = {}
        self.character_timeline: List[Dict[str, Any]] = []
        self.cross_volume_storylines: List[Dict[str, Any]] = []
        
        # Enhanced prompts for hierarchical analysis
        self.prompts = self._setup_hierarchical_prompts()
        
        logger.info("Enhanced Hierarchical Processor initialized")
    
    def _setup_hierarchical_prompts(self) -> Dict[str, str]:
        """Setup prompts specifically for hierarchical analysis"""
        return {
            'volume_analysis': """
你是专业的中文小说分析专家。请对以下卷(volume)进行全面分析。

卷信息：
- 卷号：{volume_id}
- 卷标题：{volume_title}
- 包含章节：{chapter_count}个章节
- 总字数：{word_count}字

请按以下JSON格式分析这一卷的内容：
{{
  "volume_themes": ["主题1", "主题2", "主题3"],
  "character_introductions": ["新角色1的介绍", "新角色2的介绍"],
  "character_developments": ["角色A的发展", "角色B的变化"],
  "storyline_progression": ["情节线1的发展", "情节线2的推进"],
  "volume_summary": "这一卷的整体概述(200-300字)",
  "cross_chapter_connections": [
    {{
      "from_chapter": 1,
      "to_chapter": 3,
      "connection_type": "因果关系",
      "description": "连接描述"
    }}
  ]
}}

卷内容：
{volume_content}
""",

            'chapter_in_volume': """
你是专业的中文小说分析专家。请分析以下章节，注意它在整卷中的位置和作用。

章节信息：
- 所属卷：第{volume_id}卷 {volume_title}
- 章节：第{chapter_id}章 {chapter_title}
- 在卷中位置：第{chapter_position}/{total_chapters}章

请按以下JSON格式分析：
{{
  "chapter_summary": {{
    "main_events": ["主要事件1", "主要事件2"],
    "characters_involved": ["角色1", "角色2"],
    "emotional_arc": "情感发展轨迹",
    "plot_significance": 0.8,
    "volume_role": "在整卷中的作用"
  }},
  "character_analysis": [
    {{
      "name": "角色名",
      "role_in_chapter": "在本章的作用",
      "development": "发展变化",
      "relationships": {{"其他角色": "关系变化"}}
    }}
  ],
  "storyline_threads": [
    {{
      "thread_title": "故事线名称",
      "development": "在本章的发展",
      "connects_to": ["前面章节", "后续章节"],
      "importance": 0.7
    }}
  ]
}}

章节内容：
{content}
""",

            'cross_volume_analysis': """
基于已分析的多个卷，请分析跨卷的故事发展：

已分析卷：{volume_summaries}

请按以下JSON格式分析：
{{
  "cross_volume_storylines": [
    {{
      "storyline_title": "跨卷故事线名称",
      "volume_span": [1, 2, 3],
      "development_stages": ["第一卷的发展", "第二卷的推进", "第三卷的高潮"],
      "key_characters": ["主要角色"],
      "resolution_status": "ongoing/resolved"
    }}
  ],
  "character_evolution": [
    {{
      "character_name": "角色名",
      "evolution_timeline": [
        {{"volume": 1, "state": "初始状态", "key_events": ["重要事件"]}},
        {{"volume": 2, "state": "发展状态", "key_events": ["重要事件"]}}
      ]
    }}
  ],
  "thematic_progression": ["主题在各卷中的演进"],
  "narrative_arc": "整体叙事弧线的发展"
}}
""",

            'volume_comparison': """
请比较分析以下两个卷的异同：

第{volume1_id}卷：{volume1_summary}
第{volume2_id}卷：{volume2_summary}

请分析：
1. 主题的延续与变化
2. 角色发展的对比
3. 故事节奏的差异
4. 叙事风格的变化
5. 两卷之间的连接点

请用中文回答，重点关注连续性和发展变化。
"""
        }
    
    async def process_novel_hierarchically(self, max_volumes: int = 3) -> HierarchicalNovelAnalysis:
        """Process novel with full hierarchical analysis"""
        logger.info(f"Starting hierarchical processing of up to {max_volumes} volumes...")
        
        try:
            # Step 1: Parse hierarchical structure
            structure_result = await self._parse_novel_structure()
            if not structure_result['success']:
                raise Exception(f"Structure parsing failed: {structure_result['error']}")
            
            self.novel_structure = structure_result['structure']
            
            # Step 2: Analyze each volume comprehensively
            for volume_id in sorted(self.novel_structure.keys())[:max_volumes]:
                if volume_id == 0:  # Skip unstructured chapters for now
                    continue
                    
                logger.info(f"\nProcessing Volume {volume_id}...")
                volume_analysis = await self._analyze_volume_comprehensive(volume_id)
                self.volume_analyses[volume_id] = volume_analysis
            
            # Step 3: Cross-volume analysis
            logger.info(f"\nPerforming cross-volume analysis...")
            await self._analyze_cross_volume_connections()
            
            # Step 4: Generate complete hierarchical analysis
            hierarchical_analysis = HierarchicalNovelAnalysis(
                novel_title="Unknown Novel",  # Can be detected from content
                total_volumes=len(self.novel_structure),
                total_chapters=sum(len(vol.chapters) for vol in self.novel_structure.values()),
                volume_analyses=self.volume_analyses,
                cross_volume_storylines=self.cross_volume_storylines,
                character_evolution_timeline=self.character_timeline,
                thematic_progression=[],
                overall_narrative_arc=""
            )
            
            return hierarchical_analysis
            
        except Exception as e:
            logger.error(f"Hierarchical processing error: {e}")
            raise
    
    async def _parse_novel_structure(self) -> Dict[str, Any]:
        """Parse the novel structure from Qdrant"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            # Get all content
            points = client.scroll(
                collection_name="test_novel2",
                limit=200,
                with_payload=True,
                with_vectors=False
            )
            
            # Combine content
            combined_content = ""
            for point in points[0]:
                if 'chunk' in point.payload:
                    combined_content += point.payload['chunk'] + "\n\n"
            
            # Parse hierarchical structure
            chapters = self.hierarchical_parser.parse_content_hierarchy(combined_content)
            
            return {
                'success': True,
                'structure': self.hierarchical_parser.parsed_structure,
                'chapters': chapters
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'structure': {},
                'chapters': []
            }
    
    async def _analyze_volume_comprehensive(self, volume_id: int) -> VolumeAnalysis:
        """Comprehensive analysis of a single volume"""
        volume = self.novel_structure[volume_id]
        logger.info(f"Analyzing Volume {volume_id}: {volume.volume_title}")
        
        # Combine all chapters in the volume
        volume_content = ""
        chapter_analyses = []
        
        for i, chapter in enumerate(volume.chapters):
            # Analyze individual chapter
            chapter_analysis = await self._analyze_chapter_in_volume(
                volume_id, chapter, i + 1, len(volume.chapters)
            )
            chapter_analyses.append(chapter_analysis)
            volume_content += chapter.content + "\n\n"
        
        # Analyze entire volume
        volume_analysis_result = await self._analyze_volume_as_whole(
            volume_id, volume.volume_title, volume_content, len(volume.chapters), volume.total_words
        )
        
        # Create VolumeAnalysis object
        volume_analysis = VolumeAnalysis(
            volume_id=volume_id,
            volume_title=volume.volume_title,
            chapter_analyses=chapter_analyses,
            volume_themes=volume_analysis_result.get('volume_themes', []),
            character_introductions=volume_analysis_result.get('character_introductions', []),
            character_developments=volume_analysis_result.get('character_developments', []),
            storyline_progression=volume_analysis_result.get('storyline_progression', []),
            volume_summary=volume_analysis_result.get('volume_summary', ''),
            cross_chapter_connections=volume_analysis_result.get('cross_chapter_connections', [])
        )
        
        return volume_analysis
    
    async def _analyze_chapter_in_volume(self, volume_id: int, chapter: ChapterNode, 
                                       chapter_position: int, total_chapters: int) -> Dict[str, Any]:
        """Analyze a single chapter within its volume context"""
        try:
            prompt = self.prompts['chapter_in_volume'].format(
                volume_id=volume_id,
                volume_title=chapter.volume_title or f"第{volume_id}卷",
                chapter_id=chapter.chapter_id,
                chapter_title=chapter.chapter_title,
                chapter_position=chapter_position,
                total_chapters=total_chapters,
                content=chapter.content[:2000]  # Limit content for analysis
            )
            
            response = await self._query_deepseek(prompt)
            if response:
                try:
                    return json.loads(self._extract_json_from_markdown(response))
                except json.JSONDecodeError:
                    return {'error': 'Failed to parse chapter analysis JSON'}
            else:
                return {'error': 'No response from DeepSeek'}
                
        except Exception as e:
            logger.error(f"Error analyzing chapter {chapter.hierarchy_id}: {e}")
            return {'error': str(e)}
    
    async def _analyze_volume_as_whole(self, volume_id: int, volume_title: str, 
                                     volume_content: str, chapter_count: int, word_count: int) -> Dict[str, Any]:
        """Analyze entire volume as a coherent unit"""
        try:
            prompt = self.prompts['volume_analysis'].format(
                volume_id=volume_id,
                volume_title=volume_title,
                chapter_count=chapter_count,
                word_count=word_count,
                volume_content=volume_content[:3000]  # Limit for analysis
            )
            
            response = await self._query_deepseek(prompt)
            if response:
                try:
                    return json.loads(self._extract_json_from_markdown(response))
                except json.JSONDecodeError:
                    return {
                        'volume_themes': [],
                        'character_introductions': [],
                        'character_developments': [],
                        'storyline_progression': [],
                        'volume_summary': response[:500],
                        'cross_chapter_connections': []
                    }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error analyzing volume {volume_id}: {e}")
            return {}
    
    async def _analyze_cross_volume_connections(self):
        """Analyze connections and storylines across volumes"""
        if len(self.volume_analyses) < 2:
            return
        
        try:
            # Prepare volume summaries for analysis
            volume_summaries = {}
            for vol_id, analysis in self.volume_analyses.items():
                volume_summaries[vol_id] = {
                    'title': analysis.volume_title,
                    'themes': analysis.volume_themes,
                    'summary': analysis.volume_summary,
                    'characters': analysis.character_introductions + analysis.character_developments
                }
            
            prompt = self.prompts['cross_volume_analysis'].format(
                volume_summaries=json.dumps(volume_summaries, ensure_ascii=False, indent=2)
            )
            
            response = await self._query_deepseek(prompt)
            if response:
                try:
                    cross_analysis = json.loads(self._extract_json_from_markdown(response))
                    self.cross_volume_storylines = cross_analysis.get('cross_volume_storylines', [])
                    self.character_timeline = cross_analysis.get('character_evolution', [])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse cross-volume analysis JSON")
                    
        except Exception as e:
            logger.error(f"Error in cross-volume analysis: {e}")
    
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
    
    def _extract_json_from_markdown(self, response_text: str) -> str:
        """Extract JSON from markdown code blocks"""
        import re
        
        # Try to find JSON in markdown code blocks
        markdown_pattern = r'```(?:json)?\\s*\\n(.*?)\\n```'
        match = re.search(markdown_pattern, response_text, re.DOTALL | re.MULTILINE)
        
        if match:
            return match.group(1).strip()
        
        # Try to find balanced braces
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return response_text[start_idx:i+1]
        
        return response_text.strip()
    
    def print_hierarchical_results(self):
        """Print comprehensive hierarchical analysis results"""
        print("\\n" + "="*80)
        print("HIERARCHICAL NOVEL ANALYSIS RESULTS")
        print("="*80)
        
        # Overall structure
        print(f"\\nSTRUCTURE OVERVIEW:")
        print(f"  Total Volumes: {len(self.novel_structure)}")
        print(f"  Total Chapters: {sum(len(vol.chapters) for vol in self.novel_structure.values())}")
        print(f"  Total Words: {sum(vol.total_words for vol in self.novel_structure.values()):,}")
        
        # Volume-by-volume analysis
        for vol_id in sorted(self.volume_analyses.keys()):
            analysis = self.volume_analyses[vol_id]
            print(f"\\n{'='*60}")
            print(f"第{vol_id}卷: {analysis.volume_title}")
            print("="*60)
            
            print(f"\\n📖 VOLUME SUMMARY:")
            print(f"  {analysis.volume_summary}")
            
            print(f"\\n🎭 MAIN THEMES:")
            for theme in analysis.volume_themes:
                print(f"  • {theme}")
            
            print(f"\\n👥 CHARACTER DEVELOPMENTS:")
            for dev in analysis.character_developments[:3]:
                print(f"  • {dev}")
            
            print(f"\\n📚 STORYLINE PROGRESSION:")
            for story in analysis.storyline_progression[:3]:
                print(f"  • {story}")
            
            print(f"\\n🔗 CROSS-CHAPTER CONNECTIONS:")
            for conn in analysis.cross_chapter_connections[:2]:
                print(f"  • Ch.{conn.get('from_chapter')} → Ch.{conn.get('to_chapter')}: {conn.get('description', '')}")
        
        # Cross-volume analysis
        if self.cross_volume_storylines:
            print(f"\\n{'='*60}")
            print("CROSS-VOLUME STORYLINES")
            print("="*60)
            
            for storyline in self.cross_volume_storylines:
                print(f"\\n📖 {storyline.get('storyline_title', 'Unknown Storyline')}")
                print(f"   Spans volumes: {storyline.get('volume_span', [])}")
                print(f"   Status: {storyline.get('resolution_status', 'unknown')}")
                print(f"   Key characters: {', '.join(storyline.get('key_characters', []))}")
        
        # Character evolution timeline
        if self.character_timeline:
            print(f"\\n{'='*60}")
            print("CHARACTER EVOLUTION TIMELINE")
            print("="*60)
            
            for char_evo in self.character_timeline:
                print(f"\\n👤 {char_evo.get('character_name', 'Unknown Character')}")
                for stage in char_evo.get('evolution_timeline', []):
                    vol = stage.get('volume', 0)
                    state = stage.get('state', '')
                    print(f"   第{vol}卷: {state}")

# Main execution
async def main():
    """Run enhanced hierarchical processing"""
    print("ENHANCED HIERARCHICAL NOVEL PROCESSING")
    print("="*80)
    print("Processing with proper 卷/章 hierarchy:")
    print("  • Volume-level thematic analysis")
    print("  • Chapter-in-volume contextual analysis") 
    print("  • Cross-volume storyline tracking")
    print("  • Character evolution across volumes")
    print("="*80)
    
    processor = EnhancedHierarchicalProcessor()
    
    try:
        # Process novel hierarchically
        analysis = await processor.process_novel_hierarchically(max_volumes=3)
        
        # Print results
        processor.print_hierarchical_results()
        
        # Print structure analysis
        processor.hierarchical_parser.print_structure_analysis()
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if processor.deepseek_client.session:
            await processor.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())