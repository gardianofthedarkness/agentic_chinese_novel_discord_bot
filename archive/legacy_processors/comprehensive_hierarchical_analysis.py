#!/usr/bin/env python3
"""
Comprehensive Hierarchical Analysis with DeepSeek Integration
Runs complete agentic processing pipeline with progress monitoring and token tracking
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from hierarchical_chapter_parser import HierarchicalChapterParser, ChapterNode, VolumeStructure
from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Track processing statistics and token usage"""
    start_time: datetime
    chapters_processed: int = 0
    volumes_processed: int = 0
    deepseek_calls: int = 0
    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    processing_errors: List[Dict] = None
    
    def __post_init__(self):
        if self.processing_errors is None:
            self.processing_errors = []

@dataclass 
class ChapterAnalysisResult:
    """Results from analyzing a single chapter"""
    chapter_id: str
    volume_id: int
    chapter_num: int
    analysis_success: bool
    tokens_used: int
    processing_time: float
    
    # Analysis content
    character_analysis: Dict = None
    storyline_analysis: Dict = None
    thematic_analysis: Dict = None
    summary: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        if self.character_analysis is None:
            self.character_analysis = {}
        if self.storyline_analysis is None:
            self.storyline_analysis = {}
        if self.thematic_analysis is None:
            self.thematic_analysis = {}

class ComprehensiveHierarchicalAnalyzer:
    """Complete hierarchical analyzer with DeepSeek integration"""
    
    def __init__(self, qdrant_url: str = "http://localhost:32768"):
        self.parser = HierarchicalChapterParser()
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        self.stats = ProcessingStats(start_time=datetime.now())
        self.chapter_results: List[ChapterAnalysisResult] = []
        self.volume_analyses: Dict[int, Dict] = {}
        
        # Enhanced prompts for analysis
        self.prompts = self._setup_analysis_prompts()
        
        print("="*80)
        print("COMPREHENSIVE HIERARCHICAL NOVEL ANALYZER INITIALIZED")
        print("="*80)
        print(f"Start time: {self.stats.start_time}")
        print(f"DeepSeek endpoint: {self.deepseek_config.get('api_url', 'default')}")
        print("="*80)
    
    def _setup_analysis_prompts(self) -> Dict[str, str]:
        """Setup analysis prompts for different aspects"""
        return {
            'character_analysis': """
你是专业的中文小说分析专家。请分析以下章节中的角色。

章节信息：
- 卷号：{volume_id}
- 章节号：{chapter_id}  
- 标题：{chapter_title}

请按JSON格式返回角色分析：
{{
  "main_characters": [
    {{
      "name": "角色名",
      "role": "protagonist/antagonist/supporting",
      "personality_traits": ["特征1", "特征2"],
      "actions_in_chapter": ["行动1", "行动2"],
      "development": "在本章的发展变化",
      "importance_score": 0.8
    }}
  ],
  "character_relationships": [
    {{
      "character1": "角色A",
      "character2": "角色B", 
      "relationship_type": "friend/enemy/family/romance",
      "relationship_description": "关系描述"
    }}
  ],
  "new_characters_introduced": ["新角色1", "新角色2"]
}}

章节内容：
{content}
""",

            'storyline_analysis': """
请分析以下章节的故事情节发展：

章节信息：
- 卷：{volume_id} 章：{chapter_id}
- 标题：{chapter_title}

请返回JSON格式的故事线分析：
{{
  "main_plot_events": [
    {{
      "event_description": "事件描述",
      "event_type": "conflict/resolution/development/climax",
      "characters_involved": ["角色1", "角色2"],
      "importance": 0.9,
      "consequences": "事件后果"
    }}
  ],
  "plot_threads": [
    {{
      "thread_name": "故事线名称",
      "development": "在本章的发展",
      "status": "ongoing/resolved/paused"
    }}
  ],
  "foreshadowing": ["伏笔1", "伏笔2"],
  "chapter_climax": "本章高潮描述"
}}

内容：
{content}
""",

            'thematic_analysis': """
请分析以下章节的主题内容：

章节：第{volume_id}卷第{chapter_id}章 - {chapter_title}

请返回JSON格式的主题分析：
{{
  "themes": [
    {{
      "theme_name": "主题名称",
      "description": "主题描述",
      "evidence": ["支撑证据1", "支撑证据2"],
      "prominence": 0.7
    }}
  ],
  "mood_tone": "章节整体氛围",
  "symbolic_elements": ["象征元素1", "象征元素2"],
  "conflicts": [
    {{
      "conflict_type": "internal/external/social",
      "description": "冲突描述",
      "characters_affected": ["角色1", "角色2"]
    }}
  ],
  "chapter_message": "本章传达的核心信息"
}}

内容：
{content}
""",

            'volume_synthesis': """
基于以下章节分析，请对整卷进行综合分析：

卷号：第{volume_id}卷
章节数：{chapter_count}
章节分析结果：{chapter_analyses}

请返回JSON格式的卷级分析：
{{
  "volume_themes": ["卷级主题1", "卷级主题2"],
  "character_arcs": [
    {{
      "character": "角色名", 
      "arc_description": "在整卷中的发展轨迹",
      "character_growth": "角色成长"
    }}
  ],
  "plot_progression": "整卷情节发展轨迹",
  "climax_chapters": ["高潮章节"],
  "volume_resolution": "卷级解决方案或悬念",
  "connection_to_next_volume": "与下一卷的连接点"
}}
"""
        }
    
    async def run_comprehensive_analysis(self, max_volumes: int = 2) -> Dict[str, Any]:
        """Run complete hierarchical analysis with progress monitoring"""
        
        print("\n🚀 STARTING COMPREHENSIVE HIERARCHICAL ANALYSIS")
        print("="*60)
        
        try:
            # Step 1: Parse hierarchical structure
            print("📖 Step 1: Parsing hierarchical structure...")
            structure_result = await self._parse_novel_structure()
            
            if not structure_result['success']:
                raise Exception(f"Structure parsing failed: {structure_result['error']}")
            
            volumes_to_process = min(len(structure_result['parsed_structure']), max_volumes)
            print(f"✅ Structure parsed: {len(structure_result['parsed_chapters'])} chapters in {volumes_to_process} volumes")
            
            # Step 2: Process each volume
            for volume_id in sorted(structure_result['parsed_structure'].keys())[:max_volumes]:
                if volume_id == 0:  # Skip unstructured content
                    continue
                    
                print(f"\n📚 Processing Volume {volume_id}...")
                await self._process_volume_comprehensive(volume_id, structure_result['parsed_structure'][volume_id])
                self.stats.volumes_processed += 1
            
            # Step 3: Generate final report
            print(f"\n📊 Generating comprehensive analysis report...")
            final_report = self._generate_final_report()
            
            return final_report
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            self.stats.processing_errors.append({
                'type': 'fatal_error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return {'success': False, 'error': str(e)}
    
    async def _parse_novel_structure(self) -> Dict[str, Any]:
        """Parse novel structure from Qdrant"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            points = client.scroll(
                collection_name="test_novel2",
                limit=200,
                with_payload=True,
                with_vectors=False
            )
            
            combined_content = ""
            for point in points[0]:
                if 'chunk' in point.payload:
                    combined_content += point.payload['chunk'] + "\n\n"
            
            chapters = self.parser.parse_content_hierarchy(combined_content)
            
            return {
                'success': True,
                'parsed_structure': self.parser.parsed_structure,
                'parsed_chapters': chapters,
                'total_content_length': len(combined_content)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_volume_comprehensive(self, volume_id: int, volume: VolumeStructure):
        """Process a complete volume with all chapters"""
        
        volume_start_time = time.time()
        print(f"  📖 Volume {volume_id}: {volume.volume_title}")
        print(f"  📄 Processing {len(volume.chapters)} chapters...")
        
        volume_chapter_results = []
        
        # Process each chapter in the volume
        for chapter in volume.chapters:
            chapter_result = await self._analyze_chapter_comprehensive(volume_id, chapter)
            volume_chapter_results.append(chapter_result)
            self.chapter_results.append(chapter_result)
            self.stats.chapters_processed += 1
            
            # Progress update
            if chapter_result.analysis_success:
                print(f"    ✅ Ch.{chapter.chapter_id}: {chapter_result.tokens_used} tokens, {chapter_result.processing_time:.1f}s")
            else:
                print(f"    ❌ Ch.{chapter.chapter_id}: Error - {chapter_result.error_message}")
        
        # Volume-level synthesis
        print(f"  🔄 Synthesizing volume-level analysis...")
        volume_synthesis = await self._synthesize_volume_analysis(volume_id, volume, volume_chapter_results)
        self.volume_analyses[volume_id] = volume_synthesis
        
        volume_time = time.time() - volume_start_time
        successful_chapters = sum(1 for r in volume_chapter_results if r.analysis_success)
        total_volume_tokens = sum(r.tokens_used for r in volume_chapter_results)
        
        print(f"  ✅ Volume {volume_id} complete: {successful_chapters}/{len(volume.chapters)} chapters, {total_volume_tokens} tokens, {volume_time:.1f}s")
    
    async def _analyze_chapter_comprehensive(self, volume_id: int, chapter: ChapterNode) -> ChapterAnalysisResult:
        """Analyze a single chapter comprehensively"""
        
        chapter_start_time = time.time()
        chapter_result = ChapterAnalysisResult(
            chapter_id=chapter.hierarchy_id,
            volume_id=volume_id,
            chapter_num=chapter.chapter_id,
            analysis_success=False,
            tokens_used=0,
            processing_time=0.0
        )
        
        try:
            # Limit content length for analysis
            content_for_analysis = chapter.content[:2000] if len(chapter.content) > 2000 else chapter.content
            
            # Character analysis
            char_analysis = await self._query_deepseek_with_tracking(
                self.prompts['character_analysis'].format(
                    volume_id=volume_id,
                    chapter_id=chapter.chapter_id,
                    chapter_title=chapter.chapter_title,
                    content=content_for_analysis
                )
            )
            
            if char_analysis['success']:
                chapter_result.character_analysis = char_analysis['parsed_data']
                chapter_result.tokens_used += char_analysis['tokens_used']
            
            # Storyline analysis
            story_analysis = await self._query_deepseek_with_tracking(
                self.prompts['storyline_analysis'].format(
                    volume_id=volume_id,
                    chapter_id=chapter.chapter_id,
                    chapter_title=chapter.chapter_title,
                    content=content_for_analysis
                )
            )
            
            if story_analysis['success']:
                chapter_result.storyline_analysis = story_analysis['parsed_data']
                chapter_result.tokens_used += story_analysis['tokens_used']
            
            # Thematic analysis
            theme_analysis = await self._query_deepseek_with_tracking(
                self.prompts['thematic_analysis'].format(
                    volume_id=volume_id,
                    chapter_id=chapter.chapter_id,
                    chapter_title=chapter.chapter_title,
                    content=content_for_analysis
                )
            )
            
            if theme_analysis['success']:
                chapter_result.thematic_analysis = theme_analysis['parsed_data']
                chapter_result.tokens_used += theme_analysis['tokens_used']
            
            chapter_result.analysis_success = True
            chapter_result.summary = f"Analyzed {len(content_for_analysis)} chars with {chapter_result.tokens_used} tokens"
            
        except Exception as e:
            chapter_result.error_message = str(e)
            logger.error(f"Chapter analysis error for {chapter.hierarchy_id}: {e}")
        
        chapter_result.processing_time = time.time() - chapter_start_time
        return chapter_result
    
    async def _query_deepseek_with_tracking(self, prompt: str) -> Dict[str, Any]:
        """Query DeepSeek with token tracking"""
        
        try:
            self.stats.deepseek_calls += 1
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1500,
                temperature=0.2
            )
            
            if response.get("success"):
                # Estimate tokens (rough approximation)
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                self.stats.total_tokens_used += tokens_used
                self.stats.total_cost_estimate += tokens_used * 0.00002  # Rough estimate
                
                # Try to parse JSON response
                try:
                    cleaned_response = self._extract_json_from_markdown(response["response"])
                    parsed_data = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    parsed_data = {"raw_response": response["response"]}
                
                return {
                    'success': True,
                    'parsed_data': parsed_data,
                    'raw_response': response["response"],
                    'tokens_used': tokens_used
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error'),
                    'tokens_used': 0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tokens_used': 0
            }
    
    def _extract_json_from_markdown(self, response_text: str) -> str:
        """Extract JSON from markdown code blocks"""
        import re
        
        # Try markdown code blocks first
        markdown_pattern = r'```(?:json)?\\s*\\n(.*?)\\n```'
        match = re.search(markdown_pattern, response_text, re.DOTALL)
        
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
    
    async def _synthesize_volume_analysis(self, volume_id: int, volume: VolumeStructure, 
                                        chapter_results: List[ChapterAnalysisResult]) -> Dict[str, Any]:
        """Synthesize volume-level analysis from chapter results"""
        
        try:
            # Prepare chapter analysis summary
            chapter_summaries = []
            for result in chapter_results:
                if result.analysis_success:
                    chapter_summaries.append({
                        'chapter_id': result.chapter_num,
                        'character_analysis': result.character_analysis,
                        'storyline_analysis': result.storyline_analysis,
                        'thematic_analysis': result.thematic_analysis
                    })
            
            # Query for volume synthesis
            synthesis_result = await self._query_deepseek_with_tracking(
                self.prompts['volume_synthesis'].format(
                    volume_id=volume_id,
                    chapter_count=len(volume.chapters),
                    chapter_analyses=json.dumps(chapter_summaries[:3], ensure_ascii=False)  # Limit size
                )
            )
            
            if synthesis_result['success']:
                return synthesis_result['parsed_data']
            else:
                return {'error': synthesis_result['error']}
                
        except Exception as e:
            logger.error(f"Volume synthesis error for volume {volume_id}: {e}")
            return {'error': str(e)}
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final analysis report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.stats.start_time).total_seconds()
        
        successful_chapters = sum(1 for r in self.chapter_results if r.analysis_success)
        failed_chapters = len(self.chapter_results) - successful_chapters
        
        report = {
            'analysis_metadata': {
                'start_time': self.stats.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_processing_time': f"{total_time:.1f} seconds",
                'volumes_processed': self.stats.volumes_processed,
                'chapters_processed': self.stats.chapters_processed,
                'successful_chapters': successful_chapters,
                'failed_chapters': failed_chapters
            },
            'token_usage': {
                'total_deepseek_calls': self.stats.deepseek_calls,
                'total_tokens_used': self.stats.total_tokens_used,
                'estimated_cost': f"${self.stats.total_cost_estimate:.4f}",
                'average_tokens_per_chapter': self.stats.total_tokens_used // max(successful_chapters, 1)
            },
            'volume_analyses': self.volume_analyses,
            'chapter_results': [asdict(r) for r in self.chapter_results],
            'processing_errors': self.stats.processing_errors,
            'success': len(self.stats.processing_errors) == 0
        }
        
        return report
    
    def print_comprehensive_report(self, report: Dict[str, Any]):
        """Print comprehensive analysis report"""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE HIERARCHICAL ANALYSIS REPORT")
        print("="*80)
        
        # Metadata
        meta = report['analysis_metadata']
        print(f"\n⏱️  PROCESSING SUMMARY:")
        print(f"   Start time: {meta['start_time']}")
        print(f"   End time: {meta['end_time']}")
        print(f"   Total time: {meta['total_processing_time']}")
        print(f"   Volumes processed: {meta['volumes_processed']}")
        print(f"   Chapters processed: {meta['chapters_processed']}")
        print(f"   Success rate: {meta['successful_chapters']}/{meta['chapters_processed']} ({meta['successful_chapters']/max(meta['chapters_processed'],1)*100:.1f}%)")
        
        # Token usage
        tokens = report['token_usage']
        print(f"\n💰 TOKEN USAGE & COST:")
        print(f"   DeepSeek API calls: {tokens['total_deepseek_calls']}")
        print(f"   Total tokens used: {tokens['total_tokens_used']:,}")
        print(f"   Estimated cost: {tokens['estimated_cost']}")
        print(f"   Avg tokens/chapter: {tokens['average_tokens_per_chapter']}")
        
        # Volume results
        print(f"\n📚 VOLUME ANALYSIS RESULTS:")
        for vol_id, vol_analysis in report['volume_analyses'].items():
            print(f"\n   📖 Volume {vol_id}:")
            if 'volume_themes' in vol_analysis:
                print(f"      Themes: {', '.join(vol_analysis['volume_themes'][:3])}...")
            if 'character_arcs' in vol_analysis:
                char_count = len(vol_analysis['character_arcs'])
                print(f"      Character arcs: {char_count} characters")
            if 'plot_progression' in vol_analysis:
                print(f"      Plot: {vol_analysis['plot_progression'][:100]}...")
        
        # Chapter success breakdown
        print(f"\n📄 CHAPTER ANALYSIS BREAKDOWN:")
        for result in report['chapter_results'][:10]:  # Show first 10
            status = "✅" if result['analysis_success'] else "❌"
            print(f"   {status} Ch.{result['chapter_id']}: {result['tokens_used']} tokens, {result['processing_time']:.1f}s")
        
        if len(report['chapter_results']) > 10:
            print(f"   ... and {len(report['chapter_results']) - 10} more chapters")
        
        # Errors
        if report['processing_errors']:
            print(f"\n❌ PROCESSING ERRORS ({len(report['processing_errors'])}):")
            for error in report['processing_errors'][:5]:
                print(f"   - {error['type']}: {error['message'][:80]}...")

async def main():
    """Run comprehensive hierarchical analysis"""
    
    analyzer = ComprehensiveHierarchicalAnalyzer()
    
    try:
        # Run comprehensive analysis on first 2 volumes
        report = await analyzer.run_comprehensive_analysis(max_volumes=2)
        
        # Print detailed report
        analyzer.print_comprehensive_report(report)
        
        # Save report to file
        report_file = f"hierarchical_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if analyzer.deepseek_client.session:
            await analyzer.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())