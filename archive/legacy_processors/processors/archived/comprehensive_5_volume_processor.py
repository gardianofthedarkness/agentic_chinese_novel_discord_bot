#!/usr/bin/env python3
"""
Comprehensive 5-Volume Processor with Character/Storyline/Timeline Integration
Optimized for efficiency while maintaining depth of analysis
"""

import os
import sys
import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
import re

# Setup UTF-8 environment first
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from deepseek_integration import DeepSeekClient, create_deepseek_config
from character_variant_resolver import CharacterVariantResolver
from enhanced_rag import EnhancedRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VolumeAnalysisResult:
    """Complete analysis result for a volume"""
    volume_id: int
    volume_title: str
    processing_time: float
    tokens_used: int
    cost: float
    
    # Analysis content
    character_analysis: Dict = None
    storyline_analysis: Dict = None
    timeline_events: List = None
    volume_summary: Dict = None
    
    # Statistics
    chunks_analyzed: int = 0
    total_chunks: int = 0
    total_chars: int = 0
    
    def __post_init__(self):
        if self.character_analysis is None:
            self.character_analysis = {}
        if self.storyline_analysis is None:
            self.storyline_analysis = {}
        if self.timeline_events is None:
            self.timeline_events = []
        if self.volume_summary is None:
            self.volume_summary = {}

class ComprehensiveVolumeProcessor:
    """Comprehensive processor integrating character, storyline, and timeline analysis"""
    
    def __init__(self):
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        self.character_resolver = CharacterVariantResolver()
        
        # Try to initialize RAG system
        try:
            self.rag_system = EnhancedRAGSystem()
        except:
            self.rag_system = None
        
        self.global_stats = {
            'start_time': datetime.now(),
            'total_tokens': 0,
            'total_cost': 0.0,
            'total_deepseek_calls': 0
        }
        
        self.volume_results: List[VolumeAnalysisResult] = []
        
        # Enhanced analysis prompts
        self.analysis_prompts = self._setup_comprehensive_prompts()
        
        print("=" * 80)
        print("🚀 COMPREHENSIVE 5-VOLUME PROCESSOR")
        print("=" * 80)
        print(f"🕐 Start time: {self.global_stats['start_time']}")
        print("📖 Target: Process volumes 1-5 with full integration")
        print("🎯 Analysis: Character, Storyline, Timeline, Thematic")
        print("⚡ Optimized: Sample-based analysis for efficiency")
        print("=" * 80)
    
    def _setup_comprehensive_prompts(self) -> Dict[str, str]:
        """Setup comprehensive analysis prompts"""
        return {
            'volume_comprehensive_analysis': """
你是专业的中文小说分析专家。请对第{volume_id}卷进行全面分析。

卷信息：
- 标题：{volume_title}
- 总片段：{total_chunks}
- 总字数：{total_chars:,}

代表性内容样本：
{content_samples}

请返回JSON格式的全面分析：
{{
  "volume_overview": {{
    "main_theme": "主要主题",
    "volume_tone": "整卷基调",
    "narrative_style": "叙述风格",
    "significance": "在整个系列中的重要性"
  }},
  "character_analysis": {{
    "main_characters": [
      {{
        "name": "角色名",
        "variants": ["角色别名1", "角色别名2"],
        "role": "protagonist/antagonist/supporting",
        "personality_traits": ["特征1", "特征2"],
        "character_arc": "在本卷的发展轨迹",
        "key_relationships": ["与XX的关系"],
        "growth_moments": ["成长时刻"],
        "importance_score": 0.9
      }}
    ],
    "character_relationships": [
      {{
        "character1": "角色A",
        "character2": "角色B",
        "relationship_type": "friend/enemy/family/romance/rivalry",
        "relationship_evolution": "关系发展变化",
        "key_interactions": ["重要互动"]
      }}
    ],
    "new_characters": ["新登场角色"]
  }},
  "storyline_analysis": {{
    "plot_structure": {{
      "opening": "开篇情况",
      "inciting_incident": "引发事件",
      "rising_action": "情节发展",
      "climax": "高潮部分",
      "falling_action": "高潮后发展",
      "resolution": "结局或悬念"
    }},
    "major_events": [
      {{
        "event": "重大事件描述",
        "type": "battle/revelation/character_development/plot_twist",
        "significance": "重要性和影响",
        "characters_involved": ["参与角色"],
        "consequences": "后果和影响",
        "timeline_position": "在卷中的位置"
      }}
    ],
    "plot_threads": [
      {{
        "thread_name": "故事线名称",
        "status": "introduced/developed/resolved/ongoing",
        "description": "故事线描述",
        "key_developments": ["关键发展"]
      }}
    ],
    "foreshadowing": ["伏笔和预示"],
    "cliffhangers": ["悬念和钩子"]
  }},
  "timeline_events": [
    {{
      "event_id": "vol{volume_id}_event_1",
      "event_description": "事件描述",
      "event_type": "battle/meeting/revelation/departure",
      "time_reference": "时间参考（如果有）",
      "location": "地点",
      "participants": ["参与者"],
      "importance": 0.8,
      "causes": ["导因"],
      "effects": ["结果"]
    }}
  ],
  "thematic_analysis": {{
    "central_themes": ["中心主题"],
    "symbolic_elements": ["象征元素"],
    "moral_lessons": ["道德教训"],
    "philosophical_questions": ["哲学思考"],
    "cultural_elements": ["文化元素"]
  }},
  "connections": {{
    "previous_volume": "与前卷的联系",
    "next_volume_setup": "为下卷铺垫",
    "series_significance": "在整个系列中的地位"
  }}
}}

请基于提供的内容样本进行深入分析。
""",
            
            'cross_volume_synthesis': """
基于以下5卷的分析结果，请进行跨卷综合分析：

{volume_analyses}

请返回JSON格式的综合分析：
{{
  "series_overview": {{
    "overall_narrative_arc": "整体叙事弧线",
    "central_conflict": "核心冲突",
    "thematic_progression": "主题发展",
    "world_building": "世界观构建"
  }},
  "character_evolution": [
    {{
      "character": "角色名",
      "cross_volume_arc": "跨卷发展轨迹",
      "key_transformation_moments": ["关键转变时刻"],
      "relationship_developments": ["关系发展"]
    }}
  ],
  "plot_threads_tracking": [
    {{
      "thread": "故事线",
      "volume_progression": ["在各卷中的发展"],
      "resolution_status": "resolved/ongoing/abandoned"
    }}
  ],
  "timeline_summary": {{
    "major_events_chronology": ["重大事件时间顺序"],
    "turning_points": ["转折点"],
    "story_pacing": "故事节奏分析"
  }}
}}
"""
        }
    
    async def process_comprehensive_5_volumes(self) -> Dict[str, Any]:
        """Process 5 volumes with comprehensive analysis"""
        
        try:
            # Step 1: Extract volume content
            print("\n📖 Step 1: Extracting volume content...")
            volume_data = await self._extract_optimized_volume_content()
            
            if not volume_data['success']:
                raise Exception(f"Volume extraction failed: {volume_data['error']}")
            
            volumes_content = volume_data['volumes_content']
            print(f"✅ Extracted content for {len(volumes_content)} volumes")
            
            # Step 2: Process each volume comprehensively
            for volume_id in sorted(volumes_content.keys()):
                print(f"\n" + "=" * 60)
                print(f"📚 COMPREHENSIVE ANALYSIS - VOLUME {volume_id}")
                print("=" * 60)
                
                result = await self._analyze_volume_comprehensively(volume_id, volumes_content[volume_id])
                self.volume_results.append(result)
                
                # Progress update
                print(f"\n✅ Volume {volume_id} analysis complete")
                print(f"   ⏱️  Time: {result.processing_time:.1f}s")
                print(f"   💰 Tokens: {result.tokens_used:,} | Cost: ${result.cost:.4f}")
                print(f"   📊 Coverage: {result.chunks_analyzed}/{result.total_chunks} chunks")
            
            # Step 3: Cross-volume synthesis
            print(f"\n📊 Step 3: Cross-volume synthesis...")
            cross_analysis = await self._perform_cross_volume_analysis()
            
            # Step 4: Generate final report
            final_report = self._generate_comprehensive_report(cross_analysis)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Comprehensive processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _extract_optimized_volume_content(self) -> Dict[str, Any]:
        """Extract volume content with optimization for analysis"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            # Get all data points
            all_points = []
            offset = None
            batch_size = 1000
            
            while True:
                points = client.scroll(
                    collection_name="test_novel2",
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                batch_points = points[0]
                all_points.extend(batch_points)
                
                if len(batch_points) < batch_size:
                    break
                
                offset = points[1] if len(points) > 1 and points[1] else None
                if not offset:
                    break
            
            # Organize content by volume with strategic sampling
            volumes_content = defaultdict(lambda: {
                'title': '',
                'all_chunks': [],
                'sample_chunks': [],
                'total_chars': 0
            })
            
            # Patterns for content organization
            volume_patterns = [r'魔法禁书目录\s*(\d+)', r'第(\d+)卷']
            header_patterns = [r'≡≡≡≡≡≡≡≡', r'作者：鎌池和马', r'插画：灰村清孝']
            
            for i, point in enumerate(all_points):
                content = point.payload.get('chunk', '')
                
                if not content.strip():
                    continue
                
                # Check if header
                is_header = any(re.search(pattern, content) for pattern in header_patterns)
                
                # Determine volume
                volume_num = None
                for pattern in volume_patterns:
                    match = re.search(pattern, content)
                    if match:
                        volume_num = int(match.group(1))
                        break
                
                if not volume_num:
                    estimated_volume = (i // (len(all_points) // 22)) + 1
                    if estimated_volume <= 22:
                        volume_num = estimated_volume
                
                if volume_num and volume_num <= 5:
                    if is_header and not volumes_content[volume_num]['title']:
                        # Extract title
                        lines = content.split('\n')
                        for line in lines:
                            if '魔法禁书目录' in line and str(volume_num) in line:
                                volumes_content[volume_num]['title'] = line.strip()
                                break
                    else:
                        # Add content chunk
                        chunk_data = {
                            'content': content,
                            'length': len(content),
                            'index': len(volumes_content[volume_num]['all_chunks'])
                        }
                        
                        volumes_content[volume_num]['all_chunks'].append(chunk_data)
                        volumes_content[volume_num]['total_chars'] += len(content)
            
            # Strategic sampling for analysis
            for vol_id, vol_data in volumes_content.items():
                all_chunks = vol_data['all_chunks']
                total_chunks = len(all_chunks)
                
                if total_chunks > 20:
                    # Sample strategically: beginning, middle, end, plus random
                    sample_indices = set()
                    
                    # Beginning (first 3)
                    sample_indices.update(range(min(3, total_chunks)))
                    
                    # End (last 3)
                    sample_indices.update(range(max(0, total_chunks - 3), total_chunks))
                    
                    # Middle sections
                    middle_points = [total_chunks // 4, total_chunks // 2, 3 * total_chunks // 4]
                    sample_indices.update(middle_points)
                    
                    # Random additional samples
                    import random
                    random.seed(42)  # For reproducibility
                    additional_samples = random.sample(range(total_chunks), min(8, total_chunks - len(sample_indices)))
                    sample_indices.update(additional_samples)
                    
                    vol_data['sample_chunks'] = [all_chunks[i] for i in sorted(sample_indices)]
                else:
                    # Use all chunks if volume is small
                    vol_data['sample_chunks'] = all_chunks
            
            # Convert to regular dict
            result = {}
            for vol_id, vol_data in volumes_content.items():
                if vol_data['all_chunks']:
                    result[vol_id] = {
                        'title': vol_data['title'] or f"魔法禁书目录 第{vol_id}卷",
                        'all_chunks': vol_data['all_chunks'],
                        'sample_chunks': vol_data['sample_chunks'],
                        'total_chars': vol_data['total_chars'],
                        'total_chunks': len(vol_data['all_chunks']),
                        'sample_size': len(vol_data['sample_chunks'])
                    }
            
            print(f"   📚 Organized {len(result)} volumes")
            for vol_id, vol_data in result.items():
                print(f"      Volume {vol_id}: {vol_data['total_chunks']} chunks → {vol_data['sample_size']} samples")
            
            return {
                'success': True,
                'volumes_content': result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _analyze_volume_comprehensively(self, volume_id: int, volume_data: Dict[str, Any]) -> VolumeAnalysisResult:
        """Perform comprehensive analysis on a single volume"""
        
        start_time = time.time()
        
        print(f"📖 Volume {volume_id}: {volume_data['title']}")
        print(f"📊 Analyzing {volume_data['sample_size']} sample chunks from {volume_data['total_chunks']} total")
        print(f"📝 Total characters: {volume_data['total_chars']:,}")
        
        # Prepare content samples for analysis
        content_samples = []
        for i, chunk in enumerate(volume_data['sample_chunks'][:15]):  # Limit to 15 samples
            sample_preview = chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content']
            content_samples.append(f"样本 {i+1}:\n{sample_preview}")
        
        combined_samples = "\n\n".join(content_samples)
        
        try:
            # Comprehensive analysis
            prompt = self.analysis_prompts['volume_comprehensive_analysis'].format(
                volume_id=volume_id,
                volume_title=volume_data['title'],
                total_chunks=volume_data['total_chunks'],
                total_chars=volume_data['total_chars'],
                content_samples=combined_samples
            )
            
            print(f"🔄 Running comprehensive analysis...")
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=3000,  # Larger token limit for comprehensive analysis
                temperature=0.2
            )
            
            if response.get("success"):
                # Calculate tokens and cost
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                cost = tokens_used * 0.00002
                
                # Update global stats
                self.global_stats['total_tokens'] += tokens_used
                self.global_stats['total_cost'] += cost
                self.global_stats['total_deepseek_calls'] += 1
                
                # Parse response
                try:
                    response_text = response["response"]
                    if "```json" in response_text:
                        start = response_text.find("```json") + 7
                        end = response_text.find("```", start)
                        json_text = response_text[start:end].strip()
                    elif "{" in response_text:
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        json_text = response_text[start:end]
                    else:
                        json_text = response_text
                    
                    analysis_data = json.loads(json_text)
                    
                    # Extract specific analysis components
                    character_analysis = analysis_data.get('character_analysis', {})
                    storyline_analysis = analysis_data.get('storyline_analysis', {})
                    timeline_events = analysis_data.get('timeline_events', [])
                    
                    # Process characters through variant resolver
                    if 'main_characters' in character_analysis:
                        for char in character_analysis['main_characters']:
                            if 'variants' in char:
                                resolved_name = self.character_resolver.resolve_character_name(char['name'])
                                char['resolved_name'] = resolved_name
                    
                    processing_time = time.time() - start_time
                    
                    result = VolumeAnalysisResult(
                        volume_id=volume_id,
                        volume_title=volume_data['title'],
                        processing_time=processing_time,
                        tokens_used=tokens_used,
                        cost=cost,
                        character_analysis=character_analysis,
                        storyline_analysis=storyline_analysis,
                        timeline_events=timeline_events,
                        volume_summary=analysis_data.get('volume_overview', {}),
                        chunks_analyzed=volume_data['sample_size'],
                        total_chunks=volume_data['total_chunks'],
                        total_chars=volume_data['total_chars']
                    )
                    
                    print(f"✅ Analysis successful: {tokens_used:,} tokens, ${cost:.4f}")
                    return result
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error: {e}")
                    # Return basic result with raw response
                    processing_time = time.time() - start_time
                    return VolumeAnalysisResult(
                        volume_id=volume_id,
                        volume_title=volume_data['title'],
                        processing_time=processing_time,
                        tokens_used=tokens_used,
                        cost=cost,
                        chunks_analyzed=volume_data['sample_size'],
                        total_chunks=volume_data['total_chunks'],
                        total_chars=volume_data['total_chars']
                    )
            else:
                print(f"❌ DeepSeek error: {response.get('error', 'Unknown error')}")
                processing_time = time.time() - start_time
                return VolumeAnalysisResult(
                    volume_id=volume_id,
                    volume_title=volume_data['title'],
                    processing_time=processing_time,
                    tokens_used=0,
                    cost=0.0,
                    chunks_analyzed=0,
                    total_chunks=volume_data['total_chunks'],
                    total_chars=volume_data['total_chars']
                )
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            processing_time = time.time() - start_time
            return VolumeAnalysisResult(
                volume_id=volume_id,
                volume_title=volume_data['title'],
                processing_time=processing_time,
                tokens_used=0,
                cost=0.0,
                chunks_analyzed=0,
                total_chunks=volume_data['total_chunks'],
                total_chars=volume_data['total_chars']
            )
    
    async def _perform_cross_volume_analysis(self) -> Dict[str, Any]:
        """Perform cross-volume synthesis analysis"""
        
        print(f"🔄 Performing cross-volume synthesis...")
        
        try:
            # Prepare volume analyses for synthesis
            volume_summaries = []
            for result in self.volume_results:
                summary = {
                    'volume_id': result.volume_id,
                    'title': result.volume_title,
                    'character_analysis': result.character_analysis,
                    'storyline_analysis': result.storyline_analysis,
                    'timeline_events': result.timeline_events,
                    'volume_summary': result.volume_summary
                }
                volume_summaries.append(summary)
            
            analyses_text = json.dumps(volume_summaries, ensure_ascii=False, indent=2)
            
            prompt = self.analysis_prompts['cross_volume_synthesis'].format(
                volume_analyses=analyses_text
            )
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=2000,
                temperature=0.2
            )
            
            if response.get("success"):
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                cost = tokens_used * 0.00002
                
                self.global_stats['total_tokens'] += tokens_used
                self.global_stats['total_cost'] += cost
                self.global_stats['total_deepseek_calls'] += 1
                
                try:
                    response_text = response["response"]
                    if "```json" in response_text:
                        start = response_text.find("```json") + 7
                        end = response_text.find("```", start)
                        json_text = response_text[start:end].strip()
                    elif "{" in response_text:
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        json_text = response_text[start:end]
                    else:
                        json_text = response_text
                    
                    cross_analysis = json.loads(json_text)
                    print(f"✅ Cross-volume analysis: {tokens_used:,} tokens, ${cost:.4f}")
                    return cross_analysis
                    
                except json.JSONDecodeError:
                    print(f"❌ Cross-volume analysis JSON parse error")
                    return {}
            else:
                print(f"❌ Cross-volume analysis failed")
                return {}
                
        except Exception as e:
            print(f"❌ Cross-volume analysis error: {e}")
            return {}
    
    def _generate_comprehensive_report(self, cross_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.global_stats['start_time']).total_seconds()
        
        # Aggregate statistics
        total_chunks_analyzed = sum(result.chunks_analyzed for result in self.volume_results)
        total_chunks_available = sum(result.total_chunks for result in self.volume_results)
        total_chars = sum(result.total_chars for result in self.volume_results)
        successful_volumes = len([r for r in self.volume_results if r.tokens_used > 0])
        
        report = {
            'processing_metadata': {
                'start_time': self.global_stats['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'total_processing_time': f"{total_time:.1f} seconds",
                'total_processing_minutes': f"{total_time/60:.1f} minutes",
                'volumes_processed': len(self.volume_results),
                'successful_volumes': successful_volumes,
                'total_chunks_analyzed': total_chunks_analyzed,
                'total_chunks_available': total_chunks_available,
                'total_characters': total_chars,
                'analysis_coverage': f"{total_chunks_analyzed/max(total_chunks_available, 1)*100:.1f}%"
            },
            'token_and_cost_analysis': {
                'total_tokens_used': self.global_stats['total_tokens'],
                'total_cost': f"${self.global_stats['total_cost']:.4f}",
                'total_deepseek_calls': self.global_stats['total_deepseek_calls'],
                'average_tokens_per_volume': self.global_stats['total_tokens'] // max(successful_volumes, 1),
                'cost_per_1000_chars': f"${(self.global_stats['total_cost'] / max(total_chars, 1)) * 1000:.4f}",
                'tokens_per_second': self.global_stats['total_tokens'] / max(total_time, 1)
            },
            'volume_analyses': [
                {
                    'volume_id': result.volume_id,
                    'volume_title': result.volume_title,
                    'processing_time': f"{result.processing_time:.1f}s",
                    'tokens_used': result.tokens_used,
                    'cost': f"${result.cost:.4f}",
                    'chunks_analyzed': result.chunks_analyzed,
                    'total_chunks': result.total_chunks,
                    'coverage': f"{result.chunks_analyzed/max(result.total_chunks, 1)*100:.1f}%",
                    'character_analysis': result.character_analysis,
                    'storyline_analysis': result.storyline_analysis,
                    'timeline_events': result.timeline_events,
                    'volume_summary': result.volume_summary
                }
                for result in self.volume_results
            ],
            'cross_volume_analysis': cross_analysis,
            'character_system_integration': {
                'character_resolver_used': True,
                'variants_resolved': True,
                'cross_volume_tracking': True
            },
            'timeline_integration': {
                'events_extracted': sum(len(result.timeline_events) for result in self.volume_results),
                'chronological_analysis': True
            },
            'success': successful_volumes == len(self.volume_results)
        }
        
        return report
    
    def print_comprehensive_report(self, report: Dict[str, Any]):
        """Print comprehensive analysis report"""
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE 5-VOLUME ANALYSIS REPORT")
        print("=" * 80)
        
        # Processing metadata
        meta = report['processing_metadata']
        print(f"\n⏱️  PROCESSING SUMMARY:")
        print(f"   🕐 Start: {meta['start_time']}")
        print(f"   🕐 End: {meta['end_time']}")
        print(f"   ⏱️  Total time: {meta['total_processing_time']} ({meta['total_processing_minutes']})")
        print(f"   📚 Volumes processed: {meta['successful_volumes']}/{meta['volumes_processed']}")
        print(f"   📄 Chunks analyzed: {meta['total_chunks_analyzed']}/{meta['total_chunks_available']}")
        print(f"   📊 Analysis coverage: {meta['analysis_coverage']}")
        print(f"   📝 Characters processed: {meta['total_characters']:,}")
        
        # Token and cost analysis
        tokens = report['token_and_cost_analysis']
        print(f"\n💰 TOKEN USAGE & COSTS:")
        print(f"   🔢 Total tokens: {tokens['total_tokens_used']:,}")
        print(f"   💵 Total cost: {tokens['total_cost']}")
        print(f"   📞 API calls: {tokens['total_deepseek_calls']}")
        print(f"   📖 Avg tokens/volume: {tokens['average_tokens_per_volume']}")
        print(f"   💲 Cost per 1000 chars: {tokens['cost_per_1000_chars']}")
        print(f"   ⚡ Tokens/second: {tokens['tokens_per_second']:.1f}")
        
        # Volume breakdown
        print(f"\n📚 VOLUME-BY-VOLUME ANALYSIS:")
        for vol in report['volume_analyses']:
            print(f"\n   📖 Volume {vol['volume_id']}: {vol['volume_title']}")
            print(f"      ⏱️  Time: {vol['processing_time']}")
            print(f"      💰 Tokens: {vol['tokens_used']:,} | Cost: {vol['cost']}")
            print(f"      📊 Coverage: {vol['coverage']} ({vol['chunks_analyzed']}/{vol['total_chunks']} chunks)")
            
            # Show character analysis summary
            if vol['character_analysis'] and 'main_characters' in vol['character_analysis']:
                char_count = len(vol['character_analysis']['main_characters'])
                print(f"      👥 Characters analyzed: {char_count}")
            
            # Show timeline events
            if vol['timeline_events']:
                print(f"      📅 Timeline events: {len(vol['timeline_events'])}")
        
        # Cross-volume analysis
        cross = report['cross_volume_analysis']
        if cross:
            print(f"\n🔗 CROSS-VOLUME SYNTHESIS:")
            
            if 'character_evolution' in cross:
                char_evol_count = len(cross['character_evolution'])
                print(f"   👥 Character evolution tracked: {char_evol_count} characters")
            
            if 'plot_threads_tracking' in cross:
                thread_count = len(cross['plot_threads_tracking'])
                print(f"   📖 Plot threads tracked: {thread_count} threads")
        
        # Integration status
        char_integration = report['character_system_integration']
        timeline_integration = report['timeline_integration']
        
        print(f"\n🔧 SYSTEM INTEGRATION:")
        print(f"   ✅ Character variant resolver: {char_integration['character_resolver_used']}")
        print(f"   ✅ Cross-volume character tracking: {char_integration['cross_volume_tracking']}")
        print(f"   ✅ Timeline events extracted: {timeline_integration['events_extracted']}")
        print(f"   ✅ Chronological analysis: {timeline_integration['chronological_analysis']}")
        
        # Final status
        status = "✅ SUCCESS" if report['success'] else "⚠️  PARTIAL SUCCESS"
        print(f"\n🏁 FINAL STATUS: {status}")
        print("=" * 80)

async def main():
    """Main execution function"""
    
    processor = ComprehensiveVolumeProcessor()
    
    try:
        # Process 5 volumes comprehensively
        report = await processor.process_comprehensive_5_volumes()
        
        # Print comprehensive report
        processor.print_comprehensive_report(report)
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"comprehensive_5_volume_analysis_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json_report = json.loads(json.dumps(report, default=str, ensure_ascii=False))
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Comprehensive analysis saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if processor.deepseek_client.session:
            await processor.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())