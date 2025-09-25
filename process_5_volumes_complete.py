#!/usr/bin/env python3
"""
Process First 5 Volumes from Complete Dataset (2588 points)
With comprehensive progress monitoring and token counting
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

# Setup UTF-8 environment first
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from hierarchical_chapter_parser import HierarchicalChapterParser
from deepseek_integration import DeepSeekClient, create_deepseek_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VolumeProcessingStats:
    """Track processing statistics for each volume"""
    volume_id: int
    volume_title: str
    chapters_count: int
    start_time: datetime
    end_time: datetime = None
    processing_time: float = 0.0
    chapters_processed: int = 0
    chapters_successful: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    deepseek_calls: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class Complete5VolumeProcessor:
    """Process first 5 volumes from complete dataset with detailed monitoring"""
    
    def __init__(self):
        self.parser = HierarchicalChapterParser()
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        self.global_stats = {
            'start_time': datetime.now(),
            'volumes_stats': [],
            'total_tokens': 0,
            'total_cost': 0.0,
            'total_deepseek_calls': 0,
            'total_processing_time': 0.0
        }
        
        # Analysis prompts
        self.analysis_prompts = self._setup_prompts()
        
        print("=" * 80)
        print("🚀 COMPLETE 5-VOLUME PROCESSING FROM FULL DATASET")
        print("=" * 80)
        print(f"🕐 Start time: {self.global_stats['start_time']}")
        print("📖 Target: Process volumes 1-5 from complete 22-volume dataset")
        print("💰 Monitor: Token usage, costs, and processing time")
        print("🎯 Goal: All 5 volumes with comprehensive AI analysis")
        print("=" * 80)
    
    def _setup_prompts(self) -> Dict[str, str]:
        """Setup analysis prompts"""
        return {
            'chapter_analysis': """
你是专业的中文小说分析专家。请分析以下章节：

章节信息：
- 第{volume_id}卷第{chapter_id}章
- 标题：{chapter_title}
- 字数：{word_count}

请返回JSON格式分析：
{{
  "characters": [
    {{
      "name": "角色名",
      "role": "主角/配角/反派",
      "actions": ["行动1", "行动2"],
      "development": "发展变化",
      "importance": 0.8
    }}
  ],
  "plot_events": [
    {{
      "event": "事件描述",
      "type": "conflict/resolution/development",
      "importance": 0.8,
      "consequence": "影响"
    }}
  ],
  "themes": ["主题1", "主题2"],
  "mood": "章节氛围",
  "conflicts": ["冲突1", "冲突2"],
  "summary": "章节核心内容总结（50字以内）"
}}

章节内容：
{content}
""",
            
            'volume_summary': """
请基于以下章节分析，总结第{volume_id}卷的整体内容：

卷标题：{volume_title}
章节数：{chapter_count}
总字数：{total_words}
章节分析：{chapter_analyses}

请返回JSON格式的卷级总结：
{{
  "volume_themes": ["卷级主题1", "卷级主题2"],
  "main_characters": [
    {{
      "name": "主要角色",
      "role": "角色定位",
      "arc_summary": "在本卷的发展轨迹"
    }}
  ],
  "plot_arc": "整卷情节发展概述",
  "key_events": ["关键事件1", "关键事件2"],
  "climax_chapters": ["高潮章节"],
  "character_development": "主要角色发展变化",
  "volume_conclusion": "卷结论或悬念",
  "connection_to_next": "与下一卷的连接点"
}}
"""
        }
    
    async def process_complete_5_volumes(self) -> Dict[str, Any]:
        """Process the first 5 volumes from complete dataset"""
        
        try:
            # Step 1: Get complete dataset and parse structure
            print("\n📖 Step 1: Fetching complete dataset and parsing structure...")
            structure_result = await self._parse_complete_novel_structure()
            
            if not structure_result['success']:
                raise Exception(f"Structure parsing failed: {structure_result['error']}")
            
            parsed_structure = structure_result['parsed_structure']
            all_chapters = structure_result['all_chapters']
            
            print(f"✅ Complete structure parsed successfully:")
            print(f"   📚 Total volumes available: {len(parsed_structure)}")
            print(f"   📄 Total chapters available: {len(all_chapters)}")
            print(f"   📝 Total content: {structure_result['total_content_length']:,} characters")
            print(f"   🎯 Processing first 5 volumes...")
            
            # Step 2: Identify first 5 volumes (excluding volume 0 if exists)
            volumes_to_process = []
            for vol_id in sorted(parsed_structure.keys()):
                if vol_id > 0 and len(volumes_to_process) < 5:
                    volumes_to_process.append(vol_id)
            
            print(f"\n🎯 Target volumes: {volumes_to_process}")
            
            # Verify we have 5 volumes
            if len(volumes_to_process) < 5:
                print(f"⚠️  Warning: Only {len(volumes_to_process)} volumes available")
            
            # Step 3: Process each of the first 5 volumes
            for i, volume_id in enumerate(volumes_to_process, 1):
                print(f"\n" + "=" * 60)
                print(f"📚 PROCESSING VOLUME {volume_id} ({i}/{len(volumes_to_process)})")
                print("=" * 60)
                
                await self._process_single_volume(volume_id, parsed_structure[volume_id])
                
                # Progress update
                self._print_progress_update(i, len(volumes_to_process))
            
            # Step 4: Generate final comprehensive report
            print(f"\n📊 Generating comprehensive final report...")
            final_report = self._generate_final_report()
            
            return final_report
            
        except Exception as e:
            logger.error(f"Complete volume processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _parse_complete_novel_structure(self) -> Dict[str, Any]:
        """Parse complete novel structure from all Qdrant data"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            # Get ALL data points
            print("   📥 Fetching all data points...")
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
                
                print(f"      Retrieved batch: {len(batch_points)} points (total: {len(all_points)})")
                
                if len(batch_points) < batch_size:
                    break
                
                offset = points[1] if len(points) > 1 and points[1] else None
                if not offset:
                    break
            
            print(f"   ✅ Total data points retrieved: {len(all_points)}")
            
            # Combine all content
            combined_content = ""
            for point in all_points:
                if 'chunk' in point.payload:
                    combined_content += point.payload['chunk'] + "\n\n"
            
            print(f"   📝 Combined content length: {len(combined_content):,} characters")
            
            # Parse with hierarchical parser
            print("   🔧 Running hierarchical parser...")
            chapters = self.parser.parse_content_hierarchy(combined_content)
            
            print(f"   ✅ Parsing complete: {len(self.parser.parsed_structure)} volumes, {len(chapters)} chapters")
            
            return {
                'success': True,
                'parsed_structure': self.parser.parsed_structure,
                'all_chapters': chapters,
                'total_content_length': len(combined_content),
                'total_data_points': len(all_points)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_single_volume(self, volume_id: int, volume_structure):
        """Process a single volume with detailed monitoring"""
        
        volume_start_time = datetime.now()
        volume_stats = VolumeProcessingStats(
            volume_id=volume_id,
            volume_title=volume_structure.volume_title,
            chapters_count=len(volume_structure.chapters),
            start_time=volume_start_time
        )
        
        print(f"📖 Volume {volume_id}: {volume_structure.volume_title}")
        print(f"📄 Chapters to process: {len(volume_structure.chapters)}")
        print(f"📝 Total words in volume: {sum(ch.word_count for ch in volume_structure.chapters):,}")
        print(f"🕐 Start time: {volume_start_time.strftime('%H:%M:%S')}")
        
        chapter_analyses = []
        
        # Process each chapter
        for chapter_num, chapter in enumerate(volume_structure.chapters, 1):
            print(f"\n  🔄 Processing Chapter {chapter.chapter_id} ({chapter_num}/{len(volume_structure.chapters)})...")
            print(f"      📝 Content length: {len(chapter.content):,} chars, Words: {chapter.word_count:,}")
            
            chapter_start = time.time()
            
            try:
                # Analyze chapter with DeepSeek
                analysis_result = await self._analyze_chapter_with_deepseek(
                    volume_id, chapter
                )
                
                if analysis_result['success']:
                    chapter_analyses.append(analysis_result['analysis'])
                    volume_stats.chapters_successful += 1
                    volume_stats.total_tokens += analysis_result['tokens_used']
                    volume_stats.total_cost += analysis_result['cost']
                    volume_stats.deepseek_calls += 1
                    
                    chapter_time = time.time() - chapter_start
                    print(f"    ✅ Chapter {chapter.chapter_id}: {analysis_result['tokens_used']} tokens, {chapter_time:.1f}s")
                    print(f"       💰 Cost: ${analysis_result['cost']:.4f}")
                    
                    # Show brief analysis preview
                    if 'summary' in analysis_result['analysis']:
                        summary = analysis_result['analysis']['summary'][:60] + "..." if len(analysis_result['analysis']['summary']) > 60 else analysis_result['analysis']['summary']
                        print(f"       📋 Summary: {summary}")
                    
                else:
                    volume_stats.errors.append(f"Chapter {chapter.chapter_id}: {analysis_result['error']}")
                    print(f"    ❌ Chapter {chapter.chapter_id}: {analysis_result['error']}")
                
                volume_stats.chapters_processed += 1
                
                # Rate limiting - small delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                error_msg = f"Chapter {chapter.chapter_id} error: {e}"
                volume_stats.errors.append(error_msg)
                print(f"    ❌ {error_msg}")
        
        # Generate volume summary
        print(f"\n  🔄 Generating volume-level summary...")
        try:
            volume_summary = await self._generate_volume_summary(
                volume_id, volume_structure, chapter_analyses
            )
            
            if volume_summary['success']:
                volume_stats.total_tokens += volume_summary['tokens_used']
                volume_stats.total_cost += volume_summary['cost']
                volume_stats.deepseek_calls += 1
                print(f"    ✅ Volume summary: {volume_summary['tokens_used']} tokens")
                
                # Show brief summary preview
                if 'summary' in volume_summary and 'plot_arc' in volume_summary['summary']:
                    plot_preview = volume_summary['summary']['plot_arc'][:80] + "..." if len(volume_summary['summary']['plot_arc']) > 80 else volume_summary['summary']['plot_arc']
                    print(f"       📖 Plot arc: {plot_preview}")
            
        except Exception as e:
            volume_stats.errors.append(f"Volume summary error: {e}")
            print(f"    ❌ Volume summary error: {e}")
        
        # Finalize volume stats
        volume_stats.end_time = datetime.now()
        volume_stats.processing_time = (volume_stats.end_time - volume_stats.start_time).total_seconds()
        
        # Update global stats
        self.global_stats['volumes_stats'].append(volume_stats)
        self.global_stats['total_tokens'] += volume_stats.total_tokens
        self.global_stats['total_cost'] += volume_stats.total_cost
        self.global_stats['total_deepseek_calls'] += volume_stats.deepseek_calls
        self.global_stats['total_processing_time'] += volume_stats.processing_time
        
        # Print volume completion summary
        success_rate = (volume_stats.chapters_successful / volume_stats.chapters_count) * 100
        print(f"\n✅ Volume {volume_id} completed:")
        print(f"   ⏱️  Processing time: {volume_stats.processing_time:.1f} seconds")
        print(f"   📊 Success rate: {volume_stats.chapters_successful}/{volume_stats.chapters_count} ({success_rate:.1f}%)")
        print(f"   💰 Tokens used: {volume_stats.total_tokens:,}")
        print(f"   💵 Cost: ${volume_stats.total_cost:.4f}")
        print(f"   🔧 DeepSeek calls: {volume_stats.deepseek_calls}")
        
        if volume_stats.errors:
            print(f"   ⚠️  Errors: {len(volume_stats.errors)}")
    
    async def _analyze_chapter_with_deepseek(self, volume_id: int, chapter) -> Dict[str, Any]:
        """Analyze a chapter using DeepSeek with token tracking"""
        
        try:
            # Prepare content (limit to reasonable size for analysis)
            content_for_analysis = chapter.content[:2000] if len(chapter.content) > 2000 else chapter.content
            
            # Create prompt
            prompt = self.analysis_prompts['chapter_analysis'].format(
                volume_id=volume_id,
                chapter_id=chapter.chapter_id,
                chapter_title=chapter.chapter_title,
                word_count=chapter.word_count,
                content=content_for_analysis
            )
            
            # Query DeepSeek
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1200,
                temperature=0.3
            )
            
            if response.get("success"):
                # Calculate tokens and cost
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                cost = tokens_used * 0.00002  # Rough estimate
                
                # Try to parse JSON
                try:
                    # Extract JSON from markdown if present
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
                    
                    parsed_analysis = json.loads(json_text)
                    
                except json.JSONDecodeError:
                    parsed_analysis = {"raw_response": response["response"]}
                
                return {
                    'success': True,
                    'analysis': parsed_analysis,
                    'tokens_used': tokens_used,
                    'cost': cost,
                    'raw_response': response["response"]
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown DeepSeek error'),
                    'tokens_used': 0,
                    'cost': 0.0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tokens_used': 0,
                'cost': 0.0
            }
    
    async def _generate_volume_summary(self, volume_id: int, volume_structure, chapter_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate volume-level summary"""
        
        try:
            # Prepare chapter analyses summary (limit size)
            analyses_summary = json.dumps(chapter_analyses[:5], ensure_ascii=False)
            total_words = sum(ch.word_count for ch in volume_structure.chapters)
            
            prompt = self.analysis_prompts['volume_summary'].format(
                volume_id=volume_id,
                volume_title=volume_structure.volume_title,
                chapter_count=len(volume_structure.chapters),
                total_words=total_words,
                chapter_analyses=analyses_summary
            )
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            if response.get("success"):
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                cost = tokens_used * 0.00002
                
                # Try to parse JSON
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
                    
                    parsed_summary = json.loads(json_text)
                    
                except json.JSONDecodeError:
                    parsed_summary = {"raw_response": response["response"]}
                
                return {
                    'success': True,
                    'summary': parsed_summary,
                    'tokens_used': tokens_used,
                    'cost': cost
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error'),
                    'tokens_used': 0,
                    'cost': 0.0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tokens_used': 0,
                'cost': 0.0
            }
    
    def _print_progress_update(self, completed: int, total: int):
        """Print progress update"""
        
        progress_percent = (completed / total) * 100
        elapsed_time = (datetime.now() - self.global_stats['start_time']).total_seconds()
        estimated_total_time = elapsed_time * total / completed
        remaining_time = estimated_total_time - elapsed_time
        
        print(f"\n" + "🔄" * 20 + " PROGRESS UPDATE " + "🔄" * 20)
        print(f"📊 Progress: {completed}/{total} volumes ({progress_percent:.1f}%)")
        print(f"⏱️  Elapsed time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"⏳ Estimated remaining: {remaining_time:.1f} seconds ({remaining_time/60:.1f} minutes)")
        print(f"💰 Total tokens so far: {self.global_stats['total_tokens']:,}")
        print(f"💵 Total cost so far: ${self.global_stats['total_cost']:.4f}")
        print(f"🔧 Total API calls: {self.global_stats['total_deepseek_calls']}")
        print("🔄" * 56)
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.global_stats['start_time']).total_seconds()
        
        # Calculate aggregate statistics
        total_chapters = sum(vol.chapters_count for vol in self.global_stats['volumes_stats'])
        successful_chapters = sum(vol.chapters_successful for vol in self.global_stats['volumes_stats'])
        total_errors = sum(len(vol.errors) for vol in self.global_stats['volumes_stats'])
        
        report = {
            'processing_metadata': {
                'start_time': self.global_stats['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'total_processing_time': f"{total_time:.1f} seconds",
                'total_processing_minutes': f"{total_time/60:.1f} minutes",
                'volumes_processed': len(self.global_stats['volumes_stats']),
                'chapters_processed': total_chapters,
                'successful_chapters': successful_chapters,
                'success_rate': f"{successful_chapters/max(total_chapters, 1)*100:.1f}%"
            },
            'token_and_cost_analysis': {
                'total_tokens_used': self.global_stats['total_tokens'],
                'total_cost': f"${self.global_stats['total_cost']:.4f}",
                'total_deepseek_calls': self.global_stats['total_deepseek_calls'],
                'average_tokens_per_chapter': self.global_stats['total_tokens'] // max(successful_chapters, 1),
                'average_cost_per_chapter': f"${self.global_stats['total_cost']/max(successful_chapters, 1):.4f}",
                'average_tokens_per_volume': self.global_stats['total_tokens'] // max(len(self.global_stats['volumes_stats']), 1),
                'tokens_per_second': self.global_stats['total_tokens'] / max(total_time, 1)
            },
            'volume_breakdown': [
                {
                    'volume_id': vol.volume_id,
                    'volume_title': vol.volume_title,
                    'chapters_count': vol.chapters_count,
                    'chapters_successful': vol.chapters_successful,
                    'processing_time': f"{vol.processing_time:.1f}s",
                    'processing_minutes': f"{vol.processing_time/60:.1f}m",
                    'tokens_used': vol.total_tokens,
                    'cost': f"${vol.total_cost:.4f}",
                    'deepseek_calls': vol.deepseek_calls,
                    'errors_count': len(vol.errors),
                    'success_rate': f"{vol.chapters_successful/max(vol.chapters_count, 1)*100:.1f}%"
                }
                for vol in self.global_stats['volumes_stats']
            ],
            'performance_metrics': {
                'chapters_per_minute': (successful_chapters / max(total_time, 1)) * 60,
                'volumes_per_hour': (len(self.global_stats['volumes_stats']) / max(total_time, 1)) * 3600,
                'average_processing_time_per_volume': total_time / max(len(self.global_stats['volumes_stats']), 1),
                'total_errors': total_errors,
                'error_rate': f"{total_errors/max(total_chapters, 1)*100:.1f}%"
            },
            'success': total_errors == 0
        }
        
        return report
    
    def print_final_report(self, report: Dict[str, Any]):
        """Print comprehensive final report"""
        
        print("\n" + "=" * 80)
        print("📊 FINAL 5-VOLUME PROCESSING REPORT (COMPLETE DATASET)")
        print("=" * 80)
        
        # Processing metadata
        meta = report['processing_metadata']
        print(f"\n⏱️  PROCESSING SUMMARY:")
        print(f"   🕐 Start: {meta['start_time']}")
        print(f"   🕐 End: {meta['end_time']}")
        print(f"   ⏱️  Total time: {meta['total_processing_time']} ({meta['total_processing_minutes']})")
        print(f"   📚 Volumes processed: {meta['volumes_processed']}/5")
        print(f"   📄 Chapters processed: {meta['chapters_processed']}")
        print(f"   ✅ Success rate: {meta['success_rate']}")
        
        # Token and cost analysis
        tokens = report['token_and_cost_analysis']
        print(f"\n💰 TOKEN USAGE & COSTS:")
        print(f"   🔢 Total tokens: {tokens['total_tokens_used']:,}")
        print(f"   💵 Total cost: {tokens['total_cost']}")
        print(f"   📞 API calls: {tokens['total_deepseek_calls']}")
        print(f"   📊 Avg tokens/chapter: {tokens['average_tokens_per_chapter']}")
        print(f"   💳 Avg cost/chapter: {tokens['average_cost_per_chapter']}")
        print(f"   📖 Avg tokens/volume: {tokens['average_tokens_per_volume']}")
        print(f"   ⚡ Tokens/second: {tokens['tokens_per_second']:.1f}")
        
        # Volume breakdown
        print(f"\n📚 VOLUME-BY-VOLUME BREAKDOWN:")
        for vol in report['volume_breakdown']:
            print(f"\n   📖 Volume {vol['volume_id']}: {vol['volume_title']}")
            print(f"      📄 Chapters: {vol['chapters_successful']}/{vol['chapters_count']} ({vol['success_rate']})")
            print(f"      ⏱️  Time: {vol['processing_time']} ({vol['processing_minutes']})")
            print(f"      💰 Tokens: {vol['tokens_used']:,} | Cost: {vol['cost']}")
            print(f"      📞 API calls: {vol['deepseek_calls']}")
            if vol['errors_count'] > 0:
                print(f"      ⚠️  Errors: {vol['errors_count']}")
        
        # Performance metrics
        perf = report['performance_metrics']
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   📊 Chapters/minute: {perf['chapters_per_minute']:.1f}")
        print(f"   📚 Volumes/hour: {perf['volumes_per_hour']:.1f}")
        print(f"   ⏱️  Avg time/volume: {perf['average_processing_time_per_volume']:.1f}s")
        print(f"   ❌ Total errors: {perf['total_errors']}")
        print(f"   📉 Error rate: {perf['error_rate']}")
        
        # Final status
        status = "✅ SUCCESS" if report['success'] else "⚠️  COMPLETED WITH ERRORS"
        print(f"\n🏁 FINAL STATUS: {status}")
        print("=" * 80)

async def main():
    """Main execution function"""
    
    processor = Complete5VolumeProcessor()
    
    try:
        # Process 5 volumes from complete dataset
        report = await processor.process_complete_5_volumes()
        
        # Print comprehensive report
        processor.print_final_report(report)
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"complete_5_volume_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # Convert dataclass objects to dicts for JSON serialization
            json_report = json.loads(json.dumps(report, default=str, ensure_ascii=False))
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
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