#!/usr/bin/env python3
"""
Simplified 5-Volume Processor without problematic dependencies
Focus on core processing with character and storyline analysis
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VolumeProcessingResult:
    """Processing result for a volume"""
    volume_id: int
    volume_title: str
    processing_time: float
    tokens_used: int
    cost: float
    chunks_processed: int
    total_chunks: int
    success_rate: float
    analysis_data: Dict = None
    
    def __post_init__(self):
        if self.analysis_data is None:
            self.analysis_data = {}

class SimplifiedVolumeProcessor:
    """Simplified processor focusing on essential analysis"""
    
    def __init__(self):
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        self.global_stats = {
            'start_time': datetime.now(),
            'total_tokens': 0,
            'total_cost': 0.0,
            'total_deepseek_calls': 0
        }
        
        self.volume_results: List[VolumeProcessingResult] = []
        
        print("=" * 80)
        print("🚀 SIMPLIFIED 5-VOLUME PROCESSOR")
        print("=" * 80)
        print(f"🕐 Start time: {self.global_stats['start_time']}")
        print("📖 Target: Process volume 1 for testing")
        print("🎯 Focus: Character analysis, storyline, timeline events")
        print("⚡ Strategy: Sample-based analysis for speed")
        print("=" * 80)
    
    async def process_5_volumes_efficiently(self) -> Dict[str, Any]:
        """Process 5 volumes with efficient sampling strategy"""
        
        try:
            # Step 1: Extract and organize volume content
            print("\n📖 Step 1: Extracting volume content...")
            volume_data = await self._extract_volume_content_efficiently()
            
            if not volume_data['success']:
                raise Exception(f"Volume extraction failed: {volume_data['error']}")
            
            volumes_content = volume_data['volumes_content']
            print(f"✅ Extracted content for {len(volumes_content)} volumes")
            
            # Step 2: Process each volume with sampling (testing with 1 volume first)
            for volume_id in sorted(volumes_content.keys())[:1]:  # Test with first volume only
                print(f"\n" + "=" * 60)
                print(f"📚 PROCESSING VOLUME {volume_id}")
                print("=" * 60)
                
                result = await self._process_volume_with_sampling(volume_id, volumes_content[volume_id])
                self.volume_results.append(result)
                
                # Progress update
                print(f"\n✅ Volume {volume_id} complete!")
                print(f"   ⏱️  Time: {result.processing_time:.1f}s")
                print(f"   💰 Cost: ${result.cost:.4f}")
                print(f"   📊 Success: {result.success_rate:.1f}%")
                
                # Show current totals
                current_tokens = sum(r.tokens_used for r in self.volume_results)
                current_cost = sum(r.cost for r in self.volume_results)
                print(f"   📈 Running totals: {current_tokens:,} tokens, ${current_cost:.4f}")
            
            # Step 3: Generate final report
            final_report = self._generate_final_report()
            
            return final_report
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _extract_volume_content_efficiently(self) -> Dict[str, Any]:
        """Extract volume content with efficient organization"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            # Get all data points
            all_points = []
            offset = None
            batch_size = 1000
            
            print("   📥 Fetching data points...")
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
                print(f"      Retrieved {len(all_points)} points...")
                
                if len(batch_points) < batch_size:
                    break
                
                offset = points[1] if len(points) > 1 and points[1] else None
                if not offset:
                    break
            
            print(f"   ✅ Total points: {len(all_points)}")
            
            # Organize by volume efficiently
            volumes_content = defaultdict(lambda: {
                'title': '',
                'chunks': [],
                'total_chars': 0
            })
            
            volume_patterns = [r'魔法禁书目录\s*(\d+)', r'第(\d+)卷']
            header_patterns = [r'≡≡≡≡≡≡≡≡', r'作者：鎌池和马']
            
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
                    # Estimate by position
                    estimated_volume = (i // (len(all_points) // 22)) + 1
                    if estimated_volume <= 22:
                        volume_num = estimated_volume
                
                if volume_num and volume_num <= 1:  # Only first volume for testing
                    if is_header and not volumes_content[volume_num]['title']:
                        # Extract title
                        lines = content.split('\n')
                        for line in lines:
                            if '魔法禁书目录' in line and str(volume_num) in line:
                                volumes_content[volume_num]['title'] = line.strip()
                                break
                    else:
                        # Add content chunk
                        volumes_content[volume_num]['chunks'].append({
                            'content': content,
                            'length': len(content),
                            'index': len(volumes_content[volume_num]['chunks'])
                        })
                        volumes_content[volume_num]['total_chars'] += len(content)
            
            # Convert to regular dict with metadata
            result = {}
            for vol_id, vol_data in volumes_content.items():
                if vol_data['chunks']:
                    result[vol_id] = {
                        'title': vol_data['title'] or f"魔法禁书目录 第{vol_id}卷",
                        'chunks': vol_data['chunks'],
                        'total_chars': vol_data['total_chars'],
                        'total_chunks': len(vol_data['chunks'])
                    }
            
            print(f"   📚 Organized {len(result)} volumes:")
            for vol_id, vol_data in result.items():
                print(f"      Volume {vol_id}: {vol_data['total_chunks']} chunks, {vol_data['total_chars']:,} chars")
            
            return {
                'success': True,
                'volumes_content': result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_volume_with_sampling(self, volume_id: int, volume_data: Dict[str, Any]) -> VolumeProcessingResult:
        """Process a volume using strategic sampling"""
        
        start_time = time.time()
        
        print(f"📖 Volume {volume_id}: {volume_data['title']}")
        print(f"📊 Total chunks: {volume_data['total_chunks']}")
        print(f"📝 Total characters: {volume_data['total_chars']:,}")
        
        # Strategic sampling: 20 chunks max for analysis
        total_chunks = volume_data['total_chunks']
        chunks = volume_data['chunks']
        
        if total_chunks <= 20:
            sample_chunks = chunks
        else:
            # Sample strategically
            sample_indices = set()
            
            # Beginning (first 5)
            sample_indices.update(range(min(5, total_chunks)))
            
            # End (last 5)
            sample_indices.update(range(max(0, total_chunks - 5), total_chunks))
            
            # Middle points
            middle_points = [
                total_chunks // 4,
                total_chunks // 2,
                3 * total_chunks // 4
            ]
            sample_indices.update(middle_points)
            
            # Fill remaining with evenly spaced samples
            while len(sample_indices) < 20 and len(sample_indices) < total_chunks:
                step = total_chunks // (20 - len(sample_indices))
                for i in range(0, total_chunks, step):
                    if i not in sample_indices:
                        sample_indices.add(i)
                        if len(sample_indices) >= 20:
                            break
            
            sample_chunks = [chunks[i] for i in sorted(sample_indices)]
        
        print(f"📄 Analyzing {len(sample_chunks)} sample chunks")
        
        # Process in smaller batches
        batch_size = 5
        all_analyses = []
        total_tokens = 0
        total_cost = 0.0
        successful_analyses = 0
        
        for batch_start in range(0, len(sample_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(sample_chunks))
            batch_chunks = sample_chunks[batch_start:batch_end]
            
            print(f"  🔄 Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end})...")
            
            for chunk_idx, chunk in enumerate(batch_chunks):
                global_idx = batch_start + chunk_idx + 1
                
                try:
                    # Analyze chunk
                    analysis_result = await self._analyze_chunk_focused(
                        volume_id, chunk, global_idx, len(sample_chunks)
                    )
                    
                    if analysis_result['success']:
                        all_analyses.append(analysis_result['analysis'])
                        total_tokens += analysis_result['tokens_used']
                        total_cost += analysis_result['cost']
                        successful_analyses += 1
                    
                    # Brief delay
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    print(f"    ❌ Chunk {global_idx} error: {e}")
            
            # Show batch progress
            print(f"    ✅ Batch complete, running total: {total_tokens:,} tokens")
        
        # Generate volume summary
        print(f"  🔄 Generating volume summary...")
        try:
            summary_result = await self._generate_volume_summary_focused(
                volume_id, volume_data, all_analyses[:10]  # Use first 10 analyses
            )
            
            if summary_result['success']:
                total_tokens += summary_result['tokens_used']
                total_cost += summary_result['cost']
                volume_summary = summary_result['summary']
            else:
                volume_summary = {}
                
        except Exception as e:
            print(f"    ❌ Volume summary error: {e}")
            volume_summary = {}
        
        # Update global stats
        self.global_stats['total_tokens'] += total_tokens
        self.global_stats['total_cost'] += total_cost
        self.global_stats['total_deepseek_calls'] += successful_analyses + (1 if volume_summary else 0)
        
        processing_time = time.time() - start_time
        success_rate = (successful_analyses / len(sample_chunks)) * 100
        
        result = VolumeProcessingResult(
            volume_id=volume_id,
            volume_title=volume_data['title'],
            processing_time=processing_time,
            tokens_used=total_tokens,
            cost=total_cost,
            chunks_processed=successful_analyses,
            total_chunks=len(sample_chunks),
            success_rate=success_rate,
            analysis_data={
                'chunk_analyses': all_analyses,
                'volume_summary': volume_summary,
                'total_original_chunks': volume_data['total_chunks'],
                'sampling_ratio': len(sample_chunks) / volume_data['total_chunks']
            }
        )
        
        return result
    
    async def _analyze_chunk_focused(self, volume_id: int, chunk: Dict[str, Any], 
                                   chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Focused chunk analysis"""
        
        prompt = f"""
你是专业的中文小说分析专家。请简要分析以下片段：

第{volume_id}卷，片段 {chunk_index}/{total_chunks}

请返回JSON格式分析：
{{
  "characters": ["出现的角色名"],
  "key_events": ["重要事件"],
  "emotions": ["情感氛围"],
  "themes": ["主题元素"],
  "summary": "片段要点（20字以内）"
}}

内容：
{chunk['content'][:1000]}
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            if response.get("success"):
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                cost = tokens_used * 0.00002
                
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
                    
                    analysis = json.loads(json_text)
                    
                    return {
                        'success': True,
                        'analysis': analysis,
                        'tokens_used': tokens_used,
                        'cost': cost
                    }
                    
                except json.JSONDecodeError:
                    return {
                        'success': True,
                        'analysis': {'raw_response': response["response"]},
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
    
    async def _generate_volume_summary_focused(self, volume_id: int, volume_data: Dict[str, Any],
                                             chunk_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate focused volume summary"""
        
        prompt = f"""
基于以下分析，总结第{volume_id}卷：

卷信息：
- 标题：{volume_data['title']}
- 总chunk数：{volume_data['total_chunks']}
- 分析样本：{len(chunk_analyses)}

片段分析结果：
{json.dumps(chunk_analyses, ensure_ascii=False)}

请返回JSON格式总结：
{{
  "main_characters": ["主要角色"],
  "character_relationships": ["角色关系"],
  "major_events": ["重大事件"],
  "plot_summary": "情节概述",
  "themes": ["主要主题"],
  "volume_significance": "本卷在系列中的重要性",
  "timeline_events": [
    {{
      "event": "事件",
      "characters": ["参与者"],
      "significance": "重要性"
    }}
  ]
}}
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1500,
                temperature=0.2
            )
            
            if response.get("success"):
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                cost = tokens_used * 0.00002
                
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
                    
                    summary = json.loads(json_text)
                    
                    return {
                        'success': True,
                        'summary': summary,
                        'tokens_used': tokens_used,
                        'cost': cost
                    }
                    
                except json.JSONDecodeError:
                    return {
                        'success': True,
                        'summary': {'raw_response': response["response"]},
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
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.global_stats['start_time']).total_seconds()
        
        # Aggregate statistics
        total_chunks_processed = sum(r.chunks_processed for r in self.volume_results)
        total_chunks_sampled = sum(r.total_chunks for r in self.volume_results)
        successful_volumes = len([r for r in self.volume_results if r.tokens_used > 0])
        
        # Extract all characters and events
        all_characters = set()
        all_events = []
        all_themes = set()
        
        for result in self.volume_results:
            if 'chunk_analyses' in result.analysis_data:
                for analysis in result.analysis_data['chunk_analyses']:
                    if 'characters' in analysis:
                        all_characters.update(analysis['characters'])
                    if 'key_events' in analysis:
                        all_events.extend(analysis['key_events'])
                    if 'themes' in analysis:
                        all_themes.update(analysis['themes'])
        
        report = {
            'processing_metadata': {
                'start_time': self.global_stats['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'total_processing_time': f"{total_time:.1f} seconds",
                'total_processing_minutes': f"{total_time/60:.1f} minutes",
                'volumes_processed': len(self.volume_results),
                'successful_volumes': successful_volumes,
                'total_chunks_processed': total_chunks_processed,
                'total_chunks_sampled': total_chunks_sampled
            },
            'token_and_cost_analysis': {
                'total_tokens_used': self.global_stats['total_tokens'],
                'total_cost': f"${self.global_stats['total_cost']:.4f}",
                'total_deepseek_calls': self.global_stats['total_deepseek_calls'],
                'average_cost_per_volume': f"${self.global_stats['total_cost']/max(successful_volumes, 1):.4f}",
                'tokens_per_second': self.global_stats['total_tokens'] / max(total_time, 1)
            },
            'volume_results': [
                {
                    'volume_id': result.volume_id,
                    'volume_title': result.volume_title,
                    'processing_time': f"{result.processing_time:.1f}s",
                    'tokens_used': result.tokens_used,
                    'cost': f"${result.cost:.4f}",
                    'chunks_processed': result.chunks_processed,
                    'chunks_sampled': result.total_chunks,
                    'success_rate': f"{result.success_rate:.1f}%",
                    'analysis_data': result.analysis_data
                }
                for result in self.volume_results
            ],
            'content_analysis': {
                'unique_characters_found': len(all_characters),
                'characters_list': list(all_characters)[:20],  # First 20
                'total_events_extracted': len(all_events),
                'unique_themes_found': len(all_themes),
                'themes_list': list(all_themes)
            },
            'performance_metrics': {
                'chunks_per_minute': (total_chunks_processed / max(total_time, 1)) * 60,
                'average_processing_time_per_volume': total_time / max(len(self.volume_results), 1),
                'efficiency_rating': 'HIGH' if total_time < 600 else 'MEDIUM' if total_time < 1200 else 'LOW'
            },
            'success': successful_volumes == len(self.volume_results)
        }
        
        return report
    
    def print_final_report(self, report: Dict[str, Any]):
        """Print comprehensive final report"""
        
        print("\n" + "=" * 80)
        print("📊 SIMPLIFIED 5-VOLUME PROCESSING REPORT")
        print("=" * 80)
        
        # Processing metadata
        meta = report['processing_metadata']
        print(f"\n⏱️  PROCESSING SUMMARY:")
        print(f"   🕐 Total time: {meta['total_processing_time']} ({meta['total_processing_minutes']})")
        print(f"   📚 Volumes: {meta['successful_volumes']}/{meta['volumes_processed']}")
        print(f"   📄 Chunks processed: {meta['total_chunks_processed']}")
        
        # Token and cost analysis
        tokens = report['token_and_cost_analysis']
        print(f"\n💰 COST ANALYSIS:")
        print(f"   🔢 Total tokens: {tokens['total_tokens_used']:,}")
        print(f"   💵 Total cost: {tokens['total_cost']}")
        print(f"   📞 API calls: {tokens['total_deepseek_calls']}")
        print(f"   💳 Avg cost/volume: {tokens['average_cost_per_volume']}")
        
        # Volume breakdown
        print(f"\n📚 VOLUME BREAKDOWN:")
        for vol in report['volume_results']:
            print(f"\n   📖 Volume {vol['volume_id']}: {vol['volume_title']}")
            print(f"      ⏱️  Time: {vol['processing_time']}")
            print(f"      💰 Cost: {vol['cost']}")
            print(f"      📊 Success: {vol['success_rate']}")
            print(f"      📄 Chunks: {vol['chunks_processed']}/{vol['chunks_sampled']}")
        
        # Content analysis
        content = report['content_analysis']
        print(f"\n📖 CONTENT ANALYSIS:")
        print(f"   👥 Characters found: {content['unique_characters_found']}")
        if content['characters_list']:
            print(f"      {', '.join(content['characters_list'][:10])}...")
        print(f"   📅 Events extracted: {content['total_events_extracted']}")
        print(f"   🎭 Themes found: {content['unique_themes_found']}")
        if content['themes_list']:
            print(f"      {', '.join(content['themes_list'])}")
        
        # Performance
        perf = report['performance_metrics']
        print(f"\n⚡ PERFORMANCE:")
        print(f"   📊 Chunks/minute: {perf['chunks_per_minute']:.1f}")
        print(f"   ⏱️  Avg time/volume: {perf['average_processing_time_per_volume']:.1f}s")
        print(f"   🎯 Efficiency: {perf['efficiency_rating']}")
        
        # Final status
        status = "✅ SUCCESS" if report['success'] else "⚠️  PARTIAL SUCCESS"
        print(f"\n🏁 FINAL STATUS: {status}")
        print("=" * 80)

async def main():
    """Main execution function"""
    
    processor = SimplifiedVolumeProcessor()
    
    try:
        # Process 5 volumes efficiently
        report = await processor.process_5_volumes_efficiently()
        
        # Print comprehensive report
        processor.print_final_report(report)
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"simplified_5_volume_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json_report = json.loads(json.dumps(report, default=str, ensure_ascii=False))
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Report saved to: {report_file}")
        
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