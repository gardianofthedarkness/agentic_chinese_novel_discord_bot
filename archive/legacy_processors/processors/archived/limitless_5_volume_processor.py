#!/usr/bin/env python3
"""
Limitless 5-Volume Processor - No Timeouts, Complete Processing
Processes all 5 volumes with full character/storyline/timeline analysis
Uploads results to database upon completion
"""

import os
import sys
import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
import sqlite3

# Setup UTF-8 environment first
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from deepseek_integration import DeepSeekClient, create_deepseek_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('limitless_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ChapterAnalysis:
    """Individual chapter analysis result"""
    volume_id: int
    chunk_id: int
    characters: List[str]
    events: List[str]
    themes: List[str]
    emotions: List[str]
    summary: str
    content_preview: str
    analysis_timestamp: str
    tokens_used: int
    processing_time: float

@dataclass
class VolumeAnalysis:
    """Complete volume analysis result"""
    volume_id: int
    volume_title: str
    total_chunks: int
    total_characters: int
    processing_start: str
    processing_end: str
    total_processing_time: float
    total_tokens: int
    total_cost: float
    
    # Analysis results
    main_characters: List[str]
    character_relationships: List[str]
    major_events: List[str]
    plot_summary: str
    themes: List[str]
    timeline_events: List[Dict]
    volume_significance: str
    
    # Chapter analyses
    chapter_analyses: List[ChapterAnalysis]

class LimitlessVolumeProcessor:
    """Limitless processor with database storage"""
    
    def __init__(self):
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Initialize database
        self.db_path = "limitless_processing_results.db"
        self._initialize_database()
        
        self.global_stats = {
            'start_time': datetime.now(),
            'total_tokens': 0,
            'total_cost': 0.0,
            'total_deepseek_calls': 0,
            'volumes_completed': 0
        }
        
        self.volume_results: List[VolumeAnalysis] = []
        
        print("=" * 100)
        print("🚀 LIMITLESS 5-VOLUME PROCESSOR - NO TIMEOUTS")
        print("=" * 100)
        print(f"🕐 Start time: {self.global_stats['start_time']}")
        print("📖 Target: Complete processing of volumes 1-5")
        print("🎯 Analysis: Full character, storyline, timeline, thematic analysis")
        print("💾 Database: Results will be stored in limitless_processing_results.db")
        print("⏳ Mode: No timeouts - will process until completion")
        print("=" * 100)
    
    def _initialize_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Volume analyses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS volume_analyses (
            volume_id INTEGER PRIMARY KEY,
            volume_title TEXT,
            total_chunks INTEGER,
            total_characters INTEGER,
            processing_start TEXT,
            processing_end TEXT,
            total_processing_time REAL,
            total_tokens INTEGER,
            total_cost REAL,
            main_characters TEXT,
            character_relationships TEXT,
            major_events TEXT,
            plot_summary TEXT,
            themes TEXT,
            timeline_events TEXT,
            volume_significance TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Chapter analyses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chapter_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            volume_id INTEGER,
            chunk_id INTEGER,
            characters TEXT,
            events TEXT,
            themes TEXT,
            emotions TEXT,
            summary TEXT,
            content_preview TEXT,
            analysis_timestamp TEXT,
            tokens_used INTEGER,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (volume_id) REFERENCES volume_analyses (volume_id)
        )
        ''')
        
        # Processing log table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event_type TEXT,
            message TEXT,
            data TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def _log_to_database(self, event_type: str, message: str, data: str = ""):
        """Log events to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO processing_log (timestamp, event_type, message, data)
            VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), event_type, message, data))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log to database: {e}")
    
    async def process_limitless_5_volumes(self) -> Dict[str, Any]:
        """Process all 5 volumes without limitations"""
        
        self._log_to_database("START", "Beginning limitless 5-volume processing")
        
        try:
            # Step 1: Extract all volume content
            logger.info("Step 1: Extracting volume content...")
            volume_data = await self._extract_all_volume_content()
            
            if not volume_data['success']:
                raise Exception(f"Volume extraction failed: {volume_data['error']}")
            
            volumes_content = volume_data['volumes_content']
            logger.info(f"Extracted content for {len(volumes_content)} volumes")
            
            self._log_to_database("EXTRACTION_COMPLETE", f"Extracted {len(volumes_content)} volumes", 
                                json.dumps({k: v['total_chunks'] for k, v in volumes_content.items()}))
            
            # Step 2: Process each volume completely
            for volume_id in sorted(volumes_content.keys())[:5]:
                logger.info(f"\n{'='*80}")
                logger.info(f"PROCESSING VOLUME {volume_id} - LIMITLESS MODE")
                logger.info(f"{'='*80}")
                
                volume_start_time = datetime.now()
                self._log_to_database("VOLUME_START", f"Starting volume {volume_id}", 
                                    json.dumps(volumes_content[volume_id], default=str))
                
                try:
                    result = await self._process_volume_limitless(volume_id, volumes_content[volume_id])
                    self.volume_results.append(result)
                    
                    # Save to database immediately
                    self._save_volume_to_database(result)
                    
                    self.global_stats['volumes_completed'] += 1
                    
                    volume_end_time = datetime.now()
                    volume_time = (volume_end_time - volume_start_time).total_seconds()
                    
                    logger.info(f"✅ Volume {volume_id} COMPLETED")
                    logger.info(f"   ⏱️  Time: {volume_time:.1f}s ({volume_time/60:.1f}m)")
                    logger.info(f"   💰 Tokens: {result.total_tokens:,} | Cost: ${result.total_cost:.4f}")
                    logger.info(f"   📄 Chunks: {len(result.chapter_analyses)}/{result.total_chunks}")
                    logger.info(f"   💾 Saved to database")
                    
                    # Update running totals
                    total_tokens = sum(r.total_tokens for r in self.volume_results)
                    total_cost = sum(r.total_cost for r in self.volume_results)
                    logger.info(f"   📈 Running totals: {total_tokens:,} tokens, ${total_cost:.4f}")
                    
                    self._log_to_database("VOLUME_COMPLETE", f"Completed volume {volume_id}", 
                                        json.dumps(asdict(result), default=str))
                    
                except Exception as e:
                    logger.error(f"❌ Volume {volume_id} failed: {e}")
                    self._log_to_database("VOLUME_ERROR", f"Volume {volume_id} failed", str(e))
                    continue
            
            # Step 3: Generate final comprehensive report
            logger.info("\nGenerating final comprehensive report...")
            final_report = self._generate_comprehensive_final_report()
            
            # Save final report to database
            self._save_final_report_to_database(final_report)
            
            self._log_to_database("COMPLETE", "Limitless processing completed successfully", 
                                json.dumps(final_report, default=str))
            
            return final_report
            
        except Exception as e:
            logger.error(f"Limitless processing failed: {e}")
            self._log_to_database("FATAL_ERROR", "Processing failed", str(e))
            return {'success': False, 'error': str(e)}
    
    async def _extract_all_volume_content(self) -> Dict[str, Any]:
        """Extract all volume content efficiently"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            # Get all data points
            logger.info("Fetching all data points from Qdrant...")
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
                logger.info(f"   Retrieved {len(all_points)} points...")
                
                if len(batch_points) < batch_size:
                    break
                
                offset = points[1] if len(points) > 1 and points[1] else None
                if not offset:
                    break
            
            logger.info(f"Total points retrieved: {len(all_points)}")
            
            # Organize by volume
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
                    estimated_volume = (i // (len(all_points) // 22)) + 1
                    if estimated_volume <= 22:
                        volume_num = estimated_volume
                
                if volume_num and volume_num <= 5:  # First 5 volumes
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
            
            # Convert to regular dict
            result = {}
            for vol_id, vol_data in volumes_content.items():
                if vol_data['chunks']:
                    result[vol_id] = {
                        'title': vol_data['title'] or f"魔法禁书目录 第{vol_id}卷",
                        'chunks': vol_data['chunks'],
                        'total_chars': vol_data['total_chars'],
                        'total_chunks': len(vol_data['chunks'])
                    }
            
            logger.info(f"Organized {len(result)} volumes:")
            for vol_id, vol_data in result.items():
                logger.info(f"   Volume {vol_id}: {vol_data['total_chunks']} chunks, {vol_data['total_chars']:,} chars")
            
            return {
                'success': True,
                'volumes_content': result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_volume_limitless(self, volume_id: int, volume_data: Dict[str, Any]) -> VolumeAnalysis:
        """Process a complete volume without limitations"""
        
        start_time = datetime.now()
        
        logger.info(f"📖 Volume {volume_id}: {volume_data['title']}")
        logger.info(f"📊 Total chunks: {volume_data['total_chunks']}")
        logger.info(f"📝 Total characters: {volume_data['total_chars']:,}")
        logger.info(f"🔄 Processing ALL chunks (no sampling)")
        
        chapter_analyses = []
        total_tokens = 0
        total_cost = 0.0
        
        # Process ALL chunks
        chunks = volume_data['chunks']
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            
            logger.info(f"   🔄 Processing chunk {chunk_idx + 1}/{len(chunks)}...")
            
            try:
                # Analyze chunk with comprehensive prompt
                analysis_result = await self._analyze_chunk_comprehensive(
                    volume_id, chunk, chunk_idx + 1, len(chunks)
                )
                
                if analysis_result['success']:
                    chunk_analysis = ChapterAnalysis(
                        volume_id=volume_id,
                        chunk_id=chunk_idx,
                        characters=analysis_result['analysis'].get('characters', []),
                        events=analysis_result['analysis'].get('key_events', []),
                        themes=analysis_result['analysis'].get('themes', []),
                        emotions=analysis_result['analysis'].get('emotions', []),
                        summary=analysis_result['analysis'].get('summary', ''),
                        content_preview=chunk['content'][:200],
                        analysis_timestamp=datetime.now().isoformat(),
                        tokens_used=analysis_result['tokens_used'],
                        processing_time=time.time() - chunk_start_time
                    )
                    
                    chapter_analyses.append(chunk_analysis)
                    total_tokens += analysis_result['tokens_used']
                    total_cost += analysis_result['cost']
                    
                    # Update global stats
                    self.global_stats['total_tokens'] += analysis_result['tokens_used']
                    self.global_stats['total_cost'] += analysis_result['cost']
                    self.global_stats['total_deepseek_calls'] += 1
                    
                    # Progress logging
                    if (chunk_idx + 1) % 10 == 0:
                        progress = ((chunk_idx + 1) / len(chunks)) * 100
                        logger.info(f"      📊 Progress: {progress:.1f}% ({chunk_idx + 1}/{len(chunks)})")
                        logger.info(f"      💰 Tokens so far: {total_tokens:,}, Cost: ${total_cost:.4f}")
                
                # Brief delay to be respectful to API
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"   ❌ Chunk {chunk_idx + 1} failed: {e}")
                continue
        
        # Generate comprehensive volume summary
        logger.info(f"🔄 Generating comprehensive volume summary...")
        try:
            summary_result = await self._generate_comprehensive_volume_summary(
                volume_id, volume_data, chapter_analyses
            )
            
            if summary_result['success']:
                total_tokens += summary_result['tokens_used']
                total_cost += summary_result['cost']
                volume_summary_data = summary_result['summary']
            else:
                volume_summary_data = {}
                
        except Exception as e:
            logger.error(f"❌ Volume summary failed: {e}")
            volume_summary_data = {}
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create volume analysis result
        volume_analysis = VolumeAnalysis(
            volume_id=volume_id,
            volume_title=volume_data['title'],
            total_chunks=volume_data['total_chunks'],
            total_characters=volume_data['total_chars'],
            processing_start=start_time.isoformat(),
            processing_end=end_time.isoformat(),
            total_processing_time=processing_time,
            total_tokens=total_tokens,
            total_cost=total_cost,
            main_characters=volume_summary_data.get('main_characters', []),
            character_relationships=volume_summary_data.get('character_relationships', []),
            major_events=volume_summary_data.get('major_events', []),
            plot_summary=volume_summary_data.get('plot_summary', ''),
            themes=volume_summary_data.get('themes', []),
            timeline_events=volume_summary_data.get('timeline_events', []),
            volume_significance=volume_summary_data.get('volume_significance', ''),
            chapter_analyses=chapter_analyses
        )
        
        return volume_analysis
    
    async def _analyze_chunk_comprehensive(self, volume_id: int, chunk: Dict[str, Any],
                                         chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Comprehensive chunk analysis"""
        
        prompt = f"""
你是专业的中文小说分析专家。请对以下片段进行深入分析：

卷信息：第{volume_id}卷，片段 {chunk_index}/{total_chunks}

请返回JSON格式的详细分析：
{{
  "characters": ["出现的所有角色名"],
  "key_events": ["重要事件和情节点"],
  "emotions": ["情感氛围", "角色情感状态"],
  "themes": ["体现的主题元素"],
  "dialogue_summary": ["重要对话要点"],
  "action_summary": ["重要行动描述"],
  "plot_significance": "该片段在整体情节中的重要性",
  "character_development": "角色发展变化",
  "foreshadowing": ["伏笔或预示"],
  "summary": "片段核心内容总结（50字以内）"
}}

请仔细分析以下内容：
{chunk['content']}
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1000,
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
    
    async def _generate_comprehensive_volume_summary(self, volume_id: int, volume_data: Dict[str, Any],
                                                   chapter_analyses: List[ChapterAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive volume summary"""
        
        # Aggregate data from chapter analyses
        all_characters = set()
        all_events = []
        all_themes = set()
        
        for analysis in chapter_analyses:
            all_characters.update(analysis.characters)
            all_events.extend(analysis.events)
            all_themes.update(analysis.themes)
        
        # Create summary for prompt
        analyses_summary = {
            'total_chunks_analyzed': len(chapter_analyses),
            'unique_characters': list(all_characters)[:20],  # Limit for prompt size
            'major_events': all_events[:30],  # Limit for prompt size
            'themes': list(all_themes)
        }
        
        prompt = f"""
基于对第{volume_id}卷的全面分析，请生成综合总结：

卷信息：
- 标题：{volume_data['title']}
- 总chunks：{volume_data['total_chunks']}
- 已分析：{len(chapter_analyses)}个chunks
- 总字数：{volume_data['total_chars']:,}

分析汇总：
{json.dumps(analyses_summary, ensure_ascii=False, indent=2)}

请返回JSON格式的全面总结：
{{
  "main_characters": ["主要角色名单"],
  "character_relationships": ["重要角色关系"],
  "major_events": ["重大事件列表"],
  "plot_summary": "整卷情节概述（200字以内）",
  "themes": ["主要主题"],
  "character_development": "角色发展分析",
  "volume_significance": "本卷在整个系列中的重要性",
  "timeline_events": [
    {{
      "event": "重要事件",
      "characters": ["参与角色"],
      "significance": "重要性描述",
      "location": "发生地点",
      "consequences": "后果影响"
    }}
  ],
  "cliffhangers": ["悬念和伏笔"],
  "connections_to_next_volume": "与下一卷的联系"
}}
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=2000,
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
    
    def _save_volume_to_database(self, volume_analysis: VolumeAnalysis):
        """Save volume analysis to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save volume analysis
            cursor.execute('''
            INSERT OR REPLACE INTO volume_analyses (
                volume_id, volume_title, total_chunks, total_characters,
                processing_start, processing_end, total_processing_time,
                total_tokens, total_cost, main_characters, character_relationships,
                major_events, plot_summary, themes, timeline_events, volume_significance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                volume_analysis.volume_id,
                volume_analysis.volume_title,
                volume_analysis.total_chunks,
                volume_analysis.total_characters,
                volume_analysis.processing_start,
                volume_analysis.processing_end,
                volume_analysis.total_processing_time,
                volume_analysis.total_tokens,
                volume_analysis.total_cost,
                json.dumps(volume_analysis.main_characters, ensure_ascii=False),
                json.dumps(volume_analysis.character_relationships, ensure_ascii=False),
                json.dumps(volume_analysis.major_events, ensure_ascii=False),
                volume_analysis.plot_summary,
                json.dumps(volume_analysis.themes, ensure_ascii=False),
                json.dumps(volume_analysis.timeline_events, ensure_ascii=False),
                volume_analysis.volume_significance
            ))
            
            # Save chapter analyses
            for chapter in volume_analysis.chapter_analyses:
                cursor.execute('''
                INSERT INTO chapter_analyses (
                    volume_id, chunk_id, characters, events, themes, emotions,
                    summary, content_preview, analysis_timestamp, tokens_used, processing_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chapter.volume_id,
                    chapter.chunk_id,
                    json.dumps(chapter.characters, ensure_ascii=False),
                    json.dumps(chapter.events, ensure_ascii=False),
                    json.dumps(chapter.themes, ensure_ascii=False),
                    json.dumps(chapter.emotions, ensure_ascii=False),
                    chapter.summary,
                    chapter.content_preview,
                    chapter.analysis_timestamp,
                    chapter.tokens_used,
                    chapter.processing_time
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Volume {volume_analysis.volume_id} saved to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to save volume {volume_analysis.volume_id} to database: {e}")
    
    def _save_final_report_to_database(self, report: Dict[str, Any]):
        """Save final report to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO processing_log (timestamp, event_type, message, data)
            VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                "FINAL_REPORT",
                "Complete processing report",
                json.dumps(report, ensure_ascii=False, default=str)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Final report saved to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final report: {e}")
    
    def _generate_comprehensive_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.global_stats['start_time']).total_seconds()
        
        # Aggregate all data
        total_chunks = sum(vol.total_chunks for vol in self.volume_results)
        total_chapters_analyzed = sum(len(vol.chapter_analyses) for vol in self.volume_results)
        total_characters_processed = sum(vol.total_characters for vol in self.volume_results)
        total_tokens = sum(vol.total_tokens for vol in self.volume_results)
        total_cost = sum(vol.total_cost for vol in self.volume_results)
        
        # Extract all unique characters and themes
        all_characters = set()
        all_themes = set()
        all_events = []
        
        for volume in self.volume_results:
            all_characters.update(volume.main_characters)
            all_themes.update(volume.themes)
            all_events.extend(volume.major_events)
            
            for chapter in volume.chapter_analyses:
                all_characters.update(chapter.characters)
                all_themes.update(chapter.themes)
                all_events.extend(chapter.events)
        
        report = {
            'processing_metadata': {
                'start_time': self.global_stats['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'total_processing_time_seconds': total_time,
                'total_processing_time_minutes': total_time / 60,
                'total_processing_time_hours': total_time / 3600,
                'volumes_completed': len(self.volume_results),
                'total_chunks_in_volumes': total_chunks,
                'total_chapters_analyzed': total_chapters_analyzed,
                'total_characters_processed': total_characters_processed,
                'completion_rate': f"{total_chapters_analyzed/max(total_chunks, 1)*100:.1f}%"
            },
            'cost_and_token_analysis': {
                'total_tokens_used': total_tokens,
                'total_cost_usd': total_cost,
                'total_deepseek_calls': self.global_stats['total_deepseek_calls'],
                'average_tokens_per_volume': total_tokens // max(len(self.volume_results), 1),
                'average_cost_per_volume': total_cost / max(len(self.volume_results), 1),
                'tokens_per_second': total_tokens / max(total_time, 1),
                'cost_per_1000_characters': (total_cost / max(total_characters_processed, 1)) * 1000
            },
            'content_analysis_summary': {
                'unique_characters_found': len(all_characters),
                'characters_list': sorted(list(all_characters)),
                'unique_themes_found': len(all_themes),
                'themes_list': sorted(list(all_themes)),
                'total_events_extracted': len(all_events),
                'average_events_per_volume': len(all_events) / max(len(self.volume_results), 1)
            },
            'volume_details': [
                {
                    'volume_id': vol.volume_id,
                    'volume_title': vol.volume_title,
                    'processing_time_minutes': vol.total_processing_time / 60,
                    'total_chunks': vol.total_chunks,
                    'chapters_analyzed': len(vol.chapter_analyses),
                    'tokens_used': vol.total_tokens,
                    'cost_usd': vol.total_cost,
                    'main_characters_count': len(vol.main_characters),
                    'major_events_count': len(vol.major_events),
                    'themes_count': len(vol.themes),
                    'timeline_events_count': len(vol.timeline_events)
                }
                for vol in self.volume_results
            ],
            'database_info': {
                'database_path': self.db_path,
                'volumes_stored': len(self.volume_results),
                'chapters_stored': total_chapters_analyzed,
                'storage_timestamp': end_time.isoformat()
            },
            'success': True,
            'message': f"Successfully processed {len(self.volume_results)} volumes with {total_chapters_analyzed} chapters analyzed"
        }
        
        return report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final processing summary"""
        
        print("\n" + "=" * 100)
        print("🎉 LIMITLESS 5-VOLUME PROCESSING COMPLETED")
        print("=" * 100)
        
        meta = report['processing_metadata']
        print(f"\n⏱️  PROCESSING SUMMARY:")
        print(f"   🕐 Total time: {meta['total_processing_time_hours']:.2f} hours ({meta['total_processing_time_minutes']:.1f} minutes)")
        print(f"   📚 Volumes completed: {meta['volumes_completed']}")
        print(f"   📄 Chapters analyzed: {meta['total_chapters_analyzed']}/{meta['total_chunks_in_volumes']}")
        print(f"   📝 Characters processed: {meta['total_characters_processed']:,}")
        print(f"   ✅ Completion rate: {meta['completion_rate']}")
        
        cost = report['cost_and_token_analysis']
        print(f"\n💰 COST ANALYSIS:")
        print(f"   🔢 Total tokens: {cost['total_tokens_used']:,}")
        print(f"   💵 Total cost: ${cost['total_cost_usd']:.4f}")
        print(f"   📞 API calls: {cost['total_deepseek_calls']:,}")
        print(f"   💳 Avg cost/volume: ${cost['average_cost_per_volume']:.4f}")
        
        content = report['content_analysis_summary']
        print(f"\n📖 CONTENT ANALYSIS:")
        print(f"   👥 Unique characters: {content['unique_characters_found']}")
        print(f"   🎭 Unique themes: {content['unique_themes_found']}")
        print(f"   📅 Total events: {content['total_events_extracted']}")
        
        db_info = report['database_info']
        print(f"\n💾 DATABASE STORAGE:")
        print(f"   📁 Database: {db_info['database_path']}")
        print(f"   📚 Volumes stored: {db_info['volumes_stored']}")
        print(f"   📄 Chapters stored: {db_info['chapters_stored']}")
        
        print(f"\n🏁 STATUS: ✅ {report['message']}")
        print("=" * 100)

async def main():
    """Main execution function"""
    
    processor = LimitlessVolumeProcessor()
    
    try:
        # Process all 5 volumes without limitations
        logger.info("Starting limitless 5-volume processing...")
        report = await processor.process_limitless_5_volumes()
        
        # Print final summary
        processor.print_final_summary(report)
        
        # Save report to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"limitless_5_volume_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"💾 Final report saved to: {report_file}")
        logger.info("🎉 LIMITLESS PROCESSING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"❌ Limitless processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if processor.deepseek_client.session:
            await processor.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())