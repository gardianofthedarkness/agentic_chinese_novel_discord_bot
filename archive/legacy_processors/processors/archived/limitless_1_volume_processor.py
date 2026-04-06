#!/usr/bin/env python3
"""
Limitless 1-Volume Processor - Test Version with Progress Printing
Processes volume 1 completely with detailed progress updates
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

# Configure logging with console output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('limitless_1_volume.log', encoding='utf-8'),
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

class Limitless1VolumeProcessor:
    """Limitless processor for volume 1 with detailed progress"""
    
    def __init__(self):
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Initialize database
        self.db_path = "limitless_1_volume_results.db"
        self._initialize_database()
        
        self.global_stats = {
            'start_time': datetime.now(),
            'total_tokens': 0,
            'total_cost': 0.0,
            'total_deepseek_calls': 0,
            'chunks_completed': 0,
            'chunks_failed': 0
        }
        
        print("=" * 100)
        print("🚀 LIMITLESS 1-VOLUME PROCESSOR - DETAILED PROGRESS")
        print("=" * 100)
        print(f"🕐 Start time: {self.global_stats['start_time']}")
        print("📖 Target: Complete processing of Volume 1")
        print("🎯 Analysis: Full character, storyline, timeline, thematic analysis")
        print("💾 Database: Results stored in limitless_1_volume_results.db")
        print("📊 Progress: Detailed progress updates every chunk")
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
        
        # Processing progress table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            chunk_id INTEGER,
            status TEXT,
            tokens_used INTEGER,
            processing_time REAL,
            cumulative_tokens INTEGER,
            cumulative_cost REAL,
            progress_percentage REAL,
            message TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def _log_progress(self, chunk_id: int, status: str, tokens_used: int = 0, 
                     processing_time: float = 0.0, message: str = ""):
        """Log progress to database and console"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO processing_progress 
            (timestamp, chunk_id, status, tokens_used, processing_time, 
             cumulative_tokens, cumulative_cost, progress_percentage, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                chunk_id,
                status,
                tokens_used,
                processing_time,
                self.global_stats['total_tokens'],
                self.global_stats['total_cost'],
                (self.global_stats['chunks_completed'] / max(chunk_id, 1)) * 100,
                message
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log progress: {e}")
    
    async def process_volume_1_limitless(self) -> Dict[str, Any]:
        """Process volume 1 without limitations with detailed progress"""
        
        try:
            # Step 1: Extract volume 1 content
            print("\n📖 STEP 1: EXTRACTING VOLUME 1 CONTENT")
            print("-" * 60)
            volume_data = await self._extract_volume_1_content()
            
            if not volume_data['success']:
                raise Exception(f"Volume extraction failed: {volume_data['error']}")
            
            volume_content = volume_data['volume_content']
            print(f"✅ Volume 1 content extracted:")
            print(f"   📄 Total chunks: {volume_content['total_chunks']}")
            print(f"   📝 Total characters: {volume_content['total_chars']:,}")
            print(f"   📖 Title: {volume_content['title']}")
            
            # Step 2: Process volume 1 completely
            print(f"\n📚 STEP 2: PROCESSING VOLUME 1 - ALL {volume_content['total_chunks']} CHUNKS")
            print("-" * 60)
            print("🔄 Starting chunk-by-chunk analysis...")
            print("💡 Each chunk will show: Progress | Tokens | Time | Characters | Events")
            print("-" * 60)
            
            result = await self._process_volume_1_with_progress(volume_content)
            
            # Step 3: Generate final report
            print(f"\n📊 STEP 3: GENERATING FINAL REPORT")
            print("-" * 60)
            final_report = self._generate_final_report(result)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Volume 1 processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _extract_volume_1_content(self) -> Dict[str, Any]:
        """Extract volume 1 content specifically"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            # Get all data points
            print("   📥 Fetching data from Qdrant...")
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
                print(f"      Retrieved {len(all_points)} points...")
                
                if len(batch_points) < batch_size:
                    break
                
                offset = points[1] if len(points) > 1 and points[1] else None
                if not offset:
                    break
            
            print(f"   ✅ Total points retrieved: {len(all_points)}")
            
            # Extract volume 1 specifically
            volume_1_content = {
                'title': '',
                'chunks': [],
                'total_chars': 0
            }
            
            volume_patterns = [r'魔法禁书目录\s*1', r'第1卷']
            header_patterns = [r'≡≡≡≡≡≡≡≡', r'作者：鎌池和马']
            
            for i, point in enumerate(all_points):
                content = point.payload.get('chunk', '')
                
                if not content.strip():
                    continue
                
                # Check if header
                is_header = any(re.search(pattern, content) for pattern in header_patterns)
                
                # Determine if volume 1
                is_volume_1 = False
                for pattern in volume_patterns:
                    if re.search(pattern, content):
                        is_volume_1 = True
                        break
                
                if not is_volume_1:
                    # Estimate by position (first ~118 content chunks should be volume 1)
                    estimated_volume = (i // (len(all_points) // 22)) + 1
                    if estimated_volume == 1:
                        is_volume_1 = True
                
                if is_volume_1:
                    if is_header and not volume_1_content['title']:
                        # Extract title
                        lines = content.split('\n')
                        for line in lines:
                            if '魔法禁书目录' in line and '1' in line:
                                volume_1_content['title'] = line.strip()
                                break
                    else:
                        # Add content chunk
                        volume_1_content['chunks'].append({
                            'content': content,
                            'length': len(content),
                            'index': len(volume_1_content['chunks'])
                        })
                        volume_1_content['total_chars'] += len(content)
                        
                        # Stop once we have enough chunks for volume 1 (~116 chunks)
                        if len(volume_1_content['chunks']) >= 120:
                            break
            
            if not volume_1_content['title']:
                volume_1_content['title'] = "魔法禁书目录 第一卷"
            
            result = {
                'title': volume_1_content['title'],
                'chunks': volume_1_content['chunks'],
                'total_chars': volume_1_content['total_chars'],
                'total_chunks': len(volume_1_content['chunks'])
            }
            
            print(f"   📚 Volume 1 organized: {result['total_chunks']} chunks, {result['total_chars']:,} chars")
            
            return {
                'success': True,
                'volume_content': result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _process_volume_1_with_progress(self, volume_data: Dict[str, Any]) -> VolumeAnalysis:
        """Process volume 1 with detailed progress updates"""
        
        start_time = datetime.now()
        
        print(f"📖 Volume 1: {volume_data['title']}")
        print(f"📊 Processing {volume_data['total_chunks']} chunks")
        print(f"📝 Total characters: {volume_data['total_chars']:,}")
        print(f"🕐 Start time: {start_time.strftime('%H:%M:%S')}")
        print()
        
        chapter_analyses = []
        total_tokens = 0
        total_cost = 0.0
        chunks = volume_data['chunks']
        
        # Process each chunk with detailed progress
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            chunk_num = chunk_idx + 1
            
            # Progress header
            progress_pct = (chunk_num / len(chunks)) * 100
            print(f"🔄 [{chunk_num:3d}/{len(chunks)}] ({progress_pct:5.1f}%) Processing chunk {chunk_num}...")
            
            try:
                # Analyze chunk
                analysis_result = await self._analyze_chunk_with_progress(
                    1, chunk, chunk_num, len(chunks)
                )
                
                chunk_processing_time = time.time() - chunk_start_time
                
                if analysis_result['success']:
                    chunk_analysis = ChapterAnalysis(
                        volume_id=1,
                        chunk_id=chunk_idx,
                        characters=analysis_result['analysis'].get('characters', []),
                        events=analysis_result['analysis'].get('key_events', []),
                        themes=analysis_result['analysis'].get('themes', []),
                        emotions=analysis_result['analysis'].get('emotions', []),
                        summary=analysis_result['analysis'].get('summary', ''),
                        content_preview=chunk['content'][:100],
                        analysis_timestamp=datetime.now().isoformat(),
                        tokens_used=analysis_result['tokens_used'],
                        processing_time=chunk_processing_time
                    )
                    
                    chapter_analyses.append(chunk_analysis)
                    total_tokens += analysis_result['tokens_used']
                    total_cost += analysis_result['cost']
                    
                    # Update global stats
                    self.global_stats['total_tokens'] += analysis_result['tokens_used']
                    self.global_stats['total_cost'] += analysis_result['cost']
                    self.global_stats['total_deepseek_calls'] += 1
                    self.global_stats['chunks_completed'] += 1
                    
                    # Progress output
                    chars_info = f"{chunk_analysis.characters[:3]}" if chunk_analysis.characters else "[]"
                    events_info = f"{len(chunk_analysis.events)} events" if chunk_analysis.events else "0 events"
                    
                    print(f"   ✅ Success | {analysis_result['tokens_used']:4d} tokens | {chunk_processing_time:5.1f}s | Chars: {chars_info} | {events_info}")
                    print(f"      💰 Running total: {self.global_stats['total_tokens']:,} tokens, ${self.global_stats['total_cost']:.4f}")
                    
                    # Log progress to database
                    self._log_progress(chunk_num, "SUCCESS", analysis_result['tokens_used'], 
                                     chunk_processing_time, f"Characters: {len(chunk_analysis.characters)}, Events: {len(chunk_analysis.events)}")
                
                else:
                    self.global_stats['chunks_failed'] += 1
                    print(f"   ❌ Failed | Error: {analysis_result['error']}")
                    self._log_progress(chunk_num, "FAILED", 0, chunk_processing_time, analysis_result['error'])
                
                # Estimated time remaining
                if chunk_num % 10 == 0 or chunk_num <= 5:
                    elapsed_total = (datetime.now() - start_time).total_seconds()
                    avg_time_per_chunk = elapsed_total / chunk_num
                    remaining_chunks = len(chunks) - chunk_num
                    estimated_remaining = remaining_chunks * avg_time_per_chunk
                    
                    print(f"   📊 Progress: {chunk_num}/{len(chunks)} completed")
                    print(f"   ⏱️  Average: {avg_time_per_chunk:.1f}s/chunk | ETA: {estimated_remaining/60:.1f} minutes")
                    print()
                
                # Brief delay to be respectful to API
                await asyncio.sleep(0.3)
                
            except Exception as e:
                chunk_processing_time = time.time() - chunk_start_time
                self.global_stats['chunks_failed'] += 1
                print(f"   ❌ Exception | Error: {str(e)}")
                self._log_progress(chunk_num, "ERROR", 0, chunk_processing_time, str(e))
                continue
        
        # Generate volume summary
        print(f"\n🔄 Generating comprehensive volume summary...")
        summary_start_time = time.time()
        
        try:
            summary_result = await self._generate_volume_summary_with_progress(
                1, volume_data, chapter_analyses
            )
            
            summary_time = time.time() - summary_start_time
            
            if summary_result['success']:
                total_tokens += summary_result['tokens_used']
                total_cost += summary_result['cost']
                volume_summary_data = summary_result['summary']
                print(f"   ✅ Volume summary complete | {summary_result['tokens_used']} tokens | {summary_time:.1f}s")
            else:
                volume_summary_data = {}
                print(f"   ❌ Volume summary failed: {summary_result['error']}")
                
        except Exception as e:
            volume_summary_data = {}
            print(f"   ❌ Volume summary error: {e}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create volume analysis result
        volume_analysis = VolumeAnalysis(
            volume_id=1,
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
        
        # Save to database
        self._save_volume_to_database(volume_analysis)
        
        return volume_analysis
    
    async def _analyze_chunk_with_progress(self, volume_id: int, chunk: Dict[str, Any],
                                         chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Analyze chunk with progress tracking"""
        
        prompt = f"""
你是专业的中文小说分析专家。请对以下片段进行深入分析：

第{volume_id}卷，片段 {chunk_index}/{total_chunks}

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
  "summary": "片段核心内容总结（30字以内）"
}}

请仔细分析以下内容：
{chunk['content']}
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=800,
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
    
    async def _generate_volume_summary_with_progress(self, volume_id: int, volume_data: Dict[str, Any],
                                                   chapter_analyses: List[ChapterAnalysis]) -> Dict[str, Any]:
        """Generate volume summary with progress updates"""
        
        # Aggregate data from chapter analyses
        all_characters = set()
        all_events = []
        all_themes = set()
        
        for analysis in chapter_analyses:
            all_characters.update(analysis.characters)
            all_events.extend(analysis.events)
            all_themes.update(analysis.themes)
        
        print(f"   📊 Aggregated data: {len(all_characters)} unique characters, {len(all_events)} events, {len(all_themes)} themes")
        
        # Create summary for prompt
        analyses_summary = {
            'total_chunks_analyzed': len(chapter_analyses),
            'unique_characters': list(all_characters)[:15],  # Limit for prompt size
            'major_events': all_events[:20],  # Limit for prompt size
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
  "plot_summary": "整卷情节概述（150字以内）",
  "themes": ["主要主题"],
  "character_development": "角色发展分析",
  "volume_significance": "本卷在整个系列中的重要性",
  "timeline_events": [
    {{
      "event": "重要事件",
      "characters": ["参与角色"],
      "significance": "重要性描述"
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
            
            print(f"💾 Volume 1 analysis saved to database")
            
        except Exception as e:
            logger.error(f"❌ Failed to save volume to database: {e}")
    
    def _generate_final_report(self, volume_analysis: VolumeAnalysis) -> Dict[str, Any]:
        """Generate final comprehensive report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.global_stats['start_time']).total_seconds()
        
        # Extract unique data
        all_characters = set()
        all_themes = set()
        all_events = []
        
        all_characters.update(volume_analysis.main_characters)
        all_themes.update(volume_analysis.themes)
        all_events.extend(volume_analysis.major_events)
        
        for chapter in volume_analysis.chapter_analyses:
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
                'chunks_completed': self.global_stats['chunks_completed'],
                'chunks_failed': self.global_stats['chunks_failed'],
                'success_rate': f"{self.global_stats['chunks_completed']/max(volume_analysis.total_chunks, 1)*100:.1f}%"
            },
            'cost_and_token_analysis': {
                'total_tokens_used': volume_analysis.total_tokens,
                'total_cost_usd': volume_analysis.total_cost,
                'total_deepseek_calls': self.global_stats['total_deepseek_calls'],
                'average_tokens_per_chunk': volume_analysis.total_tokens // max(len(volume_analysis.chapter_analyses), 1),
                'tokens_per_second': volume_analysis.total_tokens / max(total_time, 1),
                'cost_per_1000_characters': (volume_analysis.total_cost / max(volume_analysis.total_characters, 1)) * 1000
            },
            'content_analysis_summary': {
                'unique_characters_found': len(all_characters),
                'characters_list': sorted(list(all_characters)),
                'unique_themes_found': len(all_themes),
                'themes_list': sorted(list(all_themes)),
                'total_events_extracted': len(all_events)
            },
            'volume_analysis': {
                'volume_id': volume_analysis.volume_id,
                'volume_title': volume_analysis.volume_title,
                'total_chunks': volume_analysis.total_chunks,
                'chunks_analyzed': len(volume_analysis.chapter_analyses),
                'total_characters': volume_analysis.total_characters,
                'processing_time_minutes': volume_analysis.total_processing_time / 60,
                'main_characters': volume_analysis.main_characters,
                'major_events': volume_analysis.major_events,
                'themes': volume_analysis.themes,
                'plot_summary': volume_analysis.plot_summary,
                'timeline_events': volume_analysis.timeline_events
            },
            'database_info': {
                'database_path': self.db_path,
                'chapters_stored': len(volume_analysis.chapter_analyses),
                'storage_timestamp': end_time.isoformat()
            },
            'success': True,
            'message': f"Successfully processed Volume 1 with {len(volume_analysis.chapter_analyses)} chapters analyzed"
        }
        
        return report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final processing summary with detailed results"""
        
        print("\n" + "=" * 100)
        print("🎉 VOLUME 1 LIMITLESS PROCESSING COMPLETED")
        print("=" * 100)
        
        meta = report['processing_metadata']
        print(f"\n⏱️  PROCESSING SUMMARY:")
        print(f"   🕐 Total time: {meta['total_processing_time_hours']:.2f} hours ({meta['total_processing_time_minutes']:.1f} minutes)")
        print(f"   📄 Chunks completed: {meta['chunks_completed']}")
        print(f"   ❌ Chunks failed: {meta['chunks_failed']}")
        print(f"   ✅ Success rate: {meta['success_rate']}")
        
        cost = report['cost_and_token_analysis']
        print(f"\n💰 COST ANALYSIS:")
        print(f"   🔢 Total tokens: {cost['total_tokens_used']:,}")
        print(f"   💵 Total cost: ${cost['total_cost_usd']:.4f}")
        print(f"   📞 API calls: {cost['total_deepseek_calls']:,}")
        print(f"   📊 Avg tokens/chunk: {cost['average_tokens_per_chunk']}")
        print(f"   ⚡ Tokens/second: {cost['tokens_per_second']:.1f}")
        
        content = report['content_analysis_summary']
        print(f"\n📖 CONTENT ANALYSIS:")
        print(f"   👥 Unique characters: {content['unique_characters_found']}")
        if content['characters_list']:
            print(f"      {', '.join(content['characters_list'][:10])}{'...' if len(content['characters_list']) > 10 else ''}")
        print(f"   🎭 Unique themes: {content['unique_themes_found']}")
        if content['themes_list']:
            print(f"      {', '.join(content['themes_list'])}")
        print(f"   📅 Total events: {content['total_events_extracted']}")
        
        volume = report['volume_analysis']
        print(f"\n📚 VOLUME 1 ANALYSIS:")
        print(f"   📖 Title: {volume['volume_title']}")
        print(f"   📄 Chunks analyzed: {volume['chunks_analyzed']}/{volume['total_chunks']}")
        print(f"   📝 Characters: {volume['total_characters']:,}")
        print(f"   👥 Main characters: {len(volume['main_characters'])}")
        print(f"   📅 Timeline events: {len(volume['timeline_events'])}")
        print(f"   🎭 Themes: {len(volume['themes'])}")
        
        if volume['plot_summary']:
            print(f"\n📖 PLOT SUMMARY:")
            print(f"   {volume['plot_summary']}")
        
        db_info = report['database_info']
        print(f"\n💾 DATABASE STORAGE:")
        print(f"   📁 Database: {db_info['database_path']}")
        print(f"   📄 Chapters stored: {db_info['chapters_stored']}")
        
        print(f"\n🏁 STATUS: ✅ {report['message']}")
        print("=" * 100)

async def main():
    """Main execution function"""
    
    processor = Limitless1VolumeProcessor()
    
    try:
        # Process volume 1 without limitations
        logger.info("Starting limitless Volume 1 processing...")
        report = await processor.process_volume_1_limitless()
        
        # Print final summary
        processor.print_final_summary(report)
        
        # Save report to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"limitless_volume_1_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"💾 Final report saved to: {report_file}")
        logger.info("🎉 VOLUME 1 PROCESSING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"❌ Volume 1 processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if processor.deepseek_client.session:
            await processor.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())