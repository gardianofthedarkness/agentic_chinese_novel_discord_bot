#!/usr/bin/env python3
"""
Fixed Comprehensive Analysis with Coordinate-Based Chapter Detection
Addresses the character detection issue that missed 御坂美琴
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

from qdrant_client import QdrantClient
from deepseek_integration import DeepSeekClient, create_deepseek_config
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FixedCharacterReference:
    identifier: str
    coordinate_positions: List[int]
    estimated_chapters: List[int]
    frequency: int
    confidence: float
    descriptions: List[str]
    actions: List[str]

class FixedComprehensiveAnalyzer:
    """Enhanced analysis using coordinate-based chapter detection"""
    
    def __init__(self):
        self.qdrant_client = QdrantClient(url="http://localhost:32768", verify=False)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        self.db_engine = create_engine('postgresql://admin:admin@localhost:5432/novel_sim')
        
        # Fixed chapter mapping based on coordinate analysis
        self.chapter_boundaries = [
            (0, 46),     # Chapter 1: coordinates 0-46
            (47, 74),    # Chapter 2: coordinates 47-74
            (75, 93),    # Chapter 3: coordinates 75-93
        ]
        
        # Analysis results
        self.characters_found = {}
        self.chapter_content = {}
        self.storyline_analysis = {}
        self.total_tokens_used = 0
        self.api_calls_made = 0
        
    def map_coordinate_to_chapter(self, coordinate_y: int) -> int:
        """Map coordinate to chapter number"""
        for chapter_num, (start, end) in enumerate(self.chapter_boundaries, 1):
            if start <= coordinate_y <= end:
                return chapter_num
        return 0  # Outside our processing range
    
    async def get_fixed_chapters(self, max_chapters: int = 3) -> Dict[int, List[Dict[str, Any]]]:
        """Get chapters using coordinate-based mapping"""
        logger.info(f"Getting first {max_chapters} chapters using fixed coordinate mapping...")
        
        points = self.qdrant_client.scroll(
            collection_name="test_novel2",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        chapters = defaultdict(list)
        
        for point in points[0]:
            content = point.payload.get('chunk', '')
            coord = point.payload.get('coordinate', [0, 0])
            coord_y = coord[1] if len(coord) > 1 else 0
            
            chapter_num = self.map_coordinate_to_chapter(coord_y)
            
            if 1 <= chapter_num <= max_chapters:
                chapters[chapter_num].append({
                    'point_id': point.id,
                    'content': content,
                    'coordinate': coord,
                    'coordinate_y': coord_y
                })
        
        # Sort within each chapter by coordinate
        for chapter_num in chapters:
            chapters[chapter_num].sort(key=lambda x: x['coordinate_y'])
        
        logger.info(f"Retrieved {len(chapters)} chapters:")
        for chap_num, points_list in chapters.items():
            total_chars = sum(len(p['content']) for p in points_list)
            coord_range = f"{min(p['coordinate_y'] for p in points_list)}-{max(p['coordinate_y'] for p in points_list)}"
            logger.info(f"  Chapter {chap_num}: {len(points_list)} sections, {total_chars} chars, coords {coord_range}")
        
        self.chapter_content = dict(chapters)
        return dict(chapters)
    
    async def extract_characters_from_chapters(self) -> Dict[str, FixedCharacterReference]:
        """Extract characters using improved detection"""
        logger.info("Extracting characters with fixed detection...")
        
        # Important character patterns
        character_patterns = [
            '御坂美琴', '御坂', '美琴', 'みさか', 'ミサカ',
            '上条当麻', '当麻', '上条',
            '茵蒂克丝', '禁书目录', 'インデックス',
            '一方通行', 'アクセラレータ',
            '白井黑子', '白井', '黑子',
            '食蜂操祈',
            '土御门元春',
            '神裂火织',
            '史提尔',
            '小萌老师', '小萌'
        ]
        
        character_tracking = defaultdict(lambda: {
            'coordinates': [],
            'chapters': set(),
            'frequency': 0,
            'content_samples': []
        })
        
        for chapter_num, chapter_points in self.chapter_content.items():
            logger.info(f"Processing Chapter {chapter_num} for characters...")
            
            for point_data in chapter_points:
                content = point_data['content']
                coord_y = point_data['coordinate_y']
                
                for char_pattern in character_patterns:
                    if char_pattern in content:
                        frequency = content.count(char_pattern)
                        character_tracking[char_pattern]['coordinates'].append(coord_y)
                        character_tracking[char_pattern]['chapters'].add(chapter_num)
                        character_tracking[char_pattern]['frequency'] += frequency
                        
                        # Store content sample for context analysis
                        if len(character_tracking[char_pattern]['content_samples']) < 3:
                            character_tracking[char_pattern]['content_samples'].append(content[:500])
        
        # Convert to FixedCharacterReference objects
        characters = {}
        for char_name, data in character_tracking.items():
            if data['frequency'] >= 2:  # Filter for meaningful appearances
                
                # Extract basic descriptions and actions from content
                descriptions = []
                actions = []
                for sample in data['content_samples']:
                    # Simple pattern matching for descriptions and actions
                    if '银发' in sample or '修女' in sample:
                        descriptions.append('silver_hair_nun')
                    if '电击' in sample or '超电磁砲' in sample:
                        descriptions.append('electromaster')
                    if '说' in sample or '道' in sample:
                        actions.append('speaking')
                
                characters[char_name] = FixedCharacterReference(
                    identifier=char_name,
                    coordinate_positions=sorted(data['coordinates']),
                    estimated_chapters=sorted(list(data['chapters'])),
                    frequency=data['frequency'],
                    confidence=min(0.3 + (data['frequency'] * 0.05), 0.95),
                    descriptions=list(set(descriptions)),
                    actions=list(set(actions))
                )
        
        self.characters_found = characters
        logger.info(f"Found {len(characters)} characters with meaningful appearances")
        
        return characters
    
    async def analyze_storylines(self):
        """Analyze storylines for each chapter"""
        logger.info("Analyzing storylines with DeepSeek...")
        
        for chapter_num, chapter_points in self.chapter_content.items():
            # Combine chapter content
            combined_content = ""
            for point_data in chapter_points:
                combined_content += point_data['content'] + "\\n\\n"
            
            try:
                await self.deepseek_client.initialize()
                self.api_calls_made += 1
                
                prompt = f"""分析第{chapter_num}章的主要情节：

章节内容长度：{len(combined_content)} 字符

请简要分析：
1. 主要事件
2. 角色互动
3. 重要对话或场景

内容片段：
{combined_content[:1500]}...

请用中文简要总结主要情节。"""
                
                messages = [{"role": "user", "content": prompt}]
                response = await self.deepseek_client.generate_character_response(
                    messages=messages,
                    max_tokens=800,
                    temperature=0.1
                )
                
                if response.get("success"):
                    self.storyline_analysis[chapter_num] = response["response"]
                    self.total_tokens_used += response.get("token_usage", {}).get("total", 0)
                    logger.info(f"Chapter {chapter_num} storyline analysis complete")
                else:
                    self.storyline_analysis[chapter_num] = "Analysis failed"
                    
            except Exception as e:
                logger.error(f"Error analyzing Chapter {chapter_num}: {e}")
                self.storyline_analysis[chapter_num] = f"Error: {e}"
    
    async def store_results_in_database(self):
        """Store analysis results in PostgreSQL"""
        logger.info("Storing results in database...")
        
        with self.db_engine.connect() as conn:
            # Store chapter summaries
            for chapter_num, analysis in self.storyline_analysis.items():
                try:
                    conn.execute(text("""
                        INSERT INTO chapter_summaries 
                        (chapter_id, chapter_index, chapter_title, content_summary, created_at)
                        VALUES (:chapter_id, :chapter_index, :chapter_title, :content_summary, :created_at)
                    """), {
                        'chapter_id': f'fixed_chapter_{chapter_num}',
                        'chapter_index': chapter_num,
                        'chapter_title': f'第{chapter_num}章 (Fixed Detection)',
                        'content_summary': analysis[:500],  # Truncate for storage
                        'created_at': datetime.now()
                    })
                except Exception as e:
                    logger.error(f"Error storing chapter {chapter_num}: {e}")
            
            # Store character profiles
            for char_name, char_data in self.characters_found.items():
                try:
                    conn.execute(text("""
                        INSERT INTO character_profiles 
                        (character_id, name, character_type, first_appearance_chapter, 
                         confidence_score, created_at)
                        VALUES (:character_id, :name, :character_type, :first_appearance_chapter,
                                :confidence_score, :created_at)
                    """), {
                        'character_id': f'{char_name}_fixed',
                        'name': char_name,
                        'character_type': 'detected_character',
                        'first_appearance_chapter': min(char_data.estimated_chapters),
                        'confidence_score': char_data.confidence,
                        'created_at': datetime.now()
                    })
                except Exception as e:
                    logger.error(f"Error storing character {char_name}: {e}")
            
            conn.commit()
            logger.info("Database storage complete")
    
    def print_comprehensive_results(self):
        """Print all analysis results"""
        print("\\n" + "="*80)
        print("FIXED COMPREHENSIVE ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\\nPROCESSING STATISTICS:")
        print(f"   API calls made: {self.api_calls_made}")
        print(f"   Total tokens used: {self.total_tokens_used}")
        print(f"   Chapters processed: {len(self.chapter_content)}")
        print(f"   Characters found: {len(self.characters_found)}")
        
        print(f"\\nCHARACTER DETECTION RESULTS:")
        print("-" * 50)
        
        # Sort characters by frequency
        sorted_chars = sorted(
            self.characters_found.items(),
            key=lambda x: x[1].frequency,
            reverse=True
        )
        
        for char_name, char_data in sorted_chars:
            print(f"\\n'{char_name}':")
            print(f"   Frequency: {char_data.frequency} mentions")
            print(f"   Chapters: {char_data.estimated_chapters}")
            print(f"   Coordinates: {min(char_data.coordinate_positions)}-{max(char_data.coordinate_positions)}")
            print(f"   Confidence: {char_data.confidence:.2f}")
            
            if '御坂' in char_name:
                print(f"   >>> MISAKA CHARACTER DETECTED! <<<")
                print(f"   >>> Previously missed by original script <<<")
        
        print(f"\\nSTORYLINE ANALYSIS:")
        print("-" * 50)
        for chapter_num, analysis in self.storyline_analysis.items():
            print(f"\\nChapter {chapter_num}:")
            print(f"   {analysis[:200]}...")
    
    async def cleanup(self):
        """Clean up connections"""
        if self.deepseek_client.session:
            await self.deepseek_client.close()

async def main():
    """Run fixed comprehensive analysis"""
    print("FIXED COMPREHENSIVE ANALYSIS")
    print("Using coordinate-based chapter detection")
    print("="*60)
    
    analyzer = FixedComprehensiveAnalyzer()
    
    try:
        # Get chapters using fixed detection
        chapters = await analyzer.get_fixed_chapters(3)
        
        if not chapters:
            print("No chapters found!")
            return
        
        # Extract characters
        await analyzer.extract_characters_from_chapters()
        
        # Analyze storylines
        await analyzer.analyze_storylines()
        
        # Store in database
        await analyzer.store_results_in_database()
        
        # Print results
        analyzer.print_comprehensive_results()
        
    finally:
        await analyzer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())