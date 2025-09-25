#!/usr/bin/env python3
"""
Hierarchical Chapter Parser for Chinese Novels
Properly distinguishes between 卷 (volumes/big chapters) and 章 (small chapters)

This implementation addresses the gap in existing parsing logic by:
1. Recognizing 卷 (volume) vs 章 (chapter) hierarchy
2. Creating proper parent-child relationships
3. Supporting both traditional Chinese numerals and Arabic numbers
4. Handling mixed numbering systems
"""

import re
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ChapterNode:
    """Represents a single chapter with hierarchical information"""
    volume_id: Optional[int]           # 卷 number (None if not in a volume)
    volume_title: Optional[str]        # 卷 title
    chapter_id: int                    # 章 number within volume
    chapter_title: str                 # 章 title
    global_chapter_id: int             # Global sequential chapter number
    content: str                       # Chapter content
    word_count: int                    # Character count
    level: str                         # 'volume', 'chapter', 'section'
    parent_id: Optional[str]           # Parent chapter ID for hierarchy
    metadata: Dict                     # Additional metadata
    
    @property
    def hierarchy_id(self) -> str:
        """Generate hierarchical ID like '1.3' for 第一卷第三章"""
        if self.volume_id:
            return f"{self.volume_id}.{self.chapter_id}"
        else:
            return str(self.chapter_id)
    
    @property
    def display_title(self) -> str:
        """Generate display title with hierarchy"""
        if self.volume_id and self.volume_title:
            return f"{self.volume_title} - {self.chapter_title}"
        return self.chapter_title

@dataclass
class VolumeStructure:
    """Represents a complete volume with its chapters"""
    volume_id: int
    volume_title: str
    chapters: List[ChapterNode]
    total_words: int
    themes: List[str]
    character_introductions: List[str]
    
    @property
    def chapter_count(self) -> int:
        return len(self.chapters)

class HierarchicalChapterParser:
    """Advanced parser for Chinese novel hierarchical structure"""
    
    def __init__(self):
        # Chinese numeral mapping
        self.chinese_numerals = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
            '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20
        }
        
        # Volume patterns (卷 - big chapters)
        self.volume_patterns = [
            r'第([一二三四五六七八九十]+)卷[：:\s]*([^\n]*)',      # 第一卷：标题
            r'第(\d+)卷[：:\s]*([^\n]*)',                        # 第1卷：标题
            r'卷([一二三四五六七八九十]+)[：:\s]*([^\n]*)',        # 卷一：标题
            r'卷(\d+)[：:\s]*([^\n]*)',                          # 卷1：标题
            r'第([一二三四五六七八九十]+)部[：:\s]*([^\n]*)',      # 第一部：标题
            r'第(\d+)部[：:\s]*([^\n]*)'                         # 第1部：标题
        ]
        
        # Chapter patterns (章 - small chapters within volumes)
        self.chapter_patterns = [
            r'第([一二三四五六七八九十]+)章[：:\s]*([^\n]*)',      # 第一章：标题
            r'第(\d+)章[：:\s]*([^\n]*)',                        # 第1章：标题
            r'章([一二三四五六七八九十]+)[：:\s]*([^\n]*)',        # 章一：标题
            r'章(\d+)[：:\s]*([^\n]*)'                           # 章1：标题
        ]
        
        # Section patterns (节 - subsections)
        self.section_patterns = [
            r'第([一二三四五六七八九十]+)节[：:\s]*([^\n]*)',      # 第一节：标题
            r'第(\d+)节[：:\s]*([^\n]*)',                        # 第1节：标题
            r'([一二三四五六七八九十]+)、([^\n]*)',               # 一、标题
            r'(\d+)\.([^\n]*)'                                   # 1.标题
        ]
        
        self.parsed_structure: Dict[int, VolumeStructure] = {}
        self.flat_chapters: List[ChapterNode] = []
        self.global_chapter_counter = 0
        
    def convert_chinese_numeral(self, chinese_num: str) -> int:
        """Convert Chinese numeral to Arabic number"""
        if chinese_num in self.chinese_numerals:
            return self.chinese_numerals[chinese_num]
        elif chinese_num.isdigit():
            return int(chinese_num)
        else:
            # Handle complex Chinese numerals like 二十一
            if '十' in chinese_num and len(chinese_num) > 1:
                if chinese_num.startswith('十'):
                    return 10 + self.chinese_numerals.get(chinese_num[1:], 0)
                elif chinese_num.endswith('十'):
                    return self.chinese_numerals.get(chinese_num[:-1], 0) * 10
                else:
                    parts = chinese_num.split('十')
                    if len(parts) == 2:
                        tens = self.chinese_numerals.get(parts[0], 0) * 10
                        ones = self.chinese_numerals.get(parts[1], 0)
                        return tens + ones
            return 0
    
    def parse_content_hierarchy(self, content: str) -> List[ChapterNode]:
        """Parse content and extract hierarchical chapter structure"""
        logger.info("Starting hierarchical chapter parsing...")
        
        # Split content into potential chapters
        chapter_candidates = []
        current_volume = None
        current_volume_title = None
        
        # First pass: Identify volumes and chapters
        lines = content.split('\n')
        current_content = []
        current_chapter_info = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                current_content.append(line)
                continue
            
            # Check for volume markers
            volume_match = None
            for pattern in self.volume_patterns:
                match = re.search(pattern, line)
                if match:
                    volume_match = match
                    break
            
            if volume_match:
                # Save previous chapter if exists
                if current_chapter_info:
                    chapter_candidates.append({
                        **current_chapter_info,
                        'content': '\n'.join(current_content),
                        'word_count': len('\n'.join(current_content))
                    })
                
                # Start new volume
                volume_num = self.convert_chinese_numeral(volume_match.group(1))
                volume_title = volume_match.group(2).strip() or f"第{volume_num}卷"
                current_volume = volume_num
                current_volume_title = volume_title
                
                logger.info(f"Found volume: {volume_num} - {volume_title}")
                current_content = [line]
                current_chapter_info = None
                continue
            
            # Check for chapter markers
            chapter_match = None
            for pattern in self.chapter_patterns:
                match = re.search(pattern, line)
                if match:
                    chapter_match = match
                    break
            
            if chapter_match:
                # Save previous chapter if exists
                if current_chapter_info:
                    chapter_candidates.append({
                        **current_chapter_info,
                        'content': '\n'.join(current_content),
                        'word_count': len('\n'.join(current_content))
                    })
                
                # Start new chapter
                chapter_num = self.convert_chinese_numeral(chapter_match.group(1))
                chapter_title = chapter_match.group(2).strip() or f"第{chapter_num}章"
                self.global_chapter_counter += 1
                
                current_chapter_info = {
                    'volume_id': current_volume,
                    'volume_title': current_volume_title,
                    'chapter_id': chapter_num,
                    'chapter_title': chapter_title,
                    'global_chapter_id': self.global_chapter_counter,
                    'level': 'chapter',
                    'parent_id': f"vol_{current_volume}" if current_volume else None
                }
                
                logger.info(f"Found chapter: Vol.{current_volume} Ch.{chapter_num} - {chapter_title}")
                current_content = [line]
                continue
            
            # Add to current content
            current_content.append(line)
        
        # Don't forget the last chapter
        if current_chapter_info:
            chapter_candidates.append({
                **current_chapter_info,
                'content': '\n'.join(current_content),
                'word_count': len('\n'.join(current_content))
            })
        
        # Convert to ChapterNode objects
        chapter_nodes = []
        for candidate in chapter_candidates:
            if candidate.get('word_count', 0) > 100:  # Filter out too-short chapters
                node = ChapterNode(
                    volume_id=candidate.get('volume_id'),
                    volume_title=candidate.get('volume_title'),
                    chapter_id=candidate.get('chapter_id', 0),
                    chapter_title=candidate.get('chapter_title', ''),
                    global_chapter_id=candidate.get('global_chapter_id', 0),
                    content=candidate.get('content', ''),
                    word_count=candidate.get('word_count', 0),
                    level=candidate.get('level', 'chapter'),
                    parent_id=candidate.get('parent_id'),
                    metadata={'source_line': i}
                )
                chapter_nodes.append(node)
        
        self.flat_chapters = chapter_nodes
        self._build_volume_structure()
        
        logger.info(f"Parsed {len(chapter_nodes)} chapters across {len(self.parsed_structure)} volumes")
        return chapter_nodes
    
    def _build_volume_structure(self):
        """Build volume-based structure from flat chapters"""
        volumes = defaultdict(list)
        
        for chapter in self.flat_chapters:
            if chapter.volume_id:
                volumes[chapter.volume_id].append(chapter)
            else:
                # Handle chapters without explicit volume
                volumes[0].append(chapter)
        
        for vol_id, chapters in volumes.items():
            if vol_id == 0:
                volume_title = "散章" if chapters else "未分卷章节"
            else:
                volume_title = chapters[0].volume_title if chapters else f"第{vol_id}卷"
            
            volume_structure = VolumeStructure(
                volume_id=vol_id,
                volume_title=volume_title,
                chapters=sorted(chapters, key=lambda x: x.chapter_id),
                total_words=sum(ch.word_count for ch in chapters),
                themes=[],  # To be filled by analysis
                character_introductions=[]  # To be filled by analysis
            )
            
            self.parsed_structure[vol_id] = volume_structure
    
    def get_chapter_by_hierarchy(self, volume_id: int, chapter_id: int) -> Optional[ChapterNode]:
        """Get specific chapter by volume and chapter ID"""
        if volume_id in self.parsed_structure:
            volume = self.parsed_structure[volume_id]
            for chapter in volume.chapters:
                if chapter.chapter_id == chapter_id:
                    return chapter
        return None
    
    def get_volume_summary(self, volume_id: int) -> Optional[Dict]:
        """Get summary of a specific volume"""
        if volume_id not in self.parsed_structure:
            return None
        
        volume = self.parsed_structure[volume_id]
        return {
            'volume_id': volume.volume_id,
            'title': volume.volume_title,
            'chapter_count': volume.chapter_count,
            'total_words': volume.total_words,
            'chapters': [
                {
                    'chapter_id': ch.chapter_id,
                    'title': ch.chapter_title,
                    'hierarchy_id': ch.hierarchy_id,
                    'word_count': ch.word_count
                }
                for ch in volume.chapters
            ]
        }
    
    def get_full_structure_summary(self) -> Dict:
        """Get complete hierarchical structure summary"""
        summary = {
            'total_volumes': len(self.parsed_structure),
            'total_chapters': len(self.flat_chapters),
            'total_words': sum(ch.word_count for ch in self.flat_chapters),
            'volumes': {}
        }
        
        for vol_id in sorted(self.parsed_structure.keys()):
            volume_summary = self.get_volume_summary(vol_id)
            summary['volumes'][vol_id] = volume_summary
        
        return summary
    
    def print_structure_analysis(self):
        """Print detailed structure analysis"""
        print("\n" + "="*80)
        print("HIERARCHICAL CHAPTER STRUCTURE ANALYSIS")
        print("="*80)
        
        summary = self.get_full_structure_summary()
        
        print(f"\nOVERALL STRUCTURE:")
        print(f"  Total Volumes (卷): {summary['total_volumes']}")
        print(f"  Total Chapters (章): {summary['total_chapters']}")
        print(f"  Total Words: {summary['total_words']:,}")
        
        print(f"\nVOLUME BREAKDOWN:")
        for vol_id in sorted(summary['volumes'].keys()):
            vol_info = summary['volumes'][vol_id]
            print(f"\n  卷 {vol_id}: {vol_info['title']}")
            print(f"    Chapters: {vol_info['chapter_count']}")
            print(f"    Words: {vol_info['total_words']:,}")
            print(f"    Chapter List:")
            
            for ch_info in vol_info['chapters'][:5]:  # Show first 5 chapters
                print(f"      {ch_info['hierarchy_id']}: {ch_info['title']} ({ch_info['word_count']:,} words)")
            
            if vol_info['chapter_count'] > 5:
                print(f"      ... and {vol_info['chapter_count'] - 5} more chapters")

# Integration with existing agentic system
class HierarchicalAgenticProcessor:
    """Enhanced agentic processor with hierarchical chapter support"""
    
    def __init__(self, qdrant_url: str = "http://localhost:32768"):
        self.parser = HierarchicalChapterParser()
        self.qdrant_url = qdrant_url
        
    async def parse_novel_with_hierarchy(self, max_volumes: int = 3) -> Dict:
        """Parse novel content with proper volume/chapter hierarchy"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=self.qdrant_url, verify=False)
            
            # Get all content from Qdrant
            points = client.scroll(
                collection_name="test_novel2",
                limit=200,  # Get more points to capture full structure
                with_payload=True,
                with_vectors=False
            )
            
            # Combine all content for parsing
            combined_content = ""
            for point in points[0]:
                if 'chunk' in point.payload:
                    combined_content += point.payload['chunk'] + "\n\n"
            
            # Parse with hierarchical structure
            chapters = self.parser.parse_content_hierarchy(combined_content)
            
            # Filter to requested number of volumes
            filtered_structure = {}
            volume_count = 0
            
            for vol_id in sorted(self.parser.parsed_structure.keys()):
                if volume_count >= max_volumes:
                    break
                filtered_structure[vol_id] = self.parser.parsed_structure[vol_id]
                volume_count += 1
            
            return {
                'success': True,
                'structure': filtered_structure,
                'summary': self.parser.get_full_structure_summary(),
                'flat_chapters': chapters[:max_volumes * 10]  # Reasonable limit
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical parsing: {e}")
            return {
                'success': False,
                'error': str(e),
                'structure': {},
                'summary': {}
            }

# Testing and demonstration
async def test_hierarchical_parsing():
    """Test hierarchical parsing with sample content"""
    
    sample_content = """
第一卷：学园都市篇

第一章：超电磁炮

当麻站在学园都市的街道上，看着远处的摩天大楼。这里是世界上最先进的科学都市。

美琴从拐角处走了出来，手中闪烁着电光。"你就是那个叫上条当麻的人吗？"

第二章：幻想杀手

当麻举起右手，"幻想杀手"的力量开始显现。任何超能力都会被这只手无效化。

第三章：姐妹们的战斗

御坂妹妹们开始了她们的实验，而美琴试图阻止这一切。

第二卷：魔法侧篇

第一章：英国清教

茵蒂克丝出现了，带来了魔法世界的消息。

第二章：禁书目录

十万三千本魔道书的知识，全部储存在一个少女的脑海中。
"""
    
    parser = HierarchicalChapterParser()
    chapters = parser.parse_content_hierarchy(sample_content)
    
    parser.print_structure_analysis()
    
    print(f"\nCHAPTER DETAILS:")
    for chapter in chapters:
        print(f"\n{chapter.hierarchy_id}: {chapter.display_title}")
        print(f"  Content: {chapter.content[:100]}...")
        print(f"  Words: {chapter.word_count}")
        print(f"  Level: {chapter.level}")

if __name__ == "__main__":
    asyncio.run(test_hierarchical_parsing())