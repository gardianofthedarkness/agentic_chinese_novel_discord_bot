#!/usr/bin/env python3
"""
Analyze the full content structure to find all 22 volumes
"""

import os
import sys
import re

# Setup UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from qdrant_client import QdrantClient
from hierarchical_chapter_parser import HierarchicalChapterParser

def analyze_full_content():
    """Analyze the complete content to find all volumes"""
    
    print("=" * 80)
    print("🔍 ANALYZING FULL CONTENT FOR ALL 22 VOLUMES")
    print("=" * 80)
    
    try:
        # Connect to Qdrant and get ALL data
        client = QdrantClient(url="http://localhost:32768", verify=False)
        
        # Get all points (increase limit to get everything)
        print("📖 Fetching ALL content from Qdrant...")
        
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
            
            print(f"   Retrieved batch: {len(batch_points)} points (total: {len(all_points)})")
            
            # Check if we got a next offset
            if len(batch_points) < batch_size:
                break
            
            offset = points[1] if len(points) > 1 and points[1] else None
            if not offset:
                break
        
        print(f"✅ Total points retrieved: {len(all_points)}")
        
        # Combine all content
        print("📝 Combining all content...")
        combined_content = ""
        for point in all_points:
            if 'chunk' in point.payload:
                combined_content += point.payload['chunk'] + "\n\n"
        
        print(f"📊 Total content length: {len(combined_content):,} characters")
        
        # Look for volume patterns in the content
        print("\n🔍 Searching for volume patterns...")
        
        # Various patterns for volumes
        volume_patterns = [
            r'第([一二三四五六七八九十\d]+)卷',  # 第X卷
            r'魔法禁书目录\s*(\d+)',  # 魔法禁书目录 X
            r'卷\s*([一二三四五六七八九十\d]+)',  # 卷 X
            r'第\s*(\d+)\s*卷',  # 第 X 卷
            r'Volume\s*(\d+)',  # Volume X
        ]
        
        all_volumes_found = set()
        
        for pattern in volume_patterns:
            matches = re.findall(pattern, combined_content, re.IGNORECASE)
            for match in matches:
                # Convert Chinese numerals to numbers if needed
                if match in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
                    chinese_to_num = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
                    vol_num = chinese_to_num.get(match, match)
                else:
                    try:
                        vol_num = int(match)
                    except:
                        vol_num = match
                
                all_volumes_found.add(vol_num)
        
        print(f"📚 Volume patterns found: {list(all_volumes_found)}")
        
        # Try hierarchical parser on full content
        print("\n🔧 Running hierarchical parser on complete content...")
        parser = HierarchicalChapterParser()
        chapters = parser.parse_content_hierarchy(combined_content)
        
        print(f"✅ Hierarchical parser results:")
        print(f"   Volumes parsed: {len(parser.parsed_structure)}")
        print(f"   Chapters parsed: {len(chapters)}")
        
        # Show detailed volume structure
        print(f"\n📖 DETAILED VOLUME STRUCTURE:")
        for vol_id in sorted(parser.parsed_structure.keys()):
            volume = parser.parsed_structure[vol_id]
            print(f"\n📕 Volume {vol_id}: {volume.volume_title}")
            print(f"   📄 Chapters: {len(volume.chapters)}")
            print(f"   📝 Total words: {sum(ch.word_count for ch in volume.chapters):,}")
            
            # Show first few chapters
            for i, chapter in enumerate(volume.chapters[:5]):
                print(f"      {chapter.hierarchy_id}: {chapter.chapter_title[:50]}...")
            
            if len(volume.chapters) > 5:
                print(f"      ... and {len(volume.chapters) - 5} more chapters")
        
        # Look for specific content that might indicate more volumes
        print(f"\n🔍 CONTENT ANALYSIS FOR MISSING VOLUMES:")
        
        # Split content by obvious volume separators
        volume_separators = [
            '魔法禁书目录',
            '≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡',
            '第一卷', '第二卷', '第三卷', '第四卷', '第五卷'
        ]
        
        content_sections = []
        current_section = ""
        
        lines = combined_content.split('\n')
        for line in lines:
            if any(sep in line for sep in volume_separators):
                if current_section.strip():
                    content_sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section.strip():
            content_sections.append(current_section.strip())
        
        print(f"📑 Content sections identified: {len(content_sections)}")
        
        # Check each section for volume indicators
        for i, section in enumerate(content_sections[:10]):  # Show first 10
            print(f"\n📄 Section {i+1} (length: {len(section)} chars):")
            lines = section.split('\n')[:5]  # First 5 lines
            for line in lines:
                if line.strip():
                    print(f"   {line.strip()[:100]}")
        
        return {
            'total_points': len(all_points),
            'total_content_length': len(combined_content),
            'volumes_in_patterns': all_volumes_found,
            'parsed_volumes': len(parser.parsed_structure),
            'parsed_chapters': len(chapters),
            'content_sections': len(content_sections)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing content: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = analyze_full_content()
    
    if result:
        print(f"\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"📖 Total data points: {result['total_points']}")
        print(f"📝 Total content: {result['total_content_length']:,} characters")
        print(f"🔍 Volume patterns found: {len(result['volumes_in_patterns'])}")
        print(f"🔧 Parsed volumes: {result['parsed_volumes']}")
        print(f"📄 Parsed chapters: {result['parsed_chapters']}")
        print(f"📑 Content sections: {result['content_sections']}")