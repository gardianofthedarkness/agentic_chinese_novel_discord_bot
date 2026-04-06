#!/usr/bin/env python3
"""
Analyze Chapter Structure in Qdrant to understand big vs small chapters
"""

import asyncio
import re
from collections import defaultdict
from qdrant_client import QdrantClient

async def analyze_chapter_structure():
    """Analyze the chapter structure in Qdrant"""
    
    print("Analyzing Chapter Structure in Qdrant")
    print("="*50)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(url="http://localhost:32768", verify=False)
        
        # Get all points
        print("Fetching all points from Qdrant...")
        points = client.scroll(
            collection_name="test_novel2",
            limit=100,  # Get more points to see structure
            with_payload=True,
            with_vectors=False
        )
        
        print(f"Retrieved {len(points[0])} points")
        
        # Analyze chapter structure
        chapter_info = defaultdict(list)
        big_chapter_patterns = []
        
        for i, point in enumerate(points[0]):
            payload = point.payload
            content = payload.get('chunk', '')
            
            print(f"\n--- Point {i+1} (ID: {point.id}) ---")
            print(f"Content length: {len(content)}")
            print(f"First 300 chars: {content[:300]}...")
            
            # Look for chapter markers
            chapter_markers = []
            
            # Big chapter patterns (Chinese)
            big_patterns = [
                r'第[一二三四五六七八九十]+章',  # 第一章, 第二章, etc.
                r'第\d+章',  # 第1章, 第2章, etc.
                r'卷[一二三四五六七八九十]+',  # 卷一, 卷二, etc.
                r'第[一二三四五六七八九十]+卷'   # 第一卷, 第二卷, etc.
            ]
            
            # Small chapter patterns
            small_patterns = [
                r'第[一二三四五六七八九十]+节',  # 第一节, 第二节, etc.
                r'第\d+节',  # 第1节, 第2节, etc.
                r'[一二三四五六七八九十]+、',    # 一、二、三、etc.
                r'\d+\.',   # 1. 2. 3. etc.
            ]
            
            # Check for big chapter markers
            for pattern in big_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    chapter_markers.extend([('BIG', match) for match in matches])
                    
            # Check for small chapter markers  
            for pattern in small_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    chapter_markers.extend([('SMALL', match) for match in matches])
            
            # Look for title patterns
            title_patterns = [
                r'第.*章.*\n.*\n.*',  # Chapter title format
                r'[一二三四五六七八九十]+、.*',  # Numbered section
            ]
            
            for pattern in title_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    chapter_markers.extend([('TITLE', match) for match in matches])
            
            if chapter_markers:
                print(f"Chapter markers found: {chapter_markers}")
                chapter_info[i] = {
                    'point_id': point.id,
                    'content_length': len(content),
                    'markers': chapter_markers,
                    'content_preview': content[:200]
                }
            else:
                print("No chapter markers found")
        
        # Analyze structure
        print(f"\n{'='*60}")
        print("CHAPTER STRUCTURE ANALYSIS")
        print("="*60)
        
        big_chapters = []
        small_chapters = []
        
        for point_idx, info in chapter_info.items():
            for marker_type, marker in info['markers']:
                if marker_type == 'BIG':
                    big_chapters.append((point_idx, marker, info))
                elif marker_type == 'SMALL':
                    small_chapters.append((point_idx, marker, info))
        
        print(f"\nBIG CHAPTERS FOUND: {len(big_chapters)}")
        for point_idx, marker, info in big_chapters:
            print(f"  Point {point_idx}: '{marker}' (ID: {info['point_id']})")
            print(f"    Length: {info['content_length']} chars")
            print(f"    Preview: {info['content_preview'][:100]}...")
        
        print(f"\nSMALL CHAPTERS FOUND: {len(small_chapters)}")
        for point_idx, marker, info in small_chapters:
            print(f"  Point {point_idx}: '{marker}' (ID: {info['point_id']})")
            print(f"    Length: {info['content_length']} chars")
        
        # Try to identify the structure pattern
        print(f"\nSTRUCTURE PATTERN ANALYSIS:")
        print(f"- Total points analyzed: {len(points[0])}")
        print(f"- Points with chapter markers: {len(chapter_info)}")
        print(f"- Big chapter markers: {len(big_chapters)}")
        print(f"- Small chapter markers: {len(small_chapters)}")
        
        # Look for coordinate patterns
        print(f"\nCOORDINATE ANALYSIS:")
        coordinates = []
        for point in points[0]:
            if 'coordinate' in point.payload:
                coord = point.payload['coordinate']
                coordinates.append(coord)
        
        unique_coords = list(set(tuple(c) for c in coordinates))
        print(f"Unique coordinates found: {len(unique_coords)}")
        for coord in sorted(unique_coords)[:10]:  # Show first 10
            print(f"  {coord}")
        
        if coordinates:
            print(f"\nCoordinate ranges:")
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]
            print(f"  X: {min(x_coords)} to {max(x_coords)}")
            print(f"  Y: {min(y_coords)} to {max(y_coords)}")
        
    except Exception as e:
        print(f"Error analyzing structure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_chapter_structure())