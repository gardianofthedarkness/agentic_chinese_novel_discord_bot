#!/usr/bin/env python3
"""
Analyze Actual Content Structure in Qdrant
Find where the real novel content is stored beyond just headers
"""

import os
import sys
import re
from collections import defaultdict

# Setup UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from qdrant_client import QdrantClient

def analyze_actual_content():
    """Analyze the actual content structure to find real novel data"""
    
    print("=" * 80)
    print("🔍 ANALYZING ACTUAL CONTENT STRUCTURE")
    print("=" * 80)
    print("📋 Goal: Find real novel content beyond headers")
    print("=" * 80)
    
    try:
        # Connect to Qdrant and get all data
        client = QdrantClient(url="http://localhost:32768", verify=False)
        
        print("📖 Fetching all data points...")
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
        
        print(f"✅ Total points retrieved: {len(all_points)}")
        
        # Analyze content patterns
        print("\n🔍 Analyzing content patterns...")
        
        content_types = {
            'headers': 0,       # Title pages
            'novel_content': 0, # Actual story content
            'chapter_starts': 0, # Chapter beginnings
            'empty': 0
        }
        
        volume_content = defaultdict(list)  # volume_id -> content chunks
        
        # Patterns to identify content types
        header_patterns = [
            r'魔法禁书目录\s*\d+',
            r'≡≡≡≡≡≡≡≡',
            r'作者：鎌池和马',
            r'插画：灰村清孝',
            r'录入：',
            r'校对：',
            r'EPUB制作：'
        ]
        
        chapter_patterns = [
            r'第[一二三四五六七八九十\d]+章',
            r'第[一二三四五六七八九十\d]+节',
            r'序章',
            r'尾声',
            r'后记'
        ]
        
        volume_assignment_patterns = [
            r'魔法禁书目录\s*(\d+)',  # Extract volume number
            r'第(\d+)卷',
            r'Volume\s*(\d+)'
        ]
        
        for i, point in enumerate(all_points):
            content = point.payload.get('chunk', '')
            
            if not content.strip():
                content_types['empty'] += 1
                continue
            
            # Check if it's a header
            is_header = any(re.search(pattern, content) for pattern in header_patterns)
            
            if is_header:
                content_types['headers'] += 1
                
                # Try to extract volume number from header
                volume_num = None
                for pattern in volume_assignment_patterns:
                    match = re.search(pattern, content)
                    if match:
                        volume_num = int(match.group(1))
                        break
                
                if volume_num:
                    volume_content[volume_num].append({
                        'type': 'header',
                        'content': content,
                        'point_index': i,
                        'length': len(content)
                    })
                
            else:
                # Check if it contains chapter markers
                has_chapter = any(re.search(pattern, content) for pattern in chapter_patterns)
                
                if has_chapter:
                    content_types['chapter_starts'] += 1
                else:
                    content_types['novel_content'] += 1
                
                # Try to assign to volume based on content or position
                # Since we don't have explicit volume markers, we need to infer
                
                # Method 1: Look for volume references in the content
                volume_num = None
                for pattern in volume_assignment_patterns:
                    match = re.search(pattern, content)
                    if match:
                        volume_num = int(match.group(1))
                        break
                
                # Method 2: If no direct reference, estimate based on position
                if not volume_num:
                    # Estimate based on position in the data
                    # Assuming roughly equal distribution
                    estimated_volume = (i // (len(all_points) // 22)) + 1
                    if estimated_volume <= 22:
                        volume_num = estimated_volume
                
                if volume_num and volume_num <= 22:
                    volume_content[volume_num].append({
                        'type': 'content',
                        'content': content,
                        'point_index': i,
                        'length': len(content),
                        'has_chapter': has_chapter
                    })
            
            # Progress update
            if (i + 1) % 500 == 0:
                print(f"   Processed {i + 1}/{len(all_points)} points...")
        
        # Analysis results
        print(f"\n📊 CONTENT TYPE ANALYSIS:")
        print(f"   📄 Headers: {content_types['headers']}")
        print(f"   📖 Novel content: {content_types['novel_content']}")
        print(f"   📚 Chapter starts: {content_types['chapter_starts']}")
        print(f"   🔲 Empty: {content_types['empty']}")
        
        # Volume distribution
        print(f"\n📚 VOLUME CONTENT DISTRIBUTION:")
        
        for vol_id in sorted(volume_content.keys())[:10]:  # Show first 10 volumes
            vol_data = volume_content[vol_id]
            total_chars = sum(chunk['length'] for chunk in vol_data)
            content_chunks = [chunk for chunk in vol_data if chunk['type'] == 'content']
            header_chunks = [chunk for chunk in vol_data if chunk['type'] == 'header']
            
            print(f"\n📕 Volume {vol_id}:")
            print(f"   📄 Total chunks: {len(vol_data)}")
            print(f"   📖 Content chunks: {len(content_chunks)}")
            print(f"   📋 Header chunks: {len(header_chunks)}")
            print(f"   📝 Total characters: {total_chars:,}")
            
            if content_chunks:
                print(f"   📊 Avg content chunk size: {sum(chunk['length'] for chunk in content_chunks) // len(content_chunks):,}")
                
                # Show sample content
                sample_content = content_chunks[0]['content'][:200]
                print(f"   🔍 Sample content: {sample_content}...")
        
        # Focus on first 5 volumes for processing
        print(f"\n🎯 FIRST 5 VOLUMES FOR PROCESSING:")
        
        first_5_stats = {}
        for vol_id in range(1, 6):
            if vol_id in volume_content:
                vol_data = volume_content[vol_id]
                content_chunks = [chunk for chunk in vol_data if chunk['type'] == 'content']
                total_content_chars = sum(chunk['length'] for chunk in content_chunks)
                
                first_5_stats[vol_id] = {
                    'total_chunks': len(content_chunks),
                    'total_chars': total_content_chars,
                    'content_chunks': content_chunks
                }
                
                print(f"\n📖 Volume {vol_id}:")
                print(f"   📄 Content chunks: {len(content_chunks)}")
                print(f"   📝 Content characters: {total_content_chars:,}")
                
                if content_chunks:
                    print(f"   📊 Avg chunk size: {total_content_chars // len(content_chunks):,}")
        
        # Calculate processing estimates
        total_content_chars = sum(stats['total_chars'] for stats in first_5_stats.values())
        total_chunks = sum(stats['total_chunks'] for stats in first_5_stats.values())
        
        print(f"\n💰 PROCESSING ESTIMATES FOR FIRST 5 VOLUMES:")
        print(f"   📄 Total content chunks: {total_chunks}")
        print(f"   📝 Total content characters: {total_content_chars:,}")
        print(f"   🔢 Estimated tokens: {total_content_chars // 4:,}")
        print(f"   💵 Estimated cost: ${(total_content_chars // 4) * 0.00002:.4f}")
        
        return {
            'content_types': content_types,
            'volume_content': dict(volume_content),
            'first_5_stats': first_5_stats,
            'total_points': len(all_points)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing content: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = analyze_actual_content()
    
    if result:
        print(f"\n✅ Content analysis complete!")
        print(f"   📄 Total points: {result['total_points']}")
        print(f"   📚 Volumes identified: {len(result['volume_content'])}")
        print(f"   📖 Content chunks: {result['content_types']['novel_content']}")
    else:
        print(f"❌ Failed to analyze content")