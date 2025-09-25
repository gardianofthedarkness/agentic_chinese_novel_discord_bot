#!/usr/bin/env python3
"""
Identify Volume Structure from Qdrant Data
Understanding that volumes are separate files/rows, not hierarchical chapters
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
import json

def identify_volume_structure():
    """Identify volume structure from Qdrant data where volumes are separate files"""
    
    print("=" * 80)
    print("🔍 IDENTIFYING VOLUME STRUCTURE FROM QDRANT DATA")
    print("=" * 80)
    print("📋 Understanding: Volumes are stored as separate files/rows")
    print("📋 Goal: Identify volume boundaries and group chunks by volume")
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
            
            print(f"   Retrieved batch: {len(batch_points)} points (total: {len(all_points)})")
            
            if len(batch_points) < batch_size:
                break
            
            offset = points[1] if len(points) > 1 and points[1] else None
            if not offset:
                break
        
        print(f"✅ Total points retrieved: {len(all_points)}")
        
        # Analyze volume patterns in content
        print("\n🔍 Analyzing volume patterns in content...")
        
        volumes_content = defaultdict(list)  # volume_id -> list of chunks
        volume_metadata = {}  # volume_id -> metadata
        
        volume_patterns = [
            r'魔法禁书目录\s*(\d+)',  # 魔法禁书目录 X
            r'魔法禁书目录\s*第(\d+)卷',  # 魔法禁书目录 第X卷
            r'第(\d+)卷',  # 第X卷
        ]
        
        for i, point in enumerate(all_points):
            payload = point.payload
            content = payload.get('chunk', '')
            
            # Look for volume indicators in content
            volume_found = None
            volume_title = ""
            
            for pattern in volume_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    volume_found = int(matches[0])
                    
                    # Try to extract volume title
                    lines = content.split('\n')[:10]  # Check first 10 lines
                    for line in lines:
                        if '魔法禁书目录' in line and str(volume_found) in line:
                            volume_title = line.strip()
                            break
                        elif '第' in line and '卷' in line and str(volume_found) in line:
                            volume_title = line.strip()
                            break
                    
                    break
            
            # If we found a volume, store the data
            if volume_found:
                volumes_content[volume_found].append({
                    'point_id': i,
                    'content': content,
                    'content_length': len(content)
                })
                
                if volume_found not in volume_metadata:
                    volume_metadata[volume_found] = {
                        'volume_title': volume_title,
                        'first_content_preview': content[:200],
                        'total_chunks': 0,
                        'total_chars': 0
                    }
                
                volume_metadata[volume_found]['total_chunks'] += 1
                volume_metadata[volume_found]['total_chars'] += len(content)
            
            # Show progress every 500 points
            if (i + 1) % 500 == 0:
                print(f"   Processed {i + 1}/{len(all_points)} points...")
        
        # Analyze results
        print(f"\n📊 VOLUME STRUCTURE ANALYSIS:")
        print(f"   📚 Total volumes identified: {len(volumes_content)}")
        print(f"   📄 Total points analyzed: {len(all_points)}")
        
        volume_ids = sorted(volumes_content.keys())
        print(f"   🔢 Volume IDs found: {volume_ids}")
        
        # Detailed volume breakdown
        print(f"\n📖 DETAILED VOLUME BREAKDOWN:")
        for vol_id in volume_ids:
            vol_data = volumes_content[vol_id]
            vol_meta = volume_metadata[vol_id]
            
            print(f"\n📕 Volume {vol_id}:")
            print(f"   📝 Title: {vol_meta['volume_title']}")
            print(f"   📄 Chunks: {vol_meta['total_chunks']}")
            print(f"   📊 Total characters: {vol_meta['total_chars']:,}")
            print(f"   📖 Avg chars/chunk: {vol_meta['total_chars'] // max(vol_meta['total_chunks'], 1):,}")
            
            # Show first chunk preview
            if vol_data:
                preview = vol_data[0]['content'][:150]
                print(f"   🔍 First chunk preview: {preview}...")
        
        # Check for first 5 volumes
        print(f"\n🎯 FIRST 5 VOLUMES ANALYSIS:")
        first_5_volumes = [vol_id for vol_id in volume_ids if vol_id <= 5]
        print(f"   📚 Available in first 5: {first_5_volumes}")
        
        if len(first_5_volumes) >= 5:
            print(f"   ✅ All 5 volumes available for processing")
            
            total_chunks = sum(volume_metadata[vol_id]['total_chunks'] for vol_id in first_5_volumes)
            total_chars = sum(volume_metadata[vol_id]['total_chars'] for vol_id in first_5_volumes)
            
            print(f"   📊 Total chunks in first 5 volumes: {total_chunks}")
            print(f"   📝 Total characters in first 5 volumes: {total_chars:,}")
            
            # Estimate processing metrics
            estimated_tokens = total_chars // 4  # Rough token estimate
            estimated_cost = estimated_tokens * 0.00002  # Rough cost estimate
            
            print(f"   💰 Estimated tokens for processing: {estimated_tokens:,}")
            print(f"   💵 Estimated cost: ${estimated_cost:.4f}")
            
        else:
            print(f"   ⚠️  Only {len(first_5_volumes)} volumes available in first 5")
        
        # Save volume structure for processing
        volume_structure = {
            'volumes_content': dict(volumes_content),
            'volume_metadata': volume_metadata,
            'volume_ids': volume_ids,
            'first_5_volumes': first_5_volumes,
            'total_points': len(all_points)
        }
        
        return volume_structure
        
    except Exception as e:
        print(f"❌ Error identifying volume structure: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_volume_structure(volume_structure, filename="volume_structure.json"):
    """Save volume structure to file for later processing"""
    
    try:
        # Convert defaultdict to regular dict for JSON serialization
        serializable_structure = {
            'volume_metadata': volume_structure['volume_metadata'],
            'volume_ids': volume_structure['volume_ids'],
            'first_5_volumes': volume_structure['first_5_volumes'],
            'total_points': volume_structure['total_points'],
            'volumes_summary': {
                vol_id: {
                    'total_chunks': len(chunks),
                    'total_chars': sum(chunk['content_length'] for chunk in chunks)
                }
                for vol_id, chunks in volume_structure['volumes_content'].items()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_structure, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Volume structure saved to: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving volume structure: {e}")
        return False

if __name__ == "__main__":
    volume_structure = identify_volume_structure()
    
    if volume_structure:
        print(f"\n✅ Volume structure analysis complete!")
        print(f"   📚 Total volumes: {len(volume_structure['volume_ids'])}")
        print(f"   🎯 First 5 volumes: {volume_structure['first_5_volumes']}")
        print(f"   📄 Total data points: {volume_structure['total_points']}")
        
        # Save structure
        save_volume_structure(volume_structure)
    else:
        print(f"❌ Failed to analyze volume structure")