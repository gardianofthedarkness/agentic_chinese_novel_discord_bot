#!/usr/bin/env python3
"""
Check all volumes available in Qdrant database
"""

import os
import sys

# Setup UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from qdrant_client import QdrantClient
import json

def check_qdrant_volumes():
    """Check all volumes in Qdrant database"""
    
    print("=" * 80)
    print("🔍 CHECKING QDRANT DATABASE FOR ALL VOLUMES")
    print("=" * 80)
    
    try:
        # Connect to Qdrant
        client = QdrantClient(url="http://localhost:32768", verify=False)
        
        # Get all points
        print("📖 Fetching all points from test_novel2 collection...")
        points = client.scroll(
            collection_name="test_novel2",
            limit=1000,  # Increase limit to get all data
            with_payload=True,
            with_vectors=False
        )
        
        print(f"✅ Retrieved {len(points[0])} total points")
        
        # Analyze volume structure
        volume_info = {}
        chapter_info = {}
        
        for i, point in enumerate(points[0]):
            payload = point.payload
            
            # Print first few points to see structure
            if i < 5:
                print(f"\n📄 Point {i+1} payload keys: {list(payload.keys())}")
                if 'metadata' in payload:
                    print(f"   Metadata: {payload['metadata']}")
                if 'volume' in payload:
                    print(f"   Volume: {payload['volume']}")
                if 'chapter' in payload:
                    print(f"   Chapter: {payload['chapter']}")
                if 'volume_title' in payload:
                    print(f"   Volume Title: {payload['volume_title']}")
                if 'chapter_title' in payload:
                    print(f"   Chapter Title: {payload['chapter_title']}")
            
            # Extract volume information
            volume_id = None
            chapter_id = None
            
            # Check different possible keys for volume/chapter info
            for key in ['volume', 'volume_id', 'volume_number']:
                if key in payload:
                    volume_id = payload[key]
                    break
            
            for key in ['chapter', 'chapter_id', 'chapter_number']:
                if key in payload:
                    chapter_id = payload[key]
                    break
            
            # Also check metadata
            if 'metadata' in payload and isinstance(payload['metadata'], dict):
                meta = payload['metadata']
                if 'volume' in meta:
                    volume_id = meta['volume']
                if 'chapter' in meta:
                    chapter_id = meta['chapter']
            
            # Store volume info
            if volume_id is not None:
                if volume_id not in volume_info:
                    volume_info[volume_id] = {
                        'volume_title': payload.get('volume_title', ''),
                        'chapters': set(),
                        'points_count': 0
                    }
                volume_info[volume_id]['points_count'] += 1
                
                if chapter_id is not None:
                    volume_info[volume_id]['chapters'].add(chapter_id)
        
        # Print volume summary
        print(f"\n" + "=" * 60)
        print("📚 VOLUME ANALYSIS SUMMARY")
        print("=" * 60)
        
        if volume_info:
            print(f"📖 Total unique volumes found: {len(volume_info)}")
            
            for vol_id in sorted(volume_info.keys()):
                vol_data = volume_info[vol_id]
                print(f"\n📕 Volume {vol_id}:")
                print(f"   📄 Points in this volume: {vol_data['points_count']}")
                print(f"   📚 Unique chapters: {len(vol_data['chapters'])}")
                if vol_data['volume_title']:
                    print(f"   📝 Title: {vol_data['volume_title']}")
                if vol_data['chapters']:
                    chapters_list = sorted(list(vol_data['chapters']))[:10]  # Show first 10
                    print(f"   📋 Chapter IDs: {chapters_list}")
                    if len(vol_data['chapters']) > 10:
                        print(f"      ... and {len(vol_data['chapters']) - 10} more chapters")
        else:
            print("❌ No volume information found in payload keys")
            print("\n🔍 Let's check the structure of payload data...")
            
            # Sample payload analysis
            if points[0]:
                sample_payload = points[0][0].payload
                print(f"📋 Sample payload structure:")
                for key, value in sample_payload.items():
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                        print(f"   {key}: {type(value).__name__} - '{preview}'")
                    else:
                        print(f"   {key}: {type(value).__name__} - {value}")
        
        # Check for chunk content
        print(f"\n" + "=" * 60)
        print("📝 CONTENT ANALYSIS")
        print("=" * 60)
        
        total_content_length = 0
        points_with_content = 0
        
        for point in points[0]:
            if 'chunk' in point.payload:
                content = point.payload['chunk']
                total_content_length += len(content)
                points_with_content += 1
        
        print(f"📊 Points with content: {points_with_content}/{len(points[0])}")
        print(f"📝 Total content length: {total_content_length:,} characters")
        
        return {
            'total_points': len(points[0]),
            'volume_info': volume_info,
            'total_content_length': total_content_length,
            'points_with_content': points_with_content
        }
        
    except Exception as e:
        print(f"❌ Error checking Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = check_qdrant_volumes()
    
    if result:
        print(f"\n✅ Analysis complete!")
        print(f"   Total points: {result['total_points']}")
        print(f"   Volumes found: {len(result['volume_info'])}")
        print(f"   Content points: {result['points_with_content']}")