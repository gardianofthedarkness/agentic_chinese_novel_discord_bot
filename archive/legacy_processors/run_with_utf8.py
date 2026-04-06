#!/usr/bin/env python3
"""
UTF-8 Safe Hierarchical Analysis Runner
Automatically handles encoding issues for Chinese character processing
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime

# Setup UTF-8 environment first
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from hierarchical_chapter_parser import HierarchicalChapterParser

async def run_safe_hierarchical_analysis():
    """Run hierarchical analysis with UTF-8 safety"""
    
    print("="*80)
    print("HIERARCHICAL NOVEL ANALYSIS - UTF-8 SAFE VERSION")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print("Processing Chinese novel with proper juan/zhang distinction")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Connect to Qdrant
        print("\nStep 1: Connecting to Qdrant database...")
        
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:32768", verify=False)
        
        points = client.scroll(
            collection_name="test_novel2",
            limit=200,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"SUCCESS: Retrieved {len(points[0])} content points")
        
        # Step 2: Parse hierarchical structure
        print("\nStep 2: Parsing hierarchical structure...")
        
        combined_content = ""
        content_stats = {
            'total_points': len(points[0]),
            'total_chars': 0,
            'points_with_content': 0
        }
        
        for point in points[0]:
            if 'chunk' in point.payload:
                content = point.payload['chunk']
                combined_content += content + "\n\n"
                content_stats['total_chars'] += len(content)
                content_stats['points_with_content'] += 1
        
        print(f"Content processed:")
        print(f"  - Total points: {content_stats['total_points']}")
        print(f"  - Points with content: {content_stats['points_with_content']}")
        print(f"  - Total characters: {content_stats['total_chars']:,}")
        
        # Step 3: Run hierarchical parser
        print("\nStep 3: Running hierarchical parser...")
        parse_start = time.time()
        
        parser = HierarchicalChapterParser()
        chapters = parser.parse_content_hierarchy(combined_content)
        
        parse_time = time.time() - parse_start
        print(f"SUCCESS: Parsed {len(chapters)} chapters in {parse_time:.2f} seconds")
        
        # Step 4: Analyze structure
        if not chapters:
            print("WARNING: No chapters found")
            return {'success': False, 'error': 'No chapters parsed'}
        
        # Volume analysis
        volume_data = {}
        for chapter in chapters:
            vol_id = chapter.volume_id or 0
            if vol_id not in volume_data:
                volume_data[vol_id] = {
                    'title': chapter.volume_title or 'Unstructured',
                    'chapters': [],
                    'total_words': 0
                }
            volume_data[vol_id]['chapters'].append(chapter)
            volume_data[vol_id]['total_words'] += chapter.word_count
        
        print(f"\nStructure Summary:")
        print(f"  Volumes identified: {len(volume_data)}")
        print(f"  Chapters parsed: {len(chapters)}")
        print(f"  Total words: {sum(ch.word_count for ch in chapters):,}")
        
        # Step 5: Display results
        print("\n" + "="*60)
        print("VOLUME/CHAPTER BREAKDOWN")
        print("="*60)
        
        for vol_id in sorted(volume_data.keys()):
            vol_info = volume_data[vol_id]
            print(f"\nVolume {vol_id}: {vol_info['title']}")
            print(f"  Chapters: {len(vol_info['chapters'])}")
            print(f"  Words: {vol_info['total_words']:,}")
            
            # Show chapter details (first 3 per volume)
            for chapter in vol_info['chapters'][:3]:
                # Use ASCII-safe formatting for display
                title_safe = chapter.chapter_title.encode('ascii', 'ignore').decode('ascii')
                if not title_safe:
                    title_safe = f"Chapter_{chapter.chapter_id}"
                
                print(f"    {chapter.hierarchy_id}: {title_safe}")
                print(f"      Words: {chapter.word_count:,}")
                
                # Safe content preview
                content_preview = chapter.content[:50].encode('ascii', 'ignore').decode('ascii')
                print(f"      Preview: {content_preview}...")
            
            if len(vol_info['chapters']) > 3:
                print(f"    ... and {len(vol_info['chapters']) - 3} more chapters")
        
        # Step 6: Validation
        print(f"\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        validations = {
            'Volume detection': len(volume_data) > 0,
            'Chapter detection': len(chapters) > 0,
            'Hierarchical IDs': any('.' in ch.hierarchy_id for ch in chapters),
            'Volume titles': any(vol['title'] != 'Unstructured' for vol in volume_data.values()),
            'Word counts': all(ch.word_count > 0 for ch in chapters)
        }
        
        for check, passed in validations.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
        
        # Step 7: AI Analysis readiness
        print(f"\n" + "="*60)
        print("AI ANALYSIS PREPARATION")
        print("="*60)
        
        ready_chapters = [ch for ch in chapters if ch.word_count > 100]
        estimated_tokens = sum(min(ch.word_count, 2000) for ch in ready_chapters) // 3
        
        print(f"Chapters ready for AI: {len(ready_chapters)}")
        print(f"Estimated tokens: {estimated_tokens:,}")
        print(f"Estimated cost: ${estimated_tokens * 0.00002:.2f}")
        
        # Step 8: Performance summary
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Parse time: {parse_time:.2f} seconds")
        print(f"Processing rate: {content_stats['total_chars']/total_time:,.0f} chars/sec")
        
        return {
            'success': True,
            'volumes': len(volume_data),
            'chapters': len(chapters),
            'total_words': sum(ch.word_count for ch in chapters),
            'processing_time': total_time,
            'ready_for_ai': len(ready_chapters),
            'estimated_tokens': estimated_tokens
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    """Main execution with proper UTF-8 handling"""
    
    try:
        print("Setting up UTF-8 environment...")
        
        # Run the analysis
        result = asyncio.run(run_safe_hierarchical_analysis())
        
        if result['success']:
            print(f"\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Volumes: {result['volumes']}")
            print(f"Chapters: {result['chapters']}")
            print(f"Words: {result['total_words']:,}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Ready for AI analysis: {result['ready_for_ai']} chapters")
            print(f"Estimated tokens: {result['estimated_tokens']:,}")
        else:
            print(f"ANALYSIS FAILED: {result['error']}")
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    main()