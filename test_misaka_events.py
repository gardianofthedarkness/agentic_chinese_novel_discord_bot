#!/usr/bin/env python3
"""
Test Misaka Events Request
Simulate the exact Discord bot request: "列举御坂美琴的全部事件"
"""

import asyncio
import json
import requests
from datetime import datetime
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient

class MisakaEventsTest:
    """Test the specific Misaka events request"""
    
    def __init__(self):
        self.db_engine = create_engine('postgresql://admin:admin@localhost:5432/novel_sim')
        self.qdrant_client = QdrantClient(url="http://localhost:32768", verify=False)
        self.api_url = "http://localhost:5005"
    
    def check_misaka_in_database(self):
        """Check what Misaka-related data exists in database"""
        
        print("=== MISAKA DATA IN DATABASE ===")
        
        with self.db_engine.connect() as conn:
            # Check character profiles
            print("\n1. CHARACTER PROFILES:")
            result = conn.execute(text("""
                SELECT name, character_type, first_appearance_chapter, confidence_score
                FROM character_profiles 
                WHERE name LIKE '%御坂%' OR name LIKE '%美琴%' OR name LIKE '%Misaka%'
                ORDER BY confidence_score DESC
            """))
            
            misaka_chars = list(result)
            if misaka_chars:
                for row in misaka_chars:
                    print(f"  {row[0]} ({row[1]}) - Chapter {row[2]}, Confidence {row[3]:.2f}")
            else:
                print("  No Misaka characters found!")
            
            # Check unified characters (variant resolution)
            print("\n2. UNIFIED CHARACTERS:")
            try:
                result = conn.execute(text("""
                    SELECT primary_name, character_type, aliases, first_appearance_chapter
                    FROM unified_characters
                    WHERE primary_name LIKE '%御坂%' OR primary_name LIKE '%美琴%'
                    ORDER BY combined_confidence DESC
                """))
                
                for row in result:
                    aliases = json.loads(row[2]) if isinstance(row[2], str) else row[2] or []
                    print(f"  {row[0]} ({row[1]}) - Chapter {row[3]}")
                    if aliases:
                        print(f"    Aliases: {aliases}")
            except Exception as e:
                print(f"  Error checking unified characters: {e}")
            
            # Check timeline events
            print("\n3. TIMELINE EVENTS:")
            result = conn.execute(text("""
                SELECT event_description, chapter_context, characters_involved
                FROM timeline_events
                WHERE characters_involved::text LIKE '%御坂%' OR characters_involved::text LIKE '%美琴%'
                ORDER BY chapter_context
            """))
            
            events = list(result)
            if events:
                for i, row in enumerate(events):
                    chars = json.loads(row[2]) if isinstance(row[2], str) else row[2] or []
                    print(f"  Event {i+1}: {row[0][:100]}...")
                    print(f"    Chapter: {row[1]}")
                    print(f"    Characters: {chars}")
            else:
                print("  No timeline events found for Misaka!")
    
    def check_misaka_in_qdrant(self):
        """Check Misaka appearances in Qdrant across all chapters"""
        
        print("\n=== MISAKA DATA IN QDRANT ===")
        
        # Search for Misaka mentions across all coordinates
        points = self.qdrant_client.scroll(
            collection_name="test_novel2",
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        misaka_appearances = []
        chapter_boundaries = [
            (0, 46, "Chapter 1"),
            (47, 74, "Chapter 2"), 
            (75, 93, "Chapter 3")
        ]
        
        for point in points[0]:
            content = point.payload.get('chunk', '')
            coord = point.payload.get('coordinate', [0, 0])
            coord_y = coord[1] if len(coord) > 1 else 0
            
            # Look for Misaka mentions
            if any(name in content for name in ['御坂', '美琴', 'Misaka']):
                # Determine which chapter
                chapter = "Unknown"
                for start, end, ch_name in chapter_boundaries:
                    if start <= coord_y <= end:
                        chapter = ch_name
                        break
                
                misaka_appearances.append({
                    'coordinate': coord_y,
                    'chapter': chapter,
                    'content': content[:200]
                })
        
        print(f"\nFound {len(misaka_appearances)} Misaka mentions in Qdrant:")
        
        # Group by chapter
        by_chapter = {}
        for appearance in misaka_appearances:
            chapter = appearance['chapter']
            if chapter not in by_chapter:
                by_chapter[chapter] = []
            by_chapter[chapter].append(appearance)
        
        for chapter, appearances in by_chapter.items():
            print(f"\n{chapter}: {len(appearances)} mentions")
            for i, app in enumerate(appearances[:3]):  # Show first 3
                print(f"  {i+1}. Y={app['coordinate']}: {app['content'][:100]}...")
            if len(appearances) > 3:
                print(f"  ... and {len(appearances) - 3} more")
    
    async def test_api_request(self):
        """Test the actual API request like Discord bot would make"""
        
        print("\n=== TESTING API REQUEST ===")
        
        try:
            # First check if API is running
            response = requests.get(f"{self.api_url}/health", timeout=5)
            print(f"API Health: {response.status_code}")
            
            # Test the chat request
            request_data = {
                "message": "列举御坂美琴的全部事件",
                "history": []
            }
            
            print(f"\nSending request: {request_data}")
            
            response = requests.post(
                f"{self.api_url}/api/agent/chat",
                json=request_data,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nAPI Response:")
                print(f"Response: {result.get('response', 'No response field')}")
                print(f"Source: {result.get('source', 'No source field')}")
                if 'metadata' in result:
                    print(f"Metadata: {result['metadata']}")
            else:
                print(f"Error response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("ERROR: API server not responding. Is it running?")
        except Exception as e:
            print(f"ERROR: {e}")
    
    def analyze_chapter_coverage(self):
        """Analyze why only Chapter 1 might be showing"""
        
        print("\n=== CHAPTER COVERAGE ANALYSIS ===")
        
        with self.db_engine.connect() as conn:
            # Check what's in each chapter
            for chapter_num in [1, 2, 3]:
                print(f"\nChapter {chapter_num}:")
                
                # Characters in this chapter
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM character_profiles 
                    WHERE first_appearance_chapter = :chapter
                """), {'chapter': chapter_num})
                char_count = result.fetchone()[0]
                print(f"  Characters: {char_count}")
                
                # Events in this chapter
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM timeline_events 
                    WHERE chapter_context = :chapter
                """), {'chapter': chapter_num})
                event_count = result.fetchone()[0]
                print(f"  Events: {event_count}")
                
                # Chapter summary length
                result = conn.execute(text("""
                    SELECT LENGTH(content_summary) 
                    FROM chapter_summaries 
                    WHERE chapter_index = :chapter
                """), {'chapter': chapter_num})
                summary_len = result.fetchone()
                summary_len = summary_len[0] if summary_len else 0
                print(f"  Summary length: {summary_len} chars")

async def main():
    """Run the Misaka events test"""
    
    print("TESTING MISAKA EVENTS REQUEST")
    print("Simulating: 列举御坂美琴的全部事件")
    print("=" * 50)
    
    tester = MisakaEventsTest()
    
    # Run all tests
    tester.check_misaka_in_database()
    tester.check_misaka_in_qdrant()
    await tester.test_api_request()
    tester.analyze_chapter_coverage()
    
    print("\n" + "=" * 50)
    print("MISAKA EVENTS TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())