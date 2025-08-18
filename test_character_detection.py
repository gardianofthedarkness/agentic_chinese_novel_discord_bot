#!/usr/bin/env python3
"""
Test Character Detection Results
Verify that characters from the original text are now properly detected and stored
"""

import asyncio
from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient

class CharacterDetectionTester:
    """Test the effectiveness of our fixed character detection"""
    
    def __init__(self):
        self.db_engine = create_engine('postgresql://admin:admin@localhost:5432/novel_sim')
        self.qdrant_client = QdrantClient(url="http://localhost:32768", verify=False)
        
        # Test characters that should definitely be found in the first 3 chapters
        self.test_characters = [
            '御坂美琴',  # The one we specifically fixed
            '美琴',      # Short version
            '御坂',      # Even shorter
            '上条当麻',  # Main protagonist
            '当麻',      # Short version
            '茵蒂克丝',  # Another main character
            '禁书目录',  # Alternative name
            '小萌老师',  # Teacher character
            '白井黑子',  # If she appears
            '一方通行'   # If he appears
        ]
    
    def get_characters_from_qdrant_sample(self) -> Dict[str, List[int]]:
        """Get actual character appearances from Qdrant for verification"""
        
        print("=== SAMPLING CHARACTERS FROM QDRANT (FIRST 3 CHAPTERS) ===")
        
        points = self.qdrant_client.scroll(
            collection_name="test_novel2",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        # Chapter boundaries based on our fixed mapping
        chapter_boundaries = [
            (0, 46),     # Chapter 1
            (47, 74),    # Chapter 2  
            (75, 93),    # Chapter 3
        ]
        
        character_appearances = {}
        
        for point in points[0]:
            content = point.payload.get('chunk', '')
            coord = point.payload.get('coordinate', [0, 0])
            coord_y = coord[1] if len(coord) > 1 else 0
            
            # Check if in first 3 chapters
            chapter_num = 0
            for ch_num, (start, end) in enumerate(chapter_boundaries, 1):
                if start <= coord_y <= end:
                    chapter_num = ch_num
                    break
            
            if chapter_num > 0:  # In our target chapters
                for char_name in self.test_characters:
                    if char_name in content:
                        if char_name not in character_appearances:
                            character_appearances[char_name] = []
                        character_appearances[char_name].append(coord_y)
        
        # Sort coordinates for each character
        for char_name in character_appearances:
            character_appearances[char_name] = sorted(list(set(character_appearances[char_name])))
        
        print(f"Characters found in Qdrant (first 3 chapters):")
        for char_name, coords in character_appearances.items():
            print(f"  '{char_name}': {len(coords)} appearances at coordinates {coords[:5]}...")
        
        return character_appearances
    
    async def get_characters_from_database(self) -> Dict[str, Dict[str, Any]]:
        """Get characters that were stored in the database"""
        
        print("\\n=== CHARACTERS STORED IN DATABASE ===")
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT character_id, name, character_type, first_appearance_chapter, confidence_score
                FROM character_profiles
                ORDER BY confidence_score DESC
            """))
            
            db_characters = {}
            for row in result:
                char_id, name, char_type, first_chap, confidence = row
                db_characters[name] = {
                    'character_id': char_id,
                    'character_type': char_type,
                    'first_appearance_chapter': first_chap,
                    'confidence_score': confidence
                }
                print(f"  '{name}': {char_type}, Chapter {first_chap}, Confidence {confidence:.2f}")
        
        return db_characters
    
    async def test_detection_accuracy(self):
        """Test the accuracy of our character detection"""
        
        print("\\n" + "="*70)
        print("CHARACTER DETECTION ACCURACY TEST")
        print("="*70)
        
        # Get ground truth from Qdrant
        qdrant_characters = self.get_characters_from_qdrant_sample()
        
        # Get what we detected and stored
        db_characters = await self.get_characters_from_database()
        
        print("\\n=== DETECTION ACCURACY ANALYSIS ===")
        
        # Test each character
        found_count = 0
        missed_count = 0
        
        for test_char in self.test_characters:
            print(f"\\nTesting '{test_char}':")
            
            # Check if it exists in Qdrant
            in_qdrant = test_char in qdrant_characters
            in_database = test_char in db_characters
            
            print(f"  In Qdrant: {'YES' if in_qdrant else 'NO'}")
            print(f"  In Database: {'YES' if in_database else 'NO'}")
            
            if in_qdrant and in_database:
                print(f"  Result: CORRECTLY DETECTED ✓")
                found_count += 1
                
                # Show details
                qdrant_coords = qdrant_characters[test_char]
                db_info = db_characters[test_char]
                print(f"    Qdrant appearances: {len(qdrant_coords)} times")
                print(f"    Database confidence: {db_info['confidence_score']:.2f}")
                
            elif in_qdrant and not in_database:
                print(f"  Result: MISSED BY DETECTION ✗")
                missed_count += 1
                qdrant_coords = qdrant_characters[test_char]
                print(f"    Should have been found: {len(qdrant_coords)} appearances")
                
            elif not in_qdrant and not in_database:
                print(f"  Result: CORRECTLY NOT DETECTED (not in chapters 1-3)")
                
            elif not in_qdrant and in_database:
                print(f"  Result: FALSE POSITIVE")
                # This shouldn't happen with our test set
        
        print(f"\\n=== ACCURACY SUMMARY ===")
        total_testable = sum(1 for char in self.test_characters if char in qdrant_characters)
        if total_testable > 0:
            accuracy = found_count / total_testable * 100
            print(f"Characters found in Qdrant: {total_testable}")
            print(f"Correctly detected: {found_count}")
            print(f"Missed: {missed_count}")
            print(f"Detection accuracy: {accuracy:.1f}%")
        else:
            print("No test characters found in first 3 chapters")
        
        # Special check for 御坂美琴
        print(f"\\n=== SPECIAL CHECK: 御坂美琴 ===")
        misaka_in_qdrant = '御坂美琴' in qdrant_characters
        misaka_in_db = '御坂美琴' in db_characters
        
        print(f"御坂美琴 in Qdrant: {'YES' if misaka_in_qdrant else 'NO'}")
        print(f"御坂美琴 in Database: {'YES' if misaka_in_db else 'NO'}")
        
        if misaka_in_qdrant and misaka_in_db:
            print("SUCCESS: 御坂美琴 detection issue has been FIXED! ✓")
        elif misaka_in_qdrant and not misaka_in_db:
            print("PROBLEM: 御坂美琴 still being missed ✗")
        else:
            print("INFO: 御坂美琴 not in first 3 chapters")
    
    async def test_database_integrity(self):
        """Test database storage integrity"""
        
        print("\\n=== DATABASE INTEGRITY TEST ===")
        
        with self.db_engine.connect() as conn:
            # Check chapter summaries
            result = conn.execute(text("SELECT COUNT(*) FROM chapter_summaries"))
            chapter_count = result.fetchone()[0]
            print(f"Chapter summaries stored: {chapter_count}")
            
            # Check character profiles
            result = conn.execute(text("SELECT COUNT(*) FROM character_profiles"))
            char_count = result.fetchone()[0]
            print(f"Character profiles stored: {char_count}")
            
            # Check for any data integrity issues
            result = conn.execute(text("""
                SELECT name, first_appearance_chapter 
                FROM character_profiles 
                WHERE first_appearance_chapter NOT BETWEEN 1 AND 3
            """))
            
            invalid_chapters = list(result)
            if invalid_chapters:
                print(f"WARNING: Characters with invalid chapter numbers:")
                for name, chapter in invalid_chapters:
                    print(f"  {name}: Chapter {chapter}")
            else:
                print("All characters have valid chapter numbers (1-3)")

async def main():
    """Run comprehensive character detection tests"""
    
    print("CHARACTER DETECTION TESTING SUITE")
    print("Testing fixed character detection system")
    print("="*50)
    
    tester = CharacterDetectionTester()
    
    # Run all tests
    await tester.test_detection_accuracy()
    await tester.test_database_integrity()
    
    print("\\n" + "="*50)
    print("TESTING COMPLETE")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())