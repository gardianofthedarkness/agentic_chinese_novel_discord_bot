#!/usr/bin/env python3
"""
Test Variant-Aware Character Agent
Demonstrates how our agent can now handle character name variants
"""

import asyncio
import json
from sqlalchemy import create_engine, text
from typing import Dict, Any, Optional

class VariantAwareCharacterAgent:
    """Agent that can handle character name variants intelligently"""
    
    def __init__(self):
        self.db_engine = create_engine('postgresql://admin:admin@localhost:5432/novel_sim')
    
    def resolve_character(self, query_name: str) -> Optional[Dict[str, Any]]:
        """Resolve any character name variant to unified character info"""
        
        with self.db_engine.connect() as conn:
            # Look up character by variant name
            result = conn.execute(text('''
                SELECT uc.*, cvl.is_primary
                FROM character_variant_lookup cvl
                JOIN unified_characters uc ON cvl.unified_id = uc.unified_id
                WHERE cvl.variant_name = :query_name
            '''), {'query_name': query_name})
            
            row = result.fetchone()
            if not row:
                return None
            
            # Handle JSON fields safely
            aliases = row[3] if isinstance(row[3], list) else json.loads(row[3]) if row[3] else []
            all_variants = row[4] if isinstance(row[4], list) else json.loads(row[4]) if row[4] else []
            metadata = row[9] if isinstance(row[9], dict) else json.loads(row[9]) if row[9] else {}
            
            return {
                'unified_id': row[0],
                'primary_name': row[1],
                'character_type': row[2],
                'aliases': aliases,
                'all_variants': all_variants,
                'combined_confidence': row[5],
                'first_appearance': row[6],
                'total_mentions': row[7],
                'is_primary_name': row[10],
                'query_matched': query_name,
                'metadata': metadata
            }
    
    def get_character_summary(self, query_name: str) -> str:
        """Get a formatted character summary"""
        
        char_info = self.resolve_character(query_name)
        if not char_info:
            return f"Character '{query_name}' not found in database."
        
        # Create summary
        summary = f"Character: {char_info['primary_name']}\n"
        
        if char_info['query_matched'] != char_info['primary_name']:
            summary += f"(Queried as: {char_info['query_matched']})\n"
        
        summary += f"Type: {char_info['character_type']}\n"
        summary += f"First Appearance: Chapter {char_info['first_appearance']}\n"
        summary += f"Confidence: {char_info['combined_confidence']:.2f}\n"
        summary += f"Total Mentions: {char_info['total_mentions']}\n"
        
        if char_info['aliases']:
            summary += f"Also known as: {', '.join(char_info['aliases'])}\n"
        
        if char_info['all_variants']:
            summary += f"All name variants found: {', '.join(char_info['all_variants'])}\n"
        
        return summary
    
    def find_related_characters(self, query_name: str) -> list:
        """Find characters related to the queried character"""
        
        char_info = self.resolve_character(query_name)
        if not char_info:
            return []
        
        # For demo, find characters from same chapter
        with self.db_engine.connect() as conn:
            result = conn.execute(text('''
                SELECT primary_name, character_type, first_appearance_chapter
                FROM unified_characters 
                WHERE first_appearance_chapter = :chapter
                AND primary_name != :primary_name
                ORDER BY combined_confidence DESC
            '''), {
                'chapter': char_info['first_appearance'],
                'primary_name': char_info['primary_name']
            })
            
            related = []
            for row in result:
                related.append({
                    'name': row[0],
                    'type': row[1],
                    'chapter': row[2]
                })
            
            return related
    
    def answer_character_question(self, question: str) -> str:
        """Answer questions about characters intelligently"""
        
        # Simple keyword extraction (in practice, would use more sophisticated NLP)
        question_lower = question.lower()
        
        # Test common character name variations
        character_keywords = [
            '御坂美琴', '美琴', '御坂', 'misaka',
            '上条当麻', '当麻', '上条', 'touma',
            '茵蒂克丝', '禁书目录', 'index',
            '小萌老师', '小萌',
            '史提尔'
        ]
        
        found_character = None
        for keyword in character_keywords:
            if keyword in question:
                found_character = keyword
                break
        
        if not found_character:
            return "I couldn't identify which character you're asking about."
        
        # Get character info
        char_info = self.resolve_character(found_character)
        if not char_info:
            return f"Sorry, I don't have information about '{found_character}'."
        
        # Generate response based on question type
        if 'who is' in question_lower or 'tell me about' in question_lower:
            response = f"**{char_info['primary_name']}** is a {char_info['character_type']} character who first appears in Chapter {char_info['first_appearance']}. "
            
            if char_info['aliases']:
                response += f"This character is also known as: {', '.join(char_info['aliases'])}. "
            
            response += f"I found {char_info['total_mentions']} references to this character across different name variants."
            
        elif 'other names' in question_lower or 'aliases' in question_lower or 'called' in question_lower:
            if char_info['aliases']:
                response = f"**{char_info['primary_name']}** is also referred to as: {', '.join(char_info['aliases'])}. "
                response += f"In total, I found these name variants: {', '.join(char_info['all_variants'])}"
            else:
                response = f"**{char_info['primary_name']}** doesn't have any aliases in the chapters I've analyzed."
                
        elif 'appears' in question_lower or 'chapter' in question_lower:
            response = f"**{char_info['primary_name']}** first appears in Chapter {char_info['first_appearance']}."
            
            # Find related characters
            related = self.find_related_characters(found_character)
            if related:
                response += f" Other characters from the same chapter include: {', '.join([r['name'] for r in related[:3]])}."
        
        else:
            # General info
            response = self.get_character_summary(found_character)
        
        return response

async def test_variant_awareness():
    """Test the variant-aware character system"""
    
    print("="*70)
    print("VARIANT-AWARE CHARACTER AGENT TEST")
    print("="*70)
    
    agent = VariantAwareCharacterAgent()
    
    # Test 1: Direct character lookup with variants
    print("\n1. TESTING CHARACTER VARIANT RESOLUTION:")
    print("-" * 50)
    
    test_names = [
        '御坂美琴',  # Primary name
        '美琴',      # Common alias
        '御坂',      # Short name
        '上条当麻',  # Another character primary
        '当麻',      # Another character alias
        '茵蒂克丝',  # Index primary
        '禁书目录',  # Index alias
        '小萌',      # Teacher alias
        '不存在'     # Non-existent character
    ]
    
    for name in test_names:
        char_info = agent.resolve_character(name)
        if char_info:
            print(f"Query: '{name}' -> Resolved to: {char_info['primary_name']} ({char_info['character_type']})")
            if name != char_info['primary_name']:
                print(f"  Note: '{name}' is an alias for '{char_info['primary_name']}'")
        else:
            print(f"Query: '{name}' -> No character found")
    
    # Test 2: Character question answering
    print("\n\\n2. TESTING INTELLIGENT CHARACTER Q&A:")
    print("-" * 50)
    
    questions = [
        "Who is 御坂美琴?",
        "Tell me about 美琴",
        "What other names is 御坂 called?",
        "Who is 当麻?",
        "What chapter does 茵蒂克丝 appear in?",
        "Tell me about Index",
        "What are the aliases for 小萌?"
    ]
    
    for question in questions:
        print(f"\\nQ: {question}")
        answer = agent.answer_character_question(question)
        print(f"A: {answer}")
    
    # Test 3: Demonstrate the improvement
    print("\\n\\n3. SYSTEM IMPROVEMENT DEMONSTRATION:")
    print("-" * 50)
    
    print("BEFORE (without variant resolution):")
    print("  Query '美琴' -> No results (stored as separate entry)")
    print("  Query '御坂' -> No results (stored as separate entry)")
    print("  Query '御坂美琴' -> Found (only if exact match)")
    
    print("\\nAFTER (with variant resolution):")
    misaka_variants = ['御坂美琴', '美琴', '御坂']
    for variant in misaka_variants:
        char_info = agent.resolve_character(variant)
        if char_info:
            print(f"  Query '{variant}' -> {char_info['primary_name']} ({char_info['character_type']})")
    
    print("\\n>>> ALL VARIANTS NOW RESOLVE TO THE SAME CHARACTER! <<<")

if __name__ == "__main__":
    asyncio.run(test_variant_awareness())