#!/usr/bin/env python3
"""
Demonstrate Character Evolution Tracking with Simulated Data
Shows how "一个小女孩" evolves into "茵蒂克丝"
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CharacterReference:
    identifier: str
    descriptions: List[str]
    actions: List[str]
    chapter: int
    confidence: float

class CharacterEvolutionDemo:
    """Demonstrate character evolution with real examples"""
    
    def __init__(self):
        self.ambiguous_characters = {}
        self.named_characters = {}
        self.evolution_log = []
    
    def simulate_character_discovery(self):
        """Simulate character discovery across chapters with real examples"""
        
        print("="*60)
        print("CHARACTER EVOLUTION DEMONSTRATION")
        print("Simulating character tracking across first 3 big chapters")
        print("="*60)
        
        # === BIG CHAPTER 1 ===
        print(f"\nBIG CHAPTER 1: Initial character discovery")
        print("-" * 40)
        
        # Ambiguous character appears
        ambiguous_girl = CharacterReference(
            identifier="一个小女孩",
            descriptions=["银发", "修女服", "受伤"],
            actions=["从阳台坠落", "被救助"],
            chapter=1,
            confidence=0.3
        )
        
        self.ambiguous_characters["一个小女孩"] = ambiguous_girl
        print(f"DISCOVERED: '{ambiguous_girl.identifier}'")
        print(f"   Descriptions: {ambiguous_girl.descriptions}")
        print(f"   Actions: {ambiguous_girl.actions}")
        print(f"   Confidence: {ambiguous_girl.confidence:.1f} (LOW - just a description)")
        
        ambiguous_boy = CharacterReference(
            identifier="一名不良高中生",
            descriptions=["黄发", "刺头"],
            actions=["在便利店闹事"],
            chapter=1,
            confidence=0.2
        )
        
        self.ambiguous_characters["一名不良高中生"] = ambiguous_boy
        print(f"DISCOVERED: '{ambiguous_boy.identifier}'")
        print(f"   Descriptions: {ambiguous_boy.descriptions}")
        print(f"   Actions: {ambiguous_boy.actions}")
        print(f"   Confidence: {ambiguous_boy.confidence:.1f} (LOW - background character)")
        
        # === BIG CHAPTER 2 ===
        print(f"\nBIG CHAPTER 2: Character development")
        print("-" * 40)
        
        # The girl appears again with more context
        self.ambiguous_characters["一个小女孩"].descriptions.extend(["完全记忆能力", "英国清教"])
        self.ambiguous_characters["一个小女孩"].actions.extend(["与当麻对话", "展示知识"])
        self.ambiguous_characters["一个小女孩"].confidence = 0.6
        
        print(f"UPDATED: '一个小女孩'")
        print(f"   New descriptions: {self.ambiguous_characters['一个小女孩'].descriptions}")
        print(f"   New confidence: {self.ambiguous_characters['一个小女孩'].confidence:.1f} (MEDIUM - recurring character)")
        
        # Trigger: Mention frequency increased
        print(f"TRIGGER: '一个小女孩' appears in multiple chapters")
        
        # === BIG CHAPTER 3 ===
        print(f"\nBIG CHAPTER 3: Character identity revelation")
        print("-" * 40)
        
        # Named character appears!
        named_character = {
            "name": "茵蒂克丝",
            "descriptions": ["银发", "修女服", "完全记忆能力", "英国清教", "禁书目录"],
            "actions": ["与当麻对话", "展示魔法知识"],
            "abilities": ["十万三千册魔道书"],
            "chapter": 3,
            "confidence": 0.9
        }
        
        self.named_characters["茵蒂克丝"] = named_character
        print(f"NEW NAMED CHARACTER: '{named_character['name']}'")
        print(f"   Descriptions: {named_character['descriptions']}")
        print(f"   Abilities: {named_character['abilities']}")
        print(f"   Confidence: {named_character['confidence']:.1f} (HIGH - named character)")
        
        # === CHARACTER MATCHING ===
        print(f"\nAUTOMATIC CHARACTER MATCHING")
        print("-" * 40)
        
        # Calculate similarity
        similarity_score = self.calculate_similarity("一个小女孩", "茵蒂克丝")
        print(f"Calculating similarity between '一个小女孩' and '茵蒂克丝'...")
        print(f"   Shared descriptions: {['银发', '修女服', '完全记忆能力', '英国清教']}")
        print(f"   Shared actions: {['与当麻对话']}")
        print(f"   Similarity score: {similarity_score:.2f}")
        
        if similarity_score >= 0.7:
            print(f"HIGH SIMILARITY DETECTED!")
            print(f"   Triggering minimal LLM confirmation...")
            
            # Simulate LLM confirmation (minimal tokens)
            llm_confirmed = self.simulate_llm_confirmation("一个小女孩", "茵蒂克丝")
            
            if llm_confirmed:
                print(f"LLM CONFIRMS: '一个小女孩' = '茵蒂克丝'")
                print(f"   Token usage: ~30 tokens (very efficient!)")
                
                # === CHARACTER MERGING ===
                print(f"\nCHARACTER MERGING & RETROACTIVE HEALING")
                print("-" * 40)
                
                self.merge_characters("一个小女孩", "茵蒂克丝")
        
        # === FINAL RESULTS ===
        print(f"\nFINAL CHARACTER TRACKING RESULTS")
        print("=" * 40)
        
        for name, char in self.named_characters.items():
            if char.get('merged_from'):
                print(f"\n'{name}' (EVOLVED CHARACTER)")
                print(f"   Originally: '{char['merged_from']}'")
                print(f"   Chapters: {char['chapter_span']}")
                print(f"   Final confidence: {char['confidence']:.1f}")
                print(f"   Total descriptions: {len(char['descriptions'])}")
                print(f"   >>> CHARACTER SUCCESSFULLY EVOLVED FROM AMBIGUOUS TO NAMED <<<")
        
        # Show remaining ambiguous characters
        remaining_ambiguous = [char for char in self.ambiguous_characters.values() if not char.identifier.startswith("merged_")]
        if remaining_ambiguous:
            print(f"\nREMAINING AMBIGUOUS CHARACTERS:")
            for char in remaining_ambiguous:
                print(f"   '{char.identifier}' - Confidence: {char.confidence:.1f}")
    
    def calculate_similarity(self, ambiguous_id: str, named_char: str) -> float:
        """Calculate similarity between ambiguous and named character"""
        
        ambiguous = self.ambiguous_characters[ambiguous_id]
        named = self.named_characters[named_char]
        
        # Description similarity
        ambiguous_desc = set(ambiguous.descriptions)
        named_desc = set(named['descriptions'])
        
        shared_descriptions = ambiguous_desc.intersection(named_desc)
        desc_similarity = len(shared_descriptions) / len(ambiguous_desc) if ambiguous_desc else 0
        
        # Action similarity  
        ambiguous_actions = set(ambiguous.actions)
        named_actions = set(named['actions'])
        
        shared_actions = ambiguous_actions.intersection(named_actions)
        action_similarity = len(shared_actions) / len(ambiguous_actions) if ambiguous_actions else 0
        
        # Weighted similarity
        total_similarity = (desc_similarity * 0.7) + (action_similarity * 0.3)
        
        return total_similarity
    
    def simulate_llm_confirmation(self, ambiguous_id: str, named_char: str) -> bool:
        """Simulate minimal LLM confirmation call"""
        
        # This would be the actual LLM call:
        # prompt = f"简短回答：'{ambiguous_id}'和'{named_char}'是同一个角色吗？只回答：是 或 否"
        # response = await deepseek_client.generate(prompt, max_tokens=10)
        
        # For demo, we simulate a positive response
        return True
    
    def merge_characters(self, ambiguous_id: str, named_char: str):
        """Merge ambiguous character with named character"""
        
        ambiguous = self.ambiguous_characters[ambiguous_id]
        named = self.named_characters[named_char]
        
        # Merge data
        merged_data = {
            'name': named_char,
            'descriptions': list(set(named['descriptions'] + ambiguous.descriptions)),
            'actions': list(set(named['actions'] + ambiguous.actions)),
            'abilities': named.get('abilities', []),
            'confidence': min(named['confidence'] + 0.1, 1.0),
            'chapter_span': [ambiguous.chapter, named['chapter']],
            'merged_from': ambiguous_id,
            'first_appearance_as_ambiguous': ambiguous.chapter,
            'named_revelation_chapter': named['chapter']
        }
        
        self.named_characters[named_char] = merged_data
        
        # Mark ambiguous character as merged
        self.ambiguous_characters[f"merged_{ambiguous_id}"] = ambiguous
        del self.ambiguous_characters[ambiguous_id]
        
        print(f"MERGED: '{ambiguous_id}' -> '{named_char}'")
        print(f"   Chapter span: {merged_data['chapter_span']}")
        print(f"   Enhanced confidence: {merged_data['confidence']:.1f}")
        
        # Retroactive healing simulation
        print(f"RETROACTIVE HEALING:")
        print(f"   All events involving '{ambiguous_id}' in Chapter {ambiguous.chapter}")
        print(f"   are now properly attributed to '{named_char}'")
        print(f"   Timeline importance scores boosted for early appearances")

async def main():
    """Run the character evolution demonstration"""
    
    demo = CharacterEvolutionDemo()
    demo.simulate_character_discovery()
    
    print(f"\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print(f"This shows how our dynamic character tracking system:")
    print(f"  - Stores ambiguous references temporarily")
    print(f"  - Tracks character development across chapters")
    print(f"  - Automatically detects when identities are revealed")
    print(f"  - Uses minimal LLM tokens for confirmation (~30 tokens)")
    print(f"  - Merges and heals character data retroactively")
    print(f"  - Maintains story coherence and character importance")

if __name__ == "__main__":
    asyncio.run(main())