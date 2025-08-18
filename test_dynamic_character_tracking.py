#!/usr/bin/env python3
"""
Test Dynamic Character Tracking on First 3 Chapters
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import re

from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AmbiguousCharacter:
    """Temporary character reference before identity revealed"""
    identifier: str  # "一个小女孩", "那个不良少年"
    descriptions: List[str]
    actions: List[str] 
    locations: List[str]
    relationships: List[str]
    chapter_appearances: List[int]
    dialogue_samples: List[str]
    confidence_score: float
    first_appearance: int

@dataclass
class CharacterEvolution:
    """Track how character references evolve"""
    timeline: List[Dict[str, Any]]
    mention_frequency: Dict[int, int]  # chapter -> mention count
    description_changes: List[str]
    relationship_growth: int

class DynamicCharacterTracker:
    """Intelligent character identity resolution system"""
    
    def __init__(self, qdrant_url: str = "http://localhost:32768"):
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Storage for ambiguous characters
        self.ambiguous_characters: Dict[str, AmbiguousCharacter] = {}
        self.named_characters: Dict[str, Dict[str, Any]] = {}
        self.character_evolution: Dict[str, CharacterEvolution] = {}
        
        # Matching thresholds
        self.similarity_threshold = 0.7
        self.evolution_triggers = {
            'mention_frequency': 3,
            'chapter_span': 2,
            'description_evolution': 2
        }
        
        logger.info("Dynamic Character Tracker initialized")
    
    async def get_first_n_chapters(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get first N chapters from Qdrant - use full chunks"""
        logger.info(f"Fetching first {n} chapters from Qdrant...")
        
        try:
            # Get larger chunks directly from Qdrant
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768")
            
            # Get points with full payload
            points = client.scroll(
                collection_name="test_novel2",
                limit=20,  # Get more points
                with_payload=True,
                with_vectors=False
            )
            
            chapters = []
            for point in points[0]:
                payload = point.payload
                
                # Look for chunk content (this contains the full text)
                if 'chunk' in payload:
                    content = payload['chunk']
                    
                    # Try to extract chapter info from content
                    chapter_num = 1  # Default to chapter 1
                    
                    # Look for chapter markers in content
                    if '第一章' in content or '第1章' in content:
                        chapter_num = 1
                    elif '第二章' in content or '第2章' in content:
                        chapter_num = 2
                    elif '第三章' in content or '第3章' in content:
                        chapter_num = 3
                    
                    if chapter_num <= n:
                        chapters.append({
                            'chapter': chapter_num,
                            'content': content,
                            'metadata': payload,
                            'point_id': point.id
                        })
            
            # Remove duplicates and limit to n chapters
            unique_chapters = {}
            for chapter in chapters:
                chap_num = chapter['chapter']
                if chap_num not in unique_chapters or len(chapter['content']) > len(unique_chapters[chap_num]['content']):
                    unique_chapters[chap_num] = chapter
            
            # Convert back to list and sort
            final_chapters = list(unique_chapters.values())
            final_chapters.sort(key=lambda x: x['chapter'])
            final_chapters = final_chapters[:n]  # Limit to requested number
            
            logger.info(f"Retrieved {len(final_chapters)} full chapter chunks")
            for ch in final_chapters:
                logger.info(f"  Chapter {ch['chapter']}: {len(ch['content'])} characters")
            
            return final_chapters
            
        except Exception as e:
            logger.error(f"Failed to get chapters: {e}")
            return []
    
    def extract_character_references(self, content: str, chapter: int) -> List[Dict[str, Any]]:
        """Extract all character references using regex patterns - works with garbled text"""
        
        # Since the text is garbled, we'll work with patterns that still make sense
        # in the corrupted encoding
        character_patterns = [
            # Look for word boundaries and common structural patterns
            r'([^\s]{2,6}(?=��|��|˵|��|��|��|ȥ|��))',  # Before common verbs
            r'([^\s]{2,4}(?=��|��|��))',  # Before common particles
            r'(��[^\s]{1,4})',  # Starting with common prefix
            r'([^\s]{2,4}��)',  # Ending with common suffix
            r'([^\s]{3,5}(?=��|��|��|��))',  # Before descriptive terms
        ]
        
        references = []
        for i, pattern in enumerate(character_patterns):
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) >= 2 and len(match) <= 8:  # Reasonable length
                    references.append({
                        'identifier': match,
                        'chapter': chapter,
                        'pattern_type': f'garbled_pattern_{i}',
                        'raw_match': True
                    })
        
        # Also look for repeated character sequences (potential names)
        repeated_patterns = re.findall(r'([^\s]{2,4})', content)
        char_counts = {}
        for match in repeated_patterns:
            if len(match) >= 2:
                char_counts[match] = char_counts.get(match, 0) + 1
        
        # Add frequently mentioned sequences as potential characters
        for char_seq, count in char_counts.items():
            if count >= 2 and len(char_seq) <= 6:  # Mentioned multiple times
                references.append({
                    'identifier': char_seq,
                    'chapter': chapter,
                    'pattern_type': 'frequent_sequence',
                    'mention_count': count,
                    'raw_match': True
                })
        
        # Deduplicate
        unique_refs = []
        seen = set()
        for ref in references:
            if ref['identifier'] not in seen:
                unique_refs.append(ref)
                seen.add(ref['identifier'])
        
        logger.info(f"Chapter {chapter}: Found {len(unique_refs)} character references")
        for ref in unique_refs[:5]:  # Show first 5
            logger.info(f"  '{ref['identifier']}' (type: {ref['pattern_type']})")
        
        return unique_refs
    
    def _classify_reference_type(self, reference: str) -> str:
        """Classify the type of character reference"""
        if reference.startswith(('一个', '那个', '这个')):
            return 'ambiguous_pronoun'
        elif '银发' in reference or '修女' in reference:
            return 'description_based'
        elif '不良' in reference:
            return 'role_based'
        else:
            return 'potential_name'
    
    def extract_context_features(self, content: str, character_ref: str) -> Dict[str, List[str]]:
        """Extract contextual features around character mentions"""
        
        # Find all sentences mentioning this character
        sentences = re.split(r'[。！？]', content)
        relevant_sentences = [s for s in sentences if character_ref in s]
        
        features = {
            'descriptions': [],
            'actions': [],
            'locations': [],
            'relationships': [],
            'dialogue': []
        }
        
        for sentence in relevant_sentences:
            # Extract descriptions (adjectives)
            desc_patterns = [
                r'(银发|金发|黑发|白发)',
                r'(修女服|校服|便服)',
                r'(美丽|可爱|年轻|强壮)',
                r'(小|大|高|矮|瘦|胖)'
            ]
            
            for pattern in desc_patterns:
                matches = re.findall(pattern, sentence)
                features['descriptions'].extend(matches)
            
            # Extract actions (verbs)
            action_patterns = [
                r'(说道|说|走|跑|看|听|想|做|拿|放)',
                r'(攻击|防御|保护|救|帮助)',
                r'(哭|笑|喊|叫)'
            ]
            
            for pattern in action_patterns:
                matches = re.findall(pattern, sentence)
                features['actions'].extend(matches)
            
            # Extract locations
            location_patterns = [
                r'(学园都市|学校|教室|宿舍|阳台|屋顶)',
                r'(街道|公园|便利店|医院)'
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, sentence)
                features['locations'].extend(matches)
        
        # Deduplicate all features
        for key in features:
            features[key] = list(set(features[key]))
        
        return features
    
    def create_ambiguous_character(self, reference: str, chapter: int, context_features: Dict[str, List[str]]) -> AmbiguousCharacter:
        """Create an ambiguous character entry"""
        
        return AmbiguousCharacter(
            identifier=reference,
            descriptions=context_features['descriptions'],
            actions=context_features['actions'],
            locations=context_features['locations'],
            relationships=context_features['relationships'],
            chapter_appearances=[chapter],
            dialogue_samples=context_features['dialogue'],
            confidence_score=0.3,  # Low initial confidence
            first_appearance=chapter
        )
    
    def calculate_similarity(self, ambiguous_char: AmbiguousCharacter, named_char: Dict[str, Any]) -> float:
        """Calculate similarity between ambiguous and named character"""
        
        score = 0.0
        total_weight = 0.0
        
        # Compare descriptions
        if ambiguous_char.descriptions and named_char.get('descriptions', []):
            desc_matches = 0
            for amb_desc in ambiguous_char.descriptions:
                for named_desc in named_char['descriptions']:
                    if self._semantic_match(amb_desc, named_desc):
                        desc_matches += 1
                        break
            
            desc_score = desc_matches / len(ambiguous_char.descriptions)
            score += desc_score * 0.4  # 40% weight for descriptions
            total_weight += 0.4
        
        # Compare locations
        if ambiguous_char.locations and named_char.get('locations', []):
            loc_matches = 0
            for amb_loc in ambiguous_char.locations:
                if amb_loc in named_char['locations']:
                    loc_matches += 1
            
            loc_score = loc_matches / len(ambiguous_char.locations)
            score += loc_score * 0.3  # 30% weight for locations
            total_weight += 0.3
        
        # Compare actions/behaviors
        if ambiguous_char.actions and named_char.get('actions', []):
            action_matches = 0
            for amb_action in ambiguous_char.actions:
                if amb_action in named_char['actions']:
                    action_matches += 1
            
            action_score = action_matches / len(ambiguous_char.actions)
            score += action_score * 0.3  # 30% weight for actions
            total_weight += 0.3
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _semantic_match(self, desc1: str, desc2: str) -> bool:
        """Check if two descriptions are semantically similar"""
        
        synonym_groups = {
            '银发': ['银色头发', '白发', '银发', '银色'],
            '修女': ['修女服', '教会', '宗教服装', '修女'],
            '小女孩': ['少女', '女孩', 'loli', '小女孩'],
            '学生': ['学生', '学员', '同学']
        }
        
        # Direct match
        if desc1 == desc2 or desc1 in desc2 or desc2 in desc1:
            return True
        
        # Synonym match
        for key, synonyms in synonym_groups.items():
            if desc1 in synonyms and desc2 in synonyms:
                return True
        
        return False
    
    async def minimal_llm_confirmation(self, ambiguous_id: str, named_char: str, similarity_score: float) -> bool:
        """Use minimal LLM call to confirm character match"""
        
        prompt = f"""简短回答："{ambiguous_id}"和"{named_char}"是同一个角色吗？
相似度分数：{similarity_score:.2f}
只回答：是 或 否"""
        
        try:
            await self.deepseek_client.initialize()
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=10,
                temperature=0.1
            )
            
            if response.get("success"):
                answer = response["response"].strip()
                return "是" in answer
            
        except Exception as e:
            logger.error(f"LLM confirmation failed: {e}")
        
        return False
    
    async def process_chapters(self, chapters: List[Dict[str, Any]]):
        """Process chapters and track character evolution"""
        
        logger.info("=== PROCESSING CHAPTERS FOR DYNAMIC CHARACTER TRACKING ===")
        
        for chapter_data in chapters:
            chapter = chapter_data['chapter']
            content = chapter_data['content']
            
            logger.info(f"Processing Chapter {chapter}...")
            logger.info(f"  Content length: {len(content)} chars")
            logger.info(f"  Content preview: {content[:200]}...")
            
            # Extract character references
            references = self.extract_character_references(content, chapter)
            
            for ref_data in references:
                ref_id = ref_data['identifier']
                ref_type = ref_data['pattern_type']
                
                logger.info(f"  Found reference: '{ref_id}' (type: {ref_type})")
                
                # Extract context features
                features = self.extract_context_features(content, ref_id)
                
                if ref_type in ['ambiguous_pronoun', 'description_based', 'role_based']:
                    # This is an ambiguous reference
                    if ref_id not in self.ambiguous_characters:
                        self.ambiguous_characters[ref_id] = self.create_ambiguous_character(
                            ref_id, chapter, features
                        )
                        logger.info(f"    Created ambiguous character: {ref_id}")
                    else:
                        # Update existing ambiguous character
                        char = self.ambiguous_characters[ref_id]
                        char.chapter_appearances.append(chapter)
                        char.descriptions.extend(features['descriptions'])
                        char.actions.extend(features['actions'])
                        char.locations.extend(features['locations'])
                        
                        # Deduplicate
                        char.descriptions = list(set(char.descriptions))
                        char.actions = list(set(char.actions))
                        char.locations = list(set(char.locations))
                        
                        logger.info(f"    Updated ambiguous character: {ref_id}")
                
                elif ref_type == 'potential_name':
                    # This might be a named character
                    if ref_id not in self.named_characters:
                        self.named_characters[ref_id] = {
                            'name': ref_id,
                            'descriptions': features['descriptions'],
                            'actions': features['actions'],
                            'locations': features['locations'],
                            'chapter_appearances': [chapter],
                            'confidence': 0.8
                        }
                        logger.info(f"    Created named character: {ref_id}")
                        
                        # Check for matches with ambiguous characters
                        await self.check_character_matches(ref_id)
                    else:
                        # Update existing named character
                        char = self.named_characters[ref_id]
                        char['chapter_appearances'].append(chapter)
                        char['descriptions'].extend(features['descriptions'])
                        char['actions'].extend(features['actions'])
                        char['locations'].extend(features['locations'])
                        
                        # Deduplicate
                        char['descriptions'] = list(set(char['descriptions']))
                        char['actions'] = list(set(char['actions']))
                        char['locations'] = list(set(char['locations']))
                        
                        logger.info(f"    Updated named character: {ref_id}")
    
    async def check_character_matches(self, named_char: str):
        """Check if any ambiguous characters match this named character"""
        
        named_data = self.named_characters[named_char]
        
        for ambiguous_id, ambiguous_char in self.ambiguous_characters.items():
            similarity = self.calculate_similarity(ambiguous_char, named_data)
            
            logger.info(f"Similarity between '{ambiguous_id}' and '{named_char}': {similarity:.2f}")
            
            if similarity >= self.similarity_threshold:
                # High similarity - confirm with minimal LLM call
                confirmed = await self.minimal_llm_confirmation(ambiguous_id, named_char, similarity)
                
                if confirmed:
                    logger.info(f"CONFIRMED MATCH: '{ambiguous_id}' = '{named_char}'")
                    await self.merge_characters(ambiguous_id, named_char)
                else:
                    logger.info(f"LLM rejected match: '{ambiguous_id}' != '{named_char}'")
    
    async def merge_characters(self, ambiguous_id: str, named_char: str):
        """Merge ambiguous character with named character"""
        
        ambiguous_data = self.ambiguous_characters[ambiguous_id]
        named_data = self.named_characters[named_char]
        
        # Merge data
        named_data['ambiguous_aliases'] = named_data.get('ambiguous_aliases', [])
        named_data['ambiguous_aliases'].append(ambiguous_id)
        
        # Merge chapter appearances
        all_chapters = list(set(named_data['chapter_appearances'] + ambiguous_data.chapter_appearances))
        named_data['chapter_appearances'] = sorted(all_chapters)
        
        # Merge features
        named_data['descriptions'] = list(set(named_data['descriptions'] + ambiguous_data.descriptions))
        named_data['actions'] = list(set(named_data['actions'] + ambiguous_data.actions))
        named_data['locations'] = list(set(named_data['locations'] + ambiguous_data.locations))
        
        # Boost confidence
        named_data['confidence'] = min(1.0, named_data['confidence'] + 0.2)
        
        # Mark as resolved
        named_data['resolved_from_ambiguous'] = True
        named_data['first_ambiguous_appearance'] = ambiguous_data.first_appearance
        
        logger.info(f"MERGED: '{ambiguous_id}' into '{named_char}'")
        logger.info(f"   Total appearances: chapters {named_data['chapter_appearances']}")
        logger.info(f"   Enhanced confidence: {named_data['confidence']:.2f}")
    
    def print_results(self):
        """Print the character tracking results"""
        
        print("\n" + "="*60)
        print("DYNAMIC CHARACTER TRACKING RESULTS")
        print("="*60)
        
        print(f"\nSUMMARY:")
        print(f"   Ambiguous characters: {len(self.ambiguous_characters)}")
        print(f"   Named characters: {len(self.named_characters)}")
        
        print(f"\nAMBIGUOUS CHARACTERS:")
        for amb_id, char_data in self.ambiguous_characters.items():
            print(f"   '{amb_id}':")
            print(f"      Chapters: {char_data.chapter_appearances}")
            print(f"      Descriptions: {char_data.descriptions}")
            print(f"      Actions: {char_data.actions}")
            print(f"      Confidence: {char_data.confidence_score:.2f}")
        
        print(f"\nNAMED CHARACTERS:")
        for name, char_data in self.named_characters.items():
            print(f"   '{name}':")
            print(f"      Chapters: {char_data['chapter_appearances']}")
            print(f"      Descriptions: {char_data['descriptions']}")
            print(f"      Actions: {char_data['actions'][:3]}...")  # First 3 actions
            print(f"      Confidence: {char_data['confidence']:.2f}")
            
            if char_data.get('resolved_from_ambiguous'):
                print(f"      >> Resolved from: {char_data.get('ambiguous_aliases', [])}")
                print(f"      >> First appearance: Chapter {char_data.get('first_ambiguous_appearance', 'Unknown')}")

async def main():
    """Test the dynamic character tracking system"""
    
    print("Testing Dynamic Character Tracking on First 3 Chapters")
    print("="*60)
    
    tracker = DynamicCharacterTracker()
    
    # Get first 3 chapters
    chapters = await tracker.get_first_n_chapters(3)
    
    if not chapters:
        print("No chapters found in Qdrant")
        return
    
    print(f"Found {len(chapters)} chapter segments")
    
    # Process chapters
    await tracker.process_chapters(chapters)
    
    # Print results
    tracker.print_results()
    
    # Close connections
    if tracker.deepseek_client.session:
        await tracker.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())