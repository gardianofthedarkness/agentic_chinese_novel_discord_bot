#!/usr/bin/env python3
"""
Test Dynamic Character Tracking on First 3 BIG Chapters
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AmbiguousCharacter:
    """Temporary character reference before identity revealed"""
    identifier: str  
    descriptions: List[str]
    actions: List[str] 
    locations: List[str]
    relationships: List[str]
    big_chapter_appearances: List[int]  # Track big chapters
    dialogue_samples: List[str]
    confidence_score: float
    first_appearance: int

class BigChapterCharacterTracker:
    """Dynamic character tracking across big chapters"""
    
    def __init__(self, qdrant_url: str = "http://localhost:32768"):
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Storage for character evolution
        self.ambiguous_characters: Dict[str, AmbiguousCharacter] = {}
        self.named_characters: Dict[str, Dict[str, Any]] = {}
        
        # Big chapter structure
        self.big_chapters: Dict[int, List[Dict[str, Any]]] = {}  # big_chapter_num -> [points]
        
        logger.info("Big Chapter Character Tracker initialized")
    
    async def get_first_n_big_chapters(self, n: int = 3) -> Dict[int, List[Dict[str, Any]]]:
        """Get first N big chapters with all their content"""
        logger.info(f"Fetching first {n} big chapters from Qdrant...")
        
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            # Get more points to capture multiple chapters
            points = client.scroll(
                collection_name="test_novel2",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            # Group points by big chapter
            big_chapters = defaultdict(list)
            
            for point in points[0]:
                payload = point.payload
                if 'chunk' in payload:
                    content = payload['chunk']
                    
                    # Detect big chapter markers
                    big_chapter_num = self._extract_big_chapter_number(content)
                    
                    if big_chapter_num <= n and big_chapter_num > 0:
                        big_chapters[big_chapter_num].append({
                            'point_id': point.id,
                            'content': content,
                            'metadata': payload,
                            'chapter_num': big_chapter_num
                        })
            
            # Convert to regular dict and sort within each chapter
            final_chapters = {}
            for chap_num in sorted(big_chapters.keys())[:n]:
                # Sort points within chapter by point ID (roughly chronological)
                final_chapters[chap_num] = sorted(big_chapters[chap_num], key=lambda x: x['point_id'])
            
            logger.info(f"Retrieved {len(final_chapters)} big chapters:")
            for chap_num, points in final_chapters.items():
                total_chars = sum(len(p['content']) for p in points)
                logger.info(f"  Big Chapter {chap_num}: {len(points)} sections, {total_chars} total characters")
            
            return final_chapters
            
        except Exception as e:
            logger.error(f"Failed to get big chapters: {e}")
            return {}
    
    def _extract_big_chapter_number(self, content: str) -> int:
        """Extract big chapter number from content"""
        
        # Look for chapter markers in the garbled text
        patterns = [
            r'第([一二三四五六七八九十]+)章',  # 第一章, 第二章, etc.
            r'第(\d+)章',  # 第1章, 第2章, etc.
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    try:
                        # Convert Chinese numerals to numbers
                        chinese_numerals = {
                            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
                        }
                        
                        if match in chinese_numerals:
                            return chinese_numerals[match]
                        elif match.isdigit():
                            return int(match)
                    except:
                        continue
        
        return 0  # No chapter marker found
    
    def extract_character_references_from_big_chapter(self, big_chapter_content: List[Dict[str, Any]], big_chapter_num: int) -> List[Dict[str, Any]]:
        """Extract character references from all sections of a big chapter"""
        
        # Combine all content from the big chapter
        combined_content = ""
        for section in big_chapter_content:
            combined_content += section['content'] + "\n\n"
        
        logger.info(f"Big Chapter {big_chapter_num}: Processing {len(combined_content)} total characters from {len(big_chapter_content)} sections")
        
        # Character patterns optimized for garbled text
        character_patterns = [
            # Frequently repeated sequences (likely character names)
            r'([^\s]{2,6}(?=說|道|想|看|听|走|去|来|的|是|在|有|要|会|能|可|把|被|让|使|给|对|向|从|到|为|因|由|等|及|和|与|或|但|却|还|又|也|就|才|只|都|只|还|再|更|最|很|太|非|真|实|确|正|当|应|必|需|可|能|会|将|要|想|愿|希|期|求|请|问|答|说|讲|谈|告|知|明|解|释|释|证|实|确|定|决|选|择|取|采|用|做|作|为|成|变|化|改|换|调|整|修|正|完|结|束|终|止|停|留|住|待|等|候|望|看|观|察|听|闻|嗅|味|感|觉|思|考|想|念|记|忆|忘|失|得|获|取|收|接|受|给|送|传|递|交|换|买|卖|购|物|品|货|商|业|务|工|作|活|动|行|为|事|情|件|项|目|标|的|地|方|向|面|侧|边|角|点|线|圈|圆|方|正|直|弯|曲|高|低|大|小|长|短|宽|窄|厚|薄|深|浅|远|近|新|旧|老|少|年|月|日|时|分|秒|早|晚|前|后|左|右|上|下|内|外|里|中|间|际|通|过|经|历|验|试|测|查|检|验|证|明|示|显|现|出|入|进|出|来|去|回|返|归|还|复|再|重|复|反|对|比|较|相|同|异|别|区|分|类|种|型|形|状|色|彩|音|声|响|亮|清|晰|楚|明|白|黑|红|绿|蓝|黄|紫|橙|粉|灰|棕|金|银|铜|铁|钢|木|石|土|水|火|风|雷|电|光|热|冷|温|凉|暖|湿|干|净|脏|好|坏|美|丑|香|臭|甜|苦|酸|辣|咸|淡|硬|软|滑|粗|细|尖|钝|利|害|怕|惊|吓|喜|乐|悦|愉|快|慢|速|急|缓|松|紧|宽|严|格|准|确|精|细|粗|略|简|单|复|杂|难|易|轻|重|多|少|全|部|些|许|多|几|两|三|四|五|六|七|八|九|十|百|千|万|亿|第|首|末|尾|头|脚|手|足|眼|耳|鼻|口|嘴|唇|牙|齿|舌|喉|颈|肩|臂|肘|腕|指|掌|胸|背|腰|腹|腿|膝|踝|趾|心|肺|肝|胃|肾|脑|血|肉|骨|皮|毛|发|须|眉|睫|瞳|虹|膜|泪|汗|唾|痰|屎|尿|精|卵|胎|婴|儿|童|少|青|中|老|幼|男|女|雌|雄|公|母|父|母|子|女|兄|弟|姐|妹|夫|妻|祖|孙|亲|戚|友|朋|敌|仇|恨|爱|情|感|心|意|思|想|念|愿|望|希|求|需|要|欲|贪|嫉|妒|羡|慕|佩|服|敬|重|尊|贵|贱|穷|富|贫|困|苦|甜|乐|悲|哀|愁|忧|虑|怕|惧|恐|慌|急|忙|闲|闷|烦|恼|怒|气|愤|恨|厌|烦|喜|欢|乐|悦|愉|快|兴|奋|激|动|静|安|宁|平|和|顺|乱|混|杂|乖|听|话|顽|皮|淘|气|调|皮|活|泼|开|朗|内|向|外|向|热|情|冷|漠|温|柔|刚|强|坚|硬|软|弱|勇|敢|胆|小|怯|懦|聪|明|智|慧|愚|笨|傻|呆|机|灵|敏|捷|迟|钝|勤|奋|懒|惰|认|真|仔|细|马|虎|草|率|随|便|严|肃|正|经|幽|默|风|趣|诙|谐|搞|笑|可|爱|美|丽|漂|亮|英|俊|帅|气|丑|陋|难|看|高|矮|胖|瘦|胖|壮|结|实|虚|弱|健|康|强|壮|病|弱|残|缺|完|整|破|损|新|鲜|陈|旧|干|净|肮|脏|整|洁|凌|乱|有|序|无|序|规|律|随|机))',
            
            # Before common action words (garbled equivalents)
            r'([^\s]{2,5}(?=��|��|��|��|��|��|��|��))',
            
            # Repeated sequences that appear multiple times (names)
            r'([^\s]{2,4})',
        ]
        
        all_references = []
        char_frequencies = defaultdict(int)
        
        # Extract all potential character references
        for pattern in character_patterns:
            matches = re.findall(pattern, combined_content)
            for match in matches:
                if 2 <= len(match) <= 6:  # Reasonable name length
                    char_frequencies[match] += 1
        
        # Filter for frequently mentioned sequences (potential character names)
        for char_seq, frequency in char_frequencies.items():
            if frequency >= 3:  # Mentioned at least 3 times in the big chapter
                all_references.append({
                    'identifier': char_seq,
                    'big_chapter': big_chapter_num,
                    'frequency': frequency,
                    'pattern_type': 'frequent_mention'
                })
        
        # Sort by frequency (most mentioned first)
        all_references.sort(key=lambda x: x['frequency'], reverse=True)
        
        logger.info(f"Big Chapter {big_chapter_num}: Found {len(all_references)} potential character references")
        for ref in all_references[:10]:  # Show top 10
            logger.info(f"  '{ref['identifier']}' (mentioned {ref['frequency']} times)")
        
        return all_references
    
    def extract_context_features_from_big_chapter(self, content_sections: List[Dict[str, Any]], character_ref: str) -> Dict[str, List[str]]:
        """Extract contextual features for a character across the big chapter"""
        
        combined_content = ""
        for section in content_sections:
            combined_content += section['content'] + " "
        
        # Find sentences containing this character
        sentences = re.split(r'[。！？.\n]', combined_content)
        relevant_sentences = [s for s in sentences if character_ref in s]
        
        features = {
            'descriptions': [],
            'actions': [],
            'locations': [],
            'relationships': [],
            'dialogue': []
        }
        
        # Extract features from relevant sentences
        for sentence in relevant_sentences:
            # Look for descriptive patterns (even in garbled text)
            if len(sentence) > 10:  # Meaningful sentence
                # Extract potential descriptions (adjectives/descriptors)
                desc_patterns = [
                    r'(��[^\s]{1,3})',  # Garbled descriptors
                    r'([^\s]{1,3}��)',
                    r'(��[^\s]{1,3}��)',
                ]
                
                for pattern in desc_patterns:
                    matches = re.findall(pattern, sentence)
                    features['descriptions'].extend(matches)
        
        # Deduplicate
        for key in features:
            features[key] = list(set(features[key]))
        
        return features
    
    async def process_big_chapters(self, big_chapters: Dict[int, List[Dict[str, Any]]]):
        """Process big chapters and track character evolution"""
        
        logger.info("=== PROCESSING BIG CHAPTERS FOR CHARACTER EVOLUTION ===")
        
        for big_chapter_num in sorted(big_chapters.keys()):
            content_sections = big_chapters[big_chapter_num]
            
            logger.info(f"\nProcessing Big Chapter {big_chapter_num}...")
            logger.info(f"  Sections: {len(content_sections)}")
            
            # Extract character references from this big chapter
            references = self.extract_character_references_from_big_chapter(content_sections, big_chapter_num)
            
            for ref_data in references:
                ref_id = ref_data['identifier']
                frequency = ref_data['frequency']
                
                # Extract context features
                features = self.extract_context_features_from_big_chapter(content_sections, ref_id)
                
                # Check if this is a new character or existing one
                if ref_id not in self.ambiguous_characters and ref_id not in self.named_characters:
                    # Create new ambiguous character
                    self.ambiguous_characters[ref_id] = AmbiguousCharacter(
                        identifier=ref_id,
                        descriptions=features['descriptions'],
                        actions=features['actions'],
                        locations=features['locations'],
                        relationships=features['relationships'],
                        big_chapter_appearances=[big_chapter_num],
                        dialogue_samples=features['dialogue'],
                        confidence_score=min(0.3 + (frequency * 0.1), 0.9),  # Frequency-based confidence
                        first_appearance=big_chapter_num
                    )
                    logger.info(f"  NEW CHARACTER: '{ref_id}' (frequency: {frequency}, confidence: {self.ambiguous_characters[ref_id].confidence_score:.2f})")
                
                else:
                    # Update existing character
                    if ref_id in self.ambiguous_characters:
                        char = self.ambiguous_characters[ref_id]
                        if big_chapter_num not in char.big_chapter_appearances:
                            char.big_chapter_appearances.append(big_chapter_num)
                            char.descriptions.extend(features['descriptions'])
                            char.actions.extend(features['actions'])
                            char.locations.extend(features['locations'])
                            
                            # Boost confidence for recurring characters
                            char.confidence_score = min(char.confidence_score + 0.2, 0.95)
                            
                            logger.info(f"  UPDATED CHARACTER: '{ref_id}' (appears in chapters {char.big_chapter_appearances}, new confidence: {char.confidence_score:.2f})")
            
            # Check for character evolution triggers
            await self.check_character_evolution(big_chapter_num)
    
    async def check_character_evolution(self, current_big_chapter: int):
        """Check if any characters have evolved in importance"""
        
        evolution_triggers = []
        
        for char_id, char_data in self.ambiguous_characters.items():
            # Trigger 1: Appears in multiple big chapters
            if len(char_data.big_chapter_appearances) >= 2:
                evolution_triggers.append((char_id, 'multi_chapter_appearance', len(char_data.big_chapter_appearances)))
            
            # Trigger 2: High confidence score
            if char_data.confidence_score >= 0.7:
                evolution_triggers.append((char_id, 'high_confidence', char_data.confidence_score))
            
            # Trigger 3: Rich context (many descriptions/actions)
            context_richness = len(char_data.descriptions) + len(char_data.actions) + len(char_data.locations)
            if context_richness >= 5:
                evolution_triggers.append((char_id, 'rich_context', context_richness))
        
        if evolution_triggers:
            logger.info(f"\n⚡ CHARACTER EVOLUTION TRIGGERS in Big Chapter {current_big_chapter}:")
            for char_id, trigger_type, value in evolution_triggers:
                logger.info(f"  '{char_id}': {trigger_type} = {value}")
    
    def print_big_chapter_results(self):
        """Print character tracking results across big chapters"""
        
        print("\n" + "="*60)
        print("BIG CHAPTER CHARACTER EVOLUTION RESULTS")
        print("="*60)
        
        print(f"\nSUMMARY:")
        print(f"   Ambiguous characters tracked: {len(self.ambiguous_characters)}")
        print(f"   Named characters: {len(self.named_characters)}")
        
        # Sort characters by importance (confidence + chapter span)
        sorted_characters = sorted(
            self.ambiguous_characters.items(),
            key=lambda x: (len(x[1].big_chapter_appearances), x[1].confidence_score),
            reverse=True
        )
        
        print(f"\nCHARACTER EVOLUTION TRACKING:")
        for char_id, char_data in sorted_characters[:15]:  # Top 15 most important
            chapter_span = len(char_data.big_chapter_appearances)
            print(f"\n  '{char_id}':")
            print(f"    Big Chapters: {char_data.big_chapter_appearances}")
            print(f"    Chapter Span: {chapter_span} chapters")
            print(f"    Confidence: {char_data.confidence_score:.2f}")
            print(f"    Descriptions: {char_data.descriptions[:3]}...")
            print(f"    First Appearance: Big Chapter {char_data.first_appearance}")
            
            # Highlight potentially important characters
            if chapter_span >= 2 and char_data.confidence_score >= 0.6:
                print(f"    >>> POTENTIALLY IMPORTANT CHARACTER <<<")
            elif chapter_span >= 3:
                print(f"    >>> MAJOR CHARACTER (appears across multiple chapters) <<<")

async def main():
    """Test big chapter character tracking"""
    
    print("Testing Dynamic Character Tracking on First 3 BIG Chapters")
    print("="*60)
    
    tracker = BigChapterCharacterTracker()
    
    # Get first 3 big chapters
    big_chapters = await tracker.get_first_n_big_chapters(3)
    
    if not big_chapters:
        print("No big chapters found in Qdrant")
        return
    
    # Process big chapters
    await tracker.process_big_chapters(big_chapters)
    
    # Print results
    tracker.print_big_chapter_results()
    
    # Close connections
    if tracker.deepseek_client.session:
        await tracker.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())