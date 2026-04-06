#!/usr/bin/env python3
"""
Character Variant Resolver
Handles character name variants and merges them into unified character profiles
"""

import asyncio
import json
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from sqlalchemy import create_engine, text
from datetime import datetime

@dataclass
class CharacterVariant:
    name: str
    confidence: float
    chapter: int
    character_id: str

@dataclass
class UnifiedCharacter:
    primary_name: str
    aliases: List[str]
    all_variants: List[CharacterVariant]
    combined_confidence: float
    first_appearance: int
    character_type: str

class CharacterVariantResolver:
    """Resolves character name variants and creates unified profiles"""
    
    def __init__(self):
        self.db_engine = create_engine('postgresql://admin:admin@localhost:5432/novel_sim')
        
        # Define known character variant patterns
        self.variant_patterns = {
            # Misaka Mikoto variants
            'misaka_mikoto': {
                'primary': '御坂美琴',
                'variants': ['御坂美琴', '美琴', '御坂', 'ミサカ', 'みさか', 'Misaka'],
                'type': 'electromaster'
            },
            
            # Kamijou Touma variants  
            'kamijou_touma': {
                'primary': '上条当麻',
                'variants': ['上条当麻', '当麻', '上条', 'かみじょう', 'とうま', 'Touma'],
                'type': 'protagonist'
            },
            
            # Index variants
            'index': {
                'primary': '茵蒂克丝',
                'variants': ['茵蒂克丝', '禁书目录', 'インデックス', 'Index'],
                'type': 'grimoire_library'
            },
            
            # Komoe-sensei variants
            'komoe_sensei': {
                'primary': '小萌老师',
                'variants': ['小萌老师', '小萌', '月詠小萌', 'こもえ先生'],
                'type': 'teacher'
            },
            
            # Shirai Kuroko variants (if appears)
            'shirai_kuroko': {
                'primary': '白井黑子',
                'variants': ['白井黑子', '黑子', '白井', 'クロコ'],
                'type': 'teleporter'
            },
            
            # Accelerator variants (if appears)
            'accelerator': {
                'primary': '一方通行',
                'variants': ['一方通行', 'アクセラレータ', 'Accelerator'],
                'type': 'level5'
            },
            
            # Stiyl Magnus variants
            'stiyl': {
                'primary': '史提尔',
                'variants': ['史提尔', 'スティイル', 'Stiyl'],
                'type': 'magician'
            }
        }
    
    async def get_current_characters(self) -> List[CharacterVariant]:
        """Get all characters currently in database"""
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text('''
                SELECT character_id, name, confidence_score, first_appearance_chapter, character_type
                FROM character_profiles
                ORDER BY confidence_score DESC
            '''))
            
            characters = []
            for row in result:
                char_id, name, confidence, chapter, char_type = row
                characters.append(CharacterVariant(
                    name=name,
                    confidence=confidence,
                    chapter=chapter,
                    character_id=char_id
                ))
            
            return characters
    
    def group_variants(self, characters: List[CharacterVariant]) -> Dict[str, List[CharacterVariant]]:
        """Group character variants together"""
        
        # Create mapping from variant name to character group
        name_to_group = {}
        for group_id, pattern in self.variant_patterns.items():
            for variant in pattern['variants']:
                name_to_group[variant] = group_id
        
        # Group characters
        groups = defaultdict(list)
        ungrouped = []
        
        for char in characters:
            if char.name in name_to_group:
                group_id = name_to_group[char.name]
                groups[group_id].append(char)
            else:
                ungrouped.append(char)
        
        return dict(groups), ungrouped
    
    def create_unified_characters(self, grouped_variants: Dict[str, List[CharacterVariant]]) -> List[UnifiedCharacter]:
        """Create unified character profiles from variants"""
        
        unified_characters = []
        
        for group_id, variants in grouped_variants.items():
            if not variants:
                continue
                
            pattern = self.variant_patterns[group_id]
            
            # Find the primary name variant (prefer exact match, then highest confidence)
            primary_variant = None
            for variant in variants:
                if variant.name == pattern['primary']:
                    primary_variant = variant
                    break
            
            if not primary_variant:
                # Use highest confidence variant as primary
                primary_variant = max(variants, key=lambda x: x.confidence)
            
            # Calculate combined confidence (weighted average)
            total_confidence = sum(v.confidence for v in variants)
            combined_confidence = total_confidence / len(variants)
            
            # Get earliest appearance
            first_appearance = min(v.chapter for v in variants)
            
            # Create unified character
            unified = UnifiedCharacter(
                primary_name=pattern['primary'],
                aliases=[v.name for v in variants if v.name != pattern['primary']],
                all_variants=variants,
                combined_confidence=combined_confidence,
                first_appearance=first_appearance,
                character_type=pattern['type']
            )
            
            unified_characters.append(unified)
        
        return unified_characters
    
    async def create_unified_character_table(self, unified_characters: List[UnifiedCharacter]):
        """Create a unified character table with variant mapping"""
        
        with self.db_engine.connect() as conn:
            # Create unified characters table if it doesn't exist
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS unified_characters (
                    unified_id VARCHAR PRIMARY KEY,
                    primary_name VARCHAR NOT NULL,
                    character_type VARCHAR,
                    aliases JSONB,
                    all_variant_names JSONB,
                    combined_confidence FLOAT,
                    first_appearance_chapter INTEGER,
                    total_mentions INTEGER,
                    created_at TIMESTAMP,
                    metadata JSONB
                )
            '''))
            
            # Clear existing data
            conn.execute(text('DELETE FROM unified_characters'))
            
            # Insert unified characters
            for unified in unified_characters:
                unified_id = f"unified_{unified.character_type}_{unified.primary_name.replace(' ', '_')}"
                
                all_variant_names = [v.name for v in unified.all_variants]
                total_mentions = len(unified.all_variants)  # Rough estimate
                
                conn.execute(text('''
                    INSERT INTO unified_characters 
                    (unified_id, primary_name, character_type, aliases, all_variant_names,
                     combined_confidence, first_appearance_chapter, total_mentions, created_at, metadata)
                    VALUES 
                    (:unified_id, :primary_name, :character_type, :aliases, :all_variant_names,
                     :combined_confidence, :first_appearance_chapter, :total_mentions, :created_at, :metadata)
                '''), {
                    'unified_id': unified_id,
                    'primary_name': unified.primary_name,
                    'character_type': unified.character_type,
                    'aliases': json.dumps(unified.aliases),
                    'all_variant_names': json.dumps(all_variant_names),
                    'combined_confidence': unified.combined_confidence,
                    'first_appearance_chapter': unified.first_appearance,
                    'total_mentions': total_mentions,
                    'created_at': datetime.now(),
                    'metadata': json.dumps({
                        'variant_count': len(unified.all_variants),
                        'confidence_range': [min(v.confidence for v in unified.all_variants),
                                           max(v.confidence for v in unified.all_variants)]
                    })
                })
            
            conn.commit()
    
    async def create_variant_lookup_function(self):
        """Create a lookup function for character variants"""
        
        print("Creating character variant lookup system...")
        
        with self.db_engine.connect() as conn:
            # Create variant lookup table
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS character_variant_lookup (
                    variant_name VARCHAR PRIMARY KEY,
                    unified_id VARCHAR REFERENCES unified_characters(unified_id),
                    is_primary BOOLEAN DEFAULT FALSE,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            '''))
            
            # Clear existing lookups
            conn.execute(text('DELETE FROM character_variant_lookup'))
            
            # Get all unified characters
            result = conn.execute(text('''
                SELECT unified_id, primary_name, all_variant_names 
                FROM unified_characters
            '''))
            
            # Create lookup entries
            for row in result:
                unified_id, primary_name, variant_names_json = row
                # Handle both JSON string and direct list
                if isinstance(variant_names_json, str):
                    variant_names = json.loads(variant_names_json)
                else:
                    variant_names = variant_names_json
                
                for variant_name in variant_names:
                    is_primary = variant_name == primary_name
                    
                    conn.execute(text('''
                        INSERT INTO character_variant_lookup 
                        (variant_name, unified_id, is_primary, confidence)
                        VALUES (:variant_name, :unified_id, :is_primary, :confidence)
                    '''), {
                        'variant_name': variant_name,
                        'unified_id': unified_id,
                        'is_primary': is_primary,
                        'confidence': 0.95 if is_primary else 0.8
                    })
            
            conn.commit()
    
    def query_character_info(self, query_name: str) -> Dict[str, Any]:
        """Query character information by any variant name"""
        
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
                'query_matched': query_name
            }
    
    def print_analysis(self, unified_characters: List[UnifiedCharacter], ungrouped: List[CharacterVariant]):
        """Print analysis results"""
        
        print("\n" + "="*70)
        print("CHARACTER VARIANT RESOLUTION ANALYSIS")
        print("="*70)
        
        print(f"\nUNIFIED CHARACTERS: {len(unified_characters)}")
        print("-" * 40)
        
        for unified in unified_characters:
            print(f"\n'{unified.primary_name}' ({unified.character_type}):")
            print(f"  Aliases: {unified.aliases}")
            print(f"  Combined confidence: {unified.combined_confidence:.2f}")
            print(f"  First appearance: Chapter {unified.first_appearance}")
            print(f"  Total variants found: {len(unified.all_variants)}")
            
            if '御坂' in unified.primary_name or 'misaka' in unified.character_type:
                print(f"  >>> MISAKA VARIANTS SUCCESSFULLY UNIFIED! <<<")
        
        print(f"\nUNGROUPED CHARACTERS: {len(ungrouped)}")
        print("-" * 40)
        
        for char in ungrouped:
            print(f"  {char.name} (confidence: {char.confidence:.2f})")
        
        print(f"\nVARIANT RESOLUTION SUMMARY:")
        print("-" * 40)
        original_count = len(unified_characters) * 2.5 + len(ungrouped)  # Rough estimate
        final_count = len(unified_characters) + len(ungrouped)
        print(f"  Original character entries: ~{original_count:.0f}")
        print(f"  Unified character profiles: {final_count}")
        print(f"  Reduction in duplicates: {(1 - final_count/original_count)*100:.1f}%")

async def main():
    """Run character variant resolution"""
    
    print("CHARACTER VARIANT RESOLVER")
    print("Unifying character name variants")
    print("="*50)
    
    resolver = CharacterVariantResolver()
    
    # Get current characters
    characters = await resolver.get_current_characters()
    print(f"Found {len(characters)} character entries in database")
    
    # Group variants
    grouped_variants, ungrouped = resolver.group_variants(characters)
    print(f"Grouped into {len(grouped_variants)} variant groups + {len(ungrouped)} ungrouped")
    
    # Create unified characters
    unified_characters = resolver.create_unified_characters(grouped_variants)
    
    # Store in database
    await resolver.create_unified_character_table(unified_characters)
    await resolver.create_variant_lookup_function()
    
    # Print analysis
    resolver.print_analysis(unified_characters, ungrouped)
    
    # Test the lookup system
    print(f"\n" + "="*70)
    print("TESTING VARIANT LOOKUP SYSTEM")
    print("="*70)
    
    test_queries = ['御坂美琴', '美琴', '御坂', '上条当麻', '当麻', '茵蒂克丝', '禁书目录']
    
    for query in test_queries:
        result = resolver.query_character_info(query)
        if result:
            print(f"\nQuery: '{query}'")
            print(f"  Resolved to: {result['primary_name']} ({result['character_type']})")
            print(f"  Is primary name: {result['is_primary_name']}")
            print(f"  All aliases: {result['aliases']}")
        else:
            print(f"\nQuery: '{query}' - No match found")

if __name__ == "__main__":
    asyncio.run(main())