#!/usr/bin/env python3
"""
Enhanced RAG System for Character-Specific Conversation Retrieval
Extends the existing Qdrant RAG system with character-aware features
"""

import asyncio
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with character context"""
    text: str
    score: float
    coordinate: List[int]
    character_relevance: float
    context_type: str  # 'dialogue', 'narration', 'action'
    emotion_tags: List[str]


class CharacterRAGSystem:
    """Enhanced RAG system with character-specific retrieval and context awareness"""
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "test_novel2",
        model_name: str = "moka-ai/m3e-small"
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Initialize components
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.llm_client: Optional[ChatCompletionsClient] = None
        
        # Character-specific caches
        self.character_embeddings: Dict[str, np.ndarray] = {}
        self.conversation_patterns: Dict[str, List[str]] = {}
        self.context_weights: Dict[str, Dict[str, float]] = {}
    
    async def initialize(self):
        """Initialize all RAG components"""
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(self.qdrant_url)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.model_name)
        
        # Initialize Azure LLM client if available
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_api_key = os.getenv("AZURE_API_KEY")
        if azure_endpoint and azure_api_key:
            self.llm_client = ChatCompletionsClient(
                endpoint=azure_endpoint,
                credential=AzureKeyCredential(azure_api_key)
            )
        
        print("✅ Enhanced RAG system initialized")
    
    def _analyze_text_context(self, text: str) -> Tuple[str, List[str]]:
        """Analyze text to determine context type and emotion tags"""
        text_lower = text.lower()
        
        # Determine context type
        if any(marker in text for marker in ['"', '「', '」', '说', '道', '问', '答']):
            context_type = 'dialogue'
        elif any(marker in text for marker in ['描述', '看到', '听到', '感受', '环境']):
            context_type = 'narration'
        elif any(marker in text for marker in ['行动', '移动', '攻击', '使用', '施展']):
            context_type = 'action'
        else:
            context_type = 'narration'  # default
        
        # Extract emotion tags (simplified emotion detection)
        emotion_keywords = {
            'angry': ['愤怒', '生气', '愤然', '怒'],
            'happy': ['开心', '快乐', '高兴', '喜悦', '笑'],
            'sad': ['悲伤', '难过', '哭', '眼泪'],
            'surprised': ['惊讶', '震惊', '吃惊', '惊'],
            'fear': ['害怕', '恐惧', '担心', '紧张'],
            'calm': ['平静', '冷静', '淡然', '安静'],
            'excited': ['兴奋', '激动', '热情', '振奋']
        }
        
        emotion_tags = []
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                emotion_tags.append(emotion)
        
        return context_type, emotion_tags
    
    async def create_character_embedding(self, character_name: str, character_profile: Dict[str, Any]):
        """Create and cache character-specific embedding"""
        if not self.embedding_model:
            await self.initialize()
        
        # Combine character information for embedding
        profile_text = f"""
        Character: {character_name}
        Personality: {character_profile.get('personality', '')}
        Background: {character_profile.get('background', '')}
        Speech patterns: {', '.join(character_profile.get('speech_patterns', []))}
        Conversation style: {character_profile.get('conversation_style', 'casual')}
        """
        
        character_embedding = self.embedding_model.encode(
            profile_text.strip(), 
            convert_to_tensor=False
        )
        
        self.character_embeddings[character_name] = np.array(character_embedding)
        print(f"✅ Created character embedding for {character_name}")
    
    async def retrieve_character_examples(
        self,
        character_name: str,
        query: str,
        top_k: int = 5,
        context_filter: Optional[str] = None,
        emotion_filter: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve examples with character-specific scoring and filtering
        
        Args:
            character_name: Target character name
            query: Query text
            top_k: Number of results to return
            context_filter: Filter by context type ('dialogue', 'narration', 'action')
            emotion_filter: Filter by emotion tags
        """
        if not self.qdrant_client or not self.embedding_model:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
            
            # Perform initial vector search with higher limit for filtering
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k * 3,  # Get more results for filtering
                score_threshold=0.6
            )
            
            enhanced_results = []
            
            for result in search_results:
                text = result.payload.get("chunk", "")
                coordinate = result.payload.get("coordinate", [])
                base_score = result.score
                
                # Analyze text context
                context_type, emotion_tags = self._analyze_text_context(text)
                
                # Apply context filter
                if context_filter and context_type != context_filter:
                    continue
                
                # Apply emotion filter
                if emotion_filter and not any(emotion in emotion_tags for emotion in emotion_filter):
                    continue
                
                # Calculate character relevance score
                character_relevance = await self._calculate_character_relevance(
                    character_name, text
                )
                
                enhanced_result = RetrievalResult(
                    text=text,
                    score=base_score,
                    coordinate=coordinate,
                    character_relevance=character_relevance,
                    context_type=context_type,
                    emotion_tags=emotion_tags
                )
                
                enhanced_results.append(enhanced_result)
            
            # Sort by combined score (base score + character relevance)
            enhanced_results.sort(
                key=lambda x: (x.score * 0.7 + x.character_relevance * 0.3),
                reverse=True
            )
            
            return enhanced_results[:top_k]
            
        except Exception as e:
            print(f"❌ Error in character example retrieval: {e}")
            return []
    
    async def _calculate_character_relevance(self, character_name: str, text: str) -> float:
        """Calculate how relevant a text is to a specific character"""
        # Simple character name matching
        name_score = 1.0 if character_name.lower() in text.lower() else 0.0
        
        # Check for character-specific keywords if cached
        if character_name in self.conversation_patterns:
            patterns = self.conversation_patterns[character_name]
            pattern_score = sum(1 for pattern in patterns if pattern.lower() in text.lower())
            pattern_score = min(pattern_score / len(patterns), 1.0) if patterns else 0.0
        else:
            pattern_score = 0.0
        
        # Character embedding similarity (if available)
        embedding_score = 0.0
        if character_name in self.character_embeddings and self.embedding_model:
            text_embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            char_embedding = self.character_embeddings[character_name]
            
            # Cosine similarity
            similarity = np.dot(text_embedding, char_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(char_embedding)
            )
            embedding_score = max(0, similarity)  # Ensure non-negative
        
        # Combine scores
        final_score = (name_score * 0.4 + pattern_score * 0.3 + embedding_score * 0.3)
        return min(final_score, 1.0)
    
    async def generate_hypothetical_document(
        self,
        character_name: str,
        query: str,
        context_examples: List[RetrievalResult]
    ) -> str:
        """
        Generate character-specific hypothetical document using HyDE approach
        """
        if not self.llm_client:
            return query  # Fallback to original query
        
        # Build context from examples
        example_texts = []
        for example in context_examples[:3]:  # Use top 3 examples
            example_texts.append(f"Context: {example.text[:200]}...")
        
        context_section = "\n".join(example_texts) if example_texts else ""
        
        # Enhanced character-aware HyDE prompt
        hyde_prompt = f"""你是{character_name}的角色模拟专家。基于以下背景信息，生成一段{character_name}可能会说的话或参与的对话场景。

角色查询：{query}

参考背景片段：
{context_section}

请生成一段自然的对话或场景描述，体现{character_name}的说话风格和性格特点。保持角色一致性和语言风格的连贯性。

生成的内容应该：
1. 符合角色的性格特征
2. 使用角色典型的语言模式
3. 在给定的情境下合理自然
4. 长度适中，包含足够的细节用于语义检索

请直接输出模拟的对话或场景："""
        
        try:
            response = self.llm_client.complete(
                messages=[
                    {"role": "system", "content": "你是一位专业的角色模拟助手，擅长根据角色特征生成一致的对话内容。"},
                    {"role": "user", "content": hyde_prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            print(f"🧠 Generated hypothetical document for {character_name}")
            return hypothetical_doc
            
        except Exception as e:
            print(f"❌ Error generating hypothetical document: {e}")
            return query
    
    async def enhanced_character_retrieval(
        self,
        character_name: str,
        query: str,
        character_profile: Dict[str, Any],
        top_k: int = 5,
        use_hyde: bool = True,
        context_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform enhanced character-specific retrieval with HyDE
        
        Args:
            character_name: Target character
            query: User query
            character_profile: Character configuration
            top_k: Number of results
            use_hyde: Whether to use hypothetical document enhancement
            context_filter: Filter by context type
        """
        # Ensure character embedding exists
        if character_name not in self.character_embeddings:
            await self.create_character_embedding(character_name, character_profile)
        
        # Step 1: Initial retrieval for context
        initial_results = await self.retrieve_character_examples(
            character_name=character_name,
            query=query,
            top_k=3,
            context_filter=context_filter
        )
        
        # Step 2: Generate hypothetical document if enabled
        if use_hyde and initial_results:
            hypothetical_doc = await self.generate_hypothetical_document(
                character_name, query, initial_results
            )
            
            # Step 3: Retrieve using hypothetical document
            final_results = await self.retrieve_character_examples(
                character_name=character_name,
                query=hypothetical_doc,
                top_k=top_k,
                context_filter=context_filter
            )
        else:
            final_results = initial_results[:top_k]
        
        # Step 4: Add neighbor context (from your existing approach)
        enhanced_results = await self._add_neighbor_context(final_results)
        
        return enhanced_results
    
    async def _add_neighbor_context(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Add neighboring chunks for better context (adapted from your existing code)"""
        if not self.qdrant_client:
            return results
        
        try:
            # Get all points for coordinate mapping
            points, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=10000
            )
            
            coord_to_chunk = {}
            max_chunk_per_row = {}
            
            for point in points:
                coord = point.payload.get("coordinate")
                chunk = point.payload.get("chunk")
                
                if isinstance(coord, list) and len(coord) == 2:
                    row_id, chunk_id = coord
                    coord_tuple = (row_id, chunk_id)
                    coord_to_chunk[coord_tuple] = chunk
                    
                    if row_id not in max_chunk_per_row:
                        max_chunk_per_row[row_id] = chunk_id
                    else:
                        max_chunk_per_row[row_id] = max(max_chunk_per_row[row_id], chunk_id)
            
            # Enhance results with neighbor context
            enhanced_results = []
            
            for result in results:
                if len(result.coordinate) == 2:
                    row_id, chunk_id = result.coordinate
                    max_chunk_id = max_chunk_per_row.get(row_id, chunk_id)
                    
                    # Collect neighboring chunks
                    parts = []
                    
                    # Previous chunk
                    if chunk_id > 0:
                        prev_chunk = coord_to_chunk.get((row_id, chunk_id - 1))
                        if prev_chunk:
                            parts.append(prev_chunk)
                    
                    # Current chunk
                    parts.append(result.text)
                    
                    # Next chunk
                    if chunk_id < max_chunk_id:
                        next_chunk = coord_to_chunk.get((row_id, chunk_id + 1))
                        if next_chunk:
                            parts.append(next_chunk)
                    
                    # Create enhanced result with joined context
                    enhanced_result = RetrievalResult(
                        text="\n".join(parts).strip(),
                        score=result.score,
                        coordinate=result.coordinate,
                        character_relevance=result.character_relevance,
                        context_type=result.context_type,
                        emotion_tags=result.emotion_tags
                    )
                    
                    enhanced_results.append(enhanced_result)
                else:
                    enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            print(f"❌ Error adding neighbor context: {e}")
            return results
    
    def update_character_patterns(self, character_name: str, new_conversations: List[str]):
        """Update character conversation patterns based on new interactions"""
        if character_name not in self.conversation_patterns:
            self.conversation_patterns[character_name] = []
        
        # Extract patterns from conversations (simplified)
        patterns = []
        for conv in new_conversations:
            # Extract frequent phrases, sentence structures, etc.
            words = conv.lower().split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 5:  # Avoid very short patterns
                    patterns.append(bigram)
        
        # Add unique patterns
        existing = set(self.conversation_patterns[character_name])
        new_patterns = [p for p in patterns if p not in existing]
        self.conversation_patterns[character_name].extend(new_patterns[:10])  # Limit growth
        
        print(f"📝 Updated conversation patterns for {character_name}: +{len(new_patterns)} patterns")


# Utility functions for integration with existing system

async def migrate_existing_data_for_characters():
    """Migrate existing RAG data to support character-specific features"""
    print("🚀 Starting data migration for character support...")
    
    # This would analyze existing chunks and tag them with character information
    # For now, this is a placeholder for the migration logic
    
    print("✅ Data migration completed")


def create_character_rag_config(character_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Create RAG configuration optimized for a specific character"""
    return {
        "retrieval_strategy": "hybrid",  # Vector + keyword
        "context_window": character_profile.get("memory_limit", 10) * 2,
        "temperature": character_profile.get("temperature", 0.7),
        "top_k": 5,
        "score_threshold": 0.6,
        "use_hyde": True,
        "neighbor_context": True,
        "emotion_aware": True
    }


# Example usage and testing
async def test_enhanced_rag():
    """Test the enhanced RAG system"""
    rag_system = CharacterRAGSystem()
    await rag_system.initialize()
    
    # Test character profile
    test_character = {
        "name": "测试角色",
        "personality": "聪明、友善、好奇心强的角色",
        "speech_patterns": ["你好", "很有趣", "让我想想"],
        "conversation_style": "casual",
        "background": "来自现代都市的年轻人"
    }
    
    # Test retrieval
    results = await rag_system.enhanced_character_retrieval(
        character_name="测试角色",
        query="如何使用魔法",
        character_profile=test_character,
        top_k=3
    )
    
    print(f"📋 Retrieved {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.score:.3f}, Relevance: {result.character_relevance:.3f}")
        print(f"      Type: {result.context_type}, Emotions: {result.emotion_tags}")
        print(f"      Text: {result.text[:100]}...")


if __name__ == "__main__":
    asyncio.run(test_enhanced_rag())