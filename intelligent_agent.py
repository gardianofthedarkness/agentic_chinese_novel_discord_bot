#!/usr/bin/env python3
"""
Intelligent Literary Agent System
DeepSeek-powered agent that dynamically discovers characters, builds storylines,
and provides context-aware responses using RAG + AI reasoning
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from simple_rag import create_rag_client
from deepseek_integration import DeepSeekClient, create_deepseek_config

logger = logging.getLogger(__name__)

class AgentTask(Enum):
    CHARACTER_DISCOVERY = "character_discovery"
    STORYLINE_ANALYSIS = "storyline_analysis"
    TIMELINE_BUILDING = "timeline_building"
    CONTEXTUAL_CHAT = "contextual_chat"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"

@dataclass
class Character:
    name: str
    description: str
    personality_traits: List[str]
    relationships: Dict[str, str]
    key_scenes: List[str]
    development_arc: str
    discovered_at: datetime
    confidence: float

@dataclass
class StorylineElement:
    id: str
    title: str
    summary: str
    characters_involved: List[str]
    key_events: List[str]
    timeline_position: int
    themes: List[str]
    discovered_at: datetime

@dataclass
class AgentMemory:
    characters: Dict[str, Character]
    storylines: List[StorylineElement]
    timeline: List[Dict[str, Any]]
    knowledge_graph: Dict[str, List[str]]
    last_updated: datetime

class IntelligentLiteraryAgent:
    """Advanced agent that uses DeepSeek + RAG for dynamic literary analysis"""
    
    def __init__(self, qdrant_url: str = "http://localhost:32768", collection: str = "test_novel2"):
        self.rag_client = create_rag_client(qdrant_url)
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        self.collection = collection
        
        # Agent memory
        self.memory = AgentMemory(
            characters={},
            storylines=[],
            timeline=[],
            knowledge_graph={},
            last_updated=datetime.now()
        )
        
        # Agent prompts for different tasks
        self.prompts = {
            AgentTask.CHARACTER_DISCOVERY: """你是一位专业的中文文学分析专家。请仔细分析以下文本内容，识别和分析其中的人物角色。

请以JSON格式返回分析结果，包含：
- 人物姓名
- 人物描述和背景
- 性格特征（至少3个）
- 与其他角色的关系
- 关键场景和事件
- 人物发展轨迹

文本内容：
{content}

请提供详细且准确的分析：""",

            AgentTask.STORYLINE_ANALYSIS: """作为中文文学专家，请分析以下内容的故事情节结构。

请识别并分析：
- 主要故事线索
- 关键事件和转折点
- 故事发展的时间线
- 主要冲突和解决方案
- 文学主题和寓意

以结构化的方式总结故事情节：

文本内容：
{content}""",

            AgentTask.CONTEXTUAL_CHAT: """你是一位深谙中文文学的智能助手，拥有丰富的文学知识和背景信息。

当前对话背景：
{context}

相关文学内容：
{rag_content}

用户问题：{user_message}

请结合背景知识和相关内容，提供深入、有见地的回应。""",

            AgentTask.KNOWLEDGE_SYNTHESIS: """请综合分析以下信息，构建完整的知识图谱：

人物信息：{characters}
故事情节：{storylines}
相关内容：{rag_content}

请生成：
1. 人物关系网络
2. 故事时间线
3. 主题分析
4. 关键概念connections
"""
        }
    
    async def discover_characters(self, limit: int = 10) -> List[Character]:
        """Use RAG + DeepSeek to dynamically discover characters"""
        logger.info("Starting character discovery process...")
        
        try:
            # Get diverse content from RAG
            rag_results = self.rag_client.search_text("人物 角色 主角", collection=self.collection, limit=limit)
            
            discovered_characters = []
            
            for result in rag_results:
                content = result['content']
                
                # Use DeepSeek to analyze character information
                prompt = self.prompts[AgentTask.CHARACTER_DISCOVERY].format(content=content)
                
                messages = [{"role": "user", "content": prompt}]
                response = await self.deepseek_client.generate_character_response(
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.3  # Lower temperature for analytical tasks
                )
                
                if response.get("success"):
                    try:
                        # Parse AI response to extract character info
                        character_data = self._parse_character_response(response["response"])
                        if character_data:
                            discovered_characters.extend(character_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse character response: {e}")
                        continue
            
            # Update memory
            for char in discovered_characters:
                self.memory.characters[char.name] = char
            
            self.memory.last_updated = datetime.now()
            logger.info(f"Discovered {len(discovered_characters)} characters")
            
            return discovered_characters
            
        except Exception as e:
            logger.error(f"Character discovery error: {e}")
            return []
    
    async def analyze_storyline(self, limit: int = 5) -> List[StorylineElement]:
        """Analyze and build storyline using AI + RAG"""
        logger.info("Analyzing storyline...")
        
        try:
            # Get story content
            rag_results = self.rag_client.search_text("故事 情节 章节", collection=self.collection, limit=limit)
            
            storylines = []
            
            for i, result in enumerate(rag_results):
                content = result['content']
                
                prompt = self.prompts[AgentTask.STORYLINE_ANALYSIS].format(content=content)
                
                messages = [{"role": "user", "content": prompt}]
                response = await self.deepseek_client.generate_character_response(
                    messages=messages,
                    max_tokens=800,
                    temperature=0.2
                )
                
                if response.get("success"):
                    storyline = StorylineElement(
                        id=f"storyline_{i}",
                        title=f"故事线索 {i+1}",
                        summary=response["response"][:500],  # First 500 chars as summary
                        characters_involved=list(self.memory.characters.keys()),
                        key_events=[],
                        timeline_position=i,
                        themes=[],
                        discovered_at=datetime.now()
                    )
                    storylines.append(storyline)
            
            self.memory.storylines = storylines
            return storylines
            
        except Exception as e:
            logger.error(f"Storyline analysis error: {e}")
            return []
    
    async def contextual_chat(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """Intelligent chat with RAG-enhanced context"""
        logger.info(f"Processing contextual chat: {user_message[:50]}...")
        
        try:
            # Get relevant RAG content
            rag_results = self.rag_client.search_text(user_message, collection=self.collection, limit=3)
            rag_content = "\n\n".join([r['content'][:300] for r in rag_results])
            
            # Build context from memory
            context_parts = []
            
            if self.memory.characters:
                char_names = list(self.memory.characters.keys())[:5]
                context_parts.append(f"已知人物: {', '.join(char_names)}")
            
            if self.memory.storylines:
                story_count = len(self.memory.storylines)
                context_parts.append(f"故事线索数量: {story_count}")
            
            if conversation_history:
                recent_context = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
                context_parts.append(f"最近对话: {recent_context}")
            
            context = "\n".join(context_parts)
            
            # Generate contextual response
            prompt = self.prompts[AgentTask.CONTEXTUAL_CHAT].format(
                context=context,
                rag_content=rag_content,
                user_message=user_message
            )
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )
            
            if response.get("success"):
                return response["response"]
            else:
                return f"抱歉，处理您的问题时遇到了困难: {response.get('error', '未知错误')}"
                
        except Exception as e:
            logger.error(f"Contextual chat error: {e}")
            return "抱歉，系统暂时无法处理您的请求。"
    
    async def synthesize_knowledge(self) -> Dict[str, Any]:
        """Build comprehensive knowledge graph"""
        logger.info("Synthesizing knowledge...")
        
        try:
            # Get additional context
            rag_results = self.rag_client.search_text("总结 概括", collection=self.collection, limit=3)
            rag_content = "\n".join([r['content'][:200] for r in rag_results])
            
            characters_summary = {name: char.description for name, char in self.memory.characters.items()}
            storylines_summary = [s.summary[:100] for s in self.memory.storylines]
            
            prompt = self.prompts[AgentTask.KNOWLEDGE_SYNTHESIS].format(
                characters=json.dumps(characters_summary, ensure_ascii=False),
                storylines=storylines_summary,
                rag_content=rag_content
            )
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=1500,
                temperature=0.2
            )
            
            if response.get("success"):
                return {
                    "synthesis": response["response"],
                    "character_count": len(self.memory.characters),
                    "storyline_count": len(self.memory.storylines),
                    "last_updated": self.memory.last_updated.isoformat(),
                    "knowledge_graph": self.memory.knowledge_graph
                }
            else:
                return {"error": response.get("error", "Knowledge synthesis failed")}
                
        except Exception as e:
            logger.error(f"Knowledge synthesis error: {e}")
            return {"error": str(e)}
    
    def _parse_character_response(self, response: str) -> List[Character]:
        """Parse DeepSeek response to extract character information"""
        try:
            # Simple extraction - in production, use more sophisticated parsing
            characters = []
            
            # Look for character mentions in the response
            lines = response.split('\n')
            current_char = None
            
            for line in lines:
                line = line.strip()
                if '姓名' in line or '人物' in line:
                    # Extract character name
                    name = line.split('：')[-1].strip() if '：' in line else line
                    if len(name) > 0 and len(name) < 20:  # Reasonable name length
                        current_char = Character(
                            name=name,
                            description="",
                            personality_traits=[],
                            relationships={},
                            key_scenes=[],
                            development_arc="",
                            discovered_at=datetime.now(),
                            confidence=0.8
                        )
                elif current_char and ('描述' in line or '性格' in line):
                    current_char.description += line + " "
                elif current_char and line:
                    # Add other information as description
                    current_char.description += line + " "
                    
            if current_char and current_char.name:
                characters.append(current_char)
                
            return characters
            
        except Exception as e:
            logger.error(f"Character parsing error: {e}")
            return []
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get current agent memory state"""
        return {
            "characters": {name: {
                "name": char.name,
                "description": char.description[:100],
                "confidence": char.confidence,
                "discovered_at": char.discovered_at.isoformat()
            } for name, char in self.memory.characters.items()},
            "storylines": [{
                "id": s.id,
                "title": s.title,
                "summary": s.summary[:100],
                "timeline_position": s.timeline_position
            } for s in self.memory.storylines],
            "last_updated": self.memory.last_updated.isoformat(),
            "stats": {
                "character_count": len(self.memory.characters),
                "storyline_count": len(self.memory.storylines),
                "timeline_events": len(self.memory.timeline)
            }
        }

# Factory function
def create_intelligent_agent(qdrant_url: str = "http://localhost:32768", collection: str = "test_novel2") -> IntelligentLiteraryAgent:
    """Create intelligent literary agent"""
    return IntelligentLiteraryAgent(qdrant_url, collection)

# Test the agent
if __name__ == "__main__":
    async def test_agent():
        agent = create_intelligent_agent()
        
        print("Testing character discovery...")
        characters = await agent.discover_characters(limit=2)
        print(f"Found {len(characters)} characters")
        
        print("\nTesting contextual chat...")
        response = await agent.contextual_chat("这个故事的主要人物是谁？")
        print(f"Response: {response[:200]}...")
        
        print("\nMemory summary:")
        summary = agent.get_memory_summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2)[:500])
    
    asyncio.run(test_agent())