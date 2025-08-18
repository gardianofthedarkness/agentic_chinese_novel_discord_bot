#!/usr/bin/env python3
"""
DeepSeek API Integration for Chinese Literature Character Simulation
DeepSeek is currently the best Chinese literature model
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DeepSeekConfig:
    """DeepSeek API configuration"""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"  # or "deepseek-coder" for coding tasks
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

class DeepSeekClient:
    """DeepSeek API client for Chinese literature generation"""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        # Configure timeout to avoid Flask asyncio compatibility issues
        timeout = aiohttp.ClientTimeout(total=None)  # Disable timeout
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def generate_character_response(self, messages: List[Dict[str, str]], 
                                        character_context: Dict[str, Any] = None,
                                        **kwargs) -> Dict[str, Any]:
        """Generate character response using DeepSeek"""
        
        if not self.session:
            await self.initialize()
        
        # Prepare request
        request_data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": False
        }
        
        try:
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=request_data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "model": result.get("model", self.config.model)
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    async def generate_character_analysis(self, character_name: str, 
                                        personality: str, recent_events: List[str],
                                        query: str) -> Dict[str, Any]:
        """Generate deep character analysis using DeepSeek's Chinese literature expertise"""
        
        analysis_prompt = f"""你是一位专业的中文小说角色分析专家，请分析以下角色的心理状态和可能的反应。

角色信息：
- 姓名：{character_name}
- 性格特征：{personality}

最近发生的事件：
{chr(10).join(recent_events)}

当前情况：{query}

请从以下几个维度进行深度分析：

1. 心理状态分析：
   - 当前的情绪状态
   - 内心的冲突和挣扎
   - 潜在的担忧或期待

2. 行为动机分析：
   - 驱动角色行动的核心动机
   - 短期和长期目标
   - 价值观对行为的影响

3. 关系网络影响：
   - 重要人际关系对决策的影响
   - 社会责任感和个人需求的平衡

4. 预期反应：
   - 最可能的情绪反应
   - 典型的行为模式
   - 语言表达风格

5. 性格发展：
   - 这个情况对角色成长的意义
   - 可能的性格变化方向

请用中文回答，保持专业和深度的分析。"""

        messages = [
            {
                "role": "system",
                "content": "你是一位精通中文文学和角色心理分析的专家，擅长深度解析小说角色的内心世界和行为动机。"
            },
            {
                "role": "user", 
                "content": analysis_prompt
            }
        ]
        
        return await self.generate_character_response(messages, temperature=0.3)
    
    async def generate_roleplay_response(self, character_name: str,
                                       character_profile: Dict[str, Any],
                                       conversation_history: List[Dict[str, str]],
                                       user_message: str,
                                       rag_context: List[str] = None) -> Dict[str, Any]:
        """Generate roleplay response with character consistency"""
        
        # Build character context
        personality = character_profile.get("personality", "")
        speech_patterns = character_profile.get("speech_patterns", [])
        background = character_profile.get("background", "")
        current_emotions = character_profile.get("current_emotions", {})
        recent_goals = character_profile.get("current_goals", [])
        
        # Prepare system prompt
        system_prompt = f"""你现在要扮演角色：{character_name}

角色设定：
性格特征：{personality}
背景信息：{background}
语言风格：{', '.join(speech_patterns)}
当前情绪状态：{json.dumps(current_emotions, ensure_ascii=False)}
当前目标：{', '.join(recent_goals)}

扮演要求：
1. 完全以{character_name}的身份和语气回应
2. 保持角色的性格一致性和情绪连贯性
3. 体现角色的语言习惯和表达方式
4. 考虑角色的当前情绪和目标对回应的影响
5. 回应要自然、生动，符合中文小说的文学风格

注意：不要在回应中提及你是AI或在扮演角色，完全沉浸在角色中。"""

        # Add RAG context if available
        if rag_context:
            system_prompt += f"\n\n相关背景信息：\n{chr(10).join(rag_context)}"
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 10 messages)
        for msg in conversation_history[-10:]:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return await self.generate_character_response(messages, temperature=0.8)
    
    async def generate_story_continuation(self, story_context: str,
                                        character_perspectives: List[Dict[str, str]],
                                        desired_outcome: str = None) -> Dict[str, Any]:
        """Generate story continuation with multiple character perspectives"""
        
        continuation_prompt = f"""基于以下故事背景，请继续创作故事片段：

故事背景：
{story_context}

角色视角：
{chr(10).join([f"{char['name']}: {char['perspective']}" for char in character_perspectives])}

"""
        
        if desired_outcome:
            continuation_prompt += f"期望的故事发展方向：{desired_outcome}\n"
        
        continuation_prompt += """请创作一个500-800字的故事片段，要求：
1. 保持各角色的性格一致性
2. 推进故事情节发展
3. 体现角色间的互动和冲突
4. 使用生动的中文文学表达
5. 保持悬念和吸引力"""

        messages = [
            {
                "role": "system",
                "content": "你是一位专业的中文小说作家，擅长创作引人入胜的故事情节和生动的角色对话。"
            },
            {
                "role": "user",
                "content": continuation_prompt
            }
        ]
        
        return await self.generate_character_response(messages, temperature=0.9)

class EnhancedCharacterEngine:
    """Enhanced character engine with DeepSeek integration"""
    
    def __init__(self, deepseek_config: DeepSeekConfig):
        self.deepseek_client = DeepSeekClient(deepseek_config)
        self.characters = {}
        self.conversation_histories = {}
    
    async def initialize(self):
        """Initialize the engine"""
        await self.deepseek_client.initialize()
    
    async def close(self):
        """Close the engine"""
        await self.deepseek_client.close()
    
    async def generate_enhanced_response(self, character_name: str,
                                       user_message: str,
                                       rag_examples: List[str] = None,
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate enhanced character response with DeepSeek"""
        
        if character_name not in self.characters:
            return {"error": f"Character {character_name} not found"}
        
        character_profile = self.characters[character_name]
        conversation_history = self.conversation_histories.get(character_name, [])
        
        # Use DeepSeek for response generation
        result = await self.deepseek_client.generate_roleplay_response(
            character_name=character_name,
            character_profile=character_profile,
            conversation_history=conversation_history,
            user_message=user_message,
            rag_context=rag_examples
        )
        
        if result["success"]:
            # Update conversation history
            if character_name not in self.conversation_histories:
                self.conversation_histories[character_name] = []
            
            self.conversation_histories[character_name].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result["response"]}
            ])
            
            # Keep only recent history
            if len(self.conversation_histories[character_name]) > 20:
                self.conversation_histories[character_name] = \
                    self.conversation_histories[character_name][-20:]
            
            return {
                "success": True,
                "character_name": character_name,
                "response": result["response"],
                "model_used": "deepseek",
                "rag_enhanced": bool(rag_examples),
                "usage": result.get("usage", {})
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    
    async def analyze_character_psychology(self, character_name: str,
                                         recent_events: List[str],
                                         current_situation: str) -> Dict[str, Any]:
        """Analyze character psychology using DeepSeek"""
        
        if character_name not in self.characters:
            return {"error": f"Character {character_name} not found"}
        
        character_profile = self.characters[character_name]
        personality = character_profile.get("personality", "")
        
        result = await self.deepseek_client.generate_character_analysis(
            character_name=character_name,
            personality=personality,
            recent_events=recent_events,
            query=current_situation
        )
        
        return result
    
    def add_character(self, name: str, profile: Dict[str, Any]):
        """Add a character to the engine"""
        self.characters[name] = profile
        self.conversation_histories[name] = []

# Configuration and setup
def create_deepseek_config() -> DeepSeekConfig:
    """Create DeepSeek configuration"""
    
    # Try to get API key from environment or config
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        # For demo purposes, use a placeholder
        api_key = "your_deepseek_api_key_here"
        print("Warning: No DeepSeek API key found. Using placeholder.")
        print("Set DEEPSEEK_API_KEY environment variable for production use.")
    
    return DeepSeekConfig(
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2048
    )

# Demo and testing
async def demo_deepseek_integration():
    """Demonstrate DeepSeek integration"""
    print("DeepSeek Integration Demo")
    print("=" * 40)
    
    # Create configuration
    config = create_deepseek_config()
    
    # Initialize enhanced engine
    engine = EnhancedCharacterEngine(config)
    await engine.initialize()
    
    try:
        # Add test character
        engine.add_character("神裂火织", {
            "personality": "严肃、强大、责任感强、保护他人的女魔法师",
            "speech_patterns": ["以坚定的语气说道", "握紧七天七刀", "为了保护重要的人"],
            "background": "伦敦清教的女教皇，拥有强大的剑术和魔法能力",
            "current_emotions": {"determination": 0.8, "concern": 0.6, "confidence": 0.7},
            "current_goals": ["保护无辜的人", "维护正义", "提升实力"]
        })
        
        # Test response generation (this would work with a real API key)
        print("\n1. Testing Character Response Generation")
        print("-" * 35)
        
        if config.api_key != "your_deepseek_api_key_here":
            response = await engine.generate_enhanced_response(
                character_name="神裂火织",
                user_message="面对这样的强敌，你有什么计划吗？",
                rag_examples=[
                    "神裂火织高速挥舞着七天七刀，展现出惊人的剑术技巧",
                    "她带着凶神恶煞般的凄绝气势，一刀刀斩断攻击"
                ]
            )
            
            if response["success"]:
                print(f"Character: 神裂火织")
                print(f"Response: {response['response']}")
                print(f"Model: {response['model_used']}")
            else:
                print(f"Error: {response['error']}")
        else:
            print("Demo mode: DeepSeek API key needed for actual generation")
            print("Response would be generated using DeepSeek's Chinese literature expertise")
        
        # Test character analysis
        print("\n2. Testing Character Psychology Analysis")
        print("-" * 35)
        
        if config.api_key != "your_deepseek_api_key_here":
            analysis = await engine.analyze_character_psychology(
                character_name="神裂火织",
                recent_events=[
                    "与强大敌人激烈战斗",
                    "成功保护了重要的同伴",
                    "发现敌人的真正目的"
                ],
                current_situation="需要做出重要的战略决定"
            )
            
            if analysis["success"]:
                print("Psychology Analysis:")
                print(analysis["response"])
            else:
                print(f"Analysis Error: {analysis['error']}")
        else:
            print("Demo mode: Would provide deep psychological analysis")
            print("Including emotional state, motivations, and behavioral predictions")
        
    finally:
        await engine.close()
    
    print("\nDeepSeek integration demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_deepseek_integration())