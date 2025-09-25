#!/usr/bin/env python3
"""
Personality Engine for Character Consistency and Context Management
Handles personality consistency, emotional states, and contextual response generation
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer

from character_config import CharacterProfile, ConversationMemory, ConversationMessage


class EmotionState(Enum):
    """Predefined emotional states"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    CONFUSED = "confused"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"


@dataclass
class ContextualState:
    """Current contextual state of a character"""
    current_emotion: EmotionState
    emotion_intensity: float  # 0.0 to 1.0
    energy_level: float  # 0.0 to 1.0
    topic_focus: Optional[str]
    relationship_context: Dict[str, str]
    recent_events: List[str]
    conversation_flow: str  # "greeting", "deep_conversation", "farewell", etc.


class PersonalityEngine:
    """
    Core engine for maintaining personality consistency and managing conversational context
    """
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.embedding_model = embedding_model or SentenceTransformer("moka-ai/m3e-small")
        
        # Character states
        self.character_states: Dict[str, ContextualState] = {}
        
        # Personality consistency patterns
        self.personality_vectors: Dict[str, np.ndarray] = {}
        self.speech_pattern_vectors: Dict[str, List[np.ndarray]] = {}
        
        # Context analysis patterns
        self.emotion_keywords = {
            EmotionState.HAPPY: ["开心", "高兴", "快乐", "喜悦", "兴奋", "满意", "愉快", "笑", "哈哈"],
            EmotionState.SAD: ["伤心", "难过", "悲伤", "失望", "沮丧", "哭", "眼泪", "郁闷"],
            EmotionState.ANGRY: ["生气", "愤怒", "恼火", "烦躁", "火大", "气死", "讨厌"],
            EmotionState.EXCITED: ["激动", "兴奋", "热情", "振奋", "亢奋", "期待"],
            EmotionState.CALM: ["平静", "冷静", "安静", "淡定", "轻松", "放松"],
            EmotionState.CONFUSED: ["困惑", "迷惑", "不明白", "奇怪", "疑问", "为什么"],
            EmotionState.SURPRISED: ["惊讶", "震惊", "意外", "吃惊", "没想到", "天哪"],
        }
        
        self.conversation_flow_patterns = {
            "greeting": ["你好", "早上好", "晚上好", "初次见面", "认识你"],
            "question": ["什么", "怎么", "为什么", "哪里", "谁", "吗", "?", "？"],
            "request": ["请", "能否", "可以", "帮忙", "麻烦", "希望"],
            "farewell": ["再见", "拜拜", "下次见", "明天见", "晚安"],
            "compliment": ["很好", "棒", "厉害", "amazing", "不错", "赞"],
            "complaint": ["不好", "糟糕", "讨厌", "烦人", "问题"]
        }
        
    async def initialize_character_state(self, character: CharacterProfile) -> ContextualState:
        """Initialize contextual state for a character"""
        # Create personality vector
        personality_text = f"{character.personality} {character.background}"
        personality_vector = self.embedding_model.encode(personality_text, convert_to_tensor=False)
        self.personality_vectors[character.name] = personality_vector
        
        # Create speech pattern vectors
        if character.speech_patterns:
            pattern_vectors = []
            for pattern in character.speech_patterns:
                pattern_vector = self.embedding_model.encode(pattern, convert_to_tensor=False)
                pattern_vectors.append(pattern_vector)
            self.speech_pattern_vectors[character.name] = pattern_vectors
        
        # Initialize state
        initial_state = ContextualState(
            current_emotion=EmotionState(character.base_mood),
            emotion_intensity=0.3,
            energy_level=0.7,
            topic_focus=None,
            relationship_context={},
            recent_events=[],
            conversation_flow="neutral"
        )
        
        self.character_states[character.name] = initial_state
        return initial_state
    
    def analyze_message_context(
        self, 
        message: str, 
        character_name: str,
        conversation_history: List[ConversationMessage]
    ) -> Dict[str, Any]:
        """
        Analyze incoming message for emotional context, topics, and conversation flow
        """
        analysis = {
            "detected_emotion": EmotionState.NEUTRAL,
            "emotion_intensity": 0.0,
            "conversation_flow": "neutral",
            "topics": [],
            "requires_personality_response": False,
            "response_tone": "neutral",
            "context_shift": False
        }
        
        message_lower = message.lower()
        
        # Emotion detection
        max_emotion_score = 0
        detected_emotion = EmotionState.NEUTRAL
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > max_emotion_score:
                max_emotion_score = score
                detected_emotion = emotion
        
        analysis["detected_emotion"] = detected_emotion
        analysis["emotion_intensity"] = min(max_emotion_score / 3.0, 1.0)  # Normalize
        
        # Conversation flow detection
        for flow_type, patterns in self.conversation_flow_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                analysis["conversation_flow"] = flow_type
                break
        
        # Topic extraction (simplified)
        # In practice, you might use more sophisticated NLP
        topics = []
        if len(message) > 20:  # Longer messages likely have topics
            words = message.split()
            # Extract nouns, entities, etc. (simplified)
            potential_topics = [word for word in words if len(word) > 3 and word.isalpha()]
            topics = potential_topics[:3]  # Keep top 3
        
        analysis["topics"] = topics
        
        # Determine if personality-specific response is needed
        personal_indicators = ["你是谁", "你的性格", "你喜欢", "你觉得", character_name]
        analysis["requires_personality_response"] = any(
            indicator in message_lower for indicator in personal_indicators
        )
        
        # Context shift detection
        if conversation_history:
            recent_topics = []
            for msg in conversation_history[-3:]:  # Check last 3 messages
                if hasattr(msg, 'context_tags'):
                    recent_topics.extend(msg.context_tags)
            
            current_topics = set(topics)
            recent_topics_set = set(recent_topics)
            
            # If current topics are very different from recent ones
            if current_topics and recent_topics_set:
                overlap = len(current_topics.intersection(recent_topics_set))
                if overlap / len(current_topics) < 0.3:  # Less than 30% overlap
                    analysis["context_shift"] = True
        
        return analysis
    
    def update_character_state(
        self,
        character_name: str,
        message_analysis: Dict[str, Any],
        character_profile: CharacterProfile
    ):
        """Update character's contextual state based on message analysis"""
        if character_name not in self.character_states:
            asyncio.create_task(self.initialize_character_state(character_profile))
            return
        
        state = self.character_states[character_name]
        
        # Update emotion with volatility
        new_emotion = message_analysis["detected_emotion"]
        emotion_intensity = message_analysis["emotion_intensity"]
        
        if emotion_intensity > 0.3 and new_emotion != EmotionState.NEUTRAL:
            # Apply mood volatility from character profile
            volatility = character_profile.mood_volatility
            
            if np.random.random() < volatility:
                state.current_emotion = new_emotion
                state.emotion_intensity = emotion_intensity
            else:
                # Gradual change
                state.emotion_intensity = (state.emotion_intensity + emotion_intensity) / 2
        
        # Update conversation flow
        state.conversation_flow = message_analysis["conversation_flow"]
        
        # Update topic focus
        if message_analysis["topics"]:
            state.topic_focus = message_analysis["topics"][0]  # Primary topic
        
        # Update recent events
        event_description = f"{state.conversation_flow}_{new_emotion.value}"
        state.recent_events.append(event_description)
        
        # Keep only recent events (last 10)
        if len(state.recent_events) > 10:
            state.recent_events = state.recent_events[-10:]
        
        # Adjust energy level based on conversation flow
        if message_analysis["conversation_flow"] in ["excited", "question"]:
            state.energy_level = min(state.energy_level + 0.1, 1.0)
        elif message_analysis["conversation_flow"] in ["farewell", "sad"]:
            state.energy_level = max(state.energy_level - 0.1, 0.1)
    
    def generate_personality_consistent_prompt(
        self,
        character: CharacterProfile,
        message: str,
        conversation_history: List[ConversationMessage],
        rag_examples: List[str]
    ) -> str:
        """
        Generate a system prompt that maintains personality consistency
        """
        if character.name not in self.character_states:
            asyncio.create_task(self.initialize_character_state(character))
            return self._fallback_prompt(character, message)
        
        state = self.character_states[character.name]
        
        # Analyze current context
        message_analysis = self.analyze_message_context(message, character.name, conversation_history)
        
        # Update character state
        self.update_character_state(character.name, message_analysis, character)
        
        # Build context-aware prompt
        prompt_parts = []
        
        # Core personality
        prompt_parts.append(f"你正在扮演{character.name}。")
        prompt_parts.append(f"性格特征：{character.personality}")
        
        if character.background:
            prompt_parts.append(f"背景：{character.background}")
        
        # Current emotional state
        prompt_parts.append(f"当前情绪状态：{state.current_emotion.value}（强度：{state.emotion_intensity:.1f}）")
        prompt_parts.append(f"当前能量水平：{state.energy_level:.1f}")
        
        # Conversation style adaptation
        style_instructions = self._get_style_instructions(character.conversation_style, state)
        prompt_parts.append(style_instructions)
        
        # Speech patterns
        if character.speech_patterns:
            patterns_text = "、".join(character.speech_patterns[:5])
            prompt_parts.append(f"你的典型说话方式包括：{patterns_text}")
        
        # Context-specific instructions
        if message_analysis["conversation_flow"] == "greeting":
            prompt_parts.append("这是一次问候，请以友好的方式回应。")
        elif message_analysis["conversation_flow"] == "question":
            prompt_parts.append("用户在询问问题，请提供有帮助的回答。")
        elif message_analysis["requires_personality_response"]:
            prompt_parts.append("用户询问关于你的个人信息，请根据你的设定回答。")
        elif message_analysis["context_shift"]:
            prompt_parts.append("对话主题发生了转换，请自然地适应新话题。")
        
        # RAG examples context
        if rag_examples:
            examples_text = "\n".join([f"参考例子{i+1}：{ex[:150]}..." for i, ex in enumerate(rag_examples[:3])])
            prompt_parts.append(f"参考这些相似的对话例子来保持角色一致性：\n{examples_text}")
        
        # Response guidelines
        prompt_parts.append(self._get_response_guidelines(character, state, message_analysis))
        
        # Forbidden topics
        if character.forbidden_topics:
            forbidden_text = "、".join(character.forbidden_topics)
            prompt_parts.append(f"避免讨论这些话题：{forbidden_text}")
        
        # Final instruction
        prompt_parts.append(f"请以{character.name}的身份，根据当前的情绪状态和对话上下文，自然地回应用户的消息。保持角色一致性和个性特色。")
        
        return "\n\n".join(prompt_parts)
    
    def _get_style_instructions(self, conversation_style: str, state: ContextualState) -> str:
        """Get style-specific instructions based on character and current state"""
        base_styles = {
            "casual": "使用轻松、非正式的语言风格。",
            "formal": "使用正式、礼貌的语言风格。",
            "playful": "使用活泼、俏皮的语言风格，可以适当使用表情符号。",
            "dramatic": "使用富有感情色彩、戏剧化的表达方式。",
            "mysterious": "保持一定的神秘感，话语间留有余地。",
            "friendly": "保持友好、温暖、亲切的语调。"
        }
        
        style_instruction = base_styles.get(conversation_style, base_styles["casual"])
        
        # Modify based on current emotional state
        if state.current_emotion == EmotionState.EXCITED:
            style_instruction += " 表现出兴奋和热情。"
        elif state.current_emotion == EmotionState.SAD:
            style_instruction += " 语调稍显低沉，但仍保持基本的友善。"
        elif state.current_emotion == EmotionState.ANGRY:
            style_instruction += " 语调稍显急躁，但避免过度激烈。"
        elif state.current_emotion == EmotionState.CALM:
            style_instruction += " 保持平和、稳定的语调。"
        
        return f"语言风格：{style_instruction}"
    
    def _get_response_guidelines(
        self, 
        character: CharacterProfile, 
        state: ContextualState, 
        message_analysis: Dict[str, Any]
    ) -> str:
        """Generate specific response guidelines"""
        guidelines = []
        
        # Length guidelines
        if message_analysis["conversation_flow"] == "greeting":
            guidelines.append("回应应简洁明了（1-2句话）。")
        elif message_analysis["requires_personality_response"]:
            guidelines.append("可以提供较详细的回答（2-3句话）。")
        else:
            guidelines.append("保持适中的回应长度。")
        
        # Emotional guidelines
        if state.emotion_intensity > 0.7:
            guidelines.append("在回应中体现出当前的强烈情绪。")
        elif state.emotion_intensity > 0.4:
            guidelines.append("在回应中适度体现当前情绪。")
        
        # Energy level guidelines
        if state.energy_level > 0.8:
            guidelines.append("表现出高能量和活力。")
        elif state.energy_level < 0.3:
            guidelines.append("表现出低能量，可能稍显疲倦。")
        
        # Topic coherence
        if state.topic_focus:
            guidelines.append(f"保持与当前话题'{state.topic_focus}'的相关性。")
        
        return "回应指导：" + " ".join(guidelines)
    
    def _fallback_prompt(self, character: CharacterProfile, message: str) -> str:
        """Fallback prompt when state is not available"""
        return f"""你正在扮演{character.name}。

性格特征：{character.personality}
对话风格：{character.conversation_style}

请以{character.name}的身份回应用户的消息，保持角色一致性。"""
    
    def calculate_personality_consistency_score(
        self,
        character_name: str,
        generated_response: str,
        conversation_history: List[ConversationMessage]
    ) -> float:
        """
        Calculate how consistent a generated response is with the character's personality
        """
        if character_name not in self.personality_vectors:
            return 0.5  # Neutral score if no personality vector available
        
        try:
            # Encode the generated response
            response_vector = self.embedding_model.encode(generated_response, convert_to_tensor=False)
            personality_vector = self.personality_vectors[character_name]
            
            # Calculate cosine similarity
            similarity = np.dot(response_vector, personality_vector) / (
                np.linalg.norm(response_vector) * np.linalg.norm(personality_vector)
            )
            
            # Normalize to 0-1 range
            consistency_score = (similarity + 1) / 2
            
            # Bonus for speech pattern matching
            if character_name in self.speech_pattern_vectors:
                pattern_scores = []
                for pattern_vector in self.speech_pattern_vectors[character_name]:
                    pattern_similarity = np.dot(response_vector, pattern_vector) / (
                        np.linalg.norm(response_vector) * np.linalg.norm(pattern_vector)
                    )
                    pattern_scores.append(pattern_similarity)
                
                if pattern_scores:
                    max_pattern_score = max(pattern_scores)
                    consistency_score = consistency_score * 0.8 + max_pattern_score * 0.2
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception as e:
            print(f"❌ Error calculating consistency score: {e}")
            return 0.5
    
    def get_character_state_summary(self, character_name: str) -> Dict[str, Any]:
        """Get current state summary for a character"""
        if character_name not in self.character_states:
            return {"error": "Character state not found"}
        
        state = self.character_states[character_name]
        
        return {
            "character_name": character_name,
            "current_emotion": state.current_emotion.value,
            "emotion_intensity": state.emotion_intensity,
            "energy_level": state.energy_level,
            "topic_focus": state.topic_focus,
            "conversation_flow": state.conversation_flow,
            "recent_events": state.recent_events[-5:],  # Last 5 events
            "relationships": dict(state.relationship_context)
        }
    
    def reset_character_state(self, character_name: str):
        """Reset character state to neutral"""
        if character_name in self.character_states:
            state = self.character_states[character_name]
            state.current_emotion = EmotionState.NEUTRAL
            state.emotion_intensity = 0.3
            state.energy_level = 0.7
            state.topic_focus = None
            state.recent_events = []
            state.conversation_flow = "neutral"


class PersonalityValidator:
    """Validates responses for personality consistency"""
    
    def __init__(self, personality_engine: PersonalityEngine):
        self.engine = personality_engine
    
    def validate_response(
        self,
        character: CharacterProfile,
        user_message: str,
        generated_response: str,
        conversation_history: List[ConversationMessage]
    ) -> Dict[str, Any]:
        """
        Validate a generated response for personality consistency
        """
        validation_result = {
            "is_consistent": True,
            "consistency_score": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # Calculate consistency score
        consistency_score = self.engine.calculate_personality_consistency_score(
            character.name, generated_response, conversation_history
        )
        validation_result["consistency_score"] = consistency_score
        
        # Check for personality violations
        issues = []
        
        # Check forbidden topics
        if character.forbidden_topics:
            for forbidden in character.forbidden_topics:
                if forbidden.lower() in generated_response.lower():
                    issues.append(f"Response contains forbidden topic: {forbidden}")
        
        # Check response length
        response_length = len(generated_response)
        if response_length > character.max_response_length:
            issues.append(f"Response too long: {response_length} > {character.max_response_length}")
        
        # Check for style consistency
        style_violations = self._check_style_violations(character, generated_response)
        issues.extend(style_violations)
        
        # Overall consistency check
        if consistency_score < 0.6:
            issues.append(f"Low personality consistency score: {consistency_score:.2f}")
            validation_result["is_consistent"] = False
        
        if consistency_score < 0.8:
            validation_result["suggestions"].append("Consider incorporating more character-specific language patterns")
        
        validation_result["issues"] = issues
        
        return validation_result
    
    def _check_style_violations(self, character: CharacterProfile, response: str) -> List[str]:
        """Check for conversation style violations"""
        violations = []
        
        style = character.conversation_style
        response_lower = response.lower()
        
        if style == "formal":
            informal_markers = ["哈哈", "嘿", "呀", "啊", "哇"]
            if any(marker in response_lower for marker in informal_markers):
                violations.append("Informal language used in formal character")
        
        elif style == "casual":
            overly_formal = ["此致敬礼", "谨此", "敬请", "恳请"]
            if any(marker in response_lower for marker in overly_formal):
                violations.append("Overly formal language used in casual character")
        
        elif style == "playful":
            if len(response) > 50 and not any(char in response for char in ["!", "~", "😊", "哈", "呢"]):
                violations.append("Playful character response lacks enthusiasm markers")
        
        return violations


# Utility functions for integration

def create_personality_engine(characters: List[CharacterProfile]) -> PersonalityEngine:
    """Create and initialize personality engine with multiple characters"""
    engine = PersonalityEngine()
    
    async def init_all():
        for character in characters:
            await engine.initialize_character_state(character)
    
    asyncio.run(init_all())
    return engine


# Example usage and testing
async def test_personality_engine():
    """Test the personality engine"""
    # Create test character
    from character_config import CharacterProfile
    
    test_character = CharacterProfile(
        name="测试角色",
        personality="活泼开朗，喜欢开玩笑，对新事物充满好奇心",
        speech_patterns=["哈哈", "真有趣", "我觉得", "太棒了"],
        conversation_style="playful",
        mood_volatility=0.7
    )
    
    # Initialize engine
    engine = PersonalityEngine()
    await engine.initialize_character_state(test_character)
    
    # Test message analysis
    test_message = "你好！今天天气真好，你在做什么呢？"
    analysis = engine.analyze_message_context(test_message, "测试角色", [])
    print(f"📊 Message analysis: {analysis}")
    
    # Test prompt generation
    prompt = engine.generate_personality_consistent_prompt(
        test_character, test_message, [], ["参考对话例子"]
    )
    print(f"📝 Generated prompt:\n{prompt}")
    
    # Test state summary
    state_summary = engine.get_character_state_summary("测试角色")
    print(f"🎭 Character state: {state_summary}")


if __name__ == "__main__":
    asyncio.run(test_personality_engine())