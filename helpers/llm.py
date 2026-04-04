"""
LLM Client (DeepSeek)
======================
Async HTTP client for the DeepSeek API. Supports chat completion and streaming.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95


class DeepSeekClient:
    """Async DeepSeek API client."""

    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=None),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    async def _post(self, messages: List[Dict], stream: bool = False, **kwargs) -> aiohttp.ClientResponse:
        session = await self._get_session()
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": stream,
        }
        return await session.post(f"{self.config.base_url}/chat/completions", json=payload)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Single-shot chat completion.

        Returns:
            {"success": bool, "response": str, "usage": dict}
        """
        try:
            async with await self._post(messages, stream=False, **kwargs) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "success": True,
                        "response": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                        "model": data.get("model", self.config.model),
                    }
                error = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {error}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate(self, prompt: str, **kwargs) -> str:
        """Simple text-in, text-out helper."""
        result = await self.chat([{"role": "user", "content": prompt}], **kwargs)
        if result["success"]:
            return result["response"]
        raise RuntimeError(result.get("error", "Unknown LLM error"))

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """
        Yield response tokens as they arrive (SSE / server-sent events).

        Usage::

            async for token in llm.stream(messages):
                print(token, end="", flush=True)
        """
        try:
            async with await self._post(messages, stream=True, **kwargs) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {error}")
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise RuntimeError(f"Stream failed: {e}") from e

    # ------------------------------------------------------------------
    # Character roleplay helpers (preserved from original)
    # ------------------------------------------------------------------

    async def roleplay(
        self,
        character_name: str,
        profile: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        user_message: str,
        rag_context: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate an in-character response for a named character."""
        system = (
            f"你现在要扮演角色：{character_name}\n\n"
            f"角色设定：\n"
            f"性格特征：{profile.get('personality', '')}\n"
            f"背景信息：{profile.get('background', '')}\n"
            f"语言风格：{', '.join(profile.get('speech_patterns', []))}\n"
            f"当前情绪状态：{json.dumps(profile.get('current_emotions', {}), ensure_ascii=False)}\n"
            f"当前目标：{', '.join(profile.get('current_goals', []))}\n\n"
            f"扮演要求：\n"
            f"1. 完全以{character_name}的身份和语气回应\n"
            f"2. 保持角色的性格一致性和情绪连贯性\n"
            f"3. 不要在回应中提及你是AI或在扮演角色"
        )
        if rag_context:
            system += f"\n\n相关背景信息：\n" + "\n".join(rag_context)

        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": user_message})

        return await self.chat(messages, temperature=kwargs.get("temperature", 0.8))

    async def analyze_character(
        self,
        character_name: str,
        personality: str,
        recent_events: List[str],
        query: str,
    ) -> Dict[str, Any]:
        """Deep psychological character analysis (Chinese)."""
        prompt = (
            f"你是一位专业的中文小说角色分析专家，请分析以下角色。\n\n"
            f"角色：{character_name}\n性格：{personality}\n"
            f"近期事件：\n" + "\n".join(recent_events) +
            f"\n\n当前情况：{query}\n\n请从心理状态、行为动机、关系网络、预期反应、性格发展五个维度深度分析。"
        )
        messages = [
            {"role": "system", "content": "你是精通中文文学和角色心理分析的专家。"},
            {"role": "user", "content": prompt},
        ]
        return await self.chat(messages, temperature=0.3)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(api_key: Optional[str] = None, **kwargs) -> DeepSeekClient:
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY not set")
    cfg = DeepSeekConfig(api_key=key, **kwargs)
    return DeepSeekClient(cfg)
