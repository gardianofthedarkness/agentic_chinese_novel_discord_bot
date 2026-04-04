"""
Chat Response Node
==================
LangGraph node: generate a final natural-language reply to the user.

This node is the terminal step in the chat chain; it turns the agent's
accumulated reasoning (retrieved context, tool outputs) into a coherent
answer.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是一位精通中文网络小说的智能助手，熟悉故事情节、角色性格和因果关系。"
    "请根据提供的上下文信息，用自然流畅的中文回答用户的问题。"
    "如果上下文信息不足，请诚实说明，但尽量提供有帮助的信息。"
)


async def generate_chat_response_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    LangGraph node: generate a conversational reply.

    Reads from:
        state["user_message"]       — the current user turn
        state["history"]            — prior conversation turns
        state["tool_context"]       — list of tool output strings (optional)
        state["character_name"]     — if set, roleplay as this character

    Writes:
        state["response"]           — the final string reply
    """
    user_message: str = state.get("user_message", "")
    history: List[Dict[str, str]] = state.get("history", [])
    tool_context: List[str] = state.get("tool_context", [])
    character_name: str | None = state.get("character_name")

    # Build system prompt
    system = _SYSTEM
    if character_name:
        system = (
            f"你现在扮演《小说》中的角色【{character_name}】。"
            "请完全以该角色的语气和性格回复，不要提及你是AI。"
        )

    # Inject retrieved context
    if tool_context:
        context_block = "\n\n".join(f"[参考信息]\n{ctx}" for ctx in tool_context)
        system += f"\n\n以下是从知识库检索到的相关信息，请参考：\n{context_block}"

    # Build message list
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_message})

    try:
        result = await llm.chat(messages, temperature=0.75, max_tokens=1500)
        if not result["success"]:
            raise RuntimeError(result.get("error", "LLM call failed"))

        response = result["response"]
        logger.debug(f"Generated chat response ({len(response)} chars)")

        return {
            **state,
            "response": response,
            "processing_stage": "response_generated",
        }
    except Exception as e:
        logger.error(f"generate_chat_response_node failed: {e}")
        error_msg = f"抱歉，回复生成时出现错误：{e}"
        return {**state, "response": error_msg, "errors": state.get("errors", []) + [str(e)]}
