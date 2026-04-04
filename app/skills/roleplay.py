"""
Roleplay Skill
==============
Manages character profiles and generates in-character responses.

Character profiles can be loaded from Neo4j (persisted) or registered
at runtime via add_character().
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RoleplaySkill:
    """
    Maintains a registry of character profiles and conversation histories,
    then delegates response generation to the LLM client.
    """

    def __init__(self, llm, db=None):
        self.llm = llm
        self.db = db
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    # ------------------------------------------------------------------
    # Character management
    # ------------------------------------------------------------------

    def add_character(self, name: str, profile: Dict[str, Any]) -> None:
        """Register a character profile in memory."""
        self._profiles[name] = profile
        self._histories.setdefault(name, [])
        logger.info(f"Character registered: {name}")

    def list_characters(self) -> List[str]:
        return list(self._profiles.keys())

    def clear_history(self, name: str) -> None:
        self._histories[name] = []

    async def load_from_neo4j(self, character_name: str) -> bool:
        """
        Try to load a character profile from Neo4j.

        Returns True if found.
        """
        if self.db is None or not self.db.neo4j or not self.db.neo4j.driver:
            return False
        try:
            with self.db.neo4j.driver.session() as session:
                res = session.run(
                    "MATCH (c:Character) WHERE toLower(c.name) CONTAINS toLower($n) "
                    "RETURN c LIMIT 1",
                    n=character_name,
                )
                record = res.single()
                if record:
                    node = dict(record["c"])
                    profile = {
                        "personality": ", ".join(node.get("personality_traits") or []),
                        "background": node.get("background", ""),
                        "speech_patterns": node.get("aliases") or [],
                        "current_emotions": {},
                        "current_goals": [],
                    }
                    self.add_character(node["name"], profile)
                    return True
        except Exception as e:
            logger.warning(f"Neo4j character load failed: {e}")
        return False

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def respond(
        self,
        character_name: str,
        user_message: str,
        rag_context: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an in-character response.

        Args:
            character_name:  Name of the character to roleplay as.
            user_message:    The user's message.
            rag_context:     Optional list of retrieved passages for context.

        Returns:
            {"success": bool, "response": str, "character_name": str}
        """
        # Lazy load from DB if not in memory
        if character_name not in self._profiles:
            found = await self.load_from_neo4j(character_name)
            if not found:
                logger.warning(f"Character {character_name!r} not found; using generic profile")
                self._profiles[character_name] = {
                    "personality": "不明",
                    "background": "",
                    "speech_patterns": [],
                    "current_emotions": {},
                    "current_goals": [],
                }
                self._histories[character_name] = []

        profile = self._profiles[character_name]
        history = self._histories[character_name]

        result = await self.llm.roleplay(
            character_name=character_name,
            profile=profile,
            conversation_history=history,
            user_message=user_message,
            rag_context=rag_context,
        )

        if result.get("success"):
            # Persist conversation
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": result["response"]})
            # Cap history at 20 turns
            if len(history) > 20:
                self._histories[character_name] = history[-20:]

        return {
            "success": result.get("success", False),
            "response": result.get("response", ""),
            "character_name": character_name,
            "rag_enhanced": bool(rag_context),
        }

    async def analyze(
        self,
        character_name: str,
        recent_events: List[str],
        situation: str,
    ) -> Dict[str, Any]:
        """Run a deep psychological analysis for the character."""
        profile = self._profiles.get(character_name, {})
        personality = profile.get("personality", "")
        return await self.llm.analyze_character(
            character_name=character_name,
            personality=personality,
            recent_events=recent_events,
            query=situation,
        )
