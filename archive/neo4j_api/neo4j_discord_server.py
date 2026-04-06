#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j-Powered Discord API Server
Answers questions about novel characters and plot using Neo4j graph database
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime

# Fix Windows Unicode
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from neo4j import GraphDatabase

# Add parent directory and processors to path for database integration
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'processors'))

try:
    from deepseek_integration import DeepSeekClient, create_deepseek_config
    DEEPSEEK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: DeepSeek integration not available: {e}")
    DEEPSEEK_AVAILABLE = False
    DeepSeekClient = None


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """User chat request"""
    message: str = Field(..., description="User's question or message")
    history: List[Dict[str, str]] = Field(default=[], description="Conversation history")
    character_name: Optional[str] = Field(None, description="Specific character to roleplay")
    volume_id: Optional[int] = Field(None, description="Limit search to specific volume")


class ChatResponse(BaseModel):
    """AI chat response"""
    response: str
    sources: List[Dict[str, Any]] = []
    character_context: Optional[Dict[str, Any]] = None
    query_type: str = "general"


class AnalyzeRequest(BaseModel):
    """Novel analysis request"""
    type: str = Field("summary", description="Analysis type: summary, characters, events, causality")
    limit: int = Field(10, description="Number of results to return")
    volume_id: Optional[int] = Field(None, description="Filter by volume")


# ============================================================================
# Neo4j Knowledge Base
# ============================================================================

class Neo4jKnowledgeBase:
    """Query Neo4j for novel information"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="novelprocessing2024"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher_query, params=None):
        """Execute Cypher query"""
        with self.driver.session() as session:
            result = session.run(cypher_query, params or {})
            return [record.data() for record in result]

    def search_character(self, character_name: str, volume_id: Optional[int] = None) -> Optional[Dict]:
        """Search for character information - simplified for current schema"""
        query = """
        MATCH (c:Character)
        WHERE toLower(c.name) CONTAINS toLower($name)
        RETURN c.name as name
        LIMIT 1
        """

        results = self.query(query, {'name': character_name})
        if results:
            # Add default values for missing properties
            return {
                'name': results[0]['name'],
                'type': 'Character',
                'traits': [],
                'volume': volume_id,
                'confidence': 1.0
            }
        return None

    def get_character_events(self, character_name: str, limit: int = 10) -> List[Dict]:
        """Get events mentioning a character in description"""
        query = """
        MATCH (e:Event)
        WHERE toLower(e.description) CONTAINS toLower($character)
        RETURN e.description as description, e.event_type as type,
               e.importance_score as importance, e.volume_id as volume,
               e.chronological_order as order
        ORDER BY e.chronological_order
        LIMIT $limit
        """

        return self.query(query, {'character': character_name, 'limit': limit})

    def get_causality_chain(self, event_keyword: str, limit: int = 5) -> List[Dict]:
        """Find causality chains related to an event"""
        query = """
        MATCH (e:Event)
        WHERE toLower(e.description) CONTAINS toLower($keyword)
        WITH e LIMIT 1
        MATCH path = (e)-[:CAUSES*0..3]->(consequence:Event)
        RETURN e.description as start_event,
               consequence.description as consequence,
               length(path) as hops
        ORDER BY hops
        LIMIT $limit
        """

        return self.query(query, {'keyword': event_keyword, 'limit': limit})

    def get_all_characters(self, volume_id: Optional[int] = None, limit: int = 20) -> List[Dict]:
        """Get all characters - simplified for current schema"""
        query = """
        MATCH (c:Character)
        RETURN c.name as name
        ORDER BY c.name
        LIMIT $limit
        """

        results = self.query(query, {'limit': limit})
        # Add default properties
        return [{'name': r['name'], 'type': None, 'volume': None, 'confidence': None} for r in results]

    def get_important_events(self, volume_id: Optional[int] = None, limit: int = 10) -> List[Dict]:
        """Get most important events"""
        query = """
        MATCH (e:Event)
        """
        if volume_id:
            query += " WHERE e.volume_id = $volume_id"

        query += """
        RETURN e.description as description, e.event_type as type,
               e.importance_score as importance, e.volume_id as volume,
               e.chronological_order as order
        ORDER BY e.importance_score DESC
        LIMIT $limit
        """

        results = self.query(query, {'volume_id': volume_id, 'limit': limit})
        # Add characters field (empty for now since not in schema)
        for r in results:
            r['characters'] = []
        return results

    def search_events_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Search events by keyword"""
        query = """
        MATCH (e:Event)
        WHERE toLower(e.description) CONTAINS toLower($keyword)
        RETURN e.description as description, e.event_type as type,
               e.importance_score as importance, e.volume_id as volume
        ORDER BY e.importance_score DESC
        LIMIT $limit
        """

        return self.query(query, {'keyword': keyword, 'limit': limit})


# ============================================================================
# AI Agent with Neo4j RAG
# ============================================================================

class NovelQAAgent:
    """AI agent that answers questions using Neo4j knowledge base"""

    def __init__(self, neo4j_kb: Neo4jKnowledgeBase, deepseek_client: Optional[DeepSeekClient] = None):
        self.kb = neo4j_kb
        self.deepseek = deepseek_client

    def detect_query_intent(self, message: str) -> str:
        """Detect what the user is asking about"""
        message_lower = message.lower()

        if any(word in message_lower for word in ['summarize', 'summary', 'overview', 'about', 'chapter', 'volume', '总结', '概述', '章节']):
            return 'summary'
        elif any(word in message_lower for word in ['character', 'who is', 'who are', '角色', '人物']):
            return 'character'
        elif any(word in message_lower for word in ['event', 'what happened', 'plot', '事件', '情节']):
            return 'event'
        elif any(word in message_lower for word in ['why', 'because', 'cause', 'reason', '为什么', '原因']):
            return 'causality'
        else:
            return 'general'

    def extract_volume_number(self, message: str) -> Optional[int]:
        """Extract volume/chapter number from message"""
        import re
        # Look for patterns like "first", "1", "volume 1", "chapter 1", etc.
        patterns = [
            r'first',
            r'volume\s*(\d+)',
            r'chapter\s*(\d+)',
            r'vol\s*(\d+)',
            r'\b(\d+)\b'
        ]

        message_lower = message.lower()
        if 'first' in message_lower:
            return 1

        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match and match.groups():
                return int(match.group(1))

        return None

    def extract_keywords(self, message: str) -> List[str]:
        """Extract key terms from message (simple version)"""
        # Remove common words and split
        stop_words = {'is', 'are', 'what', 'who', 'the', 'a', 'an', 'in', 'about', 'tell', 'me'}
        words = message.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:3]  # Top 3 keywords

    async def answer_question(self, message: str, history: List[Dict] = None,
                             character_name: Optional[str] = None,
                             volume_id: Optional[int] = None) -> ChatResponse:
        """Answer user question using Neo4j + AI"""

        # Detect intent
        intent = self.detect_query_intent(message)
        keywords = self.extract_keywords(message)

        # Extract volume number from message if not provided
        if not volume_id and intent == 'summary':
            volume_id = self.extract_volume_number(message)

        # Query Neo4j based on intent
        context_data = []
        sources = []

        if intent == 'character' and keywords:
            # Search for character
            for kw in keywords:
                char_info = self.kb.search_character(kw, volume_id)
                if char_info:
                    context_data.append(f"Character: {char_info['name']} ({char_info['type']})")
                    sources.append({
                        'type': 'character',
                        'name': char_info['name'],
                        'details': char_info
                    })

                    # Get character events
                    events = self.kb.get_character_events(char_info['name'], limit=5)
                    if events:
                        context_data.append(f"Recent events for {char_info['name']}:")
                        for evt in events[:3]:
                            context_data.append(f"  - {evt['description']}")
                    break

        elif intent == 'event':
            # Search events by keywords
            for kw in keywords:
                events = self.kb.search_events_by_keyword(kw, limit=5)
                if events:
                    context_data.append(f"Related events:")
                    for evt in events:
                        context_data.append(f"  - [Vol {evt['volume']}] {evt['description']}")
                        sources.append({
                            'type': 'event',
                            'description': evt['description'],
                            'volume': evt['volume']
                        })
                    break

        elif intent == 'causality' and keywords:
            # Find causality chains
            for kw in keywords:
                chains = self.kb.get_causality_chain(kw, limit=5)
                if chains:
                    context_data.append(f"Causality chain:")
                    for chain in chains:
                        context_data.append(f"  {chain['start_event']} → {chain['consequence']}")
                        sources.append({
                            'type': 'causality',
                            'chain': chain
                        })
                    break

        elif intent == 'summary':
            # Get overview - get MORE events for better summaries
            chars = self.kb.get_all_characters(volume_id, limit=10)
            events = self.kb.get_important_events(volume_id, limit=15)

            if volume_id:
                context_data.append(f"Summary for Volume {volume_id}:")
            else:
                context_data.append(f"Overall Summary:")

            if chars:
                context_data.append(f"\nMain characters ({len(chars)} total):")
                for char in chars[:5]:
                    context_data.append(f"  - {char['name']}")

            if events:
                context_data.append(f"\nKey events ({len(events)} total):")
                # Show events in chronological order
                sorted_events = sorted(events, key=lambda x: x.get('order', 0))
                for evt in sorted_events:
                    context_data.append(f"  - [{evt.get('type', 'event')}] {evt['description']}")
                    sources.append({
                        'type': 'event',
                        'description': evt['description'],
                        'volume': evt.get('volume'),
                        'importance': evt.get('importance', 0)
                    })

        # Fallback: Get some general context
        if not context_data:
            events = self.kb.get_important_events(volume_id, limit=3)
            if events:
                context_data.append("Some key events from the novel:")
                for evt in events:
                    context_data.append(f"  - {evt['description']}")

        # Build context string
        context = "\n".join(context_data) if context_data else "No specific information found in database."

        # Generate AI response
        if self.deepseek and DEEPSEEK_AVAILABLE:
            prompt = self._build_prompt(message, context, character_name, history)

            try:
                response_text = await self.deepseek.generate_response(
                    prompt=prompt,
                    temperature=0.7 if not character_name else 0.8,
                    max_tokens=500
                )
            except Exception as e:
                print(f"DeepSeek error: {e}")
                response_text = f"Based on the novel data:\n\n{context}"
        else:
            # Fallback without AI
            response_text = f"📚 Information from novel database:\n\n{context}"

        return ChatResponse(
            response=response_text,
            sources=sources,
            query_type=intent
        )

    def _build_prompt(self, question: str, context: str, character_name: Optional[str],
                     history: Optional[List[Dict]]) -> str:
        """Build prompt for AI"""

        if character_name:
            # Character roleplay mode
            prompt = f"""你正在扮演小说中的角色 {character_name}。

根据以下小说信息回答用户问题：

{context}

用户问题: {question}

请以 {character_name} 的口吻和性格回答，保持角色设定。回答要简洁自然。"""

        else:
            # General QA mode
            prompt = f"""你是一个小说分析助手。根据以下从Neo4j数据库中提取的小说信息回答用户问题。

小说信息：
{context}

用户问题: {question}

请基于提供的信息进行准确回答。如果信息不足，请诚实说明。回答要简洁专业。"""

        return prompt


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Neo4j Novel QA Server",
    description="Discord bot API powered by Neo4j graph database",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
neo4j_kb = None
qa_agent = None
deepseek_client = None


@app.on_event("startup")
async def startup():
    """Initialize connections"""
    global neo4j_kb, qa_agent, deepseek_client

    print("🚀 Starting Neo4j Discord Server...")

    # Connect to Neo4j
    try:
        neo4j_kb = Neo4jKnowledgeBase()
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"⚠️  Neo4j connection failed: {e}")
        neo4j_kb = None

    # Initialize DeepSeek
    if DEEPSEEK_AVAILABLE:
        try:
            config = create_deepseek_config()
            deepseek_client = DeepSeekClient(config)
            await deepseek_client.initialize()
            print("✅ DeepSeek AI initialized")
        except Exception as e:
            print(f"⚠️  DeepSeek initialization failed: {e}")
            deepseek_client = None

    # Create QA agent
    qa_agent = NovelQAAgent(neo4j_kb, deepseek_client)
    print("✅ QA Agent ready")
    print()
    print("=" * 60)
    print("Server ready at http://localhost:5005")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup connections"""
    if neo4j_kb:
        neo4j_kb.close()
    if deepseek_client:
        await deepseek_client.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "neo4j_connected": neo4j_kb is not None,
        "deepseek_available": deepseek_client is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - answer questions about the novel"""
    if not qa_agent:
        raise HTTPException(status_code=503, detail="QA Agent not initialized")

    try:
        response = await qa_agent.answer_question(
            message=request.message,
            history=request.history,
            character_name=request.character_name,
            volume_id=request.volume_id
        )
        return response
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyze novel data - Discord bot compatible format"""
    if not neo4j_kb:
        raise HTTPException(status_code=503, detail="Neo4j not connected")

    try:
        # Get character and event counts
        char_count_query = "MATCH (c:Character) RETURN count(c) as count"
        event_count_query = "MATCH (e:Event) RETURN count(e) as count"
        link_count_query = "MATCH ()-[r:CAUSES]->() RETURN count(r) as count"
        volume_count_query = "MATCH (c:Character) RETURN count(DISTINCT c.volume_id) as count"

        char_count = neo4j_kb.query(char_count_query)[0]['count'] if neo4j_kb.query(char_count_query) else 0
        event_count = neo4j_kb.query(event_count_query)[0]['count'] if neo4j_kb.query(event_count_query) else 0
        link_count = neo4j_kb.query(link_count_query)[0]['count'] if neo4j_kb.query(link_count_query) else 0
        volume_count = neo4j_kb.query(volume_count_query)[0]['count'] if neo4j_kb.query(volume_count_query) else 0

        # Get character data categorized by type
        all_chars = neo4j_kb.get_all_characters(request.volume_id, request.limit)

        protagonists = []
        antagonists = []
        supporting = []

        for char in all_chars:
            char_type = (char.get('type') or '').lower()
            char_data = {"name": char['name'], "type": char.get('type', 'Unknown')}

            if '主角' in char_type or 'protagonist' in char_type:
                protagonists.append(char_data)
            elif '反派' in char_type or 'antagonist' in char_type:
                antagonists.append(char_data)
            else:
                supporting.append(char_data)

        # Get important events as "storylines"
        events = neo4j_kb.get_important_events(request.volume_id, min(request.limit, 10))
        storylines = [
            {
                "title": evt.get('description', 'Unknown Event')[:100],
                "type": evt.get('type', 'plot_event'),
                "importance": evt.get('importance', 0.5)
            }
            for evt in events
        ]

        # Return in Discord bot expected format
        return {
            "results": {
                "analysis_summary": {
                    "chapters_processed": volume_count,
                    "characters_discovered": char_count,
                    "storylines_identified": len(storylines),
                    "timeline_events": event_count,
                    "recent_chapter": {
                        "index": volume_count,
                        "title": f"Volume {volume_count}"
                    } if volume_count > 0 else None
                },
                "character_breakdown": {
                    "protagonists": protagonists[:5],
                    "antagonists": antagonists[:5],
                    "supporting": supporting[:10]
                },
                "storyline_overview": storylines,
                "system_status": {
                    "data_source": "Neo4j Graph Database",
                    "rag_enabled": True,
                    "ai_model": "DeepSeek" if deepseek_client else "None"
                }
            }
        }

    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/characters")
async def list_characters(volume_id: Optional[int] = None, limit: int = 20):
    """List all characters"""
    if not neo4j_kb:
        raise HTTPException(status_code=503, detail="Neo4j not connected")

    chars = neo4j_kb.get_all_characters(volume_id, limit)
    return {"characters": chars}


@app.get("/api/events")
async def list_events(volume_id: Optional[int] = None, limit: int = 20):
    """List important events"""
    if not neo4j_kb:
        raise HTTPException(status_code=503, detail="Neo4j not connected")

    events = neo4j_kb.get_important_events(volume_id, limit)
    return {"events": events}


@app.get("/api/agent/status")
async def agent_status():
    """Get agent status - compatibility endpoint for Discord bot"""
    return {
        "status": "online",
        "agent_type": "neo4j_rag_agent",
        "capabilities": ["character_queries", "event_queries", "causality_analysis", "roleplay"],
        "neo4j_connected": neo4j_kb is not None,
        "deepseek_available": deepseek_client is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/agent/memory")
async def agent_memory():
    """Get agent memory - compatibility endpoint for Discord bot"""
    if not neo4j_kb:
        raise HTTPException(status_code=503, detail="Neo4j not connected")

    # Return summary of what's in the knowledge base
    try:
        char_count_query = "MATCH (c:Character) RETURN count(c) as count"
        event_count_query = "MATCH (e:Event) RETURN count(e) as count"
        link_count_query = "MATCH ()-[r:CAUSES]->() RETURN count(r) as count"

        char_count = neo4j_kb.query(char_count_query)[0]['count'] if neo4j_kb.query(char_count_query) else 0
        event_count = neo4j_kb.query(event_count_query)[0]['count'] if neo4j_kb.query(event_count_query) else 0
        link_count = neo4j_kb.query(link_count_query)[0]['count'] if neo4j_kb.query(link_count_query) else 0

        return {
            "knowledge_base": {
                "total_characters": char_count,
                "total_events": event_count,
                "total_causal_links": link_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExploreRequest(BaseModel):
    """Topic exploration request"""
    topic: str
    depth: str = "medium"


@app.post("/api/agent/explore")
async def explore_topic(request: ExploreRequest):
    """Explore a topic - compatibility endpoint for Discord bot"""
    if not qa_agent:
        raise HTTPException(status_code=503, detail="QA Agent not initialized")

    try:
        # Use the chat endpoint to explore the topic
        response = await qa_agent.answer_question(
            message=f"Tell me everything about: {request.topic}",
            history=[],
            character_name=None,
            volume_id=None
        )

        return {
            "topic": request.topic,
            "depth": request.depth,
            "exploration": response.response,
            "sources": response.sources,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting Neo4j Novel QA Server...")
    print("Make sure Neo4j is running at bolt://localhost:7687")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5005,
        log_level="info"
    )
