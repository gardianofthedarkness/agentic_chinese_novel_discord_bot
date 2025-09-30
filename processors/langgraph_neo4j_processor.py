#!/usr/bin/env python3
"""
LangGraph + Neo4j Novel Processor
Advanced graph-based novel processing with orchestrated AI workflows
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# Neo4j imports
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable

# AI and processing imports
import tiktoken
from deepseek_integration import DeepSeekClient, create_deepseek_config

class ProcessingState(TypedDict):
    """State object for LangGraph workflow"""
    batch_id: int
    volume_id: int
    chunks: List[Dict[str, Any]]
    characters: Dict[str, Any]
    events: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    causality_chains: List[Dict[str, Any]]
    temporal_markers: List[str]
    processing_stage: str
    iteration_count: int
    confidence_scores: Dict[str, float]
    errors: List[str]

class GraphNodeType(Enum):
    """Types of nodes in the Neo4j graph"""
    CHARACTER = "Character"
    EVENT = "Event"
    LOCATION = "Location"
    CONCEPT = "Concept"
    EMOTION = "Emotion"
    PLOT_THREAD = "PlotThread"
    VOLUME = "Volume"
    CHAPTER = "Chapter"

class GraphRelationType(Enum):
    """Types of relationships in the Neo4j graph"""
    # Causality relationships
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    PREVENTS = "PREVENTS"
    TRIGGERS = "TRIGGERS"

    # Character relationships
    INFLUENCES = "INFLUENCES"
    INTERACTS_WITH = "INTERACTS_WITH"
    RELATES_TO = "RELATES_TO"
    PROTECTS = "PROTECTS"
    OPPOSES = "OPPOSES"

    # Narrative relationships
    PARTICIPATES_IN = "PARTICIPATES_IN"
    OCCURS_IN = "OCCURS_IN"
    FORESHADOWS = "FORESHADOWS"
    FOLLOWS = "FOLLOWS"
    LEADS_TO = "LEADS_TO"

    # Temporal relationships
    HAPPENS_BEFORE = "HAPPENS_BEFORE"
    HAPPENS_AFTER = "HAPPENS_AFTER"
    HAPPENS_DURING = "HAPPENS_DURING"

    # Emotional relationships
    FEELS = "FEELS"
    EXPERIENCES = "EXPERIENCES"
    REMEMBERS = "REMEMBERS"

@dataclass
class GraphNode:
    """Represents a node in the Neo4j graph"""
    node_id: str
    node_type: GraphNodeType
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class GraphRelationship:
    """Represents a relationship in the Neo4j graph"""
    from_node: str
    to_node: str
    rel_type: GraphRelationType
    properties: Dict[str, Any]
    strength: float = 0.5
    confidence: float = 0.5
    temporal_context: Optional[str] = None

class Neo4jNovelDatabase:
    """Neo4j database interface optimized for novel data"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._setup_constraints_and_indexes()

    def _setup_constraints_and_indexes(self):
        """Set up Neo4j constraints and indexes for optimal performance"""
        with self.driver.session() as session:
            # Create unique constraints
            constraints = [
                "CREATE CONSTRAINT character_id_unique IF NOT EXISTS FOR (c:Character) REQUIRE c.character_id IS UNIQUE",
                "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
                "CREATE CONSTRAINT volume_id_unique IF NOT EXISTS FOR (v:Volume) REQUIRE v.volume_id IS UNIQUE",
            ]

            # Create indexes for performance
            indexes = [
                "CREATE INDEX character_name_index IF NOT EXISTS FOR (c:Character) ON (c.name)",
                "CREATE INDEX event_timestamp_index IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
                "CREATE INDEX event_importance_index IF NOT EXISTS FOR (e:Event) ON (e.importance_score)",
                "CREATE INDEX relationship_strength_index IF NOT EXISTS FOR ()-[r]-() ON (r.strength)",
                "CREATE INDEX temporal_order_index IF NOT EXISTS FOR (e:Event) ON (e.chronological_order)",
                # Vector index for semantic search
                "CREATE VECTOR INDEX event_embedding_index IF NOT EXISTS FOR (e:Event) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}",
                "CREATE VECTOR INDEX character_embedding_index IF NOT EXISTS FOR (c:Character) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint creation warning: {e}")

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    print(f"Index creation warning: {e}")

    def create_character_node(self, character: Dict[str, Any]) -> str:
        """Create or update a character node"""
        query = """
        MERGE (c:Character {character_id: $character_id})
        SET c.name = $name,
            c.character_type = $character_type,
            c.personality_traits = $personality_traits,
            c.confidence_score = $confidence_score,
            c.first_appearance = $first_appearance,
            c.embedding = $embedding,
            c.updated_at = datetime()
        RETURN c.character_id as character_id
        """

        with self.driver.session() as session:
            result = session.run(query,
                character_id=character.get('character_id'),
                name=character.get('name'),
                character_type=character.get('character_type'),
                personality_traits=character.get('personality_traits', []),
                confidence_score=character.get('confidence_score', 0.5),
                first_appearance=character.get('first_appearance'),
                embedding=character.get('embedding')
            )
            return result.single()['character_id']

    def create_event_node(self, event: Dict[str, Any]) -> str:
        """Create an event node with temporal properties"""
        query = """
        CREATE (e:Event {
            event_id: $event_id,
            description: $description,
            event_type: $event_type,
            importance_score: $importance_score,
            chronological_order: $chronological_order,
            timestamp: datetime($timestamp),
            confidence_level: $confidence_level,
            volume_id: $volume_id,
            batch_id: $batch_id,
            embedding: $embedding,
            created_at: datetime()
        })
        RETURN e.event_id as event_id
        """

        with self.driver.session() as session:
            result = session.run(query, **event)
            return result.single()['event_id']

    def create_causality_relationship(self, from_event: str, to_event: str,
                                    causality_type: str, strength: float = 0.5,
                                    reasoning: str = ""):
        """Create causality relationship between events"""
        query = """
        MATCH (from:Event {event_id: $from_event})
        MATCH (to:Event {event_id: $to_event})
        CREATE (from)-[r:CAUSES {
            causality_type: $causality_type,
            strength: $strength,
            reasoning: $reasoning,
            created_at: datetime()
        }]->(to)
        RETURN r
        """

        with self.driver.session() as session:
            session.run(query,
                from_event=from_event,
                to_event=to_event,
                causality_type=causality_type,
                strength=strength,
                reasoning=reasoning
            )

    def create_character_relationship(self, char1: str, char2: str,
                                    rel_type: str, strength: float,
                                    context: str = ""):
        """Create relationship between characters"""
        query = """
        MATCH (c1:Character {character_id: $char1})
        MATCH (c2:Character {character_id: $char2})
        MERGE (c1)-[r:RELATES_TO {type: $rel_type}]->(c2)
        SET r.strength = $strength,
            r.context = $context,
            r.updated_at = datetime()
        RETURN r
        """

        with self.driver.session() as session:
            session.run(query,
                char1=char1,
                char2=char2,
                rel_type=rel_type,
                strength=strength,
                context=context
            )

    def find_causality_path(self, start_event: str, end_event: str, max_depth: int = 5):
        """Find causality path between two events using Neo4j's shortest path"""
        query = """
        MATCH (start:Event {event_id: $start_event})
        MATCH (end:Event {event_id: $end_event})
        MATCH path = shortestPath((start)-[:CAUSES*1..$max_depth]->(end))
        RETURN path, length(path) as path_length
        """

        with self.driver.session() as session:
            result = session.run(query,
                start_event=start_event,
                end_event=end_event,
                max_depth=max_depth
            )
            return result.data()

    def get_character_influence_network(self, character_id: str, depth: int = 2):
        """Get character influence network using graph traversal"""
        query = """
        MATCH (c:Character {character_id: $character_id})
        OPTIONAL MATCH (c)-[r:INFLUENCES*1..$depth]-(other:Character)
        RETURN c, collect(distinct other) as influenced_characters,
               collect(distinct r) as influence_relationships
        """

        with self.driver.session() as session:
            result = session.run(query, character_id=character_id, depth=depth)
            return result.single()

    def get_event_centrality_scores(self):
        """Calculate centrality scores for events to identify key plot points"""
        query = """
        CALL gds.betweennessCentrality.stream({
            nodeProjection: 'Event',
            relationshipProjection: 'CAUSES'
        })
        YIELD nodeId, score
        MATCH (e:Event) WHERE id(e) = nodeId
        RETURN e.event_id as event_id, e.description as description, score
        ORDER BY score DESC
        LIMIT 20
        """

        with self.driver.session() as session:
            result = session.run(query)
            return result.data()

    def temporal_event_analysis(self, volume_id: int):
        """Analyze temporal patterns in events for a volume"""
        query = """
        MATCH (e:Event {volume_id: $volume_id})
        WITH e ORDER BY e.chronological_order
        MATCH (e)-[r:CAUSES]->(next:Event)
        RETURN e.event_id as event_id,
               e.chronological_order as position,
               count(r) as causality_outgoing,
               e.importance_score as importance
        ORDER BY e.chronological_order
        """

        with self.driver.session() as session:
            result = session.run(query, volume_id=volume_id)
            return result.data()

    def semantic_event_search(self, query_embedding: List[float], limit: int = 10):
        """Semantic search using vector embeddings"""
        query = """
        CALL db.index.vector.queryNodes('event_embedding_index', $limit, $query_embedding)
        YIELD node, score
        RETURN node.event_id as event_id,
               node.description as description,
               score as similarity_score
        """

        with self.driver.session() as session:
            result = session.run(query,
                query_embedding=query_embedding,
                limit=limit
            )
            return result.data()

    def close(self):
        """Close the database connection"""
        self.driver.close()

class LangGraphNovelProcessor:
    """LangGraph-based novel processor with Neo4j backend"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 deepseek_config: Dict[str, Any]):
        self.db = Neo4jNovelDatabase(neo4j_uri, neo4j_user, neo4j_password)
        self.deepseek_client = DeepSeekClient(deepseek_config)
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.graph = self._build_processing_graph()

    def _build_processing_graph(self) -> StateGraph:
        """Build the LangGraph processing workflow"""

        # Define the workflow graph
        workflow = StateGraph(ProcessingState)

        # Add nodes for different processing stages
        workflow.add_node("extract_characters", self._extract_characters)
        workflow.add_node("extract_events", self._extract_events)
        workflow.add_node("analyze_causality", self._analyze_causality)
        workflow.add_node("build_relationships", self._build_relationships)
        workflow.add_node("temporal_analysis", self._temporal_analysis)
        workflow.add_node("validate_consistency", self._validate_consistency)
        workflow.add_node("store_to_neo4j", self._store_to_neo4j)

        # Define the workflow edges
        workflow.set_entry_point("extract_characters")
        workflow.add_edge("extract_characters", "extract_events")
        workflow.add_edge("extract_events", "analyze_causality")
        workflow.add_edge("analyze_causality", "build_relationships")
        workflow.add_edge("build_relationships", "temporal_analysis")
        workflow.add_edge("temporal_analysis", "validate_consistency")
        workflow.add_edge("validate_consistency", "store_to_neo4j")
        workflow.add_edge("store_to_neo4j", END)

        return workflow.compile(checkpointer=self.memory)

    async def _extract_characters(self, state: ProcessingState) -> ProcessingState:
        """Extract characters using AI with enhanced prompts"""
        print(f"🎭 Extracting characters from batch {state['batch_id']}")

        # Prepare content for AI analysis
        content = "\n".join([chunk.get('content', '') for chunk in state['chunks']])

        character_prompt = """
        分析以下中文小说文本，提取所有角色信息。对每个角色，请提供：
        1. 角色姓名（主要名称和所有别称）
        2. 角色类型（主角/配角/反派/路人）
        3. 个性特征列表
        4. 首次出现位置
        5. 与其他角色的关系

        文本内容：
        {content}

        请以JSON格式返回结果，包含角色的详细信息和置信度分数。
        """

        try:
            response = await self.deepseek_client.generate_response(
                character_prompt.format(content=content),
                temperature=0.3
            )

            # Parse AI response and extract characters
            characters = self._parse_character_response(response)
            state['characters'] = characters
            state['processing_stage'] = "characters_extracted"

        except Exception as e:
            state['errors'].append(f"Character extraction failed: {e}")

        return state

    async def _extract_events(self, state: ProcessingState) -> ProcessingState:
        """Extract narrative events with temporal markers"""
        print(f"📝 Extracting events from batch {state['batch_id']}")

        content = "\n".join([chunk.get('content', '') for chunk in state['chunks']])

        event_prompt = """
        分析以下中文小说文本，提取所有重要的叙事事件。对每个事件，请提供：
        1. 事件描述
        2. 事件类型（对话/行动/揭示/冲突/解决）
        3. 重要性评分（0.0-1.0）
        4. 参与的角色
        5. 时间标记（如"三天后"、"同时"、"早晨"等）
        6. 事件的叙事功能（铺垫/冲突/高潮/解决等）

        文本内容：
        {content}

        请以JSON格式返回结果，按时间顺序排列事件。
        """

        try:
            response = await self.deepseek_client.generate_response(
                event_prompt.format(content=content),
                temperature=0.3
            )

            events = self._parse_event_response(response)
            state['events'] = events
            state['processing_stage'] = "events_extracted"

        except Exception as e:
            state['errors'].append(f"Event extraction failed: {e}")

        return state

    async def _analyze_causality(self, state: ProcessingState) -> ProcessingState:
        """Analyze causality relationships between events"""
        print(f"🔗 Analyzing causality for batch {state['batch_id']}")

        events = state.get('events', [])

        causality_prompt = """
        分析以下事件列表之间的因果关系。对每个因果关系，请提供：
        1. 原因事件ID
        2. 结果事件ID
        3. 因果关系类型（直接原因/间接原因/促成条件/阻止条件/催化剂）
        4. 因果强度（0.0-1.0）
        5. 推理说明

        事件列表：
        {events}

        请以JSON格式返回因果关系链。
        """

        try:
            events_text = json.dumps(events, ensure_ascii=False, indent=2)
            response = await self.deepseek_client.generate_response(
                causality_prompt.format(events=events_text),
                temperature=0.2
            )

            causality_chains = self._parse_causality_response(response)
            state['causality_chains'] = causality_chains
            state['processing_stage'] = "causality_analyzed"

        except Exception as e:
            state['errors'].append(f"Causality analysis failed: {e}")

        return state

    async def _build_relationships(self, state: ProcessingState) -> ProcessingState:
        """Build character relationship networks"""
        print(f"👥 Building relationships for batch {state['batch_id']}")

        characters = state.get('characters', {})
        events = state.get('events', [])

        # Analyze character interactions and relationships
        relationship_prompt = """
        基于以下角色和事件信息，分析角色之间的关系网络：
        1. 角色互动类型（友好/敌对/保护/影响）
        2. 关系强度（0.0-1.0）
        3. 关系变化（如果有）
        4. 情感色彩

        角色信息：{characters}
        事件信息：{events}

        请以JSON格式返回角色关系网络。
        """

        try:
            response = await self.deepseek_client.generate_response(
                relationship_prompt.format(
                    characters=json.dumps(characters, ensure_ascii=False),
                    events=json.dumps(events, ensure_ascii=False)
                ),
                temperature=0.3
            )

            relationships = self._parse_relationship_response(response)
            state['relationships'] = relationships
            state['processing_stage'] = "relationships_built"

        except Exception as e:
            state['errors'].append(f"Relationship building failed: {e}")

        return state

    async def _temporal_analysis(self, state: ProcessingState) -> ProcessingState:
        """Analyze temporal patterns and sequences"""
        print(f"⏰ Temporal analysis for batch {state['batch_id']}")

        events = state.get('events', [])

        # Extract temporal markers and sequence information
        temporal_markers = []
        for event in events:
            if 'temporal_markers' in event:
                temporal_markers.extend(event['temporal_markers'])

        state['temporal_markers'] = list(set(temporal_markers))
        state['processing_stage'] = "temporal_analyzed"

        return state

    async def _validate_consistency(self, state: ProcessingState) -> ProcessingState:
        """Validate logical consistency of extracted information"""
        print(f"✅ Validating consistency for batch {state['batch_id']}")

        # Perform consistency checks
        validation_errors = []

        # Check character consistency
        characters = state.get('characters', {})
        events = state.get('events', [])

        # Validate that all characters mentioned in events exist
        for event in events:
            for actor in event.get('primary_actors', []):
                if actor not in characters:
                    validation_errors.append(f"Character {actor} mentioned in event but not in character list")

        if validation_errors:
            state['errors'].extend(validation_errors)

        state['processing_stage'] = "validated"
        return state

    async def _store_to_neo4j(self, state: ProcessingState) -> ProcessingState:
        """Store processed data to Neo4j graph database"""
        print(f"💾 Storing to Neo4j for batch {state['batch_id']}")

        try:
            # Store characters
            characters = state.get('characters', {})
            for char_id, char_data in characters.items():
                char_data['character_id'] = char_id
                self.db.create_character_node(char_data)

            # Store events
            events = state.get('events', [])
            for event in events:
                self.db.create_event_node(event)

            # Store causality relationships
            causality_chains = state.get('causality_chains', [])
            for chain in causality_chains:
                self.db.create_causality_relationship(
                    chain['from_event'],
                    chain['to_event'],
                    chain['causality_type'],
                    chain['strength'],
                    chain.get('reasoning', '')
                )

            # Store character relationships
            relationships = state.get('relationships', [])
            for rel in relationships:
                self.db.create_character_relationship(
                    rel['character1'],
                    rel['character2'],
                    rel['relationship_type'],
                    rel['strength'],
                    rel.get('context', '')
                )

            state['processing_stage'] = "stored"
            print(f"✅ Successfully stored batch {state['batch_id']} to Neo4j")

        except Exception as e:
            state['errors'].append(f"Neo4j storage failed: {e}")
            print(f"❌ Failed to store batch {state['batch_id']}: {e}")

        return state

    def _parse_character_response(self, response: str) -> Dict[str, Any]:
        """Parse AI character extraction response"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
            return {}

    def _parse_event_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI event extraction response"""
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
        except:
            return []

    def _parse_causality_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI causality analysis response"""
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
        except:
            return []

    def _parse_relationship_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI relationship analysis response"""
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
        except:
            return []

    async def process_batch(self, chunks: List[Dict], batch_id: int, volume_id: int) -> Dict[str, Any]:
        """Process a batch of chunks using the LangGraph workflow"""

        # Initialize processing state
        initial_state = ProcessingState(
            batch_id=batch_id,
            volume_id=volume_id,
            chunks=chunks,
            characters={},
            events=[],
            relationships=[],
            causality_chains=[],
            temporal_markers=[],
            processing_stage="initialized",
            iteration_count=0,
            confidence_scores={},
            errors=[]
        )

        # Execute the workflow
        config = {"configurable": {"thread_id": f"batch_{batch_id}"}}
        final_state = await self.graph.ainvoke(initial_state, config)

        # Return processing results
        return {
            'batch_id': batch_id,
            'volume_id': volume_id,
            'processing_stage': final_state['processing_stage'],
            'characters_extracted': len(final_state['characters']),
            'events_extracted': len(final_state['events']),
            'relationships_built': len(final_state['relationships']),
            'causality_chains': len(final_state['causality_chains']),
            'errors': final_state['errors'],
            'success': len(final_state['errors']) == 0
        }

    def close(self):
        """Clean up resources"""
        self.db.close()

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of the LangGraph + Neo4j processor"""

    # Configuration
    neo4j_config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'user': os.getenv('NEO4J_USER', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'password')
    }

    deepseek_config = create_deepseek_config()

    # Initialize processor
    processor = LangGraphNovelProcessor(
        neo4j_uri=neo4j_config['uri'],
        neo4j_user=neo4j_config['user'],
        neo4j_password=neo4j_config['password'],
        deepseek_config=deepseek_config
    )

    try:
        # Example batch processing
        sample_chunks = [
            {'content': '御坂美琴是学园都市的Level 5超能力者。', 'chunk_id': 1},
            {'content': '今天她遇到了上条当麻，两人发生了争执。', 'chunk_id': 2}
        ]

        result = await processor.process_batch(sample_chunks, batch_id=1, volume_id=1)
        print(f"Processing result: {result}")

        # Example graph queries
        print("\n📊 Graph Analysis Results:")

        # Get character influence network
        influence_network = processor.db.get_character_influence_network("misaka_mikoto")
        print(f"Character influence network: {influence_network}")

        # Get event centrality scores
        centrality_scores = processor.db.get_event_centrality_scores()
        print(f"Top important events: {centrality_scores[:5]}")

        # Find causality path
        # causality_path = processor.db.find_causality_path("event_1", "event_2")
        # print(f"Causality path: {causality_path}")

    finally:
        processor.close()

if __name__ == "__main__":
    asyncio.run(main())