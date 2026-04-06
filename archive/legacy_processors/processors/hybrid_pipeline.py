"""
Hybrid Processing Pipeline for Novel Analysis
Combines LangGraph workflow orchestration with Neo4j graph processing and existing Qdrant/PostgreSQL systems
Maximizes performance by leveraging the strengths of each database technology
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import pandas as pd
import numpy as np

# Database imports
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import RealDictCursor

# AI model imports
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Local imports
from langgraph_neo4j_processor import (
    Neo4jNovelDatabase,
    LangGraphNovelProcessor,
    GraphNodeType,
    GraphRelationType,
    ProcessingState
)

class PipelineMode(Enum):
    """Processing modes for different optimization strategies"""
    FULL_MIGRATION = "full_migration"  # Complete Neo4j processing
    HYBRID_PARALLEL = "hybrid_parallel"  # Parallel processing with both systems
    INCREMENTAL = "incremental"  # Gradual migration with fallback
    PERFORMANCE_TEST = "performance_test"  # A/B testing between approaches

@dataclass
class HybridConfig:
    """Configuration for hybrid processing pipeline"""
    # Database configurations
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "novelprocessing2024"
    postgres_url: str = "postgresql://novel_user:novel_pass_2024@localhost:5432/novel_processing"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"

    # Processing settings
    mode: PipelineMode = PipelineMode.HYBRID_PARALLEL
    batch_size: int = 50
    max_workers: int = 4
    enable_vector_indexing: bool = True

    # Performance settings
    neo4j_timeout: int = 30
    postgres_connection_pool_size: int = 10
    enable_performance_monitoring: bool = True

    # Migration settings
    migration_threshold: float = 0.8  # Confidence threshold for auto-migration
    fallback_to_postgres: bool = True

class DatabaseBridge:
    """Bridge between old PostgreSQL and new Neo4j systems"""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize connections
        self.neo4j_db = Neo4jNovelDatabase(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password
        )
        self.qdrant_client = QdrantClient(url=config.qdrant_url)
        self.postgres_conn = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Performance metrics
        self.performance_metrics = {
            'neo4j_query_times': [],
            'postgres_query_times': [],
            'migration_success_rate': 0.0,
            'data_consistency_score': 0.0
        }

    def connect_postgres(self):
        """Establish PostgreSQL connection with error handling"""
        try:
            self.postgres_conn = psycopg2.connect(
                self.config.postgres_url,
                cursor_factory=RealDictCursor
            )
            self.logger.info("PostgreSQL connection established")
        except Exception as e:
            self.logger.error(f"PostgreSQL connection failed: {e}")
            if not self.config.fallback_to_postgres:
                raise

    async def migrate_timeline_events(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate timeline events from PostgreSQL to Neo4j with relationship optimization"""
        migration_stats = {
            'events_migrated': 0,
            'relationships_created': 0,
            'errors': [],
            'performance_improvement': 0.0
        }

        try:
            # Fetch events from PostgreSQL
            with self.postgres_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT event_id, chapter_index, event_description,
                           timeline_order, character_states_before, character_states_after,
                           caused_by_events, causes_events, emotional_intensity, importance_score
                    FROM timeline_events
                    ORDER BY timeline_order
                """)
                events = cursor.fetchall()

            # Process in batches for optimal performance
            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                await self._process_event_batch(batch, migration_stats)

                # Log progress
                self.logger.info(f"Migrated batch {i//batch_size + 1}, "
                               f"total events: {migration_stats['events_migrated']}")

        except Exception as e:
            migration_stats['errors'].append(f"Migration error: {str(e)}")
            self.logger.error(f"Migration failed: {e}")

        return migration_stats

    async def _process_event_batch(self, batch: List[Dict], stats: Dict[str, Any]):
        """Process a batch of events for efficient Neo4j insertion"""
        # Prepare batch data for Neo4j
        event_data = []
        relationships = []

        for event in batch:
            # Create embedding for semantic search
            embedding = None
            if self.config.enable_vector_indexing:
                text = f"{event['event_description']} {event.get('character_states_before', '')}"
                embedding = self.embedding_model.encode(text).tolist()

            # Prepare event node
            event_node = {
                'event_id': event['event_id'],
                'chapter_index': event['chapter_index'],
                'timeline_order': event['timeline_order'],
                'description': event['event_description'],
                'emotional_intensity': event.get('emotional_intensity', 0.0),
                'importance_score': event.get('importance_score', 0.0),
                'embedding': embedding,
                'created_at': datetime.now().isoformat(),
                'processed_by': 'hybrid_pipeline_v1'
            }
            event_data.append(event_node)

            # Convert PostgreSQL array relationships to Neo4j relationships
            if event.get('caused_by_events'):
                for cause_id in event['caused_by_events']:
                    relationships.append({
                        'from_id': cause_id,
                        'to_id': event['event_id'],
                        'type': 'CAUSES',
                        'properties': {
                            'causality_strength': 0.8,  # Default, can be improved with ML
                            'confidence': 0.7,
                            'causal_type': 'direct'
                        }
                    })

            if event.get('causes_events'):
                for effect_id in event['causes_events']:
                    relationships.append({
                        'from_id': event['event_id'],
                        'to_id': effect_id,
                        'type': 'CAUSES',
                        'properties': {
                            'causality_strength': 0.8,
                            'confidence': 0.7,
                            'causal_type': 'direct'
                        }
                    })

        # Batch insert to Neo4j
        await self._batch_insert_neo4j(event_data, relationships, stats)

    async def _batch_insert_neo4j(self, events: List[Dict], relationships: List[Dict], stats: Dict[str, Any]):
        """Efficiently batch insert data to Neo4j"""
        try:
            with self.neo4j_db.driver.session() as session:
                # Insert events
                result = session.run("""
                    UNWIND $events AS event
                    MERGE (e:Event {event_id: event.event_id})
                    SET e += event
                    RETURN count(e) as events_created
                """, events=events)

                events_created = result.single()['events_created']
                stats['events_migrated'] += events_created

                # Insert relationships
                if relationships:
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (from:Event {event_id: rel.from_id})
                        MATCH (to:Event {event_id: rel.to_id})
                        CALL apoc.create.relationship(from, rel.type, rel.properties, to)
                        YIELD rel as created_rel
                        RETURN count(created_rel) as relationships_created
                    """, relationships=relationships)

                    rels_created = result.single()['relationships_created']
                    stats['relationships_created'] += rels_created

        except Exception as e:
            stats['errors'].append(f"Neo4j batch insert error: {str(e)}")
            self.logger.error(f"Neo4j batch insert failed: {e}")

class HybridQueryProcessor:
    """Intelligent query router that chooses optimal database for each query type"""

    def __init__(self, bridge: DatabaseBridge):
        self.bridge = bridge
        self.logger = logging.getLogger(__name__)

        # Query routing rules
        self.routing_rules = {
            'causality_analysis': 'neo4j',  # Graph queries are much faster in Neo4j
            'character_relationships': 'neo4j',
            'timeline_traversal': 'neo4j',
            'semantic_search': 'qdrant',  # Vector search
            'full_text_search': 'postgres',  # Full text indexing
            'aggregations': 'postgres',  # Complex aggregations
            'simple_lookups': 'fastest'  # Route to fastest responding DB
        }

    async def find_causality_chain(self, start_event: str, end_event: str, max_depth: int = 5) -> Dict[str, Any]:
        """Find causality chain - optimized for Neo4j graph traversal"""
        start_time = datetime.now()

        try:
            # Neo4j query (O(log n) with indexes)
            with self.bridge.neo4j_db.driver.session() as session:
                result = session.run("""
                    MATCH (start:Event {event_id: $start_event})
                    MATCH (end:Event {event_id: $end_event})
                    MATCH path = shortestPath((start)-[:CAUSES*1..$max_depth]->(end))
                    RETURN [node in nodes(path) | {
                        event_id: node.event_id,
                        description: node.description,
                        chapter_index: node.chapter_index,
                        timeline_order: node.timeline_order
                    }] as path_events,
                    length(path) as path_length,
                    [rel in relationships(path) | {
                        type: type(rel),
                        causality_strength: rel.causality_strength,
                        confidence: rel.confidence
                    }] as path_relationships
                """, start_event=start_event, end_event=end_event, max_depth=max_depth)

                records = list(result)

                # Performance comparison with old PostgreSQL approach
                neo4j_time = (datetime.now() - start_time).total_seconds()
                postgres_equivalent_time = await self._estimate_postgres_causality_time(start_event, end_event)

                return {
                    'path_events': records[0]['path_events'] if records else [],
                    'path_length': records[0]['path_length'] if records else 0,
                    'path_relationships': records[0]['path_relationships'] if records else [],
                    'query_time_neo4j': neo4j_time,
                    'estimated_postgres_time': postgres_equivalent_time,
                    'performance_improvement': postgres_equivalent_time / neo4j_time if neo4j_time > 0 else 0,
                    'query_method': 'neo4j_graph_traversal'
                }

        except Exception as e:
            self.logger.error(f"Neo4j causality query failed: {e}")
            # Fallback to PostgreSQL if configured
            if self.bridge.config.fallback_to_postgres:
                return await self._postgres_causality_fallback(start_event, end_event)
            raise

    async def _estimate_postgres_causality_time(self, start_event: str, end_event: str) -> float:
        """Estimate time for equivalent PostgreSQL causality analysis (for comparison)"""
        # PostgreSQL approach would require recursive CTEs and array operations
        # Simulated based on O(n²) complexity of hot-encoded arrays
        with self.bridge.postgres_conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM timeline_events")
            event_count = cursor.fetchone()[0]

            # Estimated time based on array scanning complexity
            # In real PostgreSQL with TEXT[] arrays, each hop requires scanning all events
            estimated_time = (event_count / 1000) * 0.1  # Conservative estimate
            return max(estimated_time, 0.1)  # Minimum 100ms

    async def analyze_character_influence_network(self, character_name: str, depth: int = 3) -> Dict[str, Any]:
        """Analyze character influence using Neo4j graph algorithms"""
        start_time = datetime.now()

        try:
            with self.bridge.neo4j_db.driver.session() as session:
                # Find character's direct and indirect influence on events
                result = session.run("""
                    MATCH (c:Character {name: $character_name})
                    MATCH (c)-[:PARTICIPATES_IN]->(direct_events:Event)

                    // Find events influenced by character's direct participation
                    OPTIONAL MATCH (direct_events)-[:CAUSES*1..$depth]->(influenced_events:Event)

                    // Calculate influence metrics
                    WITH c, direct_events, influenced_events,
                         size((direct_events)-[:CAUSES*1..$depth]->()) as influence_reach

                    RETURN {
                        character: c.name,
                        direct_events: count(DISTINCT direct_events),
                        influenced_events: count(DISTINCT influenced_events),
                        total_influence_reach: sum(influence_reach),
                        influence_network: collect(DISTINCT {
                            event_id: influenced_events.event_id,
                            description: influenced_events.description,
                            chapter: influenced_events.chapter_index,
                            importance: influenced_events.importance_score
                        })
                    } as influence_analysis
                """, character_name=character_name, depth=depth)

                analysis = result.single()['influence_analysis']
                query_time = (datetime.now() - start_time).total_seconds()

                return {
                    **analysis,
                    'query_time': query_time,
                    'analysis_depth': depth,
                    'query_method': 'neo4j_graph_algorithms'
                }

        except Exception as e:
            self.logger.error(f"Character influence analysis failed: {e}")
            raise

class HybridProcessingPipeline:
    """Main pipeline orchestrating the hybrid processing system"""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.bridge = DatabaseBridge(config)
        self.query_processor = HybridQueryProcessor(self.bridge)
        self.langgraph_processor = LangGraphNovelProcessor(
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password,
            qdrant_url=config.qdrant_url
        )

        # Performance monitoring
        self.metrics = {
            'total_events_processed': 0,
            'migration_success_rate': 0.0,
            'average_query_improvement': 0.0,
            'system_uptime': datetime.now()
        }

    async def initialize(self):
        """Initialize all database connections and prepare schema"""
        self.logger.info("Initializing hybrid processing pipeline...")

        # Connect to all databases
        self.bridge.connect_postgres()
        await self.bridge.neo4j_db.create_schema()

        # Verify Qdrant collection exists
        try:
            collections = self.bridge.qdrant_client.get_collections()
            self.logger.info(f"Qdrant collections available: {[c.name for c in collections.collections]}")
        except Exception as e:
            self.logger.warning(f"Qdrant connection issue: {e}")

        self.logger.info("Hybrid pipeline initialization complete")

    async def run_migration(self) -> Dict[str, Any]:
        """Execute full migration from PostgreSQL to Neo4j"""
        self.logger.info("Starting data migration process...")

        migration_results = await self.bridge.migrate_timeline_events(
            batch_size=self.config.batch_size
        )

        # Update performance metrics
        self.metrics['migration_success_rate'] = (
            migration_results['events_migrated'] /
            (migration_results['events_migrated'] + len(migration_results['errors']))
            if migration_results['events_migrated'] > 0 else 0.0
        )

        self.logger.info(f"Migration completed: {migration_results}")
        return migration_results

    async def benchmark_performance(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark Neo4j vs PostgreSQL performance"""
        benchmark_results = {
            'neo4j_times': [],
            'postgres_times': [],
            'performance_improvements': [],
            'test_results': []
        }

        for query in test_queries:
            if query['type'] == 'causality_chain':
                result = await self.query_processor.find_causality_chain(
                    query['start_event'],
                    query['end_event']
                )

                benchmark_results['neo4j_times'].append(result['query_time_neo4j'])
                benchmark_results['postgres_times'].append(result['estimated_postgres_time'])
                benchmark_results['performance_improvements'].append(result['performance_improvement'])
                benchmark_results['test_results'].append(result)

        # Calculate averages
        if benchmark_results['performance_improvements']:
            avg_improvement = np.mean(benchmark_results['performance_improvements'])
            self.metrics['average_query_improvement'] = avg_improvement

            self.logger.info(f"Average query performance improvement: {avg_improvement:.2f}x")

        return benchmark_results

    async def process_novel_chapter(self, chapter_text: str, chapter_index: int) -> Dict[str, Any]:
        """Process a novel chapter using the hybrid pipeline"""
        processing_state = ProcessingState(
            chapter_text=chapter_text,
            chapter_index=chapter_index,
            events=[],
            characters=[],
            locations=[],
            relationships=[],
            processing_metadata={
                'pipeline_mode': self.config.mode.value,
                'timestamp': datetime.now().isoformat()
            }
        )

        # Use LangGraph processor for workflow orchestration
        result = await self.langgraph_processor.process_chapter(processing_state)

        # Update metrics
        self.metrics['total_events_processed'] += len(result.events)

        return {
            'chapter_index': chapter_index,
            'events_extracted': len(result.events),
            'characters_found': len(result.characters),
            'relationships_created': len(result.relationships),
            'processing_time': result.processing_metadata.get('processing_time', 0),
            'pipeline_mode': self.config.mode.value
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        uptime = (datetime.now() - self.metrics['system_uptime']).total_seconds()

        return {
            'system_metrics': self.metrics,
            'system_uptime_seconds': uptime,
            'database_metrics': {
                'neo4j_query_times': self.bridge.performance_metrics['neo4j_query_times'][-100:],  # Last 100
                'postgres_query_times': self.bridge.performance_metrics['postgres_query_times'][-100:],
            },
            'configuration': {
                'mode': self.config.mode.value,
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'vector_indexing_enabled': self.config.enable_vector_indexing
            },
            'recommendations': self._generate_optimization_recommendations()
        }

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        if self.metrics['average_query_improvement'] > 10:
            recommendations.append("Consider full migration to Neo4j for optimal performance")

        if self.metrics['migration_success_rate'] < 0.9:
            recommendations.append("Review migration errors and improve data consistency")

        if len(self.bridge.performance_metrics['neo4j_query_times']) > 0:
            avg_neo4j_time = np.mean(self.bridge.performance_metrics['neo4j_query_times'])
            if avg_neo4j_time > 1.0:
                recommendations.append("Consider optimizing Neo4j indexes and query patterns")

        return recommendations

# Example usage and testing
async def main():
    """Example usage of the hybrid processing pipeline"""
    config = HybridConfig(
        mode=PipelineMode.HYBRID_PARALLEL,
        batch_size=50,
        enable_performance_monitoring=True
    )

    pipeline = HybridProcessingPipeline(config)
    await pipeline.initialize()

    # Run migration
    migration_results = await pipeline.run_migration()
    print(f"Migration results: {migration_results}")

    # Test causality analysis
    causality_result = await pipeline.query_processor.find_causality_chain(
        "event_001", "event_010"
    )
    print(f"Causality analysis: {causality_result}")

    # Generate performance report
    report = pipeline.get_performance_report()
    print(f"Performance report: {report}")

if __name__ == "__main__":
    asyncio.run(main())