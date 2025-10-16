#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Workflow Module - Phase 3 Integration
Orchestrated AI workflows for novel processing with Neo4j backend
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)


class ProcessingState(TypedDict):
    """State object for LangGraph workflow - passed between nodes"""
    # Input data
    batch_id: int
    volume_id: int
    chunks: List[Dict[str, Any]]  # Text chunks to process

    # Extracted entities
    characters: Dict[str, Any]  # character_id -> character data
    events: List[Dict[str, Any]]  # List of timeline events
    causal_links: List[Dict[str, Any]]  # List of causality relationships

    # Processing metadata
    processing_stage: str
    iteration_count: int
    confidence_scores: Dict[str, float]
    errors: List[str]

    # Context from previous batches (for continuity)
    previous_characters: Dict[str, Any]
    previous_events: List[str]  # event_ids


class WorkflowStage(Enum):
    """Processing stages in the workflow"""
    INITIALIZED = "initialized"
    CHARACTERS_EXTRACTED = "characters_extracted"
    EVENTS_EXTRACTED = "events_extracted"
    CAUSALITY_ANALYZED = "causality_analyzed"
    VALIDATED = "validated"
    STORED = "stored"
    COMPLETED = "completed"
    FAILED = "failed"


class LangGraphWorkflow:
    """
    LangGraph-based processing workflow
    Orchestrates AI agents for character extraction, event extraction, and causality analysis
    """

    def __init__(self, processor, config):
        """
        Initialize workflow with reference to UnifiedNovelProcessor

        Args:
            processor: UnifiedNovelProcessor instance (for AI client, database adapter)
            config: ProcessingConfig instance
        """
        self.processor = processor
        self.config = config

        # Build the workflow graph (checkpointing disabled for simplicity)
        self.graph = self._build_workflow_graph()

        logger.info("✅ LangGraph workflow initialized")

    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph processing workflow

        Workflow stages:
        1. extract_characters -> Extract character data from text
        2. extract_events -> Extract timeline events
        3. analyze_causality -> Analyze causal relationships between events
        4. validate_results -> Validate extracted data
        5. store_to_database -> Store to Neo4j/PostgreSQL via DatabaseAdapter
        """

        workflow = StateGraph(ProcessingState)

        # Add processing nodes
        workflow.add_node("extract_characters", self._extract_characters)
        workflow.add_node("extract_events", self._extract_events)
        workflow.add_node("analyze_causality", self._analyze_causality)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("store_to_database", self._store_to_database)

        # Define linear workflow edges
        workflow.set_entry_point("extract_characters")
        workflow.add_edge("extract_characters", "extract_events")
        workflow.add_edge("extract_events", "analyze_causality")
        workflow.add_edge("analyze_causality", "validate_results")
        workflow.add_edge("validate_results", "store_to_database")
        workflow.add_edge("store_to_database", END)

        # Compile workflow (checkpointing disabled for simplicity)
        return workflow.compile()

    async def _extract_characters(self, state: ProcessingState) -> ProcessingState:
        """
        Node 1: Extract characters from text using AI
        """
        batch_id = state['batch_id']
        print()
        print("=" * 100)
        print(f"🎭 CHARACTER EXTRACTION - Batch {batch_id}")
        print("=" * 100)

        try:
            # Combine chunks into single text
            content = "\n\n".join([chunk.get('content', '') for chunk in state['chunks']])
            chunk_count = len(state['chunks'])

            print(f"  📄 Processing {chunk_count} text chunks ({len(content)} characters)")

            # Truncate if too long (DeepSeek context limit)
            max_chars = 30000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
                print(f"  ⚠️  Content truncated to {max_chars} characters")

            # Show previously recognized characters
            previous_chars = state.get('previous_characters', {})
            if previous_chars:
                print(f"  📋 Previously recognized characters: {len(previous_chars)}")
                prev_names = [c.get('name', 'Unknown') for c in list(previous_chars.values())[:10]]
                print(f"     {', '.join(prev_names)}")

            # Build AI prompt for character extraction
            print(f"  🤖 Querying DeepSeek AI for character extraction...")
            prompt = self._build_character_extraction_prompt(content, previous_chars)

            # Call AI via processor's DeepSeek client
            response = await self.processor.reasoner.deepseek_client.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=2000
            )

            # Parse JSON response
            print(f"  📥 Parsing AI response...")
            characters = self._parse_json_response(response, expected_key='characters')

            # Categorize as new vs. recognized
            new_characters = []
            recognized_characters = []

            for char in characters:
                char_name = char.get('name', '')
                is_recognized = any(
                    prev.get('name', '') == char_name
                    for prev in previous_chars.values()
                )

                if is_recognized:
                    recognized_characters.append(char_name)
                else:
                    new_characters.append(char_name)

            state['characters'] = characters
            state['processing_stage'] = WorkflowStage.CHARACTERS_EXTRACTED.value
            state['confidence_scores']['character_extraction'] = 0.85

            print()
            print(f"  ✅ Character Extraction Complete:")
            print(f"     Total characters found: {len(characters)}")
            print(f"     New discoveries: {len(new_characters)}")
            print(f"     Previously recognized: {len(recognized_characters)}")

            if new_characters:
                print(f"     📝 New characters: {', '.join(new_characters[:10])}")
            if recognized_characters:
                print(f"     ✓ Recognized: {', '.join(recognized_characters[:10])}")

            print("=" * 100)

        except Exception as e:
            print()
            print(f"  ❌ Character extraction failed: {e}")
            print("=" * 100)
            state['errors'].append(f"Character extraction: {str(e)}")
            state['characters'] = {}

        return state

    async def _extract_events(self, state: ProcessingState) -> ProcessingState:
        """
        Node 2: Extract timeline events from text using AI
        """
        batch_id = state['batch_id']
        print()
        print("=" * 100)
        print(f"📝 EVENT EXTRACTION - Batch {batch_id}")
        print("=" * 100)

        try:
            content = "\n\n".join([chunk.get('content', '') for chunk in state['chunks']])

            print(f"  📄 Processing text content ({len(content)} characters)")

            # Truncate if needed
            max_chars = 30000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
                print(f"  ⚠️  Content truncated to {max_chars} characters")

            # Show character context
            characters = state.get('characters', {})
            if isinstance(characters, dict):
                char_count = len(characters)
            elif isinstance(characters, list):
                char_count = len(characters)
            else:
                char_count = 0

            print(f"  🎭 Using {char_count} characters as context")

            # Show previous events
            previous_events = state.get('previous_events', [])
            if previous_events:
                print(f"  📋 Building on {len(previous_events)} previous events")

            # Build prompt with character context
            print(f"  🤖 Querying DeepSeek AI for event extraction...")
            prompt = self._build_event_extraction_prompt(
                content,
                characters,
                previous_events
            )

            # Call AI
            response = await self.processor.reasoner.deepseek_client.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=3000
            )

            # Parse events
            print(f"  📥 Parsing AI response...")
            events = self._parse_json_response(response, expected_key='events')

            # Categorize events by type
            event_types = {}
            for event in events:
                event_type = event.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1

            state['events'] = events
            state['processing_stage'] = WorkflowStage.EVENTS_EXTRACTED.value
            state['confidence_scores']['event_extraction'] = 0.8

            print()
            print(f"  ✅ Event Extraction Complete:")
            print(f"     Total events extracted: {len(events)}")

            if event_types:
                print(f"     Event breakdown:")
                for etype, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"       • {etype}: {count}")

            # Show sample events
            if events:
                print(f"     Sample events:")
                for i, evt in enumerate(events[:3], 1):
                    desc = evt.get('description', 'No description')[:60]
                    print(f"       {i}. [{evt.get('event_type', 'unknown')}] {desc}...")

            print("=" * 100)

        except Exception as e:
            print()
            print(f"  ❌ Event extraction failed: {e}")
            print("=" * 100)
            state['errors'].append(f"Event extraction: {str(e)}")
            state['events'] = []

        return state

    async def _analyze_causality(self, state: ProcessingState) -> ProcessingState:
        """
        Node 3: Analyze causal relationships between events
        """
        batch_id = state['batch_id']
        print()
        print("=" * 100)
        print(f"🔗 CAUSALITY ANALYSIS - Batch {batch_id}")
        print("=" * 100)

        try:
            events = state.get('events', [])

            print(f"  📊 Analyzing causal relationships between {len(events)} events")

            if len(events) < 2:
                print(f"  ⚠️  Not enough events for causality analysis (minimum 2 required)")
                print("=" * 100)
                state['causal_links'] = []
                state['processing_stage'] = WorkflowStage.CAUSALITY_ANALYZED.value
                return state

            # Show event list
            print(f"  📋 Event list:")
            for i, evt in enumerate(events[:5], 1):
                desc = evt.get('description', 'No description')[:50]
                print(f"     {i}. {desc}...")

            if len(events) > 5:
                print(f"     ... and {len(events) - 5} more events")

            # Build prompt for causality analysis
            print(f"  🤖 Querying DeepSeek AI for causality analysis...")
            prompt = self._build_causality_analysis_prompt(events)

            # Call AI
            response = await self.processor.reasoner.deepseek_client.generate_response(
                prompt,
                temperature=0.2,  # Lower temperature for logical reasoning
                max_tokens=2000
            )

            # Parse causal links
            print(f"  📥 Parsing AI response...")
            causal_links = self._parse_json_response(response, expected_key='causal_links')

            # Categorize by causality type
            link_types = {}
            for link in causal_links:
                link_type = link.get('causality_type', 'unknown')
                link_types[link_type] = link_types.get(link_type, 0) + 1

            state['causal_links'] = causal_links
            state['processing_stage'] = WorkflowStage.CAUSALITY_ANALYZED.value
            state['confidence_scores']['causality_analysis'] = 0.75

            print()
            print(f"  ✅ Causality Analysis Complete:")
            print(f"     Total causal relationships: {len(causal_links)}")

            if link_types:
                print(f"     Relationship types:")
                for ltype, count in sorted(link_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"       • {ltype}: {count}")

            # Show sample causal links
            if causal_links:
                print(f"     Sample causal chains:")
                for i, link in enumerate(causal_links[:3], 1):
                    from_evt = link.get('from_event', 'unknown')
                    to_evt = link.get('to_event', 'unknown')
                    ctype = link.get('causality_type', 'unknown')
                    strength = link.get('strength', 0)
                    print(f"       {i}. [{ctype}] {from_evt} → {to_evt} (strength: {strength:.2f})")

            print("=" * 100)

        except Exception as e:
            print()
            print(f"  ❌ Causality analysis failed: {e}")
            print("=" * 100)
            state['errors'].append(f"Causality analysis: {str(e)}")
            state['causal_links'] = []

        return state

    async def _validate_results(self, state: ProcessingState) -> ProcessingState:
        """
        Node 4: Validate extracted data for consistency
        """
        batch_id = state['batch_id']
        print()
        print("=" * 100)
        print(f"✓ VALIDATION - Batch {batch_id}")
        print("=" * 100)

        try:
            validation_errors = []

            # Validate characters
            characters = state.get('characters', {})
            char_count = len(characters) if characters else 0
            print(f"  🎭 Validating {char_count} characters...")
            if not characters:
                validation_errors.append("No characters extracted")
                print(f"     ⚠️  No characters extracted")
            else:
                print(f"     ✓ Characters: OK")

            # Validate events
            events = state.get('events', [])
            event_count = len(events)
            print(f"  📝 Validating {event_count} events...")
            if not events:
                validation_errors.append("No events extracted")
                print(f"     ⚠️  No events extracted")
            else:
                print(f"     ✓ Events: OK")

            # Validate causal links reference existing events
            causal_links = state.get('causal_links', [])
            link_count = len(causal_links)
            print(f"  🔗 Validating {link_count} causal links...")

            event_ids = {evt.get('event_id') for evt in events if 'event_id' in evt}
            invalid_links = 0

            for link in causal_links:
                from_event = link.get('from_event')
                to_event = link.get('to_event')

                if from_event not in event_ids:
                    validation_errors.append(f"Causal link references unknown event: {from_event}")
                    invalid_links += 1
                if to_event not in event_ids:
                    validation_errors.append(f"Causal link references unknown event: {to_event}")
                    invalid_links += 1

            if invalid_links > 0:
                print(f"     ⚠️  {invalid_links} invalid causal link references")
            else:
                print(f"     ✓ Causal links: OK")

            print()
            if validation_errors:
                print(f"  ⚠️  Validation warnings: {len(validation_errors)}")
                for err in validation_errors[:5]:  # Show first 5
                    print(f"     - {err}")
                state['errors'].extend(validation_errors)
            else:
                print(f"  ✅ All validation checks passed")

            state['processing_stage'] = WorkflowStage.VALIDATED.value
            print("=" * 100)

        except Exception as e:
            print()
            print(f"  ❌ Validation failed: {e}")
            print("=" * 100)
            state['errors'].append(f"Validation: {str(e)}")

        return state

    async def _store_to_database(self, state: ProcessingState) -> ProcessingState:
        """
        Node 5: Store extracted data to database via DatabaseAdapter
        """
        batch_id = state['batch_id']
        print()
        print("=" * 100)
        print(f"💾 DATABASE STORAGE - Batch {batch_id}")
        print("=" * 100)

        try:
            stored_events = 0
            stored_links = 0
            failed_events = 0
            failed_links = 0

            # Store events via DatabaseAdapter
            if self.processor.db_adapter:
                from database.base_adapter import TimelineEvent as DBTimelineEvent, CausalLink as DBCausalLink

                events = state.get('events', [])
                causal_links = state.get('causal_links', [])

                print(f"  💾 Storing to {self.processor.db_adapter.__class__.__name__}...")
                print(f"  📝 Processing {len(events)} events...")

                # Store events
                for i, event_data in enumerate(events, 1):
                    try:
                        db_event = DBTimelineEvent(
                            event_id=event_data.get('event_id', f"evt_{state['batch_id']}_{stored_events}"),
                            volume_id=state['volume_id'],
                            batch_id=state['batch_id'],
                            description=event_data.get('description', ''),
                            event_type=event_data.get('event_type', 'action'),
                            importance_score=event_data.get('importance_score', 0.5),
                            chronological_order=event_data.get('chronological_order', stored_events),
                            primary_actors=event_data.get('primary_actors', []),
                            created_at=datetime.now()
                        )

                        result = await self.processor.db_adapter.store_event(db_event)
                        if result:
                            stored_events += 1
                            if i <= 3:  # Show first 3 events
                                desc = event_data.get('description', '')[:50]
                                print(f"     ✓ Event {i}: {desc}...")
                        else:
                            failed_events += 1
                    except Exception as e:
                        failed_events += 1
                        if failed_events <= 2:  # Show first 2 errors
                            print(f"     ⚠️  Failed to store event {i}: {e}")

                if len(events) > 3:
                    print(f"     ... and {len(events) - 3} more events")

                print(f"  🔗 Processing {len(causal_links)} causal links...")

                # Store causal links
                for i, link_data in enumerate(causal_links, 1):
                    try:
                        db_link = DBCausalLink(
                            from_event=link_data.get('from_event'),
                            to_event=link_data.get('to_event'),
                            causality_type=link_data.get('causality_type', 'direct_cause'),
                            strength=link_data.get('strength', 0.5),
                            reasoning=link_data.get('reasoning', ''),
                            confidence=link_data.get('confidence', 0.5)
                        )

                        result = await self.processor.db_adapter.store_causal_link(db_link)
                        if result:
                            stored_links += 1
                            if i <= 2:  # Show first 2 links
                                from_evt = link_data.get('from_event', 'unknown')
                                to_evt = link_data.get('to_event', 'unknown')
                                print(f"     ✓ Link {i}: {from_evt} → {to_evt}")
                        else:
                            failed_links += 1
                    except Exception as e:
                        failed_links += 1
                        if failed_links <= 2:  # Show first 2 errors
                            print(f"     ⚠️  Failed to store causal link {i}: {e}")

                if len(causal_links) > 2:
                    print(f"     ... and {len(causal_links) - 2} more links")

                print()
                print(f"  ✅ Database Storage Complete:")
                print(f"     Events stored: {stored_events}/{len(events)}")
                print(f"     Causal links stored: {stored_links}/{len(causal_links)}")
                if failed_events > 0 or failed_links > 0:
                    print(f"     ⚠️  Failed: {failed_events} events, {failed_links} links")

            else:
                print(f"  ⚠️  DatabaseAdapter not available, skipping storage")

            state['processing_stage'] = WorkflowStage.STORED.value
            print("=" * 100)

        except Exception as e:
            print()
            print(f"  ❌ Database storage failed: {e}")
            print("=" * 100)
            state['errors'].append(f"Database storage: {str(e)}")

        return state

    # ============================================================================
    # Helper Methods: Prompt Building
    # ============================================================================

    def _build_character_extraction_prompt(self, content: str, previous_characters: Dict) -> str:
        """Build AI prompt for character extraction"""
        return f"""分析以下中文小说文本，提取所有角色信息。

文本内容：
{content}

请以JSON格式返回结果，格式如下：
{{
    "characters": [
        {{
            "character_id": "char_001",
            "name": "角色姓名",
            "aliases": ["别称1", "别称2"],
            "character_type": "protagonist|supporting|antagonist",
            "personality_traits": ["特征1", "特征2"],
            "confidence_score": 0.9
        }}
    ]
}}

只返回JSON，不要其他说明文字。"""

    def _build_event_extraction_prompt(self, content: str, characters, previous_events: List) -> str:
        """Build AI prompt for event extraction"""
        # Handle both dict and list formats for characters
        if isinstance(characters, dict):
            char_data = list(characters.values())
        else:
            char_data = characters if isinstance(characters, list) else []

        char_list = ", ".join([c.get('name', '') for c in char_data][:10])

        return f"""分析以下中文小说文本，提取所有重要的叙事事件。

已知角色：{char_list}

文本内容：
{content}

请以JSON格式返回结果，格式如下：
{{
    "events": [
        {{
            "event_id": "evt_001",
            "description": "事件描述",
            "event_type": "action|dialogue|revelation|conflict",
            "importance_score": 0.8,
            "chronological_order": 1,
            "primary_actors": ["角色1", "角色2"]
        }}
    ]
}}

只返回JSON，不要其他说明文字。"""

    def _build_causality_analysis_prompt(self, events: List[Dict]) -> str:
        """Build AI prompt for causality analysis"""
        event_list = "\n".join([
            f"{i+1}. [{evt.get('event_id')}] {evt.get('description', '')}"
            for i, evt in enumerate(events)
        ])

        return f"""分析以下事件之间的因果关系。

事件列表：
{event_list}

请识别事件之间的因果关系，以JSON格式返回：
{{
    "causal_links": [
        {{
            "from_event": "evt_001",
            "to_event": "evt_002",
            "causality_type": "direct_cause|indirect_cause|enablement",
            "strength": 0.9,
            "reasoning": "因果关系说明",
            "confidence": 0.85
        }}
    ]
}}

只返回JSON，不要其他说明文字。"""

    def _parse_json_response(self, response: str, expected_key: str) -> Any:
        """Parse JSON from AI response, handling markdown code blocks"""
        try:
            # Remove markdown code blocks if present
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()

            # Parse JSON
            data = json.loads(response)

            # Return expected key or empty default
            if expected_key in data:
                return data[expected_key]
            else:
                logger.warning(f"Expected key '{expected_key}' not found in response")
                return [] if expected_key.endswith('s') else {}

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response was: {response[:200]}...")
            return [] if expected_key.endswith('s') else {}

    # ============================================================================
    # Public API
    # ============================================================================

    async def process_batch(self, batch_id: int, volume_id: int, chunks: List[Dict[str, Any]],
                          previous_characters: Dict = None, previous_events: List = None) -> Dict[str, Any]:
        """
        Process a batch of text chunks through the LangGraph workflow

        Args:
            batch_id: Batch identifier
            volume_id: Volume identifier
            chunks: List of text chunks to process
            previous_characters: Characters from previous batches (for continuity)
            previous_events: Event IDs from previous batches

        Returns:
            Final state dictionary with extracted data
        """
        from datetime import datetime
        batch_start_time = datetime.now()

        print()
        print("#" * 100)
        print(f"🚀 STARTING WORKFLOW - Batch {batch_id} | Volume {volume_id}")
        print("#" * 100)
        print(f"  Chunks to process: {len(chunks)}")
        print(f"  Previous characters: {len(previous_characters or {})}")
        print(f"  Previous events: {len(previous_events or [])}")
        print("#" * 100)

        # Initialize state
        initial_state = ProcessingState(
            batch_id=batch_id,
            volume_id=volume_id,
            chunks=chunks,
            characters={},
            events=[],
            causal_links=[],
            processing_stage=WorkflowStage.INITIALIZED.value,
            iteration_count=0,
            confidence_scores={},
            errors=[],
            previous_characters=previous_characters or {},
            previous_events=previous_events or []
        )

        # Run workflow
        config = {"configurable": {"thread_id": f"batch_{batch_id}"}}

        try:
            final_state = await self.graph.ainvoke(initial_state, config)

            batch_end_time = datetime.now()
            batch_duration = (batch_end_time - batch_start_time).total_seconds()

            print()
            print("#" * 100)
            print(f"✅ BATCH {batch_id} WORKFLOW COMPLETE")
            print("#" * 100)
            print(f"  Processing stage: {final_state.get('processing_stage')}")
            print(f"  Characters extracted: {len(final_state.get('characters', {}))}")
            print(f"  Events extracted: {len(final_state.get('events', []))}")
            print(f"  Causal links identified: {len(final_state.get('causal_links', []))}")
            print(f"  Processing time: {batch_duration:.2f} seconds")

            errors = final_state.get('errors', [])
            if errors:
                print(f"  ⚠️  Errors encountered: {len(errors)}")
            else:
                print(f"  ✓ No errors")

            print("#" * 100)
            print()

            return final_state

        except Exception as e:
            batch_end_time = datetime.now()
            batch_duration = (batch_end_time - batch_start_time).total_seconds()

            print()
            print("#" * 100)
            print(f"❌ BATCH {batch_id} WORKFLOW FAILED")
            print("#" * 100)
            print(f"  Error: {e}")
            print(f"  Processing time before failure: {batch_duration:.2f} seconds")
            print("#" * 100)
            print()

            import traceback
            traceback.print_exc()
            raise
