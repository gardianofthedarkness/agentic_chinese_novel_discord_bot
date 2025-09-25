#!/usr/bin/env python3
"""
Enhanced Context-Aware Iterative Batch Processor with DeepSeek Reasoner
Intelligent processing based on position in book, with RAG and SQL database context
Optimized to prevent token waste and ensure meaningful improvements
"""

import os
import sys
import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from enum import Enum

# Setup UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from deepseek_integration import DeepSeekClient, create_deepseek_config
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iterative_batch_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Different stages of novel processing requiring different strategies"""
    BEGINNING = "beginning"      # First 10-20% of book - minimal context needed
    EARLY = "early"             # 20-40% - basic character/plot establishment
    MIDDLE = "middle"           # 40-70% - full context awareness needed
    CLIMAX = "climax"           # 70-90% - maximum context utilization
    ENDING = "ending"           # 90-100% - resolution and callbacks

@dataclass
class ProcessingContext:
    """Context information for intelligent processing"""
    batch_position: int
    total_batches: int
    processing_stage: ProcessingStage
    
    @property
    def progress_percentage(self) -> float:
        return (self.batch_position / self.total_batches) * 100 if self.total_batches > 0 else 0
    
    @property
    def should_use_full_context(self) -> bool:
        return self.processing_stage in [ProcessingStage.MIDDLE, ProcessingStage.CLIMAX]
    
    @property
    def context_search_depth(self) -> int:
        """How many historical batches to search for context"""
        stage_depths = {
            ProcessingStage.BEGINNING: 1,
            ProcessingStage.EARLY: 2,
            ProcessingStage.MIDDLE: 5,
            ProcessingStage.CLIMAX: 7,
            ProcessingStage.ENDING: 5
        }
        return stage_depths.get(self.processing_stage, 3)

@dataclass
class BatchAnalysisState:
    """Enhanced state of batch analysis through iterations"""
    batch_id: int
    chunks: List[Dict]
    iteration: int
    current_analysis: Dict
    historical_context: Dict
    confidence_scores: Dict
    satisfaction_level: float
    refinement_requests: List[str]
    processing_context: ProcessingContext
    character_mentions: List[str]
    timeline_markers: List[str]
    
@dataclass
class IterationResult:
    """Enhanced result of a single reasoning iteration"""
    iteration_num: int
    analysis: Dict
    confidence_scores: Dict
    satisfaction_level: float
    identified_issues: List[str]
    refinement_requests: List[str]
    reasoning_trace: List[str]
    improvement_score: float = 0.0
    specific_improvements: List[str] = None
    context_queries_used: int = 0
    tokens_used: int = 0
    
    def __post_init__(self):
        if self.specific_improvements is None:
            self.specific_improvements = []
    
    @property
    def is_meaningful_improvement(self) -> bool:
        return self.improvement_score > 0.02  # Minimum 2% improvement
    
class DeepSeekReasonerEngine:
    """Enhanced DeepSeek Reasoner engine with context awareness"""
    
    def __init__(self):
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Initialize external systems
        self.qdrant_client = QdrantClient(url="http://localhost:32768", verify=False)
        
        # Reasoning parameters (optimized for efficiency)
        self.max_iterations = 5
        self.satisfaction_threshold = 0.85
        self.confidence_threshold = 0.8
        self.min_improvement_threshold = 0.02  # 2% minimum improvement
        self.early_termination_patience = 2  # Stop after 2 iterations without improvement
        
        print("🧠 Enhanced DeepSeek Reasoner Engine Initialized")
        print(f"   Max iterations: {self.max_iterations}")
        print(f"   Satisfaction threshold: {self.satisfaction_threshold}")
        print(f"   Improvement threshold: {self.min_improvement_threshold}")
        print(f"   Early termination patience: {self.early_termination_patience}")
    
    def _determine_processing_stage(self, batch_position: int, total_batches: int) -> ProcessingStage:
        """Determine processing stage based on position in the book"""
        progress = (batch_position / total_batches) * 100
        
        if progress <= 20:
            return ProcessingStage.BEGINNING
        elif progress <= 40:
            return ProcessingStage.EARLY
        elif progress <= 70:
            return ProcessingStage.MIDDLE
        elif progress <= 90:
            return ProcessingStage.CLIMAX
        else:
            return ProcessingStage.ENDING
    
    async def _query_rag_context(self, batch_state: BatchAnalysisState) -> Dict[str, Any]:
        """Intelligently query RAG system based on context and processing stage"""
        
        if batch_state.processing_context.processing_stage == ProcessingStage.BEGINNING:
            # At the beginning, minimal context queries to avoid wasting tokens
            return {"context": "Beginning of story - minimal context needed", "results": [], "queries_made": 0}
        
        context_queries = []
        
        # Character-focused queries
        if batch_state.character_mentions and batch_state.processing_context.should_use_full_context:
            for character in batch_state.character_mentions[:2]:  # Limit to top 2 characters
                try:
                    search_results = self.qdrant_client.search(
                        collection_name="test_novel2",
                        query_text=f"character {character} background personality",
                        limit=batch_state.processing_context.context_search_depth
                    )
                    context_queries.append({
                        "type": "character_background",
                        "character": character,
                        "results": [hit.payload.get('content', '')[:200] for hit in search_results]
                    })
                except Exception as e:
                    logger.warning(f"RAG query failed for character {character}: {e}")
        
        return {"context_queries": context_queries, "queries_made": len(context_queries)}
    
    async def reason_through_batch(self, batch_state: BatchAnalysisState) -> IterationResult:
        """Perform one enhanced iteration of reasoning on the batch"""
        
        print(f"\n🔄 Enhanced DeepSeek Reasoner - Iteration {batch_state.iteration}")
        print(f"   Batch {batch_state.batch_id}: {len(batch_state.chunks)} chunks")
        print(f"   Processing stage: {batch_state.processing_context.processing_stage.value}")
        print(f"   Current satisfaction: {batch_state.satisfaction_level:.2f}")
        
        # Query context based on processing stage
        rag_context = await self._query_rag_context(batch_state)
        print(f"   RAG queries made: {rag_context.get('queries_made', 0)}")
        
        # Build enhanced reasoning prompt with context
        reasoning_prompt = self._build_context_aware_prompt(batch_state, rag_context)
        
        # Query DeepSeek Reasoner
        reasoning_result = await self._query_deepseek_reasoner(reasoning_prompt)
        
        if reasoning_result['success']:
            # Parse reasoning result
            iteration_result = self._parse_reasoning_result(
                batch_state.iteration, 
                reasoning_result['response']
            )
            
            # Calculate improvement
            iteration_result.improvement_score = iteration_result.satisfaction_level - batch_state.satisfaction_level
            iteration_result.context_queries_used = rag_context.get('queries_made', 0)
            iteration_result.tokens_used = reasoning_result.get('tokens_used', 0)
            
            print(f"   ✅ Iteration complete:")
            print(f"      Satisfaction: {iteration_result.satisfaction_level:.3f} (Δ{iteration_result.improvement_score:+.3f})")
            print(f"      Confidence: {max(iteration_result.confidence_scores.values()) if iteration_result.confidence_scores else 0:.2f}")
            print(f"      Context queries: {iteration_result.context_queries_used}")
            print(f"      Tokens used: {iteration_result.tokens_used:,}")
            
            return iteration_result
        else:
            print(f"   ❌ Reasoning failed: {reasoning_result['error']}")
            return self._create_failure_result(batch_state.iteration)
    
    def _build_context_aware_prompt(self, batch_state: BatchAnalysisState, rag_context: Dict) -> str:
        """Build context-aware reasoning prompt based on processing stage"""
        
        # Prepare batch content
        batch_content = "\n\n".join([
            f"Chunk {i+1}:\n{chunk['content'][:800]}" 
            for i, chunk in enumerate(batch_state.chunks)
        ])
        
        # Prepare historical context
        historical_summary = json.dumps(batch_state.historical_context, 
                                      ensure_ascii=False, indent=2)
        
        # Prepare current analysis if exists
        current_analysis_summary = json.dumps(batch_state.current_analysis, 
                                            ensure_ascii=False, indent=2)
        
        # Build refinement requests
        refinement_requests = "\n".join([
            f"- {request}" for request in batch_state.refinement_requests
        ])
        
        # Adaptive prompt based on processing stage
        stage_instructions = {
            ProcessingStage.BEGINNING: "你正在分析小说的开头部分。重点关注角色介绍、设定建立和初始情节。不要过度依赖历史背景，因为这是故事的开端。",
            ProcessingStage.EARLY: "你正在分析小说的早期部分。角色和基本情节已经建立。适度使用背景信息来理解角色发展和情节进展。",
            ProcessingStage.MIDDLE: "你正在分析小说的中段部分。充分利用历史背景和角色信息来理解复杂的情节发展和角色关系。",
            ProcessingStage.CLIMAX: "你正在分析小说的高潮部分。最大化利用所有可用的背景信息和角色历史来理解关键事件和转折。",
            ProcessingStage.ENDING: "你正在分析小说的结尾部分。关注情节解决、角色成长完成和与前面内容的呼应。"
        }
        
        stage_instruction = stage_instructions.get(batch_state.processing_context.processing_stage, 
                                                  "分析以下小说片段。")
        
        # Context information (adaptive based on stage)
        context_sections = []
        
        if batch_state.processing_context.processing_stage != ProcessingStage.BEGINNING:
            if rag_context.get("context_queries"):
                context_sections.append(f"# RAG背景信息\n{json.dumps(rag_context['context_queries'], ensure_ascii=False, indent=2)}")
            
            if batch_state.historical_context:
                context_sections.append(f"# 历史上下文\n{json.dumps(batch_state.historical_context, ensure_ascii=False, indent=2)}")
        
        context_info = "\n\n".join(context_sections) if context_sections else "# 无需历史背景 - 这是故事开始"
        
        prompt = f"""
{stage_instruction}

# 当前处理信息
- 批次位置: {batch_state.batch_id}/{batch_state.processing_context.total_batches} ({batch_state.processing_context.progress_percentage:.1f}%)
- 处理阶段: {batch_state.processing_context.processing_stage.value}
- 迭代次数: {batch_state.iteration}
- 当前满意度: {batch_state.satisfaction_level:.2f}

# 当前文本内容 (5个chunk)
{batch_content}

{context_info}

# 分析任务 (基于当前阶段重点)
请按照当前处理阶段的重点进行分析：

1. **角色分析**: 识别文中角色，解析匿名引用(如"那个人"、"少女"等)
2. **时间线定位**: 确定当前事件在故事时间线中的位置
3. **情节分析**: 分析当前事件的重要性和与整体故事的关系
4. **改进识别**: 如果这是后续迭代，明确指出相比前次分析的具体改进
{historical_summary}

# 当前分析结果 (如果是后续迭代)
{current_analysis_summary}

# 需要改进的方面 (来自上一次迭代)
{refinement_requests}

# 请进行深度推理分析
请通过多步推理，分析以上内容。在每一步推理中，请考虑：

## 第一步：内容理解与整合
1. 理解这5个片段的基本内容
2. 识别片段之间的连续性和关联性
3. 提取关键信息：角色、事件、时间线索

## 第二步：历史关联分析  
1. 将当前片段与历史上下文进行对比
2. 识别角色状态的变化和发展
3. 分析事件的因果关系和时间顺序
4. 检查是否有矛盾或不一致之处

## 第三步：时间线重构
1. 基于内容线索推断真实的时间顺序
2. 区分叙述顺序和实际发生顺序
3. 识别闪回、回忆、预示等非线性叙述
4. 构建准确的事件时间线

## 第四步：深度分析
1. 角色关系发展和情感变化
2. 主题和象征意义的体现
3. 情节发展的逻辑和意义
4. 与整体故事的连接

## 第五步：质量评估与改进
1. 评估当前分析的准确性和完整性
2. 识别需要进一步澄清的问题
3. 提出改进建议
4. 给出满意度评分

# 输出格式
请返回JSON格式的分析结果：

{{
  "reasoning_trace": [
    "第一步推理：...",
    "第二步推理：...",
    "第三步推理：...",
    "第四步推理：...",
    "第五步推理：..."
  ],
  "analysis_result": {{
    "characters": [
      {{
        "name": "角色名",
        "state_changes": ["状态变化1", "状态变化2"],
        "relationships": ["与X的关系", "与Y的关系"],
        "development": "角色发展分析",
        "timeline_position": "在时间线中的位置"
      }}
    ],
    "events": [
      {{
        "event": "事件描述", 
        "chronological_position": "实际时间顺序位置",
        "narrative_position": "叙述顺序位置",
        "causality": ["导致的事件", "被什么导致"],
        "significance": "事件重要性分析"
      }}
    ],
    "timeline_reconstruction": {{
      "chronological_order": ["事件1", "事件2", "事件3"],
      "narrative_order": ["叙述1", "叙述2", "叙述3"],
      "temporal_markers": ["时间标记1", "时间标记2"],
      "flashback_sequences": ["闪回序列"],
      "parallel_storylines": ["平行故事线"]
    }},
    "thematic_analysis": {{
      "major_themes": ["主题1", "主题2"],
      "symbolic_elements": ["象征1", "象征2"],
      "emotional_arcs": ["情感弧线分析"]
    }},
    "connections_to_history": {{
      "character_continuity": ["角色连续性分析"],
      "plot_continuity": ["情节连续性分析"],
      "resolved_mysteries": ["解决的谜题"],
      "new_mysteries": ["新的谜题"]
    }}
  }},
  "confidence_scores": {{
    "character_analysis": 0.9,
    "timeline_accuracy": 0.8,
    "thematic_understanding": 0.85,
    "historical_continuity": 0.9,
    "overall_analysis": 0.88
  }},
  "quality_assessment": {{
    "identified_issues": ["问题1", "问题2"],
    "areas_for_improvement": ["改进方向1", "改进方向2"],
    "satisfaction_level": 0.85,
    "needs_further_iteration": true/false
  }},
  "refinement_requests": [
    "具体的改进要求1",
    "具体的改进要求2"
  ]
}}

请进行深入的推理分析，确保结果的准确性和完整性。
"""
        
        return prompt
    
    async def _query_deepseek_reasoner(self, prompt: str) -> Dict[str, Any]:
        """Query DeepSeek with reasoning-optimized parameters"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=4000,  # Larger for reasoning
                temperature=0.1,  # Lower for more consistent reasoning
                top_p=0.9,
                frequency_penalty=0.1
            )
            
            if response.get("success"):
                return {
                    'success': True,
                    'response': response["response"],
                    'tokens_used': len(prompt) // 4 + len(response["response"]) // 4
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error'),
                    'tokens_used': 0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tokens_used': 0
            }
    
    def _parse_reasoning_result(self, iteration_num: int, response: str) -> IterationResult:
        """Parse DeepSeek reasoning response"""
        
        try:
            # Extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_text = response[start:end].strip()
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_text = response[start:end]
            else:
                json_text = response
            
            parsed_result = json.loads(json_text)
            
            return IterationResult(
                iteration_num=iteration_num,
                analysis=parsed_result.get('analysis_result', {}),
                confidence_scores=parsed_result.get('confidence_scores', {}),
                satisfaction_level=parsed_result.get('quality_assessment', {}).get('satisfaction_level', 0.5),
                identified_issues=parsed_result.get('quality_assessment', {}).get('identified_issues', []),
                refinement_requests=parsed_result.get('refinement_requests', []),
                reasoning_trace=parsed_result.get('reasoning_trace', [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reasoning result: {e}")
            return self._create_failure_result(iteration_num)
    
    def _create_failure_result(self, iteration_num: int) -> IterationResult:
        """Create failure result when reasoning fails"""
        return IterationResult(
            iteration_num=iteration_num,
            analysis={},
            confidence_scores={},
            satisfaction_level=0.0,
            identified_issues=["Reasoning failed"],
            refinement_requests=["Retry with simpler prompt"],
            reasoning_trace=["Failed to complete reasoning"]
        )

class IterativeBatchProcessor:
    """Enhanced processor with context-aware iterative batch analysis"""
    
    def __init__(self):
        self.reasoner = DeepSeekReasonerEngine()
        self.db_path = "enhanced_iterative_results.db"
        self._initialize_database()
        
        self.global_stats = {
            'start_time': datetime.now(),
            'batches_processed': 0,
            'total_iterations': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'meaningful_improvements': 0
        }
        
        print("=" * 100)
        print("🔄 ENHANCED CONTEXT-AWARE ITERATIVE BATCH PROCESSOR")
        print("=" * 100)
        print(f"🕐 Start time: {self.global_stats['start_time']}")
        print("📊 Batch size: 5 chunks per batch")
        print("🧠 Engine: Enhanced DeepSeek Reasoner with RAG context")
        print("🎯 Goal: Context-aware analysis with intelligent iteration control")
        print("⚡ Features: Stage-based processing, token optimization, early termination")
        print("=" * 100)
    
    def _initialize_database(self):
        """Initialize database for iterative results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced batch results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS batch_results (
            batch_id INTEGER PRIMARY KEY,
            chunks_processed INTEGER,
            total_iterations INTEGER,
            final_satisfaction REAL,
            final_confidence REAL,
            processing_time REAL,
            total_tokens INTEGER,
            total_cost REAL,
            analysis_result TEXT,
            processing_stage TEXT,
            context_queries_used INTEGER,
            meaningful_improvements INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Enhanced iteration trace table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS iteration_trace (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            iteration_num INTEGER,
            satisfaction_level REAL,
            improvement_score REAL,
            context_queries_used INTEGER,
            tokens_used INTEGER,
            confidence_scores TEXT,
            identified_issues TEXT,
            refinement_requests TEXT,
            reasoning_trace TEXT,
            specific_improvements TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES batch_results (batch_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    async def process_volume_iteratively(self, volume_chunks: List[Dict], volume_id: int = 1) -> Dict[str, Any]:
        """Process entire volume using iterative batch processing"""
        
        print(f"\n📚 PROCESSING VOLUME {volume_id} ITERATIVELY")
        print(f"   Total chunks: {len(volume_chunks)}")
        print(f"   Batch size: 5 chunks")
        print(f"   Total batches: {(len(volume_chunks) + 4) // 5}")
        
        batch_results = []
        historical_context = {}
        
        # Process in batches of 5
        for batch_start in range(0, len(volume_chunks), 5):
            batch_end = min(batch_start + 5, len(volume_chunks))
            batch_chunks = volume_chunks[batch_start:batch_end]
            batch_id = (batch_start // 5) + 1
            
            print(f"\n" + "=" * 80)
            print(f"📦 BATCH {batch_id}: CHUNKS {batch_start + 1}-{batch_end}")
            print("=" * 80)
            
            # Process batch iteratively with enhanced context awareness
            batch_result = await self._process_batch_iteratively(
                batch_id, batch_chunks, historical_context, (len(volume_chunks) + 4) // 5
            )
            
            batch_results.append(batch_result)
            
            # Update historical context for next batch
            historical_context.update(batch_result['final_analysis'])
            
            print(f"✅ Batch {batch_id} completed:")
            print(f"   Iterations: {batch_result['total_iterations']}")
            print(f"   Final satisfaction: {batch_result['final_satisfaction']:.3f}")
            print(f"   Processing stage: {batch_result['processing_stage']}")
            print(f"   Meaningful improvements: {batch_result['meaningful_improvements']}")
            print(f"   Context queries: {batch_result['context_queries_used']}")
            print(f"   Tokens used: {batch_result['total_tokens']:,}")
            print(f"   Cost: ${batch_result['total_cost']:.4f}")
            
            # Update global stats
            self.global_stats['total_tokens'] += batch_result['total_tokens']
            self.global_stats['total_cost'] += batch_result['total_cost']
            self.global_stats['total_iterations'] += batch_result['total_iterations']
            self.global_stats['batches_processed'] += 1
        
        # Generate enhanced volume summary
        volume_summary = self._generate_enhanced_volume_summary(batch_results, volume_id)
        
        return volume_summary
    
    async def _process_batch_iteratively(self, batch_id: int, chunks: List[Dict], 
                                       historical_context: Dict, total_batches: int) -> Dict[str, Any]:
        """Enhanced batch processing with context awareness"""
        
        # Determine processing context
        processing_stage = self.reasoner._determine_processing_stage(batch_id, total_batches)
        processing_context = ProcessingContext(
            batch_position=batch_id,
            total_batches=total_batches,
            processing_stage=processing_stage
        )
        
        # Extract character mentions and timeline markers from chunks
        character_mentions = []
        timeline_markers = []
        for chunk in chunks:
            # Simple extraction - could be enhanced with NLP
            content = chunk.get('content', '')
            if '上条' in content:
                character_mentions.append('上条当麻')
            if '少女' in content:
                character_mentions.append('御坂美琴')
            if '电击' in content or '超电磁炮' in content:
                timeline_markers.append('能力展示')
        
        # Initialize enhanced batch state
        batch_state = BatchAnalysisState(
            batch_id=batch_id,
            chunks=chunks,
            iteration=0,
            current_analysis={},
            historical_context=historical_context,
            confidence_scores={},
            satisfaction_level=0.0,
            refinement_requests=[],
            processing_context=processing_context,
            character_mentions=character_mentions,
            timeline_markers=timeline_markers
        )
        
        iteration_results = []
        total_tokens = 0
        no_improvement_count = 0
        
        print(f"   Processing stage: {processing_context.processing_stage.value}")
        print(f"   Characters detected: {character_mentions[:3]}")
        
        # Enhanced iterative refinement loop with early termination
        while (batch_state.iteration < self.reasoner.max_iterations and 
               batch_state.satisfaction_level < self.reasoner.satisfaction_threshold):
            
            batch_state.iteration += 1
            previous_satisfaction = batch_state.satisfaction_level
            
            # Perform enhanced reasoning iteration
            iteration_result = await self.reasoner.reason_through_batch(batch_state)
            iteration_results.append(iteration_result)
            total_tokens += iteration_result.tokens_used
            
            # Update batch state
            batch_state.current_analysis = iteration_result.analysis
            batch_state.confidence_scores = iteration_result.confidence_scores
            batch_state.satisfaction_level = iteration_result.satisfaction_level
            batch_state.refinement_requests = iteration_result.refinement_requests
            
            # Enhanced iteration logging
            self._log_enhanced_iteration(batch_id, iteration_result)
            
            # Intelligent termination logic
            if iteration_result.satisfaction_level >= self.reasoner.satisfaction_threshold:
                print(f"   🎯 Satisfaction threshold reached: {iteration_result.satisfaction_level:.3f}")
                break
            elif iteration_result.is_meaningful_improvement:
                print(f"   📈 Meaningful improvement: +{iteration_result.improvement_score:.3f}")
                no_improvement_count = 0
                self.global_stats['meaningful_improvements'] += 1
            else:
                print(f"   ⚠️  Insufficient improvement: +{iteration_result.improvement_score:.3f}")
                no_improvement_count += 1
                
                if no_improvement_count >= self.reasoner.early_termination_patience:
                    print(f"   ⏹️  Early termination: {no_improvement_count} iterations without meaningful improvement")
                    break
            
            # Brief delay between iterations
            await asyncio.sleep(0.1)
        
        # Enhanced final batch result
        meaningful_improvements = sum(1 for r in iteration_results if r.is_meaningful_improvement)
        context_queries_total = sum(r.context_queries_used for r in iteration_results)
        
        final_result = {
            'batch_id': batch_id,
            'chunks_processed': len(chunks),
            'total_iterations': batch_state.iteration,
            'final_satisfaction': batch_state.satisfaction_level,
            'final_confidence': max(batch_state.confidence_scores.values()) if batch_state.confidence_scores else 0.0,
            'final_analysis': batch_state.current_analysis,
            'iteration_trace': iteration_results,
            'total_tokens': total_tokens,
            'total_cost': total_tokens * 0.00002,
            'processing_stage': processing_context.processing_stage.value,
            'context_queries_used': context_queries_total,
            'meaningful_improvements': meaningful_improvements,
            'character_mentions': character_mentions,
            'timeline_markers': timeline_markers
        }
        
        # Save batch result to database
        self._save_batch_result_to_database(final_result)
        
        return final_result
    
    def _log_enhanced_iteration(self, batch_id: int, iteration_result: IterationResult):
        """Log enhanced iteration details to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO iteration_trace 
            (batch_id, iteration_num, satisfaction_level, improvement_score, 
             context_queries_used, tokens_used, confidence_scores, 
             identified_issues, refinement_requests, reasoning_trace, specific_improvements)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                batch_id,
                iteration_result.iteration_num,
                iteration_result.satisfaction_level,
                iteration_result.improvement_score,
                iteration_result.context_queries_used,
                iteration_result.tokens_used,
                json.dumps(iteration_result.confidence_scores, ensure_ascii=False),
                json.dumps(iteration_result.identified_issues, ensure_ascii=False),
                json.dumps(iteration_result.refinement_requests, ensure_ascii=False),
                json.dumps(iteration_result.reasoning_trace, ensure_ascii=False),
                json.dumps(iteration_result.specific_improvements, ensure_ascii=False)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log enhanced iteration: {e}")
    
    def _save_batch_result_to_database(self, batch_result: Dict[str, Any]):
        """Save enhanced batch result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO batch_results 
            (batch_id, chunks_processed, total_iterations, final_satisfaction,
             final_confidence, total_tokens, total_cost, analysis_result,
             processing_stage, context_queries_used, meaningful_improvements)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                batch_result['batch_id'],
                batch_result['chunks_processed'],
                batch_result['total_iterations'],
                batch_result['final_satisfaction'],
                batch_result['final_confidence'],
                batch_result['total_tokens'],
                batch_result['total_cost'],
                json.dumps(batch_result['final_analysis'], ensure_ascii=False),
                batch_result['processing_stage'],
                batch_result['context_queries_used'],
                batch_result['meaningful_improvements']
            ))
            conn.commit()
            conn.close()
            logger.info(f"Enhanced batch {batch_result['batch_id']} saved to database")
        except Exception as e:
            logger.error(f"Failed to save enhanced batch result: {e}")
    
    def _generate_enhanced_volume_summary(self, batch_results: List[Dict], volume_id: int) -> Dict[str, Any]:
        """Generate enhanced summary for entire volume processing"""
        
        total_chunks = sum(batch['chunks_processed'] for batch in batch_results)
        total_iterations = sum(batch['total_iterations'] for batch in batch_results)
        total_tokens = sum(batch['total_tokens'] for batch in batch_results)
        total_cost = sum(batch['total_cost'] for batch in batch_results)
        total_meaningful_improvements = sum(batch['meaningful_improvements'] for batch in batch_results)
        total_context_queries = sum(batch['context_queries_used'] for batch in batch_results)
        
        avg_satisfaction = sum(batch['final_satisfaction'] for batch in batch_results) / len(batch_results)
        avg_confidence = sum(batch['final_confidence'] for batch in batch_results) / len(batch_results)
        
        # Stage breakdown analysis
        stage_stats = {}
        for batch in batch_results:
            stage = batch.get('processing_stage', 'unknown')
            if stage not in stage_stats:
                stage_stats[stage] = {
                    'batches': 0, 
                    'avg_satisfaction': 0, 
                    'avg_iterations': 0,
                    'total_tokens': 0,
                    'meaningful_improvements': 0
                }
            stage_stats[stage]['batches'] += 1
            stage_stats[stage]['avg_satisfaction'] += batch['final_satisfaction']
            stage_stats[stage]['avg_iterations'] += batch['total_iterations']
            stage_stats[stage]['total_tokens'] += batch['total_tokens']
            stage_stats[stage]['meaningful_improvements'] += batch['meaningful_improvements']
        
        for stage, stats in stage_stats.items():
            if stats['batches'] > 0:
                stats['avg_satisfaction'] /= stats['batches']
                stats['avg_iterations'] /= stats['batches']
        
        summary = {
            'volume_id': volume_id,
            'processing_metadata': {
                'total_batches': len(batch_results),
                'total_chunks': total_chunks,
                'total_iterations': total_iterations,
                'average_iterations_per_batch': total_iterations / len(batch_results),
                'average_satisfaction': avg_satisfaction,
                'average_confidence': avg_confidence,
                'processing_time': (datetime.now() - self.global_stats['start_time']).total_seconds()
            },
            'efficiency_metrics': {
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'tokens_per_chunk': total_tokens / total_chunks if total_chunks > 0 else 0,
                'cost_per_chunk': total_cost / total_chunks if total_chunks > 0 else 0,
                'meaningful_improvements': total_meaningful_improvements,
                'context_queries_used': total_context_queries,
                'improvement_rate': (total_meaningful_improvements / total_iterations * 100) if total_iterations > 0 else 0
            },
            'stage_breakdown': stage_stats,
            'batch_results': batch_results,
            'success': avg_satisfaction >= self.reasoner.satisfaction_threshold,
            'optimization_achieved': total_meaningful_improvements > 0
        }
        
        return summary

async def main():
    """Enhanced main execution function"""
    
    print("🚀 Starting Enhanced Context-Aware Iterative Batch Processing...")
    processor = IterativeBatchProcessor()
    
    try:
        # Load Volume 1 chunks (from previous processing)
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:32768", verify=False)
        
        # Get Volume 1 content
        print("📖 Loading Volume 1 content...")
        # [Implementation to load chunks - similar to previous scripts]
        
        # For now, test with sample data
        sample_chunks = [{'content': f'Sample chunk {i} content'} for i in range(1, 16)]
        
        # Process iteratively
        result = await processor.process_volume_iteratively(sample_chunks, volume_id=1)
        
        print("\n" + "=" * 100)
        print("🎉 ENHANCED ITERATIVE PROCESSING COMPLETED")
        print("=" * 100)
        
        # Basic stats
        print(f"📊 Total batches: {result['processing_metadata']['total_batches']}")
        print(f"🔄 Total iterations: {result['processing_metadata']['total_iterations']}")
        print(f"📈 Average satisfaction: {result['processing_metadata']['average_satisfaction']:.3f}")
        print(f"⏱️ Processing time: {result['processing_metadata']['processing_time']:.1f}s")
        
        # Efficiency metrics
        print("\n🚀 EFFICIENCY METRICS:")
        print(f"💰 Total cost: ${result['efficiency_metrics']['total_cost']:.4f}")
        print(f"📈 Meaningful improvements: {result['efficiency_metrics']['meaningful_improvements']}")
        print(f"🔍 Context queries used: {result['efficiency_metrics']['context_queries_used']}")
        print(f"⚡ Improvement rate: {result['efficiency_metrics']['improvement_rate']:.1f}%")
        print(f"🎯 Optimization achieved: {'YES' if result['optimization_achieved'] else 'NO'}")
        
        # Stage breakdown
        if result['stage_breakdown']:
            print("\n📊 PROCESSING STAGE BREAKDOWN:")
            for stage, stats in result['stage_breakdown'].items():
                print(f"  {stage.upper()}: {stats['batches']} batches, "
                      f"avg satisfaction: {stats['avg_satisfaction']:.3f}, "
                      f"avg iterations: {stats['avg_iterations']:.1f}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"enhanced_iterative_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📋 Detailed report saved: {report_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if processor.reasoner.deepseek_client.session:
            await processor.reasoner.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())