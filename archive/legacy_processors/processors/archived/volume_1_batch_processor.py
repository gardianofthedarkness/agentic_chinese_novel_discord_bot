#!/usr/bin/env python3
"""
Volume 1 Batch Processor with Enhanced Progress Reporting
Shows detailed batch processing with DeepSeek decision making and database interactions
"""

import os
import sys
import asyncio
import json
import time
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Setup UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from deepseek_integration import DeepSeekClient, create_deepseek_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volume_1_batch_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BatchDecision:
    """Tracks DeepSeek decisions during batch processing"""
    decision_type: str  # "continue", "adjust_database", "function_call", "satisfied"
    reasoning: str
    confidence: float
    actions_taken: List[str]
    database_changes: List[str]
    timestamp: str

@dataclass
class BatchProgress:
    """Detailed progress tracking for each batch"""
    batch_id: int
    chunks_count: int
    iteration_number: int
    satisfaction_level: float
    confidence_scores: Dict[str, float]
    decisions_made: List[BatchDecision]
    database_queries: List[str]
    function_calls: List[str]
    retrospection_count: int
    processing_time: float
    improvement_from_previous: float

class EnhancedBatchProcessor:
    """Enhanced batch processor with detailed progress reporting"""
    
    def __init__(self):
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Enhanced parameters
        self.max_iterations = 5
        self.satisfaction_threshold = 0.85
        self.confidence_threshold = 0.8
        
        # Database setup
        self.db_path = "volume_1_enhanced_batch.db"
        self._initialize_enhanced_database()
        
        # Progress tracking
        self.global_stats = {
            'start_time': datetime.now(),
            'total_batches': 0,
            'total_retrospections': 0,
            'total_decisions': 0,
            'total_database_changes': 0,
            'total_function_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
        
        print("=" * 120)
        print("🚀 ENHANCED VOLUME 1 BATCH PROCESSOR")
        print("=" * 120)
        print(f"🧠 DeepSeek Reasoner: Max {self.max_iterations} iterations, {self.satisfaction_threshold} satisfaction threshold")
        print(f"📊 Enhanced Tracking: Decisions, Database changes, Function calls, Retrospections")
        print(f"💾 Database: {self.db_path}")
        print(f"🕐 Start time: {self.global_stats['start_time']}")
        print("=" * 120)
    
    def _initialize_enhanced_database(self):
        """Initialize enhanced database with decision tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Batch progress table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS batch_progress (
            batch_id INTEGER PRIMARY KEY,
            chunks_count INTEGER,
            total_iterations INTEGER,
            final_satisfaction REAL,
            retrospection_count INTEGER,
            processing_time REAL,
            decisions_count INTEGER,
            database_changes_count INTEGER,
            function_calls_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Decisions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS batch_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            iteration_num INTEGER,
            decision_type TEXT,
            reasoning TEXT,
            confidence REAL,
            actions_taken TEXT,
            database_changes TEXT,
            timestamp TEXT,
            FOREIGN KEY (batch_id) REFERENCES batch_progress (batch_id)
        )
        ''')
        
        # Analysis results table (enhanced)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS enhanced_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            iteration_num INTEGER,
            characters TEXT,
            events TEXT,
            timeline_reconstruction TEXT,
            thematic_analysis TEXT,
            confidence_scores TEXT,
            satisfaction_level REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES batch_progress (batch_id)
        )
        ''')
        
        # Database changes log
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS database_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            iteration_num INTEGER,
            change_type TEXT,
            old_value TEXT,
            new_value TEXT,
            reasoning TEXT,
            timestamp TEXT,
            FOREIGN KEY (batch_id) REFERENCES batch_progress (batch_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Enhanced database initialized: {self.db_path}")
    
    async def process_volume_1_enhanced(self) -> Dict[str, Any]:
        """Process Volume 1 with enhanced batch tracking"""
        
        try:
            # Load Volume 1 content
            print("\n📖 LOADING VOLUME 1 CONTENT")
            print("-" * 80)
            volume_chunks = await self._load_volume_1_chunks()
            
            if not volume_chunks:
                raise Exception("Failed to load Volume 1 chunks")
            
            print(f"✅ Loaded {len(volume_chunks)} chunks from Volume 1")
            
            # Process in batches with enhanced tracking
            print(f"\n🔄 STARTING ENHANCED BATCH PROCESSING")
            print("-" * 80)
            print(f"📊 Total chunks: {len(volume_chunks)}")
            print(f"📦 Batch size: 5 chunks")
            print(f"📦 Total batches: {(len(volume_chunks) + 4) // 5}")
            print("-" * 80)
            
            batch_results = []
            historical_context = {}
            
            # Process in batches of 5
            for batch_start in range(0, len(volume_chunks), 5):
                batch_end = min(batch_start + 5, len(volume_chunks))
                batch_chunks = volume_chunks[batch_start:batch_end]
                batch_id = (batch_start // 5) + 1
                
                print(f"\n" + "=" * 100)
                print(f"📦 BATCH {batch_id}: PROCESSING CHUNKS {batch_start + 1}-{batch_end}")
                print("=" * 100)
                
                batch_result = await self._process_batch_with_enhanced_tracking(
                    batch_id, batch_chunks, historical_context
                )
                
                batch_results.append(batch_result)
                
                # Update historical context
                if batch_result['final_analysis']:
                    historical_context.update(batch_result['final_analysis'])
                
                # Print batch summary
                self._print_batch_summary(batch_result)
                
                # Brief delay between batches
                await asyncio.sleep(1.0)
            
            # Generate final report
            final_report = self._generate_enhanced_report(batch_results)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    async def _load_volume_1_chunks(self) -> List[Dict]:
        """Load Volume 1 chunks from previous processing"""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:32768", verify=False)
            
            print("   📥 Fetching Volume 1 chunks from Qdrant...")
            
            # Get all data points
            all_points = []
            offset = None
            batch_size = 1000
            
            while True:
                points = client.scroll(
                    collection_name="test_novel2",
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                batch_points = points[0]
                all_points.extend(batch_points)
                
                if len(batch_points) < batch_size:
                    break
                
                offset = points[1] if len(points) > 1 and points[1] else None
                if not offset:
                    break
            
            # Extract Volume 1 chunks (first ~120 chunks)
            volume_1_chunks = []
            for i, point in enumerate(all_points[:120]):  # Volume 1 is approximately first 120 chunks
                content = point.payload.get('chunk', '')
                if content.strip():
                    volume_1_chunks.append({
                        'chunk_id': i,
                        'content': content,
                        'length': len(content)
                    })
            
            print(f"   ✅ Extracted {len(volume_1_chunks)} Volume 1 chunks")
            return volume_1_chunks
            
        except Exception as e:
            print(f"   ❌ Failed to load chunks: {e}")
            return []
    
    async def _process_batch_with_enhanced_tracking(self, batch_id: int, chunks: List[Dict], 
                                                  historical_context: Dict) -> Dict[str, Any]:
        """Process single batch with detailed decision tracking"""
        
        batch_start_time = time.time()
        
        print(f"🧠 Batch {batch_id}: Starting iterative analysis with DeepSeek Reasoner")
        print(f"   📊 Chunks: {len(chunks)}")
        print(f"   📚 Historical context size: {len(str(historical_context))} chars")
        
        # Initialize tracking
        decisions_made = []
        database_queries = []
        function_calls = []
        retrospection_count = 0
        current_satisfaction = 0.0
        current_analysis = {}
        
        # Iterative processing loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n   🔄 Iteration {iteration}:")
            
            iteration_start_time = time.time()
            retrospection_count += 1
            
            # Generate analysis prompt
            analysis_result = await self._analyze_batch_with_decisions(
                batch_id, chunks, historical_context, current_analysis, iteration
            )
            
            iteration_time = time.time() - iteration_start_time
            
            if analysis_result['success']:
                new_satisfaction = analysis_result['satisfaction_level']
                new_analysis = analysis_result['analysis']
                decision_info = analysis_result['decision_info']
                
                # Track decision
                decision = BatchDecision(
                    decision_type=decision_info['type'],
                    reasoning=decision_info['reasoning'],
                    confidence=decision_info['confidence'],
                    actions_taken=decision_info['actions'],
                    database_changes=decision_info['db_changes'],
                    timestamp=datetime.now().isoformat()
                )
                decisions_made.append(decision)
                
                # Log decision to database
                self._log_decision_to_database(batch_id, iteration, decision)
                
                # Print iteration results
                improvement = new_satisfaction - current_satisfaction
                print(f"      📈 Satisfaction: {current_satisfaction:.2f} → {new_satisfaction:.2f} ({improvement:+.2f})")
                print(f"      🧠 Decision: {decision_info['type']}")
                print(f"      💭 Reasoning: {decision_info['reasoning'][:100]}...")
                print(f"      ⏱️  Time: {iteration_time:.1f}s")
                
                # Check for database changes
                if decision_info['db_changes']:
                    print(f"      💾 Database changes: {len(decision_info['db_changes'])}")
                    for change in decision_info['db_changes']:
                        print(f"         - {change}")
                        database_queries.append(change)
                        self._apply_database_change(batch_id, iteration, change)
                
                # Check for function calls
                if decision_info['functions']:
                    print(f"      🔧 Function calls: {len(decision_info['functions'])}")
                    for func in decision_info['functions']:
                        print(f"         - {func}")
                        function_calls.append(func)
                
                # Update current state
                current_satisfaction = new_satisfaction
                current_analysis = new_analysis
                
                # Check satisfaction threshold
                if new_satisfaction >= self.satisfaction_threshold:
                    print(f"      ✅ Satisfaction threshold reached ({new_satisfaction:.2f} >= {self.satisfaction_threshold})")
                    break
                elif improvement < 0.01 and iteration > 2:
                    print(f"      ⚠️  Minimal improvement, continuing...")
                
            else:
                print(f"      ❌ Analysis failed: {analysis_result['error']}")
                break
            
            # Brief delay between iterations
            await asyncio.sleep(0.5)
        
        batch_processing_time = time.time() - batch_start_time
        
        # Save batch progress
        self._save_batch_progress(
            batch_id, len(chunks), retrospection_count, current_satisfaction,
            len(decisions_made), len(database_queries), len(function_calls),
            batch_processing_time
        )
        
        # Update global stats
        self.global_stats['total_batches'] += 1
        self.global_stats['total_retrospections'] += retrospection_count
        self.global_stats['total_decisions'] += len(decisions_made)
        self.global_stats['total_database_changes'] += len(database_queries)
        self.global_stats['total_function_calls'] += len(function_calls)
        
        return {
            'batch_id': batch_id,
            'chunks_processed': len(chunks),
            'final_satisfaction': current_satisfaction,
            'final_analysis': current_analysis,
            'retrospection_count': retrospection_count,
            'decisions_made': decisions_made,
            'database_changes': database_queries,
            'function_calls': function_calls,
            'processing_time': batch_processing_time,
            'iterations_completed': min(iteration, self.max_iterations)
        }
    
    async def _analyze_batch_with_decisions(self, batch_id: int, chunks: List[Dict], 
                                          historical_context: Dict, current_analysis: Dict,
                                          iteration: int) -> Dict[str, Any]:
        """Analyze batch and track DeepSeek decisions"""
        
        # Prepare batch content
        batch_content = "\n\n".join([
            f"Chunk {i+1}:\n{chunk['content'][:600]}" 
            for i, chunk in enumerate(chunks)
        ])
        
        # Build comprehensive prompt
        prompt = f"""
你是专业的中文小说分析专家，使用DeepSeek推理能力进行深度分析和决策。

# 当前状态
- 批次ID: {batch_id}
- 迭代次数: {iteration}
- 当前满意度: {current_analysis.get('satisfaction_level', 0.0):.2f}

# 需要分析的内容 (5个chunk)
{batch_content}

# 历史上下文
{json.dumps(historical_context, ensure_ascii=False, indent=2)[:1000]}

# 当前分析结果 (如果存在)
{json.dumps(current_analysis, ensure_ascii=False, indent=2)[:500]}

# 任务要求
请进行深度推理分析，并在分析过程中做出以下决策：

1. **分析决策**: 是否需要调整分析方法？
2. **数据库决策**: 是否需要查询或更新数据库？
3. **功能调用决策**: 是否需要调用特定函数？
4. **满意度评估**: 当前分析是否满足质量要求？

请返回JSON格式：
{{
  "analysis_result": {{
    "characters": [
      {{
        "name": "角色名",
        "development": "角色发展",
        "relationships": ["关系1", "关系2"]
      }}
    ],
    "events": [
      {{
        "event": "事件描述",
        "chronological_position": "时间位置",
        "significance": "重要性"
      }}
    ],
    "timeline_reconstruction": {{
      "chronological_order": ["事件1", "事件2"],
      "narrative_order": ["叙述1", "叙述2"]
    }},
    "thematic_analysis": {{
      "major_themes": ["主题1", "主题2"],
      "emotional_arcs": ["情感弧线"]
    }}
  }},
  "confidence_scores": {{
    "character_analysis": 0.9,
    "timeline_accuracy": 0.8,
    "overall_analysis": 0.85
  }},
  "satisfaction_level": 0.82,
  "decision_making": {{
    "decision_type": "continue/adjust_database/function_call/satisfied",
    "reasoning": "决策推理过程",
    "confidence": 0.85,
    "recommended_actions": ["行动1", "行动2"],
    "database_operations": ["查询操作1", "更新操作2"],
    "function_calls": ["函数调用1", "函数调用2"],
    "needs_retrospection": true/false,
    "improvement_suggestions": ["改进建议1", "改进建议2"]
  }}
}}

请进行深入的推理分析，确保决策的合理性。
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=2000,
                temperature=0.2
            )
            
            if response.get("success"):
                tokens_used = len(prompt) // 4 + len(response["response"]) // 4
                self.global_stats['total_tokens'] += tokens_used
                self.global_stats['total_cost'] += tokens_used * 0.00002
                
                # Parse response
                try:
                    response_text = response["response"]
                    if "```json" in response_text:
                        start = response_text.find("```json") + 7
                        end = response_text.find("```", start)
                        json_text = response_text[start:end].strip()
                    elif "{" in response_text:
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        json_text = response_text[start:end]
                    else:
                        json_text = response_text
                    
                    parsed_result = json.loads(json_text)
                    
                    # Extract decision information
                    decision_info = parsed_result.get('decision_making', {})
                    
                    return {
                        'success': True,
                        'analysis': parsed_result.get('analysis_result', {}),
                        'confidence_scores': parsed_result.get('confidence_scores', {}),
                        'satisfaction_level': parsed_result.get('satisfaction_level', 0.5),
                        'decision_info': {
                            'type': decision_info.get('decision_type', 'continue'),
                            'reasoning': decision_info.get('reasoning', ''),
                            'confidence': decision_info.get('confidence', 0.5),
                            'actions': decision_info.get('recommended_actions', []),
                            'db_changes': decision_info.get('database_operations', []),
                            'functions': decision_info.get('function_calls', []),
                            'needs_retrospection': decision_info.get('needs_retrospection', False)
                        },
                        'tokens_used': tokens_used
                    }
                    
                except json.JSONDecodeError as e:
                    return {
                        'success': False,
                        'error': f'JSON parsing failed: {e}',
                        'tokens_used': tokens_used
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
    
    def _log_decision_to_database(self, batch_id: int, iteration: int, decision: BatchDecision):
        """Log decision to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO batch_decisions 
            (batch_id, iteration_num, decision_type, reasoning, confidence, 
             actions_taken, database_changes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                batch_id, iteration, decision.decision_type, decision.reasoning,
                decision.confidence, 
                json.dumps(decision.actions_taken, ensure_ascii=False),
                json.dumps(decision.database_changes, ensure_ascii=False),
                decision.timestamp
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
    
    def _apply_database_change(self, batch_id: int, iteration: int, change: str):
        """Apply database change and log it"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Log the change
            cursor.execute('''
            INSERT INTO database_changes 
            (batch_id, iteration_num, change_type, old_value, new_value, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                batch_id, iteration, "adjustment", "", change, 
                "DeepSeek recommended change", datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.global_stats['total_database_changes'] += 1
            
        except Exception as e:
            logger.error(f"Failed to apply database change: {e}")
    
    def _save_batch_progress(self, batch_id: int, chunks_count: int, retrospection_count: int,
                           satisfaction: float, decisions_count: int, db_changes_count: int,
                           function_calls_count: int, processing_time: float):
        """Save batch progress to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO batch_progress 
            (batch_id, chunks_count, total_iterations, final_satisfaction, retrospection_count,
             processing_time, decisions_count, database_changes_count, function_calls_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                batch_id, chunks_count, retrospection_count, satisfaction, retrospection_count,
                processing_time, decisions_count, db_changes_count, function_calls_count
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save batch progress: {e}")
    
    def _print_batch_summary(self, batch_result: Dict[str, Any]):
        """Print detailed batch summary"""
        print(f"\n📊 BATCH {batch_result['batch_id']} SUMMARY:")
        print(f"   ✅ Final satisfaction: {batch_result['final_satisfaction']:.2f}")
        print(f"   🔄 Retrospections: {batch_result['retrospection_count']}")
        print(f"   🧠 Decisions made: {len(batch_result['decisions_made'])}")
        print(f"   💾 Database changes: {len(batch_result['database_changes'])}")
        print(f"   🔧 Function calls: {len(batch_result['function_calls'])}")
        print(f"   ⏱️  Processing time: {batch_result['processing_time']:.1f}s")
        
        # Show key decisions
        if batch_result['decisions_made']:
            print(f"   🎯 Key decisions:")
            for i, decision in enumerate(batch_result['decisions_made'][-3:], 1):  # Last 3 decisions
                print(f"      {i}. {decision.decision_type} (confidence: {decision.confidence:.2f})")
                print(f"         {decision.reasoning[:80]}...")
        
        # Show database changes
        if batch_result['database_changes']:
            print(f"   💾 Database operations:")
            for change in batch_result['database_changes'][:3]:  # First 3 changes
                print(f"      - {change[:60]}...")
        
        print(f"   🏁 Status: {'✅ Satisfied' if batch_result['final_satisfaction'] >= self.satisfaction_threshold else '⚠️ Partial'}")
    
    def _generate_enhanced_report(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive enhanced report"""
        
        end_time = datetime.now()
        total_time = (end_time - self.global_stats['start_time']).total_seconds()
        
        # Aggregate statistics
        total_retrospections = sum(batch['retrospection_count'] for batch in batch_results)
        total_decisions = sum(len(batch['decisions_made']) for batch in batch_results)
        total_db_changes = sum(len(batch['database_changes']) for batch in batch_results)
        total_function_calls = sum(len(batch['function_calls']) for batch in batch_results)
        
        avg_satisfaction = sum(batch['final_satisfaction'] for batch in batch_results) / len(batch_results)
        
        report = {
            'processing_metadata': {
                'start_time': self.global_stats['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'total_processing_time': total_time,
                'total_batches': len(batch_results),
                'average_satisfaction': avg_satisfaction
            },
            'decision_analysis': {
                'total_retrospections': total_retrospections,
                'total_decisions': total_decisions,
                'total_database_changes': total_db_changes,
                'total_function_calls': total_function_calls,
                'average_retrospections_per_batch': total_retrospections / len(batch_results),
                'average_decisions_per_batch': total_decisions / len(batch_results)
            },
            'cost_analysis': {
                'total_tokens': self.global_stats['total_tokens'],
                'total_cost': self.global_stats['total_cost'],
                'cost_per_batch': self.global_stats['total_cost'] / len(batch_results)
            },
            'batch_details': batch_results,
            'database_path': self.db_path,
            'success': avg_satisfaction >= self.satisfaction_threshold
        }
        
        return report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final enhanced summary"""
        print("\n" + "=" * 120)
        print("🎉 ENHANCED VOLUME 1 BATCH PROCESSING COMPLETED")
        print("=" * 120)
        
        meta = report['processing_metadata']
        decisions = report['decision_analysis']
        cost = report['cost_analysis']
        
        print(f"\n⏱️  PROCESSING SUMMARY:")
        print(f"   🕐 Total time: {meta['total_processing_time']/60:.1f} minutes")
        print(f"   📦 Batches processed: {meta['total_batches']}")
        print(f"   📈 Average satisfaction: {meta['average_satisfaction']:.2f}")
        
        print(f"\n🧠 DEEPSEEK DECISION ANALYSIS:")
        print(f"   🔄 Total retrospections: {decisions['total_retrospections']}")
        print(f"   🎯 Total decisions made: {decisions['total_decisions']}")
        print(f"   💾 Database changes: {decisions['total_database_changes']}")
        print(f"   🔧 Function calls: {decisions['total_function_calls']}")
        print(f"   📊 Avg retrospections/batch: {decisions['average_retrospections_per_batch']:.1f}")
        print(f"   📊 Avg decisions/batch: {decisions['average_decisions_per_batch']:.1f}")
        
        print(f"\n💰 COST ANALYSIS:")
        print(f"   🔢 Total tokens: {cost['total_tokens']:,}")
        print(f"   💵 Total cost: ${cost['total_cost']:.4f}")
        print(f"   💵 Cost per batch: ${cost['cost_per_batch']:.4f}")
        
        print(f"\n💾 DATABASE: {report['database_path']}")
        print(f"✅ SUCCESS: {report['success']}")
        print("=" * 120)

async def main():
    """Main execution function"""
    
    processor = EnhancedBatchProcessor()
    
    try:
        # Process Volume 1 with enhanced tracking
        logger.info("Starting enhanced Volume 1 processing...")
        report = await processor.process_volume_1_enhanced()
        
        # Print final summary
        processor.print_final_summary(report)
        
        # Save report to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"volume_1_enhanced_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"💾 Enhanced report saved to: {report_file}")
        logger.info("🎉 ENHANCED VOLUME 1 PROCESSING COMPLETED!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if processor.deepseek_client.session:
            await processor.deepseek_client.close()

if __name__ == "__main__":
    asyncio.run(main())