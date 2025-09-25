#!/usr/bin/env python3
"""
Comprehensive Analysis Runner
Uses our working big chapter tracking with enhanced storyline analysis
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from test_big_chapter_tracking import BigChapterCharacterTracker
from deepseek_integration import DeepSeekClient, create_deepseek_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveAnalysisRunner:
    """Enhanced analysis using our working systems"""
    
    def __init__(self):
        self.character_tracker = BigChapterCharacterTracker()
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Story analysis results
        self.storyline_analysis = {}
        self.timeline_events = []
        self.character_evolution = {}
        self.recursive_recap = ""
        
        # Token usage tracking
        self.total_tokens_used = 0
        self.api_calls_made = 0
    
    async def run_comprehensive_analysis(self):
        """Run complete analysis on first 3 big chapters"""
        print("=" * 80)
        print("COMPREHENSIVE NOVEL ANALYSIS")
        print("Processing first 3 big chapters with:")
        print("  - Dynamic character tracking and evolution")
        print("  - Storyline analysis using DeepSeek")
        print("  - Timeline construction")
        print("  - Character relationship mapping")
        print("  - Recursive story recap")
        print("=" * 80)
        
        # Step 1: Get and process big chapters with character tracking
        print("\n1. PROCESSING BIG CHAPTERS WITH CHARACTER TRACKING...")
        big_chapters = await self.character_tracker.get_first_n_big_chapters(3)
        
        if not big_chapters:
            print("No big chapters found!")
            return
        
        await self.character_tracker.process_big_chapters(big_chapters)
        
        # Step 2: Enhanced storyline analysis for each chapter
        print("\n2. ENHANCED STORYLINE ANALYSIS...")
        for chapter_num in sorted(big_chapters.keys()):
            await self.analyze_chapter_storylines(chapter_num, big_chapters[chapter_num])
        
        # Step 3: Timeline construction
        print("\n3. CONSTRUCTING GLOBAL TIMELINE...")
        await self.construct_global_timeline()
        
        # Step 4: Character evolution analysis
        print("\n4. ANALYZING CHARACTER EVOLUTION...")
        await self.analyze_character_evolution()
        
        # Step 5: Generate recursive recap
        print("\n5. GENERATING RECURSIVE RECAP...")
        await self.generate_recursive_recap()
        
        # Step 6: Print comprehensive results
        print("\n6. COMPREHENSIVE RESULTS...")
        self.print_comprehensive_results()
        
        # Close connections
        await self.cleanup()
    
    async def analyze_chapter_storylines(self, chapter_num: int, chapter_sections: List[Dict[str, Any]]):
        """Detailed storyline analysis for a chapter"""
        print(f"\nAnalyzing storylines for Big Chapter {chapter_num}...")
        
        # Combine chapter content
        combined_content = ""
        for section in chapter_sections:
            combined_content += section['content'] + "\n\n"
        
        try:
            await self.deepseek_client.initialize()
            self.api_calls_made += 1
            
            prompt = f"""作为小说分析专家，请详细分析第{chapter_num}大章的故事情节：

章节内容长度：{len(combined_content)} 字符

请分析：
1. 主要故事情节和事件序列
2. 角色行动和决策
3. 冲突和转折点
4. 情感发展和心理变化
5. 故事推进的关键节点

请按以下格式返回分析：

## 主要情节线
- [情节1]
- [情节2]
- [情节3]

## 关键事件
1. [事件1]: [详细描述]
2. [事件2]: [详细描述]
3. [事件3]: [详细描述]

## 角色动态
- [角色A]: [行动和变化]
- [角色B]: [行动和变化]

## 冲突分析
- 内在冲突: [描述]
- 外在冲突: [描述]
- 解决方案: [描述]

## 情节意义
[本章对整体故事的意义和作用]

章节内容片段（供参考）：
{combined_content[:2000]}..."""
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            if response.get("success"):
                analysis = response["response"]
                self.storyline_analysis[chapter_num] = analysis
                self.total_tokens_used += response.get("token_usage", {}).get("total", 0)
                print(f"   SUCCESS: Chapter {chapter_num} storyline analysis complete")
            else:
                print(f"   FAILED: Failed to analyze Chapter {chapter_num}")
                self.storyline_analysis[chapter_num] = "Analysis failed"
                
        except Exception as e:
            print(f"   ERROR: Error analyzing Chapter {chapter_num}: {e}")
            self.storyline_analysis[chapter_num] = f"Error: {e}"
    
    async def construct_global_timeline(self):
        """Construct timeline across all chapters"""
        timeline_events = []
        
        try:
            await self.deepseek_client.initialize()
            self.api_calls_made += 1
            
            # Combine all storyline analyses
            all_analyses = ""
            for chapter_num in sorted(self.storyline_analysis.keys()):
                all_analyses += f"\n\n=== 第{chapter_num}大章分析 ===\n{self.storyline_analysis[chapter_num]}"
            
            prompt = f"""基于以下各章节的故事分析，构建一个完整的时间线：

{all_analyses}

请创建一个连贯的时间线，包括：
1. 按时间顺序排列的主要事件
2. 事件之间的因果关系
3. 角色发展的关键节点
4. 故事arc的转折点

格式：
## 全局时间线

### 第1大章时期
- 事件1: [描述] → 导致: [后果]
- 事件2: [描述] → 导致: [后果]

### 第2大章时期  
- 事件3: [描述] → 前因: [原因] → 导致: [后果]
- 事件4: [描述] → 前因: [原因] → 导致: [后果]

### 第3大章时期
- 事件5: [描述] → 前因: [原因] → 导致: [后果]

## 因果关系图
[描述主要的因果链条]

## 故事发展轨迹
[整体故事的发展模式和规律]"""
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=2500,
                temperature=0.1
            )
            
            if response.get("success"):
                timeline = response["response"]
                self.timeline_events = timeline
                self.total_tokens_used += response.get("token_usage", {}).get("total", 0)
                print("   SUCCESS: Global timeline construction complete")
            else:
                print("   FAILED: Failed to construct timeline")
                
        except Exception as e:
            print(f"   ERROR: Error constructing timeline: {e}")
    
    async def analyze_character_evolution(self):
        """Analyze how characters evolve across chapters"""
        try:
            await self.deepseek_client.initialize()
            self.api_calls_made += 1
            
            # Collect character data from tracker
            character_data = {}
            for char_id, char in self.character_tracker.ambiguous_characters.items():
                character_data[char_id] = {
                    'chapters': char.big_chapter_appearances,
                    'confidence': char.confidence_score,
                    'descriptions': char.descriptions,
                    'first_appearance': char.first_appearance
                }
            
            for char_id, char in self.character_tracker.named_characters.items():
                character_data[char_id] = {
                    'chapters': char.get('chapter_appearances', []),
                    'confidence': char.get('confidence', 0.8),
                    'type': 'named_character'
                }
            
            prompt = f"""基于角色跟踪数据，分析角色在3个大章中的演变：

角色数据：
{json.dumps(character_data, ensure_ascii=False, indent=2)}

故事线分析：
{json.dumps(self.storyline_analysis, ensure_ascii=False, indent=2)}

请分析：
1. 角色重要性的变化轨迹
2. 角色关系的发展
3. 角色成长arc
4. 潜在的重要角色识别

格式：
## 角色演变分析

### 核心角色演变
- [角色名]: [演变轨迹]

### 新兴重要角色
- [角色名]: [重要性发展]

### 角色关系网络变化
- [关系1]: [变化描述]
- [关系2]: [变化描述]

### 角色成长预测
[基于目前发展预测角色未来走向]"""
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            if response.get("success"):
                evolution = response["response"]
                self.character_evolution = evolution
                self.total_tokens_used += response.get("token_usage", {}).get("total", 0)
                print("   SUCCESS: Character evolution analysis complete")
            else:
                print("   FAILED: Failed to analyze character evolution")
                
        except Exception as e:
            print(f"   ERROR: Error analyzing character evolution: {e}")
    
    async def generate_recursive_recap(self):
        """Generate comprehensive recursive recap"""
        try:
            await self.deepseek_client.initialize()
            self.api_calls_made += 1
            
            prompt = f"""作为专业小说编辑，基于以下完整分析生成递归式总结：

## 故事线分析
{json.dumps(self.storyline_analysis, ensure_ascii=False, indent=2)}

## 时间线构建
{self.timeline_events}

## 角色演变
{self.character_evolution}

## 角色跟踪数据
发现角色: {len(self.character_tracker.ambiguous_characters)}个模糊角色, {len(self.character_tracker.named_characters)}个命名角色

请生成一个递归式总结，包括：

1. **故事整体架构**
   - 三大章的故事arc
   - 主要冲突和解决

2. **角色发展轨迹**
   - 核心角色的成长
   - 角色关系的演变

3. **主题发展**
   - 贯穿三章的主题
   - 主题的深化过程

4. **叙事技巧分析**
   - 伏笔的铺设
   - 情节的递进

5. **预测与展望**
   - 基于前三章预测后续发展
   - 潜在的故事走向

请生成深入、连贯、具有洞察力的分析。"""
            
            messages = [{"role": "user", "content": prompt}]
            response = await self.deepseek_client.generate_character_response(
                messages=messages,
                max_tokens=3000,
                temperature=0.2
            )
            
            if response.get("success"):
                recap = response["response"]
                self.recursive_recap = recap
                self.total_tokens_used += response.get("token_usage", {}).get("total", 0)
                print("   SUCCESS: Recursive recap generation complete")
            else:
                print("   FAILED: Failed to generate recap")
                
        except Exception as e:
            print(f"   ERROR: Error generating recap: {e}")
    
    def print_comprehensive_results(self):
        """Print all analysis results"""
        print("\n" + "=" * 90)
        print("COMPREHENSIVE ANALYSIS RESULTS")
        print("=" * 90)
        
        # Processing statistics
        print(f"\nPROCESSING STATISTICS:")
        print(f"   - Total API calls made: {self.api_calls_made}")
        print(f"   - Total tokens used: {self.total_tokens_used:,}")
        print(f"   - Chapters processed: 3 big chapters")
        print(f"   - Characters tracked: {len(self.character_tracker.ambiguous_characters)} ambiguous + {len(self.character_tracker.named_characters)} named")
        
        # Character tracking results (from our working system)
        print(f"\nCHARACTER TRACKING RESULTS:")
        print(f"{'-' * 50}")
        
        all_characters = []
        for char_id, char_data in self.character_tracker.ambiguous_characters.items():
            all_characters.append({
                'name': char_id,
                'chapters': char_data.big_chapter_appearances,
                'confidence': char_data.confidence_score,
                'type': 'ambiguous'
            })
        
        for char_id, char_data in self.character_tracker.named_characters.items():
            all_characters.append({
                'name': char_id,
                'chapters': char_data.get('chapter_appearances', []),
                'confidence': char_data.get('confidence', 0.8),
                'type': 'named'
            })
        
        all_characters.sort(key=lambda x: (len(x['chapters']), x['confidence']), reverse=True)
        
        for char in all_characters[:10]:  # Top 10 characters
            print(f"\n   '{char['name']}' ({char['type']}):")
            print(f"      Chapters: {char['chapters']}")
            print(f"      Confidence: {char['confidence']:.2f}")
            if len(char['chapters']) >= 2:
                print(f"      >>> RECURRING CHARACTER <<<")
        
        # Storyline analysis results
        print(f"\nSTORYLINE ANALYSIS:")
        print(f"{'-' * 50}")
        for chapter_num in sorted(self.storyline_analysis.keys()):
            print(f"\n=== BIG CHAPTER {chapter_num} ===")
            analysis = self.storyline_analysis[chapter_num]
            # Print first 500 characters of analysis
            print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
        
        # Timeline
        print(f"\nGLOBAL TIMELINE:")
        print(f"{'-' * 50}")
        if self.timeline_events:
            timeline_preview = str(self.timeline_events)[:800]
            print(timeline_preview + "..." if len(str(self.timeline_events)) > 800 else self.timeline_events)
        
        # Character evolution
        print(f"\nCHARACTER EVOLUTION:")
        print(f"{'-' * 50}")
        if self.character_evolution:
            evolution_preview = str(self.character_evolution)[:600]
            print(evolution_preview + "..." if len(str(self.character_evolution)) > 600 else self.character_evolution)
        
        # Recursive recap
        print(f"\nRECURSIVE RECAP:")
        print(f"{'=' * 70}")
        if self.recursive_recap:
            print(self.recursive_recap)
        else:
            print("No recursive recap generated")
        
        print(f"\n{'=' * 90}")
        print("ANALYSIS COMPLETE")
        print("=" * 90)
    
    async def cleanup(self):
        """Clean up connections"""
        if self.character_tracker.deepseek_client.session:
            await self.character_tracker.deepseek_client.close()
        if self.deepseek_client.session:
            await self.deepseek_client.close()

async def main():
    """Run comprehensive analysis"""
    runner = ComprehensiveAnalysisRunner()
    await runner.run_comprehensive_analysis()

if __name__ == "__main__":
    asyncio.run(main())