#!/usr/bin/env python3
"""
Final Status Check - Verify entire system completion
"""

import asyncio
import subprocess
from sqlalchemy import create_engine, text
from datetime import datetime

async def final_system_status():
    print("="*70)
    print("FINAL SYSTEM STATUS CHECK")
    print("="*70)
    print(f"Timestamp: {datetime.now()}")
    
    # 1. Check database completeness
    print("\n1. DATABASE STATUS:")
    print("-" * 30)
    
    engine = create_engine('postgresql://admin:admin@localhost:5432/novel_sim')
    
    with engine.connect() as conn:
        # Core data tables
        tables_status = {}
        
        for table in ['chapter_summaries', 'character_profiles', 'timeline_events', 'storyline_threads']:
            try:
                result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                count = result.fetchone()[0]
                tables_status[table] = count
                print(f"   {table}: {count} records")
            except Exception as e:
                tables_status[table] = f"ERROR: {e}"
                print(f"   {table}: ERROR - {e}")
        
        # Specific checks
        print("\n2. CRITICAL COMPONENTS:")
        print("-" * 30)
        
        # Chapter coverage
        result = conn.execute(text('SELECT chapter_index FROM chapter_summaries ORDER BY chapter_index'))
        chapters = [row[0] for row in result]
        print(f"   Chapters processed: {chapters}")
        chapter_complete = set(chapters) >= {1, 2, 3}
        print(f"   Chapter processing complete: {'YES' if chapter_complete else 'NO'}")
        
        # Character detection
        result = conn.execute(text('SELECT COUNT(*) FROM character_profiles'))
        char_count = result.fetchone()[0]
        char_sufficient = char_count >= 10
        print(f"   Character profiles: {char_count} ({'SUFFICIENT' if char_sufficient else 'INSUFFICIENT'})")
        
        # 御坂美琴 detection
        result = conn.execute(text("SELECT name FROM character_profiles WHERE name LIKE '%御坂%' OR name LIKE '%美琴%'"))
        misaka_chars = [row[0] for row in result]
        misaka_fixed = len(misaka_chars) > 0
        print(f"   Misaka detection: {'FIXED' if misaka_fixed else 'STILL BROKEN'}")
        if misaka_chars:
            print(f"     Found: {misaka_chars}")
        
        # Data quality
        result = conn.execute(text('''
            SELECT COUNT(*) FROM character_profiles 
            WHERE name IS NULL OR name = '' OR confidence_score IS NULL
        '''))
        bad_data = result.fetchone()[0]
        data_quality = bad_data == 0
        print(f"   Data quality: {'GOOD' if data_quality else f'ISSUES ({bad_data} bad records)'}")
    
    # 3. Check running processes
    print("\n3. BACKGROUND PROCESSES:")
    print("-" * 30)
    
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        python_processes = result.stdout.count('python.exe')
        print(f"   Python processes running: {python_processes}")
        
        # Check specific processes
        result = subprocess.run(['wmic', 'process', 'where', 'name="python.exe"', 'get', 'CommandLine'], 
                              capture_output=True, text=True, shell=True)
        if 'agentic_api.py' in result.stdout:
            print("   API server: RUNNING")
        if 'monitor' in result.stdout:
            print("   Monitor processes: RUNNING")
        
    except Exception as e:
        print(f"   Process check failed: {e}")
    
    # 4. System readiness assessment
    print("\n4. SYSTEM READINESS:")
    print("-" * 30)
    
    readiness_checks = [
        ("Chapter processing", chapter_complete),
        ("Character detection", char_sufficient),
        ("Misaka fix", misaka_fixed),
        ("Data quality", data_quality)
    ]
    
    passed_checks = sum(1 for _, status in readiness_checks if status)
    total_checks = len(readiness_checks)
    
    for check_name, status in readiness_checks:
        print(f"   {check_name}: {'PASS' if status else 'FAIL'}")
    
    readiness_percent = (passed_checks / total_checks) * 100
    print(f"\n   Overall readiness: {readiness_percent:.0f}% ({passed_checks}/{total_checks})")
    
    # 5. Final verdict
    print("\n5. FINAL VERDICT:")
    print("=" * 30)
    
    if readiness_percent >= 100:
        status = "FULLY COMPLETE"
        recommendation = "System ready for production use"
        action = "Proceed with Discord bot integration"
    elif readiness_percent >= 75:
        status = "MOSTLY COMPLETE" 
        recommendation = "Minor issues present but functional"
        action = "Can proceed with caution"
    else:
        status = "INCOMPLETE"
        recommendation = "Significant issues need resolution"
        action = "Additional work required"
    
    print(f"   STATUS: {status}")
    print(f"   RECOMMENDATION: {recommendation}")
    print(f"   NEXT ACTION: {action}")
    
    # 6. Component summary
    print("\n6. COMPONENT SUMMARY:")
    print("-" * 30)
    print(f"   Enhanced hybrid agent: {'READY' if chapter_complete else 'INCOMPLETE'}")
    print(f"   Character tracking: {'READY' if misaka_fixed else 'BROKEN'}")
    print(f"   Database storage: {'READY' if data_quality else 'ISSUES'}")
    print(f"   Storyline analysis: {'READY' if char_sufficient else 'INSUFFICIENT'}")
    
    return {
        'status': status,
        'readiness_percent': readiness_percent,
        'components': {
            'chapters': chapter_complete,
            'characters': char_sufficient, 
            'misaka_fix': misaka_fixed,
            'data_quality': data_quality
        }
    }

if __name__ == "__main__":
    result = asyncio.run(final_system_status())
    print(f"\n{'='*70}")
    print(f"FINAL STATUS: {result['status']} ({result['readiness_percent']:.0f}%)")
    print("="*70)