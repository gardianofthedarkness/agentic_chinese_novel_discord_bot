#!/usr/bin/env python3
"""
Simple Progress Monitor (no external dependencies)
"""

import time
import os
import subprocess
from datetime import datetime

def check_process():
    """Check if our analysis process is running"""
    try:
        # Use tasklist to find python processes
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        return 'python.exe' in result.stdout
    except:
        return False

def check_files():
    """Check for any new files or changes"""
    files_to_check = [
        'analysis_results.json',
        'character_evolution.json', 
        'storyline_analysis.json',
        'timeline_results.json',
        'comprehensive_analysis_output.txt'
    ]
    
    found_files = []
    for filename in files_to_check:
        if os.path.exists(filename):
            mtime = os.path.getmtime(filename)
            found_files.append(f"{filename} (modified: {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')})")
    
    return found_files

def monitor():
    """Simple monitoring loop"""
    print("SIMPLE ANALYSIS MONITOR")
    print("=" * 40)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    
    try:
        while True:
            elapsed = int(time.time() - start_time)
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Check if process is running
            process_running = check_process()
            
            # Check for output files
            files = check_files()
            
            print(f"\r[{current_time}] Elapsed: {elapsed}s | Process: {'RUNNING' if process_running else 'NOT FOUND'} | Files: {len(files)}", end="")
            
            if files:
                print(f"\n   Files found: {', '.join(files)}")
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped after {int(time.time() - start_time)} seconds")

if __name__ == "__main__":
    monitor()