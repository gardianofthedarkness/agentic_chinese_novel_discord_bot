#!/usr/bin/env python3
"""
Progress Monitor for Comprehensive Analysis
"""

import time
import psutil
import os
from datetime import datetime

def monitor_analysis():
    """Monitor the comprehensive analysis progress"""
    
    print("COMPREHENSIVE ANALYSIS MONITOR")
    print("=" * 50)
    print(f"Started monitoring at: {datetime.now()}")
    print("Press Ctrl+C to stop monitoring\n")
    
    analysis_processes = []
    start_time = time.time()
    
    try:
        while True:
            # Check for Python processes
            current_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'cpu_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if 'run_comprehensive_analysis' in cmdline:
                            current_processes.append({
                                'pid': proc.info['pid'],
                                'status': proc.info['status'],
                                'cpu': proc.cpu_percent()
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Display current status
            elapsed = time.time() - start_time
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ", end="")
            print(f"Elapsed: {elapsed:.0f}s | ", end="")
            
            if current_processes:
                for proc in current_processes:
                    print(f"PID:{proc['pid']} Status:{proc['status']} CPU:{proc['cpu']:.1f}% | ", end="")
                print("RUNNING ✓")
            else:
                print("NO ANALYSIS PROCESS FOUND")
            
            # Check for output files that might be created
            files_to_check = [
                'analysis_results.json',
                'character_evolution.json', 
                'storyline_analysis.json',
                'timeline_results.json'
            ]
            
            for filename in files_to_check:
                if os.path.exists(filename):
                    mtime = os.path.getmtime(filename)
                    print(f"   Found: {filename} (modified: {datetime.fromtimestamp(mtime)})")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. Total elapsed time: {time.time() - start_time:.0f} seconds")

if __name__ == "__main__":
    monitor_analysis()