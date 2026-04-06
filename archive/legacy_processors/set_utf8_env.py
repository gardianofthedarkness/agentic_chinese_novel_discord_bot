#!/usr/bin/env python3
"""
Set UTF-8 environment for Chinese character support
Run this before other scripts to ensure proper encoding
"""

import os
import sys
import locale

def setup_utf8_environment():
    """Setup UTF-8 environment for Chinese character support"""
    
    print("Setting up UTF-8 environment for Chinese character support...")
    
    # Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    
    # Set Python's default encoding
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    
    # Configure locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            print("Warning: Could not set UTF-8 locale")
    
    # Configure stdout/stderr encoding
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    print("UTF-8 environment configured successfully!")
    print(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING')}")
    print(f"LANG: {os.environ.get('LANG')}")
    print(f"System encoding: {sys.getdefaultencoding()}")
    print(f"Locale: {locale.getlocale()}")
    
    return True

if __name__ == "__main__":
    setup_utf8_environment()
    print("\nYou can now run Chinese character scripts safely!")
    print("Example: python run_hierarchical_analysis.py")