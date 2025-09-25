@echo off
echo ================================
echo UNIFIED NOVEL PROCESSING SYSTEM
echo ================================
echo Starting Docker services and processing volumes...
echo.

REM Set UTF-8 encoding
chcp 65001

REM Run the unified system
python run_complete_unified_system.py

pause