@echo off
echo 🧠 Starting Intelligent Novel Processor Agent...
echo =============================================

:: Set environment variables
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=novelprocessing2024
set POSTGRES_PORT=5433

:: Change to the correct directory
cd /d "C:\Users\zhuqi\Documents\agentic_chinese_novel_bot"

:: Run the Unicode-safe processor agent (Windows compatible)
python unicode_safe_processor_agent.py

echo.
echo Press any key to exit...
pause > nul