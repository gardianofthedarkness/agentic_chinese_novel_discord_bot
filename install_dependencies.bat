@echo off
echo 📦 Installing Novel Processor Dependencies...
echo =============================================

echo.
echo Installing core dependencies...
pip install neo4j
pip install langgraph
pip install tiktoken
pip install numpy
pip install requests

echo.
echo Installing optional dependencies...
pip install python-dotenv
pip install rich

echo.
echo ✅ Dependencies installation completed!
echo.
echo Run the processor with:
echo   python robust_processor_agent.py
echo.
pause