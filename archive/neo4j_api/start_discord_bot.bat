@echo off
echo ================================================================================
echo Starting Neo4j-Powered Discord Bot System
echo ================================================================================
echo.

:: Check if Neo4j is running
echo [1/3] Checking Neo4j connection...
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'novelprocessing2024')); driver.verify_connectivity(); print('   √ Neo4j is running'); driver.close()" 2>nul
if errorlevel 1 (
    echo    × Neo4j is not running!
    echo    Please start Neo4j first:
    echo    docker-compose -f docker-compose-neo4j-only.yml up -d
    pause
    exit /b 1
)

echo.
echo [2/3] Starting Python API Server (Neo4j + DeepSeek)...
start "Neo4j API Server" cmd /k "python neo4j_discord_server.py"

:: Wait for server to start
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Starting Discord Bot...
start "Discord Bot" cmd /k "node agentic-discord-bot.js"

echo.
echo ================================================================================
echo √ Discord Bot System Started!
echo ================================================================================
echo.
echo API Server: http://localhost:5005
echo Health Check: http://localhost:5005/health
echo.
echo Press any key to view logs or Ctrl+C to stop...
pause >nul
