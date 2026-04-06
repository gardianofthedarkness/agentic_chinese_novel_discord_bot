# Project Cleanup Guide

## 🧹 Redundant Files to Archive/Remove

### Old Discord Bot Versions (Keep Only Latest)
**KEEP:** `agentic-discord-bot.js` (New agentic system)  
**ARCHIVE:** These old versions:
- `discord_bot.py` - Original Python version
- `discord_bot_debug.py` - Debug version  
- `discord_bot_fast.py` - Fast version
- `discord_bot_fixed.py` - Fixed version
- `discord_bot_final.py` - "Final" version (superseded)
- `discord_bot_with_logging.py` - Logging version
- `debug_discord_bot.py` - Another debug version
- `simple_test_bot.py` - Simple test

### Old API Servers (Keep Only Latest)
**KEEP:** `agentic_api.py` (New agentic system)  
**ARCHIVE:** These old versions:
- `production_api.py` - Previous production version
- `api_server.py` - Original API server
- `api_server_simple.py` - Simplified version
- `api_server_minimal.py` - Minimal version
- `working_api.py` - Working version (superseded)
- `simple_http_server.py` - Simple HTTP version
- `test_flask.py` - Flask test

### Test Files (Consolidate or Remove)
**KEEP:** Essential tests only
- `test_discord_connection.py` - Useful for debugging
- `run_novel_analysis.py` - New main processing script

**ARCHIVE:** Redundant tests:
- `test_character_rag.py` - Old character tests
- `test_real_rag.py` - Old RAG tests  
- `test_deepseek_live.py` - DeepSeek tests (functionality now in processor)
- `test_mcp_basic.py` - Basic MCP tests
- `test_mcp_server.py` - MCP server tests
- `test_full_integration.py` - Full integration (superseded)
- `quick_test_commands.py` - Quick tests

### Utility Scripts (Keep Useful Ones)
**KEEP:**
- `simple_rag.py` - Still used by new system
- `deepseek_integration.py` - Core integration
- `enhanced_rag.py` - Enhanced RAG (may have useful components)
- `agentic_novel_processor.py` - NEW main processor
- `intelligent_agent.py` - NEW intelligent agent

**EVALUATE:**
- `personality_engine.py` - Check if still needed
- `character_config.py` - Check if still needed  
- `advanced_character_engine.py` - May have useful components
- `populate_qdrant.py` - Utility for populating vector DB

### Old MCP Versions (Keep Latest)
**KEEP:** `mcp_server.py` or most recent version  
**ARCHIVE:**
- `mcp_http_server.py` - HTTP version
- `mcp_server_simple.py` - Simplified version

## 📁 Recommended Project Structure

```
mojin_rag_project/
├── core/                          # Core system
│   ├── agentic_novel_processor.py # NEW: Main offline processor
│   ├── intelligent_agent.py      # NEW: Intelligent agent
│   ├── simple_rag.py             # RAG client  
│   └── deepseek_integration.py   # AI integration
├── api/                          # API servers
│   └── agentic_api.py           # NEW: Main API server
├── discord/                      # Discord integration
│   └── agentic-discord-bot.js   # NEW: Main Discord bot
├── scripts/                      # Utility scripts
│   ├── run_novel_analysis.py    # NEW: Run offline analysis
│   └── test_discord_connection.py # Connection testing
├── archive/                      # Old versions
│   ├── old_discord_bots/        # Archive old bot versions
│   ├── old_api_servers/         # Archive old API versions
│   └── old_tests/               # Archive old tests
└── docs/                        # Documentation
    ├── project_cleanup_guide.md # This file
    └── README.md                # Main documentation
```

## 🚀 New System Architecture

### Core Components:
1. **`agentic_novel_processor.py`** - Offline processing system
   - Analyzes entire novel chapter by chapter
   - Stores results in PostgreSQL database
   - Based on your existing notebook approach but enhanced

2. **`agentic_api.py`** - Fast API server
   - Serves pre-processed data quickly
   - No real-time AI processing during Discord interactions

3. **`agentic-discord-bot.js`** - Discord interface
   - Simple commands that query pre-processed data
   - Fast responses since data is pre-analyzed

### Workflow:
1. **Offline:** Run `python run_novel_analysis.py` to analyze entire novel
2. **Online:** Users interact with Discord bot for instant responses
3. **Data:** All analysis stored in PostgreSQL for fast retrieval

## ✅ Migration Steps

1. **Test new system:**
   ```bash
   python run_novel_analysis.py  # Process novel offline
   python agentic_api.py         # Start API server  
   node agentic-discord-bot.js   # Start Discord bot
   ```

2. **Verify functionality:** Test Discord commands work with pre-processed data

3. **Archive old files:** Move old versions to archive/ directory

4. **Update documentation:** Create proper README for new architecture

## 🎯 Benefits of New System

- **Fast Discord responses** - No waiting for AI processing
- **Comprehensive analysis** - Full novel processed offline with time for deep analysis
- **Scalable** - Can handle large novels without Discord timeouts
- **Based on proven approach** - Enhanced version of your existing notebook method
- **Agentic AI** - Uses DeepSeek intelligence to discover characters and storylines dynamically