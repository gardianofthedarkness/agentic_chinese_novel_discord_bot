# Complete File User Guide - MoJin RAG Project

This guide provides detailed information about each file in the project, including its purpose, usage, dependencies, and relationship to other components.

## 🚀 **CORE RUNNING SYSTEM FILES**

### Discord Bot & API Layer

#### `agentic-discord-bot.js` ⭐ **MAIN DISCORD BOT**
**Purpose**: Primary Discord bot with slash commands for novel analysis and character roleplay  
**Dependencies**: `discord.js`, `axios`, `winston`, `dotenv`  
**Usage**: `npm start` or `node agentic-discord-bot.js`  
**Key Features**:
- Slash commands: `/analyze`, `/chat`, `/explore`, `/memory`, `/status`
- Character roleplay mode with consistent personalities
- Intelligent response chunking for Discord message limits
- Conversation history management per channel
- Fallback error handling and rate limit management

**Configuration**: Requires Discord bot token and API server URL in `.env`
```javascript
// Example .env configuration
DISCORD_TOKEN=your_discord_bot_token
API_SERVER_URL=http://localhost:5005
DISCORD_CLIENT_ID=your_bot_client_id
```

#### `agentic_api.py` ⭐ **MAIN API SERVER**
**Purpose**: Flask API server that connects Discord bot to AI agents  
**Dependencies**: `flask`, `flask-cors`, `enhanced_hybrid_agent`, `simple_chat_agent`  
**Usage**: `python agentic_api.py` (runs on port 5005)  
**Endpoints**:
- `POST /api/agent/chat` - Intelligent conversation with optional character roleplay
- `POST /api/agent/analyze` - Novel analysis (characters, storylines, timeline)
- `POST /api/agent/explore` - Deep topic exploration
- `GET /api/agent/memory` - Agent knowledge state
- `GET /api/agent/status` - System capabilities
- `GET /health` - Health check

**Integration**: Connects Discord bot commands to Python AI agents

---

### AI Agent Core System

#### `enhanced_hybrid_agent.py` ⭐ **PRIMARY INTELLIGENT AGENT**
**Purpose**: Main AI agent combining SQL + RAG + DeepSeek + HyDE for comprehensive responses  
**Dependencies**: `sqlalchemy`, `simple_rag`, `deepseek_integration`, `asyncio`  
**Key Methods**:
- `intelligent_chat()` - Main chat with character roleplay support
- `get_sql_big_picture()` - Retrieve structured data from PostgreSQL
- `hyde_enhanced_rag()` - Enhanced retrieval using hypothetical documents
- `get_novel_analysis_summary()` - Comprehensive analysis summary

**Architecture**:
```python
# Usage example
agent = create_enhanced_hybrid_agent()
response = await agent.intelligent_chat(
    user_message="Tell me about the main character",
    character_name="御坂美琴"  # Optional roleplay mode
)
```

#### `intelligent_agent.py` ⭐ **CHARACTER DISCOVERY AGENT**
**Purpose**: AI-powered agent for dynamic character discovery and storyline analysis  
**Dependencies**: `deepseek_integration`, `simple_rag`, `character_config`  
**Key Features**:
- Dynamic character discovery from novel text
- Storyline analysis and timeline construction
- Knowledge graph building
- Memory system for learned information

#### `simple_chat_agent.py` ⭐ **FALLBACK AGENT**
**Purpose**: Non-async fallback agent for basic conversation when DeepSeek is unavailable  
**Dependencies**: `sqlalchemy`, `simple_rag`  
**Usage**: Automatic fallback when main agents fail  
**Features**:
- Rule-based responses without AI dependency
- Character and timeline data from PostgreSQL
- Pattern matching for different query types

---

### Character Simulation Systems

#### `personality_engine.py` ⭐ **CHARACTER CONSISTENCY ENGINE**
**Purpose**: Maintains character personality consistency and emotional states  
**Dependencies**: `sentence-transformers`, `numpy`, `character_config`  
**Key Classes**:
- `PersonalityEngine` - Core personality management
- `EmotionalState` - Emotional state tracking with volatility
- `ConversationContext` - Context analysis and flow detection

**Usage**:
```python
engine = PersonalityEngine()
engine.add_character("御坂美琴", personality_profile)
response = engine.generate_response(character_name, user_input, context)
```

#### `character_config.py` ⭐ **CHARACTER PROFILE MANAGEMENT**
**Purpose**: Character profile storage, validation, and conversation memory persistence  
**Dependencies**: `pydantic`, `aiofiles`, `sqlite3`  
**Data Models**:
- `CharacterProfile` - Complete character definition
- `ConversationMemory` - Chat history and context
- `CharacterAnalytics` - Usage statistics

#### `character_variant_resolver.py` ⭐ **CHARACTER NAME UNIFICATION**
**Purpose**: Resolves character name variants and creates unified profiles  
**Usage**: `python character_variant_resolver.py`  
**Key Features**:
- Unifies variants: 御坂美琴 = 美琴 = 御坂 = ミサカ = Misaka
- Creates lookup tables for character resolution
- Reduces character duplication in database

**Output**: Creates `unified_characters` and `character_variant_lookup` tables

#### `advanced_character_engine.py` ⭐ **DEEP LEARNING CHARACTER MODEL**
**Purpose**: Advanced character modeling with neural networks and state prediction  
**Dependencies**: `numpy`, deep learning models  
**Features**:
- 1024-dimensional character state vectors
- Emotional prediction networks
- Goal evolution based on experiences
- Advanced behavioral modeling

---

### Data Processing & Novel Analysis

#### `agentic_novel_processor.py` ⭐ **NOVEL PROCESSING SYSTEM**
**Purpose**: Automated novel analysis and database population  
**Dependencies**: `deepseek_integration`, `simple_rag`, `sqlalchemy`  
**Usage**: `python agentic_novel_processor.py`  
**Process Flow**:
1. Extracts chapters from RAG system
2. Analyzes each chapter with DeepSeek AI
3. Identifies characters, storylines, timeline events
4. Stores structured data in PostgreSQL

**Database Tables Created**:
- `chapter_summaries` - AI-generated chapter analysis
- `character_profiles` - Discovered character profiles
- `storyline_threads` - Tracked storylines
- `timeline_events` - Chronological event timeline

#### `deepseek_integration.py` ⭐ **AI MODEL INTEGRATION**
**Purpose**: DeepSeek API client optimized for Chinese literature analysis  
**Dependencies**: `aiohttp`, `asyncio`, `dotenv`  
**Key Classes**:
- `DeepSeekClient` - API client with async HTTP
- `EnhancedCharacterEngine` - Character-focused response generation

**Configuration**: Requires DeepSeek API key
```python
# .env configuration
DEEPSEEK_API_KEY=your_deepseek_api_key
```

---

### RAG & Search Systems

#### `enhanced_rag.py` ⭐ **ADVANCED RAG SYSTEM**
**Purpose**: Character-aware retrieval system with emotion tagging and HyDE  
**Dependencies**: `sentence-transformers`, `qdrant-client`, Azure AI  
**Features**:
- Character-specific content retrieval
- Context-aware search (dialogue/narration/action)
- Hypothetical document enhancement (HyDE)
- Emotion-tagged search results

#### `simple_rag.py` ⭐ **BASIC RAG CLIENT**
**Purpose**: Simple RAG client with mock responses for testing  
**Dependencies**: `requests`  
**Usage**: Fallback when advanced RAG unavailable  
**Features**: Basic Qdrant integration with mock novel content

---

### Database Systems

#### `database_adapter.py` ⭐ **DATABASE ABSTRACTION**
**Purpose**: Unified interface supporting SQLite (dev) and PostgreSQL (production)  
**Dependencies**: `asyncpg` (optional), `aiosqlite`  
**Features**:
- Database abstraction layer
- Connection pooling
- Proper async handling

#### `timeline_database.py` ⭐ **TIMELINE PERSISTENCE**
**Purpose**: SQLite database management for timeline system  
**Dependencies**: `sqlite3`, `asyncio`, `timeline_models`  
**Features**:
- Timeline node storage and retrieval
- Character state persistence across timeline branches
- Story event tracking

#### `timeline_models.py` ⭐ **TIMELINE DATA MODELS**
**Purpose**: Pydantic data models for timeline system  
**Dependencies**: `pydantic`, `uuid`, `hashlib`  
**Models**:
- `TimelineNode` - Individual timeline points
- `CharacterState` - Character state at specific timeline points
- `StoryEvent` - Significant story events with impact tracking

---

## 📋 **CONFIGURATION & DEPLOYMENT FILES**

#### `package.json` - **Node.js Dependencies**
**Purpose**: Defines Discord bot dependencies and scripts  
**Scripts**:
- `npm start` - Start Discord bot
- `npm run dev` - Development mode with auto-reload
- `npm test` - Run connection tests

#### `requirements.txt` - **Python Dependencies**
**Purpose**: All Python package requirements  
**Key Dependencies**: `langchain`, `transformers`, `qdrant-client`, `deepseek-api`

#### `docker-compose.yml` - **Basic Container Orchestration**
**Services**: `qdrant`, `backend`, `frontend`

#### `docker-compose-enhanced.yml` - **Production Deployment**
**Services**: Enhanced setup with PostgreSQL, monitoring, and logging

#### `docker-compose-hybrid.yml` - **Hybrid Development Setup**
**Services**: Mixed local/container development environment

---

## 🧪 **TESTING & DEBUGGING FILES**

#### `test_character_detection.py` - **Character System Tests**
**Purpose**: Tests character detection and variant resolution  
**Usage**: `python test_character_detection.py`

#### `test_variant_aware_agent.py` - **Variant Resolution Tests**
**Purpose**: Tests character variant unification system  
**Usage**: `python test_variant_aware_agent.py`

#### `debug_bot_requests.py` - **Discord Bot Debugging**
**Purpose**: Debug Discord bot API requests and responses  
**Usage**: `python debug_bot_requests.py`

#### `monitor_progress.py` - **Processing Monitor**
**Purpose**: Monitors novel processing progress and system status  
**Usage**: `python monitor_progress.py`

#### `final_status_check.py` - **System Verification**
**Purpose**: Comprehensive system status verification  
**Usage**: `python final_status_check.py`

---

## 📚 **DOCUMENTATION FILES**

#### `ADVANCED_CHARACTER_SYSTEM.md` - **Character System Documentation**
**Content**: Detailed guide to advanced character simulation features

#### `README_CHARACTER_SIMULATION.md` - **Character Roleplay Guide**
**Content**: User guide for character roleplay functionality

#### `DEPLOYMENT_OPTIONS.md` - **Deployment Guide**
**Content**: Various deployment configurations and best practices

#### `project_cleanup_guide.md` - **Maintenance Guide**
**Content**: Project maintenance and cleanup procedures

---

## 📁 **LEGACY/DEPRECATED FILES**

### Scripts Directory (`/scripts/`)
- `mojin_embedding.py` - Old embedding script (replaced by enhanced_rag.py)
- `qdrantrag-m3e_small-Copy1.py` - Old RAG implementation

### Deprecated Core Files
- `simple_monitor.py` - Basic monitoring (replaced by monitor_progress.py)
- Files with `test_` prefix in root - Old testing scripts

### Old Processing Scripts
- Various analysis scripts replaced by `agentic_novel_processor.py`

---

## 🔧 **USAGE PATTERNS & WORKFLOWS**

### Development Workflow
```bash
# 1. Start core services
docker-compose up -d qdrant postgres

# 2. Process novel data (one-time)
python agentic_novel_processor.py

# 3. Resolve character variants (one-time)
python character_variant_resolver.py

# 4. Start API server
python agentic_api.py

# 5. Start Discord bot
npm start
```

### Testing Workflow
```bash
# Test character systems
python test_character_detection.py
python test_variant_aware_agent.py

# Test Discord bot
node test/test-connection.js

# Monitor system
python monitor_progress.py
python final_status_check.py
```

### Debugging Workflow
```bash
# Check logs
tail -f agentic-bot.log
tail -f agentic-bot-error.log

# Debug specific components
python debug_bot_requests.py

# Check database status
python -c "from database_adapter import *; test_connection()"
```

## 🔗 **FILE DEPENDENCIES GRAPH**

```
agentic-discord-bot.js
├── agentic_api.py
    ├── enhanced_hybrid_agent.py
    │   ├── deepseek_integration.py
    │   ├── simple_rag.py
    │   └── database (PostgreSQL)
    ├── simple_chat_agent.py
    │   ├── simple_rag.py
    │   └── database (PostgreSQL)
    └── intelligent_agent.py
        ├── deepseek_integration.py
        ├── character_config.py
        └── personality_engine.py

agentic_novel_processor.py
├── deepseek_integration.py
├── simple_rag.py
└── database (PostgreSQL)

character_variant_resolver.py
└── database (PostgreSQL)

personality_engine.py
├── character_config.py
└── database (SQLite)

timeline_database.py
├── timeline_models.py
└── database (SQLite)
```

This comprehensive guide covers all files in the project, their relationships, and usage patterns. Each file serves a specific role in the overall agentic novel interpretation and roleplay system.