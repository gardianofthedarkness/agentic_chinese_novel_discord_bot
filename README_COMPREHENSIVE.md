# MoJin RAG Project - Agentic Novel Interpretation & Roleplay Discord Bot

An advanced agentic system for Chinese novel interpretation, character simulation, and Discord-based roleplay using state-of-the-art AI models and hybrid retrieval-augmented generation (RAG).

## 🌟 Project Overview

This project implements a sophisticated multi-agent system that:
- **Analyzes Chinese novels** using DeepSeek AI with specialized literature prompts
- **Discovers and tracks characters** dynamically through AI-powered analysis
- **Builds storyline timelines** and character relationship networks
- **Provides immersive roleplay** through Discord bot with character-consistent responses
- **Combines multiple data sources** (SQL databases, vector databases, AI models) for comprehensive understanding

### Key Features
- 🤖 **Intelligent Literary Agent**: AI-powered character discovery and analysis
- 🎭 **Character Roleplay**: Consistent character simulation with personality engines
- 📚 **Story Analysis**: Automated storyline tracking and timeline construction  
- 🔍 **Hybrid RAG**: Combines SQL queries with vector search and AI generation
- 💬 **Discord Integration**: Interactive Discord bot with slash commands
- 🐘 **Dual Database Support**: PostgreSQL for structured data, Qdrant for vector search
- 🇨🇳 **Chinese Literature Optimized**: Specialized for Chinese novel processing

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "Discord Bot Layer"
        A[Discord Bot.js] --> B[API Server Flask]
    end
    
    subgraph "Agent Layer"
        B --> C[Enhanced Hybrid Agent]
        B --> D[Simple Chat Agent]
        C --> E[Intelligent Agent]
    end
    
    subgraph "AI & NLP"
        C --> F[DeepSeek Integration]
        C --> G[Enhanced RAG]
        E --> F
        E --> G
    end
    
    subgraph "Character Systems"
        F --> H[Character Variant Resolver]
        F --> I[Personality Engine]
        F --> J[Advanced Character Engine]
    end
    
    subgraph "Data Layer"
        C --> K[PostgreSQL]
        C --> L[Qdrant Vector DB]
        G --> L
        I --> M[Timeline Database]
    end
    
    subgraph "Processing Layer"
        N[Novel Processor] --> K
        N --> L
        O[Character Config] --> K
    end
```

## 📁 Project Structure

### 🚀 **Current Running Version Files**

#### **Core Discord Bot & API**
- `agentic-discord-bot.js` - Main Discord bot with slash commands
- `agentic_api.py` - Flask API server connecting Discord to agents
- `enhanced_hybrid_agent.py` - Main agent combining SQL+RAG+DeepSeek+HyDE
- `intelligent_agent.py` - AI-powered character discovery and analysis agent
- `simple_chat_agent.py` - Fallback agent for basic functionality

#### **Character Systems (Active)**
- `personality_engine.py` - Character consistency and emotional state management
- `character_config.py` - Character profile management with persistence
- `character_variant_resolver.py` - Unifies character name variants (御坂美琴/美琴/御坂)
- `advanced_character_engine.py` - Deep learning character modeling

#### **Data Processing**
- `agentic_novel_processor.py` - Automated novel analysis and database population
- `deepseek_integration.py` - DeepSeek AI model integration for Chinese literature
- `enhanced_rag.py` - Character-aware retrieval with emotion tagging
- `simple_rag.py` - Basic RAG client with Qdrant integration

#### **Database Systems**
- `database_adapter.py` - Database abstraction (SQLite/PostgreSQL)
- `timeline_database.py` - Timeline system persistence
- `timeline_models.py` - Data models for timeline and character states

### 📋 **Configuration & Deployment**
- `docker-compose.yml` - Multi-container deployment
- `Dockerfile.discord` - Discord bot container
- `Dockerfile.mcp` - MCP server container  
- `package.json` - Node.js dependencies for Discord bot
- `requirements.txt` - Python dependencies
- `development-config.yaml` - Development configuration

### 📊 **Testing & Analysis Files**
- `test_character_detection.py` - Character detection testing
- `test_variant_aware_agent.py` - Character variant resolution testing
- `debug_bot_requests.py` - Discord bot debugging
- `monitor_progress.py` - Processing progress monitoring
- `final_status_check.py` - System status verification

### 📖 **Documentation**
- `ADVANCED_CHARACTER_SYSTEM.md` - Advanced character simulation docs
- `README_CHARACTER_SIMULATION.md` - Character roleplay documentation  
- `DEPLOYMENT_OPTIONS.md` - Deployment guide
- `project_cleanup_guide.md` - Project maintenance guide

### 🗂️ **Legacy/Deprecated Files**
- Files in `/scripts/` - Old processing scripts
- `simple_monitor.py` - Basic monitoring (replaced by enhanced versions)
- Old processing scripts - Replaced by agentic_novel_processor.py

## 🔧 Dependencies & Technology Stack

### **Backend Technologies**
- **Python 3.9+**: Main backend language
- **PostgreSQL**: Structured data storage (characters, timeline, events)
- **Qdrant**: Vector database for RAG and semantic search
- **DeepSeek AI**: Chinese literature-specialized language model
- **Flask**: API server framework
- **SQLAlchemy**: Database ORM
- **AsyncIO**: Asynchronous processing

### **Frontend & Bot**
- **Node.js 18+**: Discord bot runtime
- **Discord.js v14**: Discord API integration
- **React**: Web frontend (in development)
- **Winston**: Logging for Node.js

### **AI & NLP**
- **sentence-transformers**: Text embeddings
- **transformers**: Hugging Face model integration
- **torch**: PyTorch for deep learning models
- **tiktoken**: Token counting and management

### **Data Processing**
- **pandas**: Data manipulation
- **beautifulsoup4**: HTML parsing
- **ebooklib**: EPUB processing
- **pymupdf**: PDF processing

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Required software
- Docker & Docker Compose
- PostgreSQL 13+
- Node.js 18+
- Python 3.9+

# Environment variables
DISCORD_TOKEN=your_discord_bot_token
DEEPSEEK_API_KEY=your_deepseek_api_key
DATABASE_URL=postgresql://admin:admin@localhost:5432/novel_sim
QDRANT_URL=http://localhost:32768
```

### Installation

1. **Clone the repository**
```bash
git clone <repository_url>
cd mojin_rag_project
```

2. **Set up environment**
```bash
# Create .env file
cp .env.example .env
# Edit .env with your configuration

# Install Python dependencies  
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

3. **Start services**
```bash
# Start database services
docker-compose up -d qdrant postgres

# Process novel data (one-time setup)
python agentic_novel_processor.py

# Start API server
python agentic_api.py

# Start Discord bot
npm start
```

### Docker Deployment
```bash
# Complete system deployment
docker-compose -f docker-compose-enhanced.yml up -d

# Check service status
docker-compose ps
```

## 🎮 Discord Bot Usage

### Slash Commands
- `/analyze` - AI-powered novel analysis (characters, storylines)
- `/chat <message>` - Intelligent conversation about the novel
- `/chat <message> character:<name>` - Character roleplay mode
- `/explore <topic>` - Deep topic exploration
- `/memory` - View AI agent's learned knowledge
- `/status` - System health and capabilities
- `/clear` - Clear conversation history

### Example Usage
```
/analyze type:full
/chat message:"Tell me about the main characters"
/chat message:"How are you feeling?" character:"御坂美琴"  
/explore topic:"power systems in the novel" depth:deep
```

## 🔧 Core Components Deep Dive

### Enhanced Hybrid Agent (`enhanced_hybrid_agent.py`)
The main intelligence of the system combining:
- **SQL Big Picture**: Character profiles, timeline events, story progress from PostgreSQL
- **RAG Details**: Specific content retrieval from Qdrant vector database
- **HyDE Enhancement**: Hypothetical document generation for better retrieval
- **DeepSeek Integration**: Chinese literature-specialized AI responses

### Character Systems
#### Personality Engine (`personality_engine.py`)
- Maintains character emotional states with volatility tracking
- Ensures response consistency with personality profiles
- Context-aware conversation flow detection

#### Character Variant Resolver (`character_variant_resolver.py`) 
- Unifies character name variants (御坂美琴 = 美琴 = 御坂 = ミサカ)
- Creates unified character profiles from multiple detection instances
- Provides character lookup system for variant names

### Database Schema
```sql
-- Character profiles with AI analysis
character_profiles (character_id, name, type, personality_traits, relationships, confidence_score)

-- Timeline events with chronological ordering  
timeline_events (event_id, chapter_index, event_description, characters_involved, chronological_order)

-- Chapter summaries with AI analysis
chapter_summaries (chapter_id, chapter_index, content_summary, key_events, emotional_arc)

-- Unified character variants
unified_characters (unified_id, primary_name, aliases, character_type, combined_confidence)
```

## 📈 Advanced Features

### Character Roleplay System
- **Consistent Personalities**: Characters maintain consistent traits and speaking patterns
- **Emotional States**: Dynamic emotional tracking affects responses
- **Memory System**: Characters remember previous conversations
- **Relationship Awareness**: Characters respond based on relationship dynamics

### Timeline Analysis
- **Chronological Events**: AI-constructed timeline of story events
- **Character Development**: Track character growth across chapters
- **Plot Thread Tracking**: Multiple storyline analysis and resolution status

### Multi-Modal RAG
- **Context-Aware Search**: Different retrieval for dialogue vs. narration vs. action scenes
- **Emotion-Tagged Results**: Search results tagged with emotional context
- **Character-Specific Retrieval**: Find content specific to particular characters

## 🔍 API Endpoints

### Main Endpoints
```
GET  /health                    - System health check
POST /api/agent/analyze        - Novel analysis (characters/storylines)
POST /api/agent/chat           - Intelligent chat with optional character roleplay
POST /api/agent/explore        - Topic exploration with configurable depth
GET  /api/agent/memory         - Agent knowledge state
GET  /api/agent/status         - System capabilities and status
```

### Request/Response Examples
```json
// Character roleplay request
{
  "message": "今天的天气真不错呢",
  "character_name": "御坂美琴"
}

// Response
{
  "response": "哼，天气好又怎样？反正学园都市里每天都有奇怪的事情发生...",
  "character": {
    "name": "御坂美琴",
    "type": "electromaster",
    "traits": ["傲娇", "正义感强", "电击使"]
  },
  "context": {
    "mode": "character_roleplay",
    "sql_data": {...},
    "rag_sources": 5
  }
}
```

## 🧪 Testing & Development

### Run Tests
```bash
# Character detection tests
python test_character_detection.py

# Variant resolution tests  
python test_variant_aware_agent.py

# Discord bot connection test
node test/test-connection.js
```

### Development Mode
```bash
# API server with debug
FLASK_DEBUG=true python agentic_api.py

# Discord bot with auto-reload
npm run dev
```

### Monitoring
```bash
# Check processing progress
python monitor_progress.py

# System status verification
python final_status_check.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Discord Bot Not Responding**
   - Check `DISCORD_TOKEN` in environment
   - Verify bot permissions in Discord server
   - Check API server connectivity

2. **Character Variants Not Resolving**
   - Run `python character_variant_resolver.py`
   - Check PostgreSQL connection
   - Verify character_profiles table has data

3. **AI Responses Failing**
   - Verify `DEEPSEEK_API_KEY` is valid
   - Check API rate limits
   - Review logs in `agentic-bot-error.log`

4. **Database Connection Issues**
   - Ensure PostgreSQL is running
   - Check database credentials in `.env`
   - Verify database schema is created

### Logs & Debugging
```bash
# Check Discord bot logs
tail -f agentic-bot.log

# Check API server logs  
tail -f flask.log

# Check PostgreSQL logs
docker-compose logs postgres
```

## 🚀 Future Roadmap

- [ ] **Multi-Novel Support**: Expand beyond single novel analysis
- [ ] **Voice Integration**: Text-to-speech character voices
- [ ] **Visual Character Cards**: Generate character images
- [ ] **Interactive Web UI**: Browser-based character chat
- [ ] **Advanced Analytics**: Character relationship graphs
- [ ] **Multi-Language**: Support for other languages beyond Chinese

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Use type hints in Python code
- Test Discord bot functionality before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek AI** - Chinese literature-specialized language model
- **Qdrant** - High-performance vector database
- **Discord.js** - Discord bot development framework
- **魔法禁书目录** - Source novel for testing and development

---

**Note**: This is an active development project focused on Chinese novel analysis and character simulation. The system is designed to be educational and research-oriented, demonstrating advanced AI integration techniques for literature analysis.