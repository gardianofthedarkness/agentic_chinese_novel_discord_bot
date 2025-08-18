# 🤖 Agentic Chinese Novel Discord Bot

> **Note**: This repository is currently under reconstruction. The main project files are being restored and will be added shortly.

## 📋 Project Overview

This is an advanced agentic system for Chinese novel interpretation, character simulation, and Discord-based roleplay using state-of-the-art AI models and hybrid retrieval-augmented generation (RAG).

### Key Features (In Development)
- 🤖 **Intelligent Literary Agent**: AI-powered character discovery and analysis
- 🎭 **Character Roleplay**: Consistent character simulation with personality engines
- 📚 **Story Analysis**: Automated storyline tracking and timeline construction  
- 🔍 **Hybrid RAG**: Combines SQL queries with vector search and AI generation
- 💬 **Discord Integration**: Interactive Discord bot with slash commands
- 🐘 **Dual Database Support**: PostgreSQL for structured data, Qdrant for vector search
- 🇨🇳 **Chinese Literature Optimized**: Specialized for Chinese novel processing

## 🚧 Current Status

**Repository Status**: Under Active Development

### Available Files:
- Testing and debugging utilities
- Database configuration and Qdrant vector data
- Node.js dependencies for Discord bot
- Comprehensive test suite

### Missing Files (Being Restored):
- Main Discord bot implementation
- Enhanced hybrid agent system
- Character simulation engines
- API server and documentation

## 🔧 Technology Stack

### Backend Technologies
- **Python 3.9+**: Main backend language
- **PostgreSQL**: Structured data storage (characters, timeline, events)
- **Qdrant**: Vector database for RAG and semantic search
- **DeepSeek AI**: Chinese literature-specialized language model
- **Flask**: API server framework
- **SQLAlchemy**: Database ORM
- **AsyncIO**: Asynchronous processing

### Frontend & Bot
- **Node.js 18+**: Discord bot runtime
- **Discord.js v14**: Discord API integration
- **React**: Web frontend (in development)

### AI & NLP
- **sentence-transformers**: Text embeddings
- **transformers**: Hugging Face model integration
- **torch**: PyTorch for deep learning models

## 📁 Current Directory Structure

```
mojin_rag_project/
├── test_*.py              # Testing utilities
├── debug_*.py             # Debugging tools  
├── final_status_check.py  # System verification
├── node_modules/          # Discord.js dependencies
├── qdrant_data/          # Vector database storage
├── .env                  # Environment configuration (gitignored)
├── .gitignore           # Git ignore rules
└── package.json         # Node.js project configuration
```

## 🚀 Quick Start (When Complete)

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

### Installation (Coming Soon)
```bash
# Clone repository
git clone https://github.com/gardianofthedarkness/agentic_chinese_novel_discord_bot.git
cd agentic_chinese_novel_discord_bot

# Install dependencies
pip install -r requirements.txt
npm install

# Start services
docker-compose up -d qdrant postgres
python agentic_api.py
npm start
```

## 🧪 Testing

Current testing utilities available:
```bash
# Character detection tests
python test_character_detection.py

# Variant resolution tests  
python test_variant_aware_agent.py

# System status check
python final_status_check.py
```

## 🔐 Security

- ✅ Environment variables properly configured
- ✅ API keys stored in GitHub Secrets
- ✅ Comprehensive .gitignore for sensitive data
- ✅ No hardcoded credentials in source code

## 🤝 Contributing

This project is under active development. Main components will be added incrementally:

1. **Phase 1**: Core agent system restoration
2. **Phase 2**: Discord bot integration
3. **Phase 3**: Character simulation engines
4. **Phase 4**: Advanced RAG and timeline features

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **DeepSeek AI** - Chinese literature-specialized language model
- **Qdrant** - High-performance vector database
- **Discord.js** - Discord bot development framework
- **魔法禁书目录** - Source novel for testing and development

---

**Note**: This is an active development project focused on Chinese novel analysis and character simulation. The system is designed to be educational and research-oriented, demonstrating advanced AI integration techniques for literature analysis.