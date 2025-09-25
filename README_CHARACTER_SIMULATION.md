# MCP Character Simulation Server

A comprehensive character simulation system that transforms your existing RAG project into an MCP (Model Context Protocol) server with Discord bot integration for realistic character role-playing.

## 🌟 Features

### Core Capabilities
- **MCP Server Integration**: Full MCP protocol support with tools, resources, and prompts
- **Discord Bot Interface**: Direct Discord integration for real-time character interactions
- **Enhanced RAG System**: Character-specific conversation retrieval with HyDE enhancement
- **Personality Engine**: Advanced personality consistency and emotional state management
- **Memory Management**: Persistent conversation memory with SQLite storage
- **Character Analytics**: Comprehensive statistics and behavior analysis

### Advanced Features
- **Emotional State Tracking**: Dynamic mood and energy level management
- **Context-Aware Responses**: Conversation flow analysis and topic tracking
- **Speech Pattern Learning**: Automatic extraction and reinforcement of character patterns
- **Personality Validation**: Response consistency checking and improvement
- **Multi-Character Support**: Manage multiple characters with distinct personalities

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Discord Bot   │────│   MCP Server    │────│  Character Mgmt │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │ Personality     │    │ Memory Manager  │
         │              │ Engine          │    │ (SQLite)        │
         │              └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────▼───────────────────────┘
                        ┌─────────────────┐
                        │ Enhanced RAG    │
                        │ (Qdrant+HyDE)   │
                        └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r mcp_requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Configure Environment

Edit `.env` file with your settings:

```env
# Discord Bot
DISCORD_TOKEN=your_discord_bot_token

# Azure AI (optional, for better responses)
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_KEY=your_azure_key

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
```

### 3. Start the System

```bash
# Start the integrated MCP server
python integrated_main.py

# In another terminal, start the Discord bot
python discord_bot.py
```

### 4. Create Your First Character

Using Discord commands:
```
!create_character "Aria" "A cheerful and curious mage who loves learning new spells" | "真有趣, 让我试试, 太棒了" | "来自魔法学院的年轻学者"
```

Or using MCP tools programmatically:
```python
result = create_character_integrated(
    name="Aria",
    personality="A cheerful and curious mage who loves learning new spells",
    speech_patterns=["真有趣", "让我试试", "太棒了"],
    background="来自魔法学院的年轻学者",
    conversation_style="playful"
)
```

### 5. Activate Character in Discord

```
!activate_character Aria
```

Now the character will respond to all messages in that channel!

## 📚 System Components

### 1. MCP Server (`mcp_server.py`)
- **Tools**: Character creation, response simulation, memory management
- **Resources**: Character profiles, conversation history
- **Prompts**: Character creation assistance, conversation analysis

### 2. Discord Bot (`discord_bot.py`)
- Real-time character interactions
- Channel-specific character activation
- User-friendly command interface
- Conversation context management

### 3. Enhanced RAG (`enhanced_rag.py`)
- Character-specific embedding generation
- HyDE (Hypothetical Document Embeddings) support
- Neighbor context retrieval
- Emotion and context type analysis

### 4. Personality Engine (`personality_engine.py`)
- Dynamic emotional state management
- Conversation flow analysis
- Personality consistency scoring
- Context-aware prompt generation

### 5. Character Configuration (`character_config.py`)
- Comprehensive character profiles
- SQLite-based conversation storage
- Memory management and analytics
- Data export/import capabilities

## 🎭 Character Management

### Creating Characters

Characters are defined with rich profiles including:

```python
CharacterProfile(
    name="character_name",
    personality="Detailed personality description",
    speech_patterns=["Typical phrases", "Common expressions"],
    background="Character's backstory",
    conversation_style="casual/formal/playful/dramatic",
    base_mood="neutral/happy/calm",
    mood_volatility=0.5,  # How quickly emotions change
    memory_limit=20,      # Conversation memory size
    temperature=0.7,      # Response creativity
)
```

### Character Interactions

The system provides multiple interaction modes:

1. **Discord Channel Activation**: Characters respond to all messages
2. **Direct MCP Tool Calls**: Programmatic character simulation
3. **Batch Processing**: Multiple character interactions
4. **Analytics Mode**: Character behavior analysis

### Memory Management

- **Conversation History**: Persistent storage of all interactions
- **Contextual Memory**: Important message preservation
- **Topic Tracking**: Conversation theme continuity
- **Relationship Memory**: User-character relationship tracking

## 🔧 Configuration Options

### Character Personality Settings

```python
# Emotional Configuration
base_mood: str = "neutral"           # Default emotional state
mood_volatility: float = 0.5         # Emotion change rate
emotion_keywords: Dict[str, List]    # Custom emotion triggers

# Behavioral Settings
response_probability: float = 0.8     # Response frequency
memory_limit: int = 20               # Conversation memory
temperature: float = 0.7             # Response creativity
max_response_length: int = 300       # Response size limit

# Relationship Settings
relationships: Dict[str, str] = {}   # User relationships
relationship_memory: bool = True     # Remember relationships
```

### System Performance

```env
# Performance Settings
MAX_CONCURRENT_REQUESTS=10
RESPONSE_TIMEOUT=30
RAG_SEARCH_TIMEOUT=10

# Feature Flags
ENABLE_HYDE=true
ENABLE_PERSONALITY_VALIDATION=true
ENABLE_EMOTION_TRACKING=true
```

## 📊 Analytics and Monitoring

### Character Statistics

Get comprehensive character analytics:

```python
analytics = get_character_analytics("character_name")
# Returns:
# - Message statistics
# - Personality state
# - Conversation patterns
# - Emotional history
# - Response consistency scores
```

### System Health Monitoring

```python
health = system_health_check()
# Monitors:
# - Memory manager status
# - RAG system connectivity
# - Personality engine state
# - Database connectivity
```

## 🔌 MCP Integration

### Available Tools

- `create_character_integrated()`: Full character creation
- `simulate_character_response_integrated()`: Generate responses
- `get_character_analytics()`: Character statistics
- `system_health_check()`: System monitoring

### Available Resources

- `character://{name}`: Character profile data
- `conversations://{name}/{channel}`: Conversation history
- `analytics://{name}`: Character analytics
- `system://health`: System health status

### Available Prompts

- `character_creation_prompt()`: Assist character design
- `conversation_analysis_prompt()`: Analyze interactions

## 🛠️ Development and Extension

### Adding Custom Personality Traits

```python
# Extend the PersonalityEngine class
class CustomPersonalityEngine(PersonalityEngine):
    def analyze_custom_traits(self, message: str) -> Dict:
        # Your custom analysis logic
        pass
```

### Custom RAG Enhancement

```python
# Extend the CharacterRAGSystem class
class CustomRAGSystem(CharacterRAGSystem):
    async def custom_retrieval_strategy(self, query: str) -> List:
        # Your custom retrieval logic
        pass
```

### Adding New MCP Tools

```python
@mcp.tool()
def custom_character_tool(param1: str, param2: int) -> Dict[str, Any]:
    """Your custom tool description"""
    # Tool implementation
    return {"result": "success"}
```

## 🐛 Troubleshooting

### Common Issues

1. **Character Not Responding**
   - Check Discord bot permissions
   - Verify character activation: `!status`
   - Check MCP server connectivity

2. **RAG Retrieval Errors**
   - Ensure Qdrant is running on correct port
   - Verify embedding model download
   - Check collection exists: `test_novel2`

3. **Memory Issues**
   - Check SQLite database permissions
   - Verify character data directory exists
   - Clear corrupted memory: `!clear_memory`

4. **Personality Inconsistency**
   - Review character profile completeness
   - Check speech patterns quality
   - Enable personality validation

### Logging and Debugging

```python
# Enable debug logging
LOG_LEVEL=DEBUG

# Check system logs
tail -f character_simulation.log

# Test individual components
python character_config.py  # Test memory manager
python personality_engine.py  # Test personality system
python enhanced_rag.py  # Test RAG system
```

## 📈 Performance Optimization

### Memory Usage
- Adjust `memory_limit` per character
- Use `important_messages` for key conversations
- Regular memory cleanup with conversation summaries

### Response Speed
- Enable embedding caching
- Optimize RAG search parameters
- Use concurrent processing for multiple characters

### Scalability
- SQLite → PostgreSQL for high load
- Redis caching for personality states
- Load balancing for multiple MCP servers

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project extends your existing trading bot project and maintains the same license structure.

## 🎯 Next Steps

1. **Advanced Emotion Models**: Implement more sophisticated emotion detection
2. **Multi-Modal Support**: Add image and voice interaction capabilities
3. **Character Learning**: Implement continuous learning from interactions
4. **Advanced Analytics**: Real-time character behavior dashboards
5. **Clustering Support**: Multi-server character simulation

---

## 🚀 Migration from Your Existing System

Your current system has been enhanced with:

✅ **Preserved**: All existing RAG functionality and embeddings  
✅ **Enhanced**: Your Qdrant integration with character-specific features  
✅ **Extended**: Your HyDE implementation with personality awareness  
✅ **Integrated**: Your Azure AI setup with conversation management  
✅ **Added**: MCP protocol support and Discord bot integration  

The migration maintains compatibility with your existing `test_novel2` collection while adding powerful character simulation capabilities!