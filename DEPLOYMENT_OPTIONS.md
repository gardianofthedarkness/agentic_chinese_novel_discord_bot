# Database and Deployment Options

## Current Database Architecture Analysis

### ✅ Currently Dockerized
- **Qdrant Vector Database**: Already running in Docker with persistent volume
- **Original Backend/Frontend**: Containerized Flask API and React frontend

### ❌ Not Currently Dockerized  
- **Character Memory**: SQLite files on host filesystem (`character_data/`)
- **Timeline System**: New SQLite-based system (planned)
- **Discord Bot**: Running on host
- **MCP Server**: Running on host

## Deployment Options

### Option 1: Full Docker (Recommended for Production)

**File**: `docker-compose-enhanced.yml`

**Services**:
- PostgreSQL (timeline and character data)
- Qdrant (vector database) 
- Redis (caching and sessions)
- MCP Server (containerized)
- Discord Bot (containerized)
- Web Interface (timeline management)

**Advantages**:
- ✅ Complete isolation and portability
- ✅ Easy scaling and load balancing
- ✅ Consistent environments (dev/staging/prod)
- ✅ Better backup and recovery
- ✅ Production-ready with PostgreSQL

**Setup**:
```bash
# Copy Docker environment
cp .env.docker .env
# Edit with your tokens and keys
notepad .env
# Start full system
docker-compose -f docker-compose-enhanced.yml up --build
```

### Option 2: Hybrid Docker (Recommended for Migration)

**File**: `docker-compose-hybrid.yml`

**Keeps Existing**:
- Your current Qdrant setup
- Your Flask backend
- Your React frontend
- SQLite character data on host

**Adds in Docker**:
- PostgreSQL for timeline data
- MCP Server container
- Optional Discord Bot container

**Advantages**:
- ✅ Minimal disruption to existing setup
- ✅ Gradual migration path
- ✅ Better timeline data management
- ✅ Maintains file-based character data

**Setup**:
```bash
# Use existing .env or create from template
cp .env.example .env
# Start hybrid system
docker-compose -f docker-compose-hybrid.yml up --build
```

### Option 3: Host-Only (Current State + Enhancements)

**Keeps everything on host but with enhancements**:
- SQLite for timeline data
- Enhanced character memory system
- Host-based MCP server
- Host-based Discord bot

**Advantages**:
- ✅ No Docker complexity
- ✅ Easy debugging and development
- ✅ Direct file system access
- ✅ Lower resource overhead

**Disadvantages**:
- ❌ Less portable
- ❌ Manual dependency management
- ❌ Harder to scale
- ❌ SQLite limitations for concurrent access

## Database Adapter System

I've created a flexible database adapter (`database_adapter.py`) that supports both:

### SQLite Adapter (Development/Small Scale)
```python
# Automatically used when no DATABASE_URL is set
adapter = SQLiteAdapter("character_data/timeline.db")
```

### PostgreSQL Adapter (Production/Scale)
```python
# Automatically used when DATABASE_URL is set
adapter = PostgreSQLAdapter("postgresql://user:pass@host:port/db")
```

**The system automatically detects which to use based on environment variables.**

## Migration Paths

### From Current Setup → Hybrid Docker
1. Keep existing character data
2. Add PostgreSQL for timeline system
3. Containerize MCP server
4. Optionally containerize Discord bot

### From Hybrid → Full Docker
1. Migrate character data from SQLite to PostgreSQL
2. Move all services to containers
3. Use Docker volumes for persistence
4. Add Redis for better performance

### Data Migration Script
```python
# Migrate existing SQLite data to PostgreSQL
python migrate_to_postgres.py --source sqlite://character_data/timeline.db --target postgresql://user:pass@localhost/db
```

## Recommended Approach

### For Development
```bash
# Start with hybrid approach
run_docker_hybrid.bat
```

### For Production
```bash
# Use full Docker setup
run_docker_full.bat
```

### For Minimal Setup
```bash
# Continue with host-only
run_character_simulation.bat
```

## Environment Configuration

### Docker Full Setup (.env.docker)
```env
DATABASE_URL=postgresql://timeline_user:timeline_pass@postgres:5432/character_simulation
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379
MCP_SERVER_URL=http://mcp_server:8000
```

### Hybrid Setup (.env)
```env
TIMELINE_DATABASE_URL=postgresql://timeline_user:timeline_pass@localhost:5433/timeline_db
QDRANT_URL=http://localhost:6333
MCP_SERVER_URL=http://localhost:8000
```

### Host-Only Setup (.env)
```env
# Uses SQLite by default
QDRANT_URL=http://localhost:6333
CHARACTER_DATA_DIR=./character_data
```

## Service Health Monitoring

All containers include health checks:
- **MCP Server**: HTTP health endpoint
- **Discord Bot**: Connection status check
- **PostgreSQL**: Database connection test
- **Redis**: Ping response test

## Backup Strategies

### Docker Volumes
```bash
# Backup PostgreSQL data
docker run --rm -v mojin_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data

# Backup Qdrant data
docker run --rm -v mojin_qdrant_data:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_backup.tar.gz /data
```

### Host Files
```bash
# Backup character data directory
tar -czf character_data_backup.tar.gz character_data/
```

## Performance Considerations

### SQLite vs PostgreSQL
- **SQLite**: Good for <100 concurrent users, simple setup
- **PostgreSQL**: Better for >100 users, complex queries, JSON operations

### Resource Requirements
- **Host-Only**: ~500MB RAM, minimal CPU
- **Hybrid**: ~1GB RAM, moderate CPU
- **Full Docker**: ~2GB RAM, higher CPU (multiple containers)

## Security Considerations

### Docker Network Isolation
- Services communicate on internal Docker network
- Only necessary ports exposed to host
- Database credentials isolated in containers

### File Permissions
- Docker volumes use proper user mapping
- SQLite files need correct permissions for container access

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Change ports in docker-compose files
2. **Permission Issues**: Check Docker volume permissions
3. **Memory Issues**: Adjust container memory limits
4. **Network Issues**: Verify Docker network configuration

### Debug Commands
```bash
# Check container logs
docker-compose logs mcp_server
docker-compose logs discord_bot

# Access database
docker-compose exec postgres psql -U timeline_user character_simulation

# Check service health
curl http://localhost:8000/health
```

## Next Steps

1. **Choose your deployment option** based on your needs
2. **Set up environment variables** with your tokens/keys
3. **Run the appropriate startup script**
4. **Test the system** with Discord commands
5. **Monitor logs** for any issues
6. **Scale as needed** by moving between options

The database adapter system ensures your code works the same regardless of which option you choose!