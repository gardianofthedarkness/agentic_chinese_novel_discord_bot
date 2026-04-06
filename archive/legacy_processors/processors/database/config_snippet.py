"""
Configuration snippet to add to unified_novel_processor.py ProcessingConfig

Add these fields to the @dataclass ProcessingConfig class:
"""

# ============================================================================
# ADD TO ProcessingConfig (around line 66-88)
# ============================================================================

"""
    # ... existing fields ...

    # Neo4j settings (NEW!)
    use_neo4j: bool = False
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "novelprocessing2024")

    # Database backend selection (NEW!)
    database_backend: DatabaseBackend = DatabaseBackend.POSTGRESQL  # or HYBRID
    enable_smart_routing: bool = False  # Auto-route queries to optimal DB
    fallback_to_postgres: bool = True   # Fallback if Neo4j unavailable
"""

# ============================================================================
# IMPORT ADDITIONS (add to top of unified_novel_processor.py)
# ============================================================================

"""
# Add after existing imports:
from database import DatabaseAdapter, DatabaseBackend
from database.base_adapter import TimelineEvent, CausalLink, CharacterData
"""

# ============================================================================
# CONFIGURATION PRESETS (add after ProcessingConfig definition)
# ============================================================================

PRESET_CONFIGS = {
    'legacy': {
        'use_postgres': True,
        'use_neo4j': False,
        'database_backend': 'postgresql'
    },

    'neo4j_only': {
        'use_postgres': False,
        'use_neo4j': True,
        'database_backend': 'neo4j',
        'fallback_to_postgres': False
    },

    'hybrid_optimal': {  # RECOMMENDED
        'use_postgres': True,
        'use_neo4j': True,
        'database_backend': 'hybrid',
        'enable_smart_routing': True,
        'fallback_to_postgres': True
    },

    'testing': {
        'use_postgres': False,
        'use_neo4j': False,
        'database_backend': 'sqlite'
    }
}

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# Example 1: Use hybrid mode (recommended)
config = ProcessingConfig(
    mode=ProcessingMode.ITERATIVE,
    use_postgres=True,
    use_neo4j=True,
    enable_smart_routing=True,  # Automatic query routing
    fallback_to_postgres=True
)

# Example 2: Neo4j only (maximum performance)
config = ProcessingConfig(
    mode=ProcessingMode.ITERATIVE,
    use_postgres=False,
    use_neo4j=True,
    neo4j_uri="bolt://localhost:7687"
)

# Example 3: Legacy PostgreSQL only (backwards compatible)
config = ProcessingConfig(
    mode=ProcessingMode.ITERATIVE,
    use_postgres=True,
    use_neo4j=False
)
"""
