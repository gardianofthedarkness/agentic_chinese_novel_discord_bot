"""
Application Configuration
==========================
All configuration in one place. Values come from environment variables.
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProcessingMode(Enum):
    AGENT = "agent"          # ReAct agentic loop (default)
    BATCH = "batch"          # Bulk ingestion without agent loop


class DatabaseBackend(Enum):
    NEO4J = "neo4j"
    POSTGRESQL = "postgresql"
    HYBRID = "hybrid"        # Neo4j primary + PostgreSQL fallback


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    # ---- LLM ---------------------------------------------------------------
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7

    # ---- Neo4j -------------------------------------------------------------
    use_neo4j: bool = field(default_factory=lambda: os.getenv("USE_NEO4J", "true").lower() == "true")
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))

    # ---- PostgreSQL --------------------------------------------------------
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "novel_processing"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "novel_user"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    use_postgres: bool = field(default_factory=lambda: os.getenv("USE_POSTGRES", "true").lower() == "true")
    fallback_to_postgres: bool = True

    # ---- Qdrant ------------------------------------------------------------
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "novel_vectors"))
    use_qdrant: bool = field(default_factory=lambda: os.getenv("USE_QDRANT", "false").lower() == "true")

    # ---- Agent / Processing ------------------------------------------------
    mode: ProcessingMode = ProcessingMode.AGENT
    database_backend: DatabaseBackend = DatabaseBackend.HYBRID
    agent_max_iterations: int = 20
    agent_satisfaction_threshold: float = 0.80

    # ---- Redis -------------------------------------------------------------
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # ---- API Server --------------------------------------------------------
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "5005")))

    # ---- Logging -----------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def get_postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def validate(self) -> None:
        if self.use_postgres and not self.postgres_password:
            raise ValueError("POSTGRES_PASSWORD is required when USE_POSTGRES=true")
        if not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")


# Module-level singleton
config = AppConfig()
