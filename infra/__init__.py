"""
infra
=====
Redis-powered infrastructure layer:

  RedisPool          — shared async connection pool
  TokenStreamer      — token-level streaming over Redis pub/sub
  BackgroundExecutor — tool execution with 5s timeout → background promotion
  BackgroundHandle   — reference to a running background job
  InfraHarness       — top-level wiring object (inject this into everything)
"""

from infra.redis_client import RedisPool, STREAM_END
from infra.token_stream import TokenStreamer
from infra.background_exec import BackgroundExecutor, BackgroundHandle
from infra.harness import InfraHarness

__all__ = [
    "RedisPool",
    "STREAM_END",
    "TokenStreamer",
    "BackgroundExecutor",
    "BackgroundHandle",
    "InfraHarness",
]
