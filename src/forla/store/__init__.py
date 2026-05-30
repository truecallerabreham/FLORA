"""Forla persistence store.

Provides database-backed storage for agent runs, eval results,
datasets, and target configurations. Uses SQLModel with async
SQLAlchemy for database access.

Usage:
    from forla.store import PicoStore, get_default_store

    # Default SQLite store
    store = get_default_store()
    await store.save_agent_run(agent, response)

    # Custom connection
    store = PicoStore("postgresql+asyncpg://user:pass@host/db")
"""

from ._models import (
    DBDataset,
    DBEvalResult,
    DBEvalRun,
    DBRun,
    DBTask,
    DBTargetConfig,
)
from ._store import PicoStore, get_default_store

__all__ = [
    "PicoStore",
    "get_default_store",
    "DBRun",
    "DBDataset",
    "DBTask",
    "DBTargetConfig",
    "DBEvalRun",
    "DBEvalResult",
]
