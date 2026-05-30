"""Runs API router — CRUD for persisted agent/orchestrator runs.

Provides endpoints to list, view, and delete runs that were saved
via ``persist=True`` on ``Agent.run()`` or ``BaseOrchestrator.run()``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _get_store(request: Request):
    """Get PicoStore from app state."""
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Persistence not available — install forla[persist]",
        )
    return store


@router.get("")
async def list_runs(
    request: Request,
    run_type: Optional[str] = None,
    agent_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List persisted runs, newest first."""
    store = _get_store(request)
    return await store.list_runs(
        run_type=run_type,
        agent_name=agent_name,
        limit=limit,
        offset=offset,
    )


@router.get("/{run_id}")
async def get_run(request: Request, run_id: str):
    """Get run metadata from DB."""
    store = _get_store(request)
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/data")
async def get_run_data(request: Request, run_id: str):
    """Read full run data from JSON file."""
    store = _get_store(request)
    data = await store.get_run_data(run_id)
    if data is None:
        raise HTTPException(
            status_code=404, detail="Run data not found"
        )
    return data


@router.delete("/{run_id}")
async def delete_run(request: Request, run_id: str):
    """Delete run DB row + JSON file."""
    store = _get_store(request)
    deleted = await store.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "deleted", "run_id": run_id}
