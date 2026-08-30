from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agno.agent import _init
from agno.agent._storage import aupsert_run, aupsert_session, upsert_run, upsert_session
from agno.run import status_persist
from agno.run.agent import RunOutput
from agno.session import AgentSession


def test_upsert_session_propagates_database_errors():
    error = RuntimeError("session write failed")
    db = SimpleNamespace(upsert_session=MagicMock(side_effect=error))
    agent = SimpleNamespace(db=db)

    with pytest.raises(RuntimeError) as exc_info:
        upsert_session(agent, AgentSession(session_id="session-1"))

    assert exc_info.value is error


@pytest.mark.asyncio
async def test_aupsert_session_propagates_database_errors(monkeypatch):
    error = RuntimeError("async session write failed")
    db = SimpleNamespace(upsert_session=AsyncMock(side_effect=error))
    agent = SimpleNamespace(db=db)
    monkeypatch.setattr(_init, "has_async_db", lambda _agent: True)

    with pytest.raises(RuntimeError) as exc_info:
        await aupsert_session(agent, AgentSession(session_id="session-1"))

    assert exc_info.value is error


def test_upsert_session_propagates_not_implemented_error():
    db = SimpleNamespace(upsert_session=MagicMock(side_effect=NotImplementedError))
    agent = SimpleNamespace(db=db)

    with pytest.raises(NotImplementedError):
        upsert_session(agent, AgentSession(session_id="session-1"))


@pytest.mark.asyncio
async def test_aupsert_session_propagates_not_implemented_error(monkeypatch):
    db = SimpleNamespace(upsert_session=AsyncMock(side_effect=NotImplementedError))
    agent = SimpleNamespace(db=db)
    monkeypatch.setattr(_init, "has_async_db", lambda _agent: True)

    with pytest.raises(NotImplementedError):
        await aupsert_session(agent, AgentSession(session_id="session-1"))


def test_upsert_run_propagates_database_errors():
    error = RuntimeError("run write failed")
    db = SimpleNamespace(upsert_run=MagicMock(side_effect=error))
    agent = SimpleNamespace(db=db)

    with pytest.raises(RuntimeError) as exc_info:
        upsert_run(agent, RunOutput(run_id="run-1"), session_id="session-1")

    assert exc_info.value is error


@pytest.mark.asyncio
async def test_aupsert_run_propagates_database_errors(monkeypatch):
    error = RuntimeError("async run write failed")
    db = SimpleNamespace(upsert_run=AsyncMock(side_effect=error))
    agent = SimpleNamespace(db=db)
    monkeypatch.setattr(_init, "has_async_db", lambda _agent: True)

    with pytest.raises(RuntimeError) as exc_info:
        await aupsert_run(agent, RunOutput(run_id="run-1"), session_id="session-1")

    assert exc_info.value is error


def test_upsert_run_ignores_not_implemented_error():
    db = SimpleNamespace(upsert_run=MagicMock(side_effect=NotImplementedError))
    agent = SimpleNamespace(db=db)

    upsert_run(agent, RunOutput(run_id="run-1"), session_id="session-1")

    db.upsert_run.assert_called_once()


@pytest.mark.asyncio
async def test_aupsert_run_ignores_not_implemented_error(monkeypatch):
    db = SimpleNamespace(upsert_run=AsyncMock(side_effect=NotImplementedError))
    agent = SimpleNamespace(db=db)
    monkeypatch.setattr(_init, "has_async_db", lambda _agent: True)

    await aupsert_run(agent, RunOutput(run_id="run-1"), session_id="session-1")

    db.upsert_run.assert_awaited_once()


def test_upsert_run_does_not_ignore_not_implemented_error_from_fencing(monkeypatch):
    db = SimpleNamespace(upsert_run=MagicMock())
    agent = SimpleNamespace(db=db)
    monkeypatch.setattr(status_persist, "persist_worker_owned_run", MagicMock(side_effect=NotImplementedError))

    with pytest.raises(NotImplementedError):
        upsert_run(agent, RunOutput(run_id="run-1"), session_id="session-1")

    db.upsert_run.assert_not_called()


@pytest.mark.asyncio
async def test_aupsert_run_does_not_ignore_not_implemented_error_from_fencing(monkeypatch):
    db = SimpleNamespace(upsert_run=AsyncMock())
    agent = SimpleNamespace(db=db)
    monkeypatch.setattr(status_persist, "apersist_worker_owned_run", AsyncMock(side_effect=NotImplementedError))

    with pytest.raises(NotImplementedError):
        await aupsert_run(agent, RunOutput(run_id="run-1"), session_id="session-1")

    db.upsert_run.assert_not_awaited()
