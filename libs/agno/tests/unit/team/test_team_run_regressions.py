import gc
import inspect
import warnings
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from agno.agent.agent import Agent
from agno.db.postgres import AsyncPostgresDb
from agno.models.response import ToolExecution
from agno.run import RunContext
from agno.run.base import RunStatus
from agno.run.messages import RunMessages
from agno.run.requirement import RunRequirement
from agno.run.team import TeamRunOutput
from agno.session import TeamSession
from agno.team import _hooks
from agno.team import _run as team_run
from agno.team.remote import RemoteTeam
from agno.team.team import Team


def test_all_team_pause_handlers_accept_run_context():
    for fn in [
        _hooks.handle_team_run_paused,
        _hooks.handle_team_run_paused_stream,
        _hooks.ahandle_team_run_paused,
        _hooks.ahandle_team_run_paused_stream,
    ]:
        params = inspect.signature(fn).parameters
        assert "run_context" in params, f"{fn.__name__} missing run_context param"


def test_remote_team_acontinue_run_drops_background_tasks_from_stream_request():
    remote_team = RemoteTeam(base_url="http://localhost:7777", team_id="team-1")
    client = MagicMock()
    client.continue_team_run_stream.return_value = iter(())
    remote_team.agentos_client = client

    remote_team.acontinue_run(
        "run-1",
        stream=True,
        session_id="session-1",
        background_tasks=object(),
        extra_flag=True,
    )

    call_kwargs = client.continue_team_run_stream.call_args.kwargs
    assert "background_tasks" not in call_kwargs
    assert call_kwargs["extra_flag"] is True


def test_remote_team_acontinue_run_preserves_requirements_positional_argument():
    remote_team = RemoteTeam(base_url="http://localhost:7777", team_id="team-1")
    client = MagicMock()
    client.continue_team_run.return_value = TeamRunOutput(run_id="run-1")
    remote_team.agentos_client = client
    requirement = RunRequirement(
        tool_execution=ToolExecution(tool_call_id="call-1", tool_name="approve_me", confirmed=True)
    )

    remote_team.acontinue_run("run-1", [requirement], stream=False, session_id="session-1")

    call_kwargs = client.continue_team_run.call_args.kwargs
    assert call_kwargs["requirements"] == [requirement]
    assert call_kwargs["tools"] is None


def test_team_tool_update_creates_audit_approval_for_confirmation(monkeypatch: pytest.MonkeyPatch):
    approvals: list[dict[str, Any]] = []

    class Db:
        def create_approval(self, data):
            approvals.append(data)

    monkeypatch.setattr("agno.agent._tools.reject_tool_call", lambda *args, **kwargs: None)

    team = Team(id="team-1", name="Audit Team", members=[Agent(name="m1")])
    team.db = Db()
    tool_execution = ToolExecution(
        tool_call_id="call-1",
        tool_name="audit_tool",
        tool_args={"x": 1},
        requires_confirmation=True,
        confirmed=False,
        approval_type="audit",
    )
    run_response = TeamRunOutput(run_id="team-run", session_id="session-1", tools=[tool_execution])

    team_run._handle_team_tool_call_updates(team, run_response, RunMessages(), tools=[])

    assert len(approvals) == 1
    assert approvals[0]["approval_type"] == "audit"
    assert approvals[0]["status"] == "rejected"
    assert approvals[0]["source_type"] == "team"
    assert approvals[0]["team_id"] == "team-1"
    assert approvals[0]["tool_name"] == "audit_tool"


@pytest.mark.asyncio
async def test_async_team_tool_update_creates_audit_approval_for_external_execution(monkeypatch: pytest.MonkeyPatch):
    approvals: list[dict[str, Any]] = []

    class Db:
        async def create_approval(self, data):
            approvals.append(data)

    monkeypatch.setattr("agno.agent._tools.handle_external_execution_update", lambda *args, **kwargs: None)

    team = Team(id="team-1", name="Audit Team", members=[Agent(name="m1")])
    team.db = Db()
    tool_execution = ToolExecution(
        tool_call_id="call-1",
        tool_name="audit_external",
        tool_args={"x": 1},
        external_execution_required=True,
        result="done",
        external_execution_result_provided=True,
        approval_type="audit",
    )
    run_response = TeamRunOutput(run_id="team-run", session_id="session-1", tools=[tool_execution])

    await team_run._ahandle_team_tool_call_updates(team, run_response, RunMessages(), tools=[])

    assert len(approvals) == 1
    assert approvals[0]["approval_type"] == "audit"
    assert approvals[0]["status"] == "approved"
    assert approvals[0]["pause_type"] == "external_execution"
    assert approvals[0]["source_type"] == "team"
    assert approvals[0]["team_id"] == "team-1"
    assert approvals[0]["tool_name"] == "audit_external"


def test_team_tool_update_preserves_audit_pause_type_for_user_input(monkeypatch: pytest.MonkeyPatch):
    approvals: list[dict[str, Any]] = []

    class Db:
        def create_approval(self, data):
            approvals.append(data)

    monkeypatch.setattr("agno.agent._tools.handle_user_input_update", lambda *args, **kwargs: None)
    monkeypatch.setattr("agno.agent._tools.run_tool", lambda *args, **kwargs: iter(()))

    team = Team(id="team-1", name="Audit Team", members=[Agent(name="m1")])
    team.db = Db()
    tool_execution = ToolExecution(
        tool_call_id="call-1",
        tool_name="audit_input",
        tool_args={"x": 1},
        requires_user_input=True,
        user_input_schema=[],
        approval_type="audit",
    )
    tool_execution.user_input_schema = [type("Field", (), {"name": "answer", "value": "yes"})()]
    run_response = TeamRunOutput(run_id="team-run", session_id="session-1", tools=[tool_execution])

    team_run._handle_team_tool_call_updates(team, run_response, RunMessages(), tools=[])

    assert len(approvals) == 1
    assert approvals[0]["approval_type"] == "audit"
    assert approvals[0]["pause_type"] == "user_input"


def test_handle_team_run_paused_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def spy_cleanup(team, run_response, session, run_context=None):
        captured["run_context"] = run_context

    monkeypatch.setattr(team_run, "_cleanup_and_store", spy_cleanup)
    monkeypatch.setattr("agno.run.approval.create_approval_from_pause", lambda **kwargs: None)

    team = Team(name="test-team", members=[Agent(name="m1")])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    _hooks.handle_team_run_paused(
        team=team,
        run_response=TeamRunOutput(run_id="r1", session_id="s1", messages=[]),
        session=TeamSession(session_id="s1"),
        run_context=run_context,
    )

    assert captured["run_context"] is run_context


@pytest.mark.asyncio
async def test_ahandle_team_run_paused_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def spy_acleanup(team, run_response, session, run_context=None):
        captured["run_context"] = run_context

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(team_run, "_acleanup_and_store", spy_acleanup)
    monkeypatch.setattr("agno.run.approval.acreate_approval_from_pause", noop_acreate_approval)

    team = Team(name="test-team", members=[Agent(name="m1")])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    await _hooks.ahandle_team_run_paused(
        team=team,
        run_response=TeamRunOutput(run_id="r1", session_id="s1", messages=[]),
        session=TeamSession(session_id="s1"),
        run_context=run_context,
    )

    assert captured["run_context"] is run_context


def test_handle_team_run_paused_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(team_run, "scrub_run_output_for_storage", lambda team, run_response: None)
    monkeypatch.setattr("agno.team._session.update_session_metrics", lambda team, session, run_response: None)
    monkeypatch.setattr("agno.run.approval.create_approval_from_pause", lambda **kwargs: None)

    team = Team(name="test-team", members=[Agent(name="m1")])
    monkeypatch.setattr(team, "save_session", lambda session: None)

    session = TeamSession(session_id="s1", session_data={})
    run_response = TeamRunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"watchlist": ["AAPL"]})

    result = _hooks.handle_team_run_paused(
        team=team,
        run_response=run_response,
        session=session,
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert session.session_data["session_state"] == {"watchlist": ["AAPL"]}
    assert result.session_state == {"watchlist": ["AAPL"]}


def test_handle_team_run_paused_without_run_context_does_not_set_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(team_run, "scrub_run_output_for_storage", lambda team, run_response: None)
    monkeypatch.setattr("agno.team._session.update_session_metrics", lambda team, session, run_response: None)
    monkeypatch.setattr("agno.run.approval.create_approval_from_pause", lambda **kwargs: None)

    team = Team(name="test-team", members=[Agent(name="m1")])
    monkeypatch.setattr(team, "save_session", lambda session: None)

    session = TeamSession(session_id="s1", session_data={})

    result = _hooks.handle_team_run_paused(
        team=team,
        run_response=TeamRunOutput(run_id="r1", session_id="s1", messages=[]),
        session=session,
    )

    assert result.status == RunStatus.paused
    assert "session_state" not in session.session_data


def test_handle_team_run_paused_persists_state_when_session_data_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(team_run, "scrub_run_output_for_storage", lambda team, run_response: None)
    monkeypatch.setattr("agno.team._session.update_session_metrics", lambda team, session, run_response: None)
    monkeypatch.setattr("agno.run.approval.create_approval_from_pause", lambda **kwargs: None)

    team = Team(name="test-team", members=[Agent(name="m1")])
    monkeypatch.setattr(team, "save_session", lambda session: None)

    session = TeamSession(session_id="s1", session_data=None)
    run_response = TeamRunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"watchlist": ["AAPL"]})

    result = _hooks.handle_team_run_paused(
        team=team,
        run_response=run_response,
        session=session,
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert result.session_state == {"watchlist": ["AAPL"]}
    assert session.session_data == {"session_state": {"watchlist": ["AAPL"]}}


@pytest.mark.asyncio
async def test_ahandle_team_run_paused_persists_state_when_session_data_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(team_run, "scrub_run_output_for_storage", lambda team, run_response: None)
    monkeypatch.setattr("agno.team._session.update_session_metrics", lambda team, session, run_response: None)

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr("agno.run.approval.acreate_approval_from_pause", noop_acreate_approval)

    team = Team(name="test-team", members=[Agent(name="m1")])

    async def noop_asave(session):
        return None

    monkeypatch.setattr(team, "asave_session", noop_asave)

    session = TeamSession(session_id="s1", session_data=None)
    run_response = TeamRunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"cart": ["item-1"]})

    result = await _hooks.ahandle_team_run_paused(
        team=team,
        run_response=run_response,
        session=session,
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert result.session_state == {"cart": ["item-1"]}
    assert session.session_data == {"session_state": {"cart": ["item-1"]}}


@pytest.mark.asyncio
async def test_ahandle_team_run_paused_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(team_run, "scrub_run_output_for_storage", lambda team, run_response: None)
    monkeypatch.setattr("agno.team._session.update_session_metrics", lambda team, session, run_response: None)

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr("agno.run.approval.acreate_approval_from_pause", noop_acreate_approval)

    team = Team(name="test-team", members=[Agent(name="m1")])

    async def noop_asave(session):
        return None

    monkeypatch.setattr(team, "asave_session", noop_asave)

    session = TeamSession(session_id="s1", session_data={})
    run_response = TeamRunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"cart": ["item-1"]})

    result = await _hooks.ahandle_team_run_paused(
        team=team,
        run_response=run_response,
        session=session,
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert session.session_data["session_state"] == {"cart": ["item-1"]}
    assert result.session_state == {"cart": ["item-1"]}


def test_handle_team_run_paused_stream_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def spy_cleanup(team, run_response, session, run_context=None):
        captured["run_context"] = run_context

    monkeypatch.setattr(team_run, "_cleanup_and_store", spy_cleanup)
    monkeypatch.setattr("agno.run.approval.create_approval_from_pause", lambda **kwargs: None)

    team = Team(name="test-team", members=[Agent(name="m1")])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    events = list(
        _hooks.handle_team_run_paused_stream(
            team=team,
            run_response=TeamRunOutput(run_id="r1", session_id="s1", messages=[]),
            session=TeamSession(session_id="s1"),
            run_context=run_context,
        )
    )

    assert captured["run_context"] is run_context
    assert len(events) >= 1


@pytest.mark.asyncio
async def test_ahandle_team_run_paused_stream_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def spy_acleanup(team, run_response, session, run_context=None):
        captured["run_context"] = run_context

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(team_run, "_acleanup_and_store", spy_acleanup)
    monkeypatch.setattr("agno.run.approval.acreate_approval_from_pause", noop_acreate_approval)

    team = Team(name="test-team", members=[Agent(name="m1")])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    events = []
    async for event in _hooks.ahandle_team_run_paused_stream(
        team=team,
        run_response=TeamRunOutput(run_id="r1", session_id="s1", messages=[]),
        session=TeamSession(session_id="s1"),
        run_context=run_context,
    ):
        events.append(event)

    assert captured["run_context"] is run_context
    assert len(events) >= 1


def test_handle_team_run_paused_stream_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(team_run, "scrub_run_output_for_storage", lambda team, run_response: None)
    monkeypatch.setattr("agno.team._session.update_session_metrics", lambda team, session, run_response: None)
    monkeypatch.setattr("agno.run.approval.create_approval_from_pause", lambda **kwargs: None)

    team = Team(name="test-team", members=[Agent(name="m1")])
    monkeypatch.setattr(team, "save_session", lambda session: None)

    session = TeamSession(session_id="s1", session_data={})
    run_response = TeamRunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"watchlist": ["AAPL"]})

    events = list(
        _hooks.handle_team_run_paused_stream(
            team=team,
            run_response=run_response,
            session=session,
            run_context=run_context,
        )
    )

    assert len(events) >= 1
    assert session.session_data["session_state"] == {"watchlist": ["AAPL"]}
    assert run_response.session_state == {"watchlist": ["AAPL"]}


@pytest.mark.asyncio
async def test_ahandle_team_run_paused_stream_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(team_run, "scrub_run_output_for_storage", lambda team, run_response: None)
    monkeypatch.setattr("agno.team._session.update_session_metrics", lambda team, session, run_response: None)

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr("agno.run.approval.acreate_approval_from_pause", noop_acreate_approval)

    team = Team(name="test-team", members=[Agent(name="m1")])

    async def noop_asave(session):
        return None

    monkeypatch.setattr(team, "asave_session", noop_asave)

    session = TeamSession(session_id="s1", session_data={})
    run_response = TeamRunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"cart": ["item-1"]})

    events = []
    async for event in _hooks.ahandle_team_run_paused_stream(
        team=team,
        run_response=run_response,
        session=session,
        run_context=run_context,
    ):
        events.append(event)

    assert len(events) >= 1
    assert session.session_data["session_state"] == {"cart": ["item-1"]}
    assert run_response.session_state == {"cart": ["item-1"]}


# ---------------------------------------------------------------------------
# Sync session APIs must reject an async DB instead of silently leaking a
# coroutine (regression for the un-awaited db.<call>() path).
# ---------------------------------------------------------------------------


@pytest.fixture
def team_with_async_postgres_db() -> Team:
    engine = Mock(spec=AsyncEngine)
    db = AsyncPostgresDb(
        db_engine=engine,
        db_schema="test_schema",
        session_table="test_sessions",
    )
    return Team(members=[], name="test-team", db=db, session_id="test-session")


def _assert_raises_without_unawaited_warning(callable_to_test):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ValueError, match="Cannot use sync .* with an async database"):
            callable_to_test()
        gc.collect()

    assert not [
        warning
        for warning in caught
        if warning.category is RuntimeWarning and "was never awaited" in str(warning.message)
    ]


def test_sync_team_get_session_rejects_async_postgres_db_without_leaking_coroutines(
    team_with_async_postgres_db: Team,
):
    _assert_raises_without_unawaited_warning(lambda: team_with_async_postgres_db.get_session(session_id="test-session"))


def test_sync_team_save_session_rejects_async_postgres_db_without_leaking_coroutines(
    team_with_async_postgres_db: Team,
):
    session = TeamSession(
        session_id="test-session",
        team_id="test-team",
        session_data={},
    )

    _assert_raises_without_unawaited_warning(lambda: team_with_async_postgres_db.save_session(session=session))


def test_sync_team_delete_session_rejects_async_postgres_db_without_leaking_coroutines(
    team_with_async_postgres_db: Team,
):
    _assert_raises_without_unawaited_warning(
        lambda: team_with_async_postgres_db.delete_session(session_id="test-session")
    )
