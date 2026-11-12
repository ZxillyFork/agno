"""Check that every Agent continue path passes the continued run to the message builder.

The builder uses the run to keep it (and its fork tree) out of its own history. A call
site that stops passing it still builds valid messages, so history-level tests on other
paths keep passing while that path regresses. These tests drive the real dispatch
functions and stop them at the message-builder call with a ``BaseException`` sentinel,
which passes through every ``except Exception`` handler on the continue path, so
nothing downstream (model call, tool execution, persistence) needs mocking.

Covers the three call sites: ``continue_run_dispatch``, ``_acontinue_run`` and
``_acontinue_run_stream``.
"""

import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")

import pytest

from agno.agent import _init, _messages, _response, _run, _storage, _tools
from agno.agent.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.run import RunContext
from agno.run.agent import RunOutput
from agno.run.base import RunStatus
from agno.session import AgentSession
from agno.tools import tool

RUN_ID = "run-continue-wiring"
SESSION_ID = "session-wiring"


class _StoppedAtMessageBuilder(BaseException):
    """Raised the instant the message builder is reached."""


class _ContextReadingModel(Model):
    def _reply(self, messages):
        if any(message.role == "tool" for message in messages):
            return ModelResponse(role="assistant", content="continued")
        return ModelResponse(
            role="assistant",
            tool_calls=[
                {
                    "id": "read-resume-context",
                    "type": "function",
                    "function": {"name": "read_resume_context", "arguments": "{}"},
                }
            ],
        )

    def invoke(self, messages, **kwargs):
        return self._reply(messages)

    async def ainvoke(self, messages, **kwargs):
        return self._reply(messages)

    def invoke_stream(self, messages, **kwargs):
        yield self._reply(messages)

    async def ainvoke_stream(self, messages, **kwargs):
        yield self._reply(messages)

    def _parse_provider_response(self, response, **kwargs):
        return response

    def _parse_provider_response_delta(self, response):
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
async def test_restored_continuation_binds_messages_to_tool_run_context(async_mode, stream):
    observed = []

    @tool(requires_confirmation=True)
    def read_resume_context(run_context: RunContext) -> str:
        observed.append(run_context.messages)
        return "history-found" if run_context.messages else "missing-history"

    db = InMemoryDb()
    initial = Agent(
        id="context-reader",
        model=_ContextReadingModel(id="scripted"),
        tools=[read_resume_context],
        db=db,
        telemetry=False,
    )
    kwargs = {"session_id": SESSION_ID, "stream": stream, "stream_events": True, "yield_run_output": True}
    if async_mode:
        if stream:
            events = [event async for event in initial.arun("resume-context-token", **kwargs)]
            paused = next(event for event in reversed(events) if isinstance(event, RunOutput))
        else:
            paused = await initial.arun("resume-context-token", **kwargs)
    elif stream:
        events = list(initial.run("resume-context-token", **kwargs))
        paused = next(event for event in reversed(events) if isinstance(event, RunOutput))
    else:
        paused = initial.run("resume-context-token", **kwargs)
    assert paused.status == RunStatus.paused
    assert observed == []

    restored = RunOutput.from_dict(json.loads(json.dumps(paused.to_dict())))
    restored.requirements[0].confirm()
    resumed = Agent(
        id="context-reader",
        model=_ContextReadingModel(id="scripted"),
        tools=[read_resume_context],
        db=db,
        telemetry=False,
    )
    kwargs["requirements"] = restored.requirements
    if async_mode:
        if stream:
            events = [event async for event in resumed.acontinue_run(restored, **kwargs)]
            completed = next(event for event in reversed(events) if isinstance(event, RunOutput))
        else:
            completed = await resumed.acontinue_run(restored, **kwargs)
    elif stream:
        events = list(resumed.continue_run(restored, **kwargs))
        completed = next(event for event in reversed(events) if isinstance(event, RunOutput))
    else:
        completed = resumed.continue_run(restored, **kwargs)

    assert completed.status == RunStatus.completed
    assert len(observed) == 1
    assert observed[0] is not None
    assert any(message.role == "user" and message.content == "resume-context-token" for message in observed[0])
    assert completed.tools[0].result == "history-found"


def _make_run() -> RunOutput:
    return RunOutput(
        run_id=RUN_ID,
        session_id=SESSION_ID,
        status=RunStatus.running,
        tools=[],
        requirements=None,
        messages=[],
    )


def _patch_sync(monkeypatch: pytest.MonkeyPatch, agent: Agent, runs) -> None:
    monkeypatch.setattr(_init, "has_async_db", lambda agent: False)
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: None)
    monkeypatch.setattr(_tools, "determine_tools_for_model", lambda *a, **kw: [])
    monkeypatch.setattr(
        _storage,
        "read_or_create_session",
        lambda agent, session_id=None, user_id=None: AgentSession(session_id=session_id, user_id=user_id, runs=runs),
    )
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)


def _patch_async(monkeypatch: pytest.MonkeyPatch, agent: Agent, runs) -> None:
    async def fake_aread_or_create_session(agent, session_id=None, user_id=None):
        return AgentSession(session_id=session_id, user_id=user_id, runs=runs)

    monkeypatch.setattr(_init, "has_async_db", lambda agent: False)
    monkeypatch.setattr(_storage, "aread_or_create_session", fake_aread_or_create_session)
    monkeypatch.setattr(
        _storage,
        "read_or_create_session",
        lambda agent, session_id=None, user_id=None: AgentSession(session_id=session_id, user_id=user_id, runs=runs),
    )
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: None)
    monkeypatch.setattr(_tools, "determine_tools_for_model", lambda *a, **kw: [])
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)


def test_sync_continue_run_dispatch_passes_run_id_to_message_builder(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    _patch_sync(monkeypatch, agent, runs=[_make_run()])

    captured: dict = {}

    def fake_get_continue_run_messages(
        agent, input, session=None, add_history_to_context=None, run_context=None, run_response=None
    ):
        captured["run_id"] = run_response.run_id if run_response is not None else None
        raise _StoppedAtMessageBuilder()

    monkeypatch.setattr(_messages, "get_continue_run_messages", fake_get_continue_run_messages)

    with pytest.raises(_StoppedAtMessageBuilder):
        _run.continue_run_dispatch(agent=agent, run_id=RUN_ID, session_id=SESSION_ID, stream=False)

    assert captured.get("run_id") == RUN_ID


@pytest.mark.asyncio
async def test_async_continue_run_passes_run_id_to_message_builder(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    _patch_async(monkeypatch, agent, runs=[_make_run()])

    captured: dict = {}

    async def fake_aget_continue_run_messages(
        agent, input, session=None, add_history_to_context=None, run_context=None, run_response=None
    ):
        captured["run_id"] = run_response.run_id if run_response is not None else None
        raise _StoppedAtMessageBuilder()

    monkeypatch.setattr(_messages, "aget_continue_run_messages", fake_aget_continue_run_messages)

    with pytest.raises(_StoppedAtMessageBuilder):
        await _run.acontinue_run_dispatch(agent=agent, run_id=RUN_ID, session_id=SESSION_ID, stream=False)

    assert captured.get("run_id") == RUN_ID


@pytest.mark.asyncio
async def test_async_stream_continue_run_passes_run_id_to_message_builder(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    _patch_async(monkeypatch, agent, runs=[_make_run()])

    captured: dict = {}

    async def fake_aget_continue_run_messages(
        agent, input, session=None, add_history_to_context=None, run_context=None, run_response=None
    ):
        captured["run_id"] = run_response.run_id if run_response is not None else None
        raise _StoppedAtMessageBuilder()

    monkeypatch.setattr(_messages, "aget_continue_run_messages", fake_aget_continue_run_messages)

    agen = _run.acontinue_run_dispatch(agent=agent, run_id=RUN_ID, session_id=SESSION_ID, stream=True)

    with pytest.raises(_StoppedAtMessageBuilder):
        async for _ in agen:
            pass

    assert captured.get("run_id") == RUN_ID
