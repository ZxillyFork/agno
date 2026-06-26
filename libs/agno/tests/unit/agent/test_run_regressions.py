import inspect
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from agno.agent import _init, _messages, _response, _run, _session, _storage, _telemetry, _tools
from agno.agent.agent import Agent
from agno.db.base import SessionType
from agno.exceptions import RunCancelledException, RunNotFoundError
from agno.models.message import Message
from agno.models.response import ModelResponse, ToolExecution
from agno.run import RunContext
from agno.run.agent import RunErrorEvent, RunOutput
from agno.run.base import RunStatus
from agno.run.cancel import (
    cancel_run,
    cleanup_run,
    get_active_runs,
    get_cancellation_manager,
    is_cancelled,
    register_run,
    set_cancellation_manager,
)
from agno.run.cancellation_management.in_memory_cancellation_manager import InMemoryRunCancellationManager
from agno.run.messages import RunMessages
from agno.run.requirement import RunRequirement
from agno.session import AgentSession


@pytest.fixture(autouse=True)
def reset_cancellation_manager():
    original_manager = get_cancellation_manager()
    set_cancellation_manager(InMemoryRunCancellationManager())
    try:
        yield
    finally:
        set_cancellation_manager(original_manager)


def _orphaned_tool_call_messages() -> list[Message]:
    return [
        Message(
            role="assistant",
            content="Calling a tool",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "arguments": '{"to":"john@doe.com"}',
                    },
                }
            ],
        )
    ]


def test_repair_orphaned_tool_calls_only_runs_for_cancelled_status():
    run_response = RunOutput(
        status=RunStatus.cancelled,
        messages=_orphaned_tool_call_messages(),
    )

    _run.repair_orphaned_tool_calls(run_response)

    assert run_response.messages is not None
    assert len(run_response.messages) == 2
    assert run_response.messages[1].role == "tool"
    assert run_response.messages[1].tool_call_id == "call_123"
    assert run_response.messages[1].tool_call_error is True


def test_repair_orphaned_tool_calls_skips_paused_external_execution_runs():
    run_response = RunOutput(
        status=RunStatus.paused,
        messages=_orphaned_tool_call_messages(),
    )

    _run.repair_orphaned_tool_calls(run_response)

    assert run_response.messages is not None
    assert len(run_response.messages) == 1
    assert run_response.messages[0].role == "assistant"


def test_repair_orphaned_tool_calls_recognizes_responses_call_id_result():
    run_response = RunOutput(
        status=RunStatus.cancelled,
        messages=[
            Message(
                role="assistant",
                content="Calling a tool",
                tool_calls=[
                    {
                        "id": "fc_123",
                        "call_id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": "{}",
                        },
                    }
                ],
            ),
            Message(role="tool", tool_call_id="call_123", content="tool result"),
        ],
    )

    _run.repair_orphaned_tool_calls(run_response)

    assert run_response.messages is not None
    assert len(run_response.messages) == 2
    assert [msg.tool_call_id for msg in run_response.messages if msg.role == "tool"] == ["call_123"]


def test_repair_orphaned_tool_calls_preserves_history_marker_on_synthetic_results():
    run_response = RunOutput(
        status=RunStatus.cancelled,
        messages=[
            Message(
                role="assistant",
                content="Calling a historical tool",
                from_history=True,
                tool_calls=[
                    {
                        "id": "fc_history",
                        "call_id": "call_history",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": "{}",
                        },
                    }
                ],
            ),
            Message(role="user", content="continue"),
        ],
    )

    _run.repair_orphaned_tool_calls(run_response)

    assert run_response.messages is not None
    assert len(run_response.messages) == 3
    assert run_response.messages[1].role == "tool"
    assert run_response.messages[1].from_history is True

    _run.scrub_run_output_for_storage(Agent(name="test-agent"), run_response)

    assert run_response.messages is not None
    assert len(run_response.messages) == 1
    assert run_response.messages[0].role == "user"
    assert run_response.messages[0].content == "continue"


def _patch_sync_dispatch_dependencies(
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    runs: Optional[list[Any]] = None,
) -> None:
    monkeypatch.setattr(_init, "has_async_db", lambda agent: False)
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)
    monkeypatch.setattr(_run, "resolve_run_dependencies", lambda agent, run_context: None)
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: None)
    monkeypatch.setattr(
        _storage,
        "read_or_create_session",
        lambda agent, session_id=None, user_id=None: AgentSession(session_id=session_id, user_id=user_id, runs=runs),
    )


def test_run_dispatch_cleans_up_registered_run_on_setup_failure(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    _patch_sync_dispatch_dependencies(agent, monkeypatch, runs=[])

    def failing_initialize_agent(debug_mode=None):
        raise RuntimeError("initialize failed")

    monkeypatch.setattr(agent, "initialize_agent", failing_initialize_agent)

    run_id = "run-setup-fail"
    with pytest.raises(RuntimeError, match="initialize failed"):
        _run.run_dispatch(agent=agent, input="hello", run_id=run_id, stream=False)

    assert run_id not in get_active_runs()


def test_run_dispatch_does_not_reset_cancellation_before_impl(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    _patch_sync_dispatch_dependencies(agent, monkeypatch, runs=[])

    run_id = "run-preserve-cancelled-state"

    def initialize_and_cancel(debug_mode=None):
        # register_run now happens inside _run, so we register here to test cancellation
        register_run(run_id)
        assert cancel_run(run_id) is True

    monkeypatch.setattr(agent, "initialize_agent", initialize_and_cancel)

    observed: dict[str, bool] = {}

    def fake_run_impl(
        agent: Agent,
        run_response,
        run_context,
        session_id: str = "",
        user_id: Optional[str] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        response_format: Optional[Any] = None,
        debug_mode: Optional[bool] = None,
        background_tasks: Optional[Any] = None,
        **kwargs: Any,
    ):
        observed["cancelled_before_model"] = is_cancelled(run_response.run_id)  # type: ignore[arg-type]
        cleanup_run(run_response.run_id)  # type: ignore[arg-type]
        return run_response

    monkeypatch.setattr(_run, "_run", fake_run_impl)

    _run.run_dispatch(agent=agent, input="hello", run_id=run_id, stream=False)

    assert observed["cancelled_before_model"] is True
    assert run_id not in get_active_runs()


def test_continue_run_dispatch_handles_none_session_runs(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    monkeypatch.setattr(_init, "has_async_db", lambda agent: False)
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)
    monkeypatch.setattr(
        _storage,
        "read_or_create_session",
        lambda agent, session_id=None, user_id=None: AgentSession(session_id=session_id, user_id=user_id, runs=None),
    )

    with pytest.raises(RuntimeError, match="No runs found for run ID missing-run"):
        _run.continue_run_dispatch(
            agent=agent,
            run_id="missing-run",
            requirements=[],
            session_id="session-1",
        )


@pytest.mark.asyncio
async def test_acontinue_run_dispatch_handles_none_session_runs(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)

    async def fake_aread_or_create_session(agent, session_id: str, user_id: Optional[str] = None):
        return AgentSession(session_id=session_id, user_id=user_id, runs=None)

    async def fake_acleanup_and_store(agent, **kwargs: Any):
        return None

    async def fake_disconnect_mcp_tools(agent):
        return None

    monkeypatch.setattr(_storage, "aread_or_create_session", fake_aread_or_create_session)
    monkeypatch.setattr(_run, "acleanup_and_store", fake_acleanup_and_store)
    monkeypatch.setattr(_init, "disconnect_connectable_tools", lambda agent: None)
    monkeypatch.setattr(_init, "disconnect_mcp_tools", fake_disconnect_mcp_tools)

    # An unresolvable run_id must RAISE, not return a terminal error run. The old
    # behaviour fell through to the generic handler, which stamped an ERROR row over
    # the target run (owner, status and content) or fabricated a junk row.
    with pytest.raises(RunNotFoundError, match="No runs found for run ID missing-run"):
        await _run.acontinue_run_dispatch(
            agent=agent,
            run_id="missing-run",
            requirements=[],
            session_id="session-1",
            stream=False,
        )


@pytest.mark.asyncio
async def test_acontinue_run_stream_yields_error_event_without_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = Agent(name="test-agent")
    run_id = "missing-stream-run"

    async def fake_aread_or_create_session(agent, session_id: str, user_id: Optional[str] = None):
        return AgentSession(session_id=session_id, user_id=user_id, runs=None)

    async def fake_disconnect_mcp_tools(agent):
        return None

    monkeypatch.setattr(_storage, "aread_or_create_session", fake_aread_or_create_session)
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)
    monkeypatch.setattr(_init, "disconnect_connectable_tools", lambda agent: None)
    monkeypatch.setattr(_init, "disconnect_mcp_tools", fake_disconnect_mcp_tools)

    run_context = RunContext(
        run_id=run_id,
        session_id="session-1",
        user_id=None,
        session_state={},
    )

    # The streaming path raises for the same reason: persisting a terminal ERROR
    # run for a run_id that was never found corrupts the target row. The HTTP layer
    # turns this into a RunError SSE event, so the wire contract is unchanged.
    events = []
    with pytest.raises(RunNotFoundError, match="No runs found for run ID missing-stream-run"):
        async for event in _run._acontinue_run_stream(
            agent=agent,
            session_id="session-1",
            run_context=run_context,
            run_id=run_id,
            requirements=[],
        ):
            events.append(event)

    assert events == []


@pytest.mark.asyncio
async def test_ahandle_model_response_stream_preserves_partial_assistant_message_on_cancellation():
    agent = Agent(name="test-agent")

    class FakeModel:
        id = "fake-model"
        provider = "fake-provider"

        async def aresponse_stream(self, **kwargs):
            yield ModelResponse(content="partial content")
            raise RunCancelledException("cancelled")

    agent.model = FakeModel()
    session = AgentSession(session_id="session-1")
    run_messages = RunMessages(messages=[Message(role="user", content="hello")])
    run_response = RunOutput(
        run_id="run-1",
        session_id="session-1",
        agent_id=agent.id,
        agent_name=agent.name,
    )

    with pytest.raises(RunCancelledException, match="cancelled"):
        async for _ in _response.ahandle_model_response_stream(
            agent=agent,
            session=session,
            run_response=run_response,
            run_messages=run_messages,
        ):
            pass

    assert run_response.messages is not None
    assert len(run_response.messages) == 2
    assert run_response.messages[0].role == "user"
    assert run_response.messages[1].role == "assistant"
    assert run_response.messages[1].content == "partial content"


@pytest.mark.asyncio
async def test_arun_stream_impl_cleans_up_registered_run_on_session_read_failure(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    run_id = "arun-stream-session-fail"

    async def fail_aread_or_create_session(agent, session_id: str, user_id: Optional[str] = None):
        raise RuntimeError("session read failed")

    async def fake_disconnect_mcp_tools(agent):
        return None

    monkeypatch.setattr(_storage, "aread_or_create_session", fail_aread_or_create_session)
    monkeypatch.setattr(_init, "disconnect_connectable_tools", lambda agent: None)
    monkeypatch.setattr(_init, "disconnect_mcp_tools", fake_disconnect_mcp_tools)

    run_context = RunContext(run_id=run_id, session_id="session-1", session_state={})
    run_response = RunOutput(run_id=run_id)

    response_stream = _run._arun_stream(
        agent=agent,
        run_response=run_response,
        run_context=run_context,
        session_id="session-1",
    )

    # Consume the error event yielded by the stream
    events = []
    async for event in response_stream:
        events.append(event)

    # Verify an error event was yielded with the session read failure
    assert len(events) == 1
    assert isinstance(events[0], RunErrorEvent)
    assert "session read failed" in events[0].content

    assert run_id not in get_active_runs()


@pytest.mark.asyncio
async def test_arun_impl_preserves_original_error_when_session_read_fails(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    run_id = "arun-session-fail"
    cleanup_calls = []

    async def fail_aread_or_create_session(agent, session_id: str, user_id: Optional[str] = None):
        raise RuntimeError("session read failed")

    async def fake_acleanup_and_store(agent, **kwargs: Any):
        cleanup_calls.append(kwargs)
        return None

    async def fake_disconnect_mcp_tools(agent):
        return None

    monkeypatch.setattr(_storage, "aread_or_create_session", fail_aread_or_create_session)
    monkeypatch.setattr(_run, "acleanup_and_store", fake_acleanup_and_store)
    monkeypatch.setattr(_init, "disconnect_connectable_tools", lambda agent: None)
    monkeypatch.setattr(_init, "disconnect_mcp_tools", fake_disconnect_mcp_tools)

    run_context = RunContext(run_id=run_id, session_id="session-1", session_state={})
    run_response = RunOutput(run_id=run_id)

    response = await _run._arun(
        agent=agent,
        run_response=run_response,
        run_context=run_context,
        session_id="session-1",
    )

    assert response.status == RunStatus.error
    assert response.content == "session read failed"
    assert cleanup_calls == []
    assert run_id not in get_active_runs()


@pytest.mark.asyncio
async def test_acontinue_run_preserves_original_error_when_session_read_fails(monkeypatch: pytest.MonkeyPatch):
    agent = Agent(name="test-agent")
    run_id = "acontinue-session-fail"
    cleanup_calls = []

    async def fail_aread_or_create_session(agent, session_id: str, user_id: Optional[str] = None):
        raise RuntimeError("session read failed")

    async def fake_acleanup_and_store(agent, **kwargs: Any):
        cleanup_calls.append(kwargs)
        return None

    async def fake_disconnect_mcp_tools(agent):
        return None

    monkeypatch.setattr(_storage, "aread_or_create_session", fail_aread_or_create_session)
    monkeypatch.setattr(_run, "acleanup_and_store", fake_acleanup_and_store)
    monkeypatch.setattr(_init, "disconnect_connectable_tools", lambda agent: None)
    monkeypatch.setattr(_init, "disconnect_mcp_tools", fake_disconnect_mcp_tools)

    run_context = RunContext(run_id=run_id, session_id="session-1", session_state={})

    response = await _run._acontinue_run(
        agent=agent,
        session_id="session-1",
        run_context=run_context,
        run_id=run_id,
        requirements=[],
    )

    assert response.status == RunStatus.error
    assert response.content == "session read failed"
    assert cleanup_calls == []
    assert run_id not in get_active_runs()


def test_continue_run_stream_registers_run_for_cancellation():
    agent = Agent(name="test-agent")
    run_id = "continue-stream-register"

    run_response = RunOutput(run_id=run_id)
    run_messages = RunMessages(messages=[])
    run_context = RunContext(run_id=run_id, session_id="session-1", session_state={})
    session = AgentSession(session_id="session-1")

    response_stream = _run._continue_run_stream(
        agent=agent,
        run_response=run_response,
        run_messages=run_messages,
        run_context=run_context,
        session=session,
        tools=[],
        stream_events=True,
    )

    next(response_stream)

    assert run_id in get_active_runs()
    assert cancel_run(run_id) is True

    response_stream.close()
    assert run_id not in get_active_runs()


def test_session_read_wrappers_default_to_agent_session_type():
    read_default = inspect.signature(_storage.read_session).parameters["session_type"].default
    aread_default = inspect.signature(_storage.aread_session).parameters["session_type"].default

    assert read_default == SessionType.AGENT
    assert aread_default == SessionType.AGENT


def _make_precedence_test_agent() -> Agent:
    return Agent(
        name="precedence-agent",
        dependencies={"agent_dep": "default"},
        knowledge_filters={"agent_filter": "default"},
        metadata={"agent_meta": "default"},
        output_schema={"type": "object", "properties": {"agent": {"type": "string"}}},
    )


def _patch_continue_dispatch_dependencies(agent: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_init, "has_async_db", lambda agent: False)
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)
    monkeypatch.setattr(
        _storage,
        "read_or_create_session",
        lambda agent, session_id=None, user_id=None: AgentSession(session_id=session_id, user_id=user_id, runs=[]),
    )
    monkeypatch.setattr(_init, "set_default_model", lambda agent: None)
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: None)
    monkeypatch.setattr(agent, "get_tools", lambda **kwargs: [])
    monkeypatch.setattr(_tools, "determine_tools_for_model", lambda agent, **kwargs: [])
    monkeypatch.setattr(
        _messages, "get_continue_run_messages", lambda agent, input=None, **kwargs: RunMessages(messages=[])
    )


def test_run_dispatch_respects_run_context_precedence(monkeypatch: pytest.MonkeyPatch):
    agent = _make_precedence_test_agent()
    _patch_sync_dispatch_dependencies(agent, monkeypatch, runs=[])
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)

    def fake_run_impl(
        agent: Agent,
        run_response,
        run_context,
        session_id: str = "",
        user_id: Optional[str] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        response_format: Optional[Any] = None,
        debug_mode: Optional[bool] = None,
        background_tasks: Optional[Any] = None,
        **kwargs: Any,
    ):
        cleanup_run(run_response.run_id)  # type: ignore[arg-type]
        return run_response

    monkeypatch.setattr(_run, "_run", fake_run_impl)

    preserved_context = RunContext(
        run_id="ctx-preserve",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
        output_schema={"ctx_schema": "keep"},
    )
    _run.run_dispatch(
        agent=agent,
        input="hello",
        run_id="run-preserve",
        stream=False,
        run_context=preserved_context,
    )
    assert preserved_context.dependencies == {"ctx_dep": "keep"}
    assert preserved_context.knowledge_filters == {"ctx_filter": "keep"}
    assert preserved_context.metadata == {"ctx_meta": "keep"}
    # output_schema is always set from resolved options (for workflow reuse)
    assert preserved_context.output_schema == {"type": "object", "properties": {"agent": {"type": "string"}}}

    override_context = RunContext(
        run_id="ctx-override",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
        output_schema={"ctx_schema": "keep"},
    )
    _run.run_dispatch(
        agent=agent,
        input="hello",
        run_id="run-override",
        stream=False,
        run_context=override_context,
        dependencies={"call_dep": "override"},
        knowledge_filters={"call_filter": "override"},
        metadata={"call_meta": "override"},
        output_schema={"call_schema": "override"},
    )
    assert override_context.dependencies == {"agent_dep": "default", "call_dep": "override"}
    assert override_context.knowledge_filters == {"agent_filter": "default", "call_filter": "override"}
    assert override_context.metadata == {"call_meta": "override", "agent_meta": "default"}
    assert override_context.output_schema == {"call_schema": "override"}

    empty_context = RunContext(
        run_id="ctx-empty",
        session_id="session-1",
        session_state={},
        dependencies=None,
        knowledge_filters=None,
        metadata=None,
        output_schema=None,
    )
    _run.run_dispatch(
        agent=agent,
        input="hello",
        run_id="run-empty",
        stream=False,
        run_context=empty_context,
    )
    assert empty_context.dependencies == {"agent_dep": "default"}
    assert empty_context.knowledge_filters == {"agent_filter": "default"}
    assert empty_context.metadata == {"agent_meta": "default"}
    assert empty_context.output_schema == {"type": "object", "properties": {"agent": {"type": "string"}}}


@pytest.mark.asyncio
async def test_arun_dispatch_respects_run_context_precedence(monkeypatch: pytest.MonkeyPatch):
    agent = _make_precedence_test_agent()
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: None)

    async def fake_arun_impl(
        agent: Agent,
        run_response,
        run_context,
        user_id: Optional[str] = None,
        response_format: Optional[Any] = None,
        session_id: Optional[str] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        background_tasks: Optional[Any] = None,
        **kwargs: Any,
    ):
        return run_response

    monkeypatch.setattr(_run, "_arun", fake_arun_impl)

    preserved_context = RunContext(
        run_id="actx-preserve",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
        output_schema={"ctx_schema": "keep"},
    )
    await _run.arun_dispatch(
        agent=agent,
        input="hello",
        run_id="arun-preserve",
        stream=False,
        run_context=preserved_context,
    )
    assert preserved_context.dependencies == {"ctx_dep": "keep"}
    assert preserved_context.knowledge_filters == {"ctx_filter": "keep"}
    assert preserved_context.metadata == {"ctx_meta": "keep"}
    # output_schema is always set from resolved options (for workflow reuse)
    assert preserved_context.output_schema == {"type": "object", "properties": {"agent": {"type": "string"}}}

    override_context = RunContext(
        run_id="actx-override",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
        output_schema={"ctx_schema": "keep"},
    )
    await _run.arun_dispatch(
        agent=agent,
        input="hello",
        run_id="arun-override",
        stream=False,
        run_context=override_context,
        dependencies={"call_dep": "override"},
        knowledge_filters={"call_filter": "override"},
        metadata={"call_meta": "override"},
        output_schema={"call_schema": "override"},
    )
    assert override_context.dependencies == {"agent_dep": "default", "call_dep": "override"}
    assert override_context.knowledge_filters == {"agent_filter": "default", "call_filter": "override"}
    assert override_context.metadata == {"call_meta": "override", "agent_meta": "default"}
    assert override_context.output_schema == {"call_schema": "override"}

    empty_context = RunContext(
        run_id="actx-empty",
        session_id="session-1",
        session_state={},
        dependencies=None,
        knowledge_filters=None,
        metadata=None,
        output_schema=None,
    )
    await _run.arun_dispatch(
        agent=agent,
        input="hello",
        run_id="arun-empty",
        stream=False,
        run_context=empty_context,
    )
    assert empty_context.dependencies == {"agent_dep": "default"}
    assert empty_context.knowledge_filters == {"agent_filter": "default"}
    assert empty_context.metadata == {"agent_meta": "default"}
    assert empty_context.output_schema == {"type": "object", "properties": {"agent": {"type": "string"}}}


def test_continue_run_dispatch_respects_run_context_precedence(monkeypatch: pytest.MonkeyPatch):
    agent = _make_precedence_test_agent()
    _patch_continue_dispatch_dependencies(agent, monkeypatch)

    def fake_continue_run(
        agent: Agent,
        run_response: RunOutput,
        run_messages: RunMessages,
        run_context: RunContext,
        session: AgentSession,
        tools,
        user_id: Optional[str] = None,
        response_format: Optional[Any] = None,
        debug_mode: Optional[bool] = None,
        background_tasks: Optional[Any] = None,
        **kwargs: Any,
    ) -> RunOutput:
        return run_response

    monkeypatch.setattr(_run, "_continue_run", fake_continue_run)

    preserved_context = RunContext(
        run_id="continue-preserve",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
    )
    _run.continue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="continue-run-1", session_id="session-1", messages=[]),
        stream=False,
        run_context=preserved_context,
    )
    assert preserved_context.dependencies == {"ctx_dep": "keep"}
    assert preserved_context.knowledge_filters == {"ctx_filter": "keep"}
    assert preserved_context.metadata == {"ctx_meta": "keep"}

    override_context = RunContext(
        run_id="continue-override",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
    )
    _run.continue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="continue-run-2", session_id="session-1", messages=[]),
        stream=False,
        run_context=override_context,
        dependencies={"call_dep": "override"},
        knowledge_filters={"call_filter": "override"},
        metadata={"call_meta": "override"},
    )
    assert override_context.dependencies == {"agent_dep": "default", "call_dep": "override"}
    assert override_context.knowledge_filters == {"agent_filter": "default", "call_filter": "override"}
    assert override_context.metadata == {"call_meta": "override", "agent_meta": "default"}

    empty_context = RunContext(
        run_id="continue-empty",
        session_id="session-1",
        session_state={},
        dependencies=None,
        knowledge_filters=None,
        metadata=None,
    )
    _run.continue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="continue-run-3", session_id="session-1", messages=[]),
        stream=False,
        run_context=empty_context,
    )
    assert empty_context.dependencies == {"agent_dep": "default"}
    assert empty_context.knowledge_filters == {"agent_filter": "default"}
    assert empty_context.metadata == {"agent_meta": "default"}


def test_continue_run_dispatch_applies_admin_approval_for_provided_run_response(monkeypatch: pytest.MonkeyPatch):
    agent = _make_precedence_test_agent()
    _patch_continue_dispatch_dependencies(agent, monkeypatch)
    tool_execution = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        approval_type="required",
        requires_confirmation=True,
    )
    run_response = RunOutput(run_id="approved-run", session_id="session-1", messages=[], tools=[tool_execution])
    applied: dict[str, bool] = {}

    def fake_check_and_apply(db, run_id, run_response):
        applied["called"] = True
        tool_execution.confirmed = True

    def fake_continue_run(
        agent,
        run_response,
        run_messages,
        run_context,
        session,
        tools,
        **kwargs,
    ):
        assert run_messages is not None
        assert run_context is not None
        assert session is not None
        assert tools == []
        return run_response

    monkeypatch.setattr(_run, "check_and_apply_approval_resolution", fake_check_and_apply)
    monkeypatch.setattr(_run, "_continue_run", fake_continue_run)

    result = _run.continue_run_dispatch(agent=agent, run_response=run_response, stream=False)

    assert result is run_response
    assert applied["called"] is True
    assert tool_execution.confirmed is True


def test_apply_tool_resolution_payload_merges_partial_requirements():
    first = RunRequirement(
        ToolExecution(
            tool_call_id="call-1",
            tool_name="first",
            requires_confirmation=True,
        )
    )
    second = RunRequirement(
        ToolExecution(
            tool_call_id="call-2",
            tool_name="second",
            requires_confirmation=True,
        )
    )
    resolved_first = RunRequirement(
        ToolExecution(
            tool_call_id="call-1",
            tool_name="first",
            requires_confirmation=True,
        )
    )
    resolved_first.id = first.id
    resolved_first.confirm(metadata={"approver": "admin"})
    run_response = RunOutput(
        run_id="partial-run",
        session_id="session-1",
        messages=[],
        requirements=[first, second],
        tools=[first.tool_execution, second.tool_execution],
    )

    _run._apply_tool_resolution_payload(run_response, requirements=[resolved_first])

    assert run_response.requirements == [first, second]
    assert run_response.requirements[0].is_resolved()
    assert not run_response.requirements[1].is_resolved()
    assert run_response.tools == [first.tool_execution, second.tool_execution]


def test_apply_tool_resolution_payload_syncs_deprecated_updated_tools_to_requirements():
    paused_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        requires_confirmation=True,
        approval_type="required",
    )
    requirement = RunRequirement(paused_tool)
    updated_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        requires_confirmation=True,
        confirmed=True,
        approval_type="required",
        resume_metadata={"approver": "admin"},
    )
    run_response = RunOutput(
        run_id="legacy-tools-run",
        session_id="session-1",
        messages=[],
        tools=[paused_tool],
        requirements=[requirement],
    )

    _run._apply_tool_resolution_payload(run_response, updated_tools=[updated_tool])

    assert run_response.tools == [updated_tool]
    assert run_response.requirements == [requirement]
    assert requirement.tool_execution is updated_tool
    assert requirement.is_resolved()
    assert requirement.confirmation is True
    assert requirement.approval_metadata == {"approver": "admin"}


def test_apply_tool_resolution_payload_syncs_serialized_updated_tools_to_requirements():
    paused_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="external",
        external_execution_required=True,
    )
    requirement = RunRequirement.from_dict(RunRequirement(paused_tool).to_dict())
    updated_tool = ToolExecution.from_dict(
        {
            "tool_call_id": "call-1",
            "tool_name": "external",
            "external_execution_required": True,
            "external_execution_result_provided": True,
            "result": {"ok": True},
        }
    )
    run_response = RunOutput(
        run_id="serialized-legacy-tools-run",
        session_id="session-1",
        messages=[],
        tools=[paused_tool],
        requirements=[requirement],
    )

    _run._apply_tool_resolution_payload(run_response, updated_tools=[updated_tool])

    assert run_response.requirements == [requirement]
    assert requirement.is_resolved()
    assert requirement.external_execution_result == {"ok": True}
    assert requirement.tool_execution.external_execution_result_provided is True


def test_deprecated_updated_tools_partial_update_preserves_unresolved_tools():
    first_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="first",
        requires_confirmation=True,
        approval_type="required",
    )
    second_tool = ToolExecution(
        tool_call_id="call-2",
        tool_name="second",
        requires_confirmation=True,
        approval_type="required",
    )
    first_requirement = RunRequirement(first_tool)
    second_requirement = RunRequirement(second_tool)
    resolved_first = ToolExecution(
        tool_call_id="call-1",
        tool_name="first",
        requires_confirmation=True,
        approval_type="required",
        confirmed=True,
    )
    run_response = RunOutput(
        run_id="partial-legacy-tools-run",
        session_id="session-1",
        messages=[],
        tools=[first_tool, second_tool],
        requirements=[first_requirement, second_requirement],
    )

    _run._apply_tool_resolution_payload(run_response, updated_tools=[resolved_first])

    assert run_response.tools == [resolved_first, second_tool]
    assert first_requirement.tool_execution is resolved_first
    assert second_requirement.tool_execution is second_tool
    assert first_requirement.is_resolved()
    assert not second_requirement.is_resolved()


def test_deprecated_updated_tools_partial_update_preserves_tool_context_without_requirements():
    paused_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        tool_args={"path": ".env"},
        requires_confirmation=True,
        approval_type="required",
    )
    partial_resolution = ToolExecution(tool_call_id="call-1", confirmed=True)
    run_response = RunOutput(run_id="legacy-partial-run", session_id="session-1", messages=[], tools=[paused_tool])

    _run._apply_tool_resolution_payload(run_response, updated_tools=[partial_resolution])

    assert run_response.tools == [partial_resolution]
    assert partial_resolution.tool_name == "protected"
    assert partial_resolution.tool_args == {"path": ".env"}
    assert partial_resolution.requires_confirmation is True
    assert partial_resolution.confirmed is True


def test_requirements_payload_preserves_tool_context_without_existing_requirements():
    paused_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        tool_args={"path": ".env"},
        requires_confirmation=True,
        approval_type="required",
    )
    partial_tool = ToolExecution(tool_call_id="call-1", confirmed=True)
    partial_requirement = RunRequirement(partial_tool)
    run_response = RunOutput(run_id="partial-requirement-run", session_id="session-1", messages=[], tools=[paused_tool])

    _run._apply_tool_resolution_payload(run_response, requirements=[partial_requirement])

    assert run_response.tools == [paused_tool]
    assert run_response.requirements is not None
    assert run_response.requirements[0].tool_execution is paused_tool
    assert paused_tool.tool_name == "protected"
    assert paused_tool.tool_args == {"path": ".env"}
    assert paused_tool.requires_confirmation is True
    assert paused_tool.confirmed is True


def test_apply_tool_resolution_payload_prefers_latest_unresolved_requirement_for_same_tool_call_id():
    old_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        requires_confirmation=True,
        approval_type="required",
        confirmed=True,
    )
    old_requirement = RunRequirement(old_tool)
    old_requirement.confirmation = True
    new_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        requires_confirmation=True,
        approval_type="required",
    )
    new_requirement = RunRequirement(new_tool)
    resolved_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        requires_confirmation=True,
        approval_type="required",
        confirmed=True,
        resume_metadata={"round": 2},
    )
    payload_requirement = RunRequirement(resolved_tool)
    payload_requirement.id = "client-lost-the-active-id"
    run_response = RunOutput(
        run_id="same-call-id-run",
        session_id="session-1",
        messages=[],
        tools=[old_tool, new_tool],
        requirements=[old_requirement, new_requirement],
    )

    _run._apply_tool_resolution_payload(run_response, requirements=[payload_requirement])

    assert old_requirement.approval_metadata is None
    assert new_requirement.is_resolved()
    assert new_requirement.approval_metadata == {"round": 2}


def test_requirements_sync_keeps_active_tool_for_repeated_pause_same_tool_call_id():
    old_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        requires_confirmation=True,
        approval_type="required",
        confirmed=True,
    )
    old_requirement = RunRequirement(old_tool)
    old_requirement.confirmation = True
    new_tool = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        tool_args={"round": 2},
        requires_confirmation=True,
        approval_type="required",
    )
    new_requirement = RunRequirement(new_tool)
    payload_requirement = RunRequirement(ToolExecution(tool_call_id="call-1", confirmed=True))
    run_response = RunOutput(
        run_id="same-call-id-run",
        session_id="session-1",
        messages=[],
        tools=[new_tool],
        requirements=[old_requirement, new_requirement],
    )

    _run._apply_tool_resolution_payload(run_response, requirements=[payload_requirement])

    assert run_response.tools == [new_tool]
    assert run_response.tools[0].tool_args == {"round": 2}
    assert new_requirement.tool_execution is new_tool
    assert new_tool.confirmed is True


def test_continue_run_dispatch_stores_run_continued_event(monkeypatch: pytest.MonkeyPatch):
    agent = _make_precedence_test_agent()
    _patch_continue_dispatch_dependencies(agent, monkeypatch)
    monkeypatch.setattr(_init, "disconnect_connectable_tools", lambda agent: None)
    monkeypatch.setattr(_run, "register_run", lambda run_id: None)
    monkeypatch.setattr(_run, "cleanup_run", lambda run_id: None)
    monkeypatch.setattr(_tools, "handle_tool_call_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr(_run, "call_model_with_fallback", lambda *args, **kwargs: ModelResponse(content="done"))
    monkeypatch.setattr(_response, "generate_response_with_output_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(_response, "parse_response_with_parser_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(_response, "update_run_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(_response, "convert_response_to_structured_format", lambda *args, **kwargs: None)
    monkeypatch.setattr(_run, "store_media_util", lambda *args, **kwargs: None)
    monkeypatch.setattr(_response, "generate_followups", lambda *args, **kwargs: None)
    monkeypatch.setattr(_telemetry, "log_agent_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(_run, "cleanup_and_store", lambda *args, **kwargs: None)
    agent.store_events = True
    requirement = RunRequirement(
        ToolExecution(
            tool_call_id="call-1",
            tool_name="protected",
            approval_type="required",
            requires_confirmation=True,
        )
    )
    requirement.confirm()
    run_response = RunOutput(run_id="continued-run", session_id="session-1", messages=[], requirements=[requirement])

    result = _run.continue_run_dispatch(
        agent=agent, run_response=run_response, stream=False, requirements=[requirement]
    )

    assert result is run_response
    assert any(getattr(event, "event", None) == "RunContinued" for event in run_response.events or [])


@pytest.mark.asyncio
async def test_acontinue_run_dispatch_respects_run_context_precedence(monkeypatch: pytest.MonkeyPatch):
    agent = _make_precedence_test_agent()
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: None)

    async def fake_acontinue_run(
        agent: Agent,
        session_id: str,
        run_context: RunContext,
        run_response: Optional[RunOutput] = None,
        requirements=None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Any] = None,
        debug_mode: Optional[bool] = None,
        background_tasks: Optional[Any] = None,
        **kwargs: Any,
    ) -> RunOutput:
        return run_response if run_response is not None else RunOutput(run_id=run_id, session_id=session_id)  # type: ignore[arg-type]

    monkeypatch.setattr(_run, "_acontinue_run", fake_acontinue_run)

    preserved_context = RunContext(
        run_id="acontinue-preserve",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
    )
    await _run.acontinue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="acontinue-run-1", session_id="session-1", messages=[]),
        stream=False,
        run_context=preserved_context,
    )
    assert preserved_context.dependencies == {"ctx_dep": "keep"}
    assert preserved_context.knowledge_filters == {"ctx_filter": "keep"}
    assert preserved_context.metadata == {"ctx_meta": "keep"}

    override_context = RunContext(
        run_id="acontinue-override",
        session_id="session-1",
        session_state={},
        dependencies={"ctx_dep": "keep"},
        knowledge_filters={"ctx_filter": "keep"},
        metadata={"ctx_meta": "keep"},
    )
    await _run.acontinue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="acontinue-run-2", session_id="session-1", messages=[]),
        stream=False,
        run_context=override_context,
        dependencies={"call_dep": "override"},
        knowledge_filters={"call_filter": "override"},
        metadata={"call_meta": "override"},
    )
    assert override_context.dependencies == {"agent_dep": "default", "call_dep": "override"}
    assert override_context.knowledge_filters == {"agent_filter": "default", "call_filter": "override"}
    assert override_context.metadata == {"call_meta": "override", "agent_meta": "default"}

    empty_context = RunContext(
        run_id="acontinue-empty",
        session_id="session-1",
        session_state={},
        dependencies=None,
        knowledge_filters=None,
        metadata=None,
    )
    await _run.acontinue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="acontinue-run-3", session_id="session-1", messages=[]),
        stream=False,
        run_context=empty_context,
    )
    assert empty_context.dependencies == {"agent_dep": "default"}
    assert empty_context.knowledge_filters == {"agent_filter": "default"}
    assert empty_context.metadata == {"agent_meta": "default"}


@pytest.mark.asyncio
async def test_acontinue_run_dispatch_applies_admin_approval_for_provided_run_response(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = _make_precedence_test_agent()
    tool_execution = ToolExecution(
        tool_call_id="call-1",
        tool_name="protected",
        approval_type="required",
        requires_confirmation=True,
    )
    run_response = RunOutput(run_id="approved-run", session_id="session-1", messages=[], tools=[tool_execution])
    applied: dict[str, bool] = {}

    async def fake_check_and_apply(db, run_id, run_response):
        applied["called"] = True
        tool_execution.confirmed = True

    async def fake_aread_or_create_session(agent, session_id, user_id=None):
        return AgentSession(session_id=session_id, user_id=user_id, runs=[])

    async def fake_acall_model_with_fallback(*args, **kwargs):
        return ModelResponse(content="done")

    async def noop_async(*args, **kwargs):
        return None

    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)
    monkeypatch.setattr(agent, "aget_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: None)
    monkeypatch.setattr(_storage, "aread_or_create_session", fake_aread_or_create_session)
    monkeypatch.setattr(_storage, "load_session_state", lambda agent, session=None, session_state=None: session_state)
    monkeypatch.setattr(_storage, "update_metadata", lambda agent, session=None: None)
    monkeypatch.setattr(_tools, "determine_tools_for_model", lambda agent, **kwargs: [])
    monkeypatch.setattr(
        _messages, "get_continue_run_messages", lambda agent, input=None, **kwargs: RunMessages(messages=[])
    )
    monkeypatch.setattr(_run, "acheck_and_apply_approval_resolution", fake_check_and_apply)
    monkeypatch.setattr(_run, "aregister_run", noop_async)
    monkeypatch.setattr(_tools, "ahandle_tool_call_updates", noop_async)
    monkeypatch.setattr(_run, "acall_model_with_fallback", fake_acall_model_with_fallback)
    monkeypatch.setattr(_run, "araise_if_cancelled", noop_async)
    monkeypatch.setattr(_response, "agenerate_response_with_output_model", noop_async)
    monkeypatch.setattr(_response, "aparse_response_with_parser_model", noop_async)
    monkeypatch.setattr(
        _response,
        "update_run_response",
        lambda agent, model_response, run_response, run_messages, run_context: setattr(
            run_response, "content", model_response.content
        ),
    )
    monkeypatch.setattr(_response, "convert_response_to_structured_format", lambda *args, **kwargs: None)
    monkeypatch.setattr(_response, "agenerate_followups", noop_async)
    monkeypatch.setattr(_run, "store_media_util", lambda *args, **kwargs: None)
    monkeypatch.setattr(_run, "acleanup_and_store", noop_async)
    monkeypatch.setattr(_telemetry, "alog_agent_telemetry", noop_async)
    monkeypatch.setattr(_init, "disconnect_connectable_tools", lambda agent: None)
    monkeypatch.setattr(_init, "disconnect_mcp_tools", noop_async)
    monkeypatch.setattr(_run, "acleanup_run", noop_async)

    result = await _run.acontinue_run_dispatch(agent=agent, run_response=run_response, stream=False)

    assert result is run_response
    assert applied["called"] is True
    assert tool_execution.confirmed is True


def test_all_pause_handlers_accept_run_context():
    for fn in [
        _run.handle_agent_run_paused,
        _run.handle_agent_run_paused_stream,
        _run.ahandle_agent_run_paused,
        _run.ahandle_agent_run_paused_stream,
    ]:
        params = inspect.signature(fn).parameters
        assert "run_context" in params, f"{fn.__name__} missing run_context param"


def test_handle_agent_run_paused_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def spy_cleanup_and_store(agent, run_response, session, run_context=None, user_id=None):
        captured["run_context"] = run_context

    monkeypatch.setattr(_run, "cleanup_and_store", spy_cleanup_and_store)
    monkeypatch.setattr(_run, "create_approval_from_pause", lambda **kwargs: None)

    agent = Agent(name="test-hitl")
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    _run.handle_agent_run_paused(
        agent=agent,
        run_response=RunOutput(run_id="r1", session_id="s1", messages=[]),
        session=AgentSession(session_id="s1"),
        user_id="u1",
        run_context=run_context,
    )

    assert captured["run_context"] is run_context


def test_handle_agent_run_paused_stores_run_paused_event(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_run, "cleanup_and_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(_run, "create_approval_from_pause", lambda **kwargs: None)

    agent = Agent(name="test-hitl")
    agent.store_events = True
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])

    _run.handle_agent_run_paused(
        agent=agent,
        run_response=run_response,
        session=AgentSession(session_id="s1"),
    )

    assert run_response.events is not None
    assert any(getattr(event, "event", None) == "RunPaused" for event in run_response.events)


@pytest.mark.asyncio
async def test_ahandle_agent_run_paused_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def spy_acleanup_and_store(agent, run_response, session, run_context=None, user_id=None):
        captured["run_context"] = run_context

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(_run, "acleanup_and_store", spy_acleanup_and_store)
    monkeypatch.setattr(_run, "acreate_approval_from_pause", noop_acreate_approval)

    agent = Agent(name="test-hitl-async")
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    await _run.ahandle_agent_run_paused(
        agent=agent,
        run_response=RunOutput(run_id="r1", session_id="s1", messages=[]),
        session=AgentSession(session_id="s1"),
        user_id="u1",
        run_context=run_context,
    )

    assert captured["run_context"] is run_context


@pytest.mark.asyncio
async def test_ahandle_agent_run_paused_stores_run_paused_event(monkeypatch: pytest.MonkeyPatch):
    async def noop_acleanup_and_store(*args, **kwargs):
        return None

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(_run, "acleanup_and_store", noop_acleanup_and_store)
    monkeypatch.setattr(_run, "acreate_approval_from_pause", noop_acreate_approval)

    agent = Agent(name="test-hitl-async")
    agent.store_events = True
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])

    await _run.ahandle_agent_run_paused(
        agent=agent,
        run_response=run_response,
        session=AgentSession(session_id="s1"),
    )

    assert run_response.events is not None
    assert any(getattr(event, "event", None) == "RunPaused" for event in run_response.events)


def test_handle_agent_run_paused_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_session, "save_session", lambda agent, session: None)
    monkeypatch.setattr(_run, "create_approval_from_pause", lambda **kwargs: None)
    monkeypatch.setattr(_run, "scrub_run_output_for_storage", lambda agent, run_response: None)
    monkeypatch.setattr(_run, "save_run_response_to_file", lambda agent, **kwargs: None)
    monkeypatch.setattr(_run, "update_session_metrics", lambda agent, session, run_response: None)

    agent = Agent(name="test-hitl")
    session = AgentSession(session_id="s1", session_data={})
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"watchlist": ["AAPL"]})

    result = _run.handle_agent_run_paused(
        agent=agent,
        run_response=run_response,
        session=session,
        user_id="u1",
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert session.session_data["session_state"] == {"watchlist": ["AAPL"]}
    assert result.session_state == {"watchlist": ["AAPL"]}


def test_handle_agent_run_paused_without_run_context_does_not_set_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_session, "save_session", lambda agent, session: None)
    monkeypatch.setattr(_run, "create_approval_from_pause", lambda **kwargs: None)
    monkeypatch.setattr(_run, "scrub_run_output_for_storage", lambda agent, run_response: None)
    monkeypatch.setattr(_run, "save_run_response_to_file", lambda agent, **kwargs: None)
    monkeypatch.setattr(_run, "update_session_metrics", lambda agent, session, run_response: None)

    agent = Agent(name="test-hitl")
    session = AgentSession(session_id="s1", session_data={})

    result = _run.handle_agent_run_paused(
        agent=agent,
        run_response=RunOutput(run_id="r1", session_id="s1", messages=[]),
        session=session,
        user_id="u1",
    )

    assert result.status == RunStatus.paused
    assert "session_state" not in session.session_data


def test_handle_agent_run_paused_persists_state_when_session_data_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_session, "save_session", lambda agent, session: None)
    monkeypatch.setattr(_run, "create_approval_from_pause", lambda **kwargs: None)
    monkeypatch.setattr(_run, "scrub_run_output_for_storage", lambda agent, run_response: None)
    monkeypatch.setattr(_run, "save_run_response_to_file", lambda agent, **kwargs: None)
    monkeypatch.setattr(_run, "update_session_metrics", lambda agent, session, run_response: None)

    agent = Agent(name="test-hitl")
    session = AgentSession(session_id="s1", session_data=None)
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"watchlist": ["AAPL"]})

    result = _run.handle_agent_run_paused(
        agent=agent,
        run_response=run_response,
        session=session,
        user_id="u1",
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert result.session_state == {"watchlist": ["AAPL"]}
    assert session.session_data == {"session_state": {"watchlist": ["AAPL"]}}


@pytest.mark.asyncio
async def test_ahandle_agent_run_paused_persists_state_when_session_data_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_session, "save_session", lambda agent, session: None)
    monkeypatch.setattr(_run, "scrub_run_output_for_storage", lambda agent, run_response: None)
    monkeypatch.setattr(_run, "save_run_response_to_file", lambda agent, **kwargs: None)
    monkeypatch.setattr(_run, "update_session_metrics", lambda agent, session, run_response: None)

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(_run, "acreate_approval_from_pause", noop_acreate_approval)

    agent = Agent(name="test-hitl-async")
    session = AgentSession(session_id="s1", session_data=None)
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"cart": ["item-1"]})

    result = await _run.ahandle_agent_run_paused(
        agent=agent,
        run_response=run_response,
        session=session,
        user_id="u1",
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert result.session_state == {"cart": ["item-1"]}
    assert session.session_data == {"session_state": {"cart": ["item-1"]}}


@pytest.mark.asyncio
async def test_ahandle_agent_run_paused_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_session, "save_session", lambda agent, session: None)
    monkeypatch.setattr(_run, "scrub_run_output_for_storage", lambda agent, run_response: None)
    monkeypatch.setattr(_run, "save_run_response_to_file", lambda agent, **kwargs: None)
    monkeypatch.setattr(_run, "update_session_metrics", lambda agent, session, run_response: None)

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(_run, "acreate_approval_from_pause", noop_acreate_approval)

    agent = Agent(name="test-hitl-async")
    session = AgentSession(session_id="s1", session_data={})
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"cart": ["item-1"]})

    result = await _run.ahandle_agent_run_paused(
        agent=agent,
        run_response=run_response,
        session=session,
        user_id="u1",
        run_context=run_context,
    )

    assert result.status == RunStatus.paused
    assert session.session_data["session_state"] == {"cart": ["item-1"]}
    assert result.session_state == {"cart": ["item-1"]}


def test_handle_agent_run_paused_stream_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def spy_cleanup_and_store(agent, run_response, session, run_context=None, user_id=None):
        captured["run_context"] = run_context

    monkeypatch.setattr(_run, "cleanup_and_store", spy_cleanup_and_store)
    monkeypatch.setattr(_run, "create_approval_from_pause", lambda **kwargs: None)

    agent = Agent(name="test-hitl-stream")
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    events = list(
        _run.handle_agent_run_paused_stream(
            agent=agent,
            run_response=RunOutput(run_id="r1", session_id="s1", messages=[]),
            session=AgentSession(session_id="s1"),
            user_id="u1",
            run_context=run_context,
        )
    )

    assert captured["run_context"] is run_context
    assert len(events) >= 1


@pytest.mark.asyncio
async def test_ahandle_agent_run_paused_stream_forwards_run_context_to_cleanup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def spy_acleanup_and_store(agent, run_response, session, run_context=None, user_id=None):
        captured["run_context"] = run_context

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(_run, "acleanup_and_store", spy_acleanup_and_store)
    monkeypatch.setattr(_run, "acreate_approval_from_pause", noop_acreate_approval)

    agent = Agent(name="test-hitl-stream-async")
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"key": "val"})

    events = []
    async for event in _run.ahandle_agent_run_paused_stream(
        agent=agent,
        run_response=RunOutput(run_id="r1", session_id="s1", messages=[]),
        session=AgentSession(session_id="s1"),
        user_id="u1",
        run_context=run_context,
    ):
        events.append(event)

    assert captured["run_context"] is run_context
    assert len(events) >= 1


def test_handle_agent_run_paused_stream_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_session, "save_session", lambda agent, session: None)
    monkeypatch.setattr(_run, "create_approval_from_pause", lambda **kwargs: None)
    monkeypatch.setattr(_run, "scrub_run_output_for_storage", lambda agent, run_response: None)
    monkeypatch.setattr(_run, "save_run_response_to_file", lambda agent, **kwargs: None)
    monkeypatch.setattr(_run, "update_session_metrics", lambda agent, session, run_response: None)

    agent = Agent(name="test-hitl-stream")
    session = AgentSession(session_id="s1", session_data={})
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"watchlist": ["AAPL"]})

    events = list(
        _run.handle_agent_run_paused_stream(
            agent=agent,
            run_response=run_response,
            session=session,
            user_id="u1",
            run_context=run_context,
        )
    )

    assert len(events) >= 1
    assert session.session_data["session_state"] == {"watchlist": ["AAPL"]}
    assert run_response.session_state == {"watchlist": ["AAPL"]}


@pytest.mark.asyncio
async def test_ahandle_agent_run_paused_stream_persists_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_session, "save_session", lambda agent, session: None)
    monkeypatch.setattr(_run, "scrub_run_output_for_storage", lambda agent, run_response: None)
    monkeypatch.setattr(_run, "save_run_response_to_file", lambda agent, **kwargs: None)
    monkeypatch.setattr(_run, "update_session_metrics", lambda agent, session, run_response: None)

    async def noop_acreate_approval(**kwargs):
        return None

    monkeypatch.setattr(_run, "acreate_approval_from_pause", noop_acreate_approval)

    agent = Agent(name="test-hitl-stream-async")
    session = AgentSession(session_id="s1", session_data={})
    run_response = RunOutput(run_id="r1", session_id="s1", messages=[])
    run_context = RunContext(run_id="r1", session_id="s1", session_state={"cart": ["item-1"]})

    events = []
    async for event in _run.ahandle_agent_run_paused_stream(
        agent=agent,
        run_response=run_response,
        session=session,
        user_id="u1",
        run_context=run_context,
    ):
        events.append(event)

    assert len(events) >= 1
    assert session.session_data["session_state"] == {"cart": ["item-1"]}
    assert run_response.session_state == {"cart": ["item-1"]}


def test_continue_run_dispatch_skips_response_format_when_parser_model_set(monkeypatch: pytest.MonkeyPatch):
    """Regression for #8101: continue_run_dispatch must pass response_format=None
    when agent.parser_model is set, mirroring run_dispatch's guard."""
    agent = _make_precedence_test_agent()
    agent.parser_model = object()  # truthy non-None sentinel
    _patch_continue_dispatch_dependencies(agent, monkeypatch)

    # Override get_response_format to return a non-None sentinel that should be
    # discarded by the parser_model guard.
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: {"type": "json_object"})

    captured: dict[str, Any] = {}

    def fake_continue_run_impl(
        agent: Agent,
        run_response,
        run_messages,
        run_context,
        tools,
        user_id: Optional[str] = None,
        session=None,
        response_format: Optional[Any] = None,
        debug_mode: Optional[bool] = None,
        background_tasks: Optional[Any] = None,
        **kwargs: Any,
    ):
        captured["response_format"] = response_format
        return run_response

    monkeypatch.setattr(_run, "_continue_run", fake_continue_run_impl)

    _run.continue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="cont-parser-1", session_id="session-1", messages=[]),
        stream=False,
    )

    assert captured["response_format"] is None


@pytest.mark.asyncio
async def test_acontinue_run_dispatch_skips_response_format_when_parser_model_set(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression for #8101: acontinue_run_dispatch must pass response_format=None
    when agent.parser_model is set, mirroring arun_dispatch's guard."""
    agent = _make_precedence_test_agent()
    agent.parser_model = object()
    monkeypatch.setattr(agent, "initialize_agent", lambda debug_mode=None: None)
    monkeypatch.setattr(_response, "get_response_format", lambda agent, run_context=None: {"type": "json_object"})

    captured: dict[str, Any] = {}

    async def fake_acontinue_run(
        agent: Agent,
        session_id: str,
        run_context,
        run_response: Optional[RunOutput] = None,
        updated_tools=None,
        requirements=None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Any] = None,
        debug_mode: Optional[bool] = None,
        background_tasks: Optional[Any] = None,
        **kwargs: Any,
    ) -> RunOutput:
        captured["response_format"] = response_format
        return run_response if run_response is not None else RunOutput(run_id=run_id, session_id=session_id)  # type: ignore[arg-type]

    monkeypatch.setattr(_run, "_acontinue_run", fake_acontinue_run)

    await _run.acontinue_run_dispatch(
        agent=agent,
        run_response=RunOutput(run_id="acont-parser-1", session_id="session-1", messages=[]),
        stream=False,
    )

    assert captured["response_format"] is None
