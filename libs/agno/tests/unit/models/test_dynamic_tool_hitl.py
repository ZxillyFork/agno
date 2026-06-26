import asyncio
import json
import pickle
import time

import pytest

from agno import ApprovalRequired, CallDeferred
from agno.agent import Agent
from agno.agent import _run as agent_run
from agno.agent._response import _append_paused_requirements, _upsert_tool_executions, update_run_response
from agno.agent._tools import (
    ahandle_tool_call_updates,
    ahandle_tool_call_updates_stream,
    handle_external_execution_update,
    handle_tool_call_updates,
    handle_tool_call_updates_stream,
    run_tool,
)
from agno.exceptions import RunCancelledException, StopAgentRun, ToolApprovalRequired, ToolCallDeferred
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse, ModelResponseEvent, ToolExecution
from agno.run.agent import RunOutput
from agno.run.base import RunContext
from agno.run.cancel import acancel_run, acleanup_run, aregister_run
from agno.run.messages import RunMessages
from agno.run.requirement import RunRequirement
from agno.tools import tool
from agno.tools.function import Function, FunctionCall
from agno.tools.user_control_flow import UserControlFlowTools
from agno.tools.user_feedback import UserFeedbackTools


class DummyModel(Model):
    def invoke(self, *args, **kwargs) -> ModelResponse:  # pragma: no cover - not needed for tests
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:  # pragma: no cover - not needed for tests
        raise NotImplementedError

    def invoke_stream(self, *args, **kwargs):  # pragma: no cover - not needed for tests
        raise NotImplementedError

    async def ainvoke_stream(self, *args, **kwargs):  # pragma: no cover - not needed for tests
        raise NotImplementedError

    def _parse_provider_response(self, response, **kwargs) -> ModelResponse:  # pragma: no cover - not needed
        raise NotImplementedError

    def _parse_provider_response_delta(self, response) -> ModelResponse:  # pragma: no cover - not needed
        raise NotImplementedError


def test_tool_can_dynamically_require_approval():
    model = DummyModel(id="dummy-model")
    run_context = RunContext(run_id="run_1", session_id="session_1")

    @tool
    def protect_file(path: str, run_context: RunContext) -> str:
        if path == ".env" and not run_context.tool_call_approved:
            raise ToolApprovalRequired(metadata={"reason": "protected_file"})
        return f"updated {path}"

    protect_file._run_context = run_context
    function_call = FunctionCall(
        function=protect_file,
        arguments={"path": ".env"},
        call_id="call_approval",
    )

    function_call_results = []
    responses = list(model.run_function_calls([function_call], function_call_results))

    assert function_call_results == []
    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    tool_execution = paused.tool_executions[0]
    assert tool_execution.tool_call_id == "call_approval"
    assert tool_execution.tool_name == "protect_file"
    assert tool_execution.tool_args == {"path": ".env"}
    assert tool_execution.requires_confirmation is True
    assert tool_execution.approval_type == "required"
    assert tool_execution.metadata == {"reason": "protected_file"}


def test_pydantic_style_exception_aliases_accept_positional_metadata():
    approval = ApprovalRequired({"reason": "protected_file"})
    deferred = CallDeferred({"job_id": "job_123"})

    assert approval.metadata == {"reason": "protected_file"}
    assert deferred.metadata == {"job_id": "job_123"}


def test_dynamic_pause_exceptions_pickle_custom_messages():
    approval = ApprovalRequired({"reason": "protected_file"}, message="custom approval", approval_type="audit")
    deferred = CallDeferred({"job_id": "job_123"}, message="custom deferred")

    restored_approval = pickle.loads(pickle.dumps(approval))
    restored_deferred = pickle.loads(pickle.dumps(deferred))

    assert str(restored_approval) == "custom approval"
    assert restored_approval.metadata == {"reason": "protected_file"}
    assert restored_approval.approval_type == "audit"
    assert str(restored_deferred) == "custom deferred"
    assert restored_deferred.metadata == {"job_id": "job_123"}


def test_dynamic_approval_is_not_bypassed_by_unapproved_cache_lookup(tmp_path):
    model = DummyModel(id="dummy-model")
    run_context = RunContext(run_id="run_1", session_id="session_1")

    @tool(cache_results=True, cache_dir=str(tmp_path))
    def protect_file(path: str, run_context: RunContext) -> str:
        if not run_context.tool_call_approved:
            raise ApprovalRequired({"reason": "protected_file"})
        return f"updated {path}"

    protect_file._run_context = run_context
    run_context.tool_call_approved = True
    approved_result = FunctionCall(
        function=protect_file,
        arguments={"path": ".env"},
        call_id="call_cached_approval",
    ).execute()

    assert approved_result.result == "updated .env"

    run_context.tool_call_approved = False
    responses = list(
        model.run_function_calls(
            [
                FunctionCall(
                    function=protect_file,
                    arguments={"path": ".env"},
                    call_id="call_cached_approval",
                )
            ],
            [],
        )
    )

    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    assert paused.tool_executions[0].requires_confirmation is True
    assert paused.tool_executions[0].metadata == {"reason": "protected_file"}


def test_cached_tool_with_run_context_does_not_use_cache(tmp_path):
    run_context = RunContext(run_id="run_1", session_id="session_1")
    calls = 0

    @tool(cache_results=True, cache_dir=str(tmp_path))
    def cached_lookup(name: str, run_context: RunContext) -> str:
        nonlocal calls
        calls += 1
        return f"hello {name}"

    cached_lookup._run_context = run_context

    first = FunctionCall(
        function=cached_lookup,
        arguments={"name": "Ada"},
        call_id="call_cached_lookup_1",
    ).execute()
    second = FunctionCall(
        function=cached_lookup,
        arguments={"name": "Ada"},
        call_id="call_cached_lookup_2",
    ).execute()

    assert first.result == "hello Ada"
    assert second.result == "hello Ada"
    assert calls == 2


@pytest.mark.asyncio
async def test_async_dynamic_approval_is_not_bypassed_by_tool_cache(tmp_path):
    run_context = RunContext(run_id="run_1", session_id="session_1")
    approval_required = False
    calls = 0

    @tool(cache_results=True, cache_dir=str(tmp_path))
    async def cached_policy_tool(name: str, run_context: RunContext) -> str:
        nonlocal calls
        calls += 1
        if approval_required and not run_context.tool_call_approved:
            raise ApprovalRequired({"reason": "policy_changed"})
        return f"hello {name}"

    cached_policy_tool._run_context = run_context

    first = await FunctionCall(
        function=cached_policy_tool,
        arguments={"name": "Ada"},
        call_id="call_cached_policy_1",
    ).aexecute()

    approval_required = True

    with pytest.raises(ToolApprovalRequired) as exc_info:
        await FunctionCall(
            function=cached_policy_tool,
            arguments={"name": "Ada"},
            call_id="call_cached_policy_2",
        ).aexecute()

    assert first.result == "hello Ada"
    assert exc_info.value.metadata == {"reason": "policy_changed"}
    assert calls == 2


def test_post_hook_dynamic_approval_does_not_pause_after_tool_ran():
    model = DummyModel(id="dummy-model")
    side_effects = []

    def post_hook(**kwargs):
        raise ApprovalRequired({"reason": "post_hook"})

    @tool(post_hook=post_hook)
    def mutate_once() -> str:
        side_effects.append("ran")
        return "done"

    function_call_results = []
    function_call = FunctionCall(function=mutate_once, arguments={}, call_id="call_post_hook")
    responses = list(model.run_function_calls([function_call], function_call_results))

    assert side_effects == ["ran"]
    assert all(response.event != ModelResponseEvent.tool_call_paused.value for response in responses)
    assert function_call.error is None
    assert len(function_call_results) == 1
    assert function_call_results[0].content == "done"


def test_post_hook_does_not_run_when_tool_pauses_before_completion():
    post_hook_called = False

    def post_hook(**kwargs):
        nonlocal post_hook_called
        post_hook_called = True

    @tool(post_hook=post_hook)
    def pause_before_completion() -> str:
        raise ApprovalRequired({"reason": "before_completion"})

    with pytest.raises(ToolApprovalRequired):
        FunctionCall(function=pause_before_completion, arguments={}, call_id="call_pause").execute()

    assert post_hook_called is False


def test_post_hook_does_not_run_when_generator_pauses_during_iteration():
    model = DummyModel(id="dummy-model")
    post_hook_called = False

    def post_hook(**kwargs):
        nonlocal post_hook_called
        post_hook_called = True

    @tool(post_hook=post_hook)
    def pause_generator():
        raise ApprovalRequired({"reason": "generator"})
        yield "unreachable"

    responses = list(
        model.run_function_calls(
            [FunctionCall(function=pause_generator, arguments={}, call_id="call_generator_pause")],
            [],
        )
    )

    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert post_hook_called is False


@pytest.mark.asyncio
async def test_async_generator_pause_does_not_set_function_call_error():
    """ToolApprovalRequired/ToolCallDeferred are control-flow signals, not errors:
    pausing inside an async generator must not poison FunctionCall.error."""
    model = DummyModel(id="dummy-model")

    @tool
    async def pause_async_gen():
        raise ApprovalRequired({"reason": "async_gen"})
        yield "unreachable"  # pragma: no cover

    function_call = FunctionCall(function=pause_async_gen, arguments={}, call_id="call_async_gen")
    responses = [response async for response in model.arun_function_calls([function_call], [])]

    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].metadata == {"reason": "async_gen"}
    # Bug fix: pause exception must not be stored as a tool error
    assert function_call.error is None


def test_dynamic_approval_audit_type_is_promoted_to_required():
    """approval_type='audit' on a dynamic ToolApprovalRequired is contradictory
    (the exception always blocks) and must be promoted to 'required'."""
    model = DummyModel(id="dummy-model")

    @tool
    def audit_dynamic() -> str:
        raise ApprovalRequired({"reason": "audit"}, approval_type="audit")

    responses = list(
        model.run_function_calls(
            [FunctionCall(function=audit_dynamic, arguments={}, call_id="call_audit")],
            [],
        )
    )

    paused = responses[-1]
    te = paused.tool_executions[0]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert te.approval_type == "required"
    assert te.requires_confirmation is True


@pytest.mark.asyncio
async def test_arun_function_calls_honors_async_pre_hook_pause_for_sync_tool():
    model = DummyModel(id="dummy-model")
    side_effects = []

    async def pre_hook(**kwargs):
        raise ApprovalRequired({"reason": "async_pre_hook"})

    @tool(pre_hook=pre_hook)
    def sync_tool() -> str:
        side_effects.append("ran")
        return "done"

    responses = [
        response
        async for response in model.arun_function_calls(
            [FunctionCall(function=sync_tool, arguments={}, call_id="call_async_pre_hook")],
            [],
        )
    ]

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].metadata == {"reason": "async_pre_hook"}


@pytest.mark.asyncio
async def test_async_execution_awaits_sync_hook_wrapping_async_tool():
    model = DummyModel(id="dummy-model")
    calls = []

    def sync_hook(function, arguments):
        calls.append("hook-before")
        result = function(**arguments)
        calls.append("hook-after")
        return result

    @tool(tool_hooks=[sync_hook])
    async def async_tool() -> str:
        calls.append("tool")
        return "done"

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [FunctionCall(function=async_tool, arguments={}, call_id="call_async_tool")],
            function_call_results,
        )
    ]

    assert calls == ["hook-before", "hook-after", "tool"]
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value
    assert function_call_results[0].content == "done"


@pytest.mark.asyncio
async def test_async_callable_pre_hook_can_pause_tool():
    model = DummyModel(id="dummy-model")
    side_effects = []

    class AsyncApprovalHook:
        async def __call__(self, **kwargs):
            raise ApprovalRequired({"reason": "callable-hook"})

    @tool(pre_hook=AsyncApprovalHook())
    def guarded_tool() -> str:
        side_effects.append("ran")
        return "done"

    responses = [
        response
        async for response in model.arun_function_calls(
            [FunctionCall(function=guarded_tool, arguments={}, call_id="call_guarded")],
            [],
        )
    ]

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].metadata == {"reason": "callable-hook"}


@pytest.mark.asyncio
async def test_async_post_hook_runs_for_sync_generator_in_async_execution():
    model = DummyModel(id="dummy-model")
    calls = []

    async def post_hook(**kwargs):
        calls.append("post")

    @tool(post_hook=post_hook)
    def sync_generator():
        yield "chunk"

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [FunctionCall(function=sync_generator, arguments={}, call_id="call_sync_generator")],
            function_call_results,
        )
    ]

    assert calls == ["post"]
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value
    assert function_call_results[0].content == "chunk"


@pytest.mark.asyncio
async def test_async_sync_generator_with_async_hook_does_not_block_event_loop():
    model = DummyModel(id="dummy-model")
    ticked_during_iteration = asyncio.Event()
    iteration_finished = asyncio.Event()

    async def post_hook(**kwargs):
        pass

    @tool(post_hook=post_hook)
    def slow_sync_generator():
        time.sleep(0.2)
        yield "done"

    async def ticker():
        await asyncio.sleep(0.05)
        if not iteration_finished.is_set():
            ticked_during_iteration.set()

    ticker_task = asyncio.create_task(ticker())
    responses = []
    async for response in model.arun_function_calls(
        [FunctionCall(function=slow_sync_generator, arguments={}, call_id="call_slow_generator")],
        [],
    ):
        responses.append(response)
    iteration_finished.set()
    await ticker_task

    assert ticked_during_iteration.is_set()
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value


def test_sync_execution_consumes_sync_generator_with_async_post_hook():
    model = DummyModel(id="dummy-model")
    calls = []

    async def post_hook(**kwargs):
        calls.append("post")

    @tool(post_hook=post_hook)
    def sync_generator():
        calls.append("yield")
        yield "chunk"

    function_call_results = []
    responses = list(
        model.run_function_calls(
            [FunctionCall(function=sync_generator, arguments={}, call_id="call_sync_generator")],
            function_call_results,
        )
    )

    assert calls == ["yield", "post"]
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value
    assert function_call_results[0].content == "chunk"


@pytest.mark.asyncio
async def test_concurrent_async_hooks_restore_live_run_context_messages():
    run_context = RunContext(run_id="run-1", session_id="session-1")
    live_messages = [Message(role="user", content="hello")]
    run_context.messages = live_messages
    seen_messages = []
    hooks_entered = asyncio.Event()
    release_hooks = asyncio.Event()
    entered_count = 0

    async def pre_hook(run_context: RunContext):
        nonlocal entered_count
        seen_messages.append(run_context.messages)
        run_context.messages.append(Message(role="assistant", content="local mutation"))
        entered_count += 1
        if entered_count == 2:
            hooks_entered.set()
        await release_hooks.wait()

    @tool(pre_hook=pre_hook)
    def sync_tool(name: str) -> str:
        return name

    sync_tool._run_context = run_context

    call_one = FunctionCall(function=sync_tool, arguments={"name": "one"}, call_id="call-one")
    call_two = FunctionCall(function=sync_tool, arguments={"name": "two"}, call_id="call-two")

    task_one = asyncio.create_task(call_one.aexecute())
    task_two = asyncio.create_task(call_two.aexecute())
    await asyncio.wait_for(hooks_entered.wait(), timeout=1)
    release_hooks.set()

    await asyncio.gather(task_one, task_two)

    assert run_context.messages is live_messages
    assert [message.content for message in run_context.messages] == ["hello"]
    assert len(seen_messages) == 2
    assert all(messages is not live_messages for messages in seen_messages)
    assert seen_messages[0] is not seen_messages[1]


@pytest.mark.asyncio
async def test_concurrent_async_hook_fc_views_do_not_swap_shared_run_context():
    run_context = RunContext(run_id="run-1", session_id="session-1")
    live_messages = [Message(role="user", content="hello")]
    run_context.messages = live_messages
    seen_contexts = []
    hooks_entered = asyncio.Event()
    release_hooks = asyncio.Event()
    entered_count = 0

    async def pre_hook(fc: FunctionCall):
        nonlocal entered_count
        assert fc.function._run_context is not run_context
        assert fc.function._run_context is not None
        seen_contexts.append(fc.function._run_context)
        fc.function._run_context.messages.append(Message(role="assistant", content="hook local"))
        entered_count += 1
        if entered_count == 2:
            hooks_entered.set()
        await release_hooks.wait()

    @tool(pre_hook=pre_hook)
    def sync_tool(name: str) -> str:
        return name

    sync_tool._run_context = run_context

    task_one = asyncio.create_task(
        FunctionCall(function=sync_tool, arguments={"name": "one"}, call_id="call-one").aexecute()
    )
    task_two = asyncio.create_task(
        FunctionCall(function=sync_tool, arguments={"name": "two"}, call_id="call-two").aexecute()
    )
    await asyncio.wait_for(hooks_entered.wait(), timeout=1)
    assert sync_tool._run_context is run_context
    release_hooks.set()

    await asyncio.gather(task_one, task_two)

    assert sync_tool._run_context is run_context
    assert run_context.messages is live_messages
    assert [message.content for message in run_context.messages] == ["hello"]
    assert len(seen_contexts) == 2
    assert seen_contexts[0] is not seen_contexts[1]


@pytest.mark.asyncio
async def test_hook_receiving_fc_cannot_mutate_live_run_context_messages():
    run_context = RunContext(run_id="run-1", session_id="session-1")
    live_messages = [Message(role="user", content="hello")]
    run_context.messages = live_messages

    async def pre_hook(fc: FunctionCall):
        assert fc.function._run_context is not run_context
        assert fc.function._run_context is not None
        fc.function._run_context.messages.clear()

    @tool(pre_hook=pre_hook)
    def sync_tool() -> str:
        return "done"

    sync_tool._run_context = run_context

    result = await FunctionCall(function=sync_tool, arguments={}, call_id="call-fc-hook").aexecute()

    assert result.status == "success"
    assert sync_tool._run_context is run_context
    assert run_context.messages is live_messages
    assert [message.content for message in run_context.messages] == ["hello"]


@pytest.mark.asyncio
async def test_sync_pre_hook_returning_awaitable_in_running_loop_fails_closed():
    side_effects = []

    def pre_hook(**kwargs):
        async def pause():
            raise ApprovalRequired({"reason": "awaitable_pre_hook"})

        return pause()

    @tool(pre_hook=pre_hook)
    def guarded_tool() -> str:
        side_effects.append("ran")
        return "done"

    with pytest.raises(RuntimeError, match="use aexecute"):
        FunctionCall(function=guarded_tool, arguments={}, call_id="call-awaitable-pre-hook").execute()

    assert side_effects == []


@pytest.mark.asyncio
async def test_sync_tool_hook_returning_awaitable_in_running_loop_fails_closed():
    side_effects = []
    post_hook_calls = []

    def hook(function, arguments):
        async def run_later():
            side_effects.append("awaited")
            return function(**arguments)

        return run_later()

    def post_hook(**kwargs):
        post_hook_calls.append("post")

    @tool(tool_hooks=[hook], post_hook=post_hook)
    def guarded_tool() -> str:
        side_effects.append("ran")
        return "done"

    with pytest.raises(RuntimeError, match="use aexecute"):
        FunctionCall(function=guarded_tool, arguments={}, call_id="call-awaitable-tool-hook").execute()

    assert side_effects == []
    assert post_hook_calls == []


def test_post_hook_runs_when_sync_generator_raises_regular_exception():
    calls = []

    def post_hook(**kwargs):
        calls.append("post")

    @tool(post_hook=post_hook)
    def failing_generator():
        yield "chunk"
        raise RuntimeError("boom")

    model = DummyModel(id="dummy-model")
    function_call_results = []
    responses = list(
        model.run_function_calls(
            [FunctionCall(function=failing_generator, arguments={}, call_id="call-failing-generator")],
            function_call_results,
        )
    )

    assert calls == ["post"]
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value
    assert responses[-1].tool_executions[0].tool_call_error is True


def test_sync_generator_post_hook_stop_agent_preserves_agent_exception_semantics():
    def post_hook(**kwargs):
        raise StopAgentRun("stop now", user_message="please stop")

    @tool(post_hook=post_hook)
    def generator_tool():
        yield "chunk"

    model = DummyModel(id="dummy-model")
    function_call_results = []
    responses = list(
        model.run_function_calls(
            [FunctionCall(function=generator_tool, arguments={}, call_id="call-generator-stop")],
            function_call_results,
        )
    )

    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value
    assert responses[-1].tool_executions[0].tool_call_error is True
    assert function_call_results[0].stop_after_tool_call is True
    assert function_call_results[0].content == "stop now"
    assert function_call_results[1].role == "user"
    assert function_call_results[1].content == "please stop"
    assert function_call_results[1].stop_after_tool_call is True


@pytest.mark.asyncio
async def test_post_hook_runs_when_async_generator_raises_regular_exception():
    calls = []

    async def post_hook(**kwargs):
        calls.append("post")

    @tool(post_hook=post_hook)
    async def failing_generator():
        yield "chunk"
        raise RuntimeError("boom")

    model = DummyModel(id="dummy-model")
    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [FunctionCall(function=failing_generator, arguments={}, call_id="call-failing-generator")],
            function_call_results,
        )
    ]

    assert calls == ["post"]
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value
    assert responses[-1].tool_executions[0].tool_call_error is True


@pytest.mark.asyncio
async def test_async_generator_post_hook_stop_agent_preserves_agent_exception_semantics():
    async def post_hook(**kwargs):
        raise StopAgentRun("stop now", user_message="please stop")

    @tool(post_hook=post_hook)
    async def generator_tool():
        yield "chunk"

    model = DummyModel(id="dummy-model")
    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [FunctionCall(function=generator_tool, arguments={}, call_id="call-generator-stop")],
            function_call_results,
        )
    ]

    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value
    assert responses[-1].tool_executions[0].tool_call_error is True
    assert function_call_results[0].stop_after_tool_call is True
    assert function_call_results[0].content == "stop now"
    assert function_call_results[1].role == "user"
    assert function_call_results[1].content == "please stop"
    assert function_call_results[1].stop_after_tool_call is True


def test_sync_tool_hook_returning_awaitable_can_pause():
    model = DummyModel(id="dummy-model")
    side_effects = []

    def hook(function, arguments):
        async def pause():
            raise ApprovalRequired({"reason": "awaitable_hook"})

        return pause()

    @tool(tool_hooks=[hook])
    def guarded_tool() -> str:
        side_effects.append("ran")
        return "done"

    responses = list(
        model.run_function_calls(
            [FunctionCall(function=guarded_tool, arguments={}, call_id="call_hook_awaitable")],
            [],
        )
    )

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].metadata == {"reason": "awaitable_hook"}


def test_sync_hook_returning_awaitable_uses_copied_messages_until_await_finishes():
    live_messages = [Message(role="user", content="live")]
    run_context = RunContext(run_id="run_1", session_id="session_1", messages=live_messages)

    def hook(run_context: RunContext):
        async def mutate_messages():
            run_context.messages.clear()

        return mutate_messages()

    @tool(pre_hook=hook)
    def guarded_tool(run_context: RunContext) -> str:
        return "done"

    guarded_tool._run_context = run_context
    FunctionCall(function=guarded_tool, arguments={}, call_id="call_hook_messages").execute()

    assert run_context.messages is live_messages
    assert [message.content for message in run_context.messages] == ["live"]


@pytest.mark.asyncio
async def test_sync_execute_with_async_hook_in_running_loop_fails_closed():
    async def pre_hook(**kwargs):
        raise ApprovalRequired({"reason": "async_pre_hook"})

    @tool(pre_hook=pre_hook)
    def sync_tool() -> str:
        return "done"

    with pytest.raises(RuntimeError, match="use aexecute"):
        FunctionCall(function=sync_tool, arguments={}, call_id="call_direct_execute").execute()


def test_confirmed_dynamic_approval_reruns_instead_of_using_cache(tmp_path):
    run_context = RunContext(run_id="run_1", session_id="session_1")
    calls = 0

    @tool(cache_results=True, cache_dir=str(tmp_path))
    def protected_cached_tool(run_context: RunContext) -> str:
        nonlocal calls
        calls += 1
        if not run_context.tool_call_approved:
            raise ApprovalRequired({"reason": "approval"})
        return f"approved-call-{calls}"

    protected_cached_tool._run_context = run_context
    run_context.tool_call_approved = True

    first = FunctionCall(function=protected_cached_tool, arguments={}, call_id="call_cache_1").execute()
    second = FunctionCall(function=protected_cached_tool, arguments={}, call_id="call_cache_2").execute()

    assert first.result == "approved-call-1"
    assert second.result == "approved-call-2"
    assert calls == 2


def test_run_requirement_from_dict_syncs_top_level_resolution_to_tool_execution():
    requirement = RunRequirement.from_dict(
        {
            "tool_execution": {
                "tool_call_id": "call_approval",
                "tool_name": "protect_file",
                "tool_args": {"path": ".env"},
                "requires_confirmation": True,
            },
            "confirmation": True,
        }
    )

    assert requirement.confirmation is True
    assert requirement.tool_execution is not None
    assert requirement.tool_execution.confirmed is True


def test_run_requirement_confirm_sets_resume_metadata_without_overwriting_request_metadata():
    tool_execution = ToolExecution(
        tool_call_id="call_approval",
        tool_name="protect_file",
        tool_args={"path": ".env"},
        requires_confirmation=True,
        metadata={"reason": "request_side"},
    )
    requirement = RunRequirement(tool_execution=tool_execution)

    requirement.confirm(metadata={"approver": "alice"})

    assert requirement.approval_metadata == {"approver": "alice"}
    assert tool_execution.metadata == {"reason": "request_side"}
    assert tool_execution.resume_metadata == {"approver": "alice"}


def test_run_requirement_external_execution_accepts_none_result():
    tool_execution = ToolExecution(
        tool_call_id="call_deferred",
        tool_name="external_tool",
        tool_args={},
        external_execution_required=True,
    )
    requirement = RunRequirement(tool_execution=tool_execution)

    requirement.set_external_execution_result(None)
    restored = RunRequirement.from_dict(requirement.to_dict())

    assert requirement.needs_external_execution is False
    assert requirement.is_resolved() is True
    assert requirement.to_dict()["external_execution_result"] is None
    assert restored.needs_external_execution is False
    assert restored.tool_execution.external_execution_result_provided is True
    assert restored.tool_execution.result is None


def test_run_requirement_from_dict_null_external_result_without_flag_stays_pending():
    requirement = RunRequirement.from_dict(
        {
            "tool_execution": {
                "tool_call_id": "call_deferred",
                "tool_name": "external_tool",
                "tool_args": {},
                "external_execution_required": True,
            },
            "external_execution_result": None,
        }
    )

    assert requirement.needs_external_execution is True
    assert requirement.is_resolved() is False
    assert requirement.tool_execution.external_execution_result_provided is not True


def test_run_requirement_from_dict_hydrates_nested_tool_resolution_fields():
    requirement = RunRequirement.from_dict(
        {
            "tool_execution": {
                "tool_call_id": "call_deferred",
                "tool_name": "external_tool",
                "tool_args": {},
                "external_execution_required": True,
                "result": {"ok": True},
            }
        }
    )

    assert requirement.external_execution_result == {"ok": True}
    assert requirement.needs_external_execution is False
    assert requirement.tool_execution.external_execution_result_provided is True


def test_run_requirement_from_dict_does_not_treat_confirmation_result_as_external_result():
    requirement = RunRequirement.from_dict(
        {
            "tool_execution": {
                "tool_call_id": "call_approval",
                "tool_name": "protected_tool",
                "tool_args": {},
                "requires_confirmation": True,
                "confirmed": True,
                "result": "normal tool output",
            }
        }
    )

    assert requirement.external_execution_result is None
    assert requirement.external_execution_result_provided is False
    assert requirement.tool_execution.result == "normal tool output"


def test_run_requirement_from_dict_preserves_nested_resume_metadata():
    requirement = RunRequirement.from_dict(
        {
            "tool_execution": {
                "tool_call_id": "call_approval",
                "tool_name": "protected_tool",
                "tool_args": {},
                "requires_confirmation": True,
                "confirmed": True,
                "resume_metadata": {"approver": "alice"},
            },
            "confirmation": True,
        }
    )

    assert requirement.approval_metadata == {"approver": "alice"}
    assert requirement.tool_execution.resume_metadata == {"approver": "alice"}


def test_external_execution_update_serializes_structured_result():
    agent = Agent(model=DummyModel(id="dummy-model"), telemetry=False)
    run_messages = RunMessages()
    tool_execution = ToolExecution(
        tool_call_id="call_deferred",
        tool_name="external_tool",
        tool_args={},
        external_execution_required=True,
        result={"ok": True},
        external_execution_result_provided=True,
    )

    handle_external_execution_update(agent, run_messages=run_messages, tool=tool_execution)

    assert tool_execution.external_execution_required is False
    assert json.loads(run_messages.messages[0].content) == {"ok": True}


def test_external_execution_update_serializes_list_result_as_json_payload():
    agent = Agent(model=DummyModel(id="dummy-model"), telemetry=False)
    run_messages = RunMessages()
    tool_execution = ToolExecution(
        tool_call_id="call_deferred",
        tool_name="external_tool",
        tool_args={},
        external_execution_required=True,
        result=["a", "b"],
        external_execution_result_provided=True,
    )

    handle_external_execution_update(agent, run_messages=run_messages, tool=tool_execution)

    assert run_messages.messages[0].content == '["a", "b"]'


def test_tool_execution_round_trip_preserves_absent_schemas_as_none():
    tool_execution = ToolExecution(tool_call_id="call_1", tool_name="plain_tool", tool_args={})

    restored = ToolExecution.from_dict(tool_execution.to_dict())

    assert restored.user_input_schema is None
    assert restored.user_feedback_schema is None


def test_tool_execution_from_dict_accepts_hydrated_schema_entries():
    from agno.tools.function import UserFeedbackOption, UserFeedbackQuestion, UserInputField

    user_input_field = UserInputField(name="reason", field_type=str)
    user_feedback_question = UserFeedbackQuestion(
        question="Deploy?",
        options=[UserFeedbackOption(label="Yes"), UserFeedbackOption(label="No")],
    )

    restored = ToolExecution.from_dict(
        {
            "tool_name": "ask",
            "tool_args": {},
            "requires_user_input": True,
            "user_input_schema": [user_input_field],
            "user_feedback_schema": [user_feedback_question],
        }
    )

    assert restored.user_input_schema == [user_input_field]
    assert restored.user_feedback_schema == [user_feedback_question]


def test_run_requirement_round_trip_preserves_empty_schemas():
    requirement = RunRequirement.from_dict(
        {
            "tool_execution": {
                "tool_call_id": "call_user",
                "tool_name": "input_tool",
                "requires_user_input": True,
            },
            "user_input_schema": [],
            "user_feedback_schema": [],
        }
    )

    assert requirement.user_input_schema == []
    assert requirement.user_feedback_schema == []
    assert requirement.to_dict()["user_input_schema"] == []
    assert requirement.to_dict()["user_feedback_schema"] == []


def test_model_response_from_dict_does_not_mutate_payload_and_ignores_unknown_keys():
    payload = ModelResponse(
        tool_executions=[
            ToolExecution(
                tool_call_id="call_1",
                tool_name="tool",
                requires_confirmation=True,
            )
        ],
        event=ModelResponseEvent.tool_call_paused.value,
    ).to_dict()
    payload["future_field"] = "ignore me"

    first = ModelResponse.from_dict(payload)
    second = ModelResponse.from_dict(payload)

    assert first.tool_executions[0].tool_call_id == "call_1"
    assert second.tool_executions[0].tool_call_id == "call_1"
    assert isinstance(payload["tool_executions"][0], dict)


def test_dynamic_pause_replaces_started_tool_execution_in_run_output():
    agent = Agent(model=DummyModel(id="dummy-model"), telemetry=False)
    run_response = RunOutput(run_id="run_1", agent_id="agent_1", session_id="session_1")
    run_messages = RunMessages()
    started_tool = ToolExecution(
        tool_call_id="call_approval",
        tool_name="protect_file",
        tool_args={"path": ".env"},
    )
    paused_tool = ToolExecution(
        tool_call_id="call_approval",
        tool_name="protect_file",
        tool_args={"path": ".env"},
        requires_confirmation=True,
        metadata={"reason": "protected_file"},
    )

    update_run_response(
        agent,
        model_response=ModelResponse(tool_executions=[started_tool, paused_tool]),
        run_response=run_response,
        run_messages=run_messages,
    )

    assert run_response.tools == [paused_tool]


def test_confirmed_dynamic_approval_sets_resume_context():
    model = DummyModel(id="dummy-model")
    run_context = RunContext(run_id="run_1", session_id="session_1")

    @tool
    def protect_file(path: str, run_context: RunContext) -> str:
        if not run_context.tool_call_approved:
            raise ToolApprovalRequired(metadata={"reason": "protected_file"})
        return f"approved {path}: {run_context.tool_call_metadata['reason']}"

    protect_file._run_context = run_context
    agent = Agent(model=model, tools=[protect_file], telemetry=False)
    run_response = RunOutput(run_id="run_1", agent_id="agent_1", session_id="session_1")
    run_messages = RunMessages()
    tool_execution = next(
        response.tool_executions[0]
        for response in model.run_function_calls(
            [
                FunctionCall(
                    function=protect_file,
                    arguments={"path": ".env"},
                    call_id="call_approval",
                )
            ],
            [],
        )
        if response.event == ModelResponseEvent.tool_call_paused.value and response.tool_executions
    )

    tool_execution.confirmed = True
    tool_execution.resume_metadata = {"reason": "user_approved"}

    list(
        run_tool(
            agent,
            run_response=run_response,
            run_messages=run_messages,
            tool=tool_execution,
            functions={protect_file.name: protect_file},
        )
    )

    assert tool_execution.result == "approved .env: user_approved"
    assert run_context.tool_call_approved is False
    assert run_context.tool_call_metadata is None


def test_resumed_dynamic_tool_can_pause_again_for_external_execution():
    model = DummyModel(id="dummy-model")
    run_context = RunContext(run_id="run_1", session_id="session_1")

    @tool
    def approve_then_defer(path: str, run_context: RunContext) -> str:
        if not run_context.tool_call_approved:
            raise ApprovalRequired({"phase": "approval"})
        raise CallDeferred({"job_id": "job_123", "phase": run_context.tool_call_metadata["phase"]})

    approve_then_defer._run_context = run_context
    agent = Agent(model=model, tools=[approve_then_defer], telemetry=False)
    run_response = RunOutput(run_id="run_1", agent_id="agent_1", session_id="session_1")
    run_messages = RunMessages()
    tool_execution = next(
        response.tool_executions[0]
        for response in model.run_function_calls(
            [
                FunctionCall(
                    function=approve_then_defer,
                    arguments={"path": ".env"},
                    call_id="call_approval_then_defer",
                )
            ],
            [],
        )
        if response.event == ModelResponseEvent.tool_call_paused.value and response.tool_executions
    )
    requirement = RunRequirement(tool_execution=tool_execution)
    requirement.confirm(metadata={"phase": "approval"})
    run_response.tools = [tool_execution]
    run_response.requirements = [requirement]

    list(
        run_tool(
            agent,
            run_response=run_response,
            run_messages=run_messages,
            tool=tool_execution,
            functions={approve_then_defer.name: approve_then_defer},
        )
    )

    assert run_messages.messages == []
    assert run_response.tools is not None
    assert run_response.tools[0] is not tool_execution
    resumed_pause = run_response.tools[0]
    assert resumed_pause.tool_call_id == "call_approval_then_defer"
    assert resumed_pause.requires_confirmation is not True
    assert resumed_pause.external_execution_required is True
    assert resumed_pause.metadata == {"job_id": "job_123", "phase": "approval"}
    assert run_response.requirements is not None
    assert run_response.active_requirements == [run_response.requirements[-1]]


def test_sync_dynamic_pause_stops_later_tools_in_same_batch():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool
    def pause_first() -> str:
        raise ApprovalRequired({"reason": "stop_batch"})

    @tool
    def should_not_run() -> str:
        side_effects.append("ran")
        return "ran"

    responses = list(
        model.run_function_calls(
            [
                FunctionCall(function=pause_first, arguments={}, call_id="call_pause"),
                FunctionCall(function=should_not_run, arguments={}, call_id="call_later"),
            ],
            [],
        )
    )

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"


def test_agent_continue_unresolved_requirement_repauses_without_rejecting():
    tool_execution = ToolExecution(
        tool_call_id="call_approval",
        tool_name="protect_file",
        tool_args={"path": ".env"},
        requires_confirmation=True,
    )
    requirement = RunRequirement(tool_execution=tool_execution)
    run_response = RunOutput(
        run_id="run_1",
        agent_id="agent_1",
        session_id="session_1",
        tools=[tool_execution],
        requirements=[requirement],
        messages=[],
    )

    @tool
    def protect_file(path: str) -> str:
        return f"updated {path}"

    agent = Agent(model=DummyModel(id="dummy-model"), tools=[protect_file], telemetry=False)
    continued = agent.continue_run(run_response=run_response, stream=False)

    assert continued.is_paused
    assert tool_execution.confirmed is None
    assert tool_execution.requires_confirmation is True


def test_agent_continue_empty_requirements_does_not_mask_paused_tool():
    tool_execution = ToolExecution(
        tool_call_id="call_approval",
        tool_name="protect_file",
        tool_args={"path": ".env"},
        requires_confirmation=True,
    )
    run_response = RunOutput(
        run_id="run_1",
        agent_id="agent_1",
        session_id="session_1",
        tools=[tool_execution],
        requirements=[],
        messages=[],
    )

    @tool
    def protect_file(path: str) -> str:
        return f"updated {path}"

    agent = Agent(model=DummyModel(id="dummy-model"), tools=[protect_file], telemetry=False)
    continued = agent.continue_run(run_response=run_response, requirements=[], stream=False)

    assert continued.is_paused
    assert tool_execution.confirmed is None
    assert tool_execution.requires_confirmation is True


def test_agent_continue_applies_requirements_when_run_response_is_provided(monkeypatch):
    original_tool = ToolExecution(
        tool_call_id="call_external",
        tool_name="start_background_job",
        tool_args={"task": "summarize"},
        external_execution_required=True,
    )
    resolved_tool = ToolExecution(
        tool_call_id="call_external",
        tool_name="start_background_job",
        tool_args={"task": "summarize"},
        external_execution_required=True,
    )
    requirement = RunRequirement(tool_execution=resolved_tool)
    requirement.set_external_execution_result({"ok": True})
    run_response = RunOutput(
        run_id="run_1",
        agent_id="agent_1",
        session_id="session_1",
        tools=[original_tool],
        requirements=[RunRequirement(tool_execution=original_tool)],
        messages=[],
    )
    captured = {}

    def fake_continue_run(agent, run_response, **kwargs):
        captured["run_response"] = run_response
        return run_response

    monkeypatch.setattr(agent_run, "_continue_run", fake_continue_run)
    agent = Agent(model=DummyModel(id="dummy-model"), telemetry=False)

    continued = agent.continue_run(run_response=run_response, requirements=[requirement], stream=False)

    assert continued is run_response
    assert captured["run_response"].requirements == run_response.requirements
    assert captured["run_response"].requirements[0].tool_execution is original_tool
    assert captured["run_response"].requirements[0].is_resolved()
    assert captured["run_response"].tools[0].result == {"ok": True}


def test_agent_continue_paused_early_path_cleans_registered_run(monkeypatch):
    from agno.session.agent import AgentSession

    cleaned = []
    disconnected = []
    tool_execution = ToolExecution(
        tool_call_id="call_approval",
        tool_name="protect_file",
        tool_args={},
        requires_confirmation=True,
    )
    run_response = RunOutput(
        run_id="run_cleanup",
        agent_id="agent_1",
        session_id="session_1",
        tools=[tool_execution],
        requirements=[RunRequirement(tool_execution=tool_execution)],
        messages=[],
    )
    agent = Agent(model=DummyModel(id="dummy-model"), telemetry=False)

    monkeypatch.setattr("agno.agent._tools.handle_tool_call_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr("agno.agent._init.disconnect_connectable_tools", lambda agent: disconnected.append(True))
    monkeypatch.setattr(agent_run, "cleanup_run", lambda run_id: cleaned.append(run_id))

    continued = agent_run._continue_run(
        agent=agent,
        run_response=run_response,
        run_messages=RunMessages(messages=[]),
        run_context=RunContext(run_id="run_cleanup", session_id="session_1"),
        session=AgentSession(session_id="session_1"),
        tools=[],
    )

    assert continued.is_paused
    assert disconnected == [True]
    assert cleaned == ["run_cleanup"]


def test_tool_can_dynamically_defer_to_external_execution():
    model = DummyModel(id="dummy-model")

    @tool
    def start_background_job(task: str) -> str:
        raise ToolCallDeferred(metadata={"job_id": "job_123", "task": task})

    function_call_results = []
    responses = list(
        model.run_function_calls(
            [
                FunctionCall(
                    function=start_background_job,
                    arguments={"task": "summarize"},
                    call_id="call_deferred",
                )
            ],
            function_call_results,
        )
    )

    assert function_call_results == []
    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    tool_execution = paused.tool_executions[0]
    assert tool_execution.tool_call_id == "call_deferred"
    assert tool_execution.external_execution_required is True
    assert tool_execution.metadata == {"job_id": "job_123", "task": "summarize"}

    agent = Agent(model=model, tools=[start_background_job], telemetry=False)
    run_messages = RunMessages()
    tool_execution.result = "external result"
    handle_external_execution_update(agent, run_messages=run_messages, tool=tool_execution)

    assert len(run_messages.messages) == 1
    assert run_messages.messages[0].content == "external result"
    assert tool_execution.external_execution_required is False


@pytest.mark.asyncio
async def test_async_tool_can_dynamically_defer_to_external_execution():
    model = DummyModel(id="dummy-model")

    @tool
    async def start_background_job(task: str) -> str:
        raise ToolCallDeferred(metadata={"job_id": "job_123", "task": task})

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(
                    function=start_background_job,
                    arguments={"task": "summarize"},
                    call_id="call_deferred",
                )
            ],
            function_call_results,
        )
    ]

    assert function_call_results == []
    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    tool_execution = paused.tool_executions[0]
    assert tool_execution.external_execution_required is True
    assert tool_execution.metadata == {"job_id": "job_123", "task": "summarize"}


@pytest.mark.asyncio
async def test_async_get_user_input_pauses_once_without_mutating_function():
    model = DummyModel(id="dummy-model")
    get_user_input = UserControlFlowTools().get_functions()["get_user_input"]

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(
                    function=get_user_input,
                    arguments={
                        "user_input_fields": [
                            {
                                "field_name": "city",
                                "field_type": "str",
                                "field_description": "City name",
                            }
                        ]
                    },
                    call_id="call_user_input",
                )
            ],
            [],
        )
    ]

    paused = [response for response in responses if response.event == ModelResponseEvent.tool_call_paused.value]
    assert len(paused) == 1
    assert len(paused[0].tool_executions) == 1
    assert paused[0].tool_executions[0].tool_name == "get_user_input"
    assert get_user_input.requires_user_input is not True


@pytest.mark.asyncio
async def test_async_ask_user_pauses_once_without_mutating_function():
    model = DummyModel(id="dummy-model")
    ask_user = UserFeedbackTools().get_functions()["ask_user"]

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(
                    function=ask_user,
                    arguments={
                        "questions": [
                            {
                                "question": "Deploy now?",
                                "header": "Deploy",
                                "options": [{"label": "Yes"}, {"label": "No"}],
                            }
                        ]
                    },
                    call_id="call_ask_user",
                )
            ],
            [],
        )
    ]

    paused = [response for response in responses if response.event == ModelResponseEvent.tool_call_paused.value]
    assert len(paused) == 1
    assert len(paused[0].tool_executions) == 1
    assert paused[0].tool_executions[0].tool_name == "ask_user"
    assert ask_user.requires_user_input is not True


@pytest.mark.asyncio
async def test_async_tool_dynamic_pause_waits_for_slow_sibling_before_yielding():
    model = DummyModel(id="dummy-model")
    side_effects = []

    async def post_hook(**kwargs):
        side_effects.append("slow-post")

    @tool
    async def pause_quickly():
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"reason": "ordinary_async"})

    @tool(post_hook=post_hook)
    async def slow_tool():
        await asyncio.sleep(0.1)
        side_effects.append("slow-finished")
        return "slow"

    started_at = asyncio.get_running_loop().time()
    responses = []
    async for response in model.arun_function_calls(
        [
            FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
            FunctionCall(function=slow_tool, arguments={}, call_id="call_slow"),
        ],
        [],
    ):
        responses.append(response)
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_paused.value:
            break

    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed >= 0.1
    assert side_effects == ["slow-finished", "slow-post"]
    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    assert paused.tool_executions[0].tool_call_id == "call_pause"
    assert paused.tool_executions[0].metadata == {"reason": "ordinary_async"}
    assert any(
        response.event == ModelResponseEvent.tool_call_completed.value
        and response.tool_executions[0].tool_call_id == "call_slow"
        for response in responses
    )


@pytest.mark.asyncio
async def test_async_tool_dynamic_pause_collects_ready_sibling_pause():
    model = DummyModel(id="dummy-model")
    first_pausing = asyncio.Event()

    @tool
    async def pause_first():
        first_pausing.set()
        raise ApprovalRequired({"reason": "first"})

    @tool
    async def pause_second():
        await first_pausing.wait()
        await asyncio.sleep(0.01)
        raise ApprovalRequired({"reason": "second"})

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_first, arguments={}, call_id="call_first"),
                FunctionCall(function=pause_second, arguments={}, call_id="call_second"),
            ],
            [],
        )
    ]

    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    assert [tool.tool_call_id for tool in paused.tool_executions] == ["call_first", "call_second"]
    assert [tool.metadata["reason"] for tool in paused.tool_executions] == ["first", "second"]


@pytest.mark.asyncio
async def test_async_function_task_phase_cleans_up_when_iterator_is_cancelled():
    model = DummyModel(id="dummy-model")
    side_effects = []
    cancelled = asyncio.Event()

    @tool
    async def slow_one():
        try:
            await asyncio.sleep(1)
            side_effects.append("one-finished")
            return "one"
        except asyncio.CancelledError:
            side_effects.append("one-cancelled")
            if len(side_effects) == 2:
                cancelled.set()
            raise

    @tool
    async def slow_two():
        try:
            await asyncio.sleep(1)
            side_effects.append("two-finished")
            return "two"
        except asyncio.CancelledError:
            side_effects.append("two-cancelled")
            if len(side_effects) == 2:
                cancelled.set()
            raise

    iterator = model.arun_function_calls(
        [
            FunctionCall(function=slow_one, arguments={}, call_id="call_one"),
            FunctionCall(function=slow_two, arguments={}, call_id="call_two"),
        ],
        [],
    )
    assert (await iterator.__anext__()).event == ModelResponseEvent.tool_call_started.value
    assert (await iterator.__anext__()).event == ModelResponseEvent.tool_call_started.value

    pending_next = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0.05)
    pending_next.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_next

    await asyncio.wait_for(cancelled.wait(), timeout=0.5)
    assert "one-finished" not in side_effects
    assert "two-finished" not in side_effects


@pytest.mark.asyncio
async def test_async_dynamic_pause_does_not_start_later_sync_sibling():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool
    async def pause_quickly():
        raise ApprovalRequired({"reason": "ordinary_async"})

    @tool
    def sync_sibling():
        side_effects.append("sync-ran")
        return "sync"

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
                FunctionCall(function=sync_sibling, arguments={}, call_id="call_sync"),
            ],
            [],
        )
    ]

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"


@pytest.mark.asyncio
async def test_async_dynamic_pause_respects_earlier_sync_call_order():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool
    def slow_sync_tool() -> str:
        time.sleep(0.05)
        side_effects.append("sync-ran")
        return "sync"

    @tool
    async def pause_quickly():
        raise ApprovalRequired({"reason": "ordinary_async"})

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=slow_sync_tool, arguments={}, call_id="call_sync"),
                FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
            ],
            [],
        )
    ]

    assert side_effects == ["sync-ran"]
    completed = [
        response
        for response in responses
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_completed.value
    ]
    assert completed[0].tool_executions[0].tool_call_id == "call_sync"
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"


@pytest.mark.asyncio
async def test_async_sync_dynamic_pause_stops_later_async_sibling():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool
    def pause_sync():
        raise ApprovalRequired({"reason": "sync_first"})

    @tool
    async def later_async():
        side_effects.append("later-ran")
        return "later"

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_sync, arguments={}, call_id="call_pause"),
                FunctionCall(function=later_async, arguments={}, call_id="call_later"),
            ],
            [],
        )
    ]

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"


@pytest.mark.asyncio
async def test_async_execution_of_sync_tool_does_not_block_event_loop():
    model = DummyModel(id="dummy-model")
    ticked_during_execution = asyncio.Event()
    execution_finished = asyncio.Event()

    @tool
    def slow_sync_tool() -> str:
        time.sleep(0.2)
        return "done"

    async def ticker():
        await asyncio.sleep(0.05)
        if not execution_finished.is_set():
            ticked_during_execution.set()

    ticker_task = asyncio.create_task(ticker())
    responses = []
    async for response in model.arun_function_calls(
        [FunctionCall(function=slow_sync_tool, arguments={}, call_id="call_slow_sync")],
        [],
    ):
        responses.append(response)
    execution_finished.set()
    await ticker_task

    assert ticked_during_execution.is_set()
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value


@pytest.mark.asyncio
async def test_async_sync_tool_with_async_hook_does_not_block_event_loop():
    model = DummyModel(id="dummy-model")
    ticked_during_execution = asyncio.Event()
    execution_finished = asyncio.Event()

    async def post_hook(**kwargs):
        pass

    @tool(post_hook=post_hook)
    def slow_sync_tool() -> str:
        time.sleep(0.2)
        return "done"

    async def ticker():
        await asyncio.sleep(0.05)
        if not execution_finished.is_set():
            ticked_during_execution.set()

    ticker_task = asyncio.create_task(ticker())
    responses = []
    async for response in model.arun_function_calls(
        [FunctionCall(function=slow_sync_tool, arguments={}, call_id="call_slow_sync")],
        [],
    ):
        responses.append(response)
    execution_finished.set()
    await ticker_task

    assert ticked_during_execution.is_set()
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value


@pytest.mark.asyncio
async def test_async_hook_on_sync_tool_runs_on_caller_event_loop():
    model = DummyModel(id="dummy-model")
    caller_loop = asyncio.get_running_loop()
    hook_loop = None
    hook_released = asyncio.Event()

    async def pre_hook(**kwargs):
        nonlocal hook_loop
        hook_loop = asyncio.get_running_loop()
        await hook_released.wait()

    @tool(pre_hook=pre_hook)
    def sync_tool() -> str:
        return "done"

    async def release_hook():
        await asyncio.sleep(0.05)
        hook_released.set()

    release_task = asyncio.create_task(release_hook())
    responses = [
        response
        async for response in model.arun_function_calls(
            [FunctionCall(function=sync_tool, arguments={}, call_id="call_sync_hook")],
            [],
        )
    ]
    await release_task

    assert hook_loop is caller_loop
    assert responses[-1].event == ModelResponseEvent.tool_call_completed.value


@pytest.mark.asyncio
async def test_aexecute_handles_empty_tool_hooks_for_sync_tool():
    @tool(tool_hooks=[])
    def plain_tool() -> str:
        return "ok"

    result = await FunctionCall(function=plain_tool, arguments={}, call_id="call_plain").aexecute()

    assert result.status == "success"
    assert result.result == "ok"


@pytest.mark.asyncio
async def test_async_sync_tool_started_event_streams_before_completion():
    model = DummyModel(id="dummy-model")

    @tool
    def slow_sync_tool() -> str:
        time.sleep(0.2)
        return "done"

    iterator = model.arun_function_calls(
        [FunctionCall(function=slow_sync_tool, arguments={}, call_id="call_slow_sync")],
        [],
    )
    started_at = asyncio.get_running_loop().time()
    started = await iterator.__anext__()
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.1
    assert started.event == ModelResponseEvent.tool_call_started.value
    remaining = [response async for response in iterator]
    assert remaining[-1].event == ModelResponseEvent.tool_call_completed.value


@pytest.mark.asyncio
async def test_async_mixed_sync_async_results_preserve_tool_call_order_without_pause():
    model = DummyModel(id="dummy-model")

    @tool
    def sync_first() -> str:
        return "sync"

    @tool
    async def async_second() -> str:
        return "async"

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=sync_first, arguments={}, call_id="call_sync"),
                FunctionCall(function=async_second, arguments={}, call_id="call_async"),
            ],
            function_call_results,
        )
    ]

    completed_call_ids = [
        response.tool_executions[0].tool_call_id
        for response in responses
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_completed.value
    ]
    assert completed_call_ids == ["call_sync", "call_async"]
    assert [message.tool_call_id for message in function_call_results] == ["call_sync", "call_async"]


def test_sync_static_pause_stops_later_sibling():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool(requires_confirmation=True)
    def protected_tool() -> str:
        side_effects.append("protected-ran")
        return "protected"

    @tool
    def later_tool() -> str:
        side_effects.append("later-ran")
        return "later"

    function_call_results = []
    responses = list(
        model.run_function_calls(
            [
                FunctionCall(function=protected_tool, arguments={}, call_id="call_protected"),
                FunctionCall(function=later_tool, arguments={}, call_id="call_later"),
            ],
            function_call_results,
        )
    )

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_protected"
    assert function_call_results == []


def test_static_pause_flags_are_collapsed_into_one_requirement():
    model = DummyModel(id="dummy-model")

    def protected_external_tool() -> str:
        return "protected"

    protected_external_function = Function(
        name="protected_external_tool",
        entrypoint=protected_external_tool,
        requires_confirmation=True,
        external_execution=True,
    )

    responses = list(
        model.run_function_calls(
            [
                FunctionCall(
                    function=protected_external_function,
                    arguments={"path": ".env"},
                    call_id="call_combined_pause",
                )
            ],
            [],
        )
    )

    # Static pause emits the same "started → paused" pair dynamic pause does
    assert len(responses) == 2
    assert responses[0].event == ModelResponseEvent.tool_call_started.value
    paused = responses[1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    assert len(paused.tool_executions) == 1
    assert paused.tool_executions[0].requires_confirmation is True
    assert paused.tool_executions[0].external_execution_required is True

    run_response = RunOutput(run_id="run_1")
    _upsert_tool_executions(run_response, paused.tool_executions)
    _append_paused_requirements(run_response, paused.tool_executions)

    assert run_response.requirements is not None
    assert len(run_response.requirements) == 1
    requirement_tool = run_response.requirements[0].tool_execution
    assert requirement_tool is not None
    assert requirement_tool.tool_call_id == "call_combined_pause"
    assert requirement_tool.requires_confirmation is True
    assert requirement_tool.external_execution_required is True


def test_upsert_tool_executions_updates_existing_tool_in_place():
    existing_tool = ToolExecution(
        tool_call_id="call_in_place",
        tool_name="protected_tool",
        tool_args={"path": ".env"},
        requires_confirmation=True,
    )
    run_response = RunOutput(run_id="run_1", tools=[existing_tool])
    updated_tool = ToolExecution(
        tool_call_id="call_in_place",
        tool_name="protected_tool",
        tool_args={"path": ".env"},
        result="updated",
        requires_confirmation=False,
        confirmed=True,
        resume_metadata={"approver": "admin"},
    )

    _upsert_tool_executions(run_response, [updated_tool])

    assert run_response.tools is not None
    assert run_response.tools == [existing_tool]
    assert run_response.tools[0] is existing_tool
    assert id(run_response.tools[0]) == id(existing_tool)
    assert existing_tool.result == "updated"
    assert existing_tool.requires_confirmation is False
    assert existing_tool.confirmed is True
    assert existing_tool.resume_metadata == {"approver": "admin"}


def _snapshot_update_run_context():
    run_response = RunOutput(
        run_id="run_1",
        tools=[
            ToolExecution(
                tool_call_id="call_first",
                tool_name="first_tool",
                tool_args={},
                requires_confirmation=True,
                confirmed=True,
            ),
        ],
    )

    @tool
    def first_tool() -> str:
        return "first"

    return run_response, RunMessages(), first_tool, []


def _assert_appended_sibling_was_not_processed(run_response: RunOutput, calls: list[str]):
    assert calls == ["call_first"]
    assert run_response.tools is not None
    assert [tool_execution.tool_call_id for tool_execution in run_response.tools] == ["call_first", "call_appended"]


def test_handle_tool_call_updates_uses_snapshot_when_run_tool_appends_sibling(monkeypatch: pytest.MonkeyPatch):
    model = DummyModel(id="dummy-model")
    agent = Agent(model=model)
    run_response, run_messages, first_tool, calls = _snapshot_update_run_context()

    def fake_run_tool(agent, run_response, run_messages, tool, **kwargs):
        calls.append(tool.tool_call_id)
        if tool.tool_call_id == "call_first":
            run_response.tools.append(
                ToolExecution(
                    tool_call_id="call_appended",
                    tool_name="first_tool",
                    tool_args={},
                    requires_confirmation=True,
                    confirmed=True,
                )
            )
        return iter(())

    monkeypatch.setattr("agno.agent._tools.run_tool", fake_run_tool)

    handle_tool_call_updates(agent, run_response, run_messages, [first_tool])

    _assert_appended_sibling_was_not_processed(run_response, calls)


def test_handle_tool_call_updates_stream_uses_snapshot_when_run_tool_appends_sibling(
    monkeypatch: pytest.MonkeyPatch,
):
    model = DummyModel(id="dummy-model")
    agent = Agent(model=model)
    run_response, run_messages, first_tool, calls = _snapshot_update_run_context()

    def fake_run_tool(agent, run_response, run_messages, tool, **kwargs):
        calls.append(tool.tool_call_id)
        if tool.tool_call_id == "call_first":
            run_response.tools.append(
                ToolExecution(
                    tool_call_id="call_appended",
                    tool_name="first_tool",
                    tool_args={},
                    requires_confirmation=True,
                    confirmed=True,
                )
            )
        return iter(())

    monkeypatch.setattr("agno.agent._tools.run_tool", fake_run_tool)

    list(handle_tool_call_updates_stream(agent, run_response, run_messages, [first_tool]))

    _assert_appended_sibling_was_not_processed(run_response, calls)


@pytest.mark.asyncio
async def test_ahandle_tool_call_updates_uses_snapshot_when_arun_tool_appends_sibling(
    monkeypatch: pytest.MonkeyPatch,
):
    model = DummyModel(id="dummy-model")
    agent = Agent(model=model)
    run_response, run_messages, first_tool, calls = _snapshot_update_run_context()

    async def fake_arun_tool(agent, run_response, run_messages, tool, **kwargs):
        calls.append(tool.tool_call_id)
        if tool.tool_call_id == "call_first":
            run_response.tools.append(
                ToolExecution(
                    tool_call_id="call_appended",
                    tool_name="first_tool",
                    tool_args={},
                    requires_confirmation=True,
                    confirmed=True,
                )
            )
        if False:
            yield None

    monkeypatch.setattr("agno.agent._tools.arun_tool", fake_arun_tool)

    await ahandle_tool_call_updates(agent, run_response, run_messages, [first_tool])

    _assert_appended_sibling_was_not_processed(run_response, calls)


@pytest.mark.asyncio
async def test_ahandle_tool_call_updates_stream_uses_snapshot_when_arun_tool_appends_sibling(
    monkeypatch: pytest.MonkeyPatch,
):
    model = DummyModel(id="dummy-model")
    agent = Agent(model=model)
    run_response, run_messages, first_tool, calls = _snapshot_update_run_context()

    async def fake_arun_tool(agent, run_response, run_messages, tool, **kwargs):
        calls.append(tool.tool_call_id)
        if tool.tool_call_id == "call_first":
            run_response.tools.append(
                ToolExecution(
                    tool_call_id="call_appended",
                    tool_name="first_tool",
                    tool_args={},
                    requires_confirmation=True,
                    confirmed=True,
                )
            )
        if False:
            yield None

    monkeypatch.setattr("agno.agent._tools.arun_tool", fake_arun_tool)

    events = [
        event async for event in ahandle_tool_call_updates_stream(agent, run_response, run_messages, [first_tool])
    ]

    assert events == []
    _assert_appended_sibling_was_not_processed(run_response, calls)


@pytest.mark.asyncio
async def test_async_static_pause_stops_later_sync_sibling():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool(external_execution=True)
    def external_tool() -> str:
        side_effects.append("external-ran")
        return "external"

    @tool
    def sync_sibling() -> str:
        side_effects.append("sync-ran")
        return "sync"

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=external_tool, arguments={}, call_id="call_external"),
                FunctionCall(function=sync_sibling, arguments={}, call_id="call_sync"),
            ],
            function_call_results,
        )
    ]

    paused = [response for response in responses if response.event == ModelResponseEvent.tool_call_paused.value]
    completed = [response for response in responses if response.event == ModelResponseEvent.tool_call_completed.value]

    assert side_effects == []
    assert paused[0].tool_executions[0].tool_call_id == "call_external"
    assert completed == []
    assert function_call_results == []


@pytest.mark.asyncio
async def test_async_static_user_input_pause_stops_later_sibling():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool(requires_user_input=True, user_input_fields=["name"])
    def needs_input(name: str) -> str:
        side_effects.append("input-ran")
        return name

    @tool
    async def later_tool() -> str:
        side_effects.append("later-ran")
        return "later"

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=needs_input, arguments={}, call_id="call_input"),
                FunctionCall(function=later_tool, arguments={}, call_id="call_later"),
            ],
            [],
        )
    ]

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_input"
    assert responses[-1].tool_executions[0].requires_user_input is True


@pytest.mark.asyncio
async def test_async_static_pause_stops_later_dynamic_pause():
    model = DummyModel(id="dummy-model")

    @tool(external_execution=True)
    def external_tool() -> str:
        return "external"

    @tool
    async def pause_async():
        raise ApprovalRequired({"reason": "dynamic"})

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=external_tool, arguments={}, call_id="call_external"),
                FunctionCall(function=pause_async, arguments={}, call_id="call_pause"),
            ],
            [],
        )
    ]

    paused_call_ids = [
        response.tool_executions[0].tool_call_id
        for response in responses
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_paused.value
    ]
    assert paused_call_ids == ["call_external"]


@pytest.mark.asyncio
async def test_async_generator_dynamic_pause_is_emitted_without_waiting_for_other_generators():
    model = DummyModel(id="dummy-model")

    @tool
    async def pause_quickly():
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"reason": "async_generator"})
        yield "unreachable"

    @tool
    async def slow_generator():
        await asyncio.sleep(1)
        yield "slow"

    started_at = asyncio.get_running_loop().time()
    responses = []
    async for response in model.arun_function_calls(
        [
            FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
            FunctionCall(function=slow_generator, arguments={}, call_id="call_slow"),
        ],
        [],
    ):
        responses.append(response)
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_paused.value:
            break

    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.5
    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    assert paused.tool_executions[0].tool_call_id == "call_pause"
    assert paused.tool_executions[0].metadata == {"reason": "async_generator"}


@pytest.mark.asyncio
async def test_async_generator_dynamic_pause_cancels_sibling_before_yielding_pause():
    model = DummyModel(id="dummy-model")
    side_effects = []
    cancelled = asyncio.Event()

    async def post_hook(**kwargs):
        side_effects.append("slow-post")

    @tool
    async def pause_quickly():
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"reason": "async_generator"})
        yield "unreachable"

    @tool(post_hook=post_hook)
    async def slow_generator():
        try:
            await asyncio.sleep(1)
            side_effects.append("slow-yielded")
            yield "slow"
        except asyncio.CancelledError:
            side_effects.append("slow-cancelled")
            cancelled.set()
            raise

    responses = []
    async for response in model.arun_function_calls(
        [
            FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
            FunctionCall(function=slow_generator, arguments={}, call_id="call_slow"),
        ],
        [],
    ):
        responses.append(response)
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_paused.value:
            break

    await asyncio.wait_for(cancelled.wait(), timeout=0.5)

    assert "slow-yielded" not in side_effects
    assert "slow-cancelled" in side_effects
    assert "slow-post" not in side_effects
    paused = responses[-1]
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions is not None
    assert paused.tool_executions[0].tool_call_id == "call_pause"


@pytest.mark.asyncio
async def test_async_generator_dynamic_pause_completes_cancelled_started_sibling():
    model = DummyModel(id="dummy-model")
    function_call_results = []

    @tool
    async def pause_quickly():
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"reason": "async_generator"})
        yield "unreachable"

    @tool
    async def slow_generator():
        await asyncio.sleep(1)
        yield "slow"

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
                FunctionCall(function=slow_generator, arguments={}, call_id="call_slow"),
            ],
            function_call_results,
        )
    ]

    started_ids = {
        response.tool_executions[0].tool_call_id
        for response in responses
        if response.event == ModelResponseEvent.tool_call_started.value
    }
    closed_ids = {
        response.tool_executions[0].tool_call_id
        for response in responses
        if response.event in (ModelResponseEvent.tool_call_completed.value, ModelResponseEvent.tool_call_paused.value)
    }

    assert started_ids == {"call_pause", "call_slow"}
    assert started_ids == closed_ids
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert any(
        response.event == ModelResponseEvent.tool_call_completed.value
        and response.tool_executions[0].tool_call_id == "call_slow"
        and response.tool_executions[0].tool_call_error is True
        for response in responses
    )
    assert len(function_call_results) == 1
    assert function_call_results[0].tool_call_id == "call_slow"
    assert function_call_results[0].tool_call_error is True


@pytest.mark.asyncio
async def test_async_generator_object_sibling_is_completed_before_dynamic_pause():
    model = DummyModel(id="dummy-model")

    @tool
    async def ready_generator():
        await asyncio.sleep(0.2)
        yield "slow"

    @tool
    async def pause_quickly():
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"reason": "pause"})

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=ready_generator, arguments={}, call_id="call_generator"),
                FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
            ],
            function_call_results,
        )
    ]

    started_ids = {
        response.tool_executions[0].tool_call_id
        for response in responses
        if response.event == ModelResponseEvent.tool_call_started.value
    }
    closed_ids = {
        response.tool_executions[0].tool_call_id
        for response in responses
        if response.event in (ModelResponseEvent.tool_call_completed.value, ModelResponseEvent.tool_call_paused.value)
    }

    assert started_ids == {"call_generator", "call_pause"}
    assert started_ids == closed_ids
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert len(function_call_results) == 1
    assert function_call_results[0].tool_call_id == "call_generator"
    assert function_call_results[0].tool_call_error is True


@pytest.mark.asyncio
async def test_async_generator_dynamic_pause_waits_for_regular_async_sibling():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool
    async def pause_generator():
        await asyncio.sleep(0.01)
        raise ApprovalRequired({"reason": "generator_pause"})
        yield "unreachable"

    @tool
    async def regular_async_sibling() -> str:
        await asyncio.sleep(0.1)
        side_effects.append("regular-finished")
        return "regular"

    function_call_results = []
    responses = []
    async for response in model.arun_function_calls(
        [
            FunctionCall(function=pause_generator, arguments={}, call_id="call_pause_generator"),
            FunctionCall(function=regular_async_sibling, arguments={}, call_id="call_regular"),
        ],
        function_call_results,
    ):
        responses.append(response)
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_paused.value:
            break

    assert side_effects == ["regular-finished"]
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause_generator"
    assert any(
        response.event == ModelResponseEvent.tool_call_completed.value
        and response.tool_executions[0].tool_call_id == "call_regular"
        for response in responses
    )
    assert len(function_call_results) == 1
    assert function_call_results[0].tool_call_id == "call_regular"


@pytest.mark.asyncio
async def test_async_pause_waits_for_sync_sibling_with_async_hook_to_finish():
    model = DummyModel(id="dummy-model")
    side_effects = []

    async def post_hook(**kwargs):
        pass

    @tool(post_hook=post_hook)
    def slow_sync_tool() -> str:
        time.sleep(0.1)
        side_effects.append("slow-finished")
        return "slow"

    @tool
    async def pause_quickly():
        await asyncio.sleep(0.01)
        raise ApprovalRequired({"reason": "pause"})

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=slow_sync_tool, arguments={}, call_id="call_slow"),
                FunctionCall(function=pause_quickly, arguments={}, call_id="call_pause"),
            ],
            function_call_results,
        )
    ]

    assert side_effects == ["slow-finished"]
    assert function_call_results[0].tool_call_id == "call_slow"
    assert function_call_results[0].tool_call_error is not True
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value


@pytest.mark.asyncio
async def test_async_completed_sibling_is_recorded_before_dynamic_pause():
    model = DummyModel(id="dummy-model")

    @tool
    async def fast_tool() -> str:
        await asyncio.sleep(0.01)
        return "fast-result"

    @tool
    async def pause_later() -> str:
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"reason": "pause_later"})

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=fast_tool, arguments={}, call_id="call_fast"),
                FunctionCall(function=pause_later, arguments={}, call_id="call_pause"),
            ],
            function_call_results,
        )
    ]

    completed = [
        response
        for response in responses
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_completed.value
    ]
    paused = responses[-1]
    assert completed
    assert completed[0].tool_executions[0].tool_call_id == "call_fast"
    assert function_call_results[0].tool_call_id == "call_fast"
    assert paused.event == ModelResponseEvent.tool_call_paused.value
    assert paused.tool_executions[0].tool_call_id == "call_pause"


@pytest.mark.asyncio
async def test_async_completed_generator_sibling_is_recorded_before_dynamic_pause():
    model = DummyModel(id="dummy-model")

    @tool
    async def fast_generator():
        yield "fast-result"

    @tool
    async def pause_later():
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"reason": "pause_later"})
        yield "unreachable"

    function_call_results = []
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=fast_generator, arguments={}, call_id="call_fast_generator"),
                FunctionCall(function=pause_later, arguments={}, call_id="call_pause"),
            ],
            function_call_results,
        )
    ]

    completed = [
        response
        for response in responses
        if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_completed.value
    ]
    assert completed
    assert completed[0].tool_executions[0].tool_call_id == "call_fast_generator"
    assert function_call_results[0].tool_call_id == "call_fast_generator"
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"


@pytest.mark.asyncio
async def test_async_sync_generator_pause_prevents_later_tools():
    model = DummyModel(id="dummy-model")
    side_effects = []

    @tool
    def pause_generator():
        raise ApprovalRequired({"reason": "sync_generator"})
        yield "unreachable"

    @tool
    async def later_tool() -> str:
        side_effects.append("completed")
        return "later"

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_generator, arguments={}, call_id="call_pause"),
                FunctionCall(function=later_tool, arguments={}, call_id="call_later"),
            ],
            [],
        )
    ]

    assert side_effects == []
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"
    assert all(
        not (
            isinstance(response, ModelResponse)
            and response.event == ModelResponseEvent.tool_call_started.value
            and response.tool_executions[0].tool_call_id == "call_later"
        )
        for response in responses
    )


@pytest.mark.asyncio
async def test_async_generator_phase_observes_run_cancellation():
    model = DummyModel(id="dummy-model")
    run_id = "run_cancel_async_generator"
    await aregister_run(run_id)

    @tool
    async def silent_generator():
        await asyncio.sleep(10)
        yield "late"

    try:
        iterator = model.arun_function_calls(
            [FunctionCall(function=silent_generator, arguments={}, call_id="call_silent")],
            [],
            run_id=run_id,
        )
        started = await iterator.__anext__()
        assert started.event == ModelResponseEvent.tool_call_started.value

        await acancel_run(run_id)

        with pytest.raises(RunCancelledException):
            await iterator.__anext__()
    finally:
        await acleanup_run(run_id)


@pytest.mark.asyncio
async def test_async_function_task_phase_observes_run_cancellation():
    model = DummyModel(id="dummy-model")
    run_id = "run_cancel_async_function"
    await aregister_run(run_id)

    @tool
    async def slow_tool():
        await asyncio.sleep(10)
        return "late"

    try:
        iterator = model.arun_function_calls(
            [FunctionCall(function=slow_tool, arguments={}, call_id="call_slow")],
            [],
            run_id=run_id,
        )
        started = await iterator.__anext__()
        assert started.event == ModelResponseEvent.tool_call_started.value

        await acancel_run(run_id)

        with pytest.raises(RunCancelledException):
            await iterator.__anext__()
    finally:
        await acleanup_run(run_id)


@pytest.mark.asyncio
async def test_async_sync_tool_run_cancellation_does_not_mutate_results_after_cancel():
    model = DummyModel(id="dummy-model")
    run_id = "run_cancel_sync_function"
    function_call_results = []
    await aregister_run(run_id)

    @tool
    def slow_sync_tool():
        time.sleep(0.2)
        return "done"

    try:
        iterator = model.arun_function_calls(
            [FunctionCall(function=slow_sync_tool, arguments={}, call_id="call_slow_sync")],
            function_call_results,
            run_id=run_id,
        )
        started = await iterator.__anext__()
        assert started.event == ModelResponseEvent.tool_call_started.value

        await acancel_run(run_id)

        with pytest.raises(RunCancelledException):
            await iterator.__anext__()

        await asyncio.sleep(0.3)
        assert function_call_results == []
    finally:
        await acleanup_run(run_id)


@pytest.mark.asyncio
async def test_async_dynamic_pause_completes_timeout_started_sibling(monkeypatch):
    model = DummyModel(id="dummy-model")
    monkeypatch.setattr("agno.models.base._TOOL_CALL_BATCH_SETTLE_TIMEOUT_SECONDS", 0.05)
    function_call_results = []
    pause_ready = asyncio.Event()

    @tool
    async def pause_tool() -> str:
        await pause_ready.wait()
        raise ApprovalRequired({"reason": "pause"})

    @tool
    async def slow_tool() -> str:
        pause_ready.set()
        await asyncio.sleep(10)
        return "late"

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_tool, arguments={}, call_id="call_pause"),
                FunctionCall(function=slow_tool, arguments={}, call_id="call_slow"),
            ],
            function_call_results,
        )
    ]

    started_ids = {
        response.tool_executions[0].tool_call_id
        for response in responses
        if response.event == ModelResponseEvent.tool_call_started.value
    }
    closed_ids = {
        response.tool_executions[0].tool_call_id
        for response in responses
        if response.event in (ModelResponseEvent.tool_call_completed.value, ModelResponseEvent.tool_call_paused.value)
    }

    assert started_ids == {"call_pause", "call_slow"}
    assert started_ids == closed_ids
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"
    assert any(
        response.event == ModelResponseEvent.tool_call_completed.value
        and response.tool_executions[0].tool_call_id == "call_slow"
        and response.tool_executions[0].tool_call_error is True
        for response in responses
    )
    assert len(function_call_results) == 1
    assert function_call_results[0].tool_call_id == "call_slow"
    assert function_call_results[0].tool_call_error is True


@pytest.mark.asyncio
async def test_async_dynamic_pause_timeout_cancels_sibling_before_returning(monkeypatch):
    model = DummyModel(id="dummy-model")
    monkeypatch.setattr("agno.models.base._TOOL_CALL_BATCH_SETTLE_TIMEOUT_SECONDS", 0.05)
    pause_ready = asyncio.Event()
    cancelled = asyncio.Event()

    @tool
    async def pause_tool() -> str:
        await pause_ready.wait()
        raise ApprovalRequired({"reason": "pause"})

    @tool
    async def cancellation_resistant_tool() -> str:
        pause_ready.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.sleep(0.2)
            raise
        return "late"

    started_at = asyncio.get_running_loop().time()
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_tool, arguments={}, call_id="call_pause"),
                FunctionCall(function=cancellation_resistant_tool, arguments={}, call_id="call_slow"),
            ],
            [],
        )
    ]
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed >= 0.2
    assert cancelled.is_set()
    assert responses[-1].event == ModelResponseEvent.tool_call_paused.value
    assert responses[-1].tool_executions[0].tool_call_id == "call_pause"


@pytest.mark.asyncio
async def test_async_dynamic_pause_settlement_observes_run_cancellation(monkeypatch):
    model = DummyModel(id="dummy-model")
    monkeypatch.setattr("agno.models.base._TOOL_CALL_BATCH_SETTLE_TIMEOUT_SECONDS", 1.0)
    run_id = "run_cancel_during_pause_settlement"
    pause_ready = asyncio.Event()
    await aregister_run(run_id)

    @tool
    async def pause_tool() -> str:
        await pause_ready.wait()
        raise ApprovalRequired({"reason": "pause"})

    @tool
    async def slow_tool() -> str:
        pause_ready.set()
        await asyncio.sleep(10)
        return "late"

    async def cancel_run_after_pause() -> None:
        await pause_ready.wait()
        await asyncio.sleep(0.05)
        await acancel_run(run_id)

    cancel_task = asyncio.create_task(cancel_run_after_pause())
    try:
        with pytest.raises(RunCancelledException):
            [
                response
                async for response in model.arun_function_calls(
                    [
                        FunctionCall(function=pause_tool, arguments={}, call_id="call_pause"),
                        FunctionCall(function=slow_tool, arguments={}, call_id="call_slow"),
                    ],
                    [],
                    run_id=run_id,
                )
            ]
    finally:
        await cancel_task
        await acleanup_run(run_id)


@pytest.mark.asyncio
async def test_async_simultaneous_dynamic_pauses_are_emitted_together():
    model = DummyModel(id="dummy-model")
    barrier = asyncio.Event()

    @tool
    async def pause_a() -> str:
        await barrier.wait()
        raise ApprovalRequired({"tool": "a"})

    @tool
    async def pause_b() -> str:
        await barrier.wait()
        raise ApprovalRequired({"tool": "b"})

    async def release_barrier():
        await asyncio.sleep(0)
        barrier.set()

    release_task = asyncio.create_task(release_barrier())
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_a, arguments={}, call_id="call_a"),
                FunctionCall(function=pause_b, arguments={}, call_id="call_b"),
            ],
            [],
        )
    ]
    await release_task

    paused = [response for response in responses if response.event == ModelResponseEvent.tool_call_paused.value]

    assert len(paused) == 1
    assert [tool.tool_call_id for tool in paused[0].tool_executions] == ["call_a", "call_b"]
    assert all(tool.approval_type == "required" for tool in paused[0].tool_executions)


@pytest.mark.asyncio
async def test_async_settled_sync_hook_pause_is_emitted_with_fast_async_pause():
    model = DummyModel(id="dummy-model")

    @tool
    async def fast_pause() -> str:
        await asyncio.sleep(0.01)
        raise ApprovalRequired({"tool": "fast"})

    async def slow_pre_hook(**kwargs):
        await asyncio.sleep(0.05)
        raise ApprovalRequired({"tool": "slow"})

    @tool(pre_hook=slow_pre_hook)
    def slow_pause() -> str:
        return "should not run"

    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=slow_pause, arguments={}, call_id="call_slow"),
                FunctionCall(function=fast_pause, arguments={}, call_id="call_fast"),
            ],
            [],
        )
    ]

    paused = [response for response in responses if response.event == ModelResponseEvent.tool_call_paused.value]

    assert len(paused) == 1
    assert {tool.tool_call_id for tool in paused[0].tool_executions} == {"call_slow", "call_fast"}


@pytest.mark.asyncio
async def test_async_simultaneous_generator_dynamic_pauses_are_emitted_together():
    model = DummyModel(id="dummy-model")
    barrier = asyncio.Event()

    @tool
    async def pause_generator_a():
        await barrier.wait()
        raise ApprovalRequired({"tool": "a"})
        yield "unreachable"

    @tool
    async def pause_generator_b():
        await barrier.wait()
        raise ApprovalRequired({"tool": "b"})
        yield "unreachable"

    async def release_barrier():
        await asyncio.sleep(0)
        barrier.set()

    release_task = asyncio.create_task(release_barrier())
    responses = [
        response
        async for response in model.arun_function_calls(
            [
                FunctionCall(function=pause_generator_a, arguments={}, call_id="call_a"),
                FunctionCall(function=pause_generator_b, arguments={}, call_id="call_b"),
            ],
            [],
        )
    ]
    await release_task

    paused = [response for response in responses if response.event == ModelResponseEvent.tool_call_paused.value]

    assert len(paused) == 1
    assert {tool.tool_call_id for tool in paused[0].tool_executions} == {"call_a", "call_b"}


def test_response_adds_requirement_for_each_paused_tool_execution():
    model = DummyModel(id="dummy-model")
    run_response = RunOutput(run_id="run-1", session_id="session-1")
    messages = [Message(role="user", content="pause both")]
    provider_response = ModelResponse(
        role="assistant",
        tool_calls=[
            {"id": "call_a", "type": "function", "function": {"name": "pause_a", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "pause_b", "arguments": "{}"}},
        ],
    )
    responses = [
        ModelResponse(
            tool_executions=[
                ToolExecution(tool_call_id="call_a", tool_name="pause_a", requires_confirmation=True),
                ToolExecution(tool_call_id="call_b", tool_name="pause_b", requires_confirmation=True),
            ],
            event=ModelResponseEvent.tool_call_paused.value,
        )
    ]
    model.run_function_calls = lambda **kwargs: iter(responses)  # type: ignore[method-assign]
    model._prepare_function_calls = lambda **kwargs: []  # type: ignore[method-assign]
    model._invoke_with_retry = lambda **kwargs: provider_response  # type: ignore[method-assign]

    model_response = model.response(
        messages=messages,
        run_response=run_response,
        tools=[],
    )
    assert model_response.tool_executions is not None
    assert [tool.tool_call_id for tool in model_response.tool_executions] == ["call_a", "call_b"]

    update_run_response(
        Agent(model=model),
        model_response=model_response,
        run_response=run_response,
        run_messages=RunMessages(messages=messages),
    )

    assert run_response.requirements is not None
    assert len(run_response.requirements) == 2
    assert [req.tool_execution.tool_call_id for req in run_response.requirements] == ["call_a", "call_b"]


def test_agent_update_run_response_adds_requirement_for_paused_tool_execution():
    agent = Agent(model=DummyModel(id="dummy-model"))
    run_response = RunOutput(run_id="run-1", session_id="session-1")
    run_messages = RunMessages(messages=[Message(role="user", content="pause")])
    paused_tool = ToolExecution(tool_call_id="call-1", tool_name="protected", requires_confirmation=True)

    update_run_response(
        agent,
        model_response=ModelResponse(tool_executions=[paused_tool]),
        run_response=run_response,
        run_messages=run_messages,
    )

    assert run_response.requirements is not None
    assert run_response.requirements[0].tool_execution is paused_tool


def test_model_response_does_not_duplicate_paused_requirements_before_update():
    model = DummyModel(id="dummy-model")
    run_response = RunOutput(run_id="run-1", session_id="session-1")
    messages = [Message(role="user", content="pause")]
    provider_response = ModelResponse(
        role="assistant",
        tool_calls=[{"id": "call_a", "type": "function", "function": {"name": "pause_a", "arguments": "{}"}}],
    )
    paused_tool = ToolExecution(tool_call_id="call_a", tool_name="pause_a", requires_confirmation=True)
    responses = [ModelResponse(tool_executions=[paused_tool], event=ModelResponseEvent.tool_call_paused.value)]
    model.run_function_calls = lambda **kwargs: iter(responses)  # type: ignore[method-assign]
    model._prepare_function_calls = lambda **kwargs: []  # type: ignore[method-assign]
    model._invoke_with_retry = lambda **kwargs: provider_response  # type: ignore[method-assign]

    model_response = model.response(messages=messages, run_response=run_response, tools=[])
    update_run_response(
        Agent(model=model),
        model_response=model_response,
        run_response=run_response,
        run_messages=RunMessages(messages=messages),
    )

    assert run_response.requirements is not None
    assert len(run_response.requirements) == 1
    assert run_response.requirements[0].tool_execution is paused_tool


def test_model_response_cache_is_disabled_when_tools_are_available(tmp_path):
    model = DummyModel(id="dummy-model", cache_response=True, cache_dir=str(tmp_path))
    messages = [Message(role="user", content="call the tool")]

    @tool
    def cached_guard() -> str:
        return "done"

    saved_keys = []
    model._get_cached_model_response = lambda cache_key: {"result": {"content": "cached"}}  # type: ignore[method-assign]
    model._save_model_response_to_cache = lambda cache_key, result, is_streaming=False: saved_keys.append(cache_key)  # type: ignore[method-assign]
    model._invoke_with_retry = lambda **kwargs: ModelResponse(role="assistant", content="fresh")  # type: ignore[method-assign]

    with_tools = model.response(messages=list(messages), tools=[cached_guard])
    without_tools = model.response(messages=list(messages), tools=[])

    assert with_tools.content == "fresh"
    assert without_tools.content == "cached"
    assert saved_keys == []
