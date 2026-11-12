import asyncio
from dataclasses import dataclass

import pytest

from agno.agent import Agent, _response, _run
from agno.agent._messages import _history_for_compaction, _history_for_run
from agno.compaction import Compaction
from agno.db.in_memory import InMemoryDb
from agno.exceptions import ModelProviderError, RunCancelledException
from agno.models.base import Model
from agno.models.fallback import FallbackConfig
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput
from agno.run.base import RunStatus
from agno.run.cancel import cancel_run
from agno.run.messages import RunMessages
from agno.session import AgentSession
from agno.tools.function import Function


def checkpoint() -> str:
    return "ok"


@pytest.mark.parametrize("include_cancelled", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_compaction_history_preserves_cancelled_history_policy(include_cancelled, compact):
    agent = Agent(
        model=ScriptedModel(id="scripted"),
        include_cancelled_history=include_cancelled,
        compaction=Compaction(compact_at_tokens=1000) if compact else None,
        num_history_runs=10,
        telemetry=False,
    )
    session = AgentSession(
        session_id="history",
        runs=[
            RunOutput(
                run_id=status.value,
                status=status,
                messages=[Message(role="assistant", content=status.value)],
            )
            for status in (RunStatus.completed, RunStatus.cancelled, RunStatus.paused, RunStatus.error)
        ],
    )
    expected = [RunStatus.completed.value]
    if include_cancelled:
        expected.append(RunStatus.cancelled.value)
    history, _, _ = _history_for_run(agent, session, None, None)
    assert [message.content for message in history] == expected
    assert [message.content for message in _history_for_compaction(agent, session)] == expected


@dataclass
class ScriptedModel(Model):
    failure: str = "cancel"
    request_count: int = 0

    def invoke(self, *args, **kwargs):
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs):
        raise NotImplementedError

    def _parse_provider_response(self, response, **kwargs):
        return response

    def _parse_provider_response_delta(self, response):
        return response

    def invoke_stream(self, **kwargs):
        if self.failure == "provider":
            raise ModelProviderError("unavailable", status_code=503)
        self.request_count += 1
        if self.request_count <= 3:
            text = ["计划阶段。", "调研完成。", None][self.request_count - 1]
            if text:
                yield ModelResponse(content=text)
            yield ModelResponse(
                tool_calls=[
                    {
                        "id": f"call-{self.request_count}",
                        "type": "function",
                        "function": {"name": "checkpoint", "arguments": "{}"},
                    }
                ]
            )
        else:
            if self.failure == "flag":
                cancel_run(kwargs["run_response"].run_id)
            yield ModelResponse(content="已")
            if self.failure == "cancel":
                raise RunCancelledException("cancelled")
            if self.failure == "task":
                raise asyncio.CancelledError
            if self.failure == "keyboard":
                raise KeyboardInterrupt

    async def ainvoke_stream(self, **kwargs):
        for event in self.invoke_stream(**kwargs):
            yield event


def make_stream_state(failure, fallback=False):
    agent = Agent(model=ScriptedModel(id="scripted", failure=failure), telemetry=False)
    if fallback:
        agent.fallback_config = FallbackConfig(on_error=[agent.model])
        agent.model = ScriptedModel(id="primary", failure="provider")
    run = RunOutput(run_id="cancel-probe")
    messages = RunMessages(messages=[Message(role="user", content="go")])
    tool = Function.from_callable(checkpoint)
    return (
        agent,
        run,
        messages,
        dict(
            agent=agent,
            session=AgentSession(session_id="test"),
            run_response=run,
            run_messages=messages,
            tools=[tool],
        ),
    )


def assert_messages(run):
    assistant = [m for m in run.messages if m.role == "assistant"]
    assert [m.content for m in assistant] == ["计划阶段。", "调研完成。", None, "已"]
    assert [m.role for m in run.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert run.content == "计划阶段。调研完成。已"


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "keyboard", "close", "complete"])
def test_sync_stream_cancellation_keeps_message_boundaries(failure, fallback):
    _, run, messages, kwargs = make_stream_state(failure, fallback)
    stream = _response.handle_model_response_stream(**kwargs)
    try:
        for event in stream:
            if failure == "close" and getattr(event, "content", None) == "已":
                stream.close()
                break
    except (RunCancelledException, KeyboardInterrupt):
        pass
    _run._handle_run_cancellation(run, KeyboardInterrupt(), messages)
    assert_messages(run)


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "task", "close", "complete"])
async def test_async_stream_cancellation_keeps_message_boundaries(failure, fallback):
    _, run, messages, kwargs = make_stream_state(failure, fallback)
    stream = _response.ahandle_model_response_stream(**kwargs)
    try:
        async for event in stream:
            if failure == "close" and getattr(event, "content", None) == "已":
                await stream.aclose()
                break
    except (RunCancelledException, asyncio.CancelledError):
        pass
    _run._handle_run_cancellation(run, KeyboardInterrupt(), messages)
    assert_messages(run)


@pytest.mark.parametrize("failure", ["cancel", "flag", "complete"])
def test_sync_agent_persists_separate_messages(failure):
    agent = Agent(
        model=ScriptedModel(id="scripted", failure=failure), tools=[checkpoint], db=InMemoryDb(), telemetry=False
    )
    list(agent.run("go", stream=True))
    run = agent.get_last_run_output()
    assert run is not None
    assert_messages(run)
    assert run.status == ("COMPLETED" if failure == "complete" else "CANCELLED")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "flag", "complete"])
async def test_async_agent_persists_separate_messages(failure):
    agent = Agent(
        model=ScriptedModel(id="scripted", failure=failure), tools=[checkpoint], db=InMemoryDb(), telemetry=False
    )
    async for _ in agent.arun("go", stream=True):
        pass
    run = await agent.aget_last_run_output()
    assert run is not None
    assert_messages(run)
    assert run.status == ("COMPLETED" if failure == "complete" else "CANCELLED")
