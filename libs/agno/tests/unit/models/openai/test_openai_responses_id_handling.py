from typing import Any, Dict, List, Optional

import pytest

from agno.models.base import MessageData
from agno.models.message import Message
from agno.models.openai.responses import OpenAIResponses
from agno.models.response import ModelResponse, ModelResponseEvent


class _FakeError:
    def __init__(self, message: str):
        self.message = message


class _FakeOutputFunctionCall:
    def __init__(self, *, _id: str, call_id: Optional[str], name: str, arguments: str):
        self.type = "function_call"
        self.id = _id
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class _FakeResponse:
    def __init__(
        self,
        *,
        _id: str,
        output: List[Any],
        output_text: str = "",
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[_FakeError] = None,
    ):
        self.id = _id
        self.output = output
        self.output_text = output_text
        self.usage = usage
        self.error = error


class _FakeStreamItem:
    def __init__(self, *, _id: str, call_id: Optional[str], name: str, arguments: str):
        self.type = "function_call"
        self.id = _id
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class _FakeStreamEvent:
    def __init__(
        self,
        *,
        type: str,
        item: Optional[_FakeStreamItem] = None,
        item_id: Optional[str] = None,
        output_index: Optional[int] = None,
        name: Optional[str] = None,
        arguments: Optional[str] = None,
        delta: str = "",
        response: Any = None,
        annotation: Any = None,
    ):
        self.type = type
        self.item = item
        self.item_id = item_id
        self.output_index = output_index
        self.name = name
        self.arguments = arguments
        self.delta = delta
        self.response = response
        self.annotation = annotation


class _FakeResponsesEndpoint:
    def __init__(self, chunks: List[_FakeStreamEvent]):
        self.chunks = list(chunks)
        self.kwargs: Optional[Dict[str, Any]] = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.chunks)


class _FakeSyncClient:
    def __init__(self, chunks: List[_FakeStreamEvent]):
        self.responses = _FakeResponsesEndpoint(chunks)


class _FakeAsyncStream:
    def __init__(self, chunks: List[_FakeStreamEvent]):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeAsyncResponsesEndpoint:
    def __init__(self, chunks: List[_FakeStreamEvent]):
        self.chunks = list(chunks)
        self.kwargs: Optional[Dict[str, Any]] = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeAsyncStream(self.chunks)


class _FakeAsyncClient:
    def __init__(self, chunks: List[_FakeStreamEvent]):
        self.responses = _FakeAsyncResponsesEndpoint(chunks)


def _build_interleaved_tool_call_stream() -> List[_FakeStreamEvent]:
    item_1_added = _FakeStreamItem(_id="fc_1", call_id="call_1", name="tool_a", arguments="")
    item_2_added = _FakeStreamItem(_id="fc_2", call_id="call_2", name="tool_b", arguments="")
    item_1_done = _FakeStreamItem(_id="fc_1", call_id="call_1", name="tool_a", arguments='{"a":1}')
    item_2_done = _FakeStreamItem(_id="fc_2", call_id="call_2", name="tool_b", arguments='{"b":2}')

    return [
        _FakeStreamEvent(type="response.output_item.added", item=item_1_added, output_index=0),
        _FakeStreamEvent(type="response.output_item.added", item=item_2_added, output_index=1),
        _FakeStreamEvent(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            output_index=0,
            delta='{"a":',
        ),
        _FakeStreamEvent(
            type="response.function_call_arguments.delta",
            item_id="fc_2",
            output_index=1,
            delta='{"b":',
        ),
        _FakeStreamEvent(
            type="response.function_call_arguments.delta",
            item_id="fc_1",
            output_index=0,
            delta="1}",
        ),
        _FakeStreamEvent(
            type="response.function_call_arguments.delta",
            item_id="fc_2",
            output_index=1,
            delta="2}",
        ),
        _FakeStreamEvent(type="response.output_item.done", item=item_1_done, output_index=0),
        _FakeStreamEvent(type="response.output_item.done", item=item_2_done, output_index=1),
    ]


def test_format_messages_maps_tool_output_fc_to_call_id():
    model = OpenAIResponses(id="gpt-4.1-mini")

    # Assistant emitted a function_call with both fc_* and call_* ids
    assistant_with_tool_call = Message(
        role="assistant",
        tool_calls=[
            {
                "id": "fc_abc123",
                "call_id": "call_def456",
                "type": "function",
                "function": {"name": "execute_shell_command", "arguments": '{"command": "ls -la"}'},
            }
        ],
    )

    # Tool output referring to the fc_* id should be normalized to call_*
    tool_output = Message(role="tool", tool_call_id="fc_abc123", content="ok")

    fm = model._format_messages(
        messages=[
            Message(role="system", content="s"),
            Message(role="user", content="u"),
            assistant_with_tool_call,
            tool_output,
        ]
    )

    # Expect one function_call and one function_call_output normalized
    fc_items = [x for x in fm if x.get("type") == "function_call"]
    out_items = [x for x in fm if x.get("type") == "function_call_output"]

    assert len(fc_items) == 1
    assert fc_items[0]["id"] == "fc_abc123"
    assert fc_items[0]["call_id"] == "call_def456"

    assert len(out_items) == 1
    assert out_items[0]["call_id"] == "call_def456"


def test_format_messages_keeps_out_of_order_tool_output():
    model = OpenAIResponses(id="gpt-4.1-mini")

    assistant_with_tool_call = Message(
        role="assistant",
        tool_calls=[
            {
                "id": "fc_read",
                "call_id": "call_read",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
        ],
    )

    fm = model._format_messages(
        messages=[
            Message(role="tool", tool_call_id="fc_read", content="early result"),
            assistant_with_tool_call,
        ]
    )

    out_items = [x for x in fm if x.get("type") == "function_call_output"]

    assert out_items == [{"type": "function_call_output", "call_id": "call_read", "output": "early result"}]


def test_format_messages_drops_orphan_tool_output():
    model = OpenAIResponses(id="gpt-4.1-mini")

    fm = model._format_messages(
        messages=[
            Message(
                role="tool",
                tool_call_id="fc_missing",
                content="Tool call 'read' was cancelled before it could complete.",
            ),
            Message(role="user", content="continue"),
        ]
    )

    assert fm == [{"role": "user", "content": "continue"}]


def test_format_messages_prefers_real_result_over_cancel_placeholder():
    model = OpenAIResponses(id="gpt-4.1-mini")

    assistant_with_tool_call = Message(
        role="assistant",
        tool_calls=[
            {
                "id": "fc_read",
                "call_id": "call_read",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
        ],
    )

    fm = model._format_messages(
        messages=[
            Message(role="user", content="hello"),
            assistant_with_tool_call,
            Message(
                role="tool",
                tool_call_id="fc_read",
                content="Tool call 'read' was cancelled before it could complete.",
            ),
            Message(role="tool", tool_call_id="call_read", content="real result"),
        ]
    )

    out_items = [x for x in fm if x.get("type") == "function_call_output"]

    assert out_items == [{"type": "function_call_output", "call_id": "call_read", "output": "real result"}]


def test_format_messages_keeps_previous_response_tool_output(monkeypatch):
    model = OpenAIResponses(id="o4-mini")
    monkeypatch.setattr(model, "_using_reasoning_model", lambda: True)

    assistant_with_prev = Message(role="assistant")
    assistant_with_prev.provider_data = {"response_id": "resp_123"}  # type: ignore[attr-defined]

    fm = model._format_messages(
        messages=[
            Message(role="user", content="hello"),
            assistant_with_prev,
            Message(role="tool", tool_call_id="call_read", content="result from previous response"),
        ]
    )

    assert fm == [{"type": "function_call_output", "call_id": "call_read", "output": "result from previous response"}]


def test_parse_provider_response_maps_ids():
    model = OpenAIResponses(id="gpt-4.1-mini")

    fake_resp = _FakeResponse(
        _id="resp_1",
        output=[_FakeOutputFunctionCall(_id="fc_abc123", call_id="call_def456", name="execute", arguments="{}")],
        output_text="",
        usage=None,
        error=None,
    )

    mr: ModelResponse = model._parse_provider_response(fake_resp)  # type: ignore[arg-type]

    assert mr.tool_calls is not None and len(mr.tool_calls) == 1
    tc = mr.tool_calls[0]
    assert tc["id"] == "fc_abc123"
    assert tc["call_id"] == "call_def456"
    assert mr.extra is not None and "tool_call_ids" in mr.extra and mr.extra["tool_call_ids"][0] == "call_def456"


def test_process_stream_response_builds_tool_calls():
    model = OpenAIResponses(id="gpt-4.1-mini")
    assistant_message = Message(role="assistant")

    # Simulate function_call added and then completed
    added = _FakeStreamEvent(
        type="response.output_item.added",
        item=_FakeStreamItem(_id="fc_abc123", call_id="call_def456", name="execute", arguments="{}"),
        output_index=0,
    )
    mr, tool_uses = model._parse_provider_response_delta(added, assistant_message, {})  # type: ignore[arg-type]
    assert mr is not None
    assert mr.role is None
    assert mr.content is None
    assert mr.tool_calls == []

    # Optional: simulate args delta
    delta_ev = _FakeStreamEvent(
        type="response.function_call_arguments.delta",
        item_id="fc_abc123",
        output_index=0,
        delta='{"k":1}',
    )
    mr, tool_uses = model._parse_provider_response_delta(delta_ev, assistant_message, tool_uses)  # type: ignore[arg-type]
    assert mr is not None
    assert mr.role is None
    assert mr.content is None
    assert mr.tool_calls == []
    assert mr.event == "ToolCallArgsDelta"
    assert mr.tool_call_id == "call_def456"
    assert mr.tool_name == "execute"

    done = _FakeStreamEvent(
        type="response.output_item.done",
        item=_FakeStreamItem(_id="fc_abc123", call_id="call_def456", name="execute", arguments='{"k":1}'),
        output_index=0,
    )
    mr, tool_uses = model._parse_provider_response_delta(done, assistant_message, tool_uses)  # type: ignore[arg-type]

    assert mr is not None
    assert mr.tool_calls is not None and len(mr.tool_calls) == 1
    tc = mr.tool_calls[0]
    assert tc["id"] == "fc_abc123"
    assert tc["call_id"] == "call_def456"
    assert tc["function"]["arguments"] == '{"k":1}'
    assert assistant_message.tool_calls is not None and len(assistant_message.tool_calls) == 1
    assert tool_uses == {}


def test_process_stream_response_tracks_interleaved_tool_calls_by_item():
    model = OpenAIResponses(id="gpt-4.1-mini")
    assistant_message = Message(role="assistant")
    tool_uses: Dict[str, Dict[str, Any]] = {}
    arg_events: List[ModelResponse] = []
    done_events: List[ModelResponse] = []

    for event in _build_interleaved_tool_call_stream():
        mr, tool_uses = model._parse_provider_response_delta(event, assistant_message, tool_uses)  # type: ignore[arg-type]
        if mr.event == "ToolCallArgsDelta":
            arg_events.append(mr)
        if mr.tool_calls:
            done_events.append(mr)

    assert [(event.tool_call_id, event.tool_name, event.tool_args_delta) for event in arg_events] == [
        ("call_1", "tool_a", '{"a":'),
        ("call_2", "tool_b", '{"b":'),
        ("call_1", "tool_a", "1}"),
        ("call_2", "tool_b", "2}"),
    ]
    assert [event.extra["tool_call_ids"][0] for event in done_events if event.extra] == ["call_1", "call_2"]
    assert assistant_message.tool_calls == [
        {
            "id": "fc_1",
            "call_id": "call_1",
            "index": 0,
            "type": "function",
            "function": {"name": "tool_a", "arguments": '{"a":1}'},
        },
        {
            "id": "fc_2",
            "call_id": "call_2",
            "index": 1,
            "type": "function",
            "function": {"name": "tool_b", "arguments": '{"b":2}'},
        },
    ]
    assert tool_uses == {}


def test_process_stream_response_done_without_prior_state_still_finalizes():
    model = OpenAIResponses(id="gpt-4.1-mini")
    assistant_message = Message(role="assistant")
    stream_data = MessageData()

    done = _FakeStreamEvent(
        type="response.output_item.done",
        item=_FakeStreamItem(_id="fc_done_only", call_id="call_done_only", name="tool_done", arguments='{"ok":true}'),
        output_index=0,
    )
    mr, tool_uses = model._parse_provider_response_delta(done, assistant_message, {})  # type: ignore[arg-type]

    assert mr.tool_calls == [
        {
            "id": "fc_done_only",
            "call_id": "call_done_only",
            "index": 0,
            "type": "function",
            "function": {"name": "tool_done", "arguments": '{"ok":true}'},
        }
    ]
    events = list(model._populate_stream_data(stream_data, mr))
    # One-shot delivery (no prior args delta) now synthesizes start + a full-args delta
    # so streaming consumers can rebuild tool name/args without waiting for tool_call_started.
    start_events = [event for event in events if event.event == ModelResponseEvent.tool_call_start.value]
    args_delta_events = [event for event in events if event.event == ModelResponseEvent.tool_call_args_delta.value]
    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "call_done_only"
    assert start_events[0].tool_name == "tool_done"
    assert len(args_delta_events) == 1
    assert args_delta_events[0].tool_call_id == "call_done_only"
    assert args_delta_events[0].tool_args_delta == '{"ok":true}'
    assert mr.extra is not None and mr.extra["tool_call_ids"] == ["call_done_only"]
    assert assistant_message.tool_calls == mr.tool_calls
    assert tool_uses == {}


def test_process_stream_response_function_call_arguments_done_preserves_final_args():
    model = OpenAIResponses(id="gpt-4.1-mini")
    assistant_message = Message(role="assistant")
    stream_data = MessageData()

    added = _FakeStreamEvent(
        type="response.output_item.added",
        item=_FakeStreamItem(_id="fc_args_done", call_id="call_args_done", name="tool_done", arguments=""),
        output_index=0,
    )
    _, tool_uses = model._parse_provider_response_delta(added, assistant_message, {})  # type: ignore[arg-type]

    args_done = _FakeStreamEvent(
        type="response.function_call_arguments.done",
        item_id="fc_args_done",
        output_index=0,
        name="tool_done",
        arguments='{"done":true}',
    )
    args_done_response, tool_uses = model._parse_provider_response_delta(  # type: ignore[arg-type]
        args_done, assistant_message, tool_uses
    )
    assert args_done_response.event == ModelResponseEvent.assistant_response.value
    assert args_done_response.tool_call_id is None
    assert args_done_response.tool_args_delta is None

    done = _FakeStreamEvent(
        type="response.output_item.done",
        item=_FakeStreamItem(_id="fc_args_done", call_id="call_args_done", name="tool_done", arguments=""),
        output_index=0,
    )
    mr, tool_uses = model._parse_provider_response_delta(done, assistant_message, tool_uses)  # type: ignore[arg-type]

    assert mr.tool_calls == [
        {
            "id": "fc_args_done",
            "call_id": "call_args_done",
            "index": 0,
            "type": "function",
            "function": {"name": "tool_done", "arguments": '{"done":true}'},
        }
    ]
    events = list(model._populate_stream_data(stream_data, mr))
    # No prior tool_call_args_delta event reached _populate_stream_data for this id
    # (function_call_arguments.done is internal-only). One-shot delivery is therefore
    # synthesized here as start + full-args delta for streaming consumers.
    start_events = [event for event in events if event.event == ModelResponseEvent.tool_call_start.value]
    args_delta_events = [event for event in events if event.event == ModelResponseEvent.tool_call_args_delta.value]
    assert len(start_events) == 1
    assert len(args_delta_events) == 1
    assert args_delta_events[0].tool_args_delta == '{"done":true}'
    assert assistant_message.tool_calls == mr.tool_calls
    assert tool_uses == {}


def test_process_stream_response_args_done_without_arguments_preserves_delta_args():
    model = OpenAIResponses(id="gpt-4.1-mini")
    assistant_message = Message(role="assistant")

    added = _FakeStreamEvent(
        type="response.output_item.added",
        item=_FakeStreamItem(_id="fc_args_delta", call_id="call_args_delta", name="tool_delta", arguments=""),
        output_index=0,
    )
    _, tool_uses = model._parse_provider_response_delta(added, assistant_message, {})  # type: ignore[arg-type]

    first_delta = _FakeStreamEvent(
        type="response.function_call_arguments.delta",
        item_id="fc_args_delta",
        output_index=0,
        delta='{"city":',
    )
    _, tool_uses = model._parse_provider_response_delta(first_delta, assistant_message, tool_uses)  # type: ignore[arg-type]

    second_delta = _FakeStreamEvent(
        type="response.function_call_arguments.delta",
        item_id="fc_args_delta",
        output_index=0,
        delta='"NYC"}',
    )
    _, tool_uses = model._parse_provider_response_delta(second_delta, assistant_message, tool_uses)  # type: ignore[arg-type]

    args_done = _FakeStreamEvent(
        type="response.function_call_arguments.done",
        item_id="fc_args_delta",
        output_index=0,
        name="tool_delta",
        arguments=None,
    )
    _, tool_uses = model._parse_provider_response_delta(args_done, assistant_message, tool_uses)  # type: ignore[arg-type]

    done = _FakeStreamEvent(
        type="response.output_item.done",
        item=_FakeStreamItem(_id="fc_args_delta", call_id="call_args_delta", name="tool_delta", arguments=""),
        output_index=0,
    )
    mr, tool_uses = model._parse_provider_response_delta(done, assistant_message, tool_uses)  # type: ignore[arg-type]

    assert mr.tool_calls == [
        {
            "id": "fc_args_delta",
            "call_id": "call_args_delta",
            "index": 0,
            "type": "function",
            "function": {"name": "tool_delta", "arguments": '{"city":"NYC"}'},
        }
    ]
    assert assistant_message.tool_calls == mr.tool_calls
    assert tool_uses == {}


def test_process_stream_response_done_after_direct_delta_does_not_duplicate_args_events():
    model = OpenAIResponses(id="gpt-4.1-mini")
    assistant_message = Message(role="assistant")
    stream_data = MessageData()

    added = _FakeStreamEvent(
        type="response.output_item.added",
        item=_FakeStreamItem(_id="fc_direct", call_id="call_direct", name="tool_direct", arguments=""),
        output_index=0,
    )
    _, tool_uses = model._parse_provider_response_delta(added, assistant_message, {})  # type: ignore[arg-type]

    delta = _FakeStreamEvent(
        type="response.function_call_arguments.delta",
        item_id="fc_direct",
        output_index=0,
        delta='{"ok":true}',
    )
    delta_response, tool_uses = model._parse_provider_response_delta(delta, assistant_message, tool_uses)  # type: ignore[arg-type]
    done = _FakeStreamEvent(
        type="response.output_item.done",
        item=_FakeStreamItem(_id="fc_direct", call_id="call_direct", name="tool_direct", arguments='{"ok":true}'),
        output_index=0,
    )
    done_response, _ = model._parse_provider_response_delta(done, assistant_message, tool_uses)  # type: ignore[arg-type]

    delta_events = list(model._populate_stream_data(stream_data, delta_response))
    done_events = list(model._populate_stream_data(stream_data, done_response))
    all_args_delta_events = [
        event for event in [*delta_events, *done_events] if event.event == ModelResponseEvent.tool_call_args_delta.value
    ]

    assert len(all_args_delta_events) == 1
    assert all_args_delta_events[0].tool_call_id == "call_direct"
    assert all_args_delta_events[0].tool_name == "tool_direct"
    assert all_args_delta_events[0].tool_args_delta == '{"ok":true}'


def test_process_stream_response_completed_backfills_missing_tool_calls():
    model = OpenAIResponses(id="gpt-4.1-mini")
    assistant_message = Message(role="assistant")
    completed_response = _FakeResponse(
        _id="resp_2",
        output=[
            _FakeOutputFunctionCall(_id="fc_completed", call_id="call_completed", name="tool_completed", arguments="{}")
        ],
    )

    completed = _FakeStreamEvent(type="response.completed", response=completed_response)
    mr, tool_uses = model._parse_provider_response_delta(completed, assistant_message, {})  # type: ignore[arg-type]

    assert mr.tool_calls == [
        {
            "id": "fc_completed",
            "call_id": "call_completed",
            "type": "function",
            "function": {"name": "tool_completed", "arguments": "{}"},
        }
    ]
    assert mr.extra is not None and mr.extra["tool_call_ids"] == ["call_completed"]
    assert assistant_message.tool_calls == mr.tool_calls
    assert tool_uses == {}


def test_invoke_stream_tracks_interleaved_tool_calls_by_item(monkeypatch):
    model = OpenAIResponses(id="gpt-4.1-mini")
    fake_client = _FakeSyncClient(_build_interleaved_tool_call_stream())
    monkeypatch.setattr(model, "get_client", lambda: fake_client)

    assistant_message = Message(role="assistant")
    responses = list(
        model.invoke_stream(messages=[Message(role="user", content="u")], assistant_message=assistant_message)
    )

    assert [
        (response.tool_call_id, response.tool_name, response.tool_args_delta)
        for response in responses
        if response.event == "ToolCallArgsDelta"
    ] == [
        ("call_1", "tool_a", '{"a":'),
        ("call_2", "tool_b", '{"b":'),
        ("call_1", "tool_a", "1}"),
        ("call_2", "tool_b", "2}"),
    ]
    done_tool_calls = [response.tool_calls[0] for response in responses if response.tool_calls]
    assert done_tool_calls == [
        {
            "id": "fc_1",
            "call_id": "call_1",
            "index": 0,
            "type": "function",
            "function": {"name": "tool_a", "arguments": '{"a":1}'},
        },
        {
            "id": "fc_2",
            "call_id": "call_2",
            "index": 1,
            "type": "function",
            "function": {"name": "tool_b", "arguments": '{"b":2}'},
        },
    ]
    assert assistant_message.tool_calls == done_tool_calls


@pytest.mark.asyncio
async def test_ainvoke_stream_tracks_interleaved_tool_calls_by_item(monkeypatch):
    model = OpenAIResponses(id="gpt-4.1-mini")
    fake_client = _FakeAsyncClient(_build_interleaved_tool_call_stream())
    monkeypatch.setattr(model, "get_async_client", lambda: fake_client)

    assistant_message = Message(role="assistant")
    responses = []
    async for response in model.ainvoke_stream(
        messages=[Message(role="user", content="u")], assistant_message=assistant_message
    ):
        responses.append(response)

    assert [
        (response.tool_call_id, response.tool_name, response.tool_args_delta)
        for response in responses
        if response.event == "ToolCallArgsDelta"
    ] == [
        ("call_1", "tool_a", '{"a":'),
        ("call_2", "tool_b", '{"b":'),
        ("call_1", "tool_a", "1}"),
        ("call_2", "tool_b", "2}"),
    ]
    done_tool_calls = [response.tool_calls[0] for response in responses if response.tool_calls]
    assert done_tool_calls == [
        {
            "id": "fc_1",
            "call_id": "call_1",
            "index": 0,
            "type": "function",
            "function": {"name": "tool_a", "arguments": '{"a":1}'},
        },
        {
            "id": "fc_2",
            "call_id": "call_2",
            "index": 1,
            "type": "function",
            "function": {"name": "tool_b", "arguments": '{"b":2}'},
        },
    ]
    assert assistant_message.tool_calls == done_tool_calls


def test_reasoning_previous_response_skips_prior_function_call_items(monkeypatch):
    model = OpenAIResponses(id="o4-mini")  # reasoning

    # Force _using_reasoning_model to True
    monkeypatch.setattr(model, "_using_reasoning_model", lambda: True)

    assistant_with_prev = Message(role="assistant")
    assistant_with_prev.provider_data = {"response_id": "resp_123"}  # type: ignore[attr-defined]

    assistant_with_tool_call = Message(
        role="assistant",
        tool_calls=[
            {
                "id": "fc_abc123",
                "call_id": "call_def456",
                "type": "function",
                "function": {"name": "execute_shell_command", "arguments": "{}"},
            }
        ],
    )

    fm = model._format_messages(
        messages=[
            Message(role="system", content="s"),
            Message(role="user", content="u"),
            assistant_with_prev,
            assistant_with_tool_call,
        ]
    )

    # Expect no re-sent function_call when previous_response_id is present for reasoning models
    assert all(x.get("type") != "function_call" for x in fm)
