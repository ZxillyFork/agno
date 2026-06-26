from agno.models.base import MessageData, Model
from agno.models.response import ModelResponse, ModelResponseEvent


class DummyModel(Model):
    def invoke(self, *args, **kwargs) -> ModelResponse:  # pragma: no cover - not needed for tests
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:  # pragma: no cover - not needed for tests
        raise NotImplementedError

    def invoke_stream(self, *args, **kwargs):  # pragma: no cover - not needed for tests
        raise NotImplementedError

    async def ainvoke_stream(self, *args, **kwargs):  # pragma: no cover - not needed for tests
        raise NotImplementedError

    def _parse_provider_response(self, response, **kwargs) -> ModelResponse:  # pragma: no cover - not needed for tests
        raise NotImplementedError

    def _parse_provider_response_delta(self, response) -> ModelResponse:  # pragma: no cover - not needed for tests
        raise NotImplementedError


def test_tool_calls_snapshot_emits_start_and_full_args_delta_for_new_id():
    """One-shot tool_calls snapshot (e.g. OpenAI Responses path B) emits tool_call_start +
    a single tool_call_args_delta carrying the full arguments so streaming consumers can
    rebuild tool name/args without waiting for tool_call_started."""
    model = DummyModel(id="dummy-model")
    stream_data = MessageData()

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "do_thing", "arguments": '{"value": 1}'},
    }
    response_delta = ModelResponse(tool_calls=[tool_call])

    events = list(model._populate_stream_data(stream_data, response_delta))
    start_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_start.value]
    args_delta_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_args_delta.value]

    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "call_1"
    assert start_events[0].tool_name == "do_thing"

    assert len(args_delta_events) == 1
    assert args_delta_events[0].tool_call_id == "call_1"
    assert args_delta_events[0].tool_args_delta == '{"value": 1}'

    # Ordering: start precedes the args delta
    assert events.index(start_events[0]) < events.index(args_delta_events[0])


def test_tool_calls_snapshot_with_partial_args_still_synthesizes_start_and_delta():
    """Snapshots carrying partial arguments still synthesize a start + delta so
    consumers can begin assembling the call even before the provider finalizes it."""
    model = DummyModel(id="dummy-model")
    stream_data = MessageData()

    tool_call = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "do_thing", "arguments": '{"value":'},
    }
    response_delta = ModelResponse(tool_calls=[tool_call])

    events = list(model._populate_stream_data(stream_data, response_delta))
    start_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_start.value]
    args_delta_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_args_delta.value]

    assert len(start_events) == 1
    assert len(args_delta_events) == 1
    assert args_delta_events[0].tool_args_delta == '{"value":'


def test_tool_calls_snapshot_emits_start_only_once_for_repeated_id():
    """Subsequent tool_calls snapshots for an already-seen id (e.g. id present on the
    first OpenAI Chat chunk, omitted on subsequent index-only chunks) must not emit
    duplicate tool_call_start events."""
    model = DummyModel(id="dummy-model")
    stream_data = MessageData()

    response_delta_1 = ModelResponse(
        tool_calls=[
            {
                "id": "call_3",
                "index": 0,
                "type": "function",
                "function": {"name": "do_thing", "arguments": "{"},
            }
        ]
    )
    events_1 = list(model._populate_stream_data(stream_data, response_delta_1))
    start_events_1 = [ev for ev in events_1 if ev.event == ModelResponseEvent.tool_call_start.value]
    assert len(start_events_1) == 1

    response_delta_2 = ModelResponse(
        tool_calls=[
            {
                "index": 0,
                "type": "function",
                "function": {"arguments": '"value": 1}'},
            }
        ]
    )
    events_2 = list(model._populate_stream_data(stream_data, response_delta_2))
    start_events_2 = [ev for ev in events_2 if ev.event == ModelResponseEvent.tool_call_start.value]
    # No id on the follow-up snapshot → cannot correlate, must not re-emit
    assert len(start_events_2) == 0


def test_tool_call_start_emitted_before_first_args_delta_direct_path():
    """tool_call_start is emitted before the first tool_call_args_delta for each tool call (direct path)."""
    model = DummyModel(id="dummy-model")
    stream_data = MessageData()

    # Direct path: model provider emits tool_call_args_delta events directly
    response_delta = ModelResponse(
        event=ModelResponseEvent.tool_call_args_delta.value,
        tool_call_id="call_20",
        tool_name="direct_tool",
        tool_args_delta='{"x": 1}',
    )

    events = list(model._populate_stream_data(stream_data, response_delta))
    start_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_start.value]
    args_delta_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_args_delta.value]

    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "call_20"
    assert start_events[0].tool_name == "direct_tool"

    assert len(args_delta_events) == 1

    # Verify ordering: tool_call_start comes before tool_call_args_delta
    start_idx = events.index(start_events[0])
    delta_idx = events.index(args_delta_events[0])
    assert start_idx < delta_idx


def test_tool_call_start_emitted_only_once_per_tool_call_direct_path():
    """tool_call_start is emitted only once per tool call, even with multiple arg deltas (direct path)."""
    model = DummyModel(id="dummy-model")
    stream_data = MessageData()

    # First delta
    response_delta_1 = ModelResponse(
        event=ModelResponseEvent.tool_call_args_delta.value,
        tool_call_id="call_21",
        tool_name="direct_tool",
        tool_args_delta='{"x":',
    )
    events_1 = list(model._populate_stream_data(stream_data, response_delta_1))
    start_events_1 = [ev for ev in events_1 if ev.event == ModelResponseEvent.tool_call_start.value]
    assert len(start_events_1) == 1

    # Second delta for same tool call
    response_delta_2 = ModelResponse(
        event=ModelResponseEvent.tool_call_args_delta.value,
        tool_call_id="call_21",
        tool_name="direct_tool",
        tool_args_delta=" 1}",
    )
    events_2 = list(model._populate_stream_data(stream_data, response_delta_2))
    start_events_2 = [ev for ev in events_2 if ev.event == ModelResponseEvent.tool_call_start.value]
    # Should NOT emit another tool_call_start for the same tool call
    assert len(start_events_2) == 0


def test_indirect_tool_call_deltas_are_ignored_after_direct_delta_for_same_tool():
    """Direct tool_call_args_delta events win over later tool_calls snapshots for the same tool call."""
    model = DummyModel(id="dummy-model")
    stream_data = MessageData()

    direct_delta = ModelResponse(
        event=ModelResponseEvent.tool_call_args_delta.value,
        tool_call_id="call_22",
        tool_name="direct_tool",
        tool_args_delta='{"x":',
    )
    direct_events = list(model._populate_stream_data(stream_data, direct_delta))
    assert len([ev for ev in direct_events if ev.event == ModelResponseEvent.tool_call_args_delta.value]) == 1

    final_tool_call = ModelResponse(
        tool_calls=[
            {
                "id": "call_22",
                "type": "function",
                "function": {"name": "direct_tool", "arguments": '{"x": 1}'},
            }
        ]
    )
    final_events = list(model._populate_stream_data(stream_data, final_tool_call))
    final_args_delta_events = [ev for ev in final_events if ev.event == ModelResponseEvent.tool_call_args_delta.value]

    assert len(final_args_delta_events) == 0


def test_tool_calls_snapshot_emits_one_start_per_new_tool():
    """Each new tool_call id in a snapshot gets its own tool_call_start event."""
    model = DummyModel(id="dummy-model")
    stream_data = MessageData()

    # Two tool calls in the same response
    tool_calls = [
        {
            "id": "call_30",
            "index": 0,
            "type": "function",
            "function": {"name": "tool_a", "arguments": '{"a": 1}'},
        },
        {
            "id": "call_31",
            "index": 1,
            "type": "function",
            "function": {"name": "tool_b", "arguments": '{"b": 2}'},
        },
    ]
    response_delta = ModelResponse(tool_calls=tool_calls)

    events = list(model._populate_stream_data(stream_data, response_delta))
    start_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_start.value]
    args_delta_events = [ev for ev in events if ev.event == ModelResponseEvent.tool_call_args_delta.value]

    assert [ev.tool_call_id for ev in start_events] == ["call_30", "call_31"]
    assert [ev.tool_call_id for ev in args_delta_events] == ["call_30", "call_31"]
