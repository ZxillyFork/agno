"""
Regression tests for streaming tool-call handling on the OpenAI Chat path.

Two layers are pinned:

1. **End-to-end on the OpenAI Chat path** — `_parse_provider_response_delta`
   must dict-ify `ChoiceDeltaToolCall` chunks (so `ModelResponse.tool_calls`
   honours its `List[Dict[str, Any]]` declaration), and `_populate_stream_data`
   must then synthesize the `tool_call_start` + `tool_call_args_delta` pair
   without crashing.

2. **Defensive tolerance in `_populate_stream_data`** — a few non-OpenAI
   providers (Groq, Meta Llama, HuggingFace, Azure AI Foundry) still inject
   pydantic chunk objects into the shared aggregator. The loop must not crash
   on those until each is migrated to the dict contract.

The original production bug was `AttributeError: 'ChoiceDeltaToolCall' object
has no attribute 'get'` on the first OpenAI Chat streamed tool call.
"""

from typing import Any, Dict, List, cast

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from agno.models.base import MessageData
from agno.models.openai.chat import OpenAIChat
from agno.models.response import ModelResponse, ModelResponseEvent


def _as_tool_calls(items: List[Any]) -> List[Dict[str, Any]]:
    """Cast helper for the defensive-tolerance tests below: those deliberately
    feed pydantic objects into a `List[Dict[str, Any]]` slot to exercise the
    isinstance branch in `_populate_stream_data`."""
    return cast(List[Dict[str, Any]], items)


def _chunk_with_tool_calls(tool_calls: List[ChoiceDeltaToolCall]) -> ChatCompletionChunk:
    """Build a minimal `ChatCompletionChunk` carrying tool call deltas."""
    return ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=0,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(role="assistant", tool_calls=tool_calls),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _drain(model: OpenAIChat, stream_data: MessageData, delta: ModelResponse):
    return list(model._populate_stream_data(stream_data, delta))


# ---------------------------------------------------------------------------
# Layer 1: end-to-end OpenAI Chat path (dict contract)
# ---------------------------------------------------------------------------


def test_openai_chat_delta_parser_converts_tool_calls_to_dicts():
    """`_parse_provider_response_delta` must emit `tool_calls` as a list of
    plain dicts, not pydantic `ChoiceDeltaToolCall` instances."""
    model = OpenAIChat(id="gpt-4o-mini")
    chunk = _chunk_with_tool_calls(
        [
            ChoiceDeltaToolCall(
                index=0,
                id="call_abc",
                type="function",
                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"city":"NYC"}'),
            )
        ]
    )

    parsed = model._parse_provider_response_delta(chunk)

    assert parsed.tool_calls is not None
    assert len(parsed.tool_calls) == 1
    assert isinstance(parsed.tool_calls[0], dict)
    assert parsed.tool_calls[0]["id"] == "call_abc"
    assert parsed.tool_calls[0]["function"]["name"] == "get_weather"
    assert parsed.tool_calls[0]["function"]["arguments"] == '{"city":"NYC"}'


def test_openai_chat_stream_synthesizes_start_and_args_delta_without_crash():
    """Regression for the production AttributeError: drive the OpenAI Chat
    parser into `_populate_stream_data` and verify the synthesized events."""
    model = OpenAIChat(id="gpt-4o-mini")
    stream_data = MessageData()
    chunk = _chunk_with_tool_calls(
        [
            ChoiceDeltaToolCall(
                index=0,
                id="call_abc",
                type="function",
                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"city":"NYC"}'),
            )
        ]
    )

    delta = model._parse_provider_response_delta(chunk)
    yielded = _drain(model, stream_data, delta)

    start_events = [r for r in yielded if r.event == ModelResponseEvent.tool_call_start.value]
    args_events = [r for r in yielded if r.event == ModelResponseEvent.tool_call_args_delta.value]

    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "call_abc"
    assert start_events[0].tool_name == "get_weather"

    assert len(args_events) == 1
    assert args_events[0].tool_args_delta == '{"city":"NYC"}'

    # The dict chunk is retained for downstream `parse_tool_calls` aggregation.
    assert stream_data.response_tool_calls is not None
    assert len(stream_data.response_tool_calls) == 1
    assert isinstance(stream_data.response_tool_calls[0], dict)


def test_openai_chat_stream_arguments_concatenated_across_chunks():
    """The follow-up chunk carries only an `arguments` slice with no id —
    `_populate_stream_data` must not re-emit a start, and the chunks must
    aggregate into a single tool call once `parse_tool_calls` runs."""
    model = OpenAIChat(id="gpt-4o-mini")
    stream_data = MessageData()

    first_chunk = _chunk_with_tool_calls(
        [
            ChoiceDeltaToolCall(
                index=0,
                id="call_abc",
                type="function",
                function=ChoiceDeltaToolCallFunction(name="search", arguments='{"q":"ag'),
            )
        ]
    )
    second_chunk = _chunk_with_tool_calls(
        [
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments='no"}'),
            )
        ]
    )

    first_events = _drain(model, stream_data, model._parse_provider_response_delta(first_chunk))
    second_events = _drain(model, stream_data, model._parse_provider_response_delta(second_chunk))

    assert len([r for r in first_events if r.event == ModelResponseEvent.tool_call_start.value]) == 1
    assert len([r for r in second_events if r.event == ModelResponseEvent.tool_call_start.value]) == 0

    # Drive the same path the finalizer uses: aggregate via `parse_tool_calls`.
    assert stream_data.response_tool_calls is not None
    assembled = model.parse_tool_calls(stream_data.response_tool_calls)
    assert len(assembled) == 1
    assert assembled[0]["id"] == "call_abc"
    assert assembled[0]["function"]["name"] == "search"
    assert assembled[0]["function"]["arguments"] == '{"q":"agno"}'


def test_populate_stream_data_handles_dict_tool_calls():
    """Pin the dict-element happy path that all migrated providers (OpenAI
    Responses, OpenAI Chat post-migration, ...) flow through."""
    model = OpenAIChat(id="gpt-4o-mini")
    stream_data = MessageData()
    delta = ModelResponse(
        tool_calls=[
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"agno"}'},
            }
        ]
    )

    yielded = _drain(model, stream_data, delta)

    start_events = [r for r in yielded if r.event == ModelResponseEvent.tool_call_start.value]
    args_events = [r for r in yielded if r.event == ModelResponseEvent.tool_call_args_delta.value]

    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "call_xyz"
    assert start_events[0].tool_name == "search"

    assert len(args_events) == 1
    assert args_events[0].tool_args_delta == '{"q":"agno"}'


def test_populate_stream_data_falls_back_to_call_id_for_dict_elements():
    """OpenAI Responses uses `call_id` rather than `id` for the tool call id;
    both keys are honoured for dict elements."""
    model = OpenAIChat(id="gpt-4o-mini")
    stream_data = MessageData()
    delta = ModelResponse(
        tool_calls=[
            {
                "call_id": "call_resp",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
    )

    yielded = _drain(model, stream_data, delta)

    start_events = [r for r in yielded if r.event == ModelResponseEvent.tool_call_start.value]
    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "call_resp"


# ---------------------------------------------------------------------------
# Layer 2: defensive tolerance for not-yet-migrated providers
# ---------------------------------------------------------------------------


def test_populate_stream_data_tolerates_pydantic_tool_calls_from_legacy_providers():
    """Groq / Meta Llama / HuggingFace / Azure AI Foundry still pass pydantic
    chunks straight through. Until each is migrated to the dict contract, the
    snapshot-synthesis loop must not crash on them."""
    model = OpenAIChat(id="gpt-4o-mini")
    stream_data = MessageData()
    pydantic_chunk = ChoiceDeltaToolCall(
        index=0,
        id="call_legacy",
        type="function",
        function=ChoiceDeltaToolCallFunction(name="legacy", arguments='{"x":1}'),
    )
    delta = ModelResponse(tool_calls=_as_tool_calls([pydantic_chunk]))

    yielded = _drain(model, stream_data, delta)

    start_events = [r for r in yielded if r.event == ModelResponseEvent.tool_call_start.value]
    args_events = [r for r in yielded if r.event == ModelResponseEvent.tool_call_args_delta.value]

    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "call_legacy"
    assert start_events[0].tool_name == "legacy"
    assert len(args_events) == 1
    assert args_events[0].tool_args_delta == '{"x":1}'

    # Pydantic objects must be preserved verbatim — those providers' own
    # `parse_tool_calls` implementations depend on attribute access and on
    # receiving every fragment chunk to concatenate `function.arguments`.
    assert stream_data.response_tool_calls is not None
    assert stream_data.response_tool_calls[0] is pydantic_chunk


def test_populate_stream_data_tolerates_pydantic_fragment_without_id():
    """Mid-stream fragments from legacy providers carry `id=None`. The defence
    must skip start synthesis (no id to attach to) without crashing."""
    model = OpenAIChat(id="gpt-4o-mini")
    stream_data = MessageData()
    stream_data.tool_call_started_ids.add("call_legacy")  # start already emitted

    fragment = ChoiceDeltaToolCall(
        index=0,
        function=ChoiceDeltaToolCallFunction(arguments='{"more":'),
    )
    delta = ModelResponse(tool_calls=_as_tool_calls([fragment]))

    yielded = _drain(model, stream_data, delta)

    assert not [r for r in yielded if r.event == ModelResponseEvent.tool_call_start.value]
    assert stream_data.response_tool_calls == [fragment]
