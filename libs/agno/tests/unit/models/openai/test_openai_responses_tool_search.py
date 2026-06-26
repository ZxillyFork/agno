from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agno.models.message import Message
from agno.models.openai.responses import OpenAIResponses
from agno.models.openai.tools import ToolNamespace, ToolSearch
from agno.run import RunContext
from agno.tools import tool
from agno.tools.function import Function


class _FakeOutput:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)

    def model_dump(self, exclude_none: bool = False) -> dict:
        if exclude_none:
            return {key: value for key, value in self.__dict__.items() if value is not None}
        return dict(self.__dict__)


class _FakeResponse:
    def __init__(
        self,
        *,
        _id: str = "resp_test",
        status: str = "completed",
        output: Optional[List[Any]] = None,
        output_text: str = "",
        usage: Any = None,
        error: Any = None,
        incomplete_details: Any = None,
    ):
        self.id = _id
        self.status = status
        self.output = output or []
        self.output_text = output_text
        self.usage = usage
        self.error = error
        self.incomplete_details = incomplete_details


def _make_fake_client() -> MagicMock:
    client = MagicMock()
    client.is_closed.return_value = False
    return client


def test_tool_search_helpers_format_openai_schema():
    @tool(defer_loading=True)
    def list_open_orders(customer_id: str) -> str:
        return customer_id

    namespace = ToolNamespace(
        name="crm",
        description="CRM tools",
        tools=[list_open_orders, {"type": "function", "name": "raw_tool", "parameters": {}}],
    )

    model = OpenAIResponses(id="gpt-5.4")
    formatted = model._format_tool_params(
        messages=[Message(role="user", content="orders")],
        tools=[namespace, ToolSearch.server()],
    )

    assert formatted[0]["type"] == "namespace"
    assert formatted[0]["name"] == "crm"
    assert formatted[0]["tools"][0]["type"] == "function"
    assert formatted[0]["tools"][0]["name"] == "list_open_orders"
    assert formatted[0]["tools"][0]["defer_loading"] is True
    assert formatted[0]["tools"][1]["name"] == "raw_tool"
    assert formatted[1] == {"type": "tool_search"}


def test_raw_dict_tools_pass_through_for_provider_native_shapes():
    tools = [
        {"type": "tool_search"},
        {
            "type": "mcp",
            "server_label": "docs",
            "server_url": "https://example.com/mcp",
            "defer_loading": True,
        },
    ]

    model = OpenAIResponses(id="gpt-5.4")
    formatted = model._format_tool_params(messages=[Message(role="user", content="docs")], tools=tools)

    assert formatted == tools
    assert formatted is not tools


def test_openai_responses_formats_tools_in_deterministic_order_without_converting_helpers():
    @tool
    def beta_tool() -> str:
        return "beta"

    @tool
    def alpha_tool() -> str:
        return "alpha"

    def searcher(_call, _run_context):
        return []

    tool_search = ToolSearch.client(
        description="Search tools",
        parameters={"type": "object", "properties": {}},
        searcher=searcher,
    )
    namespace = ToolNamespace(name="crm", description="CRM tools", tools=[])

    model = OpenAIResponses(id="gpt-5.4")
    formatted_tools = model._format_tools([beta_tool, tool_search, namespace, alpha_tool])

    assert formatted_tools == [alpha_tool, beta_tool, namespace, tool_search]
    assert formatted_tools[-1].searcher is searcher


def test_parse_provider_response_preserves_hosted_tool_search_items():
    response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="server",
                status="completed",
            ),
            _FakeOutput(
                type="tool_search_output",
                id="tso_1",
                call_id="call_search",
                execution="server",
                status="completed",
                tools=[],
            ),
            _FakeOutput(
                type="function_call",
                id="fc_1",
                call_id="call_loaded",
                name="loaded_tool",
                arguments="{}",
            ),
        ]
    )

    model = OpenAIResponses(id="gpt-5.4")
    parsed = model._parse_provider_response(response)

    assert parsed.tool_calls[0]["function"]["name"] == "loaded_tool"
    assert parsed.extra is not None
    assert parsed.extra["tool_search"]["calls"][0]["execution"] == "server"
    assert parsed.extra["tool_search"]["outputs"][0]["type"] == "tool_search_output"


def test_client_tool_search_round_trip_loads_tools_and_registers_functions():
    loaded_function = Function(
        name="loaded_tool",
        description="Loaded dynamically",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
    )

    seen_calls = []

    def searcher(call, _run_context):
        seen_calls.append(call)
        return [loaded_function]

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments='{"query": "loaded"}',
                status="completed",
            )
        ]
    )
    second_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="function_call",
                id="fc_1",
                call_id="call_loaded",
                name="loaded_tool",
                arguments='{"x": 1}',
            )
        ]
    )

    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [first_response, second_response]

    model = OpenAIResponses(id="gpt-5.4", parallel_tool_calls=False)
    model.client = fake_client
    assistant_message = Message(role="assistant")

    response = model.invoke(
        messages=[Message(role="user", content="Find the loaded tool")],
        assistant_message=assistant_message,
        tools=[
            ToolSearch.client(
                description="Search available tools",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                searcher=searcher,
            )
        ],
    )

    assert len(seen_calls) == 1
    assert seen_calls[0].call_id == "call_search"
    assert seen_calls[0].arguments == {"query": "loaded"}

    assert fake_client.responses.create.call_count == 2
    _, first_kwargs = fake_client.responses.create.call_args_list[0]
    _, second_kwargs = fake_client.responses.create.call_args_list[1]

    assert first_kwargs["tools"][0]["type"] == "tool_search"
    assert first_kwargs["tools"][0]["execution"] == "client"
    assert "tools" not in second_kwargs

    tool_search_output = second_kwargs["input"][-1]
    assert tool_search_output["type"] == "tool_search_output"
    assert tool_search_output["call_id"] == "call_search"
    assert tool_search_output["tools"][0]["name"] == "loaded_tool"

    assert response.tool_calls[0]["function"]["name"] == "loaded_tool"
    assert response.extra is not None
    assert response.extra["tool_search"]["calls"][0]["type"] == "tool_search_call"
    assert response.extra["tool_search"]["outputs"][0]["type"] == "tool_search_output"

    assistant_with_tool_call = Message(role="assistant", tool_calls=response.tool_calls)
    calls_to_run = model.get_function_calls_to_run(assistant_with_tool_call, messages=[], functions={})
    assert len(calls_to_run) == 1
    assert calls_to_run[0].function.name == "loaded_tool"
    assert calls_to_run[0].arguments == {"x": 1}


def test_client_tool_search_response_passes_run_context_to_searcher():
    seen_contexts = []

    def searcher(_call, run_context):
        seen_contexts.append(run_context)
        return []

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    second_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="message",
                content=[_FakeOutput(type="output_text", annotations=[])],
            )
        ],
        output_text="done",
    )

    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [first_response, second_response]

    model = OpenAIResponses(id="gpt-5.4")
    model.client = fake_client
    run_context = RunContext(run_id="run_1", session_id="session_1")

    response = model.response(
        messages=[Message(role="user", content="Find a tool")],
        tools=[
            ToolSearch.client(
                description="Search available tools",
                parameters={"type": "object", "properties": {}},
                searcher=searcher,
            )
        ],
        run_context=run_context,
    )

    assert response.content == "done"
    assert seen_contexts == [run_context]
    assert model._current_run_context is None


def test_client_tool_search_dynamic_functions_are_available_for_resume_helpers_only():
    loaded_function = Function(
        name="loaded_for_resume",
        description="Loaded dynamically",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    model = OpenAIResponses(id="gpt-5.4")
    model._client_tool_search_functions["loaded_for_resume"] = loaded_function

    assert model._get_functions_from_tools([]) == {}
    assert model._get_functions_from_tools([], include_dynamic_functions=True)["loaded_for_resume"] is loaded_function


@pytest.mark.asyncio
async def test_async_client_tool_search_supports_async_searcher():
    loaded_function = Function(
        name="async_loaded_tool",
        description="Loaded dynamically",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    async def searcher(call, _run_context):
        assert call.call_id == "call_search"
        return [loaded_function]

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    second_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="function_call",
                id="fc_1",
                call_id="call_loaded",
                name="async_loaded_tool",
                arguments="{}",
            )
        ]
    )

    fake_client = _make_fake_client()
    fake_client.responses.create = AsyncMock(side_effect=[first_response, second_response])

    model = OpenAIResponses(id="gpt-5.4")
    model.async_client = fake_client

    response = await model.ainvoke(
        messages=[Message(role="user", content="Find the async tool")],
        assistant_message=Message(role="assistant"),
        tools=[
            ToolSearch.client(
                description="Search available tools",
                parameters={"type": "object", "properties": {}},
                searcher=searcher,
            )
        ],
    )

    assert fake_client.responses.create.call_count == 2
    assert response.tool_calls[0]["function"]["name"] == "async_loaded_tool"
    calls_to_run = model.get_function_calls_to_run(Message(role="assistant", tool_calls=response.tool_calls), [], {})
    assert calls_to_run[0].function.name == "async_loaded_tool"


class _FakeChunk:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


def test_streaming_client_tool_search_continues_with_loaded_tools():
    loaded_function = Function(
        name="stream_loaded_tool",
        description="Loaded dynamically",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    def searcher(_call, _run_context):
        return [loaded_function]

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    function_output = _FakeOutput(
        type="function_call",
        id="fc_1",
        call_id="call_stream_loaded",
        name="stream_loaded_tool",
        arguments="{}",
    )
    second_response = _FakeResponse(output=[function_output])

    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [
        iter([_FakeChunk(type="response.completed", response=first_response)]),
        iter(
            [
                _FakeChunk(type="response.output_item.done", item=function_output, output_index=0),
                _FakeChunk(type="response.completed", response=second_response),
            ]
        ),
    ]

    model = OpenAIResponses(id="gpt-5.4")
    model.client = fake_client
    assistant_message = Message(role="assistant")

    chunks = list(
        model.invoke_stream(
            messages=[Message(role="user", content="Find the stream tool")],
            assistant_message=assistant_message,
            tools=[
                ToolSearch.client(
                    description="Search available tools",
                    parameters={"type": "object", "properties": {}},
                    searcher=searcher,
                )
            ],
        )
    )

    assert fake_client.responses.create.call_count == 2
    assert any(chunk.tool_calls for chunk in chunks)
    assert assistant_message.tool_calls is not None
    assert assistant_message.tool_calls[0]["function"]["name"] == "stream_loaded_tool"
    calls_to_run = model.get_function_calls_to_run(assistant_message, [], {})
    assert calls_to_run[0].function.name == "stream_loaded_tool"


def test_client_tool_search_loaded_functions_survive_tool_result_follow_up():
    loaded_function = Function(
        name="loaded_again",
        description="Loaded dynamically",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    def searcher(_call, _run_context):
        return [loaded_function]

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    first_function_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="function_call",
                id="fc_1",
                call_id="call_loaded_once",
                name="loaded_again",
                arguments="{}",
            )
        ]
    )
    second_function_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="function_call",
                id="fc_2",
                call_id="call_loaded_twice",
                name="loaded_again",
                arguments="{}",
            )
        ]
    )

    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [first_response, first_function_response, second_function_response]

    model = OpenAIResponses(id="gpt-5.4")
    model.client = fake_client
    tool_search = ToolSearch.client(
        description="Search available tools",
        parameters={"type": "object", "properties": {}},
        searcher=searcher,
    )

    first = model.invoke(
        messages=[Message(role="user", content="Find tool")],
        assistant_message=Message(role="assistant"),
        tools=[tool_search],
    )
    assert first.tool_calls[0]["function"]["name"] == "loaded_again"

    follow_up_messages = [
        Message(role="user", content="Find tool"),
        Message(role="assistant", tool_calls=first.tool_calls),
        Message(role="tool", tool_call_id="call_loaded_once", content="ok"),
    ]
    second = model.invoke(
        messages=follow_up_messages,
        assistant_message=Message(role="assistant"),
        tools=[tool_search],
    )

    calls_to_run = model.get_function_calls_to_run(Message(role="assistant", tool_calls=second.tool_calls), [], {})
    assert calls_to_run[0].function.name == "loaded_again"


def test_sync_client_tool_search_rejects_async_searcher():
    async def searcher(_call, _run_context):
        return []

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )

    fake_client = _make_fake_client()
    fake_client.responses.create.return_value = first_response

    model = OpenAIResponses(id="gpt-5.4")
    model.client = fake_client

    with pytest.raises(Exception, match="Async ToolSearch searcher"):
        model.invoke(
            messages=[Message(role="user", content="Find tool")],
            assistant_message=Message(role="assistant"),
            tools=[
                ToolSearch.client(
                    description="Search available tools",
                    parameters={"type": "object", "properties": {}},
                    searcher=searcher,
                )
            ],
        )


def _text_response(text: str) -> _FakeResponse:
    return _FakeResponse(
        output=[_FakeOutput(type="message", content=[_FakeOutput(type="output_text", annotations=[])])],
        output_text=text,
    )


def _client_tool_search(searcher) -> ToolSearch:
    return ToolSearch.client(
        description="Search available tools",
        parameters={"type": "object", "properties": {}},
        searcher=searcher,
    )


def test_client_tool_search_state_isolated_across_runs_on_shared_model():
    """A model instance reused across runs must not leak dynamically loaded tools between runs."""
    loaded_function = Function(
        name="run_a_tool",
        description="Loaded only for run A",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    def searcher(_call, _run_context):
        return [loaded_function]

    run_a_first = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    run_a_second = _text_response("A done")
    run_b_only = _text_response("B done")

    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [run_a_first, run_a_second, run_b_only]

    model = OpenAIResponses(id="gpt-5.4")
    model.client = fake_client

    model.response(
        messages=[Message(role="user", content="A")],
        tools=[_client_tool_search(searcher)],
        run_context=RunContext(run_id="run_a", session_id="s"),
    )
    # Run A loaded the tool into its per-run state.
    assert "run_a_tool" in model._client_tool_search_functions

    # Run B (different run_id, no client tool search) must start from a clean slate.
    model.response(
        messages=[Message(role="user", content="B")],
        tools=[],
        run_context=RunContext(run_id="run_b", session_id="s"),
    )
    assert model._client_tool_search_functions == {}


def test_loaded_functions_are_prepared_and_receive_run_context():
    """Searcher-returned functions are copied, entrypoint-processed, and given the run context."""

    def needs_context(run_context=None, x: int = 0) -> str:
        return "ok"

    loaded_function = Function(name="needs_ctx", description="Needs run context", entrypoint=needs_context)

    def searcher(_call, _run_context):
        return [loaded_function]

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [first_response, _text_response("done")]

    model = OpenAIResponses(id="gpt-5.4")
    model.client = fake_client
    run_context = RunContext(run_id="run_ctx", session_id="s")

    model.response(
        messages=[Message(role="user", content="x")],
        tools=[_client_tool_search(searcher)],
        run_context=run_context,
    )

    registered = model._client_tool_search_functions["needs_ctx"]
    assert registered._run_context is run_context
    # The searcher's original object must not be mutated (a deep copy is registered).
    assert loaded_function._run_context is None


def test_resolve_allows_exactly_max_rounds_searches():
    """A run using exactly client_tool_search_max_rounds searches must succeed, not raise."""

    def searcher(_call, _run_context):
        return []

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [first_response, _text_response("done")]

    model = OpenAIResponses(id="gpt-5.4", client_tool_search_max_rounds=1)
    model.client = fake_client

    response = model.response(
        messages=[Message(role="user", content="x")],
        tools=[_client_tool_search(searcher)],
    )
    assert response.content == "done"


def test_resolve_raises_before_extra_search_round():
    searcher = MagicMock(return_value=[])

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search_1",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    second_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_2",
                call_id="call_search_2",
                execution="client",
                arguments={},
                status="completed",
            )
        ]
    )
    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [first_response, second_response]

    model = OpenAIResponses(id="gpt-5.4", client_tool_search_max_rounds=1)
    model.client = fake_client

    with pytest.raises(Exception, match="Exceeded maximum client tool search rounds"):
        model.response(messages=[Message(role="user", content="x")], tools=[_client_tool_search(searcher)])

    assert searcher.call_count == 1
    assert fake_client.responses.create.call_count == 2


def test_tool_search_follow_up_params_drop_tool_choice_without_tools():
    params = {
        "temperature": 0,
        "tools": [{"type": "tool_search"}],
        "tool_choice": {"type": "tool_search"},
        "previous_response_id": "resp_1",
    }

    follow_up = OpenAIResponses._tool_search_follow_up_params(params)

    assert follow_up == {"temperature": 0}
    assert "tool_choice" in params


def test_vector_store_id_attached_to_namespace_nested_file_search():
    model = OpenAIResponses(id="gpt-5.4")
    tool_dict = {"type": "namespace", "name": "kb", "tools": [{"type": "file_search"}]}
    model._attach_vector_store_id(tool_dict, "vs_123")
    assert tool_dict["tools"][0]["vector_store_ids"] == ["vs_123"]


def test_provider_tool_on_non_openai_responses_model_raises_clear_error():
    from agno.models.openai.chat import OpenAIChat

    model = OpenAIChat(id="gpt-4o")
    with pytest.raises(ValueError, match="only.*supported by the OpenAIResponses"):
        model._format_tools([ToolSearch.server()])


def test_saved_direct_tool_call_resolves_against_defer_loading_namespace_tool():
    """Migration case: a run saved with a directly loaded tool resumes fine when that same tool is
    later moved into a namespace with defer_loading=True. defer_loading only changes the wire format
    sent to OpenAI — the Function stays locally available, and tool calls match by plain name."""
    from agno.utils.tools import get_function_call_for_tool_call

    @tool(defer_loading=True)
    def list_open_orders(customer_id: str) -> str:
        return customer_id

    namespace = ToolNamespace(name="crm", description="CRM tools", tools=[list_open_orders])
    model = OpenAIResponses(id="gpt-5.4")

    # The resume helper collects namespace-nested functions by their plain name.
    functions = model._get_functions_from_tools([namespace], include_dynamic_functions=True)
    assert "list_open_orders" in functions

    # A tool call persisted by the OLD directly-loaded run carries the plain tool name and still
    # resolves against the new defer_loading namespace tool.
    saved_tool_call = {
        "type": "function",
        "call_id": "call_old",
        "function": {"name": "list_open_orders", "arguments": '{"customer_id": "c1"}'},
    }
    function_call = get_function_call_for_tool_call(saved_tool_call, functions)
    assert function_call is not None
    assert function_call.function.name == "list_open_orders"
    assert function_call.arguments == {"customer_id": "c1"}


def test_namespaced_function_call_uses_namespace_for_local_lookup():
    from agno.utils.tools import get_function_call_for_tool_call

    def crm_lookup(account_id: str) -> str:
        return f"crm:{account_id}"

    def billing_lookup(account_id: str) -> str:
        return f"billing:{account_id}"

    crm_function = Function.from_callable(crm_lookup, name="lookup")
    billing_function = Function.from_callable(billing_lookup, name="lookup")
    model = OpenAIResponses(id="gpt-5.4")
    functions = model._get_functions_from_tools(
        [
            ToolNamespace(name="crm", description="CRM tools", tools=[crm_function]),
            ToolNamespace(name="billing", description="Billing tools", tools=[billing_function]),
        ]
    )

    namespaced_call = {
        "type": "function",
        "call_id": "call_crm",
        "namespace": "crm",
        "function": {"name": "lookup", "arguments": '{"account_id": "a1"}'},
    }
    function_call = get_function_call_for_tool_call(namespaced_call, functions)

    assert function_call is not None
    assert function_call.function is crm_function
    assert "lookup" not in functions


def test_namespaced_function_call_namespace_is_round_tripped():
    """OpenAI emits namespaced/deferred function calls with a separate `namespace` field and rejects
    the follow-up request unless that field is round-tripped. It must survive parse -> format."""
    response = _FakeResponse(
        output=[
            _FakeOutput(
                type="function_call",
                id="fc_1",
                call_id="call_ns",
                name="list_open_orders",
                namespace="crm",
                arguments='{"customer_id": "c1"}',
            )
        ]
    )
    model = OpenAIResponses(id="gpt-5.4")
    parsed = model._parse_provider_response(response)
    assert parsed.tool_calls[0]["function"]["name"] == "list_open_orders"
    assert parsed.tool_calls[0]["namespace"] == "crm"

    assistant = Message(role="assistant", tool_calls=parsed.tool_calls)
    formatted = model._format_messages([assistant])
    function_call_items = [item for item in formatted if isinstance(item, dict) and item.get("type") == "function_call"]
    assert function_call_items
    assert function_call_items[0]["name"] == "list_open_orders"
    assert function_call_items[0]["namespace"] == "crm"


def test_default_namespace_function_call_has_no_namespace_field():
    """A plain (default-namespace) tool call must not gain a spurious namespace field."""
    response = _FakeResponse(
        output=[_FakeOutput(type="function_call", id="fc_1", call_id="call_plain", name="plain_tool", arguments="{}")]
    )
    model = OpenAIResponses(id="gpt-5.4")
    parsed = model._parse_provider_response(response)
    assert "namespace" not in parsed.tool_calls[0]

    assistant = Message(role="assistant", tool_calls=parsed.tool_calls)
    formatted = model._format_messages([assistant])
    function_call_items = [item for item in formatted if isinstance(item, dict) and item.get("type") == "function_call"]
    assert function_call_items
    assert "namespace" not in function_call_items[0]


def test_client_tool_search_items_persisted_and_reinjected_across_turns():
    """Client tool_search items are persisted on the assistant message and re-injected (in order,
    before the function_call) on a later request, so a deferred tool stays declared across turns."""
    loaded_function = Function(
        name="loaded_tool",
        description="Loaded dynamically",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    def searcher(_call, _run_context):
        return [loaded_function]

    first_response = _FakeResponse(
        output=[
            _FakeOutput(
                type="tool_search_call",
                id="ts_1",
                call_id="call_search",
                execution="client",
                arguments='{"query": "loaded"}',
                status="completed",
            )
        ]
    )
    second_response = _FakeResponse(
        output=[_FakeOutput(type="function_call", id="fc_1", call_id="call_loaded", name="loaded_tool", arguments="{}")]
    )

    fake_client = _make_fake_client()
    fake_client.responses.create.side_effect = [first_response, second_response]

    # Non-reasoning model id so history is re-sent manually (reasoning models chain via
    # previous_response_id and the server already retains the prior turn, so re-injection is
    # intentionally skipped there).
    model = OpenAIResponses(id="gpt-4.1")
    model.client = fake_client
    assistant_message = Message(role="assistant")

    response = model.invoke(
        messages=[Message(role="user", content="Find the loaded tool")],
        assistant_message=assistant_message,
        tools=[_client_tool_search(searcher)],
    )

    # The tool_search call + our constructed output are persisted on the model response...
    assert response.provider_data is not None
    persisted = response.provider_data["tool_search_items"]
    assert [item["type"] for item in persisted] == ["tool_search_call", "tool_search_output"]
    assert persisted[1]["tools"][0]["name"] == "loaded_tool"

    # ...and _populate_assistant_message would put them on the assistant message; simulate that and
    # confirm _format_messages re-injects them before the function_call on the next turn.
    assistant_message.provider_data = response.provider_data
    assistant_message.tool_calls = response.tool_calls
    follow_up = [
        Message(role="user", content="Find the loaded tool"),
        assistant_message,
        Message(role="tool", tool_call_id="call_loaded", content="ok"),
    ]
    formatted = model._format_messages(follow_up)
    types_in_order = [item.get("type") for item in formatted if isinstance(item, dict) and "type" in item]
    assert types_in_order == ["tool_search_call", "tool_search_output", "function_call", "function_call_output"]


def test_streaming_namespaced_function_call_preserves_namespace():
    item = _FakeOutput(
        type="function_call",
        id="fc_1",
        call_id="call_ns",
        name="list_open_orders",
        namespace="crm",
        arguments='{"x": 1}',
    )
    model = OpenAIResponses(id="gpt-5.4")
    assistant = Message(role="assistant")
    model._parse_provider_response_delta(
        _FakeChunk(type="response.output_item.done", item=item, output_index=0),
        assistant,
        {},
    )
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0]["namespace"] == "crm"
