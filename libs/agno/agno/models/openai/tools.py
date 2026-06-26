from __future__ import annotations

from dataclasses import dataclass, field, replace
from inspect import isawaitable
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from agno.run import RunContext
from agno.tools.function import Function


@dataclass
class ToolSearchCall:
    """OpenAI client-executed tool_search call."""

    call_id: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    execution: Literal["client"] = "client"
    status: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


ToolSearchResolver = Callable[
    [ToolSearchCall, Optional[RunContext]],
    Union[
        Sequence[Union[Function, Dict[str, Any], "ToolNamespace"]],
        Any,
    ],
]


@dataclass
class ToolSearch:
    """OpenAI Responses tool_search provider tool."""

    execution: Literal["server", "client"] = "server"
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    searcher: Optional[ToolSearchResolver] = None
    type: Literal["tool_search"] = field(init=False, default="tool_search")

    @classmethod
    def server(cls) -> "ToolSearch":
        return cls(execution="server")

    @classmethod
    def client(
        cls,
        *,
        description: str,
        parameters: Dict[str, Any],
        searcher: Optional[ToolSearchResolver] = None,
    ) -> "ToolSearch":
        return cls(execution="client", description=description, parameters=parameters, searcher=searcher)

    def to_dict(self) -> Dict[str, Any]:
        tool: Dict[str, Any] = {"type": self.type}
        if self.execution == "client":
            tool["execution"] = "client"
        if self.description is not None:
            tool["description"] = self.description
        if self.parameters is not None:
            tool["parameters"] = self.parameters
        return tool

    def is_async_searcher(self) -> bool:
        if self.searcher is None:
            return False

        from inspect import iscoroutinefunction

        return iscoroutinefunction(self.searcher)


@dataclass
class ToolNamespace:
    """OpenAI Responses namespace containing deferred or regular tools."""

    name: str
    description: str
    tools: List[Union[Function, Dict[str, Any], ToolSearch]]
    type: Literal["namespace"] = field(init=False, default="namespace")

    def with_tools(self, tools: List[Union[Function, Dict[str, Any], ToolSearch]]) -> "ToolNamespace":
        return replace(self, tools=tools)

    def to_dict(self, format_tool: Optional[Callable[[Any], Dict[str, Any]]] = None) -> Dict[str, Any]:
        formatter = format_tool or _default_format_tool
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "tools": [formatter(tool) for tool in self.tools],
        }


def _default_format_tool(tool: Any) -> Dict[str, Any]:
    if isinstance(tool, Function):
        tool_dict = tool.to_dict()
        tool_dict["type"] = "function"
        if tool.defer_loading is not None:
            tool_dict["defer_loading"] = tool.defer_loading
        return tool_dict
    if isinstance(tool, ToolSearch):
        return tool.to_dict()
    if isinstance(tool, ToolNamespace):
        return tool.to_dict()
    if isinstance(tool, dict):
        return tool
    if hasattr(tool, "to_dict"):
        return tool.to_dict()
    raise TypeError(f"Unsupported OpenAI tool type: {type(tool).__name__}")


async def maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value
