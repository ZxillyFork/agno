import asyncio
import collections.abc
import json
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from time import sleep, time
from types import AsyncGeneratorType, GeneratorType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

if TYPE_CHECKING:
    from agno.compression.manager import CompressionManager
    from agno.offload.store import ResultStore
from uuid import uuid4

from pydantic import BaseModel

from agno.exceptions import (
    AgentRunException,
    ContextWindowExceededError,
    ModelProviderError,
    RetryableModelProviderError,
    RunCancelledException,
    ToolApprovalRequired,
    ToolCallDeferred,
)
from agno.media import Audio, File, Image, Video
from agno.metrics import MessageMetrics, ModelType, ToolCallMetrics
from agno.models.message import Citations, Message
from agno.models.response import ModelResponse, ModelResponseEvent, ToolExecution
from agno.run.agent import RUN_OUTPUT_EVENT_TYPES, CustomEvent, RunContentEvent, RunOutput, RunOutputEvent
from agno.run.team import TEAM_RUN_OUTPUT_EVENT_TYPES, TeamRunOutput, TeamRunOutputEvent
from agno.run.team import RunContentEvent as TeamRunContentEvent
from agno.run.workflow import WORKFLOW_RUN_OUTPUT_EVENT_TYPES
from agno.tools.function import (
    Function,
    FunctionCall,
    FunctionExecutionResult,
    UserFeedbackOption,
    UserFeedbackQuestion,
    UserInputField,
)
from agno.utils.log import log_debug, log_error, log_info, log_warning
from agno.utils.timer import Timer
from agno.utils.tools import get_function_call_for_tool_call, get_function_call_for_tool_execution

# Every run-event type a tool-result generator can bubble up, cached once:
# these isinstance checks run per streamed item on the hot path.
_ALL_RUN_OUTPUT_EVENT_TYPES = RUN_OUTPUT_EVENT_TYPES + TEAM_RUN_OUTPUT_EVENT_TYPES + WORKFLOW_RUN_OUTPUT_EVENT_TYPES


@dataclass
class MessageData:
    response_role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    response_content: Any = ""
    response_reasoning_content: Any = ""
    response_redacted_reasoning_content: Any = ""
    response_citations: Optional[Citations] = None
    response_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_started_ids: Set[str] = field(default_factory=set)
    tool_call_direct_args_delta_ids: Set[str] = field(default_factory=set)
    # Provider-native ordinal -> tool_call_id (e.g. OpenAI Chat `index`,
    # OpenAI Responses `output_index`, Anthropic content_block `index`).
    # Lets snapshot synthesis attribute mid-stream args fragments whose chunks
    # carry only `index` (id only appears on the first chunk).
    tool_call_provider_index_to_id: Dict[int, str] = field(default_factory=dict)

    response_audio: Optional[Audio] = None
    response_image: Optional[Image] = None
    response_video: Optional[Video] = None
    response_file: Optional[File] = None

    response_metrics: Optional[MessageMetrics] = None

    # Data from the provider that we might need on subsequent messages
    response_provider_data: Optional[Dict[str, Any]] = None

    extra: Optional[Dict[str, Any]] = None


def _log_messages(messages: List[Message]) -> None:
    """
    Log messages for debugging.
    """
    for m in messages:
        # Don't log metrics for input messages
        m.log(metrics=False)


def _handle_agent_exception(a_exc: AgentRunException, additional_input: Optional[List[Message]] = None) -> None:
    """Handle AgentRunException and collect additional messages."""
    if additional_input is None:
        additional_input = []
    if a_exc.user_message is not None:
        msg = (
            Message(role="user", content=a_exc.user_message)
            if isinstance(a_exc.user_message, str)
            else a_exc.user_message
        )
        additional_input.append(msg)

    if a_exc.agent_message is not None:
        msg = (
            Message(role="assistant", content=a_exc.agent_message)
            if isinstance(a_exc.agent_message, str)
            else a_exc.agent_message
        )
        additional_input.append(msg)

    if a_exc.messages:
        for m in a_exc.messages:
            if isinstance(m, Message):
                additional_input.append(m)
            elif isinstance(m, dict):
                try:
                    additional_input.append(Message(**m))
                except Exception as e:
                    log_warning(f"Failed to convert dict to Message: {str(e)}")

    if a_exc.stop_execution:
        for m in additional_input:
            m.stop_after_tool_call = True


ToolPauseException = Union[ToolApprovalRequired, ToolCallDeferred]
_TOOL_CALL_BATCH_SETTLE_TIMEOUT_SECONDS = 10.0


def _is_tool_pause_exception(exc: Any) -> bool:
    return isinstance(exc, (ToolApprovalRequired, ToolCallDeferred))


def _create_paused_tool_execution(function_call: FunctionCall, exc: ToolPauseException) -> ToolExecution:
    """Create a paused ToolExecution from a dynamic HITL exception raised by a tool."""

    metadata = getattr(exc, "metadata", None)
    func = function_call.function

    if isinstance(exc, ToolApprovalRequired):
        approval_type = exc.approval_type or func.approval_type or "required"
        # A dynamic exception inherently aborts the tool and forces a pause. That
        # is incompatible with approval_type="audit" (a non-blocking audit trail),
        # so promote to "required" and warn rather than silently producing a
        # contradictory ToolExecution that would still block while claiming to
        # be audit-only.
        if approval_type == "audit":
            log_warning(
                f"Tool '{func.name}' raised ToolApprovalRequired with approval_type='audit', "
                "but a dynamic pause exception always blocks the run; treating as approval_type='required'."
            )
            approval_type = "required"

        return ToolExecution(
            tool_call_id=function_call.call_id,
            tool_name=func.name,
            tool_args=function_call.arguments,
            requires_confirmation=True,
            approval_type=approval_type,
            metadata=metadata,
            external_execution_silent=func.external_execution_silent,
        )

    return ToolExecution(
        tool_call_id=function_call.call_id,
        tool_name=func.name,
        tool_args=function_call.arguments,
        external_execution_required=True,
        approval_type=func.approval_type,
        metadata=metadata,
        external_execution_silent=func.external_execution_silent,
    )


def _create_tool_call_paused_response(function_call: FunctionCall, exc: ToolPauseException) -> ModelResponse:
    return ModelResponse(
        tool_executions=[_create_paused_tool_execution(function_call, exc)],
        event=ModelResponseEvent.tool_call_paused.value,
    )


def _create_tool_calls_paused_response(
    paused_calls: Sequence[Tuple[FunctionCall, ToolPauseException]],
) -> ModelResponse:
    return ModelResponse(
        tool_executions=[_create_paused_tool_execution(function_call, exc) for function_call, exc in paused_calls],
        event=ModelResponseEvent.tool_call_paused.value,
    )


def _handle_agent_exception_from_tool_call(
    function_call: FunctionCall,
    a_exc: AgentRunException,
    additional_input: Optional[List[Message]] = None,
) -> bool:
    _handle_agent_exception(a_exc, additional_input)
    function_call.error = str(a_exc)
    return a_exc.stop_execution


def _cancelled_tool_call_content(function_call: FunctionCall) -> str:
    return f"Tool call '{function_call.function.name}' was cancelled before it could complete."


def _is_get_user_input_call(function_call: FunctionCall) -> bool:
    return bool(
        function_call.function.name == "get_user_input"
        and function_call.arguments
        and function_call.arguments.get("user_input_fields")
    )


def _is_ask_user_call(function_call: FunctionCall) -> bool:
    return bool(
        function_call.function.name == "ask_user"
        and function_call.arguments
        and function_call.arguments.get("questions")
    )


def _is_static_pause_call(function_call: FunctionCall) -> bool:
    return bool(
        function_call.function.requires_confirmation
        or function_call.function.external_execution
        or (
            function_call.function.requires_user_input
            and not (_is_get_user_input_call(function_call) or _is_ask_user_call(function_call))
        )
        or _is_get_user_input_call(function_call)
        or _is_ask_user_call(function_call)
    )


def _create_static_paused_tool_executions(function_call: FunctionCall) -> List[ToolExecution]:
    paused_tool_execution: Optional[ToolExecution] = None

    def get_paused_tool_execution() -> ToolExecution:
        nonlocal paused_tool_execution
        if paused_tool_execution is None:
            paused_tool_execution = ToolExecution(
                tool_call_id=function_call.call_id,
                tool_name=function_call.function.name,
                tool_args=function_call.arguments,
                approval_type=function_call.function.approval_type,
                external_execution_silent=function_call.function.external_execution_silent,
            )
        return paused_tool_execution

    if function_call.function.requires_confirmation:
        get_paused_tool_execution().requires_confirmation = True

    if function_call.function.requires_user_input and not (
        _is_get_user_input_call(function_call) or _is_ask_user_call(function_call)
    ):
        user_input_schema = function_call.function.user_input_schema
        if function_call.arguments and user_input_schema:
            for name, value in function_call.arguments.items():
                for user_input_field in user_input_schema:
                    if user_input_field.name == name:
                        user_input_field.value = value

        paused_tool_execution = get_paused_tool_execution()
        paused_tool_execution.requires_user_input = True
        paused_tool_execution.user_input_schema = user_input_schema

    if _is_get_user_input_call(function_call):
        arguments = function_call.arguments or {}
        user_input_schema = []
        for input_field in arguments.get("user_input_fields", []):
            field_type = input_field.get("field_type")
            if isinstance(field_type, str):
                type_mapping = {
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                }
                python_type = type_mapping.get(field_type, str)
            elif isinstance(field_type, type):
                python_type = field_type
            else:
                python_type = str
            user_input_schema.append(
                UserInputField(
                    name=input_field.get("field_name"),
                    field_type=python_type,
                    description=input_field.get("field_description"),
                )
            )

        paused_tool_execution = get_paused_tool_execution()
        paused_tool_execution.requires_user_input = True
        paused_tool_execution.user_input_schema = user_input_schema

    if _is_ask_user_call(function_call):
        arguments = function_call.arguments or {}
        user_feedback_schema = []
        for question in arguments.get("questions", []):
            options = None
            if question.get("options"):
                options = [
                    UserFeedbackOption(label=option.get("label", ""), description=option.get("description"))
                    for option in question["options"]
                ]
            user_feedback_schema.append(
                UserFeedbackQuestion(
                    question=question.get("question", ""),
                    header=question.get("header"),
                    options=options,
                    multi_select=question.get("multi_select", False),
                )
            )

        paused_tool_execution = get_paused_tool_execution()
        paused_tool_execution.requires_user_input = True
        paused_tool_execution.user_feedback_schema = user_feedback_schema

    if function_call.function.external_execution:
        get_paused_tool_execution().external_execution_required = True

    return [paused_tool_execution] if paused_tool_execution is not None else []


def _function_call_uses_thread(function_call: FunctionCall) -> bool:
    from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction

    entrypoint = function_call.function.entrypoint
    return not (
        iscoroutinefunction(entrypoint)
        or isasyncgenfunction(entrypoint)
        or iscoroutine(entrypoint)
        or function_call._requires_async_execution()
    )


def _function_call_has_async_entrypoint(function_call: FunctionCall) -> bool:
    from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction

    entrypoint = function_call.function.entrypoint
    return bool(iscoroutinefunction(entrypoint) or isasyncgenfunction(entrypoint) or iscoroutine(entrypoint))


def _function_call_has_sync_entrypoint(function_call: FunctionCall) -> bool:
    return not _function_call_has_async_entrypoint(function_call)


def _function_call_is_sync_generator(function_call: FunctionCall) -> bool:
    from inspect import isgeneratorfunction, unwrap

    entrypoint = function_call.function.entrypoint
    if entrypoint is None:
        return False

    return bool(isgeneratorfunction(entrypoint) or isgeneratorfunction(unwrap(entrypoint)))


def _has_executable_tools(tools: Optional[List[Union[Function, dict]]]) -> bool:
    return bool(tools)


# Per-call run context, stored in a ContextVar so a single Model instance shared
# across concurrent runs (asyncio tasks / threads) never leaks one run's context
# into another. Each task/thread gets its own value; sync save/restore uses tokens.
_CURRENT_RUN_CONTEXT_VAR: ContextVar[Optional[Any]] = ContextVar("agno_current_run_context", default=None)


@dataclass
class Model(ABC):
    # ID of the model to use.
    id: str
    # Name for this Model. This is not sent to the Model API.
    name: Optional[str] = None
    # Provider for this Model. This is not sent to the Model API.
    provider: Optional[str] = None
    # Functional role of this model (e.g., MODEL, OUTPUT_MODEL, PARSER_MODEL).
    # Set by the agent during initialization; defaults to MODEL.
    model_type: ModelType = ModelType.MODEL

    # -*- Do not set the following attributes directly -*-
    # -*- Set them on the Agent instead -*-

    # True if the Model supports structured outputs natively (e.g. OpenAI)
    supports_native_structured_outputs: bool = False
    # True if the Model requires a json_schema for structured outputs (e.g. LMStudio)
    supports_json_schema_outputs: bool = False

    # Controls which (if any) function is called by the model.
    # "none" means the model will not call a function and instead generates a message.
    # "auto" means the model can pick between generating a message or calling a function.
    # Specifying a particular function via {"type: "function", "function": {"name": "my_function"}}
    #   forces the model to call that function.
    # "none" is the default when no functions are present. "auto" is the default if functions are present.
    _tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # System prompt from the model added to the Agent.
    system_prompt: Optional[str] = None
    # Instructions from the model added to the Agent.
    instructions: Optional[List[str]] = None

    # The role of the tool message.
    tool_message_role: str = "tool"
    # The role of the assistant message.
    assistant_message_role: str = "assistant"

    # Cache model responses to avoid redundant API calls during development
    cache_response: bool = False
    cache_ttl: Optional[int] = None
    cache_dir: Optional[str] = None

    # Retry configuration for model provider errors
    # Number of retries to attempt when a ModelProviderError occurs
    retries: int = 0
    # Delay between retries (in seconds)
    delay_between_retries: int = 1
    # Exponential backoff: if True, the delay between retries is doubled each time
    exponential_backoff: bool = False
    # Enable retrying a model invocation once with a guidance message.
    # This is useful for known errors avoidable with extra instructions.
    retry_with_guidance: bool = True
    # Set the number of times to retry the model invocation with guidance.
    retry_with_guidance_limit: int = 1

    def __post_init__(self):
        if self.provider is None and self.name is not None:
            self.provider = f"{self.name} ({self.id})"

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate the delay before the next retry attempt."""
        if self.exponential_backoff:
            return self.delay_between_retries * (2**attempt)
        return self.delay_between_retries

    def _is_retryable_error(self, error: ModelProviderError) -> bool:
        """Determine if an error is worth retrying.

        Non-retryable errors include:
        - ContextWindowExceededError (fast path after ModelProviderError.classify)
        - Client errors (400, 401, 403, 404, 413, 422) that won't change on retry
        - Context window/token limit patterns in error message (defense-in-depth)

        Retryable errors include:
        - Rate limit errors (429)
        - Server errors (500, 502, 503, 504)
        - Anything else not explicitly non-retryable
        """
        # Fast path: already classified by ModelProviderError.classify()
        if isinstance(error, ContextWindowExceededError):
            return False

        non_retryable_codes = {400, 401, 403, 404, 413, 422}
        if error.status_code in non_retryable_codes:
            return False

        # Defense-in-depth: catch context window errors even if not pre-classified
        error_msg = str(error.message).lower()
        if any(pattern in error_msg for pattern in ModelProviderError.CONTEXT_WINDOW_PATTERNS):
            return False

        return True

    def _invoke_with_retry(self, **kwargs) -> ModelResponse:
        """
        Invoke the model with retry logic for ModelProviderError.

        This method wraps the invoke() call and retries on ModelProviderError
        with optional exponential backoff.
        """
        last_exception: Optional[ModelProviderError] = None
        retries_with_guidance_count = kwargs.pop("retries_with_guidance_count", 0)

        for attempt in range(self.retries + 1):
            try:
                return self.invoke(**kwargs)
            except ModelProviderError as e:
                last_exception = ModelProviderError.classify(e)
                # Check if error is non-retryable
                if not self._is_retryable_error(last_exception):
                    log_error(f"Non-retryable model provider error: {str(e)}")
                    raise last_exception from e
                if attempt < self.retries:
                    delay = self._get_retry_delay(attempt)
                    log_warning(
                        f"Model provider error (attempt {attempt + 1}/{self.retries + 1}): {last_exception}. Retrying in {delay}s...: {e}",
                    )

                    sleep(delay)
                else:
                    if self.retries > 0:
                        log_error(f"Model provider error after {self.retries + 1} attempts: {str(e)}")
            except RetryableModelProviderError as e:
                current_count = retries_with_guidance_count
                if current_count >= self.retry_with_guidance_limit:
                    raise ModelProviderError(
                        message=f"Max retries with guidance reached. Error: {e.original_error}",
                        model_name=self.name,
                        model_id=self.id,
                    )
                kwargs.pop("retry_with_guidance", None)
                kwargs["retries_with_guidance_count"] = current_count + 1

                # Append the guidance message to help the model avoid the error in the next invoke.
                kwargs["messages"].append(Message(role="user", content=e.retry_guidance_message, temporary=True))

                return self._invoke_with_retry(**kwargs, retry_with_guidance=True)

        # If we've exhausted all retries, raise the last exception
        raise last_exception  # type: ignore

    async def _ainvoke_with_retry(self, **kwargs) -> ModelResponse:
        """
        Asynchronously invoke the model with retry logic for ModelProviderError.

        This method wraps the ainvoke() call and retries on ModelProviderError
        with optional exponential backoff.
        """
        last_exception: Optional[ModelProviderError] = None
        retries_with_guidance_count = kwargs.pop("retries_with_guidance_count", 0)

        for attempt in range(self.retries + 1):
            try:
                return await self.ainvoke(**kwargs)
            except ModelProviderError as e:
                last_exception = ModelProviderError.classify(e)
                # Check if error is non-retryable
                if not self._is_retryable_error(last_exception):
                    log_error(f"Non-retryable model provider error: {str(e)}")
                    raise last_exception from e
                if attempt < self.retries:
                    delay = self._get_retry_delay(attempt)
                    log_warning(
                        f"Model provider error (attempt {attempt + 1}/{self.retries + 1}): {last_exception}. Retrying in {delay}s...: {e}",
                    )

                    await asyncio.sleep(delay)
                else:
                    if self.retries > 0:
                        log_error(f"Model provider error after {self.retries + 1} attempts: {str(e)}")
            except RetryableModelProviderError as e:
                current_count = retries_with_guidance_count
                if current_count >= self.retry_with_guidance_limit:
                    raise ModelProviderError(
                        message=f"Max retries with guidance reached. Error: {e.original_error}",
                        model_name=self.name,
                        model_id=self.id,
                    )

                kwargs.pop("retry_with_guidance", None)
                kwargs["retries_with_guidance_count"] = current_count + 1

                # Append the guidance message to help the model avoid the error in the next invoke.
                kwargs["messages"].append(Message(role="user", content=e.retry_guidance_message, temporary=True))

                return await self._ainvoke_with_retry(**kwargs, retry_with_guidance=True)

        # If we've exhausted all retries, raise the last exception
        raise last_exception  # type: ignore

    def _invoke_stream_with_retry(self, **kwargs) -> Iterator[ModelResponse]:
        """
        Invoke the model stream with retry logic for ModelProviderError.

        This method wraps the invoke_stream() call and retries on ModelProviderError
        with optional exponential backoff. Note that retries restart the entire stream.
        """
        last_exception: Optional[ModelProviderError] = None
        retries_with_guidance_count = kwargs.pop("retries_with_guidance_count", 0)

        for attempt in range(self.retries + 1):
            try:
                yield from self.invoke_stream(**kwargs)
                return  # Success, exit the retry loop
            except ModelProviderError as e:
                last_exception = ModelProviderError.classify(e)
                # Check if error is non-retryable (e.g., context window exceeded, auth errors)
                if not self._is_retryable_error(last_exception):
                    log_error(f"Non-retryable model provider error: {str(e)}")
                    raise last_exception from e
                if attempt < self.retries:
                    delay = self._get_retry_delay(attempt)
                    log_warning(
                        f"Model provider error during stream (attempt {attempt + 1}/{self.retries + 1}): {last_exception}. : {e}"
                        f"Retrying in {delay}s...: {e}",
                    )

                    sleep(delay)
                else:
                    if self.retries > 0:
                        log_error(f"Model provider error after {self.retries + 1} attempts: {str(e)}")
            except RetryableModelProviderError as e:
                current_count = retries_with_guidance_count
                if current_count >= self.retry_with_guidance_limit:
                    raise ModelProviderError(
                        message=f"Max retries with guidance reached. Error: {e.original_error}",
                        model_name=self.name,
                        model_id=self.id,
                    )

                kwargs.pop("retry_with_guidance", None)
                kwargs["retries_with_guidance_count"] = current_count + 1

                # Append the guidance message to help the model avoid the error in the next invoke.
                kwargs["messages"].append(Message(role="user", content=e.retry_guidance_message, temporary=True))

                yield from self._invoke_stream_with_retry(**kwargs, retry_with_guidance=True)
                return  # Success, exit after regeneration

        # If we've exhausted all retries, raise the last exception
        raise last_exception  # type: ignore

    async def _ainvoke_stream_with_retry(self, **kwargs) -> AsyncIterator[ModelResponse]:
        """
        Asynchronously invoke the model stream with retry logic for ModelProviderError.

        This method wraps the ainvoke_stream() call and retries on ModelProviderError
        with optional exponential backoff. Note that retries restart the entire stream.
        """
        last_exception: Optional[ModelProviderError] = None
        retries_with_guidance_count = kwargs.pop("retries_with_guidance_count", 0)

        for attempt in range(self.retries + 1):
            try:
                async for response in self.ainvoke_stream(**kwargs):
                    yield response
                return  # Success, exit the retry loop
            except ModelProviderError as e:
                last_exception = ModelProviderError.classify(e)
                # Check if error is non-retryable
                if not self._is_retryable_error(last_exception):
                    log_error(f"Non-retryable model provider error: {str(e)}")
                    raise last_exception from e
                if attempt < self.retries:
                    delay = self._get_retry_delay(attempt)
                    log_warning(
                        f"Model provider error during stream (attempt {attempt + 1}/{self.retries + 1}): {last_exception}. : {e}"
                        f"Retrying in {delay}s...: {e}",
                    )

                    await asyncio.sleep(delay)
                else:
                    if self.retries > 0:
                        log_error(f"Model provider error after {self.retries + 1} attempts: {str(e)}")
            except RetryableModelProviderError as e:
                current_count = retries_with_guidance_count
                if current_count >= self.retry_with_guidance_limit:
                    raise ModelProviderError(
                        message=f"Max retries with guidance reached. Error: {e.original_error}",
                        model_name=self.name,
                        model_id=self.id,
                    )

                kwargs.pop("retry_with_guidance", None)
                kwargs["retries_with_guidance_count"] = current_count + 1

                # Append the guidance message to help the model avoid the error in the next invoke.
                kwargs["messages"].append(Message(role="user", content=e.retry_guidance_message, temporary=True))

                async for response in self._ainvoke_stream_with_retry(**kwargs, retry_with_guidance=True):
                    yield response
                return  # Success, exit after regeneration

        # If we've exhausted all retries, raise the last exception
        raise last_exception  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        fields = {"name", "id", "provider"}
        _dict = {field: getattr(self, field) for field in fields if getattr(self, field) is not None}
        return _dict

    def _remove_temporary_messages(self, messages: List[Message]) -> None:
        """Remove temporary messages from the given list.

        Args:
            messages: The list of messages to filter (modified in place).
        """
        messages[:] = [m for m in messages if not m.temporary]

    def get_provider(self) -> str:
        return self.provider or self.name or self.__class__.__name__

    def _get_model_cache_key(self, messages: List[Message], stream: bool, **kwargs: Any) -> str:
        """Generate a cache key based on model messages and core parameters."""
        message_data = []
        for msg in messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
            }
            message_data.append(msg_dict)

        # Include tools parameter in cache key
        has_tools = bool(kwargs.get("tools"))

        cache_data = {
            "model_id": self.id,
            "messages": message_data,
            "has_tools": has_tools,
            "response_format": kwargs.get("response_format"),
            "stream": stream,
        }

        def _cache_default(obj: Any) -> Any:
            if isinstance(obj, type):
                return obj.__name__
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        cache_str = json.dumps(cache_data, sort_keys=True, default=_cache_default)
        return md5(cache_str.encode()).hexdigest()

    def _get_model_cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
        else:
            cache_dir = Path.home() / ".agno" / "cache" / "model_responses"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}.json"

    def _get_cached_model_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if it exists and is not expired."""
        cache_file = self._get_model_cache_file_path(cache_key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            # Check TTL if set (None means no expiration)
            if self.cache_ttl is not None:
                if time() - cached_data["timestamp"] > self.cache_ttl:
                    return None

            return cached_data
        except Exception:
            return None

    def _save_model_response_to_cache(self, cache_key: str, result: ModelResponse, is_streaming: bool = False) -> None:
        """Save a model response to cache."""
        try:
            cache_file = self._get_model_cache_file_path(cache_key)

            cache_data = {
                "timestamp": int(time()),
                "is_streaming": is_streaming,
                "result": result.to_dict(),
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
        except Exception:
            pass

    def _save_streaming_responses_to_cache(self, cache_key: str, responses: List[ModelResponse]) -> None:
        """Save streaming responses to cache."""
        cache_file = self._get_model_cache_file_path(cache_key)

        cache_data = {
            "timestamp": int(time()),
            "is_streaming": True,
            "streaming_responses": [r.to_dict() for r in responses],
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
        except Exception:
            pass

    def _model_response_from_cache(self, cached_data: Dict[str, Any]) -> ModelResponse:
        """Reconstruct a ModelResponse from cached data."""
        return ModelResponse.from_dict(cached_data["result"])

    def _streaming_responses_from_cache(self, cached_data: list) -> Iterator[ModelResponse]:
        """Reconstruct streaming responses from cached data."""
        for cached_response in cached_data:
            yield ModelResponse.from_dict(cached_response)

    @abstractmethod
    def invoke(self, *args, **kwargs) -> ModelResponse:
        pass

    @abstractmethod
    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        pass

    @abstractmethod
    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        pass

    @abstractmethod
    def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        pass

    @abstractmethod
    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        """
        Parse the raw response from the model provider into a ModelResponse.

        Args:
            response: Raw response from the model provider

        Returns:
            ModelResponse: Parsed response data
        """
        pass

    @abstractmethod
    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        """
        Parse the streaming response from the model provider into ModelResponse objects.

        Args:
            response: Raw response chunk from the model provider

        Returns:
            ModelResponse: Parsed response delta
        """
        pass

    @staticmethod
    def _tool_name(t: Any) -> str:
        """Extract a tool's name for deterministic sorting."""
        if isinstance(t, dict):
            fn = t.get("function")
            if isinstance(fn, dict):
                return str(fn.get("name", ""))
            return str(t.get("name", ""))
        # Non-dict objects (e.g. provider tool configs) — sort by name/type attribute.
        return str(getattr(t, "name", None) or getattr(t, "type", "") or "")

    def _format_tools(self, tools: Optional[List[Union[Function, dict]]]) -> List[Dict[str, Any]]:
        _tool_dicts = []
        for tool in tools or []:
            if isinstance(tool, Function):
                _tool_dicts.append({"type": "function", "function": tool.to_dict()})
            elif getattr(tool, "type", None) in {"namespace", "tool_search"} and not isinstance(tool, dict):
                # ToolSearch / ToolNamespace are OpenAI Responses-specific helpers. Reaching the
                # base formatter means they were passed to a provider that cannot consume them.
                raise ValueError(
                    f"Tool of type '{getattr(tool, 'type', None)}' ({type(tool).__name__}) is only "
                    f"supported by the OpenAIResponses model; it cannot be used with {self.name or type(self).__name__}."
                )
            else:
                # If a dict is passed, it is a builtin tool
                _tool_dicts.append(tool)
        # Deterministic ordering so prompt caching gets consistent cache hits.
        # Applies across providers — Anthropic, OpenAI, and Gemini prompt/context
        # caching all require stable request prefixes.
        _tool_dicts.sort(key=self._tool_name)
        return _tool_dicts

    @property
    def _current_run_context(self) -> Optional[Any]:
        """The run context for the in-flight model call (per task/thread, never shared)."""
        return _CURRENT_RUN_CONTEXT_VAR.get()

    def _get_functions_from_tools(
        self, tools: Optional[List[Union[Function, dict]]], include_dynamic_functions: bool = False
    ) -> Dict[str, Function]:
        return {tool.name: tool for tool in tools if isinstance(tool, Function)} if tools is not None else {}

    def _ensure_message_metrics_initialized(self, assistant_message: Message) -> None:
        """
        Ensure message metrics are initialized and timer is started.

        Args:
            assistant_message: The assistant message to initialize metrics for
        """
        if assistant_message.metrics is None:
            assistant_message.metrics = MessageMetrics()
        if assistant_message.metrics.timer is None or assistant_message.metrics.timer.start_time is None:
            assistant_message.metrics.start_timer()

    def count_tokens(
        self,
        messages: List[Message],
        tools: Optional[Sequence[Union[Function, Dict[str, Any]]]] = None,
        output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> int:
        from agno.utils.tokens import count_tokens

        return count_tokens(
            messages,
            tools=list(tools) if tools else None,
            model_id=self.id,
            output_schema=output_schema,
        )

    async def acount_tokens(
        self,
        messages: List[Message],
        tools: Optional[Sequence[Union[Function, Dict[str, Any]]]] = None,
        output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> int:
        return self.count_tokens(messages, tools, output_schema=output_schema)

    def response(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        send_media_to_model: bool = True,
        compression_manager: Optional["CompressionManager"] = None,
        result_store: Optional["ResultStore"] = None,
        after_tool_results: Optional[Callable[["ModelResponse"], None]] = None,
        run_context: Optional[Any] = None,
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            messages: List of messages to send to the model
            response_format: Response format to use
            tools: List of tools to use. This includes the original Function objects and dicts for built-in tools.
            tool_choice: Tool choice to use
            tool_call_limit: Tool call limit
            run_response: Run response to use
            send_media_to_model: Whether to send media to the model
            after_tool_results: Optional callback invoked once per tool batch, after tool result
                messages are appended to ``messages`` and before the next model call (or break).
                Receives the current ``ModelResponse`` (with accumulated ``tool_executions``)
                as its single argument. Used by Agent-level checkpointing
                (``checkpoint="tool-batch"``) to persist mid-run state. Exceptions are caught and
                logged — a failed callback must not kill the run.
        """
        _run_context_token = _CURRENT_RUN_CONTEXT_VAR.set(run_context)
        try:
            # Check cache if enabled
            cache_key = None
            cache_model_response = self.cache_response and not _has_executable_tools(tools)
            if cache_model_response:
                cache_key = self._get_model_cache_key(
                    messages, stream=False, response_format=response_format, tools=tools
                )
                cached_data = self._get_cached_model_response(cache_key)

                if cached_data:
                    log_info("Cache hit for model response")
                    return self._model_response_from_cache(cached_data)

            log_debug(f"{self.get_provider()} Response Start", center=True, symbol="-")
            log_debug(f"Model: {self.id}", center=True, symbol="-")

            _log_messages(messages)
            model_response = ModelResponse()

            function_call_count = 0

            _tool_dicts = self._format_tools(tools) if tools is not None else []
            _functions = self._get_functions_from_tools(tools)

            _compress_tool_results = compression_manager is not None and compression_manager.compress_tool_results
            _compression_manager = compression_manager if _compress_tool_results else None

            while True:
                if _compression_manager is not None and _compression_manager.should_compress(
                    messages, tools, model=self, response_format=response_format
                ):
                    _compression_manager.compress(
                        messages, run_metrics=run_response.metrics if run_response is not None else None
                    )

                assistant_message = Message(role=self.assistant_message_role)
                self._ensure_message_metrics_initialized(assistant_message)
                self._process_model_response(
                    messages=messages,
                    assistant_message=assistant_message,
                    model_response=model_response,
                    response_format=response_format,
                    tools=_tool_dicts,
                    tool_choice=tool_choice or self._tool_choice,
                    run_response=run_response,
                    compress_tool_results=_compress_tool_results,
                )

                if run_response is not None and model_response.response_usage is not None:
                    from agno.metrics import accumulate_model_metrics

                    accumulate_model_metrics(model_response, self, self.model_type, run_response.metrics)

                messages.append(assistant_message)
                assistant_message.log(metrics=True, use_compressed_content=_compress_tool_results)

                if assistant_message.tool_calls:
                    function_calls_to_run = self._prepare_function_calls(
                        assistant_message=assistant_message,
                        messages=messages,
                        model_response=model_response,
                        functions=_functions,
                    )
                    function_call_results: List[Message] = []
                    tool_call_paused = False

                    for function_call_response in self.run_function_calls(
                        function_calls=function_calls_to_run,
                        function_call_results=function_call_results,
                        current_function_call_count=function_call_count,
                        function_call_limit=tool_call_limit,
                        result_store=result_store,
                    ):
                        if isinstance(function_call_response, ModelResponse):
                            if function_call_response.updated_session_state is not None:
                                model_response.updated_session_state = function_call_response.updated_session_state

                            if function_call_response.images is not None:
                                if model_response.images is None:
                                    model_response.images = []
                                model_response.images.extend(function_call_response.images)

                            if function_call_response.audios is not None:
                                if model_response.audios is None:
                                    model_response.audios = []
                                model_response.audios.extend(function_call_response.audios)

                            if function_call_response.videos is not None:
                                if model_response.videos is None:
                                    model_response.videos = []
                                model_response.videos.extend(function_call_response.videos)

                            if function_call_response.files is not None:
                                if model_response.files is None:
                                    model_response.files = []
                                model_response.files.extend(function_call_response.files)

                            if (
                                function_call_response.event
                                in [
                                    ModelResponseEvent.tool_call_completed.value,
                                    ModelResponseEvent.tool_call_paused.value,
                                ]
                                and function_call_response.tool_executions is not None
                            ):
                                if model_response.tool_executions is None:
                                    model_response.tool_executions = []
                                model_response.tool_executions.extend(function_call_response.tool_executions)

                                if function_call_response.event == ModelResponseEvent.tool_call_paused.value:
                                    tool_call_paused = True

                            elif function_call_response.event not in [
                                ModelResponseEvent.tool_call_started.value,
                                ModelResponseEvent.tool_call_completed.value,
                            ]:
                                if function_call_response.content:
                                    model_response.content += function_call_response.content  # type: ignore

                    function_call_count += self._limit_charge_for(function_call_results, result_store)

                    self.format_function_call_results(
                        messages=messages,
                        function_call_results=function_call_results,
                        compress_tool_results=_compress_tool_results,
                        **model_response.extra or {},
                    )

                    if any(msg.images or msg.videos or msg.audio or msg.files for msg in function_call_results):
                        self._handle_function_call_media(
                            messages=messages,
                            function_call_results=function_call_results,
                            send_media_to_model=send_media_to_model,
                        )

                    for function_call_result in function_call_results:
                        function_call_result.log(metrics=True, use_compressed_content=_compress_tool_results)

                    if any(m.stop_after_tool_call for m in function_call_results):
                        break

                    if after_tool_results is not None:
                        try:
                            after_tool_results(model_response)
                        except Exception as e:
                            log_error(f"after_tool_results callback failed: {e}")

                    if tool_call_paused:
                        break

                    if any(tc.requires_confirmation for tc in model_response.tool_executions or []):
                        break

                    if any(tc.external_execution_required for tc in model_response.tool_executions or []):
                        break

                    if any(tc.requires_user_input for tc in model_response.tool_executions or []):
                        break

                    if run_response is not None and run_response.requirements:
                        if any(not req.is_resolved() for req in run_response.requirements):
                            break

                    continue

                break

            log_debug(f"{self.get_provider()} Response End", center=True, symbol="-")

            if cache_model_response and cache_key is not None:
                self._save_model_response_to_cache(cache_key, model_response, is_streaming=False)
        finally:
            _CURRENT_RUN_CONTEXT_VAR.reset(_run_context_token)
            if self.__class__.__name__ == "Gemini" and self.client is not None:  # type: ignore
                try:
                    self.client.close()  # type: ignore
                    self.client = None
                except AttributeError as e:
                    log_warning(
                        f"Your Gemini client is outdated. For Agno to properly handle the lifecycle of the client,: {e}"
                        f" please upgrade Gemini to the latest version: pip install -U google-genai: {e}",
                    )

        return model_response

    async def aresponse(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        send_media_to_model: bool = True,
        compression_manager: Optional["CompressionManager"] = None,
        result_store: Optional["ResultStore"] = None,
        after_tool_results: Optional[Callable[["ModelResponse"], Awaitable[None]]] = None,
        run_context: Optional[Any] = None,
    ) -> ModelResponse:
        """
        Generate an asynchronous response from the model.

        ``after_tool_results``: optional async callback invoked once per tool batch, after tool
        result messages are appended to ``messages`` and before the next model call (or break).
        Receives the current ``ModelResponse`` (with accumulated ``tool_executions``) as its
        single argument. Used by Agent-level checkpointing (``checkpoint="tool-batch"``) to persist
        mid-run state. Exceptions are caught and logged — a failed callback must not kill the run.
        """

        _run_context_token = _CURRENT_RUN_CONTEXT_VAR.set(run_context)
        try:
            # Check cache if enabled
            cache_key = None
            cache_model_response = self.cache_response and not _has_executable_tools(tools)
            if cache_model_response:
                cache_key = self._get_model_cache_key(
                    messages, stream=False, response_format=response_format, tools=tools
                )
                cached_data = self._get_cached_model_response(cache_key)

                if cached_data:
                    log_info("Cache hit for model response")
                    return self._model_response_from_cache(cached_data)

            log_debug(f"{self.get_provider()} Async Response Start", center=True, symbol="-")
            log_debug(f"Model: {self.id}", center=True, symbol="-")
            _log_messages(messages)
            model_response = ModelResponse()

            _tool_dicts = self._format_tools(tools) if tools is not None else []
            _functions = self._get_functions_from_tools(tools)

            _compress_tool_results = compression_manager is not None and compression_manager.compress_tool_results
            _compression_manager = compression_manager if _compress_tool_results else None

            function_call_count = 0

            while True:
                if _compression_manager is not None and await _compression_manager.ashould_compress(
                    messages, tools, model=self, response_format=response_format
                ):
                    await _compression_manager.acompress(
                        messages, run_metrics=run_response.metrics if run_response is not None else None
                    )

                assistant_message = Message(role=self.assistant_message_role)
                self._ensure_message_metrics_initialized(assistant_message)
                await self._aprocess_model_response(
                    messages=messages,
                    assistant_message=assistant_message,
                    model_response=model_response,
                    response_format=response_format,
                    tools=_tool_dicts,
                    tool_choice=tool_choice or self._tool_choice,
                    run_response=run_response,
                    compress_tool_results=_compress_tool_results,
                )

                if run_response is not None and model_response.response_usage is not None:
                    from agno.metrics import accumulate_model_metrics

                    accumulate_model_metrics(model_response, self, self.model_type, run_response.metrics)

                messages.append(assistant_message)
                assistant_message.log(metrics=True)

                if assistant_message.tool_calls:
                    function_calls_to_run = self._prepare_function_calls(
                        assistant_message=assistant_message,
                        messages=messages,
                        model_response=model_response,
                        functions=_functions,
                    )
                    function_call_results: List[Message] = []
                    tool_call_paused = False

                    async for function_call_response in self.arun_function_calls(
                        function_calls=function_calls_to_run,
                        function_call_results=function_call_results,
                        current_function_call_count=function_call_count,
                        function_call_limit=tool_call_limit,
                        result_store=result_store,
                        run_id=run_response.run_id if run_response else None,
                    ):
                        if isinstance(function_call_response, ModelResponse):
                            if function_call_response.updated_session_state is not None:
                                model_response.updated_session_state = function_call_response.updated_session_state

                            if function_call_response.images is not None:
                                if model_response.images is None:
                                    model_response.images = []
                                model_response.images.extend(function_call_response.images)

                            if function_call_response.audios is not None:
                                if model_response.audios is None:
                                    model_response.audios = []
                                model_response.audios.extend(function_call_response.audios)

                            if function_call_response.videos is not None:
                                if model_response.videos is None:
                                    model_response.videos = []
                                model_response.videos.extend(function_call_response.videos)

                            if function_call_response.files is not None:
                                if model_response.files is None:
                                    model_response.files = []
                                model_response.files.extend(function_call_response.files)

                            if (
                                function_call_response.event
                                in [
                                    ModelResponseEvent.tool_call_completed.value,
                                    ModelResponseEvent.tool_call_paused.value,
                                ]
                                and function_call_response.tool_executions is not None
                            ):
                                if model_response.tool_executions is None:
                                    model_response.tool_executions = []
                                model_response.tool_executions.extend(function_call_response.tool_executions)

                                if function_call_response.event == ModelResponseEvent.tool_call_paused.value:
                                    tool_call_paused = True

                            elif function_call_response.event not in [
                                ModelResponseEvent.tool_call_started.value,
                                ModelResponseEvent.tool_call_completed.value,
                            ]:
                                if function_call_response.content:
                                    model_response.content += function_call_response.content  # type: ignore

                    function_call_count += self._limit_charge_for(function_call_results, result_store)

                    self.format_function_call_results(
                        messages=messages,
                        function_call_results=function_call_results,
                        compress_tool_results=_compress_tool_results,
                        **model_response.extra or {},
                    )

                    if any(msg.images or msg.videos or msg.audio or msg.files for msg in function_call_results):
                        self._handle_function_call_media(
                            messages=messages,
                            function_call_results=function_call_results,
                            send_media_to_model=send_media_to_model,
                        )

                    for function_call_result in function_call_results:
                        function_call_result.log(metrics=True, use_compressed_content=_compress_tool_results)

                    if any(m.stop_after_tool_call for m in function_call_results):
                        break

                    if after_tool_results is not None:
                        try:
                            await after_tool_results(model_response)
                        except Exception as e:
                            log_error(f"after_tool_results callback failed: {e}")

                    if tool_call_paused:
                        break

                    if any(tc.requires_confirmation for tc in model_response.tool_executions or []):
                        break

                    if any(tc.external_execution_required for tc in model_response.tool_executions or []):
                        break

                    if any(tc.requires_user_input for tc in model_response.tool_executions or []):
                        break

                    if run_response is not None and run_response.requirements:
                        if any(not req.is_resolved() for req in run_response.requirements):
                            break

                    continue

                break

            log_debug(f"{self.get_provider()} Async Response End", center=True, symbol="-")

            if cache_model_response and cache_key is not None:
                self._save_model_response_to_cache(cache_key, model_response, is_streaming=False)
        finally:
            _CURRENT_RUN_CONTEXT_VAR.reset(_run_context_token)
            if self.__class__.__name__ == "Gemini" and self.client is not None:
                try:
                    await self.client.aio.aclose()  # type: ignore
                    self.client = None
                except AttributeError as e:
                    log_warning(
                        f"Your Gemini client is outdated. For Agno to properly handle the lifecycle of the client,: {e}"
                        f" please upgrade Gemini to the latest version: pip install -U google-genai: {e}",
                    )

        return model_response

    def _process_model_response(
        self,
        messages: List[Message],
        assistant_message: Message,
        model_response: ModelResponse,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> None:
        """
        Process a single model response and return the assistant message and whether to continue.

        Returns:
            Tuple[Message, bool]: (assistant_message, should_continue)
        """
        # Generate response with retry logic for ModelProviderError
        provider_response = self._invoke_with_retry(
            assistant_message=assistant_message,
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice or self._tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

        # Set TTFT after response arrives (guard ensures first-call-wins)
        if run_response and run_response.metrics:
            run_response.metrics.set_time_to_first_token()

        # Populate the assistant message
        self._populate_assistant_message(assistant_message=assistant_message, provider_response=provider_response)

        # Update model response with assistant message content and audio
        if assistant_message.content is not None:
            if model_response.content is None:
                model_response.content = assistant_message.get_content_string()
            else:
                model_response.content += assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        if assistant_message.redacted_reasoning_content is not None:
            model_response.redacted_reasoning_content = assistant_message.redacted_reasoning_content
        if assistant_message.citations is not None:
            model_response.citations = assistant_message.citations
        if assistant_message.audio_output is not None:
            if isinstance(assistant_message.audio_output, Audio):
                model_response.audio = assistant_message.audio_output
        if assistant_message.image_output is not None:
            model_response.images = [assistant_message.image_output]
        if assistant_message.video_output is not None:
            model_response.videos = [assistant_message.video_output]
        if provider_response.extra is not None:
            if model_response.extra is None:
                model_response.extra = {}
            model_response.extra.update(provider_response.extra)
        if provider_response.provider_data is not None:
            model_response.provider_data = provider_response.provider_data
        if provider_response.response_usage is not None:
            model_response.response_usage = provider_response.response_usage
        # Providers (e.g. GeminiInteractions on the agent path) can produce
        # already-executed ToolExecution records server-side; carry them
        # through so run_response.tools / AgentOS UI sees the audit.
        if provider_response.tool_executions:
            if model_response.tool_executions is None:
                model_response.tool_executions = []
            model_response.tool_executions.extend(provider_response.tool_executions)

    async def _aprocess_model_response(
        self,
        messages: List[Message],
        assistant_message: Message,
        model_response: ModelResponse,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> None:
        """
        Process a single async model response and return the assistant message and whether to continue.

        Returns:
            Tuple[Message, bool]: (assistant_message, should_continue)
        """
        # Generate response with retry logic for ModelProviderError
        provider_response = await self._ainvoke_with_retry(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice or self._tool_choice,
            assistant_message=assistant_message,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

        # Set TTFT after response arrives (guard ensures first-call-wins)
        if run_response and run_response.metrics:
            run_response.metrics.set_time_to_first_token()

        # Populate the assistant message
        self._populate_assistant_message(assistant_message=assistant_message, provider_response=provider_response)

        # Update model response with assistant message content and audio
        if assistant_message.content is not None:
            if model_response.content is None:
                model_response.content = assistant_message.get_content_string()
            else:
                model_response.content += assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        if assistant_message.redacted_reasoning_content is not None:
            model_response.redacted_reasoning_content = assistant_message.redacted_reasoning_content
        if assistant_message.citations is not None:
            model_response.citations = assistant_message.citations
        if assistant_message.audio_output is not None:
            if isinstance(assistant_message.audio_output, Audio):
                model_response.audio = assistant_message.audio_output
        if assistant_message.image_output is not None:
            model_response.images = [assistant_message.image_output]
        if assistant_message.video_output is not None:
            model_response.videos = [assistant_message.video_output]
        if provider_response.extra is not None:
            if model_response.extra is None:
                model_response.extra = {}
            model_response.extra.update(provider_response.extra)
        if provider_response.provider_data is not None:
            model_response.provider_data = provider_response.provider_data
        if provider_response.response_usage is not None:
            model_response.response_usage = provider_response.response_usage
        # Providers (e.g. GeminiInteractions on the agent path) can produce
        # already-executed ToolExecution records server-side; carry them
        # through so run_response.tools / AgentOS UI sees the audit.
        if provider_response.tool_executions:
            if model_response.tool_executions is None:
                model_response.tool_executions = []
            model_response.tool_executions.extend(provider_response.tool_executions)

    def _populate_assistant_message(
        self,
        assistant_message: Message,
        provider_response: ModelResponse,
    ) -> Message:
        """
        Populate an assistant message with the provider response data.

        Args:
            assistant_message: The assistant message to populate
            provider_response: Parsed response from the model provider

        Returns:
            Message: The populated assistant message
        """
        if provider_response.role is not None:
            assistant_message.role = provider_response.role

        # Add content to assistant message
        if provider_response.content is not None:
            assistant_message.content = provider_response.content
            # Set time_to_first_token when we receive content if not already set
            if assistant_message.metrics is not None and assistant_message.metrics.time_to_first_token is None:
                assistant_message.metrics.set_time_to_first_token()

        # Add tool calls to assistant message
        if provider_response.tool_calls is not None and len(provider_response.tool_calls) > 0:
            # Ensure every tool call has an id — some providers (e.g. Ollama) omit it
            for tc in provider_response.tool_calls:
                if not tc.get("id"):
                    tc["id"] = str(uuid4())
            assistant_message.tool_calls = provider_response.tool_calls

        # Add audio to assistant message
        if provider_response.audio is not None:
            assistant_message.audio_output = provider_response.audio

        # Add image to assistant message
        if provider_response.images is not None:
            if provider_response.images:
                assistant_message.image_output = provider_response.images[-1]  # Taking last (most recent) image

        # Add video to assistant message
        if provider_response.videos is not None:
            if provider_response.videos:
                assistant_message.video_output = provider_response.videos[-1]  # Taking last (most recent) video

        if provider_response.files is not None:
            if provider_response.files:
                assistant_message.file_output = provider_response.files[-1]  # Taking last (most recent) file

        if provider_response.audios is not None:
            if provider_response.audios:
                assistant_message.audio_output = provider_response.audios[-1]  # Taking last (most recent) audio

        # Add redacted thinking content to assistant message
        if provider_response.redacted_reasoning_content is not None:
            assistant_message.redacted_reasoning_content = provider_response.redacted_reasoning_content

        # Add reasoning content to assistant message
        if provider_response.reasoning_content is not None:
            assistant_message.reasoning_content = provider_response.reasoning_content

        # Add provider data to assistant message
        if provider_response.provider_data is not None:
            assistant_message.provider_data = provider_response.provider_data

        # Add citations to assistant message
        if provider_response.citations is not None:
            assistant_message.citations = provider_response.citations

        # Add usage metrics if provided
        if provider_response.response_usage is not None:
            self._ensure_message_metrics_initialized(assistant_message)
            # Update Metrics with usage data from response
            usage = provider_response.response_usage
            # Use in-place addition to preserve timer automatically
            assistant_message.metrics += usage
            # Set time_to_first_token if we have content and it's not already set
            if provider_response.content is not None and assistant_message.metrics.time_to_first_token is None:
                assistant_message.metrics.set_time_to_first_token()

        return assistant_message

    def process_response_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        stream_data: MessageData,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> Iterator[ModelResponse]:
        """
        Process a streaming response from the model with retry logic for ModelProviderError.
        """

        for response_delta in self._invoke_stream_with_retry(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice or self._tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        ):
            # Set TTFT when first chunk arrives (guard ensures first-call-wins)
            if run_response and run_response.metrics:
                run_response.metrics.set_time_to_first_token()
            for model_response_delta in self._populate_stream_data(
                stream_data=stream_data,
                model_response_delta=response_delta,
            ):
                yield model_response_delta

        # Populate assistant message from stream data after the stream ends
        self._populate_assistant_message_from_stream_data(assistant_message=assistant_message, stream_data=stream_data)

    def response_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        stream_model_response: bool = True,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        send_media_to_model: bool = True,
        compression_manager: Optional["CompressionManager"] = None,
        result_store: Optional["ResultStore"] = None,
        after_tool_results: Optional[Callable[["ModelResponse"], None]] = None,
        run_context: Optional[Any] = None,
    ) -> Iterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        """
        Generate a streaming response from the model.

        ``after_tool_results``: optional callback invoked once per tool batch, after tool result
        messages are appended to ``messages`` and before the next model call (or break). Receives
        the current ``ModelResponse`` (with accumulated ``tool_executions``) as its single
        argument. Used by Agent-level checkpointing (``checkpoint="tool-batch"``) to persist mid-run
        state. Exceptions are caught and logged — a failed callback must not kill the run.
        """
        _run_context_token = _CURRENT_RUN_CONTEXT_VAR.set(run_context)
        try:
            # Check cache if enabled - capture key BEFORE streaming to avoid mismatch
            cache_key = None
            cache_streaming_response = self.cache_response and not _has_executable_tools(tools)
            if cache_streaming_response:
                cache_key = self._get_model_cache_key(
                    messages, stream=True, response_format=response_format, tools=tools
                )
                cached_data = self._get_cached_model_response(cache_key)

                if cached_data:
                    log_info("Cache hit for streaming model response")
                    for response in self._streaming_responses_from_cache(cached_data["streaming_responses"]):
                        yield response
                    return

                log_info("Cache miss for streaming model response")

            streaming_responses: List[ModelResponse] = []

            log_debug(f"{self.get_provider()} Response Stream Start", center=True, symbol="-")
            log_debug(f"Model: {self.id}", center=True, symbol="-")
            _log_messages(messages)

            _tool_dicts = self._format_tools(tools) if tools is not None else []
            _functions = self._get_functions_from_tools(tools)

            _compress_tool_results = compression_manager is not None and compression_manager.compress_tool_results
            _compression_manager = compression_manager if _compress_tool_results else None

            function_call_count = 0

            while True:
                if _compression_manager is not None and _compression_manager.should_compress(
                    messages, tools, model=self, response_format=response_format
                ):
                    yield ModelResponse(event=ModelResponseEvent.compression_started.value)
                    _compression_manager.compress(
                        messages, run_metrics=run_response.metrics if run_response is not None else None
                    )
                    yield ModelResponse(
                        event=ModelResponseEvent.compression_completed.value,
                        compression_stats=_compression_manager.stats.copy(),
                    )

                assistant_message = Message(role=self.assistant_message_role)
                stream_data = MessageData()
                model_response = ModelResponse()

                yield ModelResponse(event=ModelResponseEvent.model_request_started.value)

                if stream_model_response:
                    stream_data.response_metrics = MessageMetrics()
                    stream_data.response_metrics.start_timer()
                    self._ensure_message_metrics_initialized(assistant_message)
                    try:
                        for response in self.process_response_stream(
                            messages=messages,
                            assistant_message=assistant_message,
                            stream_data=stream_data,
                            response_format=response_format,
                            tools=_tool_dicts,
                            tool_choice=tool_choice or self._tool_choice,
                            run_response=run_response,
                            compress_tool_results=_compress_tool_results,
                        ):
                            if cache_streaming_response and isinstance(response, ModelResponse):
                                streaming_responses.append(response)
                            yield response
                    finally:
                        if run_response is not None and assistant_message.metrics is not None:
                            from agno.metrics import accumulate_model_metrics

                            _stream_model_response = ModelResponse()
                            _stream_model_response.response_usage = assistant_message.metrics
                            accumulate_model_metrics(
                                _stream_model_response, self, self.model_type, run_response.metrics
                            )

                else:
                    self._ensure_message_metrics_initialized(assistant_message)
                    self._process_model_response(
                        messages=messages,
                        assistant_message=assistant_message,
                        model_response=model_response,
                        response_format=response_format,
                        tools=_tool_dicts,
                        tool_choice=tool_choice or self._tool_choice,
                        run_response=run_response,
                        compress_tool_results=_compress_tool_results,
                    )
                    if run_response is not None and model_response.response_usage is not None:
                        from agno.metrics import accumulate_model_metrics

                        accumulate_model_metrics(model_response, self, self.model_type, run_response.metrics)
                    if cache_streaming_response:
                        streaming_responses.append(model_response)
                    yield model_response

                messages.append(assistant_message)
                assistant_message.log(metrics=True)

                llm_metrics = assistant_message.metrics
                yield ModelResponse(
                    event=ModelResponseEvent.model_request_completed.value,
                    input_tokens=llm_metrics.input_tokens if llm_metrics else None,
                    output_tokens=llm_metrics.output_tokens if llm_metrics else None,
                    total_tokens=llm_metrics.total_tokens if llm_metrics else None,
                    time_to_first_token=llm_metrics.time_to_first_token if llm_metrics else None,
                    reasoning_tokens=llm_metrics.reasoning_tokens if llm_metrics else None,
                    cache_read_tokens=llm_metrics.cache_read_tokens if llm_metrics else None,
                    cache_write_tokens=llm_metrics.cache_write_tokens if llm_metrics else None,
                )

                if assistant_message.tool_calls is not None:
                    function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(
                        assistant_message=assistant_message, messages=messages, functions=_functions
                    )
                    function_call_results: List[Message] = []
                    tool_call_paused = False

                    for function_call_response in self.run_function_calls(
                        function_calls=function_calls_to_run,
                        function_call_results=function_call_results,
                        current_function_call_count=function_call_count,
                        function_call_limit=tool_call_limit,
                        result_store=result_store,
                    ):
                        if (
                            isinstance(function_call_response, ModelResponse)
                            and function_call_response.event == ModelResponseEvent.tool_call_paused.value
                        ):
                            tool_call_paused = True
                        if cache_streaming_response and isinstance(function_call_response, ModelResponse):
                            streaming_responses.append(function_call_response)
                        yield function_call_response

                    function_call_count += self._limit_charge_for(function_call_results, result_store)

                    if stream_data and stream_data.extra is not None:
                        self.format_function_call_results(
                            messages=messages,
                            function_call_results=function_call_results,
                            compress_tool_results=_compress_tool_results,
                            **stream_data.extra,
                        )
                    elif model_response and model_response.extra is not None:
                        self.format_function_call_results(
                            messages=messages,
                            function_call_results=function_call_results,
                            compress_tool_results=_compress_tool_results,
                            **model_response.extra,
                        )
                    else:
                        self.format_function_call_results(
                            messages=messages,
                            function_call_results=function_call_results,
                            compress_tool_results=_compress_tool_results,
                        )

                    if any(msg.images or msg.videos or msg.audio or msg.files for msg in function_call_results):
                        self._handle_function_call_media(
                            messages=messages,
                            function_call_results=function_call_results,
                            send_media_to_model=send_media_to_model,
                        )

                    for function_call_result in function_call_results:
                        function_call_result.log(metrics=True, use_compressed_content=_compress_tool_results)

                    if any(m.stop_after_tool_call for m in function_call_results):
                        break

                    if after_tool_results is not None:
                        try:
                            after_tool_results(model_response)
                        except Exception as e:
                            log_error(f"after_tool_results callback failed: {e}")

                    if tool_call_paused:
                        break

                    if any(fc.function.requires_confirmation for fc in function_calls_to_run):
                        break

                    if any(fc.function.external_execution for fc in function_calls_to_run):
                        break

                    if any(fc.function.requires_user_input for fc in function_calls_to_run):
                        break

                    if run_response is not None and run_response.requirements:
                        if any(not req.is_resolved() for req in run_response.requirements):
                            break

                    continue

                break

            log_debug(f"{self.get_provider()} Response Stream End", center=True, symbol="-")

            if cache_streaming_response and cache_key and streaming_responses:
                self._save_streaming_responses_to_cache(cache_key, streaming_responses)
        finally:
            _CURRENT_RUN_CONTEXT_VAR.reset(_run_context_token)
            if self.__class__.__name__ == "Gemini" and self.client is not None:
                try:
                    self.client.close()  # type: ignore
                    self.client = None
                except AttributeError as e:
                    log_warning(
                        f"Your Gemini client is outdated. For Agno to properly handle the lifecycle of the client,: {e}"
                        f" please upgrade Gemini to the latest version: pip install -U google-genai: {e}",
                    )

    async def aprocess_response_stream(
        self,
        messages: List[Message],
        assistant_message: Message,
        stream_data: MessageData,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> AsyncIterator[ModelResponse]:
        """
        Process a streaming response from the model with retry logic for ModelProviderError.
        """
        try:
            async for response_delta in self._ainvoke_stream_with_retry(
                messages=messages,
                assistant_message=assistant_message,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice or self._tool_choice,
                run_response=run_response,
                compress_tool_results=compress_tool_results,
            ):
                if run_response and run_response.metrics:
                    run_response.metrics.set_time_to_first_token()
                for model_response_delta in self._populate_stream_data(
                    stream_data=stream_data,
                    model_response_delta=response_delta,
                ):
                    yield model_response_delta

            self._populate_assistant_message_from_stream_data(
                assistant_message=assistant_message, stream_data=stream_data
            )
        except BaseException:
            self._populate_assistant_message_from_stream_data(
                assistant_message=assistant_message, stream_data=stream_data
            )
            raise

    async def aresponse_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_call_limit: Optional[int] = None,
        stream_model_response: bool = True,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        send_media_to_model: bool = True,
        compression_manager: Optional["CompressionManager"] = None,
        result_store: Optional["ResultStore"] = None,
        after_tool_results: Optional[Callable[["ModelResponse"], Awaitable[None]]] = None,
        run_context: Optional[Any] = None,
    ) -> AsyncIterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        """
        Generate an asynchronous streaming response from the model.

        ``after_tool_results``: optional async callback invoked once per tool batch, after tool
        result messages are appended to ``messages`` and before the next model call (or break).
        Receives the current ``ModelResponse`` (with accumulated ``tool_executions``) as its
        single argument. Used by Agent-level checkpointing (``checkpoint="tool-batch"``) to persist
        mid-run state. Exceptions are caught and logged — a failed callback must not kill the run.
        """
        _run_context_token = _CURRENT_RUN_CONTEXT_VAR.set(run_context)
        try:
            # Check cache if enabled - capture key BEFORE streaming to avoid mismatch
            cache_key = None
            cache_streaming_response = self.cache_response and not _has_executable_tools(tools)
            if cache_streaming_response:
                cache_key = self._get_model_cache_key(
                    messages, stream=True, response_format=response_format, tools=tools
                )
                cached_data = self._get_cached_model_response(cache_key)

                if cached_data:
                    log_info("Cache hit for async streaming model response")
                    for response in self._streaming_responses_from_cache(cached_data["streaming_responses"]):
                        yield response
                    return

                log_info("Cache miss for async streaming model response")

            # Track streaming responses for caching
            streaming_responses: List[ModelResponse] = []

            log_debug(f"{self.get_provider()} Async Response Stream Start", center=True, symbol="-")
            log_debug(f"Model: {self.id}", center=True, symbol="-")
            _log_messages(messages)

            _tool_dicts = self._format_tools(tools) if tools is not None else []
            _functions = self._get_functions_from_tools(tools)

            _compress_tool_results = compression_manager is not None and compression_manager.compress_tool_results
            _compression_manager = compression_manager if _compress_tool_results else None

            function_call_count = 0

            while True:
                # Compress existing tool results BEFORE making API call to avoid context overflow
                if _compression_manager is not None and await _compression_manager.ashould_compress(
                    messages, tools, model=self, response_format=response_format
                ):
                    # Emit compression started event
                    yield ModelResponse(event=ModelResponseEvent.compression_started.value)
                    await _compression_manager.acompress(
                        messages, run_metrics=run_response.metrics if run_response is not None else None
                    )
                    # Emit compression completed event with stats
                    yield ModelResponse(
                        event=ModelResponseEvent.compression_completed.value,
                        compression_stats=_compression_manager.stats.copy(),
                    )

                # Create assistant message and stream data
                assistant_message = Message(role=self.assistant_message_role)
                stream_data = MessageData()
                model_response = ModelResponse()
                _assistant_message_appended = False

                # Emit LLM request started event
                yield ModelResponse(event=ModelResponseEvent.model_request_started.value)

                try:
                    if stream_model_response:
                        stream_data.response_metrics = MessageMetrics()
                        stream_data.response_metrics.start_timer()
                        self._ensure_message_metrics_initialized(assistant_message)
                        async for model_response_delta in self.aprocess_response_stream(
                            messages=messages,
                            assistant_message=assistant_message,
                            stream_data=stream_data,
                            response_format=response_format,
                            tools=_tool_dicts,
                            tool_choice=tool_choice or self._tool_choice,
                            run_response=run_response,
                            compress_tool_results=_compress_tool_results,
                        ):
                            if cache_streaming_response and isinstance(model_response_delta, ModelResponse):
                                streaming_responses.append(model_response_delta)
                            yield model_response_delta

                        if run_response is not None and assistant_message.metrics is not None:
                            from agno.metrics import accumulate_model_metrics

                            _stream_model_response = ModelResponse()
                            _stream_model_response.response_usage = assistant_message.metrics
                            accumulate_model_metrics(
                                _stream_model_response, self, self.model_type, run_response.metrics
                            )

                    else:
                        self._ensure_message_metrics_initialized(assistant_message)
                        await self._aprocess_model_response(
                            messages=messages,
                            assistant_message=assistant_message,
                            model_response=model_response,
                            response_format=response_format,
                            tools=_tool_dicts,
                            tool_choice=tool_choice or self._tool_choice,
                            run_response=run_response,
                            compress_tool_results=_compress_tool_results,
                        )
                        if run_response is not None and model_response.response_usage is not None:
                            from agno.metrics import accumulate_model_metrics

                            accumulate_model_metrics(model_response, self, self.model_type, run_response.metrics)
                        if cache_streaming_response:
                            streaming_responses.append(model_response)
                        yield model_response

                    messages.append(assistant_message)
                    _assistant_message_appended = True
                except BaseException:
                    if not _assistant_message_appended:
                        self._populate_assistant_message_from_stream_data(
                            assistant_message=assistant_message, stream_data=stream_data
                        )
                        if assistant_message.content:
                            messages.append(assistant_message)
                    raise
                assistant_message.log(metrics=True)

                # Emit LLM request completed event with metrics
                llm_metrics = assistant_message.metrics
                yield ModelResponse(
                    event=ModelResponseEvent.model_request_completed.value,
                    input_tokens=llm_metrics.input_tokens if llm_metrics else None,
                    output_tokens=llm_metrics.output_tokens if llm_metrics else None,
                    total_tokens=llm_metrics.total_tokens if llm_metrics else None,
                    time_to_first_token=llm_metrics.time_to_first_token if llm_metrics else None,
                    reasoning_tokens=llm_metrics.reasoning_tokens if llm_metrics else None,
                    cache_read_tokens=llm_metrics.cache_read_tokens if llm_metrics else None,
                    cache_write_tokens=llm_metrics.cache_write_tokens if llm_metrics else None,
                )

                # Handle tool calls if present
                if assistant_message.tool_calls is not None:
                    # Prepare function calls
                    function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(
                        assistant_message=assistant_message, messages=messages, functions=_functions
                    )
                    function_call_results: List[Message] = []
                    tool_call_paused = False

                    # Execute function calls
                    async for function_call_response in self.arun_function_calls(
                        function_calls=function_calls_to_run,
                        function_call_results=function_call_results,
                        current_function_call_count=function_call_count,
                        function_call_limit=tool_call_limit,
                        result_store=result_store,
                        run_id=run_response.run_id if run_response else None,
                    ):
                        if (
                            isinstance(function_call_response, ModelResponse)
                            and function_call_response.event == ModelResponseEvent.tool_call_paused.value
                        ):
                            tool_call_paused = True
                        if cache_streaming_response and isinstance(function_call_response, ModelResponse):
                            streaming_responses.append(function_call_response)
                        yield function_call_response

                    # Add a function call for each successful execution
                    function_call_count += self._limit_charge_for(function_call_results, result_store)

                    # Format and add results to messages
                    if stream_data and stream_data.extra is not None:
                        self.format_function_call_results(
                            messages=messages,
                            function_call_results=function_call_results,
                            compress_tool_results=_compress_tool_results,
                            **stream_data.extra,
                        )
                    elif model_response and model_response.extra is not None:
                        self.format_function_call_results(
                            messages=messages,
                            function_call_results=function_call_results,
                            compress_tool_results=_compress_tool_results,
                            **model_response.extra or {},
                        )
                    else:
                        self.format_function_call_results(
                            messages=messages,
                            function_call_results=function_call_results,
                            compress_tool_results=_compress_tool_results,
                        )

                    # Handle function call media
                    if any(msg.images or msg.videos or msg.audio or msg.files for msg in function_call_results):
                        self._handle_function_call_media(
                            messages=messages,
                            function_call_results=function_call_results,
                            send_media_to_model=send_media_to_model,
                        )

                    for function_call_result in function_call_results:
                        function_call_result.log(metrics=True, use_compressed_content=_compress_tool_results)

                    # Check if we should stop after tool calls
                    if any(m.stop_after_tool_call for m in function_call_results):
                        break

                    if after_tool_results is not None:
                        try:
                            await after_tool_results(model_response)
                        except Exception as e:
                            log_error(f"after_tool_results callback failed: {e}")

                    if tool_call_paused:
                        break

                    # If we have any tool calls that require confirmation, break the loop
                    if any(fc.function.requires_confirmation for fc in function_calls_to_run):
                        break

                    # If we have any tool calls that require external execution, break the loop
                    if any(fc.function.external_execution for fc in function_calls_to_run):
                        break

                    # If we have any tool calls that require user input, break the loop
                    if any(fc.function.requires_user_input for fc in function_calls_to_run):
                        break

                    # Check if run_response has requirements (e.g., from member agent HITL)
                    # This handles cases where a tool (like delegate_task_to_member) propagates
                    # HITL requirements from a member agent to the team's run_response
                    if run_response is not None and run_response.requirements:
                        if any(not req.is_resolved() for req in run_response.requirements):
                            break

                    # Continue loop to get next response
                    continue

                # No tool calls or finished processing them
                break

            log_debug(f"{self.get_provider()} Async Response Stream End", center=True, symbol="-")

            # Save streaming responses to cache if enabled
            if cache_streaming_response and cache_key and streaming_responses:
                self._save_streaming_responses_to_cache(cache_key, streaming_responses)

        finally:
            _CURRENT_RUN_CONTEXT_VAR.reset(_run_context_token)
            # Close the Gemini client
            if self.__class__.__name__ == "Gemini" and self.client is not None:
                try:
                    await self.client.aio.aclose()  # type: ignore
                    self.client = None
                except AttributeError as e:
                    log_warning(
                        f"Your Gemini client is outdated. For Agno to properly handle the lifecycle of the client,: {e}"
                        f" please upgrade Gemini to the latest version: pip install -U google-genai: {e}",
                    )

    def _populate_assistant_message_from_stream_data(
        self, assistant_message: Message, stream_data: MessageData
    ) -> None:
        """
        Populate an assistant message with the stream data.
        """
        if stream_data.response_role is not None:
            assistant_message.role = stream_data.response_role
        if stream_data.response_metrics is not None:
            assistant_message.metrics = stream_data.response_metrics
        if stream_data.response_content:
            assistant_message.content = stream_data.response_content
        if stream_data.response_reasoning_content:
            assistant_message.reasoning_content = stream_data.response_reasoning_content
        if stream_data.response_redacted_reasoning_content:
            assistant_message.redacted_reasoning_content = stream_data.response_redacted_reasoning_content
        if stream_data.response_provider_data:
            assistant_message.provider_data = stream_data.response_provider_data
        if stream_data.response_citations:
            assistant_message.citations = stream_data.response_citations
        if stream_data.response_audio:
            assistant_message.audio_output = stream_data.response_audio
        if stream_data.response_image:
            assistant_message.image_output = stream_data.response_image
        if stream_data.response_video:
            assistant_message.video_output = stream_data.response_video
        if stream_data.response_file:
            assistant_message.file_output = stream_data.response_file
        if stream_data.response_tool_calls and len(stream_data.response_tool_calls) > 0:
            parsed_tool_calls = self.parse_tool_calls(stream_data.response_tool_calls)
            # Ensure every tool call has an id — some providers (e.g. Ollama) omit it
            for tc in parsed_tool_calls:
                if not tc.get("id"):
                    tc["id"] = str(uuid4())
            assistant_message.tool_calls = parsed_tool_calls

    def _populate_stream_data(
        self, stream_data: MessageData, model_response_delta: ModelResponse
    ) -> Iterator[ModelResponse]:
        """Update the stream data and assistant message with the model response."""

        if model_response_delta.event == ModelResponseEvent.tool_call_args_delta.value:
            if model_response_delta.tool_call_id is not None:
                stream_data.tool_call_direct_args_delta_ids.add(model_response_delta.tool_call_id)
            if (
                model_response_delta.tool_call_id
                and model_response_delta.tool_call_id not in stream_data.tool_call_started_ids
            ):
                stream_data.tool_call_started_ids.add(model_response_delta.tool_call_id)
                yield ModelResponse(
                    event=ModelResponseEvent.tool_call_start.value,
                    tool_call_id=model_response_delta.tool_call_id,
                    tool_name=model_response_delta.tool_name,
                )
            yield model_response_delta
            return

        should_yield = False
        if model_response_delta.role is not None:
            stream_data.response_role = model_response_delta.role  # type: ignore

        if model_response_delta.response_usage is not None:
            if stream_data.response_metrics is None:
                # Initialize if not already initialized (shouldn't happen, but safety check)
                stream_data.response_metrics = MessageMetrics()
                stream_data.response_metrics.start_timer()
            # Update Metrics with usage data from response
            usage = model_response_delta.response_usage
            # Use in-place addition to preserve timer automatically
            stream_data.response_metrics += usage

        # Update stream_data content
        if model_response_delta.content is not None:
            stream_data.response_content += model_response_delta.content
            # Set time_to_first_token on first content chunk if not already set
            if stream_data.response_metrics is not None and stream_data.response_metrics.time_to_first_token is None:
                stream_data.response_metrics.set_time_to_first_token()
            should_yield = True

        if model_response_delta.reasoning_content is not None:
            stream_data.response_reasoning_content += model_response_delta.reasoning_content
            should_yield = True

        if model_response_delta.redacted_reasoning_content is not None:
            stream_data.response_redacted_reasoning_content += model_response_delta.redacted_reasoning_content
            should_yield = True

        if model_response_delta.citations is not None:
            stream_data.response_citations = model_response_delta.citations
            should_yield = True

        if model_response_delta.provider_data:
            if stream_data.response_provider_data is None:
                stream_data.response_provider_data = {}
            # List-aware merge: extend lists (e.g. server_tool_blocks), replace scalars
            for key, value in model_response_delta.provider_data.items():
                existing = stream_data.response_provider_data.get(key)
                if isinstance(existing, list) and isinstance(value, list):
                    existing.extend(value)
                else:
                    stream_data.response_provider_data[key] = value

        # Update stream_data tool calls
        if model_response_delta.tool_calls is not None and len(model_response_delta.tool_calls) > 0:
            # When a provider delivers a tool call as a snapshot (e.g. OpenAI Responses
            # `response.output_item.done` with full arguments and no preceding args
            # deltas), no `tool_call_args_delta` event ever fires for this id, so the
            # paired `tool_call_start` is never emitted by the args-delta branch above.
            # Synthesize them here for any tool_call_id we have not seen yet so that
            # streaming consumers receive the same start + args-delta pair on both
            # delta and one-shot delivery paths.
            #
            # `ModelResponse.tool_calls` is declared `List[Dict[str, Any]]` and the
            # OpenAI Chat / OpenAILike providers now honour that contract. A handful
            # of other providers (Groq, Meta Llama, HuggingFace, Azure AI Foundry)
            # still inject pydantic chunk objects here, and their `parse_tool_calls`
            # implementations depend on receiving them verbatim for per-chunk
            # `function.arguments` concatenation. Be tolerant of both element shapes
            # so this loop never crashes on those providers, and do not coerce the
            # elements before extending `response_tool_calls` below, or those
            # fragment-concat paths break. Each remaining provider should migrate
            # to the dict contract in a follow-up.
            #
            # Mid-stream fragments (OpenAI Chat / Groq / Azure / HF) carry only an
            # `index` after the first chunk; we look up the previously cached
            # id from `tool_call_provider_index_to_id` so subsequent args fragments
            # can still be emitted as `tool_call_args_delta` events instead of
            # being dropped.
            for tc in model_response_delta.tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("call_id") or tc.get("id")
                    tc_index = tc.get("index")
                    tc_function = tc.get("function") if isinstance(tc.get("function"), dict) else None
                    tc_name = tc_function.get("name") if tc_function else None
                    tc_args = tc_function.get("arguments") if tc_function else None
                else:
                    tc_id = getattr(tc, "call_id", None) or getattr(tc, "id", None)
                    tc_index = getattr(tc, "index", None)
                    tc_function = getattr(tc, "function", None)
                    tc_name = getattr(tc_function, "name", None) if tc_function is not None else None
                    tc_args = getattr(tc_function, "arguments", None) if tc_function is not None else None

                # If this is the first chunk for an `index`, remember the (index -> id)
                # mapping so that subsequent fragments with id=None can still be routed.
                if tc_index is not None and tc_id and tc_index not in stream_data.tool_call_provider_index_to_id:
                    stream_data.tool_call_provider_index_to_id[tc_index] = tc_id

                # Fragment chunks (post-first) typically have id=None; recover the id
                # from the previously cached mapping so we can still emit args_delta.
                if not tc_id and tc_index is not None:
                    tc_id = stream_data.tool_call_provider_index_to_id.get(tc_index)

                if not tc_id:
                    continue

                if tc_id not in stream_data.tool_call_started_ids:
                    stream_data.tool_call_started_ids.add(tc_id)
                    yield ModelResponse(
                        event=ModelResponseEvent.tool_call_start.value,
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                    )
                if tc_args and tc_id not in stream_data.tool_call_direct_args_delta_ids:
                    yield ModelResponse(
                        event=ModelResponseEvent.tool_call_args_delta.value,
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                        tool_args_delta=tc_args,
                    )
            if stream_data.response_tool_calls is None:
                stream_data.response_tool_calls = []
            stream_data.response_tool_calls.extend(model_response_delta.tool_calls)
            should_yield = True

        if model_response_delta.audio is not None and isinstance(model_response_delta.audio, Audio):
            if stream_data.response_audio is None:
                stream_data.response_audio = Audio(id=str(uuid4()), content="", transcript="")

            from typing import cast

            audio_response = cast(Audio, model_response_delta.audio)

            # Update the stream data with audio information
            if audio_response.id is not None:
                stream_data.response_audio.id = audio_response.id  # type: ignore
            if audio_response.content is not None:
                stream_data.response_audio.content += audio_response.content  # type: ignore
            if audio_response.transcript is not None:
                stream_data.response_audio.transcript += audio_response.transcript  # type: ignore
            if audio_response.expires_at is not None:
                stream_data.response_audio.expires_at = audio_response.expires_at
            if audio_response.mime_type is not None:
                stream_data.response_audio.mime_type = audio_response.mime_type
            stream_data.response_audio.sample_rate = audio_response.sample_rate
            stream_data.response_audio.channels = audio_response.channels

            should_yield = True

        if model_response_delta.images:
            if stream_data.response_image is None:
                stream_data.response_image = model_response_delta.images[-1]
            should_yield = True

        if model_response_delta.videos:
            if stream_data.response_video is None:
                stream_data.response_video = model_response_delta.videos[-1]
            should_yield = True

        if model_response_delta.extra is not None:
            if stream_data.extra is None:
                stream_data.extra = {}
            for key in model_response_delta.extra:
                if isinstance(model_response_delta.extra[key], list):
                    if not stream_data.extra.get(key):
                        stream_data.extra[key] = []
                    stream_data.extra[key].extend(model_response_delta.extra[key])
                else:
                    stream_data.extra[key] = model_response_delta.extra[key]

        if should_yield:
            yield model_response_delta

    def parse_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse the tool calls from the model provider into a list of tool calls.
        """
        return tool_calls_data

    def get_function_call_to_run_from_tool_execution(
        self,
        tool_execution: ToolExecution,
        functions: Optional[Dict[str, Function]] = None,
    ) -> FunctionCall:
        function_call = get_function_call_for_tool_execution(
            tool_execution=tool_execution,
            functions=functions,
        )
        if function_call is None:
            raise ValueError("Function call not found")
        return function_call

    def get_function_calls_to_run(
        self,
        assistant_message: Message,
        messages: List[Message],
        functions: Optional[Dict[str, Function]] = None,
    ) -> List[FunctionCall]:
        """
        Prepare function calls for the assistant message.
        """
        function_calls_to_run: List[FunctionCall] = []
        if assistant_message.tool_calls is not None:
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("call_id") or tool_call.get("id")
                _tool_call_name = tool_call.get("function", {}).get("name")
                _function_call = get_function_call_for_tool_call(tool_call, functions)
                if _function_call is None:
                    messages.append(
                        Message(
                            role=self.tool_message_role,
                            tool_call_id=_tool_call_id,
                            tool_name=_tool_call_name,
                            content="Error: The requested tool does not exist or is not available.",
                        )
                    )
                    continue
                if _function_call.error is not None:
                    messages.append(
                        Message(
                            role=self.tool_message_role,
                            tool_call_id=_tool_call_id,
                            tool_name=_tool_call_name,
                            content=_function_call.error,
                        )
                    )
                    continue
                function_calls_to_run.append(_function_call)
        return function_calls_to_run

    @staticmethod
    def _offload_candidate(
        result_store: "ResultStore",
        function_call: FunctionCall,
        success: bool,
        output: str,
    ) -> Optional[str]:
        """The output as text when it qualifies for offloading, else None.

        Never offloaded: failed calls (the model needs the error text
        verbatim), empty results, sub-threshold results, the read-back tools'
        own output, and any result that ends the run. A result that ends the
        run is the answer the caller receives, so a pointer in its place would
        replace the answer with a reference to it. Only the message content is
        ever replaced - media on the FunctionExecutionResult is untouched.
        """
        if not success or not output:
            return None
        if function_call.function.stop_after_tool_call:
            return None
        text = output
        if not result_store.should_offload(function_call.function.name, text):
            return None
        if function_call.function._run_context is None:
            return None
        return text

    def _substitute_tool_result(
        self,
        result_store: "ResultStore",
        function_call: FunctionCall,
        success: bool,
        output: str,
    ) -> str:
        """Replace an oversized successful tool result with its envelope."""
        text = self._offload_candidate(result_store, function_call, success, output)
        if text is None:
            return output
        run_context = function_call.function._run_context
        assert run_context is not None
        return result_store.offload_for_model(
            session_id=run_context.session_id,
            run_id=run_context.run_id,
            tool_call_id=function_call.call_id or function_call.function.name,
            tool_name=function_call.function.name,
            tool_args=function_call.arguments,
            output=text,
            user_id=run_context.user_id,
            shared=function_call.function._team is not None,
        )

    async def _asubstitute_tool_result(
        self,
        result_store: "ResultStore",
        function_call: FunctionCall,
        success: bool,
        output: str,
    ) -> str:
        """Async variant of ``_substitute_tool_result``."""
        text = self._offload_candidate(result_store, function_call, success, output)
        if text is None:
            return output
        run_context = function_call.function._run_context
        assert run_context is not None
        return await result_store.aoffload_for_model(
            session_id=run_context.session_id,
            run_id=run_context.run_id,
            tool_call_id=function_call.call_id or function_call.function.name,
            tool_name=function_call.function.name,
            tool_args=function_call.arguments,
            output=text,
            user_id=run_context.user_id,
            shared=function_call.function._team is not None,
        )

    def create_function_call_result(
        self,
        function_call: FunctionCall,
        success: bool,
        output: Optional[Union[List[Any], str]] = None,
        timer: Optional[Timer] = None,
        function_execution_result: Optional[FunctionExecutionResult] = None,
    ) -> Message:
        """Create a function call result message."""
        kwargs: Dict[str, Any] = {}
        # Tool messages don't get metrics - only assistant messages do

        # Include media artifacts from function execution result in the tool message
        images = None
        videos = None
        audios = None
        files = None

        if success and function_execution_result:
            # With unified classes, no conversion needed - use directly
            images = function_execution_result.images
            videos = function_execution_result.videos
            audios = function_execution_result.audios
            files = function_execution_result.files

        return Message(
            role=self.tool_message_role,
            content=output if success else function_call.error,
            tool_call_id=function_call.call_id,
            tool_name=function_call.function.name,
            tool_args=function_call.arguments,
            tool_call_error=not success,
            stop_after_tool_call=function_call.function.stop_after_tool_call,
            images=images,
            videos=videos,
            audio=audios,
            files=files,
            **kwargs,  # type: ignore
        )

    @staticmethod
    def _limit_charge_for(function_call_results: List[Message], result_store: Optional["ResultStore"]) -> int:
        """How many of this batch's results spend the tool call limit.

        With offloading on, the read-back tools are exempt: they exist only
        because a result was replaced with a pointer the model was told to
        follow, so the running total charges the same calls the per-batch
        check charges.
        """
        if result_store is None:
            return len(function_call_results)
        from agno.offload.types import NEVER_OFFLOADED_TOOLS

        return sum(1 for m in function_call_results if m.tool_name not in NEVER_OFFLOADED_TOOLS)

    def create_tool_call_limit_error_result(self, function_call: FunctionCall) -> Message:
        return Message(
            role=self.tool_message_role,
            content=f"Tool call limit reached. Tool call {function_call.function.name} not executed. Don't try to execute it again.",
            tool_call_id=function_call.call_id,
            tool_name=function_call.function.name,
            tool_args=function_call.arguments,
            tool_call_error=True,
        )

    def run_function_call(
        self,
        function_call: FunctionCall,
        function_call_results: List[Message],
        additional_input: Optional[List[Message]] = None,
        result_store: Optional["ResultStore"] = None,
    ) -> Iterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        # Start function call
        function_call_timer = Timer()
        function_call_timer.start()
        # Yield a tool_call_started event
        yield ModelResponse(
            content=function_call.get_call_str(),
            tool_executions=[
                ToolExecution(
                    tool_call_id=function_call.call_id,
                    tool_name=function_call.function.name,
                    tool_args=function_call.arguments,
                )
            ],
            event=ModelResponseEvent.tool_call_started.value,
        )

        # Run function calls sequentially
        function_execution_result: FunctionExecutionResult = FunctionExecutionResult(status="failure")
        stop_after_tool_call_from_exception = False
        try:
            function_execution_result = function_call.execute()
        except (ToolApprovalRequired, ToolCallDeferred) as pause_exc:
            function_call_timer.stop()
            yield _create_tool_call_paused_response(function_call, pause_exc)
            return
        except AgentRunException as a_exc:
            stop_after_tool_call_from_exception = _handle_agent_exception_from_tool_call(
                function_call, a_exc, additional_input
            )
        except RunCancelledException:
            raise
        except Exception as e:
            log_error(f"Error executing function {function_call.function.name}: {str(e)}")
            raise e

        function_call_success = function_execution_result.status == "success"

        # Stop function call timer
        function_call_timer.stop()

        # Process function call output
        function_call_output: str = ""

        if isinstance(function_execution_result.result, (GeneratorType, collections.abc.Iterator)):
            try:
                for item in function_execution_result.result:
                    # This function yields agent/team/workflow run events
                    if isinstance(item, _ALL_RUN_OUTPUT_EVENT_TYPES):
                        # We only capture content events for output accumulation
                        if isinstance(item, RunContentEvent) or isinstance(item, TeamRunContentEvent):
                            if item.content is not None and isinstance(item.content, BaseModel):
                                function_call_output += item.content.model_dump_json()
                            else:
                                # Capture output
                                function_call_output += item.content or ""

                            if function_call.function.show_result and item.content is not None:
                                yield ModelResponse(content=item.content)

                        if isinstance(item, CustomEvent):
                            function_call_output += str(item)
                            item.tool_call_id = function_call.call_id

                        # For WorkflowCompletedEvent, extract content for final output
                        from agno.run.workflow import WorkflowCompletedEvent

                        if isinstance(item, WorkflowCompletedEvent):
                            if item.content is not None:
                                if isinstance(item.content, BaseModel):
                                    function_call_output += item.content.model_dump_json()
                                else:
                                    function_call_output += str(item.content)

                        # Yield the event itself to bubble it up. The isinstance guards
                        # above narrow item at runtime, but mypy cannot see through
                        # the cached union-member tuple.
                        yield item  # type: ignore[misc]

                    else:
                        function_call_output += str(item)
                        if function_call.function.show_result and item is not None:
                            yield ModelResponse(content=str(item))
            except RunCancelledException:
                raise
            except (ToolApprovalRequired, ToolCallDeferred) as pause_exc:
                yield _create_tool_call_paused_response(function_call, pause_exc)
                return
            except AgentRunException as a_exc:
                stop_after_tool_call_from_exception = _handle_agent_exception_from_tool_call(
                    function_call, a_exc, additional_input
                )
                function_call_success = False
            except Exception as e:
                log_error(
                    f"Error while iterating function result generator for {function_call.function.name}: {str(e)}"
                )
                function_call.error = str(e)
                function_call_success = False

            # For generators, re-capture updated_session_state after consumption
            # since session_state modifications were made during iteration
            if function_execution_result.updated_session_state is None:
                if (
                    function_call.function._run_context is not None
                    and function_call.function._run_context.session_state is not None
                ):
                    function_execution_result.updated_session_state = function_call.function._run_context.session_state
        else:
            from agno.tools.function import ToolResult

            if isinstance(function_execution_result.result, ToolResult):
                # Extract content and media from ToolResult
                tool_result = function_execution_result.result
                function_call_output = tool_result.content

                # Transfer media from ToolResult to FunctionExecutionResult
                if tool_result.images:
                    function_execution_result.images = tool_result.images
                if tool_result.videos:
                    function_execution_result.videos = tool_result.videos
                if tool_result.audios:
                    function_execution_result.audios = tool_result.audios
                if tool_result.files:
                    function_execution_result.files = tool_result.files
            else:
                function_call_output = str(function_execution_result.result) if function_execution_result.result else ""

            if function_call.function.show_result and function_call_output is not None:
                yield ModelResponse(content=function_call_output)

        # Create ToolCallMetrics for the tool execution
        tool_metrics = None
        if function_call_timer is not None and function_call_timer.elapsed > 0:
            from time import time

            tool_metrics = ToolCallMetrics()
            tool_metrics.timer = function_call_timer
            tool_metrics.duration = function_call_timer.elapsed
            # Calculate Unix timestamps (Timer uses perf_counter which is relative)
            current_time = time()
            tool_metrics.end_time = current_time
            tool_metrics.start_time = current_time - function_call_timer.elapsed

        # Replace an oversized successful result with its stored envelope
        # BEFORE the tool message (and the ToolExecution derived from it) is
        # built. With no result_store this is a no-op passthrough.
        if result_store is not None:
            function_call_output = self._substitute_tool_result(
                result_store, function_call, function_call_success, function_call_output
            )
        # Create and yield function call result
        function_call_result = self.create_function_call_result(
            function_call,
            success=function_call_success,
            output=function_call_output,
            timer=function_call_timer,
            function_execution_result=function_execution_result,
        )
        # Override stop_after_tool_call if set by exception
        if stop_after_tool_call_from_exception:
            function_call_result.stop_after_tool_call = True
        yield ModelResponse(
            content=f"{function_call.get_call_str()} completed in {function_call_timer.elapsed:.4f}s. ",
            tool_executions=[
                ToolExecution(
                    tool_call_id=function_call_result.tool_call_id,
                    tool_name=function_call_result.tool_name,
                    tool_args=function_call_result.tool_args,
                    tool_call_error=function_call_result.tool_call_error,
                    result=str(function_call_result.content),
                    stop_after_tool_call=function_call_result.stop_after_tool_call,
                    metrics=tool_metrics,
                )
            ],
            event=ModelResponseEvent.tool_call_completed.value,
            updated_session_state=function_execution_result.updated_session_state,
            # Add media artifacts from function execution
            images=function_execution_result.images,
            videos=function_execution_result.videos,
            audios=function_execution_result.audios,
            files=function_execution_result.files,
        )

        # Add function call to function call results
        function_call_results.append(function_call_result)

    def run_function_calls(
        self,
        function_calls: List[FunctionCall],
        function_call_results: List[Message],
        additional_input: Optional[List[Message]] = None,
        current_function_call_count: int = 0,
        function_call_limit: Optional[int] = None,
        result_store: Optional["ResultStore"] = None,
    ) -> Iterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        from agno.offload.types import NEVER_OFFLOADED_TOOLS

        # Additional messages from function calls that will be added to the function call results
        if additional_input is None:
            additional_input = []

        for fc in function_calls:
            # The read-back tools exist only because offloading replaced a result
            # the model was told to go and read. Counting them against the limit
            # can refuse the very read the run needs to answer.
            counts_against_limit = result_store is None or fc.function.name not in NEVER_OFFLOADED_TOOLS
            if function_call_limit is not None and counts_against_limit:
                current_function_call_count += 1
                # We have reached the function call limit, so we add an error result to the function call results
                if current_function_call_count > function_call_limit:
                    log_debug(
                        f"Tool call limit ({function_call_limit}) reached. "
                        f"Skipping: {fc.function.name} (call #{current_function_call_count})"
                    )
                    function_call_results.append(self.create_tool_call_limit_error_result(fc))
                    continue

            paused_tool_executions = _create_static_paused_tool_executions(fc)
            if paused_tool_executions:
                # Mirror the dynamic-pause sequence: emit tool_call_started before
                # tool_call_paused so downstream SSE consumers see a uniform
                # "started → paused" pair regardless of pause origin.
                yield ModelResponse(
                    content=fc.get_call_str(),
                    tool_executions=[
                        ToolExecution(
                            tool_call_id=fc.call_id,
                            tool_name=fc.function.name,
                            tool_args=fc.arguments,
                        )
                    ],
                    event=ModelResponseEvent.tool_call_started.value,
                )
                yield ModelResponse(
                    tool_executions=paused_tool_executions,
                    event=ModelResponseEvent.tool_call_paused.value,
                )
                if additional_input:
                    function_call_results.extend(additional_input)
                return

            for response in self.run_function_call(
                function_call=fc,
                function_call_results=function_call_results,
                additional_input=additional_input,
                result_store=result_store,
            ):
                yield response
                if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_paused.value:
                    if additional_input:
                        function_call_results.extend(additional_input)
                    return

        # Add any additional messages at the end
        if additional_input:
            function_call_results.extend(additional_input)

    async def arun_function_call(
        self,
        function_call: FunctionCall,
    ) -> Tuple[
        Union[bool, AgentRunException, ToolApprovalRequired, ToolCallDeferred],
        Timer,
        FunctionCall,
        FunctionExecutionResult,
    ]:
        """Run a single function call and return its success status, timer, and the FunctionCall object."""
        from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction

        function_call_timer = Timer()
        function_call_timer.start()
        success: Union[bool, AgentRunException, ToolApprovalRequired, ToolCallDeferred] = False
        result: FunctionExecutionResult = FunctionExecutionResult(status="failure")

        try:
            if (
                iscoroutinefunction(function_call.function.entrypoint)
                or isasyncgenfunction(function_call.function.entrypoint)
                or iscoroutine(function_call.function.entrypoint)
            ):
                result = await function_call.aexecute()
                success = result.status == "success"

            # If any of the hooks are async, we need to run the function call asynchronously
            elif function_call._requires_async_execution():
                result = await function_call.aexecute()
                success = result.status == "success"
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, function_call.execute)
                success = result.status == "success"
        except (ToolApprovalRequired, ToolCallDeferred) as e:
            success = e
        except AgentRunException as e:
            success = e
        except RunCancelledException:
            raise
        except Exception as e:
            log_error(f"Error executing function {function_call.function.name}: {str(e)}")
            success = False
            raise e

        function_call_timer.stop()
        return success, function_call_timer, function_call, result

    async def arun_function_calls(
        self,
        function_calls: List[FunctionCall],
        function_call_results: List[Message],
        additional_input: Optional[List[Message]] = None,
        current_function_call_count: int = 0,
        function_call_limit: Optional[int] = None,
        skip_pause_check: bool = False,
        result_store: Optional["ResultStore"] = None,
        run_id: Optional[str] = None,
    ) -> AsyncIterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        # Additional messages from function calls that will be added to the function call results
        if additional_input is None:
            additional_input = []

        from agno.offload.types import NEVER_OFFLOADED_TOOLS

        function_calls_to_run = []
        for fc in function_calls:
            # The read-back tools exist only because offloading replaced a result
            # the model was told to go and read. Counting them against the limit
            # can refuse the very read the run needs to answer.
            counts_against_limit = result_store is None or fc.function.name not in NEVER_OFFLOADED_TOOLS
            if function_call_limit is not None and counts_against_limit:
                current_function_call_count += 1
                # We have reached the function call limit, so we add an error result to the function call results
                if current_function_call_count > function_call_limit:
                    log_debug(
                        f"Tool call limit ({function_call_limit}) reached. "
                        f"Skipping: {fc.function.name} (call #{current_function_call_count})"
                    )
                    function_call_results.append(self.create_tool_call_limit_error_result(fc))
                    # Skip this function call
                    continue
            function_calls_to_run.append(fc)

        if any(_function_call_uses_thread(fc) for fc in function_calls_to_run):
            thread_function_calls = []
            # For each statically-paused call, remember the started + paused responses
            # so we can emit "started → paused" together, matching the dynamic flow.
            static_pause_responses: Dict[str, Tuple[ModelResponse, ModelResponse]] = {}

            for fc in function_calls_to_run:
                if not skip_pause_check:
                    paused_tool_executions = _create_static_paused_tool_executions(fc)
                    if paused_tool_executions:
                        response_key = fc.call_id or fc.function.name or str(id(fc))
                        started_response = ModelResponse(
                            content=fc.get_call_str(),
                            tool_executions=[
                                ToolExecution(
                                    tool_call_id=fc.call_id,
                                    tool_name=fc.function.name,
                                    tool_args=fc.arguments,
                                )
                            ],
                            event=ModelResponseEvent.tool_call_started.value,
                        )
                        paused_response = ModelResponse(
                            tool_executions=paused_tool_executions,
                            event=ModelResponseEvent.tool_call_paused.value,
                        )
                        static_pause_responses[response_key] = (started_response, paused_response)
                        continue

                if _function_call_uses_thread(fc):
                    thread_function_calls.append(fc)

            async def _yield_sync_function_call(
                fc: FunctionCall,
            ) -> AsyncIterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue[Any] = asyncio.Queue()
                done_marker = object()
                local_function_call_results: List[Message] = []

                def _run_sync_function_call() -> List[Message]:
                    try:
                        for response in self.run_function_call(
                            function_call=fc,
                            function_call_results=local_function_call_results,
                            additional_input=None,
                            result_store=result_store,
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait, response)
                    except BaseException as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, exc)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, done_marker)
                    return local_function_call_results

                sync_future = loop.run_in_executor(None, _run_sync_function_call)
                try:
                    while True:
                        if run_id:
                            try:
                                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                            except asyncio.TimeoutError:
                                from agno.run.cancel import ais_cancelled

                                if await ais_cancelled(run_id):
                                    raise RunCancelledException("Run cancelled while executing tool calls")
                                continue
                        else:
                            item = await queue.get()

                        if item is done_marker:
                            break
                        if isinstance(item, BaseException):
                            raise item
                        yield item
                    completed_results = await sync_future
                    function_call_results.extend(completed_results)
                except (asyncio.CancelledError, RunCancelledException):
                    sync_future.cancel()
                    raise

            thread_call_object_ids = {id(fc) for fc in thread_function_calls}
            pending_non_thread_function_calls: List[FunctionCall] = []

            async def _flush_pending_non_thread_function_calls() -> AsyncIterator[
                Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]
            ]:
                if not pending_non_thread_function_calls:
                    return

                batch = list(pending_non_thread_function_calls)
                pending_non_thread_function_calls.clear()
                async for response in self.arun_function_calls(
                    batch,
                    function_call_results,
                    skip_pause_check=True,
                    result_store=result_store,
                    run_id=run_id,
                ):
                    yield response
                    if (
                        isinstance(response, ModelResponse)
                        and response.event == ModelResponseEvent.tool_call_paused.value
                    ):
                        return
                return

            for fc in function_calls_to_run:
                response_key = fc.call_id or fc.function.name or str(id(fc))
                static_pause_pair = static_pause_responses.get(response_key)
                if static_pause_pair is not None:
                    async for response in _flush_pending_non_thread_function_calls():
                        yield response
                        if (
                            isinstance(response, ModelResponse)
                            and response.event == ModelResponseEvent.tool_call_paused.value
                        ):
                            if additional_input:
                                function_call_results.extend(additional_input)
                            return
                    started_response, paused_response = static_pause_pair
                    yield started_response
                    yield paused_response
                    if additional_input:
                        function_call_results.extend(additional_input)
                    return

                if id(fc) not in thread_call_object_ids:
                    pending_non_thread_function_calls.append(fc)
                    continue

                async for response in _flush_pending_non_thread_function_calls():
                    yield response
                    if (
                        isinstance(response, ModelResponse)
                        and response.event == ModelResponseEvent.tool_call_paused.value
                    ):
                        if additional_input:
                            function_call_results.extend(additional_input)
                        return

                async for response in _yield_sync_function_call(fc):
                    yield response
                    if (
                        isinstance(response, ModelResponse)
                        and response.event == ModelResponseEvent.tool_call_paused.value
                    ):
                        if additional_input:
                            function_call_results.extend(additional_input)
                        return

            async for response in _flush_pending_non_thread_function_calls():
                yield response
                if isinstance(response, ModelResponse) and response.event == ModelResponseEvent.tool_call_paused.value:
                    if additional_input:
                        function_call_results.extend(additional_input)
                    return

            if additional_input:
                function_call_results.extend(additional_input)
            return

        first_static_pause_response: Optional[ModelResponse] = None
        first_static_pause_started: Optional[ModelResponse] = None
        if not skip_pause_check:
            runnable_function_calls = []
            for fc in function_calls_to_run:
                paused_tool_executions = _create_static_paused_tool_executions(fc)
                if paused_tool_executions:
                    # Build the paired started event so we emit the same
                    # "started → paused" sequence dynamic pauses produce.
                    first_static_pause_started = ModelResponse(
                        content=fc.get_call_str(),
                        tool_executions=[
                            ToolExecution(
                                tool_call_id=fc.call_id,
                                tool_name=fc.function.name,
                                tool_args=fc.arguments,
                            )
                        ],
                        event=ModelResponseEvent.tool_call_started.value,
                    )
                    first_static_pause_response = ModelResponse(
                        tool_executions=paused_tool_executions,
                        event=ModelResponseEvent.tool_call_paused.value,
                    )
                    break
                runnable_function_calls.append(fc)
            function_calls_to_run = runnable_function_calls

        # Yield tool_call_started events for all calls that can run before the first static pause.
        for fc in function_calls_to_run:
            yield ModelResponse(
                content=fc.get_call_str(),
                tool_executions=[
                    ToolExecution(
                        tool_call_id=fc.call_id,
                        tool_name=fc.function.name,
                        tool_args=fc.arguments,
                    )
                ],
                event=ModelResponseEvent.tool_call_started.value,
            )

        # Create and run all function calls in parallel.
        results: List[Any] = [None] * len(function_calls_to_run)
        paused_function_calls: List[Tuple[int, FunctionCall, ToolPauseException]] = []
        paused_generator_calls: List[Tuple[int, FunctionCall, ToolPauseException]] = []
        paused_generator_ids: Set[int] = set()
        async_generator_results: List[Any] = []
        async_generator_outputs: Dict[int, Tuple[Any, str, Optional[BaseException]]] = {}
        async_generator_result_ids: Dict[int, int] = {}
        event_queue: asyncio.Queue = asyncio.Queue()
        active_generators_count = 0
        completed_generators_count = 0
        child_run_ids: Dict[int, str] = {}
        generator_tasks: List[asyncio.Task] = []
        generator_task_ids: Dict[asyncio.Task, int] = {}
        cancelled_generator_result_ids: Set[int] = set()
        function_call_indices = {id(fc): index for index, fc in enumerate(function_calls_to_run)}

        async def process_async_generator(result, generator_id):
            function_call_success, function_call_timer, function_call, function_execution_result = result
            function_call_output = ""

            try:
                async for item in function_call.result:
                    if generator_id not in child_run_ids and hasattr(item, "run_id") and item.run_id:
                        child_run_ids[generator_id] = item.run_id
                    # This function yields agent/team/workflow run events
                    if isinstance(item, _ALL_RUN_OUTPUT_EVENT_TYPES):
                        # We only capture content events
                        if isinstance(item, RunContentEvent) or isinstance(item, TeamRunContentEvent):
                            if item.content is not None and isinstance(item.content, BaseModel):
                                function_call_output += item.content.model_dump_json()
                            else:
                                # Capture output
                                function_call_output += item.content or ""

                            if function_call.function.show_result and item.content is not None:
                                await event_queue.put(ModelResponse(content=item.content))
                                continue

                        if isinstance(item, CustomEvent):
                            function_call_output += str(item)
                            item.tool_call_id = function_call.call_id

                            # For WorkflowCompletedEvent, extract content for final output
                            from agno.run.workflow import WorkflowCompletedEvent

                            if isinstance(item, WorkflowCompletedEvent):
                                if item.content is not None:
                                    if isinstance(item.content, BaseModel):
                                        function_call_output += item.content.model_dump_json()
                                    else:
                                        function_call_output += str(item.content)

                        # Put the event into the queue to be yielded
                        await event_queue.put(item)

                    # Yield custom events emitted by the tool
                    else:
                        function_call_output += str(item)
                        if function_call.function.show_result and item is not None:
                            await event_queue.put(ModelResponse(content=str(item)))

                # Store the final output for this generator
                async_generator_outputs[generator_id] = (result, function_call_output, None)

            except (ToolApprovalRequired, ToolCallDeferred) as e:
                # Pause exceptions are control-flow signals, not errors: don't poison function_call.error
                async_generator_outputs[generator_id] = (result, "", e)
            except Exception as e:
                function_call.error = str(e)
                async_generator_outputs[generator_id] = (result, "", e)

            # Signal that this generator is done
            await event_queue.put(("GENERATOR_DONE", generator_id))

        def _start_async_generator(result) -> None:
            nonlocal active_generators_count

            generator_id = len(async_generator_results)
            async_generator_results.append(result)
            async_generator_result_ids[id(result)] = generator_id
            active_generators_count += 1
            task = asyncio.create_task(process_async_generator(result, generator_id))
            generator_tasks.append(task)
            generator_task_ids[task] = generator_id

        def _record_generator_pause(generator_id: int) -> bool:
            if generator_id in paused_generator_ids or generator_id not in async_generator_outputs:
                return False

            result, _, error = async_generator_outputs[generator_id]
            if not _is_tool_pause_exception(error):
                return False

            _, _, function_call, _ = result
            paused_generator_calls.append(
                (
                    function_call_indices[id(function_call)],
                    function_call,
                    cast(ToolPauseException, error),
                )
            )
            paused_generator_ids.add(generator_id)
            return True

        _CHILD_CANCEL_TIMEOUT_SECONDS = 10.0

        async def _cancel_pending_child_tasks(
            tasks: List[asyncio.Task],
            child_ids: Dict[int, str],
        ) -> None:
            from agno.run.cancel import acancel_run

            for task in tasks:
                task.cancel()

            for c_run_id in child_ids.values():
                try:
                    await acancel_run(c_run_id)
                except Exception:
                    pass

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=_CHILD_CANCEL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                log_warning(
                    f"Child agent tasks did not finish within {_CHILD_CANCEL_TIMEOUT_SECONDS}s "
                    "after cancellation signal; they will continue running in the background."
                )

        async def _cancel_remaining_generators() -> None:
            pending_tasks = [task for task in generator_tasks if not task.done()]
            if not pending_tasks:
                return

            cancelled_generator_ids = [generator_task_ids[task] for task in pending_tasks]
            await _cancel_pending_child_tasks(pending_tasks, child_run_ids)
            await asyncio.sleep(0)
            for generator_id in cancelled_generator_ids:
                if generator_id in async_generator_outputs:
                    continue

                original_result = async_generator_results[generator_id]
                _, function_call_timer, function_call, function_execution_result = original_result
                function_call.error = _cancelled_tool_call_content(function_call)
                cancelled_result = (False, function_call_timer, function_call, function_execution_result)
                async_generator_results[generator_id] = cancelled_result
                async_generator_result_ids.pop(id(original_result), None)
                async_generator_result_ids[id(cancelled_result)] = generator_id
                cancelled_generator_result_ids.add(id(cancelled_result))
                for result_index, result in enumerate(results):
                    if result is original_result:
                        results[result_index] = cancelled_result
                        break

        if function_calls_to_run:
            task_to_index = {
                asyncio.create_task(self.arun_function_call(fc)): index
                for index, fc in enumerate(function_calls_to_run)
            }
            pending_function_tasks: Set[asyncio.Task] = set(task_to_index.keys())
            event_get_task: Optional[asyncio.Task] = None

            async def _drain_cancelled_function_tasks(tasks: List[asyncio.Task]) -> None:
                await asyncio.gather(*tasks, return_exceptions=True)

            async def _settle_pending_tasks_after_pause(tasks: List[asyncio.Task]) -> None:
                ordered_tasks = sorted(tasks, key=lambda task: task_to_index[task])
                pending_tasks = set(ordered_tasks)
                task_results: Dict[asyncio.Task, Any] = {}
                settle_deadline = asyncio.get_running_loop().time() + _TOOL_CALL_BATCH_SETTLE_TIMEOUT_SECONDS

                while pending_tasks:
                    timeout_remaining = settle_deadline - asyncio.get_running_loop().time()
                    if timeout_remaining <= 0:
                        break

                    wait_timeout = min(0.1, timeout_remaining) if run_id else timeout_remaining
                    done_tasks, pending_tasks = await asyncio.wait(
                        pending_tasks,
                        timeout=wait_timeout,
                        return_when=asyncio.ALL_COMPLETED,
                    )
                    for settled_task in done_tasks:
                        try:
                            task_results[settled_task] = settled_task.result()
                        except BaseException as exc:
                            task_results[settled_task] = exc

                    if pending_tasks and run_id:
                        from agno.run.cancel import ais_cancelled

                        if await ais_cancelled(run_id):
                            raise RunCancelledException("Run cancelled while executing tool calls")

                if pending_tasks:
                    log_warning(
                        f"Tool calls did not finish within {_TOOL_CALL_BATCH_SETTLE_TIMEOUT_SECONDS}s after a dynamic pause; "
                        "cancelling unfinished calls before returning the pause."
                    )
                    for pending_task in pending_tasks:
                        pending_task.cancel()
                    settled_results = await asyncio.gather(*pending_tasks, return_exceptions=True)
                    for settled_task, settled_result in zip(pending_tasks, settled_results):
                        task_results[settled_task] = settled_result
                    pending_tasks = set()

                for settled_task in ordered_tasks:
                    if settled_task in task_results:
                        settled_result = task_results[settled_task]
                    elif settled_task.done():
                        try:
                            settled_result = settled_task.result()
                        except BaseException as exc:
                            settled_result = exc
                    elif settled_task.cancelled():
                        settled_result = asyncio.CancelledError()
                    else:
                        settled_result = asyncio.CancelledError()

                    result_index = task_to_index[settled_task]
                    results[result_index] = settled_result
                    if not isinstance(settled_result, BaseException) and isinstance(
                        settled_result[0], (ToolApprovalRequired, ToolCallDeferred)
                    ):
                        _, _, function_call, _ = settled_result
                        paused_function_calls.append((result_index, function_call, settled_result[0]))
                        continue
                    if not isinstance(settled_result, BaseException) and isinstance(
                        settled_result[2].result, (AsyncGeneratorType, collections.abc.AsyncIterator)
                    ):
                        _start_async_generator(settled_result)

            try:
                while pending_function_tasks or completed_generators_count < active_generators_count:
                    wait_tasks: Set[asyncio.Task] = set(pending_function_tasks)
                    if completed_generators_count < active_generators_count:
                        if event_get_task is None:
                            event_get_task = asyncio.create_task(event_queue.get())
                        wait_tasks.add(event_get_task)

                    if not wait_tasks:
                        break

                    if run_id:
                        done_tasks, _ = await asyncio.wait(wait_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                        if not done_tasks:
                            from agno.run.cancel import ais_cancelled

                            if await ais_cancelled(run_id):
                                for pending_task in pending_function_tasks:
                                    pending_task.cancel()
                                if pending_function_tasks:
                                    await _drain_cancelled_function_tasks(list(pending_function_tasks))
                                    pending_function_tasks = set()
                                await _cancel_remaining_generators()
                                raise RunCancelledException("Run cancelled while executing tool calls")
                            continue
                    else:
                        done_tasks, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

                    pause_seen = False
                    if event_get_task is not None and event_get_task in done_tasks:
                        event = event_get_task.result()
                        event_get_task = None
                        if isinstance(event, tuple) and event[0] == "GENERATOR_DONE":
                            completed_generators_count += 1
                            generator_id = event[1]
                            pause_seen = _record_generator_pause(generator_id) or pause_seen
                        else:
                            yield event

                    done_function_tasks = done_tasks & pending_function_tasks
                    pending_function_tasks -= done_function_tasks
                    for task in sorted(done_function_tasks, key=lambda done_task: task_to_index[done_task]):
                        result_index = task_to_index[task]
                        try:
                            result = task.result()
                        except BaseException as exc:
                            results[result_index] = exc
                            continue

                        results[result_index] = result
                        function_call_success, _, function_call, _ = result
                        if isinstance(function_call_success, (ToolApprovalRequired, ToolCallDeferred)):
                            paused_function_calls.append((result_index, function_call, function_call_success))
                            pause_seen = True
                        elif isinstance(function_call.result, (AsyncGeneratorType, collections.abc.AsyncIterator)):
                            _start_async_generator(result)

                    if pause_seen:
                        while not event_queue.empty():
                            event = event_queue.get_nowait()
                            if isinstance(event, tuple) and event[0] == "GENERATOR_DONE":
                                completed_generators_count += 1
                                generator_id = event[1]
                                _record_generator_pause(generator_id)
                            else:
                                yield event

                        if pending_function_tasks:
                            await _settle_pending_tasks_after_pause(list(pending_function_tasks))
                            pending_function_tasks = set()
                        await _cancel_remaining_generators()
                        await asyncio.sleep(0)
                        break

                    if run_id:
                        from agno.run.cancel import ais_cancelled

                        if await ais_cancelled(run_id):
                            for pending_task in pending_function_tasks:
                                pending_task.cancel()
                            if pending_function_tasks:
                                await _drain_cancelled_function_tasks(list(pending_function_tasks))
                                pending_function_tasks = set()
                            await _cancel_remaining_generators()
                            raise RunCancelledException("Run cancelled while executing tool calls")
            finally:
                if event_get_task is not None:
                    event_get_task.cancel()
                    await asyncio.gather(event_get_task, return_exceptions=True)
                if pending_function_tasks:
                    for pending_task in pending_function_tasks:
                        pending_task.cancel()
                    await _drain_cancelled_function_tasks(list(pending_function_tasks))
                await _cancel_remaining_generators()

        for generator_id in list(async_generator_outputs):
            _record_generator_pause(generator_id)

        if paused_function_calls or paused_generator_calls:
            for index, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    cancelled_function_call = function_calls_to_run[index]
                    cancelled_content = _cancelled_tool_call_content(cancelled_function_call)
                    cancelled_function_call.error = cancelled_content
                    cancelled_timer = Timer()
                    cancelled_timer.start()
                    cancelled_timer.stop()
                    results[index] = (
                        False,
                        cancelled_timer,
                        cancelled_function_call,
                        FunctionExecutionResult(status="failure"),
                    )

        if paused_function_calls:
            results = [
                result
                for result in results
                if result is not None
                and not isinstance(result, BaseException)
                and not isinstance(result[0], (ToolApprovalRequired, ToolCallDeferred))
            ]

        if paused_generator_calls:
            filtered_results = []
            paused_results = []
            for result in results:
                if isinstance(result, BaseException):
                    filtered_results.append(result)
                    continue

                _, _, function_call, _ = result
                if isinstance(function_call.result, (AsyncGeneratorType, collections.abc.AsyncIterator)):
                    generator_id = async_generator_result_ids.get(id(result))
                    if generator_id is None:
                        continue
                    if generator_id not in async_generator_outputs:
                        if id(result) in cancelled_generator_result_ids:
                            filtered_results.append(result)
                        continue

                    _, _, error = async_generator_outputs[generator_id]
                    if _is_tool_pause_exception(error):
                        paused_results.append(result)
                        continue

                filtered_results.append(result)

            filtered_results.extend(paused_results)
            results = filtered_results

        # Now process all results (non-async generators and completed async generators)
        for i, original_result in enumerate(results):
            # If result is an exception, skip processing it
            if isinstance(original_result, BaseException):
                # Cancellation is intentional, not an error — re-raise without logging
                if isinstance(original_result, RunCancelledException):
                    raise original_result
                log_error(f"Error during function call: {original_result}")
                raise original_result

            # Unpack result
            function_call_success, function_call_timer, function_call, function_execution_result = original_result

            # Check if this was an async generator that was already processed
            async_function_call_output = None
            if isinstance(function_call.result, (AsyncGeneratorType, collections.abc.AsyncIterator)):
                generator_id = async_generator_result_ids.get(id(original_result))
                if generator_id is not None and generator_id in async_generator_outputs:
                    _, async_function_call_output, error = async_generator_outputs[generator_id]
                    if error:
                        if isinstance(error, RunCancelledException):
                            raise error
                        if _is_tool_pause_exception(error):
                            function_call_success = error
                        elif isinstance(error, AgentRunException):
                            function_call_success = error
                        else:
                            # Handle async generator exceptions gracefully like sync generators
                            log_error(
                                f"Error while iterating async generator for {function_call.function.name}: {error}"
                            )
                            function_call.error = str(error)
                            function_call_success = False

            updated_session_state = function_execution_result.updated_session_state

            # Handle AgentRunException
            stop_after_tool_call_from_exception = False
            if isinstance(function_call_success, (ToolApprovalRequired, ToolCallDeferred)):
                continue

            if isinstance(function_call_success, AgentRunException):
                stop_after_tool_call_from_exception = _handle_agent_exception_from_tool_call(
                    function_call, function_call_success, additional_input
                )
                function_call_success = False

            # Process function call output
            function_call_output: str = ""

            # Check if this was an async generator that was already processed
            if async_function_call_output is not None:
                function_call_output = async_function_call_output
                # Events from async generators were already yielded in real-time above
            elif isinstance(function_call.result, (GeneratorType, collections.abc.Iterator)):
                try:
                    for item in function_call.result:
                        # This function yields agent/team/workflow run events
                        if isinstance(item, _ALL_RUN_OUTPUT_EVENT_TYPES):
                            # We only capture content events
                            if isinstance(item, RunContentEvent) or isinstance(item, TeamRunContentEvent):
                                if item.content is not None and isinstance(item.content, BaseModel):
                                    function_call_output += item.content.model_dump_json()
                                else:
                                    # Capture output
                                    function_call_output += item.content or ""

                                if function_call.function.show_result and item.content is not None:
                                    yield ModelResponse(content=item.content)
                                    continue

                            elif isinstance(item, CustomEvent):
                                function_call_output += str(item)
                                item.tool_call_id = function_call.call_id

                            # Yield the event itself to bubble it up
                            yield item
                        else:
                            function_call_output += str(item)
                            if function_call.function.show_result and item is not None:
                                yield ModelResponse(content=str(item))
                except RunCancelledException:
                    raise
                except (ToolApprovalRequired, ToolCallDeferred) as pause_exc:
                    yield _create_tool_call_paused_response(function_call, pause_exc)
                    return
                except AgentRunException as a_exc:
                    stop_after_tool_call_from_exception = _handle_agent_exception_from_tool_call(
                        function_call, a_exc, additional_input
                    )
                    function_call_success = False
                except Exception as e:
                    log_error(
                        f"Error while iterating function result generator for {function_call.function.name}: {str(e)}"
                    )
                    function_call.error = str(e)
                    function_call_success = False

            # For generators (sync or async), re-capture updated_session_state after consumption
            # since session_state modifications were made during iteration
            if async_function_call_output is not None or isinstance(
                function_call.result,
                (GeneratorType, collections.abc.Iterator, AsyncGeneratorType, collections.abc.AsyncIterator),
            ):
                if updated_session_state is None:
                    if (
                        function_call.function._run_context is not None
                        and function_call.function._run_context.session_state is not None
                    ):
                        updated_session_state = function_call.function._run_context.session_state

            if not (
                async_function_call_output is not None
                or isinstance(
                    function_call.result,
                    (GeneratorType, collections.abc.Iterator, AsyncGeneratorType, collections.abc.AsyncIterator),
                )
            ):
                from agno.tools.function import ToolResult

                if isinstance(function_execution_result.result, ToolResult):
                    tool_result = function_execution_result.result
                    function_call_output = tool_result.content

                    if tool_result.images:
                        function_execution_result.images = tool_result.images
                    if tool_result.videos:
                        function_execution_result.videos = tool_result.videos
                    if tool_result.audios:
                        function_execution_result.audios = tool_result.audios
                    if tool_result.files:
                        function_execution_result.files = tool_result.files
                else:
                    function_call_output = str(function_call.result)

                if function_call.function.show_result and function_call_output is not None:
                    yield ModelResponse(content=function_call_output)

            # Create ToolCallMetrics for the tool execution
            tool_metrics = None
            if function_call_timer is not None and function_call_timer.elapsed > 0:
                from time import time

                tool_metrics = ToolCallMetrics()
                tool_metrics.timer = function_call_timer
                tool_metrics.duration = function_call_timer.elapsed
                # Calculate Unix timestamps (Timer uses perf_counter which is relative)
                current_time = time()
                tool_metrics.end_time = current_time
                tool_metrics.start_time = current_time - function_call_timer.elapsed

            # Replace an oversized successful result with its stored envelope
            # BEFORE the tool message (and the ToolExecution derived from it)
            # is built. Async parity: the a-prefixed store methods do the I/O.
            if result_store is not None:
                function_call_output = await self._asubstitute_tool_result(
                    result_store, function_call, function_call_success, function_call_output
                )
            # Create and yield function call result
            function_call_result = self.create_function_call_result(
                function_call,
                success=function_call_success,
                output=function_call_output,
                timer=function_call_timer,
                function_execution_result=function_execution_result,
            )
            # Override stop_after_tool_call if set by exception
            if stop_after_tool_call_from_exception:
                function_call_result.stop_after_tool_call = True
            yield ModelResponse(
                content=f"{function_call.get_call_str()} completed in {function_call_timer.elapsed:.4f}s. ",
                tool_executions=[
                    ToolExecution(
                        tool_call_id=function_call_result.tool_call_id,
                        tool_name=function_call_result.tool_name,
                        tool_args=function_call_result.tool_args,
                        tool_call_error=function_call_result.tool_call_error,
                        result=str(function_call_result.content),
                        stop_after_tool_call=function_call_result.stop_after_tool_call,
                        metrics=tool_metrics,
                    )
                ],
                event=ModelResponseEvent.tool_call_completed.value,
                updated_session_state=updated_session_state,
                images=function_execution_result.images,
                videos=function_execution_result.videos,
                audios=function_execution_result.audios,
                files=function_execution_result.files,
            )

            # Add function call result to function call results
            function_call_results.append(function_call_result)

        indexed_paused_calls = paused_function_calls + paused_generator_calls
        if indexed_paused_calls:
            if additional_input:
                function_call_results.extend(additional_input)
            paused_calls = [
                (function_call, pause_exc)
                for _, function_call, pause_exc in sorted(indexed_paused_calls, key=lambda item: item[0])
            ]
            yield _create_tool_calls_paused_response(paused_calls)
            return

        if first_static_pause_response is not None:
            if additional_input:
                function_call_results.extend(additional_input)
            if first_static_pause_started is not None:
                yield first_static_pause_started
            yield first_static_pause_response
            return

        # Add any additional messages at the end
        if additional_input:
            function_call_results.extend(additional_input)

    def _prepare_function_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        model_response: ModelResponse,
        functions: Optional[Dict[str, Function]] = None,
    ) -> List[FunctionCall]:
        """
        Prepare function calls from tool calls in the assistant message.
        """
        if model_response.content is None:
            model_response.content = ""
        if model_response.tool_calls is None:
            model_response.tool_calls = []

        function_calls_to_run: List[FunctionCall] = self.get_function_calls_to_run(
            assistant_message=assistant_message, messages=messages, functions=functions
        )
        return function_calls_to_run

    def format_function_call_results(
        self,
        messages: List[Message],
        function_call_results: List[Message],
        compress_tool_results: bool = False,
        **kwargs,
    ) -> None:
        """
        Format function call results.
        """
        if len(function_call_results) > 0:
            messages.extend(function_call_results)

    def _handle_function_call_media(
        self, messages: List[Message], function_call_results: List[Message], send_media_to_model: bool = True
    ) -> None:
        """
        Handle media artifacts from function calls by adding follow-up user messages for generated media if needed.
        """
        if not function_call_results:
            return

        # Collect all media artifacts from function calls
        all_images: List[Image] = []
        all_videos: List[Video] = []
        all_audio: List[Audio] = []
        all_files: List[File] = []

        for result_message in function_call_results:
            if result_message.images:
                all_images.extend(result_message.images)
                # Remove images from tool message to avoid errors on the LLMs
                result_message.images = None

            if result_message.videos:
                all_videos.extend(result_message.videos)
                result_message.videos = None

            if result_message.audio:
                all_audio.extend(result_message.audio)
                result_message.audio = None

            if result_message.files:
                all_files.extend(result_message.files)
                result_message.files = None

        # Only add media message if we should send media to model
        if send_media_to_model and (all_images or all_videos or all_audio or all_files):
            # If we have media artifacts, add a follow-up "user" message instead of a "tool"
            # message with the media artifacts which throws error for some models
            media_message = Message(
                role="user",
                content="The tool call above generated the attached media.",
                images=all_images if all_images else None,
                videos=all_videos if all_videos else None,
                audio=all_audio if all_audio else None,
                files=all_files if all_files else None,
            )
            messages.append(media_message)

    def get_system_message_for_model(self, tools: Optional[List[Any]] = None) -> Optional[str]:
        return self.system_prompt

    def get_instructions_for_model(self, tools: Optional[List[Any]] = None) -> Optional[List[str]]:
        return self.instructions

    def __deepcopy__(self, memo):
        """Create a deep copy of the Model instance.

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass.

        Returns:
            Model: A new Model instance with deeply copied attributes.
        """
        from copy import copy, deepcopy

        # Create a new instance without calling __init__
        cls = self.__class__
        new_model = cls.__new__(cls)
        memo[id(self)] = new_model

        # Deep copy all attributes except client objects
        for k, v in self.__dict__.items():
            if k in {"response_format", "_tools", "_functions"}:
                continue
            # Skip client objects
            if k in {"client", "async_client", "http_client", "mistral_client", "model_client"}:
                setattr(new_model, k, None)
                continue
            try:
                setattr(new_model, k, deepcopy(v, memo))
            except Exception:
                try:
                    setattr(new_model, k, copy(v))
                except Exception:
                    setattr(new_model, k, v)

        return new_model
