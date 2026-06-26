"""Approval record creation and resolution gating for HITL tool runs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from agno.run.base import RunStatus
from agno.utils.dttm import now_epoch_s
from agno.utils.log import log_debug, log_warning


def _approval_belongs_to_run(approval: Any, run_response: Any) -> bool:
    return bool(
        isinstance(approval, dict)
        and approval.get("run_id") == getattr(run_response, "run_id", None)
        and approval.get("approval_type", "required") == "required"
    )


def _get_existing_approval_for_run(db: Any, approval_id: str, run_response: Any) -> Optional[Dict[str, Any]]:
    get_approval = getattr(db, "get_approval", None)
    if get_approval is None:
        return None
    try:
        approval = get_approval(approval_id)
    except Exception:
        return None
    return approval if _approval_belongs_to_run(approval, run_response) else None


async def _aget_existing_approval_for_run(db: Any, approval_id: str, run_response: Any) -> Optional[Dict[str, Any]]:
    get_approval = getattr(db, "get_approval", None)
    if get_approval is None:
        return None
    try:
        from inspect import iscoroutinefunction

        approval = await get_approval(approval_id) if iscoroutinefunction(get_approval) else get_approval(approval_id)
    except Exception:
        return None
    return approval if _approval_belongs_to_run(approval, run_response) else None


def _get_pause_type(tool_execution: Any) -> str:
    """Determine the pause type from a tool execution's HITL flags."""
    if getattr(tool_execution, "requires_user_input", False):
        return "user_input"
    if getattr(tool_execution, "external_execution_required", False):
        return "external_execution"
    return "confirmation"


def _get_first_approval_tool(tools: Optional[List[Any]], requirements: Optional[List[Any]] = None) -> Any:
    """Return the first tool execution that has approval_type set."""
    if tools:
        for tool in tools:
            if getattr(tool, "approval_type", None) is not None:
                return tool
    if requirements:
        for req in requirements:
            te = getattr(req, "tool_execution", None)
            if te and getattr(te, "approval_type", None) is not None:
                return te
    return None


def _is_active_approval_tool(tool: Any) -> bool:
    if tool is None or getattr(tool, "approval_type", None) != "required":
        return False

    if getattr(tool, "requires_confirmation", None) is True:
        return getattr(tool, "confirmed", None) is None
    if getattr(tool, "requires_user_input", None) is True:
        return getattr(tool, "answered", None) is not True
    if getattr(tool, "external_execution_required", None) is True:
        return getattr(tool, "external_execution_result_provided", None) is not True

    if (
        getattr(tool, "requires_confirmation", None) is False
        or getattr(tool, "requires_user_input", None) is False
        or getattr(tool, "external_execution_required", None) is False
        or getattr(tool, "confirmed", None) is not None
        or getattr(tool, "answered", None) is not None
        or getattr(tool, "external_execution_result_provided", None) is not None
    ):
        return False

    # Backwards-compatible default: older paused approval tools may only carry
    # approval_type="required", which semantically means confirmation.
    return True


def _tool_feedback_answered(tool: Any) -> bool:
    feedback_schema = getattr(tool, "user_feedback_schema", None) or []
    return bool(feedback_schema and all(question.selected_options is not None for question in feedback_schema))


def _get_first_active_approval_tool(tools: Optional[List[Any]], requirements: Optional[List[Any]] = None) -> Any:
    if tools:
        for tool in tools:
            if _is_active_approval_tool(tool):
                return tool
    if requirements:
        for req in requirements:
            te = getattr(req, "tool_execution", None)
            if te and _is_active_approval_tool(te):
                return te
    return None


def _has_approval_requirement(tools: Optional[List[Any]], requirements: Optional[List[Any]] = None) -> bool:
    """Check if any paused tool execution has approval_type set.

    Checks both run_response.tools (agent-level) and run_response.requirements
    (team-level, where member tools are propagated via requirements).
    """
    tool = _get_first_approval_tool(tools, requirements)
    return tool is not None and getattr(tool, "approval_type", None) == "required"


def _stamp_approval_id_on_tools(
    tools: Optional[List[Any]], requirements: Optional[List[Any]], approval_id: str
) -> None:
    """Stamp approval_id on active approval tools for the current pause."""
    if tools:
        for tool in tools:
            if _is_active_approval_tool(tool):
                tool.approval_id = approval_id
    if requirements:
        for req in requirements:
            te = getattr(req, "tool_execution", None)
            if te is not None and _is_active_approval_tool(te):
                te.approval_id = approval_id


def _get_existing_approval_id(tools: Optional[List[Any]], requirements: Optional[List[Any]]) -> Optional[str]:
    if tools:
        for tool in tools:
            if not _is_active_approval_tool(tool):
                continue
            approval_id = getattr(tool, "approval_id", None)
            if approval_id:
                return approval_id
    if requirements:
        for req in requirements:
            te = getattr(req, "tool_execution", None)
            if te is None or not _is_active_approval_tool(te):
                continue
            approval_id = getattr(te, "approval_id", None)
            if approval_id:
                return approval_id
    return None


def _build_approval_dict(
    run_response: Any,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    team_id: Optional[str] = None,
    team_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
    user_id: Optional[str] = None,
    schedule_id: Optional[str] = None,
    schedule_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the approval record dict from run response and context."""
    # Determine source type
    source_type = "agent"
    source_name = agent_name
    if team_id:
        source_type = "team"
        source_name = team_name
    elif workflow_id:
        source_type = "workflow"
        source_name = workflow_name

    # Serialize requirements
    requirements_data: Optional[List[Dict[str, Any]]] = None
    if hasattr(run_response, "requirements") and run_response.requirements:
        requirements_data = []
        for req in run_response.requirements:
            if hasattr(req, "to_dict"):
                requirements_data.append(req.to_dict())
            elif isinstance(req, dict):
                requirements_data.append(req)

    # Find the first approval tool to extract pause_type, tool_name, tool_args
    tools = getattr(run_response, "tools", None)
    requirements = getattr(run_response, "requirements", None)
    first_tool = _get_first_active_approval_tool(tools, requirements) or _get_first_approval_tool(tools, requirements)

    pause_type = _get_pause_type(first_tool) if first_tool else "confirmation"
    tool_name = getattr(first_tool, "tool_name", None) if first_tool else None
    tool_args = getattr(first_tool, "tool_args", None) if first_tool else None

    # Build context with tool names for UI display.
    tool_names: List[str] = []
    if hasattr(run_response, "requirements") and run_response.requirements:
        for req in run_response.requirements:
            te = getattr(req, "tool_execution", None)
            if te and getattr(te, "approval_type", None) is not None:
                name = getattr(te, "tool_name", None)
                if name:
                    tool_names.append(name)
    # Fallback: extract from run_response.tools
    if not tool_names and tools:
        for t in tools:
            if hasattr(t, "tool_name") and t.tool_name:
                tool_names.append(t.tool_name)

    context: Dict[str, Any] = {}
    if tool_names:
        context["tool_names"] = tool_names
    if source_name:
        context["source_name"] = source_name

    return {
        "id": str(uuid4()),
        "run_id": getattr(run_response, "run_id", None) or str(uuid4()),
        "session_id": getattr(run_response, "session_id", None) or "",
        "status": "pending",
        "approval_type": "required",
        "pause_type": pause_type,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "source_type": source_type,
        "agent_id": agent_id,
        "team_id": team_id,
        "workflow_id": workflow_id,
        "user_id": user_id,
        "schedule_id": schedule_id,
        "schedule_run_id": schedule_run_id,
        "source_name": source_name,
        "requirements": requirements_data,
        "context": context if context else None,
        "resolved_by": None,
        "resolved_at": None,
        "created_at": now_epoch_s(),
        "updated_at": None,
        # Run status is PAUSED when the approval is created (run is paused waiting for approval)
        "run_status": RunStatus.paused.value,
    }


def create_approval_from_pause(
    db: Any,
    run_response: Any,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    team_id: Optional[str] = None,
    team_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
    user_id: Optional[str] = None,
    schedule_id: Optional[str] = None,
    schedule_run_id: Optional[str] = None,
) -> Optional[str]:
    """Create an approval record when a run pauses for a tool with approval_type set.

    Returns the approval_id if a record was created, None otherwise.
    Silently returns None if no approval requirement is found or if DB doesn't support approvals.
    """
    if db is None:
        return None

    tools = getattr(run_response, "tools", None)
    requirements = getattr(run_response, "requirements", None)
    if _get_first_active_approval_tool(tools, requirements) is None:
        return None
    existing_approval_id = _get_existing_approval_id(tools, requirements)
    if existing_approval_id:
        if _get_existing_approval_for_run(db, existing_approval_id, run_response) is not None:
            return existing_approval_id

    try:
        approval_data = _build_approval_dict(
            run_response,
            agent_id=agent_id,
            agent_name=agent_name,
            team_id=team_id,
            team_name=team_name,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            user_id=user_id,
            schedule_id=schedule_id,
            schedule_run_id=schedule_run_id,
        )
        db.create_approval(approval_data)
        approval_id: str = approval_data["id"]
        # Stamp the approval_id only on tools that are still actively paused
        _stamp_approval_id_on_tools(tools, requirements, approval_id)
        log_debug(f"Created approval {approval_id} for run {approval_data['run_id']}")
        return approval_id
    except NotImplementedError:
        pass
    except Exception as e:
        log_warning(f"Error creating approval record (sync): {str(e)}")
    return None


async def acreate_approval_from_pause(
    db: Any,
    run_response: Any,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    team_id: Optional[str] = None,
    team_name: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
    user_id: Optional[str] = None,
    schedule_id: Optional[str] = None,
    schedule_run_id: Optional[str] = None,
) -> Optional[str]:
    """Async variant of create_approval_from_pause.

    Returns the approval_id if a record was created, None otherwise.
    """
    if db is None:
        return None

    tools = getattr(run_response, "tools", None)
    requirements = getattr(run_response, "requirements", None)
    if _get_first_active_approval_tool(tools, requirements) is None:
        return None
    existing_approval_id = _get_existing_approval_id(tools, requirements)
    if existing_approval_id:
        if await _aget_existing_approval_for_run(db, existing_approval_id, run_response) is not None:
            return existing_approval_id

    try:
        approval_data = _build_approval_dict(
            run_response,
            agent_id=agent_id,
            agent_name=agent_name,
            team_id=team_id,
            team_name=team_name,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            user_id=user_id,
            schedule_id=schedule_id,
            schedule_run_id=schedule_run_id,
        )
        # Try async first, fall back to sync
        create_fn = getattr(db, "create_approval", None)
        if create_fn is None:
            return None
        from inspect import iscoroutinefunction

        if iscoroutinefunction(create_fn):
            await create_fn(approval_data)
        else:
            create_fn(approval_data)
        approval_id: str = approval_data["id"]
        # Stamp the approval_id on all tools with approval_type
        _stamp_approval_id_on_tools(tools, requirements, approval_id)
        log_debug(f"Created approval {approval_id} for run {approval_data['run_id']}")
        return approval_id
    except NotImplementedError:
        pass
    except Exception as e:
        log_warning(f"Error creating approval record (async): {str(e)}")
    return None


def create_audit_approval(
    db: Any,
    tool_execution: Any,
    run_response: Any,
    status: str,  # "approved" or "rejected"
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    team_id: Optional[str] = None,
    team_name: Optional[str] = None,
    user_id: Optional[str] = None,
    pause_type: Optional[str] = None,
) -> None:
    """Create an audit approval record AFTER a HITL interaction resolves.

    Unlike create_approval_from_pause (which creates a 'pending' record before resolution),
    this creates a completed record (status='approved'/'rejected') for audit logging.
    Only called for tools with approval_type='audit'.
    """
    if db is None:
        return
    try:
        source_type = "agent"
        source_name = agent_name
        if team_id:
            source_type = "team"
            source_name = team_name

        tool_name = getattr(tool_execution, "tool_name", None)
        tool_args = getattr(tool_execution, "tool_args", None)
        pause_type = pause_type or _get_pause_type(tool_execution)

        context: Dict[str, Any] = {}
        if tool_name:
            context["tool_names"] = [tool_name]
        if source_name:
            context["source_name"] = source_name

        approval_data = {
            "id": str(uuid4()),
            "run_id": getattr(run_response, "run_id", None) or str(uuid4()),
            "session_id": getattr(run_response, "session_id", None) or "",
            "status": status,
            "approval_type": "audit",
            "pause_type": pause_type,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "source_type": source_type,
            "agent_id": agent_id,
            "team_id": team_id,
            "user_id": user_id,
            "source_name": source_name,
            "context": context if context else None,
            "resolved_at": now_epoch_s(),
            "created_at": now_epoch_s(),
            "updated_at": None,
        }
        db.create_approval(approval_data)
        log_debug(f"Audit approval {approval_data['id']} for tool {tool_name}")
    except NotImplementedError:
        pass
    except Exception as e:
        log_warning(f"Error creating audit approval record (sync): {str(e)}")


# ---------------------------------------------------------------------------
# Approval gate: enforce external resolution before continue
# ---------------------------------------------------------------------------


def _tool_user_input_ready(tool: Any) -> bool:
    user_input_schema = getattr(tool, "user_input_schema", None) or []
    return bool(user_input_schema) and all(getattr(field, "value", None) is not None for field in user_input_schema)


def _tool_user_feedback_ready(tool: Any) -> bool:
    user_feedback_schema = getattr(tool, "user_feedback_schema", None) or []
    return bool(user_feedback_schema) and all(
        getattr(question, "selected_options", None) is not None for question in user_feedback_schema
    )


def _mapping_value(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _resolution_values_and_selections(
    resolution_data: Optional[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not isinstance(resolution_data, dict):
        return None, None

    values = _mapping_value(resolution_data.get("values"))
    selections = _mapping_value(resolution_data.get("selections"))
    if selections is None:
        selections = _mapping_value(resolution_data.get("feedback"))

    if values is None and selections is None:
        values = _mapping_value(resolution_data)
    if selections is None:
        selections = values
    return values, selections


def _has_non_null_value(values: Optional[Dict[str, Any]], name: Any) -> bool:
    return values is not None and name in values and values[name] is not None


def _is_valid_feedback_selection(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _has_valid_feedback_selection(selections: Optional[Dict[str, Any]], question: Any) -> bool:
    return selections is not None and question in selections and _is_valid_feedback_selection(selections[question])


def _has_usable_approval_resolution(tool: Any, approval_status: str, resolution_data: Optional[Dict[str, Any]]) -> bool:
    if approval_status == "rejected":
        return True

    if approval_status != "approved":
        return False

    if getattr(tool, "requires_confirmation", False):
        return True

    if getattr(tool, "requires_user_input", False):
        if _tool_user_input_ready(tool) or _tool_user_feedback_ready(tool):
            return True
        if not isinstance(resolution_data, dict) or not resolution_data:
            return False

        values, selections = _resolution_values_and_selections(resolution_data)
        user_input_schema = getattr(tool, "user_input_schema", None) or []
        user_feedback_schema = getattr(tool, "user_feedback_schema", None) or []

        input_ready = bool(user_input_schema) and all(
            getattr(field, "value", None) is not None or _has_non_null_value(values, getattr(field, "name", None))
            for field in user_input_schema
        )
        feedback_ready = bool(user_feedback_schema) and all(
            getattr(question, "selected_options", None) is not None
            or _has_valid_feedback_selection(selections, getattr(question, "question", None))
            for question in user_feedback_schema
        )
        schema_less_ready = not user_input_schema and not user_feedback_schema
        return input_ready or feedback_ready or schema_less_ready

    if getattr(tool, "external_execution_required", False):
        return isinstance(resolution_data, dict) and "result" in resolution_data

    # Backwards-compatible default approval tools are confirmation approvals.
    return True


def _apply_approval_to_tools(tools: List[Any], approval_status: str, resolution_data: Optional[Dict[str, Any]]) -> None:
    """Apply approval resolution status to tools that require approval.

    For 'approved': sets confirmed=True, applies resolution_data to user_input/external_execution fields.
    For 'rejected': sets confirmed=False.
    """
    resume_metadata = None
    if isinstance(resolution_data, dict):
        resume_metadata = resolution_data.get("metadata", resolution_data.get("approval_metadata"))

    for tool in tools:
        if getattr(tool, "approval_type", None) != "required":
            continue

        if approval_status == "approved":
            if (
                not getattr(tool, "requires_confirmation", False)
                and not getattr(tool, "requires_user_input", False)
                and not getattr(tool, "external_execution_required", False)
            ):
                tool.requires_confirmation = True

            # Confirmation tools
            if getattr(tool, "requires_confirmation", False):
                tool.confirmed = True
                if resume_metadata is not None:
                    tool.resume_metadata = resume_metadata

            # User input tools: apply resolution_data values to user_input_schema
            if getattr(tool, "requires_user_input", False) and resolution_data:
                values, selections = _resolution_values_and_selections(resolution_data)
                values = values or {}
                selections = selections or {}
                for ufield in tool.user_input_schema or []:
                    if _has_non_null_value(values, ufield.name):
                        ufield.value = values[ufield.name]
                for question in getattr(tool, "user_feedback_schema", None) or []:
                    if _has_valid_feedback_selection(selections, question.question):
                        question.selected_options = selections[question.question]
                        if question.options:
                            for option in question.options:
                                option.selected = option.label in question.selected_options
                if (
                    (tool.user_input_schema and all(field.value is not None for field in tool.user_input_schema))
                    or _tool_feedback_answered(tool)
                    or (
                        not getattr(tool, "user_input_schema", None) and not getattr(tool, "user_feedback_schema", None)
                    )
                ):
                    tool.answered = True
                if resume_metadata is not None:
                    tool.resume_metadata = resume_metadata

            # External execution tools: apply resolution_data result
            if getattr(tool, "external_execution_required", False) and resolution_data:
                if "result" in resolution_data:
                    tool.result = resolution_data["result"]
                    tool.external_execution_result_provided = True
                if resume_metadata is not None:
                    tool.resume_metadata = resume_metadata

        elif approval_status == "rejected":
            note = None
            if isinstance(resolution_data, dict):
                note = resolution_data.get("note") or resolution_data.get("reason")
            if (
                not getattr(tool, "requires_confirmation", False)
                and not getattr(tool, "requires_user_input", False)
                and not getattr(tool, "external_execution_required", False)
            ):
                tool.requires_confirmation = True
            if getattr(tool, "requires_confirmation", False):
                tool.confirmed = False
                if note:
                    tool.confirmation_note = note
            if getattr(tool, "requires_user_input", False):
                tool.confirmed = False
                if note:
                    tool.confirmation_note = note
                tool.answered = True
            if getattr(tool, "external_execution_required", False):
                tool.confirmed = False
                if note:
                    tool.confirmation_note = note
                if not getattr(tool, "external_execution_result_provided", False):
                    tool.result = note or "Tool call was rejected"
                    tool.external_execution_result_provided = True


def _sync_requirements_from_tools(run_response: Any) -> None:
    requirements = getattr(run_response, "requirements", None) or []
    tools = getattr(run_response, "tools", None) or []
    if not requirements:
        return

    matched_tool_indexes: set[int] = set()
    for requirement in requirements:
        tool_execution = getattr(requirement, "tool_execution", None)
        if tool_execution is None:
            continue
        resolved_tool = None
        for index, tool in enumerate(tools):
            if index in matched_tool_indexes:
                continue
            if tool is tool_execution:
                resolved_tool = tool
                matched_tool_indexes.add(index)
                break

        if resolved_tool is None:
            requirement_approval_id = getattr(tool_execution, "approval_id", None)
            fallback_match = None
            fallback_index = None
            for index, tool in enumerate(tools):
                if index in matched_tool_indexes:
                    continue
                if getattr(tool, "tool_call_id", None) is None:
                    continue
                if getattr(tool, "tool_call_id", None) != getattr(tool_execution, "tool_call_id", None):
                    continue
                tool_approval_id = getattr(tool, "approval_id", None)
                if requirement_approval_id is not None and tool_approval_id not in (requirement_approval_id, None):
                    continue
                if (
                    requirement_approval_id is None
                    and getattr(tool_execution, "confirmed", None) is not None
                    and getattr(tool, "confirmed", None) is None
                ):
                    continue
                if _is_active_approval_tool(tool_execution) and _is_active_approval_tool(tool):
                    resolved_tool = tool
                    matched_tool_indexes.add(index)
                    break
                if fallback_match is None:
                    fallback_match = tool
                    fallback_index = index
            if resolved_tool is None and fallback_match is not None and fallback_index is not None:
                resolved_tool = fallback_match
                matched_tool_indexes.add(fallback_index)

        resolved_tool = resolved_tool or tool_execution
        if resolved_tool is not None:
            requirement.tool_execution = resolved_tool
            if getattr(resolved_tool, "resume_metadata", None) is not None:
                requirement.approval_metadata = getattr(resolved_tool, "resume_metadata", None)
            if getattr(resolved_tool, "confirmed", None) is not None:
                requirement.confirmation = resolved_tool.confirmed
                requirement.confirmation_note = getattr(resolved_tool, "confirmation_note", None)
            if getattr(resolved_tool, "external_execution_result_provided", None):
                requirement.external_execution_result = getattr(resolved_tool, "result", None)
                requirement.external_execution_result_provided = True
            if getattr(resolved_tool, "user_input_schema", None) is not None:
                requirement.user_input_schema = resolved_tool.user_input_schema
            if getattr(resolved_tool, "user_feedback_schema", None) is not None:
                requirement.user_feedback_schema = resolved_tool.user_feedback_schema

    requirement_tools = [req.tool_execution for req in requirements if getattr(req, "tool_execution", None) is not None]
    if requirement_tools and not getattr(run_response, "tools", None):
        run_response.tools = requirement_tools


def _get_active_approval_id(approval_tools: List[Any]) -> Optional[str]:
    for tool in approval_tools:
        if _is_active_approval_tool(tool):
            approval_id = getattr(tool, "approval_id", None)
            if approval_id:
                return approval_id
    return None


def _approval_tools_for_resolution(approval_tools: List[Any], approval_id: Optional[str]) -> List[Any]:
    active_tools = [tool for tool in approval_tools if _is_active_approval_tool(tool)]
    if approval_id is None:
        return active_tools

    matching_tools = [tool for tool in active_tools if getattr(tool, "approval_id", None) == approval_id]
    if matching_tools:
        return matching_tools

    # Legacy paused runs may predate per-tool approval ids. Only fall back to
    # unbound tools when no active tool carries any explicit approval id.
    if not any(getattr(tool, "approval_id", None) is not None for tool in active_tools):
        return active_tools

    return []


def _get_approval_for_run(db: Any, run_id: str, approval_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up the active approval for a run_id (sync)."""
    try:
        if approval_id:
            get_approval = getattr(db, "get_approval", None)
            if get_approval is not None:
                approval = get_approval(approval_id)
                if (
                    isinstance(approval, dict)
                    and approval.get("run_id") == run_id
                    and approval.get("approval_type", "required") == "required"
                ):
                    return approval
        approvals, _ = db.get_approvals(run_id=run_id, approval_type="required", limit=1)
        return approvals[0] if approvals else None
    except (NotImplementedError, Exception):
        return None


async def _aget_approval_for_run(db: Any, run_id: str, approval_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up the active approval for a run_id (async)."""
    try:
        from inspect import iscoroutinefunction

        if approval_id:
            get_approval = getattr(db, "get_approval", None)
            if get_approval is not None:
                if iscoroutinefunction(get_approval):
                    approval = await get_approval(approval_id)
                else:
                    approval = get_approval(approval_id)
                if (
                    isinstance(approval, dict)
                    and approval.get("run_id") == run_id
                    and approval.get("approval_type", "required") == "required"
                ):
                    return approval

        get_fn = getattr(db, "get_approvals", None)
        if get_fn is None:
            return None

        if iscoroutinefunction(get_fn):
            approvals, _ = await get_fn(run_id=run_id, approval_type="required", limit=1)
        else:
            approvals, _ = get_fn(run_id=run_id, approval_type="required", limit=1)
        return approvals[0] if approvals else None
    except (NotImplementedError, Exception):
        return None


def _collect_all_run_ids(run_id: str, run_response: Any) -> List[str]:
    """Gather all candidate run_ids when looking up an approval record.

    Approvals may be stored under the team's run_id (team-level tools) or a
    member agent's run_id (member-level tools where the agent created the
    approval before the team propagated the pause). This returns the team
    run_id first, followed by any member_run_id values from the requirements,
    so the lookup can try each until a match is found.
    """
    ids = [run_id]
    for req in getattr(run_response, "requirements", None) or []:
        mid = getattr(req, "member_run_id", None)
        if mid and mid not in ids:
            ids.append(mid)
    return ids


def _collect_all_approval_tools(run_response: Any) -> List[Any]:
    """Collect all tool executions that carry an approval_type from the run response.

    Searches both run_response.tools (team-level tools) and
    run_response.requirements[*].tool_execution (member-level tools propagated
    via _propagate_member_pause). Deduplicates by tool_call_id.
    """
    result: List[Any] = []
    for t in getattr(run_response, "tools", None) or []:
        if getattr(t, "approval_type", None) is not None:
            result.append(t)
    for req in getattr(run_response, "requirements", None) or []:
        te = getattr(req, "tool_execution", None)
        if te and getattr(te, "approval_type", None) is not None:
            # Avoid duplicates (same tool_call_id already in result)
            if not any(getattr(r, "tool_call_id", None) == te.tool_call_id for r in result):
                result.append(te)
    return result


def _attach_resolved_approval(run_response: Any, approval: Dict[str, Any]) -> None:
    """Expose the resolved approval record to post-hooks via run_response.metadata["approval"]."""
    if run_response.metadata is None:
        run_response.metadata = {}
    run_response.metadata["approval"] = approval


def check_and_apply_approval_resolution(db: Any, run_id: str, run_response: Any) -> None:
    """Gate: if any tool has approval_type='required', verify the approval is resolved before continuing.

    Checks both run_response.tools AND requirements' tool_execution objects so that
    member-level approvals (where run_response.tools = [delegate_task_to_member]) are found.

    Raises RuntimeError if the approval is still pending or not found.
    No-op if no tools require approval or if db is None.
    """
    if db is None:
        return

    approval_tools = _collect_all_approval_tools(run_response)
    if not any(getattr(t, "approval_type", None) == "required" for t in approval_tools):
        return

    # Approvals may be stored under the team's run_id or any member agent's run_id;
    # also try the active approval_id stamped on the tool.
    active_approval_id = _get_active_approval_id(approval_tools)
    approval = None
    for rid in _collect_all_run_ids(run_id, run_response):
        approval = _get_approval_for_run(db, rid, approval_id=active_approval_id)
        if approval is not None:
            break
    if approval is None:
        raise RuntimeError(
            "No approval record found for this run. Cannot continue a run that requires external approval."
        )

    status = approval.get("status", "pending")
    if status == "pending":
        raise RuntimeError("Approval is still pending. Resolve the approval before continuing this run.")

    resolution_tools = _approval_tools_for_resolution(approval_tools, approval.get("id") or active_approval_id)
    if not resolution_tools:
        raise RuntimeError("Resolved approval does not match the active HITL requirement for this run.")
    resolution_data = approval.get("resolution_data")
    unresolved_tools = [
        tool for tool in resolution_tools if not _has_usable_approval_resolution(tool, status, resolution_data)
    ]
    if unresolved_tools:
        raise RuntimeError("Approval resolution data is incomplete. Resolve the approval before continuing this run.")

    _apply_approval_to_tools(resolution_tools, status, resolution_data)
    _sync_requirements_from_tools(run_response)
    _attach_resolved_approval(run_response, approval)


async def acheck_and_apply_approval_resolution(db: Any, run_id: str, run_response: Any) -> None:
    """Async variant of check_and_apply_approval_resolution."""
    if db is None:
        return

    approval_tools = _collect_all_approval_tools(run_response)
    if not any(getattr(t, "approval_type", None) == "required" for t in approval_tools):
        return

    # Approvals may be stored under the team's run_id or any member agent's run_id;
    # also try the active approval_id stamped on the tool.
    active_approval_id = _get_active_approval_id(approval_tools)
    approval = None
    for rid in _collect_all_run_ids(run_id, run_response):
        approval = await _aget_approval_for_run(db, rid, approval_id=active_approval_id)
        if approval is not None:
            break
    if approval is None:
        raise RuntimeError(
            "No approval record found for this run. Cannot continue a run that requires external approval."
        )

    status = approval.get("status", "pending")
    if status == "pending":
        raise RuntimeError("Approval is still pending. Resolve the approval before continuing this run.")

    resolution_tools = _approval_tools_for_resolution(approval_tools, approval.get("id") or active_approval_id)
    if not resolution_tools:
        raise RuntimeError("Resolved approval does not match the active HITL requirement for this run.")
    resolution_data = approval.get("resolution_data")
    unresolved_tools = [
        tool for tool in resolution_tools if not _has_usable_approval_resolution(tool, status, resolution_data)
    ]
    if unresolved_tools:
        raise RuntimeError("Approval resolution data is incomplete. Resolve the approval before continuing this run.")

    _apply_approval_to_tools(resolution_tools, status, resolution_data)
    _sync_requirements_from_tools(run_response)
    _attach_resolved_approval(run_response, approval)


async def acreate_audit_approval(
    db: Any,
    tool_execution: Any,
    run_response: Any,
    status: str,  # "approved" or "rejected"
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    team_id: Optional[str] = None,
    team_name: Optional[str] = None,
    user_id: Optional[str] = None,
    pause_type: Optional[str] = None,
) -> None:
    """Async variant of create_audit_approval."""
    if db is None:
        return
    try:
        source_type = "agent"
        source_name = agent_name
        if team_id:
            source_type = "team"
            source_name = team_name

        tool_name = getattr(tool_execution, "tool_name", None)
        tool_args = getattr(tool_execution, "tool_args", None)
        pause_type = pause_type or _get_pause_type(tool_execution)

        context: Dict[str, Any] = {}
        if tool_name:
            context["tool_names"] = [tool_name]
        if source_name:
            context["source_name"] = source_name

        approval_data = {
            "id": str(uuid4()),
            "run_id": getattr(run_response, "run_id", None) or str(uuid4()),
            "session_id": getattr(run_response, "session_id", None) or "",
            "status": status,
            "approval_type": "audit",
            "pause_type": pause_type,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "source_type": source_type,
            "agent_id": agent_id,
            "team_id": team_id,
            "user_id": user_id,
            "source_name": source_name,
            "context": context if context else None,
            "resolved_at": now_epoch_s(),
            "created_at": now_epoch_s(),
            "updated_at": None,
        }
        create_fn = getattr(db, "create_approval", None)
        if create_fn is None:
            return
        from inspect import iscoroutinefunction

        if iscoroutinefunction(create_fn):
            await create_fn(approval_data)
        else:
            create_fn(approval_data)
        log_debug(f"Audit approval {approval_data['id']} for tool {tool_name}")
    except NotImplementedError:
        pass
    except Exception as e:
        log_warning(f"Error creating audit approval record (async): {str(e)}")


# ---------------------------------------------------------------------------
# Update approval run_status when run completes
# ---------------------------------------------------------------------------


def update_approval_run_status(db: Any, run_id: str, run_status: RunStatus) -> None:
    """Update run_status on all approvals for a given run_id.

    Called when a run completes, errors, or is cancelled after being paused.
    This allows the UI to know if the run has already been continued.

    Args:
        db: Database adapter instance.
        run_id: The run ID to match.
        run_status: The new run status.
    """
    if db is None:
        return

    try:
        update_fn = getattr(db, "update_approval_run_status", None)
        if update_fn is None:
            return
        count = update_fn(run_id, run_status)
        if count > 0:
            log_debug(f"Updated run_status to {run_status} for {count} approval(s) on run {run_id}")
    except NotImplementedError:
        pass
    except Exception as e:
        log_warning(f"Error updating approval run_status (sync): {str(e)}")


async def aupdate_approval_run_status(db: Any, run_id: str, run_status: RunStatus) -> None:
    """Async variant of update_approval_run_status.

    Called when a run completes, errors, or is cancelled after being paused.
    This allows the UI to know if the run has already been continued.

    Args:
        db: Database adapter instance.
        run_id: The run ID to match.
        run_status: The new run status.
    """
    if db is None:
        return

    try:
        update_fn = getattr(db, "update_approval_run_status", None)
        if update_fn is None:
            return
        from inspect import iscoroutinefunction

        if iscoroutinefunction(update_fn):
            count = await update_fn(run_id, run_status)
        else:
            count = update_fn(run_id, run_status)
        if count > 0:
            log_debug(f"Updated run_status to {run_status} for {count} approval(s) on run {run_id}")
    except NotImplementedError:
        pass
    except Exception as e:
        log_warning(f"Error updating approval run_status (async): {str(e)}")


# ---------------------------------------------------------------------------
# Resolve approval record (for interface HITL resume)
# ---------------------------------------------------------------------------


async def aresolve_approval(
    db: Any,
    approval_id: str,
    status: str,
    resolved_by: Optional[str],
    resolved_at: int,
    resolution_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve an approval record by stamping status and resolution fields.

    Called by interface HITL handlers (Slack, etc.) when a user approves or
    rejects a paused tool. Completes the audit trail for the approval record.

    Args:
        db: Database adapter instance.
        approval_id: The approval record ID.
        status: New status ("approved" or "rejected").
        resolved_by: User ID of the resolver.
        resolved_at: Unix timestamp of resolution.
        resolution_data: Optional dict with note/values/result/feedback.
    """
    if db is None or not hasattr(db, "update_approval"):
        return None

    from agno.db.base import AsyncBaseDb

    kwargs: Dict[str, Any] = {
        "status": status,
        "resolved_by": resolved_by,
        "resolved_at": resolved_at,
    }
    if resolution_data:
        kwargs["resolution_data"] = resolution_data

    try:
        if isinstance(db, AsyncBaseDb):
            result = await db.update_approval(approval_id, expected_status="pending", **kwargs)
        else:
            result = db.update_approval(approval_id, expected_status="pending", **kwargs)
        if result is None:
            log_debug(f"Approval {approval_id} already resolved or missing")
        return result
    except NotImplementedError:
        return None
    except Exception as e:
        log_warning(f"Error resolving approval {approval_id}: {str(e)}")
        return None
