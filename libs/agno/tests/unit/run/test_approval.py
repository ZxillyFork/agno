"""Unit tests for agno.run.approval — approval record creation and resolution gating."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agno.models.response import ToolExecution
from agno.run.approval import (
    _apply_approval_to_tools,
    _build_approval_dict,
    _get_first_approval_tool,
    _get_pause_type,
    _has_approval_requirement,
    acheck_and_apply_approval_resolution,
    acreate_approval_from_pause,
    acreate_audit_approval,
    check_and_apply_approval_resolution,
    create_approval_from_pause,
    create_audit_approval,
)
from agno.run.requirement import RunRequirement

# =============================================================================
# Helpers: lightweight stand-ins for ToolExecution / RunResponse / UserInputField
# =============================================================================


@dataclass
class FakeToolExecution:
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    approval_type: Optional[str] = None
    approval_id: Optional[str] = None
    requires_confirmation: Optional[bool] = None
    requires_user_input: Optional[bool] = None
    external_execution_required: Optional[bool] = None
    user_input_schema: Optional[list] = None
    user_feedback_schema: Optional[list] = None
    confirmed: Optional[bool] = None
    answered: Optional[bool] = None
    result: Optional[str] = None
    external_execution_result_provided: Optional[bool] = None
    resume_metadata: Optional[Dict[str, Any]] = None
    confirmation_note: Optional[str] = None


@dataclass
class FakeRequirement:
    tool_execution: Optional[FakeToolExecution] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool_execution": self.tool_execution.tool_name if self.tool_execution else None}


@dataclass
class FakeRunResponse:
    run_id: Optional[str] = "run-123"
    session_id: Optional[str] = "sess-456"
    tools: Optional[list] = None
    requirements: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FakeUserInputField:
    name: str = ""
    value: Optional[str] = None


@dataclass
class FakeFeedbackOption:
    label: str = ""
    selected: bool = False


@dataclass
class FakeFeedbackQuestion:
    question: str = ""
    selected_options: Optional[list] = None
    options: Optional[list] = None


# =============================================================================
# _get_pause_type
# =============================================================================


class TestGetPauseType:
    def test_user_input(self):
        te = FakeToolExecution(requires_user_input=True)
        assert _get_pause_type(te) == "user_input"

    def test_external_execution(self):
        te = FakeToolExecution(external_execution_required=True)
        assert _get_pause_type(te) == "external_execution"

    def test_confirmation_default(self):
        te = FakeToolExecution()
        assert _get_pause_type(te) == "confirmation"

    def test_user_input_takes_precedence(self):
        """user_input is checked before external_execution."""
        te = FakeToolExecution(requires_user_input=True, external_execution_required=True)
        assert _get_pause_type(te) == "user_input"


# =============================================================================
# _get_first_approval_tool
# =============================================================================


class TestGetFirstApprovalTool:
    def test_returns_none_when_empty(self):
        assert _get_first_approval_tool(None) is None
        assert _get_first_approval_tool([]) is None

    def test_finds_tool_in_tools_list(self):
        t1 = FakeToolExecution(tool_name="t1", approval_type=None)
        t2 = FakeToolExecution(tool_name="t2", approval_type="required")
        assert _get_first_approval_tool([t1, t2]) is t2

    def test_finds_tool_in_requirements(self):
        te = FakeToolExecution(tool_name="req_tool", approval_type="audit")
        req = FakeRequirement(tool_execution=te)
        assert _get_first_approval_tool(None, requirements=[req]) is te

    def test_tools_list_takes_precedence(self):
        t_in_tools = FakeToolExecution(tool_name="from_tools", approval_type="required")
        t_in_reqs = FakeToolExecution(tool_name="from_reqs", approval_type="required")
        req = FakeRequirement(tool_execution=t_in_reqs)
        result = _get_first_approval_tool([t_in_tools], requirements=[req])
        assert result is t_in_tools


# =============================================================================
# _has_approval_requirement
# =============================================================================


class TestHasApprovalRequirement:
    def test_false_when_no_tools(self):
        assert _has_approval_requirement(None) is False

    def test_false_when_approval_type_is_audit(self):
        t = FakeToolExecution(approval_type="audit")
        assert _has_approval_requirement([t]) is False

    def test_true_when_approval_type_is_required(self):
        t = FakeToolExecution(approval_type="required")
        assert _has_approval_requirement([t]) is True

    def test_true_via_requirements(self):
        te = FakeToolExecution(approval_type="required")
        req = FakeRequirement(tool_execution=te)
        assert _has_approval_requirement(None, requirements=[req]) is True


# =============================================================================
# _build_approval_dict
# =============================================================================


class TestBuildApprovalDict:
    def test_basic_agent_source(self):
        rr = FakeRunResponse(
            tools=[FakeToolExecution(tool_name="delete_file", approval_type="required", requires_confirmation=True)]
        )
        result = _build_approval_dict(rr, agent_id="a1", agent_name="MyAgent")
        assert result["source_type"] == "agent"
        assert result["source_name"] == "MyAgent"
        assert result["agent_id"] == "a1"
        assert result["tool_name"] == "delete_file"
        assert result["approval_type"] == "required"
        assert result["status"] == "pending"
        assert result["run_id"] == "run-123"
        assert result["session_id"] == "sess-456"
        assert isinstance(result["id"], str)
        assert isinstance(result["created_at"], int)

    def test_team_source_overrides_agent(self):
        rr = FakeRunResponse(tools=[FakeToolExecution(tool_name="t", approval_type="required")])
        result = _build_approval_dict(rr, agent_id="a1", agent_name="A", team_id="t1", team_name="MyTeam")
        assert result["source_type"] == "team"
        assert result["source_name"] == "MyTeam"

    def test_workflow_source(self):
        rr = FakeRunResponse(tools=[FakeToolExecution(tool_name="t", approval_type="required")])
        result = _build_approval_dict(rr, workflow_id="w1", workflow_name="MyWorkflow")
        assert result["source_type"] == "workflow"
        assert result["source_name"] == "MyWorkflow"

    def test_session_id_falls_back_to_empty_string(self):
        rr = FakeRunResponse(session_id=None, tools=[FakeToolExecution(approval_type="required")])
        result = _build_approval_dict(rr)
        assert result["session_id"] == ""

    def test_run_id_falls_back_to_uuid(self):
        rr = FakeRunResponse(run_id=None, tools=[FakeToolExecution(approval_type="required")])
        result = _build_approval_dict(rr)
        assert isinstance(result["run_id"], str)
        assert len(result["run_id"]) > 0

    def test_context_includes_tool_names_from_requirements(self):
        te1 = FakeToolExecution(tool_name="tool_a", approval_type="required")
        te2 = FakeToolExecution(tool_name="tool_b", approval_type="required")
        rr = FakeRunResponse(requirements=[FakeRequirement(tool_execution=te1), FakeRequirement(tool_execution=te2)])
        result = _build_approval_dict(rr)
        assert result["context"]["tool_names"] == ["tool_a", "tool_b"]

    def test_context_falls_back_to_tools_list(self):
        t1 = FakeToolExecution(tool_name="my_tool", approval_type="required")
        rr = FakeRunResponse(tools=[t1])
        result = _build_approval_dict(rr)
        assert result["context"]["tool_names"] == ["my_tool"]

    def test_pause_type_from_user_input_tool(self):
        t = FakeToolExecution(tool_name="ask", approval_type="required", requires_user_input=True)
        rr = FakeRunResponse(tools=[t])
        result = _build_approval_dict(rr)
        assert result["pause_type"] == "user_input"

    def test_pause_type_from_external_execution_tool(self):
        t = FakeToolExecution(tool_name="ext", approval_type="required", external_execution_required=True)
        rr = FakeRunResponse(tools=[t])
        result = _build_approval_dict(rr)
        assert result["pause_type"] == "external_execution"

    def test_schedule_fields_passed_through(self):
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        result = _build_approval_dict(rr, schedule_id="sched-1", schedule_run_id="sr-1")
        assert result["schedule_id"] == "sched-1"
        assert result["schedule_run_id"] == "sr-1"


# =============================================================================
# create_approval_from_pause (sync)
# =============================================================================


class TestCreateApprovalFromPause:
    def test_noop_when_db_is_none(self):
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        create_approval_from_pause(db=None, run_response=rr)  # should not raise

    def test_noop_when_no_approval_requirement(self):
        db = MagicMock()
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type=None)])
        create_approval_from_pause(db=db, run_response=rr)
        db.create_approval.assert_not_called()

    def test_creates_approval_record(self):
        db = MagicMock()
        rr = FakeRunResponse(tools=[FakeToolExecution(tool_name="delete", approval_type="required")])
        create_approval_from_pause(db=db, run_response=rr, agent_id="a1", agent_name="Agent")
        db.create_approval.assert_called_once()
        data = db.create_approval.call_args[0][0]
        assert data["status"] == "pending"
        assert data["agent_id"] == "a1"

    def test_silently_handles_not_implemented(self):
        db = MagicMock()
        db.create_approval.side_effect = NotImplementedError
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        create_approval_from_pause(db=db, run_response=rr)  # should not raise

    def test_silently_handles_generic_exception(self):
        db = MagicMock()
        db.create_approval.side_effect = RuntimeError("db down")
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        create_approval_from_pause(db=db, run_response=rr)  # should not raise

    def test_passes_user_id(self):
        db = MagicMock()
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        create_approval_from_pause(db=db, run_response=rr, user_id="user-1")
        data = db.create_approval.call_args[0][0]
        assert data["user_id"] == "user-1"

    def test_passes_team_context(self):
        db = MagicMock()
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        create_approval_from_pause(db=db, run_response=rr, team_id="t1", team_name="Team", user_id="u1")
        data = db.create_approval.call_args[0][0]
        assert data["team_id"] == "t1"
        assert data["source_type"] == "team"
        assert data["source_name"] == "Team"
        assert data["user_id"] == "u1"

    def test_returns_approval_id_on_success(self):
        db = MagicMock()
        tool = FakeToolExecution(tool_name="delete", approval_type="required")
        rr = FakeRunResponse(tools=[tool])
        result = create_approval_from_pause(db=db, run_response=rr, agent_id="a1", agent_name="Agent")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # The returned ID must match what was passed to db.create_approval
        data = db.create_approval.call_args[0][0]
        assert result == data["id"]
        # approval_id must also be stamped on the tool itself
        assert tool.approval_id == result

    def test_reuses_existing_approval_id(self):
        db = MagicMock()
        db.get_approval.return_value = {
            "id": "approval-1",
            "run_id": "run-123",
            "approval_type": "required",
            "status": "pending",
        }
        tool = FakeToolExecution(
            tool_name="delete",
            approval_type="required",
            approval_id="approval-1",
            requires_confirmation=True,
        )
        rr = FakeRunResponse(tools=[tool])

        result = create_approval_from_pause(db=db, run_response=rr)

        assert result == "approval-1"
        db.create_approval.assert_not_called()

    def test_copied_approval_id_from_other_run_does_not_suppress_new_approval(self):
        db = MagicMock()
        db.get_approval.return_value = {
            "id": "member-approval",
            "run_id": "member-run",
            "approval_type": "required",
            "status": "pending",
        }
        tool = FakeToolExecution(
            tool_name="delete",
            approval_type="required",
            approval_id="member-approval",
            requires_confirmation=True,
        )
        rr = FakeRunResponse(run_id="team-run", tools=[tool])

        result = create_approval_from_pause(db=db, run_response=rr)

        assert result is not None
        assert result != "member-approval"
        assert tool.approval_id == result
        db.create_approval.assert_called_once()

    def test_old_resolved_approval_id_does_not_suppress_new_pause(self):
        db = MagicMock()
        old_tool = FakeToolExecution(
            tool_name="old_tool",
            approval_type="required",
            approval_id="old-approval",
            requires_confirmation=False,
        )
        new_tool = FakeToolExecution(tool_name="new_tool", approval_type="required", requires_confirmation=True)
        rr = FakeRunResponse(tools=[old_tool, new_tool])

        result = create_approval_from_pause(db=db, run_response=rr)

        assert result != "old-approval"
        assert new_tool.approval_id == result
        db.create_approval.assert_called_once()


# =============================================================================
# acreate_approval_from_pause (async)
# =============================================================================


class TestAsyncCreateApprovalFromPause:
    @pytest.mark.asyncio
    async def test_noop_when_db_is_none(self):
        await acreate_approval_from_pause(db=None, run_response=FakeRunResponse())

    @pytest.mark.asyncio
    async def test_calls_async_create_approval(self):
        db = MagicMock()
        db.create_approval = AsyncMock()
        rr = FakeRunResponse(tools=[FakeToolExecution(tool_name="t", approval_type="required")])
        await acreate_approval_from_pause(db=db, run_response=rr)
        db.create_approval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_create_approval(self):
        db = MagicMock()
        db.create_approval = MagicMock()  # sync
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        await acreate_approval_from_pause(db=db, run_response=rr)
        db.create_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_create_approval_missing(self):
        db = MagicMock(spec=[])  # no create_approval attribute
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        await acreate_approval_from_pause(db=db, run_response=rr)  # should not raise

    @pytest.mark.asyncio
    async def test_returns_approval_id_on_success(self):
        db = MagicMock()
        db.create_approval = AsyncMock()
        tool = FakeToolExecution(tool_name="delete", approval_type="required")
        rr = FakeRunResponse(tools=[tool])
        result = await acreate_approval_from_pause(db=db, run_response=rr, agent_id="a1", agent_name="Agent")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        data = db.create_approval.call_args[0][0]
        assert result == data["id"]
        # approval_id must also be stamped on the tool itself
        assert tool.approval_id == result

    @pytest.mark.asyncio
    async def test_copied_approval_id_from_other_run_does_not_suppress_new_approval_async(self):
        db = MagicMock()
        db.create_approval = AsyncMock()
        db.get_approval = AsyncMock(
            return_value={
                "id": "member-approval",
                "run_id": "member-run",
                "approval_type": "required",
                "status": "pending",
            }
        )
        tool = FakeToolExecution(
            tool_name="delete",
            approval_type="required",
            approval_id="member-approval",
            requires_confirmation=True,
        )
        rr = FakeRunResponse(run_id="team-run", tools=[tool])

        result = await acreate_approval_from_pause(db=db, run_response=rr)

        assert result is not None
        assert result != "member-approval"
        assert tool.approval_id == result
        db.create_approval.assert_awaited_once()


# =============================================================================
# create_audit_approval (sync)
# =============================================================================


class TestCreateAuditApproval:
    def test_noop_when_db_is_none(self):
        te = FakeToolExecution(tool_name="t")
        rr = FakeRunResponse()
        create_audit_approval(db=None, tool_execution=te, run_response=rr, status="approved")

    def test_creates_audit_record(self):
        db = MagicMock()
        te = FakeToolExecution(tool_name="send_email", tool_args={"to": "a@b.com"}, requires_confirmation=True)
        rr = FakeRunResponse()
        create_audit_approval(
            db=db, tool_execution=te, run_response=rr, status="approved", agent_id="a1", agent_name="Bot"
        )
        db.create_approval.assert_called_once()
        data = db.create_approval.call_args[0][0]
        assert data["approval_type"] == "audit"
        assert data["status"] == "approved"
        assert data["tool_name"] == "send_email"
        assert data["source_type"] == "agent"
        assert data["source_name"] == "Bot"

    def test_team_source_name_set(self):
        """Verify the fix: source_name is set to team_name when team_id is present."""
        db = MagicMock()
        te = FakeToolExecution(tool_name="t")
        rr = FakeRunResponse()
        create_audit_approval(
            db=db, tool_execution=te, run_response=rr, status="rejected", team_id="t1", team_name="TheTeam"
        )
        data = db.create_approval.call_args[0][0]
        assert data["source_type"] == "team"
        assert data["source_name"] == "TheTeam"

    def test_rejected_status(self):
        db = MagicMock()
        te = FakeToolExecution(tool_name="t")
        rr = FakeRunResponse()
        create_audit_approval(db=db, tool_execution=te, run_response=rr, status="rejected")
        data = db.create_approval.call_args[0][0]
        assert data["status"] == "rejected"

    def test_explicit_pause_type_is_preserved_after_tool_flags_are_cleared(self):
        db = MagicMock()
        te = FakeToolExecution(tool_name="external", external_execution_required=False)
        rr = FakeRunResponse()
        create_audit_approval(
            db=db,
            tool_execution=te,
            run_response=rr,
            status="approved",
            pause_type="external_execution",
        )
        data = db.create_approval.call_args[0][0]
        assert data["pause_type"] == "external_execution"

    def test_silently_handles_not_implemented(self):
        db = MagicMock()
        db.create_approval.side_effect = NotImplementedError
        te = FakeToolExecution(tool_name="t")
        rr = FakeRunResponse()
        create_audit_approval(db=db, tool_execution=te, run_response=rr, status="approved")


# =============================================================================
# acreate_audit_approval (async)
# =============================================================================


class TestAsyncCreateAuditApproval:
    @pytest.mark.asyncio
    async def test_creates_audit_record_async(self):
        db = MagicMock()
        db.create_approval = AsyncMock()
        te = FakeToolExecution(tool_name="send_email")
        rr = FakeRunResponse()
        await acreate_audit_approval(
            db=db, tool_execution=te, run_response=rr, status="approved", agent_id="a1", agent_name="Bot"
        )
        db.create_approval.assert_awaited_once()
        data = db.create_approval.call_args[0][0]
        assert data["approval_type"] == "audit"
        assert data["status"] == "approved"

    @pytest.mark.asyncio
    async def test_team_source_name_set(self):
        """Verify the fix: source_name is set to team_name when team_id is present."""
        db = MagicMock()
        db.create_approval = AsyncMock()
        te = FakeToolExecution(tool_name="t")
        rr = FakeRunResponse()
        await acreate_audit_approval(
            db=db, tool_execution=te, run_response=rr, status="approved", team_id="t1", team_name="TheTeam"
        )
        data = db.create_approval.call_args[0][0]
        assert data["source_type"] == "team"
        assert data["source_name"] == "TheTeam"

    @pytest.mark.asyncio
    async def test_falls_back_to_sync(self):
        db = MagicMock()
        db.create_approval = MagicMock()  # sync
        te = FakeToolExecution(tool_name="t")
        rr = FakeRunResponse()
        await acreate_audit_approval(db=db, tool_execution=te, run_response=rr, status="approved")
        db.create_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_pause_type_is_preserved_after_tool_flags_are_cleared_async(self):
        db = MagicMock()
        db.create_approval = AsyncMock()
        te = FakeToolExecution(tool_name="input", requires_user_input=False)
        rr = FakeRunResponse()
        await acreate_audit_approval(
            db=db,
            tool_execution=te,
            run_response=rr,
            status="approved",
            pause_type="user_input",
        )
        data = db.create_approval.call_args[0][0]
        assert data["pause_type"] == "user_input"


# =============================================================================
# _apply_approval_to_tools
# =============================================================================


class TestApplyApprovalToTools:
    def test_approved_sets_confirmed_true(self):
        t = FakeToolExecution(approval_type="required", requires_confirmation=True)
        _apply_approval_to_tools([t], "approved", {"metadata": {"approver": "admin"}})
        assert t.confirmed is True
        assert t.resume_metadata == {"approver": "admin"}

    def test_rejected_sets_confirmed_false(self):
        t = FakeToolExecution(approval_type="required", requires_confirmation=True)
        _apply_approval_to_tools([t], "rejected", None)
        assert t.confirmed is False

    def test_skips_tools_without_approval_type_required(self):
        t = FakeToolExecution(approval_type="audit", requires_confirmation=True)
        _apply_approval_to_tools([t], "approved", None)
        assert t.confirmed is None  # untouched

    def test_approved_applies_user_input_values(self):
        ufield = FakeUserInputField(name="reason")
        t = FakeToolExecution(
            approval_type="required",
            requires_user_input=True,
            user_input_schema=[ufield],
        )
        _apply_approval_to_tools([t], "approved", {"values": {"reason": "looks good"}})
        assert ufield.value == "looks good"
        assert t.answered is True

    def test_approved_applies_user_feedback_selections(self):
        yes = FakeFeedbackOption(label="Yes")
        no = FakeFeedbackOption(label="No")
        question = FakeFeedbackQuestion(question="Deploy?", options=[yes, no])
        t = FakeToolExecution(
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[question],
        )

        _apply_approval_to_tools([t], "approved", {"selections": {"Deploy?": ["Yes"]}})

        assert question.selected_options == ["Yes"]
        assert yes.selected is True
        assert no.selected is False
        assert t.answered is True

    def test_approved_applies_user_feedback_key(self):
        yes = FakeFeedbackOption(label="Yes")
        question = FakeFeedbackQuestion(question="Deploy?", options=[yes])
        t = FakeToolExecution(
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[question],
        )

        _apply_approval_to_tools([t], "approved", {"feedback": {"Deploy?": ["Yes"]}})

        assert question.selected_options == ["Yes"]
        assert yes.selected is True
        assert t.answered is True

    def test_direct_user_feedback_rejects_non_list_selection(self):
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="ask_feedback",
                requires_user_input=True,
                user_feedback_schema=[FakeFeedbackQuestion(question="Deploy?")],
            )
        )

        with pytest.raises(ValueError, match="lists of option labels"):
            requirement.provide_user_feedback({"Deploy?": "Yes"})  # type: ignore[arg-type]

    def test_approved_legacy_confirmation_tool_sets_confirmed_true(self):
        t = FakeToolExecution(approval_type="required")
        _apply_approval_to_tools([t], "approved", None)
        assert t.requires_confirmation is True
        assert t.confirmed is True

    def test_rejected_legacy_confirmation_tool_sets_confirmed_false(self):
        t = FakeToolExecution(approval_type="required")
        _apply_approval_to_tools([t], "rejected", {"reason": "no"})
        assert t.requires_confirmation is True
        assert t.confirmed is False
        assert t.confirmation_note == "no"

    def test_approved_applies_external_execution_result(self):
        t = FakeToolExecution(approval_type="required", external_execution_required=True)
        _apply_approval_to_tools([t], "approved", {"result": "done"})
        assert t.result == "done"

    def test_rejected_user_input_sets_confirmed_false(self):
        t = FakeToolExecution(approval_type="required", requires_user_input=True)
        _apply_approval_to_tools([t], "rejected", {"reason": "not needed"})
        assert t.confirmed is False
        assert t.answered is True
        assert t.confirmation_note == "not needed"

    def test_rejected_external_execution_sets_confirmed_false(self):
        t = FakeToolExecution(approval_type="required", external_execution_required=True)
        _apply_approval_to_tools([t], "rejected", {"reason": "unsafe"})
        assert t.confirmed is False
        assert t.external_execution_result_provided is True
        assert t.result == "unsafe"


# =============================================================================
# check_and_apply_approval_resolution (sync)
# =============================================================================


class TestCheckAndApplyApprovalResolution:
    def test_noop_when_db_is_none(self):
        rr = FakeRunResponse()
        check_and_apply_approval_resolution(db=None, run_id="r1", run_response=rr)

    def test_noop_when_no_tools_require_approval(self):
        db = MagicMock()
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type=None)])
        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        db.get_approvals.assert_not_called()

    def test_raises_when_no_approval_record_found(self):
        db = MagicMock()
        db.get_approvals.return_value = ([], 0)
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        with pytest.raises(RuntimeError, match="No approval record found"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    def test_raises_when_approval_still_pending(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "pending"}], 1)
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        with pytest.raises(RuntimeError, match="still pending"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    def test_applies_approved_status(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": None}], 1)
        t = FakeToolExecution(approval_type="required", requires_confirmation=True)
        rr = FakeRunResponse(tools=[t])
        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        assert t.confirmed is True

    def test_applies_approved_status_to_requirements(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": {"result": None}}], 1)
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="dynamic_tool",
            approval_type="required",
            external_execution_required=True,
        )
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="dynamic_tool",
                approval_type="required",
                external_execution_required=True,
            )
        )
        rr = FakeRunResponse(tools=[tool], requirements=[requirement])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert rr.requirements[0].tool_execution is tool
        assert rr.requirements[0].external_execution_result is None
        assert rr.requirements[0].external_execution_result_provided is True
        assert rr.requirements[0].is_resolved()

    def test_applies_approved_status_to_requirements_without_top_level_tools(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": None}], 1)
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="dynamic_tool",
                approval_type="required",
                requires_confirmation=True,
            )
        )
        rr = FakeRunResponse(tools=None, requirements=[requirement])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert rr.requirements[0].confirmation is True
        assert rr.requirements[0].tool_execution.confirmed is True
        assert rr.requirements[0].is_resolved()
        assert rr.tools == [rr.requirements[0].tool_execution]

    def test_approved_user_input_without_values_stays_blocked(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": None}], 1)
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_reason",
            approval_type="required",
            requires_user_input=True,
            user_input_schema=[FakeUserInputField(name="reason")],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    def test_malformed_user_input_values_stays_blocked_without_crashing(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": {"values": ["reason"]}}], 1)
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_reason",
            approval_type="required",
            requires_user_input=True,
            user_input_schema=[FakeUserInputField(name="reason")],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    def test_null_user_input_value_stays_blocked(self):
        db = MagicMock()
        db.get_approvals.return_value = (
            [{"status": "approved", "resolution_data": {"values": {"reason": None}}}],
            1,
        )
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_reason",
            approval_type="required",
            requires_user_input=True,
            user_input_schema=[FakeUserInputField(name="reason")],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert tool.user_input_schema[0].value is None
        assert tool.answered is None

    def test_malformed_user_feedback_selections_stays_blocked_without_crashing(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": {"selections": ["Yes"]}}], 1)
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_feedback",
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[FakeFeedbackQuestion(question="Deploy?")],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    def test_string_user_feedback_selection_stays_blocked(self):
        db = MagicMock()
        db.get_approvals.return_value = (
            [{"status": "approved", "resolution_data": {"selections": {"Deploy?": "Yes"}}}],
            1,
        )
        yes = FakeFeedbackOption(label="Yes")
        prefix = FakeFeedbackOption(label="Ye")
        question = FakeFeedbackQuestion(question="Deploy?", options=[prefix, yes])
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_feedback",
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[question],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert question.selected_options is None
        assert prefix.selected is False
        assert yes.selected is False

    def test_null_user_feedback_selection_stays_blocked(self):
        db = MagicMock()
        db.get_approvals.return_value = (
            [{"status": "approved", "resolution_data": {"selections": {"Deploy?": None}}}],
            1,
        )
        question = FakeFeedbackQuestion(question="Deploy?", options=[FakeFeedbackOption(label="Yes")])
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_feedback",
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[question],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert question.selected_options is None
        assert question.options[0].selected is False

    def test_approved_user_feedback_requirement_is_resolved(self):
        db = MagicMock()
        db.get_approvals.return_value = (
            [{"status": "approved", "resolution_data": {"selections": {"Deploy?": ["Yes"]}}}],
            1,
        )
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_feedback",
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[FakeFeedbackQuestion(question="Deploy?")],
        )
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="ask_feedback",
                approval_type="required",
                requires_user_input=True,
                user_feedback_schema=[FakeFeedbackQuestion(question="Deploy?")],
            )
        )
        rr = FakeRunResponse(tools=[tool], requirements=[requirement])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert rr.requirements[0].is_resolved()
        assert rr.requirements[0].tool_execution.answered is True
        assert rr.requirements[0].user_feedback_schema[0].selected_options == ["Yes"]

    def test_approved_legacy_approval_type_only_tool_is_resolved(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": None}], 1)
        t = FakeToolExecution(approval_type="required")
        rr = FakeRunResponse(tools=[t])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert t.requires_confirmation is True
        assert t.confirmed is True

    def test_approved_external_execution_without_result_stays_blocked(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": {}}], 1)
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="external_tool",
            approval_type="required",
            external_execution_required=True,
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    def test_applies_resume_metadata_to_requirements(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "approved", "resolution_data": {"metadata": {"m": 1}}}], 1)
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="dynamic_tool",
            approval_type="required",
            requires_confirmation=True,
        )
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="dynamic_tool",
                approval_type="required",
                requires_confirmation=True,
            )
        )
        rr = FakeRunResponse(tools=[tool], requirements=[requirement])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert rr.requirements[0].approval_metadata == {"m": 1}
        assert rr.requirements[0].tool_execution.resume_metadata == {"m": 1}

    def test_approved_user_input_requirement_is_resolved_with_metadata(self):
        db = MagicMock()
        db.get_approvals.return_value = (
            [{"status": "approved", "resolution_data": {"values": {"reason": "ok"}, "metadata": {"m": 1}}}],
            1,
        )
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_reason",
            approval_type="required",
            requires_user_input=True,
            user_input_schema=[FakeUserInputField(name="reason")],
        )
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="ask_reason",
                approval_type="required",
                requires_user_input=True,
                user_input_schema=[FakeUserInputField(name="reason")],
            )
        )
        rr = FakeRunResponse(tools=[tool], requirements=[requirement])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert rr.requirements[0].is_resolved()
        assert rr.requirements[0].approval_metadata == {"m": 1}
        assert rr.requirements[0].tool_execution.answered is True

    def test_resolution_prefers_active_tool_approval_id(self):
        db = MagicMock()
        db.get_approval.return_value = {
            "id": "approval-active",
            "run_id": "r1",
            "approval_type": "required",
            "status": "approved",
            "resolution_data": None,
        }
        db.get_approvals.return_value = ([{"id": "approval-old", "status": "rejected", "resolution_data": None}], 1)
        tool = FakeToolExecution(
            approval_type="required",
            approval_id="approval-active",
            requires_confirmation=True,
        )
        rr = FakeRunResponse(tools=[tool])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        db.get_approval.assert_called_once_with("approval-active")
        assert tool.confirmed is True

    def test_active_approval_id_from_another_run_is_ignored(self):
        db = MagicMock()
        db.get_approval.return_value = {
            "id": "approval-active",
            "run_id": "other-run",
            "approval_type": "required",
            "status": "approved",
            "resolution_data": None,
        }
        db.get_approvals.return_value = ([{"id": "approval-current", "status": "pending"}], 1)
        tool = FakeToolExecution(
            approval_type="required",
            approval_id="approval-active",
            requires_confirmation=True,
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="still pending"):
            check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert tool.confirmed is None

    def test_new_approval_id_is_only_stamped_on_active_pause(self):
        db = MagicMock()
        old_tool = FakeToolExecution(
            approval_type="required",
            approval_id="approval-old",
            requires_confirmation=False,
        )
        new_tool = FakeToolExecution(
            approval_type="required",
            requires_confirmation=True,
        )
        rr = FakeRunResponse(tools=[old_tool, new_tool])

        approval_id = create_approval_from_pause(db=db, run_response=rr)

        assert old_tool.approval_id == "approval-old"
        assert new_tool.approval_id == approval_id

    def test_resolution_sync_keeps_repeated_tool_call_requirements_separate(self):
        db = MagicMock()
        db.get_approval.return_value = {
            "id": "approval-new",
            "run_id": "r1",
            "approval_type": "required",
            "status": "approved",
            "resolution_data": {"metadata": {"round": 2}},
        }
        old_tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="protected",
            approval_type="required",
            approval_id="approval-old",
            requires_confirmation=True,
            confirmed=True,
            resume_metadata={"round": 1},
        )
        old_requirement = RunRequirement(old_tool)
        old_requirement.confirmation = True
        old_requirement.approval_metadata = {"round": 1}
        new_tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="protected",
            approval_type="required",
            approval_id="approval-new",
            requires_confirmation=True,
        )
        new_requirement = RunRequirement(new_tool)
        rr = FakeRunResponse(tools=[old_tool, new_tool], requirements=[old_requirement, new_requirement])

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert old_requirement.tool_execution is old_tool
        assert old_requirement.approval_metadata == {"round": 1}
        assert new_requirement.tool_execution is new_tool
        assert new_requirement.is_resolved()
        assert new_requirement.approval_metadata == {"round": 2}

    def test_specific_approval_id_does_not_resolve_unbound_active_requirement(self):
        db = MagicMock()
        db.get_approval.return_value = {
            "id": "approval-a",
            "run_id": "r1",
            "approval_type": "required",
            "status": "approved",
            "resolution_data": {"metadata": {"approval": "a"}},
        }
        approved_tool = ToolExecution(
            tool_call_id="call-a",
            tool_name="approved_tool",
            approval_type="required",
            approval_id="approval-a",
            requires_confirmation=True,
        )
        unbound_tool = ToolExecution(
            tool_call_id="call-b",
            tool_name="still_pending",
            approval_type="required",
            approval_id=None,
            requires_confirmation=True,
        )
        approved_requirement = RunRequirement(approved_tool)
        unbound_requirement = RunRequirement(unbound_tool)
        rr = FakeRunResponse(
            tools=[approved_tool, unbound_tool], requirements=[approved_requirement, unbound_requirement]
        )

        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert approved_requirement.is_resolved()
        assert approved_requirement.approval_metadata == {"approval": "a"}
        assert unbound_requirement.is_resolved() is False
        assert unbound_tool.confirmed is None

    def test_applies_rejected_status(self):
        db = MagicMock()
        db.get_approvals.return_value = ([{"status": "rejected", "resolution_data": None}], 1)
        t = FakeToolExecution(approval_type="required", requires_confirmation=True)
        rr = FakeRunResponse(tools=[t])
        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        assert t.confirmed is False

    def test_attaches_resolved_approval_to_metadata(self):
        approval = {"status": "approved", "resolution_data": None, "resolved_by": "alice", "resolved_at": 1700000000}
        db = MagicMock()
        db.get_approvals.return_value = ([approval], 1)
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required", requires_confirmation=True)])
        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        assert rr.metadata is not None
        assert rr.metadata["approval"] == approval

    def test_preserves_existing_metadata(self):
        approval = {"status": "approved", "resolution_data": None}
        db = MagicMock()
        db.get_approvals.return_value = ([approval], 1)
        rr = FakeRunResponse(
            tools=[FakeToolExecution(approval_type="required", requires_confirmation=True)],
            metadata={"existing": "value"},
        )
        check_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        assert rr.metadata["existing"] == "value"
        assert rr.metadata["approval"] == approval


# =============================================================================
# acheck_and_apply_approval_resolution (async)
# =============================================================================


class TestAsyncCheckAndApplyApprovalResolution:
    @pytest.mark.asyncio
    async def test_noop_when_db_is_none(self):
        rr = FakeRunResponse()
        await acheck_and_apply_approval_resolution(db=None, run_id="r1", run_response=rr)

    @pytest.mark.asyncio
    async def test_raises_when_no_approval_record_found(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(return_value=([], 0))
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        with pytest.raises(RuntimeError, match="No approval record found"):
            await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    @pytest.mark.asyncio
    async def test_raises_when_approval_still_pending(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(return_value=([{"status": "pending"}], 1))
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required")])
        with pytest.raises(RuntimeError, match="still pending"):
            await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    @pytest.mark.asyncio
    async def test_applies_approved_status_async(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(return_value=([{"status": "approved", "resolution_data": None}], 1))
        t = FakeToolExecution(approval_type="required", requires_confirmation=True)
        rr = FakeRunResponse(tools=[t])
        await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        assert t.confirmed is True

    @pytest.mark.asyncio
    async def test_applies_approved_status_to_requirements_async(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(return_value=([{"status": "approved", "resolution_data": None}], 1))
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="dynamic_tool",
            approval_type="required",
            requires_confirmation=True,
        )
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="dynamic_tool",
                approval_type="required",
                requires_confirmation=True,
            )
        )
        rr = FakeRunResponse(tools=[tool], requirements=[requirement])

        await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert rr.requirements[0].tool_execution is tool
        assert rr.requirements[0].confirmation is True
        assert rr.requirements[0].is_resolved()
        assert rr.tools == [tool]

    @pytest.mark.asyncio
    async def test_approved_external_execution_without_result_stays_blocked_async(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(return_value=([{"status": "approved", "resolution_data": None}], 1))
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="external_tool",
            approval_type="required",
            external_execution_required=True,
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

    @pytest.mark.asyncio
    async def test_active_approval_id_from_another_run_is_ignored_async(self):
        db = MagicMock()
        db.get_approval = AsyncMock(
            return_value={
                "id": "approval-active",
                "run_id": "other-run",
                "approval_type": "required",
                "status": "approved",
                "resolution_data": None,
            }
        )
        db.get_approvals = AsyncMock(return_value=([{"id": "approval-current", "status": "pending"}], 1))
        tool = FakeToolExecution(
            approval_type="required",
            approval_id="approval-active",
            requires_confirmation=True,
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="still pending"):
            await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert tool.confirmed is None

    @pytest.mark.asyncio
    async def test_rejected_user_feedback_sets_answered_async(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(return_value=([{"status": "rejected", "resolution_data": {"reason": "no"}}], 1))
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_feedback",
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[FakeFeedbackQuestion(question="Deploy?")],
        )
        rr = FakeRunResponse(tools=[tool])

        await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert tool.confirmed is False
        assert tool.answered is True
        assert tool.confirmation_note == "no"

    @pytest.mark.asyncio
    async def test_approved_user_feedback_requirement_is_resolved_async(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(
            return_value=(
                [{"status": "approved", "resolution_data": {"selections": {"Deploy?": ["Yes"]}}}],
                1,
            )
        )
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_feedback",
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[FakeFeedbackQuestion(question="Deploy?")],
        )
        requirement = RunRequirement(
            tool_execution=ToolExecution(
                tool_call_id="call-1",
                tool_name="ask_feedback",
                approval_type="required",
                requires_user_input=True,
                user_feedback_schema=[FakeFeedbackQuestion(question="Deploy?")],
            )
        )
        rr = FakeRunResponse(tools=[tool], requirements=[requirement])

        await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert rr.requirements[0].is_resolved()
        assert rr.requirements[0].tool_execution.answered is True

    @pytest.mark.asyncio
    async def test_null_user_input_value_stays_blocked_async(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(
            return_value=([{"status": "approved", "resolution_data": {"values": {"reason": None}}}], 1)
        )
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_reason",
            approval_type="required",
            requires_user_input=True,
            user_input_schema=[FakeUserInputField(name="reason")],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert tool.user_input_schema[0].value is None
        assert tool.answered is None

    @pytest.mark.asyncio
    async def test_string_user_feedback_selection_stays_blocked_async(self):
        db = MagicMock()
        db.get_approvals = AsyncMock(
            return_value=([{"status": "approved", "resolution_data": {"selections": {"Deploy?": "Yes"}}}], 1)
        )
        yes = FakeFeedbackOption(label="Yes")
        prefix = FakeFeedbackOption(label="Ye")
        question = FakeFeedbackQuestion(question="Deploy?", options=[prefix, yes])
        tool = ToolExecution(
            tool_call_id="call-1",
            tool_name="ask_feedback",
            approval_type="required",
            requires_user_input=True,
            user_feedback_schema=[question],
        )
        rr = FakeRunResponse(tools=[tool])

        with pytest.raises(RuntimeError, match="incomplete"):
            await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)

        assert question.selected_options is None
        assert prefix.selected is False
        assert yes.selected is False

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_get_approvals(self):
        db = MagicMock()
        db.get_approvals = MagicMock(return_value=([{"status": "approved", "resolution_data": None}], 1))
        t = FakeToolExecution(approval_type="required", requires_confirmation=True)
        rr = FakeRunResponse(tools=[t])
        await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        assert t.confirmed is True

    @pytest.mark.asyncio
    async def test_attaches_resolved_approval_to_metadata(self):
        approval = {"status": "approved", "resolution_data": None, "resolved_by": "alice"}
        db = MagicMock()
        db.get_approvals = AsyncMock(return_value=([approval], 1))
        rr = FakeRunResponse(tools=[FakeToolExecution(approval_type="required", requires_confirmation=True)])
        await acheck_and_apply_approval_resolution(db=db, run_id="r1", run_response=rr)
        assert rr.metadata is not None
        assert rr.metadata["approval"] == approval
