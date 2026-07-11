from copy import deepcopy

import pytest

from agno.models.response import ToolExecution
from agno.run.agent import RunOutput
from agno.run.requirement import RunRequirement
from agno.run.team import TeamRunOutput
from agno.session import AgentSession


class _LegacyInteger(int):
    pass


@pytest.mark.parametrize(
    ("run_class", "owner_field"),
    ((RunOutput, "agent_id"), (TeamRunOutput, "team_id")),
)
@pytest.mark.parametrize("legacy_value", (50358656, _LegacyInteger(50358656)))
@pytest.mark.parametrize(
    "field_name",
    ("run_id", "parent_run_id", "forked_from_run_id", "regenerated_from"),
)
def test_from_dict_converts_legacy_integer_identity_without_mutating_input(
    run_class,
    owner_field: str,
    legacy_value: int,
    field_name: str,
):
    data = {
        "run_id": "run-1",
        owner_field: "owner-1",
        "session_id": "session-1",
        field_name: legacy_value,
    }
    original = deepcopy(data)

    run = run_class.from_dict(data)

    assert getattr(run, field_name) == "50358656"
    assert data == original


@pytest.mark.parametrize(
    ("run_class", "owner_field"),
    ((RunOutput, "agent_id"), (TeamRunOutput, "team_id")),
)
@pytest.mark.parametrize("invalid_value", (True, 1.5))
@pytest.mark.parametrize(
    "field_name",
    ("run_id", "parent_run_id", "forked_from_run_id", "regenerated_from"),
)
def test_from_dict_rejects_invalid_identity(
    run_class,
    owner_field: str,
    field_name: str,
    invalid_value: object,
):
    with pytest.raises(TypeError, match=field_name):
        run_class.from_dict(
            {
                "run_id": "run-1",
                owner_field: "owner-1",
                "session_id": "session-1",
                field_name: invalid_value,
            }
        )


@pytest.mark.parametrize(
    ("run_class", "owner_field"),
    ((RunOutput, "agent_id"), (TeamRunOutput, "team_id")),
)
@pytest.mark.parametrize(
    "field_name",
    ("run_id", "parent_run_id", "forked_from_run_id", "regenerated_from"),
)
def test_to_dict_rejects_invalid_identity(
    run_class,
    owner_field: str,
    field_name: str,
):
    run = run_class(run_id="run-1", session_id="session-1", **{owner_field: "owner-1"})
    setattr(run, field_name, 50358656)

    with pytest.raises(TypeError, match=field_name):
        run.to_dict()


def test_agent_session_from_dict_returns_canonical_run_identities():
    data = {
        "session_id": "session-1",
        "agent_id": "agent-1",
        "runs": [
            {
                "run_id": 50358656,
                "parent_run_id": 11,
                "forked_from_run_id": 12,
                "regenerated_from": 13,
                "agent_id": "agent-1",
                "session_id": "session-1",
                "requirements": [
                    {
                        "tool_execution": {"tool_name": "lookup", "tool_args": {}},
                        "member_run_id": 14,
                        "routed_member_run_id": 15,
                    }
                ],
            },
            {
                "run_id": "50358656",
                "team_id": "team-1",
                "session_id": "session-1",
            },
        ],
    }

    session = AgentSession.from_dict(data)

    assert session is not None
    assert session.runs is not None
    assert [run.run_id for run in session.runs] == ["50358656", "50358656"]
    assert isinstance(session.runs[0], RunOutput)
    assert isinstance(session.runs[1], TeamRunOutput)
    assert session.runs[0].parent_run_id == "11"
    assert session.runs[0].forked_from_run_id == "12"
    assert session.runs[0].regenerated_from == "13"
    assert session.runs[0].requirements is not None
    assert session.runs[0].requirements[0].member_run_id == "14"
    assert session.runs[0].requirements[0].routed_member_run_id == "15"


@pytest.mark.parametrize(
    ("run_class", "owner_field"),
    ((RunOutput, "agent_id"), (TeamRunOutput, "team_id")),
)
def test_from_dict_converts_identity_in_wrapped_run_payload(run_class, owner_field: str):
    run = run_class.from_dict(
        {
            "run": {
                "run_id": 50358656,
                owner_field: "owner-1",
                "session_id": "session-1",
            }
        }
    )

    assert run.run_id == "50358656"


@pytest.mark.parametrize("field_name", ("member_run_id", "routed_member_run_id"))
@pytest.mark.parametrize("legacy_value", (50358656, _LegacyInteger(50358656)))
def test_run_requirement_from_dict_converts_legacy_integer_identity(field_name: str, legacy_value: int):
    requirement = RunRequirement.from_dict(
        {
            "tool_execution": {"tool_name": "lookup", "tool_args": {}},
            field_name: legacy_value,
        }
    )

    assert getattr(requirement, field_name) == "50358656"


@pytest.mark.parametrize("field_name", ("member_run_id", "routed_member_run_id"))
@pytest.mark.parametrize("invalid_value", (True, 1.5))
def test_run_requirement_from_dict_rejects_invalid_identity(field_name: str, invalid_value: object):
    with pytest.raises(TypeError, match=field_name):
        RunRequirement.from_dict(
            {
                "tool_execution": {"tool_name": "lookup", "tool_args": {}},
                field_name: invalid_value,
            }
        )


@pytest.mark.parametrize("field_name", ("member_run_id", "routed_member_run_id"))
def test_run_requirement_to_dict_rejects_invalid_identity(field_name: str):
    requirement = RunRequirement(tool_execution=ToolExecution(tool_name="lookup", tool_args={}))
    setattr(requirement, field_name, 50358656)

    with pytest.raises(TypeError, match=field_name):
        requirement.to_dict()
