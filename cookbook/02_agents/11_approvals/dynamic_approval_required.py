"""
Dynamic Approval Required
=============================

Dynamic HITL: ask for approval only when the tool's arguments are sensitive.

Unlike @tool(requires_confirmation=True) — which pauses every call — the tool
itself decides at runtime whether the user needs to approve, by raising
ApprovalRequired from inside its body. The Agent pauses, the caller confirms
on the requirement, and the tool re-executes with
run_context.tool_call_approved = True so the same call no longer pauses.

Useful when most invocations are safe and only a small subset (sensitive
paths, large amounts, irreversible actions, …) need human sign-off.
"""

from pathlib import Path

from agno import ApprovalRequired
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools import tool
from agno.utils import pprint
from rich.console import Console
from rich.prompt import Prompt

console = Console()

# Paths the tool refuses to write to without explicit human approval.
PROTECTED_PATHS = {".env", ".git/config", "secrets.json", "credentials.json"}


def _is_protected(path: str) -> bool:
    p = Path(path).as_posix()
    while p.startswith("./"):
        p = p[2:]
    return p in PROTECTED_PATHS or p.startswith(".git/")


# Note: no requires_confirmation / external_execution here. The HITL decision
# is made dynamically from inside the tool body.
@tool
def write_file(run_context: RunContext, path: str, content: str) -> str:
    """Write text to a file under the workspace.

    Args:
        path (str): Destination file path (relative to the workspace).
        content (str): Text content to write.

    Returns:
        str: A short status message.
    """
    # First-time call for a sensitive path: pause and request approval.
    # On resume after `requirement.confirm()`, tool_call_approved becomes True
    # and we proceed to the write.
    if _is_protected(path) and not run_context.tool_call_approved:
        raise ApprovalRequired(
            metadata={"path": path, "reason": "protected_path"},
            message=f"Writing to protected path '{path}' needs human approval.",
        )

    target = Path("tmp/workspace") / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} bytes to {target}"


# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5-mini"),
    tools=[write_file],
    db=SqliteDb(session_table="dynamic_approval_session", db_file="tmp/example.db"),
    markdown=True,
)


def _resolve_pending_requirements(run_response):
    """Prompt for each paused requirement and confirm or reject it."""
    for requirement in run_response.active_requirements:
        if not requirement.needs_confirmation:
            continue
        te = requirement.tool_execution
        console.print(
            f"[yellow]Pause:[/] [bold blue]{te.tool_name}({te.tool_args})[/] "
            f"— metadata: {te.metadata}"
        )
        choice = Prompt.ask("Approve?", choices=["y", "n"], default="y").strip().lower()
        if choice == "y":
            requirement.confirm()
        else:
            requirement.reject(note="Operator denied write to protected path.")


# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Two requests in one run: notes.txt (no pause) + .env (dynamic pause).
    run_response = agent.run(
        "Create two files in my workspace:\n"
        "1) notes.txt with the text 'hello world'\n"
        "2) .env with the line 'API_KEY=sk-demo'\n"
    )

    # Loop until the run finishes (a single .env write only pauses once, but the
    # loop also handles models that batch multiple sensitive calls together).
    while run_response.is_paused:
        _resolve_pending_requirements(run_response)
        run_response = agent.continue_run(
            run_id=run_response.run_id,
            requirements=run_response.requirements,
        )

    pprint.pprint_run_response(run_response)
