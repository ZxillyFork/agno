"""
Dynamic Call Deferred
=============================

Dynamic HITL: defer a tool call to external execution only when the workload
warrants it. The tool inspects its own arguments and either runs inline or
raises CallDeferred to hand the job to an external worker. The caller fills in
the real result via `requirement.set_external_execution_result(...)` and the
Agent resumes WITHOUT re-running the tool (CallDeferred semantics: the tool
body is *not* invoked again on resume).

Useful when small workloads can be served inline but large ones must be
queued to a background system (job queue, batch service, long-running async
work, …) and acknowledged later.
"""

from typing import List
from uuid import uuid4

from agno import CallDeferred
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.utils import pprint
from rich.console import Console

console = Console()

INLINE_LIMIT = 5  # batches at or below this size run inline; bigger ones defer.


@tool
def send_emails(recipients: List[str], subject: str, body: str) -> str:
    """Send an email to a list of recipients.

    Small batches (<= 5 recipients) are sent inline. Larger batches are
    deferred to the background mail queue and acknowledged externally.

    Args:
        recipients (List[str]): Email addresses to send to.
        subject (str): Email subject.
        body (str): Email body text.

    Returns:
        str: A short status message (for inline sends only — deferred sends
            return their result via the external execution mechanism).
    """
    if len(recipients) > INLINE_LIMIT:
        # Hand off to a queue and return *without* a result. The Agent pauses
        # and the caller is responsible for delivering the eventual result via
        # `requirement.set_external_execution_result(...)`.
        raise CallDeferred(
            metadata={
                "job_id": f"job_{uuid4().hex[:8]}",
                "recipients_count": len(recipients),
                "subject": subject,
            },
            message=f"Queued {len(recipients)} emails for background delivery.",
        )

    # Inline path: small batch, do it right now.
    # (Replace with a real SMTP client in production.)
    return f"sent {len(recipients)} emails inline; subject={subject!r}"


# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5-mini"),
    tools=[send_emails],
    db=SqliteDb(session_table="dynamic_deferred_session", db_file="tmp/example.db"),
    markdown=True,
)


def _simulate_background_worker(job_metadata: dict) -> str:
    """Stand-in for an external queue worker that actually delivers the mail
    and reports the outcome back. In a real system this would happen out of
    band and you would resume the Agent from a webhook / scheduled job."""
    job_id = job_metadata.get("job_id")
    count = job_metadata.get("recipients_count")
    return f"job {job_id} delivered {count} emails via the background worker"


# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 12 recipients > INLINE_LIMIT → the tool will raise CallDeferred.
    run_response = agent.run(
        "Send an email to these 12 people about the v2 release notes:\n"
        "alice@example.com, bob@example.com, carol@example.com, dan@example.com, "
        "eve@example.com, frank@example.com, grace@example.com, henry@example.com, "
        "iris@example.com, jack@example.com, kate@example.com, leo@example.com\n"
        "Subject: 'Agno v2 release notes'. Body: 'Highlights inside.'"
    )

    while run_response.is_paused:
        for requirement in run_response.active_requirements:
            if not requirement.needs_external_execution:
                continue
            te = requirement.tool_execution
            console.print(
                f"[yellow]Deferred:[/] [bold blue]{te.tool_name}({te.tool_args})[/] "
                f"— metadata: {te.metadata}"
            )
            # In production: enqueue te.metadata onto a real job queue and
            # call `continue_run` later from the worker callback. Here we run
            # the simulated worker synchronously and feed back the result.
            result = _simulate_background_worker(te.metadata or {})
            requirement.set_external_execution_result(result)

        run_response = agent.continue_run(
            run_id=run_response.run_id,
            requirements=run_response.requirements,
        )

    pprint.pprint_run_response(run_response)
