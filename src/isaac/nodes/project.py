"""Deterministic local voice commands. Only the active microphone turn can approve."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage

from isaac.core.voice_turn import current_voice_turn
from isaac.identity.address import strip_selene_address
from isaac.mcp.projects import execute
from isaac.security.audit import audit
from isaac.security.project_approval import Proposal
from isaac.security.projects import ProjectRegistry

_PATTERNS = (
    ("project_git_status", r"(?:check|show|get) (?:the )?git status (?:of|for) (.+)"),
    ("project_run_tests", r"run (?:the )?tests (?:of|for) (.+)"),
    ("project_list_files", r"list (?:the )?files in (.+)"),
    ("project_read_file", r"read (.+?) (?:in|from) (.+)"),
    ("history", r"(?:show|recall) (?:the )?project history for (.+)"),
)


def _request(state):
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        return "", {}
    text = strip_selene_address(str(messages[-1].content)).strip().rstrip(".!?")
    if text.lower() == "confirm tests":
        return "confirm", {}
    for tool, pattern in _PATTERNS:
        if match := re.fullmatch(pattern, text, re.I):
            fields = match.groups()
            args = {"project": fields[-1]}
            if len(fields) > 1:
                args["path"] = fields[0]
            return tool, args
    return "", {}


def handles(state):
    turn = current_voice_turn.get()
    return (
        turn is not None and hasattr(turn, "approval") and bool(turn.approval or _request(state)[0])
    )


def _reply(tool, result):
    if "error" in result:
        return "The project request failed. See the diagnostic in your terminal."
    if tool == "project_git_status":
        return (
            f"Git reports {len(result['modified'])} modified, {len(result['staged'])} staged "
            f"and {len(result['untracked'])} untracked files."
        )
    if tool == "project_list_files":
        return f"I found {len(result['entries'])} entries. The list is in your terminal."
    if tool == "project_read_file":
        return "I've displayed the requested file excerpt in your terminal."
    if result["timed_out"]:
        return "The configured test run timed out."
    return f"The configured test run exited with code {result['exit_code']}."


def project_node(state):
    """Handle one explicit request and keep raw results outside message history."""
    turn = current_voice_turn.get()
    turn.check()
    session = state.get("session_id", "")
    tool, args = _request(state)
    result = {}
    proposal = turn.approval
    if proposal and tool != "confirm":
        turn.approval = None
        audit("approval", "project_cancelled", details={"origin": proposal.origin, "turn": turn.id})
        reply = "The test request was cancelled."
    elif tool == "confirm":
        if proposal is None:
            reply = "There is no pending test request."
        else:
            result = execute(
                "project_run_tests",
                {"project": proposal.project},
                session=session,
                confirmation=proposal,
            )
            turn.approval = None
            reply = _reply("project_run_tests", result)
    elif tool in {"project_run_tests", "history"}:
        try:
            project = ProjectRegistry().get(args["project"])
            if tool == "history":
                from isaac.meta.learner import get_learner

                result = {"history": get_learner().project_history(project["id"])}
                reply = f"I found {len(result['history'])} recent project observations."
            else:
                if tool not in project["operations"] or not project["command"]:
                    raise PermissionError
                audit(
                    "approval",
                    "project_proposed",
                    details={
                        "origin": turn.id,
                        "project": project["id"],
                        "tool": tool,
                        "argv": project["command"],
                    },
                )
                turn.check()
                turn.offer = Proposal.create(turn, session, project)
                reply = (
                    f"Run {' '.join(project['command'])} in the sandbox for {project['name']} "
                    f"at {project['root']}? Say confirm tests to approve."
                )
        except (PermissionError, ValueError):
            reply = "This project or operation is unavailable."
    else:
        result = execute(tool, args, session=session)
        reply = _reply(tool, result)
    return {
        "project_result": result,
        "messages": [AIMessage(content=reply)],
        "current_phase": "project",
    }
