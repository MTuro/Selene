"""Shared policy boundary for project MCP and local voice operations."""

from __future__ import annotations

import json
import os
import time
from configparser import Error as ConfigError
from pathlib import Path
from sqlite3 import Error as DatabaseError

from isaac.meta.learner import get_learner
from isaac.sandbox.project import sandbox
from isaac.security.audit import audit
from isaac.security.projects import (
    OPERATIONS,
    ProjectRegistry,
    directory,
    permitted,
    project_root,
    read_regular,
)
from isaac.security.redact import redact_secrets

PROPERTIES = {
    "project_list_files": {
        "path": {"type": "string"},
        "depth": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "project_read_file": {
        "path": {"type": "string"},
        "start": {"type": "integer", "minimum": 1},
        "end": {"type": "integer", "minimum": 1},
    },
    "project_git_status": {},
    "project_run_tests": {},
}
DESCRIPTIONS = (
    "List visible project files.",
    "Read lines of a project text file.",
    "Inspect structured project Git status.",
    "Run the fixed test suite with local voice approval.",
)


def schemas():
    return [
        {
            "name": tool,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": {"project": {"type": "string"}, **PROPERTIES[tool]},
                "required": ["project", "path"] if tool == "project_read_file" else ["project"],
                "additionalProperties": False,
            },
        }
        for tool, description in zip(OPERATIONS, DESCRIPTIONS, strict=True)
    ]


def _listing(fd, path, depth):
    rows = []
    scanned = 0
    truncated = False
    denied = 0

    def visit(parent, prefix, levels):
        nonlocal scanned, truncated, denied
        with os.scandir(parent) as entries:
            for entry in entries:
                scanned += 1
                if scanned > 4096 or len(rows) >= 200:
                    truncated = True
                    break
                item = prefix / entry.name
                if not permitted(item) or entry.is_symlink():
                    continue
                try:
                    folder = entry.is_dir(follow_symlinks=False)
                    if not folder and not entry.is_file(follow_symlinks=False):
                        continue
                    rows.append({"path": str(item), "directory": folder})
                    if folder and levels > 1:
                        with directory(parent, Path(entry.name)) as child:
                            visit(child, item, levels - 1)
                except (PermissionError, FileNotFoundError):
                    denied += 1

    with directory(fd, path) as parent:
        visit(parent, path, depth)
    return dict(entries=rows, truncated=truncated, skipped=denied)


def _status(fd):
    result = sandbox(fd)
    if result["exit_code"] != 0 or result["truncated"]:
        raise ValueError(
            "Git status unavailable or exceeds the output limit. " + result["stderr"][:180]
        )
    status = dict(branch=None, ahead=None, behind=None, modified=[], staged=[], untracked=[])
    for row in result["stdout"].split("\0"):
        if row.startswith("# branch.head "):
            status["branch"] = row[14:]
        elif row.startswith("# branch.ab "):
            ahead, behind = row[12:].split()
            status.update(ahead=int(ahead), behind=abs(int(behind)))
        elif row.startswith("? ") and permitted(Path(row[2:])):
            status["untracked"].append(row[2:])
        elif row.startswith(("1 ", "u ")):
            parts = row.split(" ", 8 if row.startswith("1 ") else 10)
            path = parts[-1]
            if permitted(Path(path)):
                if parts[1][0] != ".":
                    status["staged"].append(path)
                if parts[1][1] != ".":
                    status["modified"].append(path)
    return status


def execute(tool, args, *, confirmation=None, session=""):
    """Recheck the current grant and root, then record a bounded observation."""
    started = time.monotonic()
    details = {"tool": tool}
    project_id = None
    try:
        if (
            tool not in OPERATIONS
            or not isinstance(args, dict)
            or set(args) - {"project", *PROPERTIES[tool]}
        ):
            raise ValueError("Invalid project tool arguments")
        with ProjectRegistry().locked(args["project"]) as (_, project):
            project_id = project["id"]
            details.update(project=project_id, session=session)
            if tool not in project["operations"]:
                raise PermissionError("Operation is not allowed for this project")
            audit("tool", "project_request", details=details)
            with project_root(project) as fd:
                if tool == "project_run_tests":
                    from isaac.core.voice_turn import current_voice_turn
                    from isaac.security.project_approval import Proposal

                    turn = current_voice_turn.get()
                    if (
                        turn is None
                        or not isinstance(confirmation, Proposal)
                        or not project["command"]
                    ):
                        raise PermissionError(
                            "Local confirmation and a configured command are required"
                        )
                    result = sandbox(
                        fd,
                        project["command"],
                        check=lambda: confirmation.consume(turn, session, project, tool, args),
                    )
                elif tool == "project_git_status":
                    result = _status(fd)
                else:
                    path = Path(args.get("path", "."))
                    if not permitted(path):
                        raise PermissionError("Restricted project path")
                    if tool == "project_list_files":
                        depth = args.get("depth", 1)
                        if type(depth) is not int or not 1 <= depth <= 5:
                            raise ValueError("Depth must be between 1 and 5")
                        result = _listing(fd, path, depth)
                    else:
                        start = args.get("start", 1)
                        end = args.get("end", start + 199)
                        if type(start) is not int or type(end) is not int or not 1 <= start <= end:
                            raise ValueError("Invalid line range")
                        with directory(fd, path.parent) as parent:
                            data = read_regular(parent, path.name, 1 << 20)
                        if b"\x00" in data:
                            raise ValueError("Binary file")
                        lines = redact_secrets(data.decode()).splitlines()
                        selected = "\n".join(lines[start - 1 : end])
                        result = dict(
                            text=selected[:16384],
                            start=start,
                            end=min(end, len(lines)),
                            truncated=len(selected) > 16384 or len(lines) > end,
                        )
    except (KeyError, TypeError, ValueError, OSError, ConfigError, DatabaseError) as exc:
        result = {"error": redact_secrets(str(exc))[:240]}
    ok = "error" not in result and result.get("exit_code", 0) == 0
    audit("tool", "project_result", details={**details, "execution_ok": ok})
    if project_id:
        from isaac.improvement.performance import get_tracker

        duration_ms = (time.monotonic() - started) * 1000
        get_tracker().record_node(tool, duration_ms, ok, session_id=session)
        summary = {
            k: v
            for k, v in result.items()
            if k in {"error", "exit_code", "timed_out", "truncated", "start", "end"}
        }
        for key in ("entries", "modified", "staged", "untracked"):
            if key in result:
                summary[key] = len(result[key])
        result["experience"] = get_learner().observe_project(
            project_id, tool, session, duration_ms, ok, json.dumps(summary)
        )
    return result
