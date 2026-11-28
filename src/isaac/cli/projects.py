"""Operator-only management of local project permissions and verified feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from isaac.security.projects import ProjectRegistry

app = typer.Typer(help="Manage authorized local projects.")


def show(value):
    typer.echo(json.dumps(value, indent=2))


@app.command()
def register(
    name: str,
    root: Path,
    allow: Annotated[list[str], typer.Option()],
    runner: str = typer.Option(""),
):
    """Register a directory. Choose unittest or pytest for an optional fixed runner."""
    command = ["python3", "-m", runner] if runner else []
    show(ProjectRegistry().register(name, root, allow, command))


@app.command("list")
def list_projects():
    """List registered projects."""
    show(ProjectRegistry().list())


@app.command()
def inspect(project: str):
    """Display the current project configuration."""
    show(ProjectRegistry().get(project))


@app.command()
def permissions(project: str, allow: Annotated[list[str] | None, typer.Option()] = None):
    """Replace the entire operation allowlist."""
    show(ProjectRegistry().update(project, allow or []))


@app.command()
def remove(project: str):
    """Forget a grant while preserving the project directory."""
    ProjectRegistry().remove(project)


@app.command()
def history(project: str):
    """Recall recent project experiences."""
    from isaac.meta.learner import get_learner

    show(get_learner().project_history(ProjectRegistry().get(project)["id"]))


@app.command()
def verify(
    project: str,
    experience: int,
    fulfilled: bool = typer.Option(..., "--fulfilled/--unfulfilled"),
    feedback: str = "",
):
    """Explicitly assess whether an observed operation fulfilled the user's objective."""
    from isaac.meta.learner import get_learner

    get_learner().verify_project(
        ProjectRegistry().get(project)["id"], experience, fulfilled, feedback
    )
