"""Shared contract for the Perception prompt, Ollama decoder and local validation."""

from pydantic import BaseModel, Field

from isaac.core.state import TaskMode


class PerceptionResult(BaseModel):
    observations: list[str]
    hypothesis: str
    task_mode: TaskMode
    confidence: float = Field(default=0.6, ge=0, le=1)
    active_window_title: str | None = None
    current_url: str | None = None
    screen_width: int | None = None
    screen_height: int | None = None
