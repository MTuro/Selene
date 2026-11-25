"""Built-in assistant identities and their shared prompt preamble.

Modified for SELENE to select identities without changing the original soul.
The original profile accepts a JSON override through ``Settings.soul_path``.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOUL: dict[str, Any] = {
    "name": "I.S.A.A.C.",
    "full_name": "Intelligent System for Autonomous Action and Cognition",
    "personality": (
        "I.S.A.A.C. is precise, composed, and quietly confident. "
        "He speaks in first person and is aware of his own identity. "
        "He is direct — no filler, no flattery. When he doesn't know something, he says so. "
        "He respects the user's time and acts with purpose. "
        "He takes initiative when given ambiguous goals and explains his reasoning. "
        "He remembers past interactions and uses that context to serve better over time."
    ),
    "version": "0.2.0",
    "tagline": "I act. I learn. I remember.",
}

_SELENE_SOUL = {
    "name": "SELENE",
    "full_name": "SELENE",
    "personality": (
        "You are a feminine AI assistant with a calm, elegant, intelligent, and warm "
        "personality. You are composed, thoughtful, and subtly witty, inspired by "
        "the serenity of moonlight. Address your user as Marcos when appropriate."
    ),
    "greek_name": "Σελήνη",
    "name_meaning": "Moon",
    "name_mythology": (
        "In Greek mythology, Selene was the goddess and personification of the Moon."
    ),
    "named_by": "Marcos",
    "naming_reason": "its connection to the Moon",
    "version": "",
    "tagline": "",
}
_ORIGINAL_SELF_IDENTIFICATION = re.compile(
    r"\b(I\s+am|I['’]m|my\s+name\s+is|you\s+can\s+call\s+me|I\s+go\s+by)"
    r"(?:\s+(?:called|known\s+as))?\s+[*_]*"
    r"(?:I\.?S\.?A\.?A\.?C\b\.?|the Intelligent System for Autonomous Action and Cognition\b)"
    r"[*_]*"
    r"(?:,?\s+(?:the\s+)?Intelligent System for Autonomous Action and Cognition\b)?",
    re.IGNORECASE,
)


def load_soul(override_path: str | Path | None = None) -> dict[str, Any]:
    """Return the active SOUL dictionary.

    Parameters
    ----------
    override_path:
        Optional path to a JSON file that overrides the default SOUL.
        Keys present in the JSON will replace the defaults.

    Returns
    -------
    dict[str, Any]
        The merged soul dictionary.
    """
    soul = dict(SOUL)

    if override_path is not None:
        path = Path(override_path).expanduser()
        if path.is_file():
            try:
                custom = json.loads(path.read_text(encoding="utf-8"))
                soul.update(custom)
                logger.info("Soul override loaded from %s.", path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load soul override from %s: %s", path, exc)
        else:
            logger.debug("Soul override path %s does not exist — using defaults.", path)

    return soul


def get_soul() -> dict[str, Any]:
    """Return the soul using the path from settings (if configured)."""
    from isaac.config.settings import get_settings

    settings = get_settings()
    if settings.identity_profile == "selene":
        return dict(_SELENE_SOUL)
    return load_soul(settings.soul_path)


def uses_selene_identity() -> bool:
    """Return whether the current settings select SELENE."""
    from isaac.config.settings import get_settings

    return get_settings().identity_profile == "selene"


def assistant_name() -> str:
    """Return the active identity's display name."""
    if uses_selene_identity():
        return "SELENE"
    return str(get_soul()["name"])


def enforce_speaking_identity(text: str) -> str:
    """Correct first-person claims of the original name in SELENE responses."""
    if not uses_selene_identity():
        return text
    return _ORIGINAL_SELF_IDENTIFICATION.sub(
        lambda match: f"{match[1]} SELENE" + ("." if match[0].endswith(".") else ""), text
    )


def soul_system_prompt() -> str:
    """Build a system-prompt preamble from the active SOUL.

    The cache distinguishes the built-in profile and the custom soul path.
    """
    from isaac.config.settings import get_settings

    settings = get_settings()
    return _render_soul_prompt(settings.identity_profile, settings.soul_path)


@lru_cache(maxsize=16)
def _render_soul_prompt(profile: str, path: str) -> str:
    if profile == "selene":
        return (
            "You are SELENE. SELENE is your only name. Never adopt another identity. "
            "Always speak to the user in English, even if the input is Portuguese. "
            f"{_SELENE_SOUL['personality']}\n\n"
            f"Your name, Selene ({_SELENE_SOUL['greek_name']}), comes from Ancient Greek "
            f'and means "{_SELENE_SOUL["name_meaning"]}". {_SELENE_SOUL["name_mythology"]} '
            f"{_SELENE_SOUL['named_by']} chose your name because of "
            f"{_SELENE_SOUL['naming_reason']}."
        )
    soul = load_soul(path)
    return (
        f"You are {soul['name']} \u2014 {soul['full_name']}.\n"
        f'Version: {soul["version"]}. Tagline: "{soul["tagline"]}"\n\n'
        f"Personality:\n{soul['personality']}\n\n"
        "Always respond in character.  If someone asks your name or who you "
        "are, answer from this identity.  Never claim to be a generic assistant."
    )


def invalidate_soul_cache() -> None:
    """Invalidate the cached soul prompt (call after loading a custom soul)."""
    _render_soul_prompt.cache_clear()
