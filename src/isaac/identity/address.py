"""Recognize SELENE as a conversational form of address."""

import re

from isaac.identity.soul import get_soul, uses_selene_identity

_REQUEST = (
    r"(?:what|who|why|when|where|how|can|could|would|please|tell|show|help|"
    r"como|qual|quem|o\s+que|voc[eê]|pod(?:e|eria)|por\s+favor|me|tudo)\b"
)
_ADDRESS = re.compile(
    r"^(?P<greeting>(?:(?:hey|hi|hello|oi|olá)[ ,]+)?)"
    rf"(?:selene(?=\s*[,!:]|\s*[?.]?\s*$|\s+{_REQUEST})|"
    rf"(?:celene|selena|selina|celina|celine|céline)(?=\s*[,!:]\s*{_REQUEST}|\s+{_REQUEST}))",
    re.IGNORECASE,
)
_NAME_QUESTION = re.compile(
    r"(?:what(?:['’]s|\s+is)\s+your\s+name|"
    r"what\s+should\s+i\s+call\s+you|"
    r"(?:como\s+(?:(?:voc[eê]\s+)?se\s+chama|[eé]\s+seu\s+nome))|"
    r"qual\s+[eé]\s+o\s+seu\s+nome)[\s?.!]*",
    re.IGNORECASE,
)
_NAME_ORIGIN_QUESTION = re.compile(
    r"(?:why\s+are\s+you\s+called\s+selene|why\s+is\s+your\s+name\s+selene)[\s?.!]*",
    re.IGNORECASE,
)
_NAMER_QUESTION = re.compile(
    r"(?:who\s+named\s+you|who\s+gave\s+you\s+your\s+name)[\s?.!]*",
    re.IGNORECASE,
)


def normalize_selene_address(text: str) -> str:
    """Canonicalize only a name at the start of a conversational call."""
    return _ADDRESS.sub(r"\g<greeting>SELENE", text, count=1)


def strip_selene_address(text: str) -> str:
    """Return the request without its leading SELENE vocative."""
    return _ADDRESS.sub("", text, count=1).lstrip(" ,!:")


def selene_identity_reply(text: str) -> str | None:
    """Return a fixed English reply for an exact SELENE identity question."""
    if not uses_selene_identity():
        return None

    question = strip_selene_address(text.strip())
    if _NAME_QUESTION.fullmatch(question):
        return "My name is SELENE."
    if _NAME_ORIGIN_QUESTION.fullmatch(question):
        soul = get_soul()
        return (
            f"My name comes from the Greek word for {soul['name_meaning']}. "
            f"{soul['name_mythology']} "
            f"{soul['named_by']} chose the name because of that connection."
        )
    if _NAMER_QUESTION.fullmatch(question):
        return f"{get_soul()['named_by']} named me Selene."
    return None
