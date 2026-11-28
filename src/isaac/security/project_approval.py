"""One-use request-bound capabilities held only by the local voice host."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from isaac.security.audit import audit
from isaac.security.capabilities import CapabilityToken


@dataclass
class Proposal:
    origin: str
    session: str
    project: str
    token: CapabilityToken

    def binding(self, project, tool, args):
        return hashlib.sha256(
            json.dumps([self.origin, self.session, project, tool, args], sort_keys=True).encode()
        ).hexdigest()

    @classmethod
    def create(cls, turn, session, project):
        token = CapabilityToken(
            token_id=secrets.token_hex(16),
            tool_name="project_run_tests",
            expires_at=(datetime.now(UTC) + timedelta(seconds=60)).isoformat(),
            max_uses=1,
        )
        p = cls(turn.id, session, project["id"], token)
        token.action = p.binding(project, token.tool_name, {"project": project["id"]})
        return p

    def consume(self, turn, session, project, tool, args):
        turn.check()
        if (
            turn.spoken_command.casefold().rstrip(".!?") != "confirm tests"
            or turn.approval is not self
            or self.session != session
            or self.origin == turn.id
            or not self.token.matches(tool, self.binding(project, tool, args))
        ):
            raise PermissionError("Approval is stale or does not match this request")
        self.token.use_count += 1
        turn.approval = None
        audit(
            "approval",
            "project_confirmed",
            details={"tool": tool, "project": self.project, "origin": self.origin, "turn": turn.id},
        )
        turn.check()
