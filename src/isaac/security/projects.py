"""Local operator grants and descriptor-relative access to project trees."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import uuid
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from isaac.security.path_policy import is_sensitive_path

OPERATIONS = ("project_list_files", "project_read_file", "project_git_status", "project_run_tests")
COMMANDS = (("python3", "-m", "unittest"), ("python3", "-m", "pytest"))
SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".isaac", ".codex", ".agents"}
DIRECTORY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def permitted(path: Path) -> bool:
    return (not path.is_absolute() and ".." not in path.parts
            and not set(path.parts) & SKIP and not is_sensitive_path(path)
            and all(c.isprintable() for c in str(path)))


@contextmanager
def opened(parent: int, name: str, flags: int):
    fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def directory(root: int, path: Path):
    if not permitted(path):
        raise PermissionError("Restricted project path")
    fd = os.dup(root)
    try:
        for part in path.parts:
            child = os.open(part, DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def project_root(project):
    fd = os.open(project["root"], DIRECTORY)
    try:
        info = os.fstat(fd)
        if [info.st_dev, info.st_ino] != project["identity"]:
            raise PermissionError("Project directory has changed. Register it again.")
        yield fd
    finally:
        os.close(fd)


def read_regular(fd, name, limit):
    with opened(fd, name, os.O_RDONLY) as file:
        info = os.fstat(file)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PermissionError("Only regular, single-link files are supported")
        if info.st_size > limit:
            raise ValueError("File exceeds the size limit")
        with os.fdopen(os.dup(file), "rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("File exceeds the size limit")
        return data


def _operations(values):
    values = sorted(set(values))
    if not set(values).issubset(OPERATIONS):
        raise ValueError("Unknown project operation")
    return values


class ProjectRegistry:
    """SQLite serializes grant edits across local processes."""

    def __init__(self, home=None):
        if home is None:
            from isaac.config.settings import get_settings

            home = get_settings().isaac_home
        home.mkdir(exist_ok=True, parents=True)
        self.path = home / "projects.sqlite3"
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, "
                       "name TEXT UNIQUE COLLATE NOCASE, config TEXT NOT NULL)")

    def register(self, name, root, operations, command=()):
        root = Path(root)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,63}", name) or name != name.strip():
            raise ValueError("Use a short alphanumeric project name")
        if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError("Use an absolute, canonical directory")
        if (root in (Path.home(), *Path.home().parents) or is_sensitive_path(root)
                or set(root.parts) & SKIP or self.path.is_relative_to(root)):
            raise PermissionError("Choose a dedicated development directory")
        if command and tuple(command) not in COMMANDS:
            raise ValueError("Unsupported test command")
        info = root.stat()
        now = datetime.now(UTC).isoformat()
        project = dict(id=uuid.uuid4().hex, name=name, root=str(root),
                       identity=[info.st_dev, info.st_ino], operations=_operations(operations),
                       command=list(command), git=(root / ".git").is_dir(),
                       created=now, modified=now)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO projects VALUES (?,?,?)", (project["id"], name, json.dumps(project)))
        return project

    def list(self):
        with closing(sqlite3.connect(self.path)) as db:
            return [json.loads(row[0]) for row in db.execute("SELECT config FROM projects ORDER BY name")]

    @contextmanager
    def locked(self, key):
        with closing(sqlite3.connect(self.path, timeout=35)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT config FROM projects WHERE id=? OR name=?", (key,key)).fetchall()
            if len(rows) != 1:
                raise PermissionError("Unknown or ambiguous project")
            yield db, json.loads(rows[0][0])

    def get(self, key):
        with self.locked(key) as (_, project):
            return project

    def update(self, key, operations):
        with self.locked(key) as (db, project):
            project.update(operations=_operations(operations), modified=datetime.now(UTC).isoformat())
            db.execute("UPDATE projects SET config=? WHERE id=?", (json.dumps(project),project["id"]))
        return project

    def remove(self, key):
        with self.locked(key) as (db, project):
            db.execute("DELETE FROM projects WHERE id=?", (project["id"],))
