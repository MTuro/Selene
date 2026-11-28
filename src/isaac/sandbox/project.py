"""Bounded single-process Arch Linux sandbox for local project inspection and tests."""

from __future__ import annotations

import configparser
import json
import os
import re
import stat
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path

from isaac.sandbox.security import _default_seccomp_profile
from isaac.security.projects import COMMANDS, DIRECTORY, opened, permitted, read_regular
from isaac.security.redact import redact_secrets


def _snapshot(fd, target, remaining, depth=0):
    if depth > 40:
        raise ValueError("Project exceeds snapshot depth limit")
    with os.scandir(fd) as entries:
        for entry in entries:
            remaining[0] -= 1
            if remaining[0] < 0:
                raise ValueError("Project exceeds snapshot entry limit")
            if not permitted(Path(entry.name)):
                continue
            dest = target / entry.name
            if entry.is_dir(follow_symlinks=False):
                with opened(fd, entry.name, DIRECTORY) as child:
                    dest.mkdir()
                    _snapshot(child, dest, remaining, depth + 1)
            elif entry.is_file(follow_symlinks=False):
                data = read_regular(fd, entry.name, min(remaining[1], 8 << 20))
                remaining[1] -= len(data)
                dest.write_bytes(data)


def _git_mounts(stack, fd, tmp):
    """Use only Git's index, objects and references, never repository hooks or filters."""
    git = stack.enter_context(opened(fd, ".git", DIRECTORY))
    mounts = []
    fds = []
    for name in ("objects", "refs", "HEAD", "index", "packed-refs", "shallow"):
        try:
            item = stack.enter_context(
                opened(git, name, DIRECTORY if name in {"objects", "refs"} else os.O_RDONLY)
            )
        except FileNotFoundError:
            if name in {"objects", "refs", "HEAD"}:
                raise
            continue
        if name not in {"objects", "refs"} and not stat.S_ISREG(os.fstat(item).st_mode):
            raise ValueError("Unsupported Git metadata")
        mounts.extend(["--ro-bind-fd", str(item), "/metadata/" + name])
        fds.append(item)

    # Keep conventional upstream tracking only. Other configuration can name executables.
    cfg = configparser.RawConfigParser(strict=False)
    cfg.read_string(read_regular(git, "config", 256 << 10).decode())
    tracking = []
    for section in cfg.sections():
        match = re.fullmatch(r'branch "([\w./-]+)"', section)
        if match:
            remote = cfg.get(section, "remote", fallback="")
            merge = cfg.get(section, "merge", fallback="")
            if re.fullmatch(r"[\w-]+", remote) and re.fullmatch(r"refs/heads/[\w./-]+", merge):
                tracking.extend(
                    [
                        f"[{section}]",
                        f"remote = {remote}",
                        f"merge = {merge}",
                        f'[remote "{remote}"]',
                        f"fetch = +refs/heads/*:refs/remotes/{remote}/*",
                    ]
                )
    config = tmp / "gitconfig"
    config.write_text("\n".join(tracking))
    mounts += ["--ro-bind", str(config), "/metadata/config"]
    return mounts, fds


def sandbox(fd, command=None, *, check=None):
    """Run fixed argv with a clean environment, no sockets and no process creation."""
    allowed = set(_default_seccomp_profile()["syscalls"][0]["names"])
    allowed -= {"socket", "socketpair", "clone", "clone3", "fork", "vfork", "memfd_create"}
    with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
        tmp = Path(temp)
        mounts = []
        fds = []
        if command is None:
            mounts, fds = _git_mounts(stack, fd, tmp)
            mounts += ["--ro-bind-fd", str(fd), "/project"]
            fds.append(fd)
            argv = [
                "/usr/bin/git",
                "--git-dir=/metadata",
                "--work-tree=/project",
                "--no-optional-locks",
                "-c",
                "safe.directory=*",
                "-c",
                "core.preloadIndex=false",
                "-c",
                "index.threads=1",
                "status",
                "--porcelain=v2",
                "-z",
                "--branch",
                "--no-renames",
                "--ignore-submodules=all",
                "-uall",
            ]
        else:
            if check is None or tuple(command) not in COMMANDS:
                raise PermissionError("Invalid test configuration")
            snapshot = tmp / "project"
            snapshot.mkdir()
            _snapshot(fd, snapshot, [20000, 64 << 20])
            mounts = ["--ro-bind", str(snapshot), "/project"]
            argv = ["/usr/bin/python3", *command[1:]]
        args = [
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--disable-userns",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "GIT_CONFIG_NOSYSTEM",
            "1",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--symlink",
            "usr/bin",
            "/bin",
            "--dev-bind",
            "/dev/null",
            "/dev/null",
            "--size",
            str(16 << 20),
            "--tmpfs",
            "/tmp",
            *mounts,
            "--ro-bind",
            str(Path(__file__).with_name("project_child.py")),
            "/bootstrap.py",
            "--chdir",
            "/project",
            "--remount-ro",
            "/",
            "/usr/bin/python3",
            "-I",
            "-S",
            "/bootstrap.py",
            json.dumps(sorted(allowed)),
            *argv,
        ]
        # Regular output files are subject to the child's hard file size limit.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            if check is not None:
                check()
            try:
                result = subprocess.run(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    shell=False,
                    timeout=25,
                    env={},
                    pass_fds=tuple(fds),
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = None
            out.seek(0)
            err.seek(0)
            stdout = out.read(65537)
            stderr = err.read(8193)
            return dict(
                exit_code=code,
                timed_out=code is None,
                truncated=len(stdout) > 65536 or len(stderr) > 8192,
                stdout=redact_secrets(stdout[:65536].decode(errors="replace")),
                stderr=redact_secrets(stderr[:8192].decode(errors="replace")),
            )
