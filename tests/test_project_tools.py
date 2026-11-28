from pathlib import Path

import pytest

from isaac.security.projects import OPERATIONS, ProjectRegistry


@pytest.fixture
def registry(tmp_path):
    return ProjectRegistry(tmp_path / "profile")


@pytest.fixture
def project(tmp_path, registry):
    root = tmp_path / "sample"
    root.mkdir()
    return registry.register("Tapyr", root, OPERATIONS, ["python3", "-m", "unittest"])


def test_registry_restart_and_remove(project, registry):
    reopened = ProjectRegistry(registry.path.parent)
    assert reopened.get("tapyr") == reopened.get(project["id"]) == project
    assert reopened.list() == [project]
    reopened.update("Tapyr", [])
    assert registry.get("Tapyr")["operations"] == []
    reopened.remove(project["id"])
    assert Path(project["root"]).is_dir()
    with pytest.raises(PermissionError):
        registry.get("Tapyr")


def test_broad_root_and_unknown_command(registry, tmp_path):
    with pytest.raises(PermissionError):
        registry.register("Home", Path.home(), OPERATIONS)
    (tmp_path / "repo").mkdir()
    with pytest.raises(ValueError):
        registry.register("Bad", tmp_path.joinpath("repo"), OPERATIONS, ["bash"]) 
