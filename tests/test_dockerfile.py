"""Ensure the Docker image builds successfully."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_dockerfile_builds(tmp_path) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI not available in test environment")
    tag = f"aidaytrading-ci-{uuid.uuid4().hex}"
    build_log = tmp_path / "docker_build.log"
    try:
        result = subprocess.run(
            ["docker", "build", "-t", tag, "."],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        build_log.write_text(result.stdout, encoding="utf-8")
    finally:
        subprocess.run(
            ["docker", "rmi", "-f", tag],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
