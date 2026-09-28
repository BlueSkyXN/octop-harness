"""Local execution timeout and cancellation reap shell descendants."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from octop_harness.backends.local_shell import HarnessLocalShellBackend
from octop_harness.backends.process import execution_scope

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process-group assertions require POSIX")


def _command(tmp_path: Path) -> str:
    code = (
        "import time; from pathlib import Path; "
        f"Path({str(tmp_path / 'started')!r}).touch(); "
        "time.sleep(1.5); "
        f"Path({str(tmp_path / 'escaped')!r}).touch()"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} & wait"


@pytest.mark.asyncio
async def test_cancel_kills_descendant(tmp_path: Path) -> None:
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False)
    task = asyncio.create_task(backend.aexecute(_command(tmp_path), timeout=10))
    async with asyncio.timeout(2):
        while not (tmp_path / "started").exists():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(1.6)
    assert not (tmp_path / "escaped").exists()


@pytest.mark.asyncio
async def test_scope_clamps_default_execute_timeout(tmp_path: Path) -> None:
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False)
    async with execution_scope(timeout_s=0.1):
        result = await backend.aexecute(_command(tmp_path))
    assert result.exit_code == 124
    await asyncio.sleep(1.6)
    assert not (tmp_path / "escaped").exists()


@pytest.mark.asyncio
async def test_scope_isolates_other_turns(tmp_path: Path) -> None:
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False)

    async def timed() -> int | None:
        async with execution_scope(timeout_s=0.01):
            return (await backend.aexecute(f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(1)'")).exit_code

    async def normal() -> int | None:
        async with execution_scope():
            return (await backend.aexecute(f"{shlex.quote(sys.executable)} -c 'print(1)'")).exit_code

    assert await asyncio.gather(timed(), normal()) == [124, 0]
