"""Cancellation-aware local command execution and optional turn deadlines."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any


class _ExecutionScope:
    def __init__(self, timeout_s: float | None = None) -> None:
        self.deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[str]] = set()
        self._cancelled = False

    def register(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._cancelled:
                _kill_process(process)
            else:
                self._processes.add(process)

    def unregister(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.discard(process)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            processes = list(self._processes)
        for process in processes:
            _kill_process(process)


_CURRENT_SCOPE: ContextVar[_ExecutionScope | None] = ContextVar("harness_execution_scope", default=None)


@asynccontextmanager
async def execution_scope(timeout_s: float | None = None) -> AsyncIterator[None]:
    """Bound local shell commands to a turn deadline and reap them on exit.

    The scope is task-local and propagates through ``asyncio.to_thread``.
    Remote sandbox backends retain their own cancellation semantics.
    """
    scope = _ExecutionScope(timeout_s)
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield
    finally:
        try:
            await asyncio.shield(asyncio.to_thread(scope.cancel))
        finally:
            _CURRENT_SCOPE.reset(token)


def _kill_process(process: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError):
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()


def run_process(
    args: str | Sequence[str],
    *,
    shell: bool,
    timeout: float,
    env: dict[str, str] | None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, killing its POSIX process group on timeout or cancellation."""
    scope = _CURRENT_SCOPE.get()
    if scope is not None and scope.deadline is not None:
        timeout = min(timeout, scope.deadline - time.monotonic())
    if timeout <= 0:
        raise subprocess.TimeoutExpired(args, timeout)
    with subprocess.Popen(
        args,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=os.name == "posix",
    ) as process:
        if scope is not None:
            scope.register(process)
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except BaseException:
                _kill_process(process)
                process.communicate()
                raise
            return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        finally:
            if scope is not None:
                scope.unregister(process)


async def aexecute(backend: Any, command: str, *, timeout: int | None) -> Any:
    """Keep the sync backend worker tied to its caller's cancellation."""
    scope = _CURRENT_SCOPE.get() or _ExecutionScope()
    token = _CURRENT_SCOPE.set(scope)
    try:
        return await asyncio.to_thread(backend.execute, command, timeout=timeout)
    except asyncio.CancelledError:
        await asyncio.shield(asyncio.to_thread(scope.cancel))
        raise
    finally:
        _CURRENT_SCOPE.reset(token)
