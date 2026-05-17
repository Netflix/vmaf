# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Subprocess execution utilities."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def run_cmd(
    cmd: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    timeout: float | None = None,
    **kwargs: object,
) -> subprocess.CompletedProcess:
    """Run a subprocess command with standard error handling.

    Args:
        cmd: Command and arguments to execute.
        capture: If True, capture stdout and stderr (text mode). Equivalent to
            ``capture_output=True, text=True``.
        check: If True (default), raise CalledProcessError on non-zero exit.
            Pass False to inspect returncode manually.
        timeout: Optional timeout in seconds. Raises TimeoutExpired on breach.
        **kwargs: Additional keyword arguments forwarded to subprocess.run().

    Returns:
        CompletedProcess instance. When capture=True, stdout and stderr are
        available as strings on the returned object.

    Raises:
        subprocess.CalledProcessError: If check=True and the command exits non-zero.
        subprocess.TimeoutExpired: If timeout is set and the command exceeds it.
        FileNotFoundError: If the executable is not found on PATH.
    """
    if capture:
        kwargs.setdefault("capture_output", True)
        kwargs.setdefault("text", True)
    return subprocess.run(cmd, check=check, timeout=timeout, **kwargs)
