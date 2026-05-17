# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Shared helper utilities for AI scripts.

This package centralizes common patterns (file hashing, time formatting,
CLI setup, Parquet I/O) to reduce code duplication across the ai/scripts/
directory and establish standard interfaces for new scripts.
"""

from aiutils.file_utils import sha256
from aiutils.jsonl_utils import iter_jsonl
from aiutils.parquet_utils import write_parquet_atomic
from aiutils.time_utils import now_iso_8601

__all__ = [
    "iter_jsonl",
    "now_iso_8601",
    "sha256",
    "write_parquet_atomic",
]
