# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Test that K150K extraction schema does not include ssimulacra2."""

import sys
from pathlib import Path

# Ensure ai package is importable
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "ai" / "scripts"))

# noinspection PyUnresolvedReference
import extract_k150k_features as extractor  # noqa: E402


def test_feature_names_excludes_ssimulacra2() -> None:
    """Verify ssimulacra2 is not in the output feature list."""
    assert "ssimulacra2" not in extractor.FEATURE_NAMES
    assert len(extractor.FEATURE_NAMES) == 25


def test_cuda_extractor_names_excludes_ssimulacra2_cuda() -> None:
    """Verify ssimulacra2_cuda is not in the CUDA extractor list."""
    assert "ssimulacra2_cuda" not in extractor.CUDA_EXTRACTOR_NAMES
    assert len(extractor.CUDA_EXTRACTOR_NAMES) == 9


def test_metric_aliases_excludes_ssimulacra2() -> None:
    """Verify ssimulacra2 lookup key is not in aliases."""
    assert "ssimulacra2" not in extractor._METRIC_ALIASES
    assert len(extractor._METRIC_ALIASES) == 25
