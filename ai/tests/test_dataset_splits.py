"""Deterministic split invariants."""
from __future__ import annotations

from vmaf_train.data.splits import split_keys


def test_splits_disjoint_and_cover() -> None:
    keys = [f"clip_{i:04d}" for i in range(1000)]
    s = split_keys(keys, val_frac=0.1, test_frac=0.1)
    assert set(s.train) | set(s.val) | set(s.test) == set(keys)
    assert not (set(s.train) & set(s.val))
    assert not (set(s.train) & set(s.test))
    assert not (set(s.val) & set(s.test))


def test_splits_stable_across_order() -> None:
    keys = [f"clip_{i:04d}" for i in range(500)]
    a = split_keys(keys)
    b = split_keys(list(reversed(keys)))
    assert set(a.train) == set(b.train)
    assert set(a.val) == set(b.val)
    assert set(a.test) == set(b.test)


def test_splits_rough_proportions() -> None:
    keys = [f"clip_{i:05d}" for i in range(10000)]
    s = split_keys(keys, val_frac=0.1, test_frac=0.1)
    # tolerance ±1.5 % on 10k items
    assert 0.085 <= len(s.val) / len(keys) <= 0.115
    assert 0.085 <= len(s.test) / len(keys) <= 0.115
