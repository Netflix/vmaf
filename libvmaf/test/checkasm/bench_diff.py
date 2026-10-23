#!/usr/bin/env python3
"""Compare two checkasm --bench --json outputs and print a markdown summary.

Each input file may contain leading plain-text log lines before the JSON
object (checkasm always prints its human-readable summary to stdout first);
this script finds the JSON object and ignores anything before it.

Usage: bench_diff.py <base.json> <pr.json>
"""
import json
import sys

# A change smaller than this (in percent) is treated as noise and excluded
# from the "significant" table, since CI runners have enough jitter that
# small deltas aren't meaningful.
NOISE_THRESHOLD_PCT = 10.0
MAX_ROWS = 30


def load(path):
    text = open(path).read()
    start = text.index("{")
    return json.loads(text[start:])


def flatten(data):
    """Returns {(function, version): median_cycles}."""
    out = {}
    for name, fn in data.get("functions", {}).items():
        for ver, v in fn.get("versions", {}).items():
            cycles = v.get("adjustedCycles", {}).get("median")
            if cycles is not None:
                out[(name, ver)] = cycles
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: bench_diff.py <base.json> <pr.json>", file=sys.stderr)
        return 1

    base = flatten(load(sys.argv[1]))
    pr = flatten(load(sys.argv[2]))

    keys = sorted(set(base) | set(pr))
    rows = []
    added, removed = [], []
    for key in keys:
        name, ver = key
        label = f"{name}_{ver}"
        if key not in base:
            added.append(label)
            continue
        if key not in pr:
            removed.append(label)
            continue
        base_cycles, pr_cycles = base[key], pr[key]
        if base_cycles <= 0:
            continue
        pct = (pr_cycles - base_cycles) / base_cycles * 100.0
        rows.append((label, base_cycles, pr_cycles, pct))

    significant = [r for r in rows if abs(r[3]) >= NOISE_THRESHOLD_PCT]
    significant.sort(key=lambda r: -abs(r[3]))

    print(f"Compared {len(rows)} functions common to both master and this PR.")
    print()

    if not significant and not added and not removed:
        print(f"No changes >= {NOISE_THRESHOLD_PCT:.0f}% detected.")
        return 0

    if significant:
        shown = significant[:MAX_ROWS]
        print(f"| Function | master (cycles) | PR (cycles) | Change |")
        print(f"|---|---:|---:|---:|")
        for label, base_cycles, pr_cycles, pct in shown:
            marker = "🔴" if pct > 0 else "🟢"
            print(f"| {label} | {base_cycles:,.0f} | {pr_cycles:,.0f} "
                  f"| {marker} {pct:+.1f}% |")
        if len(significant) > MAX_ROWS:
            print()
            print(f"_...and {len(significant) - MAX_ROWS} more with "
                  f"|change| >= {NOISE_THRESHOLD_PCT:.0f}%._")

    if added:
        print()
        print(f"**New in this PR:** {', '.join(sorted(added))}")
    if removed:
        print()
        print(f"**Removed in this PR:** {', '.join(sorted(removed))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
