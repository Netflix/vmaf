---
name: run-netflix-bench
description: Run the Netflix benchmark suite (testdata/bench_all.sh) and diff the output against testdata/netflix_benchmark_results.json. Reports any delta per (resolution, feature, backend).
---

# /run-netflix-bench

## Steps

1. `./build/tools/vmaf` and `./build/tools/vmaf_bench` must exist. If not, run
   `/build-vmaf --backend=<target>` first.
2. `cd testdata && ./bench_all.sh > /tmp/bench-run.json`.
3. Compare: `python3 testdata/compare_combined.py /tmp/bench-run.json \
   testdata/netflix_benchmark_results.json`.
4. For any row where delta > 1e-6 (relative), emit a summary line with
   `(resolution, feature, backend, expected, got, delta)`.
5. Exit 0 if all rows within tolerance; exit 1 otherwise.

## Notes

- `testdata/netflix_benchmark_results.json` is fork-committed (not Netflix golden data);
  regenerate only via `/regen-snapshots` with justification.
- This is NOT the Netflix CPU golden-data gate — that is the Python `assertAlmostEqual`
  suite invoked by `make test-netflix-golden`.
