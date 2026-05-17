# Research-0135: CHUG extraction I/O cost breakdown and Win 1 + Win 2 optimisations

- **Status**: Active
- **Workstream**: perf-audit-pipeline-2026-05-16.md §6 (F6-B, F6-C)
- **Last updated**: 2026-05-16

## Question

The CHUG/K150K feature extractor (`ai/scripts/extract_k150k_features.py`) was
profiled during the end-to-end pipeline perf audit on 2026-05-16.  Two avoidable
costs were identified outside the vmaf binary hot path:

1. **F6-C (Win 1)**: Every 200 completed clips the script re-read the entire
   growing parquet, concatenated the new batch, deduplicated, and wrote it back.
   Total I/O is O(N²) over the 5992-clip CHUG corpus.  Specifically: 30 flush
   cycles, each reading an increasingly large file (200 → 400 → … → 5992 rows ×
   ~48 float64 columns ≈ growing from 0.4 MB to 2.2 MB).  Estimated total
   redundant read mass: ~1.1 GB across the 30 cycles.

2. **F6-B (Win 2)**: For every clip, a `ffprobe` subprocess was spawned to get
   width / height / fps.  CHUG has a JSONL sidecar (`.corpus/chug/chug.jsonl`)
   that already carries `chug_width_manifest`, `chug_height_manifest`, and
   `chug_framerate_manifest` for every distorted clip.  At ~100–300 ms per
   `ffprobe` invocation and 5992 clips this is 600 s – 1800 s (10–30 min) of
   pure syscall overhead amortised over 8 workers.

The vmaf binary startup cost (Win 3 in the task brief, F6-A in the audit) is the
largest per-clip fixed cost (~150–300 ms per invocation) but requires a resident-
daemon architecture; deferred to a follow-up PR.

## Sources

- `.workingdir/perf-audit-pipeline-2026-05-16.md` §6 — per-surface audit findings
  F6-A, F6-B, F6-C.
- `ai/scripts/extract_k150k_features.py` — implementation under study (original
  version at commit `7b10b27a2`).
- [`ADR-0362`](../adr/0362-fr-from-nr-adapter.md) — FR-from-NR adapter rationale.
- [`ADR-0382`](../adr/0382-k150k-parallel-extraction.md) — parallelism model and
  worker isolation.

## Findings

### Win 1: Replace per-flush parquet with at-end-only write

**Old strategy**: every 200 completed clips:
```python
# Pseudocode of old _flush_parquet
existing = pd.read_parquet(out_path)          # grows with N
combined = pd.concat([existing, new_df])
combined.to_parquet(tmp, index=False)
tmp.rename(out_path)
```
Total I/O = sum_{k=1}^{30} k×200 rows = 30×31÷2×200 = 93 000 row-reads for a
5992-clip run.  At ~48 float64 columns, each row is ~384 bytes compressed, so
~35 MB of redundant reads; the real figure is higher due to parquet metadata
and the per-cycle write amplification.

**New strategy**:
- Rows are accumulated in memory throughout the run.
- Each completed row is appended to a JSONL staging file
  (`<out>.rows.jsonl`) for crash durability (single-writer, main-process-only,
  per the checkpoint thread-safety invariant in `ai/AGENTS.md`).
- The parquet is written **exactly once** at the end of the run.
- On an interrupted + resumed run: the `.done` checkpoint file is the primary
  restartability gate (unchanged); the staging file lets the main process recover
  rows that were completed before the crash but not yet in the parquet.

Total I/O = 1 × N_rows × row_size — linear, not quadratic.

**Correctness**: the bit-equivalence test
`TestStagingIO::test_parquet_matches_legacy_flush_equivalent` builds 600 rows,
runs the old three-batch concat strategy and the new single-write strategy, and
asserts that columns, row count, ordering, and MOS values are identical.  NaN
propagation through the JSONL round-trip (`json.dumps(allow_nan=True)` /
`json.loads`) is verified by `test_write_parquet_preserves_nan`.

### Win 2: Skip ffprobe when JSONL sidecar has geometry

CHUG's public `chug.jsonl` includes `chug_width_manifest`, `chug_height_manifest`,
and `chug_framerate_manifest` for every clip.  The `--metadata-jsonl` flag already
loads this sidecar into `jsonl_meta` for metadata enrichment.

The new `_geometry_from_sidecar(meta)` helper reads those three fields (plus
optional `chug_bit_depth` for pixel-format inference) and returns
`(width, height, pix_fmt, fps)` without any subprocess.  The helper returns
`None` for K150K-A clips (no sidecar) or incomplete sidecar rows, triggering
the existing `_probe_geometry(mp4)` ffprobe fallback.

The sidecar pixel-format inference policy: `chug_bit_depth == 10` →
`yuv420p10le`; otherwise `yuv420p`.  This matches the policy that `_probe_geometry`
applies to ffprobe output (`"yuv420p10le" if "10" in pix_fmt else "yuv420p"`).

**Per-clip saving**: one `subprocess.run(["ffprobe", ...])` per clip, which
measured 100–300 ms on the RTX 4090 host.  With 8 workers and 5992 clips,
effective savings are 5992 × ~200 ms / 8 = ~150 s (2.5 min) of wall time.

## Alternatives explored

### Win 1 alternatives

| Alternative | Assessment | Decision |
|---|---|---|
| **Separate per-batch parquet shards + merge at end** | Avoids growing reads; correct. Adds post-processing step and a second parquet layout. | Rejected — single in-memory accumulation + one write is simpler; 5992 rows × ~48 columns ≈ ~10 MB RAM, negligible on any modern host. |
| **DuckDB / Delta Lake for append-only writes** | Truly O(1) per append. Zero-dependency overhead (DuckDB is not in the training venv). | Rejected for this PR — dependency cost not justified for a one-time extraction job. Could revisit for 150k-clip runs. |
| **Keep per-flush but use pyarrow streaming writer** | Avoids the read-back; correct I/O complexity. More code than the in-memory approach. | Rejected — in-memory accumulation is simpler and sufficient at this corpus size. |
| **JSONL as final output instead of parquet** | Eliminates parquet entirely. JSONL is append-safe and crash-durable natively. | Rejected — downstream loaders (`ai/scripts/train_*.py`) read parquet; changing the output format is a wider refactor. |

### Win 2 alternatives

| Alternative | Assessment | Decision |
|---|---|---|
| **Pre-build a geometry CSV from chug.jsonl once** | Decouples the sidecar from the extraction script. Adds a prep step operators can forget. | Rejected — the JSONL sidecar is already loaded for metadata enrichment; reading geometry from it costs zero extra I/O. |
| **Cache ffprobe results on disk** | Avoids repeated probes on resume. Adds a cache invalidation problem. | Rejected — the sidecar approach is simpler and has no invalidation surface. |

## Open questions

- **Win 3 (resident vmaf daemon)**: the vmaf binary startup cost (~150–300 ms ×
  5992 clips / 8 workers ≈ 28–56 min) dominates both wins combined.  A persistent
  subprocess with stdin/stdout job framing would eliminate it.  Deferred: requires
  a protocol design and vmaf binary changes or a socket-based wrapper.
- **JSONL staging file size**: for a 150k-clip run (KoNViD-150k-A), the staging
  file grows to ~150k lines × ~2 kB per row ≈ 300 MB.  Still within acceptable
  bounds for a run that takes multiple hours; the staging file is deleted
  immediately after the final parquet write.
- **Concurrent staging writers**: the single-writer invariant (staging writes happen
  only in the main process, post `fut.result()`) is preserved by the existing
  `as_completed()` loop structure.  Parallel workers must never touch the staging
  file — they return row dicts only.

## Related

- `.workingdir/perf-audit-pipeline-2026-05-16.md` — source audit document
- `ai/tests/test_extract_k150k_perf.py` — test suite for this optimisation
- `ai/AGENTS.md` §K150K-A corpus extraction invariants — updated with at-end-only
  parquet write invariant
- `changelog.d/perf/chug-extraction-io.md` — changelog fragment
