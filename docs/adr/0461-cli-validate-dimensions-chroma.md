# ADR-0461: CLI validates positive dimensions and chroma-alignment on input videos

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `cli`, `validation`, `correctness`

## Context

`validate_videos()` in `libvmaf/tools/vmaf.c` checked that the two input streams
had matching dimensions, matching pixel formats, and a supported bitdepth for the
reference stream — but it did not:

1. Check the *distorted* stream's bitdepth range (8–16).
2. Reject zero-width or zero-height frames; these produce divide-by-zero or
   zero-stride allocations in `copy_picture_data` and downstream allocators.
3. Reject odd luma widths for 4:2:0 / 4:2:2 formats or odd luma heights for
   4:2:0, where the corresponding chroma plane would be a fractional number of
   pixels.

A `//TODO: more validations are possible.` marker (line 107) had tracked this
gap since the function was written. The gap was surfaced in the
`audit-todo-fixme-2026-05-16.md` TODO audit as item #9.

## Decision

We add `validate_video_info()` (per-stream bitdepth + positive-dimensions check)
and `validate_chroma_alignment()` (even-dimension check for subsampled formats) as
separate static helpers, called from `validate_videos()`. This keeps each helper
under the `readability-function-size` threshold without a NOLINT.  The `//TODO`
marker is removed.

No ABI, protocol, or golden-data change. Error messages go to stderr and the
function returns a non-zero error count; the call site already propagates that
count to a non-zero exit.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Inline all checks in `validate_videos` with a NOLINT | Fewest files touched | Violates touched-file lint-clean rule (ADR-0141); NOLINT without a load-bearing invariant is not allowed (ADR-0278) | Rejected |
| Leave as-is with a `//TODO` | Zero effort | Crash path on zero-dim / odd-chroma input; audit marker left open | Rejected |
| Add a separate validation pass after `vidinput` open | Would allow richer diagnostics | Requires threading state through a second call site; no additional benefit for these simple checks | Rejected |

## Consequences

- **Positive**: the CLI exits with a clear error message instead of crashing or
  producing undefined output for malformed input dimensions.
- **Negative**: none — the checks run once at startup on small structs.
- **Neutral / follow-ups**: the `//TODO` marker is removed; the audit item is
  closed.

## References

- `.workingdir/audit-todo-fixme-2026-05-16.md` item #9 (`libvmaf/tools/vmaf.c:107`).
- ADR-0141 (touched-file lint-clean rule).
- ADR-0278 (NOLINT citation rule).
