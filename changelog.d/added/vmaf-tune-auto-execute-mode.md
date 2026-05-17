### vmaf-tune auto --execute — Phase F real encode/score execution (ADR-0454)

`vmaf-tune auto` gains an `--execute` flag that realises the planned cell(s) as
actual FFmpeg encodes followed by libvmaf scores. Results land in
`<--runs-dir>/tune_results.jsonl` (one row per cell, appending across runs).
Two companion flags: `--runs-dir PATH` (default `runs/`) and `--execute-all`
(run every plan cell rather than only the selected winner). Plan-only mode
is unchanged when `--execute` is absent.
