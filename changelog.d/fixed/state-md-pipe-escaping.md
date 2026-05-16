Fix three broken table rows in `docs/state.md` caused by `\|` outside backtick
spans (FINDING-10, HP-2, fr_regressor_v2 seed-redo row). Backslash-pipe outside
backticks is parsed as a literal backslash followed by a cell boundary, creating
phantom extra columns. Fixed by rewriting as prose or using bare pipes inside
backtick code spans.
