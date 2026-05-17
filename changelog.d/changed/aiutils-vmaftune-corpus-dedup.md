## Deduplicate file-hash and UTC-timestamp helpers in vmaf-tune corpus

`tools/vmaf-tune/src/vmaftune/corpus.py` previously contained private inline
implementations of `_sha256_of` (streaming SHA-256 file hash) and `_utc_now_iso`
(second-precision UTC ISO-8601 timestamp) that duplicated logic already
centralised in `ai/src/aiutils/file_utils.py` and `ai/src/aiutils/time_utils.py`.

Both inline functions are removed; `corpus.py` now imports `sha256` and
`now_iso_8601` from `aiutils` directly, resolving `ai/src` via a module-level
`sys.path` insertion (the same pattern established in `merge_corpora.py` and
other scripts). All 1,011 existing vmaf-tune tests pass unchanged.
