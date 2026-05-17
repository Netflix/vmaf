- **CHUG sidecar `chug_bit_depth` not loaded** (`ai/scripts/extract_k150k_features.py`):
  `_load_jsonl_metadata` filtered JSONL fields through an explicit `keep` allowlist
  that omitted `chug_bit_depth`. As a result `_geometry_from_sidecar` always saw
  `meta.get("chug_bit_depth") == None` and fell back to `yuv420p` regardless of
  actual bit depth, silently mis-decoding 10-bit CHUG clips as 8-bit YUV (wrong
  scores, wasted half the dynamic range). Added `"chug_bit_depth"` to `keep` so
  the sidecar-based geometry path correctly infers `yuv420p10le` for 10-bit clips.
  Corrected the module and `_process_clip` docstrings that overstated the ffprobe
  skip (ffprobe still runs for HDR color metadata) (F6-B / Research-0135).
