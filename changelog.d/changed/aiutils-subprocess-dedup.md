**aiutils: extract `run_cmd` subprocess helper (ADR-0486)** — `ai/src/aiutils/subprocess_utils.py`
adds a typed `run_cmd()` wrapper consolidating the `capture_output=True, text=True` idiom used
across `ai/scripts/`. Removes inline `subprocess` imports from `bvi_dvc_to_full_features.py`
and `collect_gpu_calibration_data.py`; closes Group 6 of the 2026-05-16 dedup audit.
