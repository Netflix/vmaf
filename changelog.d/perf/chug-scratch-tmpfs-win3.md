- `ai/scripts/extract_k150k_features.py`: auto-select `/dev/shm` as the
  YUV scratch directory when `/dev/shm` is writable and has at least 20 GiB
  free (Win 3 — Research-0135).  Eliminates NVMe I/O for the ~1.5 GiB
  per-clip raw YUV intermediate, saving an estimated 5–15 s per clip on
  NVMe-bound hosts.  Falls back to the OS temp directory when `/dev/shm` is
  absent, unwritable, or has insufficient free space.  Pass `--scratch-dir`
  to override auto-selection.
