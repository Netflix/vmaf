# CLAUDE.md — VMAF Fork (Lusoris)

High-signal orientation for Claude Code sessions opened in this repo.
Non-Claude agents: see [AGENTS.md](AGENTS.md) (same content, tool-agnostic).

## 1. What this repo is

- Fork of [Netflix/vmaf](https://github.com/Netflix/vmaf) — perceptual video quality assessment.
- Additions on top of upstream:
  - SYCL / CUDA / HIP GPU backends (runtime-selected).
  - AVX2 / AVX-512 / NEON SIMD paths.
  - Full-precision `--precision` CLI flag (default `%.17g`, IEEE-754 round-trip lossless).
  - Tiny-AI (ONNX Runtime) model surface — see `ai/`, `libvmaf/src/dnn/`.
  - MCP server — see `mcp-server/vmaf-mcp/`.
- License: BSD-3-Clause-Plus-Patent (upstream license preserved). See [LICENSE](LICENSE).
- Default branch on this fork: `master`. Upstream is tracked as remote `upstream`.

## 2. How to build

Meson + Ninja (NOT CMake).

```bash
# CPU only (fastest build, no GPU deps required)
meson setup build -Denable_cuda=false -Denable_sycl=false
ninja -C build

# With CUDA (requires /opt/cuda + nvcc)
meson setup build -Denable_cuda=true -Denable_sycl=false
ninja -C build

# With SYCL (requires oneAPI / icpx)
meson setup build -Denable_cuda=false -Denable_sycl=true
ninja -C build

# Full (both backends)
meson setup build -Denable_cuda=true -Denable_sycl=true
ninja -C build
```

Shortcut: `/build-vmaf --backend=cpu|cuda|sycl|all` (skill).

## 3. How to test

```bash
meson test -C build                     # all unit tests
meson test -C build --suite=fast        # fast subset (pre-push gate)
make test                               # full suite + ASan + UBSan
make test-netflix-golden                # the 3 Netflix CPU golden-data tests (see §8)
```

## 4. How to lint

```bash
make lint        # clang-tidy + cppcheck + iwyu + ruff + semgrep
make format      # clang-format + black + isort (write)
make format-check  # same, no writes (pre-commit / CI)
```

Skills: `/format-all`, `/lint-all`.

## 5. Where the code is

```
libvmaf/
  src/                        # C sources (metric engine, feature extractors)
    feature/                  # per-feature CPU implementations
      x86/                    # AVX2 / AVX-512 SIMD paths
      arm64/                  # NEON SIMD paths
      cuda/                   # CUDA feature kernels
      sycl/                   # SYCL feature kernels
    cuda/                     # CUDA backend runtime (picture, dispatch)
    sycl/                     # SYCL backend runtime (queue, USM, dmabuf)
    dnn/                      # ONNX Runtime integration (tiny AI, Phase 3k)
  include/libvmaf/            # public C API headers
  tools/                      # CLI: vmaf.c, vmaf_bench.c, cli_parse.c
  test/                       # C unit tests
python/vmaf/                  # Python bindings + training harness
python/test/                  # Python tests — contains Netflix golden assertions
ai/                           # Tiny-AI training (Python / PyTorch + Lightning)
mcp-server/vmaf-mcp/          # MCP JSON-RPC server (Python)
model/                        # .json / .pkl / .onnx VMAF models
testdata/                     # YUV fixtures + benchmark JSONs (fork-added)
docs/                         # architecture + best-practice docs
resource/doc/                 # upstream docs (kept in sync when possible)
.claude/                      # Claude Code config (skills, agents, hooks)
.workingdir2/                 # planning dossier (read-only at runtime)
```

## 6. Coding standards

**Read [docs/principles.md](docs/principles.md) before writing C.** Summary:

- NASA/JPL Power of 10 (`.clang-tidy` enforces)
- JPL Institutional Coding Standard for C — applicable subset
- SEI CERT C & CERT C++ — mandatory
- MISRA C:2012 — informative only
- Style: K&R, 4-space, 100-char columns
- Banned functions: `gets`, `strcpy`, `strcat`, `sprintf`, `strtok`, `atoi`, `atof`,
  `rand`, `system` — see `docs/principles.md §1.2 rule 30`.
- Every non-void return value is checked or explicitly `(void)`-discarded.

## 7. When adding new functionality

| Task                                | Skill invocation                              |
|-------------------------------------|-----------------------------------------------|
| New GPU backend (hip, vulkan, metal)| `/add-gpu-backend <name>`                     |
| New SIMD path                       | `/add-simd-path <isa> <feature>`              |
| New feature extractor               | `/add-feature-extractor <name>`               |
| Register a new model JSON           | `/add-model <path>`                           |
| Profile a hot path                  | `/profile-hotpath <backend> <feature>`        |
| Bisect a regression                 | `/bisect-regression`                          |
| Cross-backend numeric diff          | `/cross-backend-diff`                         |
| Port an upstream commit             | `/port-upstream-commit <sha>`                 |
| Sync with upstream master           | `/sync-upstream`                              |
| Regenerate test snapshots           | `/regen-snapshots` (justification required)   |

## 8. Netflix golden-data gate (do not modify)

The fork preserves Netflix's 3 canonical CPU reference test pairs as the source of truth
for VMAF numerical correctness:

1. Normal: `src01_hrc00_576x324.yuv` ↔ `src01_hrc01_576x324.yuv`
2. Checkerboard (1-px shift): `checkerboard_1920_1080_10_3_0_0.yuv` ↔ `..._1_0.yuv`
3. Checkerboard (10-px shift): `checkerboard_1920_1080_10_3_0_0.yuv` ↔ `..._10_0.yuv`

YUVs: `python/test/resource/yuv/`. Golden assertions: `python/test/quality_runner_test.py`,
`vmafexec_test.py`, `vmafexec_feature_extractor_test.py`, `feature_extractor_test.py`,
`result_test.py` (hardcoded `assertAlmostEqual` values). **Never modify these assertions.**
They run in CI as a required status check. Fork-added tests go in separate files.

## 9. Snapshot regeneration

Fork-added snapshot JSONs under `testdata/scores_cpu_*.json` and
`testdata/netflix_benchmark_results.json` are GPU/SIMD numerical snapshots, NOT Netflix
golden data. Regenerate intentionally via `/regen-snapshots`; include the justification
in the commit message.

## 10. Upstream sync

```bash
git remote add upstream https://github.com/Netflix/vmaf.git  # once
/sync-upstream                                               # creates PR
```

Use `/port-upstream-commit <sha>` for single-commit cherry-picks.

## 11. Release

Automated by `release-please` on pushes to `master`. Version scheme: `v3.x.y-lusoris.N`
(tracks Netflix upstream version + fork suffix). Signing is keyless via Sigstore.
Use `/prep-release` to dry-run locally before merging a release PR.

## 12. Hard rules for every session

1. **Never** modify Netflix golden assertions (§8).
2. **Never** `git push --force` to `master`.
3. **Never** commit to `master` directly — branch + PR, merge via squash or ff only.
4. **Never** skip `make lint` before pushing.
5. **Never** commit benchmark output files (`testdata/netflix_benchmark_results.json`
   is usually noise from an ad-hoc run; stash it unless the run is formal).
6. **Every** commit message is Conventional Commits (`type(scope): subject`). Enforced
   by the `commit-msg` git hook.
7. **Every** new `.c` / `.h` / `.cpp` / `.cu` starts with the license header. Use
   `Copyright 2024-2026 Lusoris and Claude (Anthropic)` for wholly-new files, Netflix
   header for upstream-touched files.
