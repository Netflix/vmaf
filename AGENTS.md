# AGENTS.md — VMAF Fork (Lusoris)

Orientation for any coding agent (Cursor, Copilot, Aider, Continue, Cody, Codeium, etc.)
opened in this repo. For Claude Code–specific tooling (skills, hooks), see
[CLAUDE.md](CLAUDE.md) — the same operational content with Claude-specific commands.

## 1. What this repo is

- Fork of [Netflix/vmaf](https://github.com/Netflix/vmaf) — perceptual video quality
  assessment.
- Additions over upstream: SYCL / CUDA / HIP GPU backends, AVX2/AVX-512/NEON SIMD,
  a full-precision CLI flag (default `%.17g`), tiny-AI surface (ONNX Runtime), MCP server.
- License: BSD-3-Clause-Plus-Patent (upstream license preserved — see [LICENSE](LICENSE)).
- Default branch: `master`. Upstream tracked as remote `upstream`.

## 2. Build

Meson + Ninja.

```
meson setup build [-Denable_cuda=true|false] [-Denable_sycl=true|false]
ninja -C build
```

## 3. Test

```
meson test -C build                     # unit tests
meson test -C build --suite=fast        # fast subset
make test                               # full + ASan + UBSan
make test-netflix-golden                # Netflix CPU golden-data gate (see §8)
```

## 4. Lint / format

```
make lint          # clang-tidy + cppcheck + iwyu + ruff + semgrep
make format        # clang-format + black + isort (writes)
make format-check  # dry-run (CI / pre-commit)
```

## 5. Repository layout

See [CLAUDE.md §5](CLAUDE.md) — identical. Briefly:

- `libvmaf/src/` — C engine
- `libvmaf/src/{cuda,sycl,dnn}/` — GPU / DNN backends
- `libvmaf/src/feature/{x86,arm64,cuda,sycl}/` — per-platform feature implementations
- `libvmaf/tools/` — `vmaf` CLI + `vmaf_bench`
- `python/vmaf/` + `python/test/` — Python bindings + tests (golden-data here)
- `ai/` — PyTorch tiny-model training
- `mcp-server/` — MCP JSON-RPC server
- `model/` — VMAF models (.json / .pkl / .onnx)
- `testdata/` — fork-added YUV + snapshot JSONs
- `docs/principles.md` — canonical engineering standards

## 6. Coding standards

All C code must conform to:

- NASA/JPL Power of 10 (enforced by `.clang-tidy`)
- JPL Institutional Coding Standard for C (applicable subset)
- SEI CERT C & CERT C++ (mandatory)
- MISRA C:2012 (informative subset)

Banned functions, pointer/loop/alloc restrictions, and the exact list of `.clang-tidy`
checks that codify them are in [docs/principles.md](docs/principles.md).

Style: K&R, 4-space indent, 100-char line budget (see `.clang-format`).

## 7. Conventional entry points for common tasks

These operational workflows are each codified as a Claude skill under `.claude/skills/`.
Agents without slash-command routing should read the corresponding `SKILL.md` and follow
the steps manually.

| Task                         | See                                               |
|------------------------------|---------------------------------------------------|
| Build against a backend      | `.claude/skills/build-vmaf/SKILL.md`              |
| Add new GPU backend          | `.claude/skills/add-gpu-backend/SKILL.md`         |
| Add SIMD path                | `.claude/skills/add-simd-path/SKILL.md`           |
| Add feature extractor        | `.claude/skills/add-feature-extractor/SKILL.md`   |
| Add a model                  | `.claude/skills/add-model/SKILL.md`               |
| Cross-backend numeric diff   | `.claude/skills/cross-backend-diff/SKILL.md`      |
| Profile a hot path           | `.claude/skills/profile-hotpath/SKILL.md`         |
| Bisect a regression          | `.claude/skills/bisect-regression/SKILL.md`       |
| Port upstream commit         | `.claude/skills/port-upstream-commit/SKILL.md`    |
| Sync with upstream master    | `.claude/skills/sync-upstream/SKILL.md`           |
| Regenerate test snapshots    | `.claude/skills/regen-snapshots/SKILL.md`         |
| Release dry-run              | `.claude/skills/prep-release/SKILL.md`            |

## 8. Netflix golden-data gate — never modify

The fork preserves three Netflix-authored CPU reference test pairs as the
numerical-correctness ground truth:

1. **Normal** — `src01_hrc00_576x324.yuv` ↔ `src01_hrc01_576x324.yuv`
2. **Checkerboard 1-px** — `checkerboard_1920_1080_10_3_0_0.yuv` ↔ `..._1_0.yuv`
3. **Checkerboard 10-px** — `checkerboard_1920_1080_10_3_0_0.yuv` ↔ `..._10_0.yuv`

YUV files: `python/test/resource/yuv/`. Golden-score assertions are hardcoded as
`assertAlmostEqual(...)` calls in `python/test/`. **These assertions are never modified
by any PR.** They run in CI as a required status check. Fork-added tests live in separate
files and directories.

## 9. Snapshot regeneration

`testdata/scores_cpu_*.json` and `testdata/netflix_benchmark_results.json` are
fork-added GPU/SIMD snapshots — NOT Netflix golden data. If an intentional numerical
change is needed, regenerate them and include the justification in the commit message.
Non-justified changes in these files will be rejected in review.

## 10. Upstream sync

`git remote add upstream https://github.com/Netflix/vmaf.git` (once), then follow
`.claude/skills/sync-upstream/SKILL.md` to open a sync PR. Individual commits go
through `port-upstream-commit`.

## 11. Release

Release-please is triggered by pushes to `master`. Version scheme `v3.x.y-lusoris.N`,
tracking upstream version + a fork suffix. Signing is keyless via Sigstore / GitHub OIDC.

## 12. Hard rules

1. Never modify Netflix golden-score assertions (§8).
2. Never `git push --force` to `master`.
3. Never commit directly to `master` — PR with squash or fast-forward only.
4. Never merge without `make lint` + `make test` green locally.
5. Every commit message is Conventional Commits (`type(scope): subject`) — enforced by
   the `commit-msg` git hook.
6. Every new `.c` / `.h` / `.cpp` / `.cu` file starts with the applicable license
   header (wholly-new fork files: `Copyright 2024-2026 Lusoris and Claude (Anthropic)`;
   files touching Netflix code: Netflix header preserved).
