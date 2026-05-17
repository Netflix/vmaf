# Research-0034 — CI pipeline audit (2026-05-01)

| Field      | Value                                                                  |
| ---------- | ---------------------------------------------------------------------- |
| **Date**   | 2026-05-01                                                             |
| **Status** | Audit; top items consumed by PRs #235 (FFmpeg+Vulkan lane) + #236 (pre-commit parity) |
| **Tags**   | ci, devx, pre-commit, ffmpeg, vulkan, mcp, hip                         |

## TL;DR — top 3 highest-impact changes

1. **FFmpeg CI is missing Vulkan, confirmed.**
   `ffmpeg-integration.yml` builds two `--enable-libvmaf` lanes (gcc,
   clang) and one SYCL lane, but **zero** lanes pass
   `-Denable_vulkan=enabled` to libvmaf or wire patch
   `0006-libvmaf-add-libvmaf-vulkan-filter.patch` against a
   Vulkan-capable `vf_libvmaf.c`. PR #234's `VK_NO_PROTOTYPES`
   regression would not have been caught here. Closed by **PR #235**
   (adds `ffmpeg-vulkan` lavapipe lane).

2. **Pre-commit ↔ CI parity gap is real and cheap to close.**
   Local pre-commit is missing 4 CI gates that all run in <10 s on a
   single-file diff: `assertion-density.sh`, `mypy`, `semgrep --error`,
   and `ffmpeg-patches apply --check`. Closed by **PR #236**.

3. ~~**HIP backend has zero CI coverage**~~ — **false positive** (the
   audit agent worked off a stale worktree). Verified post-audit:
   `libvmaf/src/hip/` ships 7 files on master (common.{c,h},
   dispatch_strategy.{c,h}, picture_hip.{c,h}, meson.build) plus 4
   kernel stubs at `libvmaf/src/feature/hip/`, all landed by PR #200
   (T7-10), and the `Build — Ubuntu HIP (T7-10 scaffold)` CI matrix
   row already exercises them with `-Denable_hip=true`. Status: not
   affecting the fork.

## Deliverable 1 — CI duplicate / low-value scan

13 workflow files; 30+ jobs total. Wall-time figures are estimates.

### Net cuts recommended (saves ~25-35 min/PR)

| Workflow | Job | Recommendation |
| --- | --- | --- |
| `tests-and-quality-gates.yml` | `cross-backend` | drop dead `if: false` block (ADR-0024 already gated by `vulkan-vif-cross-backend`) |
| `tests-and-quality-gates.yml` | `vulkan-vif-arc-nightly` | drop until `vmaf-arc` self-hosted runner exists; carry recipe in `.github/workflows-disabled/` if wanted |
| `tests-and-quality-gates.yml` | `sanitizers[thread]` (PR-time TSan) | drop; already runs nightly. Keep ASan + UBSan |
| `tests-and-quality-gates.yml` | per-feature VIF steps in `vulkan-vif-cross-backend` | replaced by the parity-matrix gate (`vulkan-parity-matrix-gate`) — drop the 14 hand-rolled `cross_backend_vif_diff.py` steps |
| `lint-and-format.yml` | `python-lint` ruff/black/isort steps | duplicate of pre-commit hooks; reduce job to **mypy only** |
| `lint-and-format.yml` | `shellcheck` lane | duplicate of pre-commit `shellcheck` + `shfmt` hooks; drop |
| `libvmaf-build-matrix.yml` | `Build — Ubuntu SYCL Static` | dynamic-link SYCL leg already proves the icpx + meson contract; static is a packaging exercise that doesn't catch new bugs |
| `docker-image.yml` | `docker` | demote to advisory; no smoke test of resulting image |

### Keep, no change

Critical correctness/security gates (Netflix golden, sanitizers,
CodeQL, gitleaks, semgrep, clang-tidy, cppcheck, doc-substance,
ADR-0108), build matrix variants for cross-platform coverage, the
nightly TSan/clang-tidy-full pair.

## Deliverable 2 — CI gap scan

| Surface | Current CI lane | Proposed lane | Priority |
| --- | --- | --- | --- |
| **FFmpeg + Vulkan integration** | none | `ffmpeg-vulkan` matrix entry: install `mesa-vulkan-drivers libvulkan-dev glslc lavapipe`, build vmaf with `-Denable_vulkan=enabled`, apply patches, configure FFmpeg `--enable-libvmaf-vulkan`, build `vf_libvmaf_vulkan.o`, `nm` symbol assertion | **urgent — closed by PR #235** |
| **`ffmpeg-patches/` apply check vs pinned `n8.1`** | partial (covered by `ffmpeg-integration.yml` build) | 2-min job: `git clone -b n8.1 --depth=1 FFmpeg && for p in ffmpeg-patches/000*-*.patch; do git -C ffmpeg apply --check "$p"; done`. Catches CLAUDE §12 r14 drift fast | nice-to-have (covered locally by PR #236 hook) |
| ~~**HIP backend**~~ | ~~empty dir~~ → **already covered** | n/a — `Build — Ubuntu HIP (T7-10 scaffold)` lane in `libvmaf-build-matrix.yml` exercises the 7-file scaffold from PR #200 + 4 kernel stubs at `libvmaf/src/feature/hip/`. Audit agent's worktree was stale. **False positive.** | n/a (closed) |
| **MCP server smoke test** | `supply-chain.yml::mcp-build` builds the wheel; nothing runs `pytest mcp-server/vmaf-mcp/tests/` | `mcp-smoke` job: build with `-Denable_mcp=true`, `pip install -e mcp-server/vmaf-mcp[test]`, `pytest`. ~5 min | **urgent** (per CLAUDE §12 r10) |
| **Tiny-AI training scripts smoke** | `dnn` job runs `ai/tests/`; `ai/scripts/*.py` (parquet producers) have no smoke runs | `ai-scripts-smoke` step: dry-run `--help` + 1-frame `extract_full_features.py` against cached YUV. ~30 s | nice-to-have |
| **Cross-backend bit-exactness** beyond Vulkan | none usable on hosted runners | document gap in `docs/development/self-hosted-runner.md`; consider Cirun.io / BuildJet for managed GPU minutes. **No new lane** until runner exists | nice-to-have (blocked) |
| **Tiny-Model Registry validation** | ~~none~~ → **already exists** (verified post-audit) | n/a — `lint-and-format.yml::registry-validate` already calls `ai/scripts/validate_model_registry.py`, which does both schema and sha256 cross-check (lines 100-103 of that script). The audit's grep missed the job because the job name `registry-validate` doesn't contain "Tiny-Model" verbatim while the surface label does. **False positive.** | n/a (closed) |
| **Public C-API ABI** | none | weekly `abidiff` against last released `.so`. Out of scope for this audit | nice-to-have |

## Deliverable 3 — Pre-commit ↔ CI parity

### CI lint surface (what runs)

- `lint-and-format.yml::pre-commit`: `pre-commit run --all-files` (covers
  everything in `.pre-commit-config.yaml`).
- `python-lint`: ruff, black, isort, mypy (advisory).
- `shellcheck`: shellcheck + shfmt -d.
- `assertion-density`: `scripts/ci/assertion-density.sh` (Power-of-10
  §5).
- Standalone clang-tidy + cppcheck (build-required, not in scope for
  cheap pre-commit).

### Local pre-commit hooks (what runs locally)

- `pre-commit-hooks` standard set, `clang-format`, `black`, `isort`,
  `ruff-check`, `shfmt`, `shellcheck`, `gitleaks`,
  `conventional-pre-commit`, local `check-copyright`.

### Gap (closed by PR #236)

CI checks NOT in local pre-commit, all <10 s on a single-file diff:

1. `assertion-density.sh` — runs in CI; was not local.
2. `mypy` on `ai/` + `scripts/` — advisory in CI; was not local.
3. `semgrep --config=.semgrep.yml --error` — runs in CI; was not local.
4. `ffmpeg-patches apply --check` — was not run anywhere except inside
   the FFmpeg build job.

PR #236 adds these as `local` hooks (last three at `stages:
[pre-push]` so they don't slow micro-commits) and a new
`scripts/ci/ffmpeg-patches-check.sh`. Combined wall-clock <8 s on a
single-file diff.

### Pre-existing violations PR #236 surfaced (not fixed by it)

1. `mypy` finds 4 errors in `ai/scripts/phase3_subset_sweep.py`
   (missing annotations, unused `# type: ignore`, pandas-stubs,
   generic type args). Matches CI's advisory posture.
2. **4 of 6 `ffmpeg-patches/` no longer apply cleanly against
   `release/8.1`** (`0003`–`0006`). The patch series needs a refresh
   — exactly the kind of rebase debt the new gate exists to catch.
   **Open** — see backlog item below.
3. `semgrep` exits 2 only on `--all-files` (700+ explicit paths);
   fine on real per-file diffs.

## Backlog candidates surfaced by this audit

| ID | Title | Effort |
| --- | --- | --- |
| `T7-CI-DEDUP` | Drop dead jobs (`cross-backend`, `vulkan-vif-arc-nightly`), drop PR-time TSan, merge per-feature VIF steps into matrix gate, demote docker-image to advisory | S (~3h) |
| `T7-MCP-SMOKE-CI` | Add `mcp-smoke` job to `tests-and-quality-gates.yml` per CLAUDE §12 r10 | S (~3h) |
| ~~`T7-REGISTRY-CI`~~ | ~~Add 30-s JSON-schema + sha256 validate job for `model/tiny/registry.json`~~ — **closed as not-affecting-the-fork**: `lint-and-format.yml::registry-validate` already does this. | n/a |
| ~~`T7-HIP-STATUS`~~ | ~~Clarify HIP backend~~ — **closed as not-affecting-the-fork**: PR #200 already shipped the scaffold + CI matrix row. | n/a |
| `T7-FFMPEG-PATCH-REFRESH` | Refresh `ffmpeg-patches/0003`–`0006` against `release/8.1` (caught by PR #236's new local gate) | S–M |

## What this audit closed (in flight)

- **PR #235** — FFmpeg + Vulkan CI lane (lavapipe). Catches PR #234's
  `-DVK_NO_PROTOTYPES` class.
- **PR #236** — pre-commit hooks: assertion-density, mypy-local,
  semgrep-local, ffmpeg-patches-apply-check.
