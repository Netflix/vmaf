# VMAF Fork — Engineering Principles

This document defines the non-negotiable standards for code merged to `master`. Every
requirement here is codified in one of: `.clang-tidy`, `.cppcheck-suppressions.txt`,
`.semgrep.yml`, `.pre-commit-config.yaml`, `.github/workflows/{lint,security,supply-chain}.yml`.
If a rule here is not yet codified in tooling, it is tracked as an OPEN item in
[.workingdir2/OPEN.md](../.workingdir2/OPEN.md).

## 1. Coding

### 1.1 NASA/JPL "Power of 10" rules (adapted for C)

1. **Simple control flow.** No `goto`, no `setjmp`/`longjmp`, no recursion (static or dynamic).
2. **Bounded loops.** Every loop has a statically-verifiable upper bound on iterations.
3. **No dynamic allocation in hot paths.** `malloc`/`calloc`/`realloc` are permitted at
   initialization only; frame-loop code paths are allocation-free.
4. **Short functions.** Function bodies ≤ 60 lines (a single printed page). Enforced by
   `readability-function-size` in clang-tidy with `LineThreshold: 60`.
5. **Assertion density.** ≥ 2 runtime assertions per function on average. Use `assert()`
   for invariants; in hot loops where `assert()` cost matters, use `VMAF_ASSERT_DEBUG()`
   which compiles away in release.
6. **Minimal scope.** Declare variables at the smallest possible scope.
7. **Check return values.** All non-void function returns must be checked or explicitly
   discarded with `(void)`. Enforced by `cert-err33-c`.
8. **Preprocessor restraint.** Only header inclusion (`#include`) and simple macros. No
   token pasting, no recursive macros, no conditional compilation beyond platform/feature
   toggles established at build-system level.
9. **Pointer restraint.** Single level of dereferencing in most cases. No function
   pointers except where essential for dispatch (feature extractor registry, SIMD
   runtime selection).
10. **Max strictness.** Compile with `-Wall -Wextra -Wpedantic -Werror`. Static analyzers
    (clang-tidy, cppcheck, scan-build) must finish with zero findings at enabled levels.

### 1.2 JPL Institutional Coding Standard for C — applicable subset

The JPL-C-STD is the 31-rule superset of Power of 10. The following rules extend 1.1 and
are additionally enforced:

- **Rule 11.** Place `#include` directives only at the top of a file.
- **Rule 12.** Do not place executable code inside a `#include`d file.
- **Rule 13.** Do not use reserved names (names beginning with `_` followed by uppercase
  letter, or `__`).
- **Rule 14.** All type casts must be explicit.
- **Rule 15.** The evaluation order of operands must not matter. No `i++` or `++i` used
  within a larger expression; split into separate statements.
- **Rule 16.** No assignments inside expressions.
- **Rule 17.** All variables must be initialized before use.
- **Rule 18.** All global variables must have a single declaration, and it must be in the
  file that owns them.
- **Rule 19.** All non-trivial types used in a function's public interface must be
  defined in a header.
- **Rule 20.** Use `const` aggressively. Anything not modified is `const`.
- **Rule 21.** Use `static` on every symbol not referenced outside its translation unit.
- **Rule 22.** No side effects in conditional expressions.
- **Rule 23.** All switch statements must have a default case.
- **Rule 24.** No fall-through in switch statements except where explicitly marked with
  `/* FALLTHROUGH */` or `__attribute__((fallthrough))`.
- **Rule 25.** Functions must have a single exit point where practical. Early returns are
  permitted for validation at function entry.
- **Rule 26.** Use of pointer arithmetic is restricted to array traversal within bounded
  loops.
- **Rule 27.** No use of `<setjmp.h>`, `<signal.h>` handlers that do real work, or any
  non-async-signal-safe function inside a signal handler.
- **Rule 28.** No variadic functions except standard `printf`/`scanf` family.
- **Rule 29.** No use of `<stdarg.h>` macros outside of printf/scanf wrappers.
- **Rule 30.** Banned functions: `gets`, `strcpy`, `strcat`, `sprintf`, `strtok`
  (non-reentrant), `atoi`, `atol`, `atof` (no error reporting), `rand` (non-cryptographic
  and non-reproducible), `system` (outside build scripts). Use: `fgets`, `strncpy` (with
  explicit null-termination) / `snprintf` / `strlcpy`, `snprintf`, `strtok_r`, `strtol`
  + errno checking, `arc4random` or a seeded `xorshift`, direct process calls with arg arrays.
- **Rule 31.** All headers must be self-contained (include what they need) and
  include-guard protected.

### 1.3 SEI CERT C & CERT C++

Full compliance required for:

- **INT (Integers)** — no signed overflow, always check `size_t` arithmetic, narrow casts
- **STR (Strings)** — no off-by-one, always check buffer bounds
- **MEM (Memory management)** — every `malloc` has a matching `free`; no double-free; no
  use-after-free (ASan enforced)
- **FIO (I/O)** — check every `fopen`, `fread`, `fwrite`, `fclose` return
- **EXP (Expressions)** — no undefined behavior; no sequence-point violations
- **CON (Concurrency)** — use atomics correctly (`_Atomic`); no data races (TSan nightly)
- **ENV (Environment)** — never trust `getenv` input without validation

Enforcement: `cert-*` checks in `.clang-tidy` all enabled. The noisy
`clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling` subset
(Microsoft C11 `_s` functions) is explicitly disabled — it does not map to
any portable POSIX API.

### 1.4 MISRA C:2012 (informative subset)

Applied where it does not conflict with existing libvmaf conventions:

- **Rule 8.5** (one external declaration per identifier in a single file)
- **Rule 10.1–10.8** (essential type rules — stricter than standard C integer promotions)
- **Rule 17.7** (function return values must be used or explicitly discarded) — matches
  Power of 10 #7
- **Rule 21.x** (banned functions) — matches JPL rule 30

Not enforced as PR-blocking; informational in review.

### 1.5 Style

- **C:** Existing libvmaf conventions preserved (K&R braces, 4-space indent, 100-char
  columns). Codified in [.clang-format](../.clang-format).
- **C++ (SYCL code):** same as C where applicable; RAII encouraged for queue/context
  wrappers; no exceptions in hot paths.
- **Python:** PEP 8 + black (line-length 100) + isort + ruff. Codified in
  [pyproject.toml](../pyproject.toml).
- **CUDA (`.cu`):** follows C style; kernel names `kernel_*`; device helpers `device_*`.
- **Shell:** shfmt + shellcheck; `#!/usr/bin/env bash`; `set -euo pipefail`.

## 2. Security

See [SECURITY.md](../SECURITY.md) for reporting policy.

### 2.1 Input validation at boundaries

- All CLI inputs validated at parse time (see `cli_parse.c`).
- All public libvmaf API entry points validate pointer non-null + struct version.
- Video frame dimensions bounded; no negative or zero-size frames accepted.
- File paths validated for NULL and checked with `access()` before open.

### 2.2 Memory safety

- **ASan + UBSan** on every PR (debug builds).
- **TSan** on nightly cron.
- **valgrind memcheck** on release-candidate audit.
- `_FORTIFY_SOURCE=3` in release builds.
- Stack protector `-fstack-protector-strong`.
- PIE (`-fPIE`, `-pie`) for all binaries.
- RELRO + BIND_NOW: `-Wl,-z,relro,-z,now`.

### 2.3 Banned functions

See JPL rule 30 above — enforced by `.semgrep.yml` custom rules +
`bugprone-unsafe-functions` in clang-tidy.

### 2.4 Dependency auditing

- **trivy** scans Dockerfile + filesystem for known CVEs on every PR.
- **pip-audit** for Python dependencies.
- **OSV** scanner enabled in CodeQL config.
- Dependabot opens weekly PRs for github-actions / pip / docker.

## 3. Quality gates (required for PR merge to master)

- ✅ `make lint` — zero clang-tidy / cppcheck / ruff / black findings
- ✅ `make test` — all unit + integration tests pass with ASan + UBSan
- ✅ `make sec` — zero gitleaks / semgrep critical / bandit high / trivy critical findings
- ✅ CodeQL — zero security-and-quality issues at `security-extended` suite
- ✅ Conventional commit messages (enforced by commit-msg hook)
- ✅ CI matrix green on Linux/macOS/Windows
- ✅ Netflix source-of-truth golden tests pass (CPU, 3 pairs: 1 normal + 2 checkerboard)
- ✅ Coverage ≥ 70% overall, ≥ 85% for security-critical code (validation, parsing,
  crypto-adjacent)

### 3.1 Netflix golden-data gate

The fork preserves the three Netflix-authored reference test pairs as the canonical
ground-truth gate for VMAF numerical correctness (CPU only):

| # | Type          | Reference                                      | Distorted                                      |
|---|---------------|------------------------------------------------|------------------------------------------------|
| 1 | normal        | `src01_hrc00_576x324.yuv`                      | `src01_hrc01_576x324.yuv`                      |
| 2 | checkerboard  | `checkerboard_1920_1080_10_3_0_0.yuv`          | `checkerboard_1920_1080_10_3_1_0.yuv` (1-px)   |
| 3 | checkerboard  | `checkerboard_1920_1080_10_3_0_0.yuv`          | `checkerboard_1920_1080_10_3_10_0.yuv` (10-px) |

YUVs live in [python/test/resource/yuv/](../python/test/resource/yuv/). Golden expected
scores are hardcoded `assertAlmostEqual` assertions in `python/test/` (primarily
`quality_runner_test.py`, `vmafexec_test.py`, `vmafexec_feature_extractor_test.py`,
`feature_extractor_test.py`, `result_test.py`). **These assertions are never modified.**
They run in CI as a required status check on every PR. They are not run as a pre-commit
hook because their runtime is longer than the acceptable pre-commit latency budget.

Fork-added tests (SYCL, CUDA, SIMD snapshots, performance benchmarks) live in separate
files and directories, and must not modify or override Netflix golden behavior.

## 4. Deterministic builds

- `SOURCE_DATE_EPOCH` respected throughout the build
- All build-time random data (tempdir names, etc.) is seeded from `SOURCE_DATE_EPOCH`
- Meson pinned to a specific minor version via `dev-setup` scripts
- Compiler/linker flags logged and archived per release

## 5. Supply chain

See [phases/03-framework/3c-supply-chain.md](../.workingdir2/phases/03-framework/3c-supply-chain.md):

- **SLSA Level 3** provenance attestation on every release
- **CycloneDX + SPDX SBOMs** on every release
- **Keyless cosign** signatures via Sigstore / GitHub OIDC
- **Transparency log** (Rekor) makes signatures publicly verifiable

## 6. Compliance targets

- **OpenSSF Best Practices** — Gold badge goal
- **OpenSSF Scorecard** — ≥ 8.5 score required on master
- **OWASP ASVS** — Level 2 applicable controls (auth/authz not applicable — library)
- **NIST SSDF** — PW.4 (design), PW.5 (implement), PW.7 (review), PW.9 (archive) aligned

## 7. Non-goals

We explicitly do **not** pursue:

- Full MISRA C compliance (too restrictive for numerical code — informative subset only)
- CII Best Practices "Passing" badge deadline commitment (will earn it when it earns
  itself)
- FIPS 140 (no cryptography in this library)
- DO-178C (aerospace software — out of scope)

## 8. Revising this document

Changes to this document require:

1. A dedicated PR titled `docs(principles): <summary>`
2. Two approvals from CODEOWNERS
3. A corresponding change to tooling if the standard is newly enforceable in CI
4. Linked rationale — either a design doc, a referenced security incident, or a
   standards-body update
