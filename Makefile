# Use bash for shell
SHELL := /bin/bash

# Path and environment setup
VENV := .venv
VIRTUAL_ENV_PATH := $(VENV)/bin

# Build tools configured in the virtual environment
PYTHON_INTERPRETER := python3
VENV_PIP := $(VIRTUAL_ENV_PATH)/pip
VENV_PYTHON := $(VIRTUAL_ENV_PATH)/python
MESON := $(VIRTUAL_ENV_PATH)/meson
MESON_SETUP := $(MESON) setup
NINJA := $(VIRTUAL_ENV_PATH)/ninja

# Build types and options
BUILDTYPE_RELEASE := --buildtype release
BUILDTYPE_DEBUG := --buildtype debug
ENABLE_FLOAT := -Denable_float=true
ENABLE_NVCC :=	true
ENABLE_CUDA := -Denable_cuda=true -Denable_nvcc=$(ENABLE_NVCC)

# Directories
LIBVMAF_DIR := libvmaf
BUILD_DIR := $(LIBVMAF_DIR)/build
DEBUG_DIR := $(LIBVMAF_DIR)/debug

.PHONY: default all debug build install cythonize clean distclean cythonize-deps

default: build

all: build debug install test cythonize

$(BUILD_DIR): $(MESON) $(NINJA)
	PATH="$(VENV)/bin:$$PATH" $(MESON_SETUP) $(BUILD_DIR) $(LIBVMAF_DIR) $(BUILDTYPE_RELEASE) $(ENABLE_FLOAT) $(ENABLE_CUDA)

$(DEBUG_DIR): $(MESON) $(NINJA)
	PATH="$(VENV)/bin:$$PATH" $(MESON_SETUP) $(DEBUG_DIR) $(LIBVMAF_DIR) $(BUILDTYPE_DEBUG) $(ENABLE_FLOAT) $(ENABLE_CUDA)

cythonize: cythonize-deps
	pushd python && ../$(VENV_PYTHON) setup.py build_ext --build-lib . && popd || exit 1

build: $(BUILD_DIR) $(NINJA)
	PATH="$(VENV)/bin:$$PATH" $(NINJA) -vC $(BUILD_DIR)

test: build $(NINJA)
	PATH="$(VENV)/bin:$$PATH" $(NINJA) -vC $(BUILD_DIR) test

debug: $(DEBUG_DIR) $(NINJA)
	PATH="$(VENV)/bin:$$PATH" $(NINJA) -vC $(DEBUG_DIR)

install: $(BUILD_DIR) $(NINJA)
	PATH="$(VENV)/bin:$$PATH" $(NINJA) -vC $(BUILD_DIR) install

clean:
	rm -rf $(BUILD_DIR) $(DEBUG_DIR)
	rm -f python/vmaf/core/adm_dwt2_cy.c*

distclean: clean
	rm -rf $(VENV)

# Set up or rebuild virtual environment
$(VENV_PIP):
	@echo "Setting up the virtual environment..."
	$(PYTHON_INTERPRETER) -m venv $(VENV) || { echo "Failed to create virtual environment"; exit 1; }
	$(VENV_PIP) install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }
	@echo "Virtual environment setup complete."

$(MESON): $(VENV_PIP)
	$(VENV_PIP) install meson || { echo "Failed to install meson"; exit 1; }

$(NINJA): $(VENV_PIP)
	$(VENV_PIP) install ninja || { echo "Failed to install ninja"; exit 1; }

cythonize-deps: $(VENV_PIP)
	$(VENV_PIP) install setuptools cython numpy || { echo "Failed to install dependencies"; exit 1; }

# ============================================================================
# Fork-specific targets (lusoris). The upstream targets above are preserved as-is.
# ============================================================================

.PHONY: lint lint-c lint-py lint-sh format format-check sec sbom \
        test-netflix-golden test-sanitizers test-fast hooks-install help \
        coverage coverage-html coverage-check assertion-density pr-check

# Top-level lint — runs every analyzer we own. Uses the meson compile_commands.json.
lint: lint-c lint-py lint-sh docs-fragments-check
	@echo "=== all lints passed ==="

# Fragment-tree drift check (ADR-0221). Verifies CHANGELOG.md and
# docs/adr/README.md are in sync with their per-PR fragment trees.
docs-fragments-check:
	@echo "--- changelog.d/ vs CHANGELOG.md ---"
	@bash scripts/release/concat-changelog-fragments.sh --check
	@echo "--- docs/adr/_index_fragments/ vs docs/adr/README.md ---"
	@bash scripts/docs/concat-adr-index.sh --check

# Regenerate consolidated outputs from fragments (ADR-0221).
docs-fragments-write:
	@bash scripts/release/concat-changelog-fragments.sh --write
	@bash scripts/docs/concat-adr-index.sh --write

lint-c: $(BUILD_DIR)
	@command -v clang-tidy >/dev/null || { echo "clang-tidy not found; skipping"; exit 0; }
	@command -v cppcheck >/dev/null   || { echo "cppcheck not found; skipping"; exit 0; }
	@echo "--- clang-tidy ---"
	@FILES=$$(git ls-files 'libvmaf/src/**/*.c' 'libvmaf/src/**/*.cpp' 'libvmaf/tools/*.c' \
	         | grep -v '^subprojects/'); \
	 clang-tidy -p $(BUILD_DIR) --quiet $$FILES
	@echo "--- cppcheck ---"
	cppcheck --enable=all --inline-suppr \
	         --suppressions-list=.cppcheck-suppressions.txt \
	         --project=$(BUILD_DIR)/compile_commands.json \
	         --error-exitcode=1

lint-py:
	@command -v ruff >/dev/null || { echo "ruff not found; skipping"; exit 0; }
	ruff check python/ ai/ scripts/ 2>/dev/null || ruff check python/
	@command -v black >/dev/null && black --check python/ ai/ scripts/ 2>/dev/null || true
	@command -v mypy  >/dev/null && mypy ai/scripts/ ai/tests/ ai/train/ ai/lpips_export.py scripts/ 2>/dev/null || true

lint-sh:
	@command -v shellcheck >/dev/null || { echo "shellcheck not found; skipping"; exit 0; }
	shellcheck $$(git ls-files '*.sh')

# Formatters — writes changes.
format:
	@command -v clang-format >/dev/null && \
	 clang-format -i $$(git ls-files '*.c' '*.h' '*.cpp' '*.hpp' '*.cu' '*.cuh' \
	                   | grep -v '^subprojects/' | grep -v '^libvmaf/test/data/') || true
	@command -v black >/dev/null && black python/ ai/ scripts/ 2>/dev/null || true
	@command -v isort >/dev/null && isort python/ ai/ scripts/ 2>/dev/null || true
	@command -v shfmt >/dev/null && shfmt -w -i 2 -ci $$(git ls-files '*.sh') || true

# Formatters — check-only (CI gate, no writes).
format-check:
	@command -v clang-format >/dev/null && \
	 clang-format --dry-run --Werror \
	   $$(git ls-files '*.c' '*.h' '*.cpp' '*.hpp' '*.cu' '*.cuh' \
	      | grep -v '^subprojects/' | grep -v '^libvmaf/test/data/') || true
	@command -v black >/dev/null && black --check python/ ai/ scripts/ 2>/dev/null || true
	@command -v isort >/dev/null && isort --check-only python/ ai/ scripts/ 2>/dev/null || true
	@command -v shfmt >/dev/null && shfmt -d -i 2 -ci $$(git ls-files '*.sh') || true

# Security scan (semgrep custom + CERT-C + CWE Top 25).
sec:
	@command -v semgrep >/dev/null || { echo "semgrep not installed — see .semgrep.yml"; exit 1; }
	semgrep scan --config=.semgrep.yml \
	             --config=p/cwe-top-25 \
	             --config=p/cert-c-strict \
	             --error

# SBOM generation (Software Bill of Materials, both SPDX and CycloneDX).
sbom:
	@command -v syft >/dev/null || { echo "syft not installed"; exit 1; }
	@mkdir -p build/sbom
	syft . -o spdx-json=build/sbom/sbom.spdx.json
	syft . -o cyclonedx-json=build/sbom/sbom.cdx.json
	@echo "SBOM: build/sbom/sbom.{spdx,cdx}.json"

# Netflix CPU golden-data gate (D24) — the 3 test pairs that MUST pass.
# Runs the Python tests whose hardcoded CPU scores are the source of truth
# for VMAF numerical correctness.
test-netflix-golden: build
	@echo "=== Netflix CPU golden-data gate (D24) ==="
	PYTHONPATH=$(CURDIR)/python python3 -m pytest \
	    python/test/quality_runner_test.py \
	    python/test/feature_extractor_test.py \
	    python/test/vmafexec_test.py \
	    python/test/vmafexec_feature_extractor_test.py \
	    python/test/result_test.py \
	    -v -m "not slow" --tb=short

# Sanitizer build (ASan + UBSan) — used by CI and `/build-vmaf --sanitizers`.
test-sanitizers:
	@mkdir -p build-san
	meson setup build-san $(LIBVMAF_DIR) --buildtype=debug \
	    -Db_sanitize=address,undefined \
	    -Denable_cuda=false -Denable_sycl=false \
	    --reconfigure 2>/dev/null || \
	meson setup build-san $(LIBVMAF_DIR) --buildtype=debug \
	    -Db_sanitize=address,undefined \
	    -Denable_cuda=false -Denable_sycl=false
	ninja -C build-san
	meson test -C build-san --print-errorlogs

test-fast: build
	PATH="$(VENV)/bin:$$PATH" meson test -C $(BUILD_DIR) --suite=fast

# ============================================================================
# Coverage gate (docs/principles.md §3 — ≥70% overall, ≥85% security-critical)
# ============================================================================

COVERAGE_DIR := build-coverage
COVERAGE_MIN_OVERALL := 70
COVERAGE_MIN_CRITICAL := 85

# Build with gcov instrumentation, run tests, emit lcov report.
# Uses a dedicated build dir so normal `make build` isn't instrumented.
coverage:
	@command -v lcov >/dev/null || { echo "lcov not found — install lcov"; exit 1; }
	@command -v gcov >/dev/null || { echo "gcov not found — install gcc"; exit 1; }
	@mkdir -p $(COVERAGE_DIR)
	meson setup $(COVERAGE_DIR) $(LIBVMAF_DIR) --buildtype=debug -Db_coverage=true \
	    -Denable_cuda=false -Denable_sycl=false --reconfigure 2>/dev/null || \
	meson setup $(COVERAGE_DIR) $(LIBVMAF_DIR) --buildtype=debug -Db_coverage=true \
	    -Denable_cuda=false -Denable_sycl=false
	ninja -C $(COVERAGE_DIR)
	meson test -C $(COVERAGE_DIR) --print-errorlogs
	@echo "--- gathering coverage ---"
	lcov --capture --directory $(COVERAGE_DIR) --output-file $(COVERAGE_DIR)/coverage.info \
	     --ignore-errors mismatch,gcov,source --rc geninfo_unexecuted_blocks=1
	lcov --remove $(COVERAGE_DIR)/coverage.info \
	     '/usr/*' '*/subprojects/*' '*/test/*' '*/tests/*' \
	     --output-file $(COVERAGE_DIR)/coverage.filtered.info \
	     --ignore-errors unused
	lcov --list $(COVERAGE_DIR)/coverage.filtered.info | tee $(COVERAGE_DIR)/coverage.summary.txt

# Render HTML coverage report (open $(COVERAGE_DIR)/html/index.html).
coverage-html: coverage
	genhtml $(COVERAGE_DIR)/coverage.filtered.info \
	    --output-directory $(COVERAGE_DIR)/html \
	    --demangle-cpp --legend --title "libvmaf coverage"
	@echo "open $(COVERAGE_DIR)/html/index.html"

# Enforce the coverage thresholds from docs/principles.md §3.
# Overall: ≥70% line coverage. Security-critical (libvmaf/src/dnn/, opt.c,
# read_json_model.c): ≥85% line coverage.
coverage-check: coverage
	@scripts/ci/coverage-check.sh $(COVERAGE_DIR)/coverage.filtered.info \
	    $(COVERAGE_MIN_OVERALL) $(COVERAGE_MIN_CRITICAL)

# Power-of-10 rule 5 density check (≥2 asserts per function average across
# fork-added code). Warns on any non-trivial fork-added function with 0 asserts.
assertion-density:
	@scripts/ci/assertion-density.sh

# Install the pre-commit + pre-push git hooks.
#
# - pre-commit / commit-msg come from .pre-commit-config.yaml (including
#   the `agent-worktree-drift-guard` local hook; ADR-0332).
# - pre-push is the PR-body deliverables validator at
#   scripts/git-hooks/pre-push (mirrors the rule-enforcement.yml
#   deep-dive-checklist gate locally; ADR-0108).
#
# Idempotent: re-running replaces stale symlinks/copies. Existing
# non-symlink pre-push hooks are preserved with a `.local-backup`
# suffix so a contributor's hand-rolled hook is never silently
# overwritten.
hooks-install:
	@command -v pre-commit >/dev/null || pip install pre-commit
	pre-commit install --install-hooks
	pre-commit install --hook-type commit-msg
	@hooks_dir="$$(git rev-parse --git-path hooks)"; \
	src="$$(git rev-parse --show-toplevel)/scripts/git-hooks/pre-push"; \
	dst="$$hooks_dir/pre-push"; \
	if [ ! -x "$$src" ]; then \
	    echo "hooks-install: $$src missing or not executable" >&2; exit 1; \
	fi; \
	if [ -e "$$dst" ] && [ ! -L "$$dst" ]; then \
	    echo "hooks-install: preserving existing $$dst as $$dst.local-backup"; \
	    mv "$$dst" "$$dst.local-backup"; \
	fi; \
	ln -sfn "$$src" "$$dst"; \
	echo "hooks-install: pre-push -> $$src"

# pr-check — local equivalent of the rule-enforcement.yml deliverables gate.
# Runs scripts/ci/deliverables-check.sh against an existing PR's body
# (PR=<num>) or against a local body file (BODY=<path>). Exits non-zero
# if the six-deliverable checklist or any ticked file reference is
# inconsistent with the diff vs origin/master.
#
# Usage:
#   make pr-check PR=260
#   make pr-check BODY=pr-body.md
pr-check:
	@if [ -n "$(PR)" ]; then \
	    echo "--- pr-check: fetching PR $(PR) body via gh ---"; \
	    PR_BODY="$$(gh pr view $(PR) --json body -q .body)" \
	        bash scripts/ci/deliverables-check.sh; \
	elif [ -n "$(BODY)" ]; then \
	    echo "--- pr-check: reading body from $(BODY) ---"; \
	    PR_BODY="$$(cat "$(BODY)")" \
	        bash scripts/ci/deliverables-check.sh; \
	else \
	    echo "Usage: make pr-check PR=<number>  OR  make pr-check BODY=<file>" >&2; \
	    exit 2; \
	fi

help:
	@echo "Fork-specific targets:"
	@echo "  make lint             — clang-tidy + cppcheck + ruff + shellcheck"
	@echo "  make format           — clang-format + black + isort + shfmt (writes)"
	@echo "  make format-check     — same, no writes (CI gate)"
	@echo "  make sec              — semgrep (CERT-C + CWE + fork rules)"
	@echo "  make sbom             — SPDX + CycloneDX SBOMs via syft"
	@echo "  make pr-check         — ADR-0108 deliverables gate (PR=<num> or BODY=<file>)"
	@echo "  make test-netflix-golden — D24 gate: 3 Netflix CPU test pairs"
	@echo "  make test-sanitizers  — ASan + UBSan build + run"
	@echo "  make test-fast        — meson --suite=fast (pre-push gate)"
	@echo "  make coverage         — gcov/lcov line+branch coverage report"
	@echo "  make coverage-html    — render HTML coverage report"
	@echo "  make coverage-check   — enforce ≥70% overall / ≥85% critical"
	@echo "  make assertion-density — Power-of-10 rule 5 density check"
	@echo "  make hooks-install    — wire up pre-commit git hooks"
	@echo ""
	@echo "Upstream targets: build, test, debug, install, clean, distclean, cythonize"
