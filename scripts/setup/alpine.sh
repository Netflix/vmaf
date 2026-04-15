#!/usr/bin/env bash
# Alpine 3.20+. Alpine uses musl libc; CUDA and oneAPI do NOT officially
# support musl — GPU backends are effectively unavailable on Alpine.
# Included for minimal CPU-only container images.
set -euo pipefail

ENABLE_CUDA="${ENABLE_CUDA:-false}"
ENABLE_SYCL="${ENABLE_SYCL:-false}"
INSTALL_LINTERS="${INSTALL_LINTERS:-true}"

if [[ "$ENABLE_CUDA" == "true" || "$ENABLE_SYCL" == "true" ]]; then
  echo "ERROR: CUDA and SYCL are not supported on Alpine (musl libc)."
  echo "       Use the Ubuntu, Fedora, or Arch setup instead."
  exit 2
fi

need_sudo() { [[ $EUID -ne 0 ]] && echo "sudo" || echo ""; }
SUDO="$(need_sudo)"

echo "=== Alpine setup for vmaf fork (CPU-only) ==="
$SUDO apk add --no-cache \
  build-base clang clang-extra-tools cppcheck \
  meson ninja nasm pkgconf \
  python3 py3-pip py3-virtualenv \
  git curl wget \
  doxygen \
  ffmpeg-dev

if [[ "$INSTALL_LINTERS" == "true" ]]; then
  $SUDO apk add --no-cache shellcheck shfmt
  python3 -m pip install --break-system-packages --user --upgrade \
    pre-commit ruff black isort mypy semgrep
fi

echo ""
echo "=== done. Alpine is CPU-only ==="
echo "  meson setup build -Denable_cuda=false -Denable_sycl=false"
