#!/usr/bin/env bash
# Arch / Manjaro / CachyOS / EndeavourOS setup.
# CUDA and oneAPI are both available in official repos — no AUR required for core deps.
set -euo pipefail

ENABLE_CUDA="${ENABLE_CUDA:-false}"
ENABLE_SYCL="${ENABLE_SYCL:-false}"
INSTALL_LINTERS="${INSTALL_LINTERS:-true}"

need_sudo() { [[ $EUID -ne 0 ]] && echo "sudo" || echo ""; }
SUDO="$(need_sudo)"

echo "=== Arch setup for vmaf fork ==="
$SUDO pacman -Syu --noconfirm --needed \
  base-devel clang cppcheck \
  meson ninja nasm pkgconf \
  python python-pip python-virtualenv \
  git curl wget \
  doxygen \
  ffmpeg # libav* headers via ffmpeg-headers alt; ffmpeg meta-package pulls all

if [[ "$INSTALL_LINTERS" == "true" ]]; then
  $SUDO pacman -S --noconfirm --needed shellcheck shfmt
  python -m pip install --user --break-system-packages --upgrade \
    pre-commit ruff black isort mypy semgrep
fi

if [[ "$ENABLE_CUDA" == "true" ]]; then
  $SUDO pacman -S --noconfirm --needed cuda cuda-tools
  echo "Note: /opt/cuda/bin needs to be in PATH (add to ~/.zshrc or ~/.bashrc):"
  echo "  export PATH=/opt/cuda/bin:\$PATH"
fi

if [[ "$ENABLE_SYCL" == "true" ]]; then
  # oneAPI on Arch: via AUR (intel-oneapi-basekit) or official Intel installer.
  echo "oneAPI on Arch:"
  echo "  1. AUR:   yay -S intel-oneapi-basekit intel-oneapi-hpckit"
  echo "  2. Direct: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
  echo "Then:   source /opt/intel/oneapi/setvars.sh"
fi

echo ""
echo "=== done. next steps ==="
echo "  meson setup build -Denable_cuda=$ENABLE_CUDA -Denable_sycl=$ENABLE_SYCL"
