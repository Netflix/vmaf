#!/usr/bin/env bash
# Fedora 40+ / RHEL 9 / Rocky 9 / Alma 9. RHEL-family needs EPEL for shellcheck.
set -euo pipefail

ENABLE_CUDA="${ENABLE_CUDA:-false}"
ENABLE_SYCL="${ENABLE_SYCL:-false}"
INSTALL_LINTERS="${INSTALL_LINTERS:-true}"

need_sudo() { [[ $EUID -ne 0 ]] && echo "sudo" || echo ""; }
SUDO="$(need_sudo)"

# Detect Fedora vs RHEL-family for EPEL handling.
. /etc/os-release
if [[ "$ID" != "fedora" ]]; then
  $SUDO dnf install -y epel-release
fi

echo "=== Fedora/RHEL setup for vmaf fork ==="
$SUDO dnf groupinstall -y "Development Tools"
$SUDO dnf install -y \
  clang clang-tools-extra cppcheck \
  meson ninja-build nasm pkgconf-pkg-config \
  python3 python3-pip python3-virtualenv \
  git curl wget \
  doxygen \
  ffmpeg-devel \
  shellcheck

if [[ "$INSTALL_LINTERS" == "true" ]]; then
  if ! command -v shfmt >/dev/null; then
    $SUDO curl -fsSL -o /usr/local/bin/shfmt \
      https://github.com/mvdan/sh/releases/latest/download/shfmt_v3.9.0_linux_amd64
    $SUDO chmod +x /usr/local/bin/shfmt
  fi
  python3 -m pip install --user --upgrade \
    pre-commit ruff black isort mypy semgrep
fi

if [[ "$ENABLE_CUDA" == "true" ]]; then
  # RPM Fusion nonfree contains cuda for Fedora; on RHEL, use NVIDIA's repo.
  $SUDO dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora$VERSION_ID/x86_64/cuda-fedora$VERSION_ID.repo || true
  $SUDO dnf install -y cuda-toolkit
  echo "Note: export PATH=/usr/local/cuda/bin:\$PATH"
fi

if [[ "$ENABLE_SYCL" == "true" ]]; then
  # Intel oneAPI yum repo.
  $SUDO tee /etc/yum.repos.d/oneAPI.repo >/dev/null <<'EOF'
[oneAPI]
name=Intel(R) oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
  $SUDO dnf install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-runtime-libs \
                       level-zero-devel libva-devel
  echo "Then: source /opt/intel/oneapi/setvars.sh"
fi

echo ""
echo "=== done. next steps ==="
echo "  meson setup build -Denable_cuda=$ENABLE_CUDA -Denable_sycl=$ENABLE_SYCL"
