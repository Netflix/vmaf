# Installing on Fedora (40 / 41)

```bash
bash scripts/setup/fedora.sh
ENABLE_CUDA=1 bash scripts/setup/fedora.sh      # + CUDA (RPMFusion + NVIDIA repo)
ENABLE_SYCL=1 bash scripts/setup/fedora.sh      # + intel-basekit
INSTALL_LINTERS=1 bash scripts/setup/fedora.sh  # + clang-tools-extra/cppcheck/iwyu
```

## Manual install

```bash
sudo dnf install -y \
    @development-tools meson ninja-build pkgconf-pkg-config nasm \
    python3 python3-pip \
    clang clang-tools-extra cppcheck doxygen
```

### CUDA (optional)

```bash
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/fedora40/x86_64/cuda-fedora40.repo
sudo dnf install -y cuda-toolkit-12-6
```

### SYCL / oneAPI (optional)

```bash
tee /tmp/oneAPI.repo <<'EOF'
[oneAPI]
name=Intel(R) oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
sudo mv /tmp/oneAPI.repo /etc/yum.repos.d/
sudo dnf install -y intel-basekit
source /opt/intel/oneapi/setvars.sh
```

## Build

```bash
cd libvmaf
meson setup ../build -Denable_cuda=true -Denable_sycl=true
ninja -C ../build
```
