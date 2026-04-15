# Installing on Arch Linux

```bash
bash scripts/setup/arch.sh                     # CPU-only
ENABLE_CUDA=1 bash scripts/setup/arch.sh       # + CUDA toolkit from extra
ENABLE_SYCL=1 bash scripts/setup/arch.sh       # + intel-oneapi-basekit (AUR)
INSTALL_LINTERS=1 bash scripts/setup/arch.sh   # + clang-tidy/cppcheck/iwyu
```

## Manual install

```bash
sudo pacman -S --needed \
    base-devel meson ninja pkgconf nasm \
    python python-pip \
    clang cppcheck doxygen
```

### CUDA (optional)

```bash
sudo pacman -S --needed cuda cuda-tools
```

### SYCL (optional, AUR)

```bash
yay -S intel-oneapi-basekit
source /opt/intel/oneapi/setvars.sh
```

## Build

```bash
cd libvmaf
meson setup ../build -Denable_cuda=true -Denable_sycl=true
ninja -C ../build
```
