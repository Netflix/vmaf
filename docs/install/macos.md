# Installing on macOS

```bash
bash scripts/setup/macos.sh                     # CPU-only, Intel or Apple silicon
INSTALL_LINTERS=1 bash scripts/setup/macos.sh   # + Homebrew LLVM (clang-tidy/clang-format)
```

## Caveats

- **CUDA** is unsupported on macOS by NVIDIA. Use Linux or Windows for CUDA work.
- **SYCL via Intel oneAPI** is unsupported on Apple Silicon; the setup script blocks it.
- On Apple Silicon, Apple's built-in `clang` lacks `clang-tidy`/`clang-format`.
  The setup script installs `llvm` from Homebrew and exports its bin dir into `PATH`.

## Manual install

```bash
brew install meson ninja pkg-config nasm doxygen
brew install --with-toolchain llvm     # clang-tidy, clang-format, clangd
```

Then append to your shell RC:

```bash
export PATH="$(brew --prefix llvm)/bin:$PATH"
```

## Build

```bash
cd libvmaf
meson setup ../build
ninja -C ../build
```
