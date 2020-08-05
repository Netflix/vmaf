# libvmaf

## Prerequisites

For building, you need the following:

- [Meson](https://mesonbuild.com/) (0.47 or higher)
- [Ninja](https://ninja-build.org/)
- [NASM](https://www.nasm.us/) (for x86 builds only, 2.14 or higher)

Install these dependencies under Ubuntu with:

```
sudo apt update -qq && \
sudo apt install python3 python3-pip python3-setuptools python3-wheel ninja-build doxygen nasm
pip3 install --user meson
```

Make sure your user install executable directory is on your PATH.
```
export PATH="$PATH:$HOME/.local/bin"
```

Under macOS, install [Homebrew](https://brew.sh), then:

```
brew install meson doxygen nasm
```

## Compile


Run:

```
meson build --buildtype release
```

Build with:

```
ninja -vC build
```

## Test

Build and run tests with:

```
ninja -vC build test
```

## Install

Install the libraries and models to `/usr/local` using:

```
ninja -vC build install
```

Under Linux, you may need `sudo` for the above command.

## Documentation

Generate HTML documentation with:

```
ninja -vC build doc/html
```

## Example

The following example shows a comparison using a pair of y4m inputs (`reference.y4m`, `distorted.y4m`).
In addition to VMAF which is enabled with the model `../model/vmaf_v0.6.1.pkl`, the following metrics
are computed and logged: `psnr`, `ssim`, `ms-ssim`.

```sh
./build/tools/vmaf_rc \
    --reference reference.y4m \
    --distorted distorted.y4m \
    --model path=../model/vmaf_v0.6.1.pkl \
    --feature psnr --feature float_ssim --feature float_ms_ssim \
    --output /dev/stdout
```
