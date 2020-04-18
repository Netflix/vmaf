# libvmaf

## Prerequisites

For building, you need the following:

- [Meson](https://mesonbuild.com/) (0.47 or higher)
- [Ninja](https://ninja-build.org/)

Install these dependencies under Ubuntu with:

```
sudo apt update -qq && \
sudo apt install python3 python3-pip python3-setuptools python3-wheel ninja-build doxygen
pip3 install --user meson
```

Make sure your user install executable directory is on your PATH. Add this to the end of `~/.bashrc` (or `~/.bash_profile` under macOS) and restart your shell:

```
export PATH="$PATH:$HOME/.local/bin"
```

Under macOS, install [Homebrew](https://brew.sh), then:

```
brew install meson doxygen
```

## Compile

First, change to the `libvmaf` directory:

```
cd libvmaf
```

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
