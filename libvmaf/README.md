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

Under macOS, install [Homebrew](https://brew.sh), then:

```
brew install meson doxygen
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

Install using:

```
ninja -vC build install
```

## Documentation

Generate HTML documentation with:

```
ninja -vC build doc/html
```
