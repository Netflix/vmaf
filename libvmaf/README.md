# libvmaf

## Compile
1. Install [Meson](https://mesonbuild.com/) (0.47 or higher) and [Ninja](https://ninja-build.org/)
2. Run `meson build --buildtype release`
3. Build with `ninja -vC build`

## Test
Build and run tests with `ninja -vC build test`

## Install
Install using `ninja -vC build install`

## Documentation
Generate HTML documentation with `ninja -vC build doc/html`
