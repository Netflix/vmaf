rm -rf build
meson build --buildtype release
DESTDIR=install ninja -vC build install
