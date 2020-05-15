all:
	cd third_party/libsvm && make lib

	cd libvmaf && mkdir -p build && \
	meson . build --buildtype release && \
	ninja -vC build

clean:
	cd third_party/libsvm && make clean && cd -
	rm -rf libvmaf/build

install:
	cd libvmaf && mkdir -p build && \
	meson . build --buildtype release && \
	ninja -vC build install
