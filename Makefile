all:
	cd third_party/libsvm && make lib

	meson setup libvmaf/build libvmaf --buildtype debug && \
	ninja -vC libvmaf/build

clean:
	cd third_party/libsvm && make clean && cd -
	rm -rf libvmaf/build

install:
	meson setup libvmaf/build libvmaf --buildtype release && \
	ninja -vC libvmaf/build install