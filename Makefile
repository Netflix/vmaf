all:
	cd third_party/libsvm && make lib

	mkdir -p libvmaf/build && cd libvmaf/build && \
	meson .. --buildtype release && \
	ninja -vC .

clean:
	cd third_party/libsvm && make clean && cd -
	rm -rf libvmaf/build

install:
	mkdir -p libvmaf/build && cd libvmaf/build && \
	meson .. --buildtype release && \
	ninja -vC . install
