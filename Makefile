all:
	cd third_party/libsvm && make lib

	meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true && \
	ninja -vC libvmaf/build
	cd python && python3 setup.py build_ext --build-lib .

clean:
	cd third_party/libsvm && make clean && cd -
	rm -rf libvmaf/build
	rm -f python/vmaf/core/adm_dwt2_cy.c*

install:
	meson setup libvmaf/build libvmaf --buildtype release && \
	ninja -vC libvmaf/build install
