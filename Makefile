all:
	cd third_party/libsvm && make lib

	meson setup libvmaf/build libvmaf --buildtype release && \
	ninja -vC libvmaf/build
	python3 python/setup.py build_ext --build-lib python

clean:
	cd third_party/libsvm && make clean && cd -
	rm -rf libvmaf/build
	rm python/vmaf/core/adm_dwt2_cy.c*

install:
	meson setup libvmaf/build libvmaf --buildtype release && \
	ninja -vC libvmaf/build install