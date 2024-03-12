VENV=.venv
.PHONY: all install clean distclean deps

all: deps
	$(VENV)/bin/meson setup libvmaf/build libvmaf --buildtype release -Denable_float=true && \
	$(VENV)/bin/ninja -vC libvmaf/build && \
	cd python && \
	../$(VENV)/bin/python setup.py build_ext --build-lib .

install: deps
	$(VENV)/bin/meson setup libvmaf/build libvmaf --buildtype release && \
	$(VENV)/bin/ninja -vC libvmaf/build install

clean:
	rm -rf libvmaf/build
	rm -f python/vmaf/core/adm_dwt2_cy.c*

distclean: clean
	rm -rf $(VENV)

deps:
	test -d $(VENV) || python3 -mvenv $(VENV)
	$(VENV)/bin/pip install meson ninja cython numpy
