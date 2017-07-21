TARGETS = \
	ptools \
	feature \
	libsvm \
	wrapper

all:
	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE); cd ..; \
	done

	cd libsvm; $(MAKE) lib; cd ..;
	cd wrapper; $(MAKE) libvmaf.a; cd ..;

clean:
	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE) clean; cd ..; \
	done

test:
	@echo hello;

install:
	cd wrapper; $(MAKE) install; cd ..;

uninstall:
	cd wrapper; $(MAKE) uninstall; cd ..;

testlib:
	cd wrapper; $(MAKE) testlib; cd ..;

.PHONY: clean $(TARGETS)


