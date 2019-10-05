all:
	cd src/ptools; $(MAKE); cd ../..;
	cd src/libsvm; $(MAKE); cd ../..;
	cd src/libsvm; $(MAKE) lib; cd ../..;
	cd src/libvmaf; $(MAKE); cd ../..;
	cd src/feature; $(MAKE); cd ../..;


clean:
	cd src/ptools; $(MAKE) clean; cd ../..;
	cd src/libsvm; $(MAKE) clean; cd ../..;
	cd src/libvmaf; $(MAKE) clean; cd ../..;
	cd src/feature; $(MAKE) clean; cd ../..;

test:
	@echo hello;

install:
	cd src/libvmaf; $(MAKE) install; cd ../..;

uninstall:
	cd src/libvmaf; $(MAKE) uninstall; cd ../..;

testlib:
	cd src/libvmaf; $(MAKE) testlib; cd ../..;

.PHONY: all clean $(TARGETS)


