all:
	cd src/ptools; $(MAKE); cd ../..;
	cd src/libsvm; $(MAKE); cd ../..;
	cd src/libsvm; $(MAKE) lib; cd ../..;
	cd src/wrapper; $(MAKE); cd ../..;
	cd feature; $(MAKE); cd ..;


clean:
	cd src/ptools; $(MAKE) clean; cd ../..;
	cd src/libsvm; $(MAKE) clean; cd ../..;
	cd src/wrapper; $(MAKE) clean; cd ../..;
	cd feature; $(MAKE) clean; cd ..;

test:
	@echo hello;

install:
	cd src/wrapper; $(MAKE) install; cd ../..;

uninstall:
	cd src/wrapper; $(MAKE) uninstall; cd ../..;

testlib:
	cd src/wrapper; $(MAKE) testlib; cd ../..;

.PHONY: all clean $(TARGETS)


