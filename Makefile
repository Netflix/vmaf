all:
	cd src/ptools; $(MAKE); cd ../..;
	cd src/libsvm; $(MAKE); cd ../..;
	cd src/libsvm; $(MAKE) lib; cd ../..;
	cd wrapper; $(MAKE); cd ..;
	cd feature; $(MAKE); cd ..;


clean:
	cd src/ptools; $(MAKE) clean; cd ../..;
	cd src/libsvm; $(MAKE) clean; cd ../..;
	cd wrapper; $(MAKE) clean; cd ..;
	cd feature; $(MAKE) clean; cd ..;

test:
	@echo hello;

install:
	cd wrapper; $(MAKE) install; cd ..;

uninstall:
	cd wrapper; $(MAKE) uninstall; cd ..;

testlib:
	cd wrapper; $(MAKE) testlib; cd ..;

.PHONY: all clean $(TARGETS)


