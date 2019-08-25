TARGETS = \
	ptools \
	libsvm \
	wrapper \
	feature

all:
	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE); cd ..; \
	done

	cd libsvm; $(MAKE) lib; cd ..;

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

.PHONY: all clean $(TARGETS)


