TARGETS = \
	feature \
	libsvm \
	wrapper

all:
	cd ptools; $(MAKE) libptools; cd ..;

	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE); cd ..; \
	done

	cd libsvm; $(MAKE) lib; cd ..;

clean:
	cd ptools; $(MAKE) clean; cd ..;

	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE) clean; cd ..; \
	done

.PHONY: clean $(TARGETS)
