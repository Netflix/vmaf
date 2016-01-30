TARGETS = \
	feature \
	libsvm

all:
	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE); cd ..; \
	done

clean:
	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE) clean; cd ..; \
	done

.PHONY: clean $(TARGETS)
