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

clean:
	-for dir in $(TARGETS); do \
		cd $${dir}; $(MAKE) clean; cd ..; \
	done

test:
	@echo hello;
 
.PHONY: clean $(TARGETS)


PREFIX = /usr/local

alib = libvmaf.a
obj = $(wildcard wrapper/obj/*.o) $(wildcard model/*.model) $(wildcard ptools/*.o)

$(alib): $(obj)
	ar rcs $@ $^

.PHONY: install
install: $(alib)
	mkdir -p $(DESTDIR)$(PREFIX)/lib
	mkdir -p $(DESTDIR)$(PREFIX)/include
	cp $(alib) $(DESTDIR)$(PREFIX)/lib/$(alib)
	cp wrapper/src/libvmaf.h $(DESTDIR)$(PREFIX)/include/

.PHONY: uninstall
uninstall:
	rm -f $(alib)
	rm -f $(DESTDIR)$(PREFIX)/lib/$(alib)
	rm -f $(DESTDIR)$(PREFIX)/include/libvmaf.h
