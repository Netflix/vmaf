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
obj =  $(wildcard ptools/*.o) $(wildcard wrapper/libobj/*.o)

$(alib): $(obj)
	ar rcs $@ $^

.PHONY: install
install: $(alib)
	mkdir -p $(DESTDIR)$(PREFIX)/lib
	mkdir -p $(DESTDIR)$(PREFIX)/include
	cp $(alib) $(DESTDIR)$(PREFIX)/lib/$(alib)
	cp wrapper/libsrc/libvmaf.h $(DESTDIR)$(PREFIX)/include/
	cp -r model $(DESTDIR)$(PREFIX)/share/

.PHONY: uninstall
uninstall:
	rm -f $(alib)
	rm -f $(DESTDIR)$(PREFIX)/lib/$(alib)
	rm -f $(DESTDIR)$(PREFIX)/include/libvmaf.h
	rm -fr $(DESTDIR)$(PREFIX)/share/model
