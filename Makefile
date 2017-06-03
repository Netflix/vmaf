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


PREFIX = /usr

alib = libvmaf.a
obj = $(wildcard wrapper/obj/*.o)

$(alib): $(obj)
	ar rcs $@ $^

.PHONY: install
install: $(alib)
	mkdir -p $(DESTDIR)$(PREFIX)/lib
	mkdir -p $(DESTDIR)$(PREFIX)/include
	cp $(alib) $(DESTDIR)$(PREFIX)/lib/$(alib)
	cp wrapper/src/vmaf.h $(DESTDIR)$(PREFIX)/include/

.PHONY: uninstall
uninstall:
	rm -f $(DESTDIR)$(PREFIX)/lib/$(alib)
	rm -f $(DESTDIR)$(PREFIX)/include/vmaf.h
