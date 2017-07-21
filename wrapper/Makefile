TOP = $(shell pwd)

SRCDIR = $(TOP)/src
OBJDIR = $(TOP)/obj
FEATURESRCDIR = $(TOP)/../feature/src
PTOOLSDIR = $(TOP)/../ptools

OBJS = \
	$(OBJDIR)/alloc.o \
	$(OBJDIR)/frame.o \
	$(OBJDIR)/file_io.o \
	$(OBJDIR)/cpu.o \
	$(OBJDIR)/convolution.o \
	$(OBJDIR)/convolution_avx.o \
	$(OBJDIR)/adm.o \
	$(OBJDIR)/adm_tools.o \
	$(OBJDIR)/ansnr.o \
	$(OBJDIR)/ansnr_tools.o \
	$(OBJDIR)/vif.o \
	$(OBJDIR)/vif_tools.o \
	$(OBJDIR)/motion.o \
	$(OBJDIR)/psnr.o \
	$(OBJDIR)/math_utils.o \
	$(OBJDIR)/convolve.o \
	$(OBJDIR)/decimate.o \
	$(OBJDIR)/ssim_tools.o \
	$(OBJDIR)/ssim.o \
	$(OBJDIR)/ms_ssim.o \
	$(OBJDIR)/svm.o \
	$(OBJDIR)/combo.o \
	$(OBJDIR)/vmaf.o \
	$(OBJDIR)/darray.o \
	$(OBJDIR)/libvmaf.o \
	$(OBJDIR)/main.o \
	$(OBJDIR)/pugixml.o \

AVX_OBJS := \
	$(OBJDIR)/convolution_avx.o \

.PHONY: all
all: vmafossexec

CFLAGS_COMMON = -g -O3 -fPIC -w -Wextra -pedantic
# CFLAGS_COMMON = -g -O0 -fPIC -Wall -Wextra -pedantic -D BUILD_O0 #-D PRINT_PROGRESS

CFLAGS   := -std=c99 $(CFLAGS_COMMON) $(CFLAGS)
CXXFLAGS := -std=c++11 $(CFLAGS_COMMON) $(CXXFLAGS)
CPPFLAGS := $(CPPFLAGS)
LDFLAGS  := $(LDFLAGS)

$(AVX_OBJS): EXTRA_CFLAGS := -mavx

$(OBJDIR)/alloc.o: $(FEATURESRCDIR)/common/alloc.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/file_io.o: $(FEATURESRCDIR)/common/file_io.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/frame.o: $(FEATURESRCDIR)/common/frame.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/convolution.o: $(FEATURESRCDIR)/common/convolution.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/cpu.o: $(FEATURESRCDIR)/common/cpu.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/convolution_avx.o: $(FEATURESRCDIR)/common/convolution_avx.c
	$(CC) -c -o $@ $(EXTRA_CFLAGS) $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/adm.o: $(FEATURESRCDIR)/adm.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/adm_tools.o: $(FEATURESRCDIR)/adm_tools.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/ansnr.o: $(FEATURESRCDIR)/ansnr.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/ansnr_tools.o: $(FEATURESRCDIR)/ansnr_tools.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/vif.o: $(FEATURESRCDIR)/vif.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/vif_tools.o: $(FEATURESRCDIR)/vif_tools.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/motion.o: $(FEATURESRCDIR)/motion.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/psnr.o: $(FEATURESRCDIR)/psnr.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/math_utils.o: $(FEATURESRCDIR)/iqa/math_utils.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/convolve.o: $(FEATURESRCDIR)/iqa/convolve.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/decimate.o: $(FEATURESRCDIR)/iqa/decimate.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/ssim_tools.o: $(FEATURESRCDIR)/iqa/ssim_tools.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/ssim.o: $(FEATURESRCDIR)/ssim.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/ms_ssim.o: $(FEATURESRCDIR)/ms_ssim.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/svm.o: $(SRCDIR)/svm.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/libvmaf.o: $(SRCDIR)/libvmaf.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) -I $(SRCDIR) -I $(FEATURESRCDIR)/common $<

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) -I $(FEATURESRCDIR) -I $(FEATURESRCDIR)/common $<

$(OBJDIR)/vmaf.o: $(SRCDIR)/vmaf.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) \
	-DOC_NEW_STYLE_INCLUDES -I $(PTOOLSDIR) \
	-I $(PTOOLSDIR)/opencontainers_1_8_4/include  $<

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) -I $(FEATURESRCDIR) $<

$(OBJDIR)/%.o: $(SRCDIR)/pugixml/%.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

PREFIX = /usr/local

lib = libvmaf.a
$(lib): $(OBJS) $(wildcard ../ptools/*.o)
	ar rcs $@ $^

vmafossexec: $(lib)
	$(CXX) -s -o $@ $(LDFLAGS) $^ -pthread

.PHONY: install
install:
	ar -dv $(lib) main.o
	mkdir -p $(DESTDIR)$(PREFIX)/lib
	mkdir -p $(DESTDIR)$(PREFIX)/include
	mkdir -p $(DESTDIR)$(PREFIX)/share
	mkdir -p $(DESTDIR)$(PREFIX)/lib/pkgconfig
	cp $(lib) $(DESTDIR)$(PREFIX)/lib/$(lib)
	cp src/libvmaf.h $(DESTDIR)$(PREFIX)/include/
	cp -r ../model $(DESTDIR)$(PREFIX)/share/
	cp libvmaf.pc $(DESTDIR)$(PREFIX)/lib/pkgconfig/

.PHONY: uninstall
uninstall:
	rm -f $(lib)
	rm -f $(DESTDIR)$(PREFIX)/lib/$(lib)
	rm -f $(DESTDIR)$(PREFIX)/include/libvmaf.h
	rm -fr $(DESTDIR)$(PREFIX)/share/model
	rm -f $(DESTDIR)$(PREFIX)/lib/pkgconfig/libvmaf.pc

testlib: $(SRCDIR)/main.cpp
	$(CXX) -s -o $@  $(CXXFLAGS) $^ -I $(FEATURESRCDIR) -I $(DESTDIR)$(PREFIX)/include -L$(DESTDIR)$(PREFIX)/lib -lvmaf -pthread

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o
	rm -f libvmaf.a
	rm -f vmafossexec
	rm -f testlib

.DEFAULT: all
