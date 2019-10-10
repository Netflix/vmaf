LIBVMAF = libvmaf.a
LIBVMAF_DYN = libvmafdyn.so

.PHONY: all
all: $(LIBVMAF) $(LIBVMAF_DYN) libvmaf.pc vmafossexec vmaf psnr moment ssim ms_ssim

OS = $(shell uname)
TOP = $(shell pwd)
SRCDIR = $(TOP)/src
OBJDIR = $(TOP)/obj
TOOLDIR = $(TOP)/tool
FEATURESRCDIR = $(TOP)/src
PTOOLSDIR = $(TOP)/../ptools
INSTALL_PREFIX = /usr/local
INCLUDES += -I$(TOP)/src
INCLUDES += -I$(TOP)/src/common

OBJS = \
	$(OBJDIR)/alloc.o \
	$(OBJDIR)/alignment.o \
	$(OBJDIR)/frame.o \
	$(OBJDIR)/file_io.o \
	$(OBJDIR)/cpu.o \
	$(OBJDIR)/convolution.o \
	$(OBJDIR)/convolution_avx.o \
	$(OBJDIR)/psnr_tools.o \
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
	$(OBJDIR)/moment.o \
	$(OBJDIR)/svm.o \
	$(OBJDIR)/combo.o \
	$(OBJDIR)/all.o \
	$(OBJDIR)/vmaf.o \
	$(OBJDIR)/darray.o \
	$(OBJDIR)/libvmaf.o \
	$(OBJDIR)/pugixml.o \
	$(OBJDIR)/blur_array.o \
	$(OBJDIR)/cpu_info.o

AVX_OBJS := \
	$(OBJDIR)/convolution_avx.o \

CFLAGS_COMMON = -g -O3 -fPIC -w -Wextra -pedantic
#CFLAGS_COMMON = -g -O3 -fPIC -w -Wextra -pedantic -D PRINT_PROGRESS
#CFLAGS_COMMON = -g -O0 -fPIC -Wall -Wextra -pedantic -D PRINT_PROGRESS

CFLAGS   := -std=c99 $(CFLAGS_COMMON) $(CFLAGS)
CXXFLAGS := -std=c++11 $(CFLAGS_COMMON) $(CXXFLAGS)
CPPFLAGS := -D_FILE_OFFSET_BITS=64 $(CPPFLAGS)
LDFLAGS  := $(LDFLAGS)
LIBS     := $(LIBS) -lm

$(AVX_OBJS): EXTRA_CFLAGS := -mavx

$(OBJDIR)/alloc.o: $(FEATURESRCDIR)/common/alloc.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/alignment.o: $(FEATURESRCDIR)/common/alignment.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/file_io.o: $(FEATURESRCDIR)/common/file_io.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/frame.o: $(FEATURESRCDIR)/common/frame.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/convolution.o: $(FEATURESRCDIR)/common/convolution.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $<

$(OBJDIR)/cpu.o: $(FEATURESRCDIR)/common/cpu.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/convolution_avx.o: $(FEATURESRCDIR)/common/convolution_avx.c
	$(CC) -c -o $@ $(EXTRA_CFLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $<

$(OBJDIR)/psnr_tools.o: $(FEATURESRCDIR)/psnr_tools.c
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
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $<

$(OBJDIR)/vif_tools.o: $(FEATURESRCDIR)/vif_tools.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $<

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

$(OBJDIR)/moment.o: $(FEATURESRCDIR)/moment.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/all.o: $(FEATURESRCDIR)/all.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/blur_array.o: $(FEATURESRCDIR)/common/blur_array.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/cpu_info.o: $(SRCDIR)/cpu_info.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/svm.o: $(SRCDIR)/svm.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/libvmaf.o: $(SRCDIR)/libvmaf.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) -I $(SRCDIR) -I $(FEATURESRCDIR)/common -DOC_NEW_STYLE_INCLUDES -I $(PTOOLSDIR) \
	-I $(PTOOLSDIR)/opencontainers_1_8_4/include $<

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) -I $(FEATURESRCDIR) -I $(FEATURESRCDIR)/common $<

$(OBJDIR)/vmaf.o: $(SRCDIR)/vmaf.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) \
	-DOC_NEW_STYLE_INCLUDES -I $(PTOOLSDIR) \
	-I $(PTOOLSDIR)/opencontainers_1_8_4/include -I $(FEATURESRCDIR) $<

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) -I $(FEATURESRCDIR) $<

$(OBJDIR)/%.o: $(SRCDIR)/pugixml/%.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

$(LIBVMAF): $(OBJS) $(wildcard $(PTOOLSDIR)/*.o)
	ar rcs $@ $^

$(LIBVMAF_DYN): $(OBJS) $(wildcard $(PTOOLSDIR)/*.o)
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-shared -Wl,-install_name,$@"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,$@"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} -o $@ $^

vmafossexec: $(OBJDIR)/main.o $(LIBVMAF)
	$(CXX) -s -o $@ $(LDFLAGS) $^ -pthread
#	$(CXX) -o $@ $(LDFLAGS) $^ -pthread

libvmaf.pc: libvmaf.pc.in
	sed 's!@@PREFIX@@!$(INSTALL_PREFIX)!g' $< >$@

$(OBJDIR)/vmaf_main.o: $(SRCDIR)/vmaf_main.c
	$(CC) $(EXTRA_CFLAGS) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/psnr_main.o: $(SRCDIR)/psnr_main.c
	$(CC) $(EXTRA_CFLAGS) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/moment_main.o: $(SRCDIR)/moment_main.c
	$(CC) $(EXTRA_CFLAGS) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/ssim_main.o: $(SRCDIR)/ssim_main.c
	$(CC) $(EXTRA_CFLAGS) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/ms_ssim_main.o: $(SRCDIR)/ms_ssim_main.c
	$(CC) $(EXTRA_CFLAGS) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

vmaf: $(OBJDIR)/vmaf_main.o $(LIBVMAF)
	$(CXX) -s -o $@ $(LDFLAGS) $^ $(LIBS) -pthread

psnr: $(OBJDIR)/psnr_main.o $(LIBVMAF)
	$(CC) -s -o $@ $(LDFLAGS) $^ $(LIBS)

moment: $(OBJDIR)/moment_main.o $(LIBVMAF)
	$(CC) -s -o $@ $(LDFLAGS) $^ $(LIBS)

ssim: $(OBJDIR)/ssim_main.o $(LIBVMAF)
	$(CC) -s -o $@ $(LDFLAGS) $^ $(LIBS)

ms_ssim: $(OBJDIR)/ms_ssim_main.o $(LIBVMAF)
	$(CC) -s -o $@ $(LDFLAGS) $^ $(LIBS)

.PHONY: install
install: libvmaf.pc
	mkdir -p $(DESTDIR)$(INSTALL_PREFIX)/lib
	mkdir -p $(DESTDIR)$(INSTALL_PREFIX)/include
	mkdir -p $(DESTDIR)$(INSTALL_PREFIX)/share
	mkdir -p $(DESTDIR)$(INSTALL_PREFIX)/lib/pkgconfig
	cp $(LIBVMAF) $(DESTDIR)$(INSTALL_PREFIX)/lib/$(LIBVMAF)
	cp $(LIBVMAF_DYN) $(DESTDIR)$(INSTALL_PREFIX)/lib/$(LIBVMAF_DYN)
	cp src/libvmaf.h $(DESTDIR)$(INSTALL_PREFIX)/include/
	cp -r ../../model $(DESTDIR)$(INSTALL_PREFIX)/share/
	cp libvmaf.pc $(DESTDIR)$(INSTALL_PREFIX)/lib/pkgconfig/

.PHONY: uninstall
uninstall:
	rm -f $(DESTDIR)$(INSTALL_PREFIX)/lib/$(LIBVMAF)
	rm -f $(DESTDIR)$(INSTALL_PREFIX)/lib/$(LIBVMAF_DYN)
	rm -f $(DESTDIR)$(INSTALL_PREFIX)/include/libvmaf.h
	rm -fr $(DESTDIR)$(INSTALL_PREFIX)/share/model
	rm -f $(DESTDIR)$(INSTALL_PREFIX)/lib/pkgconfig/libvmaf.pc

testlib: $(SRCDIR)/main.cpp
	$(CXX) -s -o $@  $(CXXFLAGS) $^ -I $(FEATURESRCDIR) -I $(FEATURESRCDIR)/common -I $(DESTDIR)$(INSTALL_PREFIX)/include -L$(DESTDIR)$(INSTALL_PREFIX)/lib -lvmaf -pthread

testlibdyn: $(SRCDIR)/main.cpp
	$(CXX) -s -o $@  $(CXXFLAGS) $^ -I $(FEATURESRCDIR) -I $(FEATURESRCDIR)/common -I $(DESTDIR)$(INSTALL_PREFIX)/include -L$(DESTDIR)$(INSTALL_PREFIX)/lib -lvmafdyn -pthread

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(LIBVMAF)
	rm -f $(LIBVMAF_DYN)
	rm -f vmafossexec
	rm -f testlib
	rm -f testlibdyn
	rm -f vmaf psnr moment ssim ms_ssim
