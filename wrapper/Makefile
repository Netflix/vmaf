TOP = $(shell pwd)

SRCDIR = $(TOP)/src
OBJDIR = $(TOP)/obj
LIBSRCDIR = $(TOP)/libsrc
LIBOBJDIR = $(TOP)/libobj
FEATURESRCDIR = $(TOP)/../feature/src
PTOOLSDIR = $(TOP)/../ptools

OBJS = \
	$(OBJDIR)/alloc.o \
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
	$(OBJDIR)/main.o \
	$(OBJDIR)/pugixml.o \

LIBOBJS = \
	$(LIBOBJDIR)/alloc.o \
	$(LIBOBJDIR)/file_io.o \
	$(LIBOBJDIR)/cpu.o \
	$(LIBOBJDIR)/convolution.o \
	$(LIBOBJDIR)/adm.o \
	$(LIBOBJDIR)/adm_tools.o \
	$(LIBOBJDIR)/ansnr.o \
	$(LIBOBJDIR)/ansnr_tools.o \
	$(LIBOBJDIR)/vif.o \
	$(LIBOBJDIR)/vif_tools.o \
	$(LIBOBJDIR)/motion.o \
	$(LIBOBJDIR)/psnr.o \
	$(LIBOBJDIR)/math_utils.o \
	$(LIBOBJDIR)/convolve.o \
	$(LIBOBJDIR)/decimate.o \
	$(LIBOBJDIR)/ssim_tools.o \
	$(LIBOBJDIR)/ssim.o \
	$(LIBOBJDIR)/ms_ssim.o \
    $(LIBOBJDIR)/svm.o \
	$(LIBOBJDIR)/darray.o \
	$(LIBOBJDIR)/pugixml.o \
	$(LIBOBJDIR)/combo_lib.o \
	$(LIBOBJDIR)/vmaf_lib.o \
    $(LIBOBJDIR)/vmaf_wrapper.o \
    $(LIBOBJDIR)/libvmaf.o \

AVX_OBJS := \
	$(OBJDIR)/convolution_avx.o \

all: vmafossexec lib

CFLAGS_COMMON = -g -O3 -fPIC -w -Wextra -pedantic
# CFLAGS_COMMON = -g -O0 -fPIC -Wall -Wextra -pedantic -D BUILD_O0 #-D PRINT_PROGRESS

CFLAGS   := -std=c99 $(CFLAGS_COMMON) $(CFLAGS)
CXXFLAGS := -std=c++11 $(CFLAGS_COMMON) $(CXXFLAGS)
CPPFLAGS := $(CPPFLAGS)
LIBS     := $(LIBS) -lm -lpthread -lptools
LDFLAGS  := $(LDFLAGS)

$(AVX_OBJS): EXTRA_CFLAGS := -mavx

$(OBJDIR)/alloc.o: $(FEATURESRCDIR)/common/alloc.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/file_io.o: $(FEATURESRCDIR)/common/file_io.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/convolution.o: $(FEATURESRCDIR)/common/convolution.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/cpu.o: $(FEATURESRCDIR)/common/cpu.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(OBJDIR)/convolution_avx.o: $(FEATURESRCDIR)/common/convolution_avx.c
	$(CC) -c -o $@ $(EXTRA_CFLAGS) $(CFLAGS) $(CPPFLAGS) $<
	cp $(OBJDIR)/convolution_avx.o $(LIBOBJDIR)

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


$(LIBOBJDIR)/%.o: $(FEATURESRCDIR)/common/%.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(LIBOBJDIR)/%.o: $(FEATURESRCDIR)/%.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(LIBOBJDIR)/%.o: $(FEATURESRCDIR)/iqa/%.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

$(LIBOBJDIR)/svm.o: $(SRCDIR)/svm.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

$(LIBOBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) -I $(FEATURESRCDIR) $<

$(LIBOBJDIR)/%.o: $(SRCDIR)/pugixml/%.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $<

$(LIBOBJDIR)/combo_lib.o: $(LIBSRCDIR)/combo_lib.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) -I $(FEATURESRCDIR) -I $(FEATURESRCDIR)/common -I $(FEATURESRCDIR)/iqa -I $(SRCDIR) $<

$(LIBOBJDIR)/vmaf_lib.o: $(LIBSRCDIR)/vmaf_lib.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) \
	-DOC_NEW_STYLE_INCLUDES -I $(PTOOLSDIR) -I $(SRCDIR) -I $(FEATURESRCDIR)/common \
	-I $(PTOOLSDIR)/opencontainers_1_8_4/include  $<

$(LIBOBJDIR)/vmaf_wrapper.o: $(LIBSRCDIR)/vmaf_wrapper.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) -I $(SRCDIR) $<

$(LIBOBJDIR)/libvmaf.o: $(LIBSRCDIR)/libvmaf.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) -I $(SRCDIR) -I $(FEATURESRCDIR)/common $<

vmafossexec: $(OBJS)
ifeq ($(shell uname),Darwin)
	$(CXX) -o $@ $(LDFLAGS) $^ -L$(PTOOLSDIR) $(LIBS) -Wl
else
	$(CXX) -o $@ $(LDFLAGS) $^ -L$(PTOOLSDIR) $(LIBS) -Wl,-rpath=$(PTOOLSDIR)
endif

lib: $(LIBOBJS)

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(LIBOBJDIR)/*.o
	rm -f vmafossexec

.PHONY: all clean
.DEFAULT: all
