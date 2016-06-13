TOP = $(shell pwd)

SRCDIR  = $(TOP)/src
OBJDIR  = $(TOP)/obj
TOOLDIR = $(TOP)/tool

OBJS_VMAF = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/common/convolution.o \
	$(OBJDIR)/adm.o \
	$(OBJDIR)/adm_tools.o \
	$(OBJDIR)/ansnr.o \
	$(OBJDIR)/ansnr_tools.o \
	$(OBJDIR)/vif.o \
	$(OBJDIR)/vif_tools.o \
	$(OBJDIR)/motion.o \
	$(OBJDIR)/all.o \
	$(OBJDIR)/vmaf.o \

OBJS_PSNR = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/psnr_tools.o \
	$(OBJDIR)/psnr.o \

OBJS_MOMENT = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/moment.o \

OBJS_SSIM = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/iqa/math_utils.o \
	$(OBJDIR)/iqa/convolve.o \
	$(OBJDIR)/iqa/decimate.o \
	$(OBJDIR)/iqa/ssim_tools.o \
	$(OBJDIR)/ssim.o \

OBJS_MS_SSIM = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/iqa/math_utils.o \
	$(OBJDIR)/iqa/convolve.o \
	$(OBJDIR)/iqa/decimate.o \
	$(OBJDIR)/iqa/ssim_tools.o \
	$(OBJDIR)/ms_ssim.o \

all: vmaf psnr moment ssim ms_ssim

CFLAGS_COMMON = -g -O3 -fPIC -Wall -Wextra -pedantic
#CFLAGS_COMMON = -g -O0 -fPIC -Wall -Wextra -pedantic -D BUILD_O0

CFLAGS   := -std=c99 $(CFLAGS_COMMON) $(CFLAGS)
CXXFLAGS := -std=c++11 $(CFLAGS_COMMON) $(CXXFLAGS)
CPPFLAGS := $(CPPFLAGS)
LIBS     := $(LIBS) -lm
LDFLAGS  := $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

vmaf: $(OBJS_VMAF)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

psnr: $(OBJS_PSNR)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

moment: $(OBJS_MOMENT)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

ssim: $(OBJS_SSIM)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

ms_ssim: $(OBJS_MS_SSIM)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(OBJDIR)/common/*.o
	rm -f $(OBJDIR)/iqa/*.o
	rm -f vmaf psnr moment ssim ms_ssim

.PHONY: all clean