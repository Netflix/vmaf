.PHONY: all
all: vmaf psnr moment ssim ms_ssim

TOP = $(shell pwd)
SRCDIR  = $(TOP)/src
OBJDIR  = $(TOP)/obj
TOOLDIR = $(TOP)/tool
LIBVMAF = $(TOP)/../wrapper/libvmaf.a

CFLAGS_COMMON = -g -O3 -fPIC -w -Wextra -pedantic
# CFLAGS_COMMON = -g -O0 -fPIC -Wall -Wextra -pedantic

CFLAGS   := -std=c99 $(CFLAGS_COMMON) $(CFLAGS)
CXXFLAGS := -std=c++11 $(CFLAGS_COMMON) $(CXXFLAGS)
CPPFLAGS := $(CPPFLAGS)
LIBS     := $(LIBS) -lm
LDFLAGS  := $(LDFLAGS)

$(AVX_OBJS): EXTRA_CFLAGS := -mavx

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
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

moment: $(OBJDIR)/moment_main.o $(LIBVMAF)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

ssim: $(OBJDIR)/ssim_main.o $(LIBVMAF)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

ms_ssim: $(OBJDIR)/ms_ssim_main.o $(LIBVMAF)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o
	rm -f vmaf psnr moment ssim ms_ssim
