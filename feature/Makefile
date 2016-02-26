TOP = $(shell pwd)

SRCDIR  = $(TOP)/src
OBJDIR  = $(TOP)/obj
TOOLDIR = $(TOP)/tool

HEADERS = \
	$(SRCDIR)/adm_options.h \
	$(SRCDIR)/adm_tools.h \
	$(SRCDIR)/ansnr_options.h \
	$(SRCDIR)/ansnr_tools.h \
	$(SRCDIR)/common/alloc.h \
	$(SRCDIR)/common/file_io.h \
	$(SRCDIR)/main.h \
	$(SRCDIR)/vif_options.h \
	$(SRCDIR)/vif_tools.h \
	$(SRCDIR)/common/macros.h \
	$(SRCDIR)/common/alignment.h \
	$(SRCDIR)/common/convolution.h \
	$(SRCDIR)/common/convolution_internal.h \
	$(SRCDIR)/all_options.h \
	$(SRCDIR)/psnr_options.h

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
	$(OBJDIR)/main.o

OBJS_PSNR = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/psnr.o

all: vmaf psnr

CFLAGS_COMMON = -g -O3 -fPIC -Wall -Wextra -pedantic

CFLAGS   := -std=c99 $(CFLAGS_COMMON) $(CFLAGS)
CXXFLAGS := -std=c++11 $(CFLAGS_COMMON) $(CXXFLAGS)
CPPFLAGS := $(CPPFLAGS)
LIBS     := $(LIBS) -lm
LDFLAGS  := $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADERS)
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

vmaf: $(OBJS_VMAF)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

psnr: $(OBJS_PSNR)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(OBJDIR)/common/*.o
	rm -f vmaf psnr

.PHONY: all clean
