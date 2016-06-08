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
	$(OBJDIR)/main.o

OBJS_PSNR = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/psnr.o

OBJS_MOMENT = \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/moment.o

all: vmaf psnr moment

CFLAGS_COMMON = -g -O3 -fPIC -Wall -Wextra -pedantic

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

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(OBJDIR)/common/*.o
	rm -f vmaf psnr moment

.PHONY: all clean
