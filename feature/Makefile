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
	$(SRCDIR)/common/convolution_internal.h

OBJS = \
	$(OBJDIR)/adm.o \
	$(OBJDIR)/adm_tools.o \
	$(OBJDIR)/ansnr.o \
	$(OBJDIR)/ansnr_tools.o \
	$(OBJDIR)/common/alloc.o \
	$(OBJDIR)/common/file_io.o \
	$(OBJDIR)/main.o \
	$(OBJDIR)/vif.o \
	$(OBJDIR)/vif_tools.o \
	$(OBJDIR)/motion.o \
	$(OBJDIR)/common/convolution.o

all: vmaf

CCLD  := $(CC)
CXXLD := $(CXX)

CFLAGS_COMMON = -g -O3 -fPIC -Wall -Wextra -pedantic

CFLAGS   := -std=c99 $(CFLAGS_COMMON) $(CFLAGS)
CXXFLAGS := -std=c++11 $(CFLAGS_COMMON) $(CXXFLAGS)
CPPFLAGS := $(CPPFLAGS)
LIBS     := $(LIBS) -lm
LDFLAGS  := $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADERS)
	$(CC) -c -o $@ $(CFLAGS) $(CPPFLAGS) $<

vmaf: $(OBJS)
	$(CC) -o $@ $(LDFLAGS) $^ $(LIBS)

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(OBJDIR)/common/*.o
	rm -f vmaf

.PHONY: all clean
