CC = g++
# CC = icc     # Intel seems to work

## With g++ --version==4.4.x, you may need -fno-strict-aliasing to get rid 
## of some annoying warnings when you compile with -O2 or above

# *****CAREFUL! g++ 4.5.1 20100924 DOES NOT WORK with Ptools and -O.  You must
# compile without optimization for that compiler to work

OC = ./opencontainers_1_8_4
OCINC = $(OC)/include

CFLAGS = -Wall -Wextra -fpic -O -fno-strict-aliasing -DLINUX_ -DOC_NEW_STYLE_INCLUDES -Wno-deprecated -I$(OCINC) -pthread -D_REENTRANT

## How to build with OCString
##CFLAGS = -Wall -O4 -DLINUX_ -DOC_USE_OC_STRING -DOC_USE_OC_EXCEPTIONS -DOC_ONLY_NEEDED_STL -DOC_NEW_STYLE_INCLUDES -Wno-deprecated -I$(OCINC) 

###### CFLAGS = -Wall -O4 -DLINUX_-DOC_NEW_STYLE_INCLUDES -Wno-deprecated -I$(OCINC) 


CCFLAGS = -pthread $(CFLAGS)

COM_OBJS = m2pythontools.o valpython.o midassocket.o valprotocol2.o m2ser.o m2streamdataenc.o m2convertrep.o timeconv.o

OBJS = midastalker_ex.o midastalker_ex2.o httpclient_ex.o httpserver_ex.o $(COM_OBJS) load.o save.o sharedmemory.o 

all: libptools

.cc.o:
	$(CC) $(CFLAGS) -c $<

libptools : $(COM_OBJS)
ifeq ($(shell uname),Darwin)
	$(CC) $(CCFLAGS) $(COM_OBJS) -dynamiclib -install_name $(shell pwd)/libptools.dylib -o libptools.dylib
else
	$(CC) $(CCFLAGS) $(COM_OBJS) -shared -o libptools.so
endif

clean :
	/bin/rm -rf *.o *.so *.dylib *~

