include ../../common.mk

LIBPATH = $(ROOT)/src
LIBNAME = libdwt
CXXFLAGS = -I$(LIBPATH) `pkg-config --cflags OpenEXR` -std=c++98 -pedantic -Wall -Wextra -O2
CFLAGS += -I$(LIBPATH)
LDLIBS += `pkg-config --libs OpenEXR` $(LIBPATH)/$(LIBNAME).a
BIN = hdr

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) *.o $(BIN)

distclean: clean
	-$(RM) *.pgm

# link application
$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).a exr.o $(LIBPATH)/eaw-experimental.o

# compile application
$(BIN).o: $(BIN).c $(LIBPATH)/$(LIBNAME).h exr.h

$(LIBPATH)/eaw-experimental.o: $(LIBPATH)/eaw-experimental.c $(LIBPATH)/eaw-experimental.h
	$(MAKE) -C $(LIBPATH) eaw-experimental.o

# compile dependency
$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a

# compile dependency
exr.o: exr.cpp exr.h
