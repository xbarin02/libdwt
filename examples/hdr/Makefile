include ../../common.mk

LIBPATH = $(ROOT)/src
LIBNAME = libdwt
CXXFLAGS = -I$(LIBPATH) `pkg-config --cflags OpenEXR` -std=c++98 -pedantic -Wall -Wextra -O2
CFLAGS += -I$(LIBPATH)
LDLIBS += `pkg-config --libs OpenEXR` $(LIBPATH)/$(LIBNAME).a -lstdc++
BIN = hdr

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) *.o $(BIN)

distclean: clean
	-$(RM) *.pgm

# link application
$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).a exr.o

# compile application
$(BIN).o: $(BIN).c $(LIBPATH)/$(LIBNAME).h exr.h

# compile dependency
$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a

# compile dependency
exr.o: exr.cpp exr.h
