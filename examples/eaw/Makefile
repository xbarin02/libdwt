include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH)
BIN = simple

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) $(BIN) *.o *.elf *.gdb

$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).o $(LIBPATH)/eaw-experimental.o

$(BIN).o: $(BIN).c $(LIBPATH)/$(LIBNAME).h

$(LIBPATH)/$(LIBNAME).o: $(LIBPATH)/$(LIBNAME).c $(LIBPATH)/$(LIBNAME).h
	$(MAKE) -C $(LIBPATH) $(LIBNAME).o

$(LIBPATH)/eaw-experimental.o: $(LIBPATH)/eaw-experimental.c $(LIBPATH)/eaw-experimental.h
	$(MAKE) -C $(LIBPATH) eaw-experimental.o

.PHONY: distclean
distclean: clean
	$(RM) *.pgm
