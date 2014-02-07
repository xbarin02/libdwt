include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH)
BIN = vectors

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) $(BIN) *.o *.elf *.gdb

distclean: clean
	-$(RM) *.pgm

$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).a

$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a
