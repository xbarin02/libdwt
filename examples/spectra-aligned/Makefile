include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH) -D_POSIX_C_SOURCE=199309L -D_GNU_SOURCE
BIN = spectra-aligned

.PHONY: all clean distclean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) -- $(BIN) *.o *.elf *.gdb

distclean: clean
	-$(RM) -- *.pgm *.mat

$(BIN): $(BIN).o ../spectra-blobs/spectra-experimental.o $(LIBPATH)/$(LIBNAME).a

$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a
