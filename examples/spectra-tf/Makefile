include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH) -D_POSIX_C_SOURCE=199309L -D_GNU_SOURCE
BIN = tf

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) $(BIN) *.o *.elf *.gdb

$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).a

$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a
