include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH) -D_POSIX_C_SOURCE=199309L -D_GNU_SOURCE
BIN = mra

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) $(BIN) *.o *.elf *.gdb

$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).a

$(BIN).o: $(BIN).c

$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a

.PHONY: plot
plot:
	-gnuplot plot.txt
	for f in *.svg; do test -s "$${f}" && inkscape -A "$${f/.svg/.pdf}" "$${f}"; done
