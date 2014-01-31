include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH)
BIN = perf

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) $(BIN) *.o *.elf *.gdb
	-$(RM) data/*.txt

$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).a

$(BIN).o: $(BIN).c $(LIBPATH)/$(LIBNAME).h

$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a

.PHONY: plot
plot:
	-gnuplot plot.txt
	-for f in *.svg; do test -s "$${f}" && inkscape -A "$${f/.svg/.pdf}" "$${f}"; done

.PHONY: distclean
distclean: clean
	-$(RM) *.pdf *.svg
