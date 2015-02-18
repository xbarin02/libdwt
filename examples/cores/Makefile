include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH)
ifeq ($(ARCH),x86_64)
	CFLAGS += -msse3 -msse4.1
endif
BIN = main

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) $(BIN) *.o *.elf *.gdb

$(BIN): $(BIN).o cores.o cores-nsls.o fix.o image2.o clock.o $(LIBPATH)/$(LIBNAME).a

$(BIN).o: $(BIN).c

cores.o: cores.c cores.h

cores-nsls.o: cores-nsls.c cores-nsls.h

fix.o: fix.c fix.h

image2.o: image2.c image2.h

clock.o: clock.c clock.h

$(LIBPATH)/$(LIBNAME).a:
	$(MAKE) -C $(LIBPATH) $(LIBNAME).a

.PHONY: distclean
distclean: clean
	$(RM) *.pgm
