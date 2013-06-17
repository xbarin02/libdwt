include ../../common.mk

LIBNAME = libdwt
LIBPATH = $(ROOT)/src
CFLAGS += -I$(LIBPATH)
BIN = start

.PHONY: all clean

all: $(BIN)

clean:
	$(MAKE) -C $(LIBPATH) $@
	-$(RM) $(BIN) *.o *.elf *.gdb

$(BIN): $(BIN).o $(LIBPATH)/$(LIBNAME).o

$(BIN).o: $(BIN).c $(LIBPATH)/$(LIBNAME).h

$(LIBPATH)/$(LIBNAME).o: $(LIBPATH)/$(LIBNAME).c $(LIBPATH)/$(LIBNAME).h
	$(MAKE) -C $(LIBPATH) $(LIBNAME).o
