include ../common.mk

LIBNAME = libdwt
CFLAGS += -finline-functions -D_POSIX_C_SOURCE=199309L -D_GNU_SOURCE -DPACKAGE_VERSION="$(PACKAGE_VERSION)" -DPACKAGE_NAME="$(PACKAGE_NAME)" -DPACKAGE_STRING="$(PACKAGE_STRING)" -DARCH="$(ARCH)"

.PHONY: all clean version

all: $(LIBNAME).o

clean:
	-$(RM) *.o *.a *.so *.S

# ARCH=asvp needs firmwares to be compiled
ifeq ($(ARCH),asvp)
FWDIR = firmware
FIRMWARES = fw_fp01_lift4sa fw_fp01_lift4sb

$(LIBNAME).c: $(addprefix $(FWDIR)/,$(addsuffix .h,$(FIRMWARES)))

$(FWDIR)/%.h: $(FWDIR)/%.c
	$(MAKE) -C $(FWDIR) $(@F)
endif

$(LIBNAME).o: $(LIBNAME).c $(LIBNAME).h

util.o: util.c util.h

signal.o: signal.c signal.h

image.o: image.c image.h

swt.o: swt.c swt.h

dwt.o: dwt.c dwt.h

dwt-simple.o: dwt-simple.c dwt-simple.h

eaw-experimental.o: eaw-experimental.c eaw-experimental.h

dwt-core.o: dwt-core.c dwt-core.h

dwt-sym.o: dwt-sym.c dwt-sym.h

dwt-sym-ms.o: dwt-sym-ms.c dwt-sym-ms.h

gabor.o: gabor.c gabor.h

denoise.o: denoise.c denoise.h

system.o: system.c system.h

spectra.o: spectra.c spectra.h

volume.o: volume.c volume.h

volume-dwt.o: volume-dwt.c volume-dwt.h

core-int.o: core-int.c core-int.h

dwt-core.S: dwt-core.c dwt-core.h
	$(CC) $(CFLAGS) -c -Wa,-ahl=dwt-core.S -g -fverbose-asm $< -o /dev/null

$(LIBNAME).S: $(LIBNAME).c $(LIBNAME).h
	$(CC) $(CFLAGS) -S -Wa,-adhln -g -fverbose-asm $< -o $@

$(LIBNAME).a: $(LIBNAME).o util.o signal.o image.o swt.o dwt.o dwt-simple.o eaw-experimental.o dwt-core.o gabor.o dwt-sym.o dwt-sym-ms.o system.o spectra.o volume.o volume-dwt.o core-int.o
	$(AR) -rsc $@ $^

# $(LIBNAME).so: $(LIBNAME).o
# 	$(CC) -shared -o $@ $^

LIBNAME_CV=cvdwt

$(LIBNAME_CV).o: $(LIBNAME_CV).cpp $(LIBNAME_CV).h
	$(CXX) $(CXXFLAGS) `pkg-config --cflags opencv` -c -o $@ $<

$(LIBNAME_CV).a: $(LIBNAME_CV).o
	$(AR) -rsc $@ $^
