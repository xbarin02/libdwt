include ../common.mk

SUBDIRS = $(wildcard */)

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

clean: $(SUBDIRS)

$(SUBDIRS):
	-$(MAKE) ARCH="$(ARCH)" BUILD="$(BUILD)" -C "$@" $(MAKECMDGOALS)

.PHONY: distclean
distclean: $(SUBDIRS)
