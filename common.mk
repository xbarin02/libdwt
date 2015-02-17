ROOT = $(dir $(lastword $(MAKEFILE_LIST)))

PACKAGE_NAME = libdwt
PACKAGE_VERSION := $(shell cat $(ROOT)/VERSION)
PACKAGE_STRING = $(PACKAGE_NAME) $(PACKAGE_VERSION)
DATE := $(shell date -R)
AUTHORS := $(shell cat $(ROOT)/AUTHORS)

.DEFAULT_GOAL := all

.PHONY: version help

version:
	@echo "$(PACKAGE_VERSION)"

help:
	@echo "You are going to build $(PACKAGE_STRING)."
	@echo "To print this help type 'make help'."
	@echo "To print version type 'make version'."
	@echo "To make an example application type 'make' in example folder."
	@echo "If you want to build any binary for another architecture type 'make ARCH=<arch>', where <arch> is one of 'x86_64' (for PC) or 'microblaze' (for EdkDSP platform)."
	@echo "For release build type 'make BUILD=release'."
	@echo "In case of any problems contact $(AUTHORS)."

include $(ROOT)/arch.mk

EDKDSP_SERVER = ibarina@pcbarina.fit.vutbr.cz
EDKDSP_CLIENT = 192.168.0.10
.PHONY: upload
upload: all
	cat $(BIN) | ssh $(EDKDSP_SERVER) "curl -n -T - -Q \"-SITE CHMOD 777 /tmp/$(BIN)\" ftp://$(EDKDSP_CLIENT)/tmp/$(BIN)"

.PHONY: run
run: all
	./$(BIN)

.PHONY: valgrind
valgrind: all
	valgrind --leak-check=full ./$(BIN)

.PHONY: time
time: all
	\time -- ./$(BIN)
