include ../common.mk

.PHONY: all
all: html

html:
	PACKAGE_VERSION=$(PACKAGE_VERSION) doxygen Doxyfile

.PHONY: clean
clean:
	-$(RM) -r html
