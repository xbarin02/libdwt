# get architecture
ARCH ?= $(shell uname -m)

# ARCH was specified as empty value
ifeq ($(ARCH),)
	override ARCH = $(shell uname -m)
endif

# common part
CROSS_COMPILE ?= 
CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
# FLAGS = -fprofile-generate
# FLAGS = -fprofile-use
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O3 $(FLAGS)
CXXFLAGS = -std=c++98 -pedantic -Wall -Wextra -O2 $(FLAGS)
LDLIBS = -lm
LDFLAGS = $(FLAGS)

# ASVP platform
ifeq ($(ARCH),asvp)
	# asvp-demo_v0
	CROSS_COMPILE = microblaze-uclinux-
	CFLAGS += -mno-xl-soft-mul -mhard-float -mcpu=v8.00.b -DEMBED -Dlinux -D__linux__ -Dunix -D__uClinux__ -DLINUX -I$(ROOT)/api/22-mb-petalinux/libwal -I$(ROOT)/api/22-mb-petalinux/libbce_config_step4 -Wno-unknown-pragmas -D__asvp__
	LDFLAGS += -L$(ROOT)/api/22-mb-petalinux/libwal -L$(ROOT)/api/22-mb-petalinux/libbce_config_step4
	LDLIBS += -lwal -lbce_config
endif

# MicroBlaze
ifeq ($(ARCH),microblaze)
	CROSS_COMPILE = microblaze-uclinux-
	CFLAGS += -mno-xl-soft-mul -mhard-float -mcpu=v8.00.b -Wno-unknown-pragmas
endif

# AMD64
ifeq ($(ARCH),x86_64)
	CROSS_COMPILE = 
	CFLAGS += -fopenmp -fPIC
	CFLAGS += -O3 -ftree-vectorize
	LDFLAGS += -fopenmp
	LDLIBS += -lrt
endif

# ARM11 (Raspberry Pi)
ifeq ($(ARCH),armv6l)
	CROSS_COMPILE = 
	CFLAGS += -O3 -fPIC -Wno-unknown-pragmas
	LDLIBS += -lrt
endif

# Cortex-A8 (N900)
ifeq ($(ARCH),armv7l)
	CROSS_COMPILE = 
	CFLAGS += -fopenmp -fPIC
#	CFLAGS += -O3 -ftree-vectorize -mfpu=neon -march=armv7-a -mfloat-abi=softfp -mvectorize-with-neon-quad -funsafe-math-optimizations
#	CFLAGS += -O3 -ftree-vectorize -mfpu=neon -march=armv7-a -mvectorize-with-neon-quad -funsafe-math-optimizations
	CFLAGS += -O3 -ftree-vectorize -mfpu=neon -mcpu=cortex-a7 -mtune=cortex-a7 -mvectorize-with-neon-quad -funsafe-math-optimizations
	LDFLAGS += -fopenmp
	LDLIBS += -lrt
endif

ifeq ($(BUILD),release)
	CFLAGS += -DNDEBUG
endif

ifeq ($(BUILD),debug)
	CFLAGS += -DDEBUG -g
endif

ifeq ($(LINKER),static)
	LDFLAGS += -static-libgcc -static
endif
