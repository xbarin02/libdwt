API_PATH=../../api

OUT=fw_bce2_eval_op.h

SYSINC=$(shell dirname $(shell which pblaze-cc))/../target/include
SYSLIB=$(shell dirname $(shell which pblaze-cc))/../target/lib-pb3a

#   -regalloc                           - Register allocator to use
#     =default                          -   pick register allocator based on -O option
#     =basic                            -   basic register allocator
#     =fast                             -   fast register allocator
#     =greedy                           -   greedy register allocator
#     =linearscan                       -   linear scan register allocator
#     =pbqp                             -   PBQP register allocator
LLCFLAGS=-regalloc=pbqp

.PHONY: clean all

all: $(OUT)

clean:
	-$(RM) $(OUT)

%.O0.ll: %.c
	pblaze-clang -Wall -ccc-host-triple pblaze -ffunction-sections -I$(API_PATH)/20-pb-firmware -I$(SYSINC) -S -emit-llvm  $< -o $@

%.O3.ll: %.O0.ll
	pblaze-opt -O3 -S $< -o $@

%.s: %.ll
	pblaze-llc $(LLCFLAGS) -asm-verbose -mcpu=pb3a -ffunction-sections $< -o $@

%.o: %.O3.s
	pblaze-llvm-mc -triple=pblaze -mcpu=pb3a -filetype=obj $< -o $@

%.o: %.c

%.elf: %.o
	pblaze-ld -L$(SYSLIB) --gc-sections $(SYSLIB)/startup.o $< $(API_PATH)/20-pb-firmware/pbbcelib.o -lpblibc -o $@

%.bin : %.elf
	objcopy -I elf32-little -O binary -j .text $^ $@
	@if [ `stat -c%s $@` -ge 4096 ]; then echo -e "\e[0;31mERROR: Firmware image file $@ is bigger than 1024 words (4kB)!!\e[00m"; fi

%.h : %.bin
	pblaze-bincopy --format=c $< $@
