/**
 * @brief signal_t infrastructure
 */

#include "signal.h"

#include <stdlib.h> // malloc, free

struct signal_t {
	void *ptr;		// pointer to first element (float)
	int stride;		// in bytes
	unsigned int size;	// in elements
	unsigned int center;	// in elements
};

struct signal_const_t {
	const void *ptr;	// pointer to first element (float)
	int stride;		// in bytes
	unsigned int size;	// in elements
	unsigned int center;	// in elements
};

static
float *addr1_s(
	void *ptr,
	int i,
	int stride
)
{
	return (float *)((char *)ptr+i*stride);
}

static
const float *addr1_const_s(
	const void *ptr,
	int i,
	int stride
)
{
	return (const float *)((const char *)ptr+i*stride);
}

static
int saturate_i(int val, int lo, int hi)
{
	if( val < lo )
		return lo;
	if( val > hi )
		return hi;
	return val;
}

signal_t *signal_create(void *ptr, int stride, unsigned int size, unsigned int center)
{
	signal_t *signal = malloc(sizeof(signal_t));

	*signal = (signal_t){ ptr, stride, size, center };

	return signal;
}

signal_const_t *signal_const_create(const void *ptr, int stride, unsigned int size, unsigned int center)
{
	signal_const_t *signal = malloc(sizeof(signal_const_t));

	*signal = (signal_const_t){ ptr, stride, size, center };

	return signal;
}

void signal_destroy(signal_t *signal)
{
	free(signal);
}

void signal_const_destroy(signal_const_t *signal)
{
	free(signal);
}

float *signal_get_s(const signal_t *signal, int index)
{
	unsigned int index0 = saturate_i(index + signal->center, 0, signal->size-1);

	return addr1_s(signal->ptr, index0, signal->stride);
}

const float *signal_const_get_s(const signal_const_t *signal, int index)
{
	unsigned int index0 = saturate_i(index + signal->center, 0, signal->size-1);

	return addr1_const_s(signal->ptr, index0, signal->stride);
}

int signal_left(const signal_t *signal)
{
	return -signal->center;
}

int signal_const_left(const signal_const_t *signal)
{
	return -signal->center;
}

int signal_right(const signal_t *signal)
{
	return signal->size - signal->center - 1;
}

int signal_const_right(const signal_const_t *signal)
{
	return signal->size - signal->center - 1;
}
