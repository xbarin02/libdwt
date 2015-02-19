#ifndef COORDS_H
#define COORDS_H

#include "inline.h"

// symmetric border extension (whole point symmetry)
UNUSED_FUNC
static
int virt2real(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		real *= -1;
	if( real > size-1 )
		real = 2*(size-1) - real;

	return real;
}

// constant padding with first/last value
UNUSED_FUNC
static
int virt2real_copy(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		real = 0;
	if( real > size-1 )
		real = size-1;

	return real;
}

// error outside of image area
UNUSED_FUNC
static
int virt2real_error(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		return -1;
	if( real > size-1 )
		return -1;

	return real;
}

#endif
