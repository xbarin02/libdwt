#ifndef FIX_H
#define FIX_H

#include <math.h>
#include <stdint.h>
#include "inline.h"

#define FIX32_T int32_t
#define FIX32_M 15
#define FIX32_N 16
#define FIX32_ONE (((FIX32_T)1) << FIX32_N)
#define FIX32_HALF (((FIX32_T)1) << (FIX32_N-1))

#if 0
#define FIX16_T int16_t
#define FIX16_M 7
#define FIX16_N 8
#define FIX16_ONE (((FIX16_T)1) << FIX16_N)
#define FIX16_HALF (((FIX16_T)1) << (FIX16_N-1))
#else
#define FIX16_T int16_t
#define FIX16_M 6
#define FIX16_N 9
#define FIX16_ONE (((FIX16_T)1) << FIX16_N)
#define FIX16_HALF (((FIX16_T)1) << (FIX16_N-1))
#endif

UNUSED_FUNC
static
FIX32_T conv_float32_to_fix32(float x)
{
	return (FIX32_T)roundf( x * FIX32_ONE );
}

UNUSED_FUNC
static
FIX16_T conv_float32_to_fix16(float x)
{
	return (FIX16_T)roundf( x * FIX16_ONE );
}

UNUSED_FUNC
static
float conv_fix32_to_float32(FIX32_T x)
{
	return (float)x / FIX32_ONE;
}

UNUSED_FUNC
static
float conv_fix16_to_float32(FIX16_T x)
{
	return (float)x / FIX16_ONE;
}

UNUSED_FUNC
static
FIX32_T fix32_mul(FIX32_T x, FIX32_T y)
{
	// FIXME HACK: int64_t
	return ( (int64_t)x * y + FIX32_HALF ) >> FIX32_N;
}

UNUSED_FUNC
static
FIX16_T fix16_mul(FIX16_T x, FIX16_T y)
{
	// FIXME HACK: int32_t
	return ( (int32_t)x * y + FIX16_HALF ) >> FIX16_N;
}

#endif
