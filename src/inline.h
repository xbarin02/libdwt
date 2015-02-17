#ifndef INLINE_H
#define INLINE_H

#include <assert.h>
#include <limits.h>
#include <complex.h>
#include <stdint.h>

#ifdef __GNUC__
	#define UNUSED_FUNC __attribute__ ((unused))
#else
	#define UNUSED_FUNC
#endif

#define UNUSED(expr) do { (void)(expr); } while (0)

#define ALIGNED(align) __attribute((aligned(align)))

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (float *).
 */
UNUSED_FUNC
static
float *addr1_s(
	void *ptr,
	int i,
	int stride
)
{
	return (float *)((char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (const float *).
 */
UNUSED_FUNC
static
const float *addr1_const_s(
	const void *ptr,
	int i,
	int stride
)
{
	return (const float *)((const char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (const float complex *).
 */
UNUSED_FUNC
static
const float complex *addr1_const_cs(
	const void *ptr,
	int i,
	int stride
)
{
	return (const float complex *)((const char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (float complex *).
 */
UNUSED_FUNC
static
float complex *addr1_cs(
	void *ptr,
	int i,
	int stride
)
{
	return (float complex *)((char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (double *).
 */
UNUSED_FUNC
static
double *addr1_d(
	void *ptr,
	int i,
	int stride
)
{
	return (double *)((char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (const double *).
 */
UNUSED_FUNC
static
const double *addr1_const_d(
	const void *ptr,
	int i,
	int stride
)
{
	return (const double *)((const char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (int *).
 */
UNUSED_FUNC
static
int *addr1_i(
	void *ptr,
	int i,
	int stride
)
{
	return (int *)((char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (const int *).
 */
UNUSED_FUNC
static
const int *addr1_const_i(
	const void *ptr,
	int i,
	int stride
)
{
	return (const int *)((const char *)ptr+i*stride);
}

UNUSED_FUNC
static
void *addr2(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (void *)((char *)ptr+y*stride_x+x*stride_y);
}

UNUSED_FUNC
static
const void *addr2_const(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (const void *)((const char *)ptr+y*stride_x+x*stride_y);
}

/**
 * @brief Helper function returning address of given pixel.
 *
 * Evaluate address of (x,y) image element, returns (float *).
 */
UNUSED_FUNC
static
float *addr2_s(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (float *)((char *)ptr+y*stride_x+x*stride_y);
}

UNUSED_FUNC
static
const float *addr2_const_s(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (const float *)((const char *)ptr+y*stride_x+x*stride_y);
}

UNUSED_FUNC
static
const float complex *addr2_const_cs(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (const float complex *)((const char *)ptr + y*stride_x + x*stride_y);
}

/**
 * @brief Helper function returning address of given pixel.
 *
 * Evaluate address of (x,y) image element, returns (double *).
 */
UNUSED_FUNC
static
double *addr2_d(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (double *)((char *)ptr+y*stride_x+x*stride_y);
}

UNUSED_FUNC
static
const double *addr2_const_d(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (const double *)((const char *)ptr+y*stride_x+x*stride_y);
}

/**
 * @brief Helper function returning address of given pixel.
 *
 * Evaluate address of (x,y) image element, returns (int *).
 */
UNUSED_FUNC
static
int *addr2_i(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (int *)((char *)ptr+y*stride_x+x*stride_y);
}

UNUSED_FUNC
static
const int *addr2_const_i(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (const int *)((const char *)ptr+y*stride_x+x*stride_y);
}

UNUSED_FUNC
static
int16_t *addr2_i16(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return (int16_t *)((char *)ptr + y*stride_x + x*stride_y);
}

/**
 * @{
 * @brief CDF 9/7 lifting scheme constants
 * These constants are found in S. Mallat. A Wavelet Tour of Signal Processing: The Sparse Way (Third Edition). 3rd edition, 2009 on page 370.
 */
static const int    dwt_cdf97_k_s  =    2;
static const float  dwt_cdf97_p1_s =    1.58613434342059;
static const float  dwt_cdf97_u1_s =   -0.0529801185729;
static const float  dwt_cdf97_p2_s =   -0.8829110755309;
static const float  dwt_cdf97_u2_s =    0.4435068520439;
static const float  dwt_cdf97_s1_s =    1.1496043988602;
static const float  dwt_cdf97_s2_s =  1/1.1496043988602; // FIXME: unnecessary

static const int    dwt_cdf97_k_d  =    2;
static const double dwt_cdf97_p1_d =    1.58613434342059;
static const double dwt_cdf97_u1_d =   -0.0529801185729;
static const double dwt_cdf97_p2_d =   -0.8829110755309;
static const double dwt_cdf97_u2_d =    0.4435068520439;
static const double dwt_cdf97_s1_d =    1.1496043988602;
static const double dwt_cdf97_s2_d =  1/1.1496043988602; // FIXME: unnecessary
/**@}*/

/**
 * @{
 * @brief CDF 5/3 lifting scheme constants
 * These constants are found in S. Mallat. A Wavelet Tour of Signal Processing: The Sparse Way (Third Edition). 3rd edition, 2009 on page 369.
 */
static const int    dwt_cdf53_k_s  =    1;
static const float  dwt_cdf53_p1_s =    0.5;
static const float  dwt_cdf53_u1_s =    0.25;
static const float  dwt_cdf53_s1_s =    1.41421356237309504880;
static const float  dwt_cdf53_s2_s =    0.70710678118654752440; // FIXME: unnecessary

static const int    dwt_cdf53_k_d  =    1;
static const double dwt_cdf53_p1_d =    0.5;
static const double dwt_cdf53_u1_d =    0.25;
static const double dwt_cdf53_s1_d =    1.41421356237309504880;
static const double dwt_cdf53_s2_d =    0.70710678118654752440; // FIXME: unnecessary
/**@}*/

/**
 * @returns (int)ceil(x/(double)y)
 */
UNUSED_FUNC
static
int ceil_div(
	int x,
	int y
)
{
	return (x + y - 1) / y;
}

/**
 * @returns (int)floor(x/(double)y)
 */
UNUSED_FUNC
static
int floor_div(
	int x,
	int y
)
{
	return x / y;
}

/**
 * @brief Minimum of two integers.
 */
UNUSED_FUNC
static
int min(
	int a,
	int b
)
{
	return a > b ? b : a;
}

/**
 * @brief Maximum of two integers.
 */
UNUSED_FUNC
static
int max(
	int a,
	int b
)
{
	return a > b ? a : b;
}

/**
 * @brief Power of two using greater or equal to x, i.e. 2^(ceil(log_2(x)).
 */
UNUSED_FUNC
static
int pow2_ceil_log2(
	int x
)
{
	assert( x > 0 );

	x--;

	unsigned shift = 1;

	while(shift < sizeof(int) * CHAR_BIT)
	{
		x |= x >> shift;
		shift <<= 1;
	}

	x++;

	return x;
}

/**
 * @brief Number of 1-bits in x, in parallel.
 */
UNUSED_FUNC
static
int bits(
	unsigned x
)
{
	x -= x >> 1 & (unsigned)~(unsigned)0/3;
	x = (x & (unsigned)~(unsigned)0/15*3) + (x >> 2 & (unsigned)~(unsigned)0/15*3);
	x = (x + (x >> 4)) & (unsigned)~(unsigned)0/255*15;
	return (x * ((unsigned)~(unsigned)0/255)) >> (sizeof(unsigned) - 1) * CHAR_BIT;
}

/**
 * @brief Smallest integer not less than the base 2 logarithm of x, i.e. ceil(log_2(x)).
 * @returns (int)ceil(log2(x))
 */
UNUSED_FUNC
static
int ceil_log2(
	int x
)
{
	return bits(pow2_ceil_log2(x) - 1);
}

/**
 * @returns (int)ceil(i/(double)(1<<j))
 */
UNUSED_FUNC
static
int ceil_div_pow2(
	int i,
	int j
)
{
	return (i + (1 << j) - 1) >> j;
}

/**
 * @returns (int)floor(i/(double)(1<<j))
 */
UNUSED_FUNC
static
int floor_div_pow2(
	int i,
	int j
)
{
	return i >> j;
}

/**
 * @brief returns closest even (2^1) integer not larger than x; works also for negative numbers
 */
UNUSED_FUNC
static
int to_even(
	int x
)
{
	return x & ~1;
}

/**
 * @brief returns closest odd integer not larger than x; works also for negative numbers
 */
UNUSED_FUNC
static
int to_odd(
	int x
)
{
	return x - (1 & ~x);
}

UNUSED_FUNC
static
int up_to_even(
	int x
)
{
	return (x+1) & ~1;
}

/**
 * @brief returns closest integer what is multiple of 4 (2^2) and is not larger than x; works also for negative numbers
 */
UNUSED_FUNC
static
int to_even4(
	int x
)
{
	return x & ~3;
}

UNUSED_FUNC
static
int up_to_mul4(
	int x
)
{
	return (x+3) & ~3;
}

/**
 * @brief returns closest integer what is multiple of 8 (2^3) and is not larger than x; works also for negative numbers
 */
UNUSED_FUNC
static
int to_even8(
	int x
)
{
	return x & ~(8-1);
}

/**
 * @brief Round up to the closest power of two.
 */
UNUSED_FUNC
static
int up_to_pow2(
	int x,
	int j
)
{
	return (x+((1<<j)-1)) & ~((1<<j)-1);
}

/**
 * @brief returns 1 if x is even, 0 otherwise; works also for negative numbers
 */
UNUSED_FUNC
static
int is_even(
	int x
)
{
	return 1 & ~x;
}

/**
 * @returns (i * 2^j) = (i * (1<<j)) = (i << j)
 */
UNUSED_FUNC
static
int mul_pow2(int i, int j)
{
	return i << j;
}

#ifdef __GNUC__
	#define likely(x)   __builtin_expect(!!(x), 1)
	#define unlikely(x) __builtin_expect(!!(x), 0)
#else
	#define likely(x)   (x)
	#define unlikely(x) (x)
#endif

/**
 * @returns (int)floor(x/(double)2)
 */
UNUSED_FUNC
static
int floor_div2(
	int x
)
{
	return x >> 1;
}

/**
 * @returns (int)ceil(x/(double)2)
 */
UNUSED_FUNC
static
int ceil_div2(
	int x
)
{
	return (x + 1) >> 1;
}

UNUSED_FUNC
static
int is_pow2(
	int x
)
{
	return 0 == (x & (x - 1));
}

/**
 * @brief returns 1 if x is odd, 0 otherwise; works also for negative numbers
 */
UNUSED_FUNC
static
int is_odd(
	int x
)
{
	return x & 1;
}

#endif
