#ifndef INLINE_H
#define INLINE_H

#ifdef __GNUC__
	#define UNUSED_FUNC __attribute__ ((unused))
#else
	#define UNUSED_FUNC
#endif

#define UNUSED(expr) do { (void)(expr); } while (0)

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

#endif
