#include "swt.h"
#include "util.h"

#define sizeof_arr(a) (sizeof(a)/sizeof(*a))

// CDF 9/7, low-pass
static const float dwt_cdf97_g_s[9] = { +0.03782846, -0.02384947, -0.11062438, +0.37740287, +0.85269880, +0.37740287, -0.11062438, -0.02384947, +0.03782846 };

// CDF 9/7, high-pass
static const float dwt_cdf97_h_s[7] = { +0.06453887, -0.04068942, -0.41809219, +0.78848559, -0.41809219, -0.04068942, +0.06453887 };

// CDF 5/3, low-pass
static const float dwt_cdf53_g_s[5] = { -0.17677669, +0.35355338, +1.06066012, +0.35355338, -0.17677669 };

// CDF 5/3, high-pass
static const float dwt_cdf53_h_s[3] = { -0.35355338, +0.70710677, -0.35355338 };

void swt_cdf97_f_ex_stride_s(
	const void *src,
	void *dst_l,
	void *dst_h,
	int N,
	int stride,
	int level
)
{
	// filter src => dst_l with g upsampled by factor 2^l
	dwt_util_convolve1_s(
		// output response
		dst_l,
		stride,
		N,
		N/2,
		// input signal
		src,
		stride,
		N,
		N/2,
		// kernel
		dwt_cdf97_g_s,
		sizeof(float),
		sizeof_arr(dwt_cdf97_g_s),
		sizeof_arr(dwt_cdf97_g_s)/2,
		// parameters
		1,
		1<<level
	);

	// filter src => dst_h with h upsampled by factor 2^l
	dwt_util_convolve1_s(
		// output response
		dst_h,
		stride,
		N,
		N/2,
		// input signal
		src,
		stride,
		N,
		N/2,
		// kernel
		dwt_cdf97_h_s,
		sizeof(float),
		sizeof_arr(dwt_cdf97_h_s),
		sizeof_arr(dwt_cdf97_h_s)/2,
		// parameters
		1,
		1<<level
	);
}

void swt_cdf53_f_ex_stride_s(
	const void *src,
	void *dst_l,
	void *dst_h,
	int N,
	int stride,
	int level
)
{
	// filter src => dst_l with g upsampled by factor 2^l
	dwt_util_convolve1_s(
		// output response
		dst_l,
		stride,
		N,
		N/2,
		// input signal
		src,
		stride,
		N,
		N/2,
		// kernel
		dwt_cdf53_g_s,
		sizeof(float),
		sizeof_arr(dwt_cdf53_g_s),
		sizeof_arr(dwt_cdf53_g_s)/2,
		// parameters
		1,
		1<<level
	);

	// filter src => dst_h with h upsampled by factor 2^l
	dwt_util_convolve1_s(
		// output response
		dst_h,
		stride,
		N,
		N/2,
		// input signal
		src,
		stride,
		N,
		N/2,
		// kernel
		dwt_cdf53_h_s,
		sizeof(float),
		sizeof_arr(dwt_cdf53_h_s),
		sizeof_arr(dwt_cdf53_h_s)/2,
		// parameters
		1,
		1<<level
	);
}
