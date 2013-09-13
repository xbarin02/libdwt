/**
 * @brief Utility functions.
 */

#include "util.h"
#include "signal.h"

#define sizeof_arr(a) (sizeof(a)/sizeof(*a))

// CDF9/7, low-pass
static const float dwt_cdf97_g_s[9] = { +0.03782846, -0.02384947, -0.11062438, +0.37740287, +0.85269880, +0.37740287, -0.11062438, -0.02384947, +0.03782846 };

// CDF9/7, high-pass
static const float dwt_cdf97_h_s[7] = { +0.06453887, -0.04068942, -0.41809219, +0.78848559, -0.41809219, -0.04068942, +0.06453887 };

void dwt_util_convolve1_s(
	// output response
	void *y_ptr,
	int y_stride,
	int y_size,
	int y_center,
	// input signal
	const void *x_ptr,
	int x_stride,
	int x_size,
	int x_center,
	// kernel
	const void *g_ptr,
	int g_stride,
	int g_size,
	int g_center,
	// parameters
	int y_downsample_factor,
	int g_upsample_factor
)
{
	signal_t *y_signal = signal_create(y_ptr, y_stride, y_size, y_center);
	signal_const_t *x_signal = signal_const_create(x_ptr, x_stride, x_size, x_center);
	signal_const_t *g_signal = signal_const_create(g_ptr, g_stride, g_size, g_center);

	for(int y_idx = signal_left(y_signal); y_idx <= signal_right(y_signal); y_idx++)
	{
		float *y_coeff = signal_get_s(y_signal, y_idx);

		*y_coeff = 0.f;

		for(int g_idx = signal_const_left(g_signal); g_idx <= signal_const_right(g_signal); g_idx++)
		{
			float x_coeff  = *signal_const_get_s(x_signal, y_downsample_factor*y_idx - g_upsample_factor*g_idx);
			float g_coeff  = *signal_const_get_s(g_signal, g_idx);

			*y_coeff += x_coeff * g_coeff;
		}
	}

	signal_destroy(y_signal);
	signal_const_destroy(x_signal);
	signal_const_destroy(g_signal);
}

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
