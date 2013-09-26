/**
 * @brief Utility functions.
 */

#ifndef UTIL_H
#define UTIL_H

/**
 * @brief Convolution.
 *
 * @warning experimental
 */
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
);

#endif
