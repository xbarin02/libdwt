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

const float *dwt_util_find_max_pos_s(
	// input
	const void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	// output
	int *pos_x,
	int *pos_y
);

#endif
