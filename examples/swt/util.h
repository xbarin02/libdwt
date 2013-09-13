/**
 * @brief Utility functions.
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

/**
 * @brief One level of forward stationary wavelet transform.
 *
 * @warning experimental
 */
void swt_cdf97_f_ex_stride_s(
	const void *src,	///< pointer to input data
	void *dst_l,		///< pointer to output low-pass coefficients
	void *dst_h,		///< pointer to output high-pass coefficients
	int N,			///< size of src, dst_l and dst_h
	int stride,		///< stride of src, dst_l and dst_h
	int level		///< level of decomposition
);
