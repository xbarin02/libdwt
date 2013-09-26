#ifndef SWT_H
#define SWT_H

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

/**
 * @brief One level of forward stationary wavelet transform.
 *
 * @warning experimental
 */
void swt_cdf53_f_ex_stride_s(
	const void *src,	///< pointer to input data
	void *dst_l,		///< pointer to output low-pass coefficients
	void *dst_h,		///< pointer to output high-pass coefficients
	int N,			///< size of src, dst_l and dst_h
	int stride,		///< stride of src, dst_l and dst_h
	int level		///< level of decomposition
);

#endif
