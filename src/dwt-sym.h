#ifndef DWT_SYM_H
#define DWT_SYM_H

/**
 * @brief Forward DWT with CDF 9/7 over SP-FP, in-place subband organization.
 *
 * This function can work with in-place data (@p src == @p dst) as well as with not-in-place data (@p src != @p dst).
 * The single-loop core approach is used in in the heart of the function.
 * This approach is implemented using the @f$ 4\times4 @f$ core with vertical vectorization.
 * The vectorized core is written using SSE instruction set.
 * The image borders are extended using a virtual symmetric padding.
 *
 * @warning experimental
 */
void cdf97_2f_dl_4x4_s(
	int size_x,
	int size_y,
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y
);

#endif
