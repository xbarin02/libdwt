#ifndef DWT_SYM_H
#define DWT_SYM_H

#include <stdio.h> // FILE

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

/**
 * @brief Inverse DWT with CDF 9/7 over SP-FP, in-place subband organization.
 *
 * This function can work with in-place data (@p src == @p dst) as well as with not-in-place data (@p src != @p dst).
 * The single-loop core approach is used in in the heart of the function.
 * This approach is implemented using the @f$ 4\times4 @f$ core with vertical vectorization.
 * The vectorized core is written using SSE instruction set.
 * The image borders are extended using a virtual symmetric padding.
 *
 * @warning experimental
 */
void cdf97_2i_dl_4x4_s(
	int size_x,
	int size_y,
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y
);

/**
 * @brief Performance test.
 *
 * @warning experimental
 */
void dwt_util_perf_cdf97_2f_dl_4x4_s(
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs,
	int flush,
	int type
);

/**
 * @brief Measure performance.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_2f_dl_4x4_s(
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet, in-place version.
 *
 * @note for compatibility with @ref dwt_cdf97_2f_inplace_s
 * @warning experimental
 */
void dwt_cdf97_2f_dl_4x4_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_x,		///< width of nested image (in elements)
	int size_y,		///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 9/7 wavelet, in-place version.
 *
 * @note for compatibility with @ref dwt_cdf97_2i_inplace_s
 * @warning experimental
 */
void dwt_cdf97_2i_dl_4x4_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_x,		///< width of nested image (in elements)
	int size_y,		///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

void dwt_util_perf_dwt_cdf97_2f_dl_4x4_s(
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs,
	int flush,
	int type,
	int J
);

void dwt_util_measure_perf_dwt_cdf97_2f_dl_4x4_s(
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data,
	int J
);

void dwt_util_measure_perf_wrapper_cdf97_2_inplace_sep_s(
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data,
	int J
);

#endif
