#ifndef DWT_SYM_MS_H
#define DWT_SYM_MS_H

#include <stdio.h> // FILE

/**
 * @brief Multi-scale single-loop forward implementation using SSE vectorized core.
 *
 * @warning experimental
 */
void ms_cdf97_2f_dl_4x4_s(
	int size_x,
	int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	int J
);

/**
 * @brief Multi-scale transform. Several levels of the transform are fused in a single loop.
 * @warning experimental
 */
void ms_cdf97_2f_dl_4x4_fused_s(
	int size_x,
	int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	int J
);

/**
 * @brief Multi-scale transform. Several levels of the transform are fused in a single loop.
 * @warning experimental
 */
void ms_cdf97_2f_dl_4x4_fused2_s(
	int size_x,
	int size_y,
	void *src,
	int src_stride_x,
	int src_stride_y,
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	int J
);

void dwt_util_perf_ms_cdf97_2f_dl_4x4_s(
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

void dwt_util_measure_perf_ms_cdf97_2f_dl_4x4_s(
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

void cdf97_2f_dl_2x2_s(
	int size_x,
	int size_y,
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y
);

void dwt_cdf97_2f_dl_2x2_s(
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

void ms_cdf97_2f_dl_2x2_s(
	int size_x,
	int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	int J
);

void dwt_util_perf_ms_cdf97_2f_dl_2x2_s(
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

void dwt_util_measure_perf_ms_cdf97_2f_dl_2x2_s(
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
