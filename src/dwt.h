/**
 * @brief DWT related functions
 */

#ifndef DWT_H
#define DWT_H

#include <stdio.h>
#include "dwt-core.h"

enum dwt_alg {
	DWT_ALG_DL = 0,				// naive/separable
	DWT_ALG_SL_DL = 1,			// SL, DL
	DWT_ALG_SL_SDL = 2,			// SL, SDL
	DWT_ALG_SL_CORE_DL,			// core: DL
	DWT_ALG_SL_CORE_DL_SSE,			// core: DL+SSE
	DWT_ALG_SL_CORE_SDL,			// core: SDL
	DWT_ALG_SL_CORE_SDL_SSE,		// core: SDL+SSE
	DWT_ALG_SL_CORE_DL_SC,			// core: DL merged scaling
	DWT_ALG_SL_CORE_DL_SC_SSE,
	DWT_ALG_SL_CORE_SDL_SC,			// core: SDL, merged scaling
	DWT_ALG_SL_CORE_SDL_SC_SSE,		// core: SDL, merged scaling
	DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0,	// w/o offset 1
	DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1,	// offset=1
	DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0_OVL1,	// off=1 overlay=1(C) src==dst
	DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1_OVL1,
	DWT_ALG_SL_CORE_DL_SC_SSE_OFF1,		// coreSL.DL SSE OFF=1 OVL=0/B THREADS
	DWT_ALG_SL_CORE_DL_SC_SSE_OFF1_4X4,
	DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1_6X2,
	DWT_ALG_LAST
};

/**
 * @brief Allocate new image and copy ... inside of the new one.
 * @warning experimental
 */
void dwt_util_wrap_image(
	void *src_ptr,
	int src_size_x,
	int src_size_y,
	int src_stride_x,
	int src_stride_y,
	enum dwt_alg alg,
	void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y,
	int *dst_stride_x,
	int *dst_stride_y,
	int *offset_x,
	int *offset_y,
	void **view_ptr,
	int opt_stride
);

int dwt_alg_get_shift(
	enum dwt_alg alg
);

void dwt_util_perf_cdf97_2_inplace_alg_s(
	enum dwt_alg alg,
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs
);

void dwt_util_measure_perf_cdf97_2_inplace_alg_s(
	enum dwt_alg alg,
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data
);

void dwt_util_alloc_zero(
	void **ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void dwt_util_copy2_s(
	const void *src,
	int src_stride_x,
	int src_stride_y,
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
);

void dwt_util_perf_cdf97_2_inplaceB_alg_s(
	enum dwt_alg alg,
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs,
	int flush,
	int overlap
);

void dwt_util_measure_perf_cdf97_2_inplaceABC_alg_s(
	enum dwt_alg alg,
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data,
	int overlap // -1=A 0=B 1=C
);

#ifdef __SSE__
void fdwt_cdf97_diag_core2x2_sse_s(
	float *ptr_y0_x0,
	float *ptr_y0_x1,
	float *ptr_y1_x0,
	float *ptr_y1_x1,
	float *out_y0_x0,
	float *out_y0_x1,
	float *out_y1_x0,
	float *out_y1_x1,
	float *buff_y0,
	float *buff_y1,
	float *buff_x0,
	float *buff_x1
);
#endif

#ifdef __SSE__
void idwt_cdf97_diag_core2x2_sse_s(
	float *ptr_y0_x0,
	float *ptr_y0_x1,
	float *ptr_y1_x0,
	float *ptr_y1_x1,
	float *out_y0_x0,
	float *out_y0_x1,
	float *out_y1_x0,
	float *out_y1_x1,
	float *buff_y0,
	float *buff_y1,
	float *buff_x0,
	float *buff_x1
);
#endif

void dwt_util_perf_cdf97_2_inplace_diag2x2_frame4_s(
	enum dwt_alg alg,
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs,
	int flush
);

void dwt_util_perf_cdf97_2_inplace_new_s(
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs,
	int flush,
	// new
	enum order order,
	int strip_x,
	int strip_y
);

void dwt_util_perf_cdf97_2_inplace_new_VERT_s(
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs,
	int flush,
	// new
	enum order order,
	int strip_x,
	int strip_y
);


void dwt_util_measure_perf_cdf97_2_inplace_new_s(
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data,
	enum order order,
	int strip_x,
	int strip_y
);

void dwt_util_measure_perf_cdf97_2_inplace_new_VERT_s(
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data,
	enum order order,
	int strip_x,
	int strip_y
);

#endif
