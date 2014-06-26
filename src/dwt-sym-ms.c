#include "dwt-sym-ms.h"
#include "libdwt.h"
#include "inline.h"
#include <math.h>
#ifdef __SSE__
	#include <xmmintrin.h>
#endif
#define MEASURE_FACTOR 1
#define MEASURE_PER_PIXEL

// vert_2x4
#ifdef __SSE__
static
void vert_2x4(
	// left input column [4]
	__m128 in0,
	// right input column [4]
	__m128 in1,
	// output 0 [4]
	__m128 *out0,
	// output 1 [4]
	__m128 *out1,
	// 4x buffer "L" with stride = (1*4) * sizeof(float)
	float *buff
)
{
	// weights
	const __m128 w0 = { +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s };
	const __m128 w1 = { -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s };
	const __m128 w2 = { +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s };
	const __m128 w3 = { -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s };

	// variables
	__m128 l0, l1, l2, l3;
	__m128 c0, c1, c2, c3;
	__m128 r0, r1, r2, r3;
	__m128 x0, x1;
	__m128 y0, y1;

	// load "L"
	l0 = _mm_load_ps(&buff[0*(1*4)]);
	l1 = _mm_load_ps(&buff[1*(1*4)]);
	l2 = _mm_load_ps(&buff[2*(1*4)]);
	l3 = _mm_load_ps(&buff[3*(1*4)]);
	//_MM_TRANSPOSE4_PS(l0, l1, l2, l3);

	// inputs
	x0 = in0;
	x1 = in1;

	// shuffles
	y0 = l0;
	c0 = l1;
	c1 = l2;
	c2 = l3;
	c3 = x0;

	// operation
	r3 = x1;
	r2 = c3 + w3 * (l3 + r3);
	r1 = c2 + w2 * (l2 + r2);
	r0 = c1 + w1 * (l1 + r1);
	y1 = c0 + w0 * (l0 + r0);

	// update
	l0 = r0;
	l1 = r1;
	l2 = r2;
	l3 = r3;

	// outputs
	*out0 = y0;
	*out1 = y1;

	// store "L"
	//_MM_TRANSPOSE4_PS(l0, l1, l2, l3);
	_mm_store_ps(&buff[0*(1*4)], l0);
	_mm_store_ps(&buff[1*(1*4)], l1);
	_mm_store_ps(&buff[2*(1*4)], l2);
	_mm_store_ps(&buff[3*(1*4)], l3);
}
#endif

#define CDF97_S1_FWD_SQR_S 1.3215902738787f
#define CDF97_S1_INV_SQR_S 0.75666416420054f

#define CORE_4X4_SCALE(t0, t1, t2, t3) \
do { \
	(t0) *= (const __m128){ (CDF97_S1_INV_SQR_S),                  1.f, (CDF97_S1_INV_SQR_S),                  1.f }; \
	(t1) *= (const __m128){                  1.f, (CDF97_S1_FWD_SQR_S),                  1.f, (CDF97_S1_FWD_SQR_S) }; \
	(t2) *= (const __m128){ (CDF97_S1_INV_SQR_S),                  1.f, (CDF97_S1_INV_SQR_S),                  1.f }; \
	(t3) *= (const __m128){                  1.f, (CDF97_S1_FWD_SQR_S),                  1.f, (CDF97_S1_FWD_SQR_S) }; \
} while(0)

#define CORE_4X4_CALC(t0, t1, t2, t3, buff_h, buff_v) \
do { \
	vert_2x4((t0), (t1), &(t0), &(t1), (buff_h)); \
	vert_2x4((t2), (t3), &(t2), &(t3), (buff_h)); \
	\
	_MM_TRANSPOSE4_PS((t0), (t1), (t2), (t3)); \
	\
	vert_2x4((t0), (t1), &(t0), &(t1), (buff_v)); \
	vert_2x4((t2), (t3), &(t2), &(t3), (buff_v)); \
	\
	CORE_4X4_SCALE((t0), (t1), (t2), (t3)); \
} while(0)

static
int virt2real(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		real *= -1;
	if( real > size-1 )
		real = 2*(size-1) - real;

#if 0
	if( real < 0 || real > size-1 )
	{
		dwt_util_error("out of range\n");
	}
#endif
	return real;
}

static
int virt2real_error(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		return -1;
	if( real > size-1 )
		return -1;

	return real;
}

static
void unified_4x4(
	int x, int y,
	int size_x,
	int size_y,
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	void *buffer_x,
	void *buffer_y
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	__m128 t[4];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real(x, xx, 0, size_x);
			const int pos_y = virt2real(y, yy, 0, size_y);

			t[xx][yy] = *addr2_const_s(src_ptr, pos_y, pos_x, src_stride_x, src_stride_y);
		}
	}

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
}

// store LL and HL/LH/HH separately
static
void unified_4x4_separately(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	// output LL
	void *low_ptr,
	int low_stride_x,
	int low_stride_y,
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	__m128 t[4];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real(x, xx, 0, size_x);
			const int pos_y = virt2real(y, yy, 0, size_y);

			t[xx][yy] = *addr2_const_s(src_ptr, pos_y, pos_x, src_stride_x, src_stride_y);
		}
	}

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			if( (xx&1) && (yy&1) )
				*addr2_s(low_ptr, pos_y, pos_x, low_stride_x, low_stride_y) = t[yy][xx];
			else
				*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
}

// FIXME: lag
static
void ms_loop_unified_4x4(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int size_x, int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	float *buffer_x,
	float *buffer_y,
	int J,
	int super_x,
	int super_y,
	int buffer_offset
)
{
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	// core size
	const int step_y = 4;
	const int step_x = 4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

#if 1
	// order=horizontal
	for(int y = base_y; y < stop_y; y += step_y)
	{
		for(int x = base_x; x < stop_x; x += step_x)
		{
#else
	// order=vertical
	for(int x = base_x; x < stop_x; x += step_x)
	{
		for(int y = base_y; y < stop_y; y += step_y)
		{
#endif
			for(int j = 0; j < J; j++)
			{
				// mod == 0
#if 0
				if( ((x)%(step_x<<j)) == (0) && ((y)%(step_y<<j)) == (0) )
#else
				if( (x&((4<<j)-1)) == (0) && (y&((4<<j)-1)) == (0) )
#endif
				{
					// FIXME: what distance between read and write head is really needed?
					// 18 (ok), 4 (minimum), 8 (fixed left/top)
					const int lag = 8;

					// TODO: try to hardcode -(step_x-1) - lag as constant
					const int x_j = ceil_div_pow2(x,j) -(step_x-1) - lag;
					const int y_j = ceil_div_pow2(y,j) -(step_y-1) - lag;

					const int size_x_j = ceil_div_pow2(size_x,j);
					const int size_y_j = ceil_div_pow2(size_y,j);
					const int stride_x_j = mul_pow2(stride_x,j);
					const int stride_y_j = mul_pow2(stride_y,j);

					unified_4x4(
						x_j, y_j,
						size_x_j, size_y_j,
						ptr, stride_x_j, stride_y_j,
						ptr, stride_x_j, stride_y_j,
						// TODO: try to use incremental pointers
						buffer_x + j*buffer_x_elems + (buffer_offset+x_j)*buff_elem_size,
						buffer_y + j*buffer_y_elems + (buffer_offset+y_j)*buff_elem_size
					);
				}
			}
		}
	}
}

// FIXME: lag
static
void ms_loop_unified_4x4_buffers(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int size_x, int size_y,
	void *ptr,	// unused
	int stride_x,
	int stride_y,
	float *buffer_x,
	float *buffer_y,
	int J,
	int super_x,
	int super_y,
	int buffer_offset,
	void *ib[J+1]
)
{
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	// core size
	const int step_y = 4;
	const int step_x = 4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

#if 1
	// order=horizontal
	for(int y = base_y; y < stop_y; y += step_y)
	{
		for(int x = base_x; x < stop_x; x += step_x)
		{
#else
	// order=vertical
	for(int x = base_x; x < stop_x; x += step_x)
	{
		for(int y = base_y; y < stop_y; y += step_y)
		{
#endif
			for(int j = 0; j < J; j++)
			{
				// mod == 0
#if 0
				if( ((x)%(step_x<<j)) == (0) && ((y)%(step_y<<j)) == (0) )
#else
				if( (x&((4<<j)-1)) == (0) && (y&((4<<j)-1)) == (0) )
#endif
				{
					// FIXME: what distance between read and write head is really needed?
					// 18 (ok), 4 (minimum), 8 (fixed left/top)
					const int lag = 4;

					// TODO: try to hardcode -(step_x-1) - lag as constant
					const int x_j = ceil_div_pow2(x,j) -(step_x-1) - lag;
					const int y_j = ceil_div_pow2(y,j) -(step_y-1) - lag;

					const int size_x_j = ceil_div_pow2(size_x,j);
					const int size_y_j = ceil_div_pow2(size_y,j);
					const int stride_x_j = mul_pow2(stride_x,j);
					const int stride_y_j = mul_pow2(stride_y,j);

					unified_4x4_separately(
						x_j, y_j,
						size_x_j, size_y_j,
#if 0
						// use several aux. buffers
						ib[j+0], stride_x_j, stride_y_j, // input
						ib[j+1], stride_x_j, stride_y_j, // output LL
#else
						// use single aux. buffer
						ib[0], stride_x_j, stride_y_j, // input
						ib[0], stride_x_j, stride_y_j, // output LL
#endif
						ptr, stride_x_j, stride_y_j, // output HL/LH/HH
						// TODO: try to use incremental pointers
						buffer_x + j*buffer_x_elems + (buffer_offset+x_j)*buff_elem_size,
						buffer_y + j*buffer_y_elems + (buffer_offset+y_j)*buff_elem_size
					);
				}
			}
		}
	}
}

// FIXME: does not work for arbitrary sizes
void ms_cdf97_2f_dl_4x4_s(
	int size_x,
	int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	int J
)
{
	// TODO: assert

	// offset_x, offset_y
// 	const int offset = 1;

	// core size
// 	const int step_y = 4;
// 	const int step_x = 4;

	// vertical vectorization
// 	const int shift = 4;

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buff_guard = 32; // 32: maximal coordinate in negative value (currently -(step_x-1) - lag = -3 -8 = -11)
	const int overlap_L = 4; // 4
	// FIXME: what overlap is really needed?
	// 13<<J ... with lag = 18
	//  7<<J ... with lag =  8 (artifacts)
	//  5<<J ... with lag =  4 (buffers)
	const int overlap_R = 7<<J;

	const int super_x = buff_guard + overlap_L + size_x + overlap_R;
	const int super_y = buff_guard + overlap_L + size_y + overlap_R;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	// alloc buffers
	float buffer_x[J*buffer_x_elems] ALIGNED(16);
	float buffer_y[J*buffer_y_elems] ALIGNED(16);

	const int buffer_offset = buff_guard+overlap_L;

	// unified loop
	{
		ms_loop_unified_4x4(
			/* base */ -overlap_L, -overlap_L, // x, y
			/* stop */ size_x+overlap_R, size_y+overlap_R, // x, y
			/* size */ size_x, size_y,
			ptr,
			stride_x, stride_y,
			buffer_x, buffer_y,
			J,
			super_x, super_y,
			buffer_offset
		);
	}
}

// FIXME: does not work for arbitrary sizes
void ms_cdf97_2f_dl_4x4_buffers_s(
	int size_x,
	int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	int J
)
{
	// TODO: assert

	// offset_x, offset_y
// 	const int offset = 1;

	// core size
// 	const int step_y = 4;
// 	const int step_x = 4;

	// vertical vectorization
// 	const int shift = 4;

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buff_guard = 32; // 32: maximal coordinate in negative value (currently -(step_x-1) - lag = -3 -8 = -11)
	const int overlap_L = 4; // 4
	// FIXME: what overlap is really needed?
	// 13<<J ... with lag = 18
	//  7<<J ... with lag =  8 (artifacts)
	//  5<<J ... with lag =  4 (buffers)
	const int overlap_R = 5<<J;

	const int super_x = buff_guard + overlap_L + size_x + overlap_R;
	const int super_y = buff_guard + overlap_L + size_y + overlap_R;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	// alloc buffers
	float buffer_x[J*buffer_x_elems] ALIGNED(16);
	float buffer_y[J*buffer_y_elems] ALIGNED(16);

	const int buffer_offset = buff_guard+overlap_L;

	// alloc J image buffers
	void *ib[J+1];
	ib[0] = ptr;
	for(int j = 1; j <= J; j++)
	{
		ib[j] = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);
	}

	// unified loop
	{
		ms_loop_unified_4x4_buffers(
			/* base */ -overlap_L, -overlap_L, // x, y
			/* stop */ size_x+overlap_R, size_y+overlap_R, // x, y
			/* size */ size_x, size_y,
			ptr,
			stride_x, stride_y,
			buffer_x, buffer_y,
			J,
			super_x, super_y,
			buffer_offset,
			ib
		);
	}

#if 1
#if 0
	// copy ib[J] => ptr
	// NOTE: coefficiens for different "j" are spread over ib[...]
	for(int j = 1; j <= J; j++)
	{
		// FIXME: +1 -1 problems?
		const int size_x_j = ceil_div_pow2(size_x,j+0);
		const int size_y_j = ceil_div_pow2(size_y,j+0);
		const int stride_x_j = mul_pow2(stride_x,j+0);
		const int stride_y_j = mul_pow2(stride_y,j+0);
		const int stride_x_j2 = mul_pow2(stride_x,j-1);
		const int stride_y_j2 = mul_pow2(stride_y,j-1);

		// FIXME: 3times for HL/LH/HH: (1,0)*stride; (0,1)*stride; (1,1)*stride
		void *src_10 = addr2_s(ib[j], 1, 0, stride_x_j2, stride_y_j2);
		void *src_01 = addr2_s(ib[j], 0, 1, stride_x_j2, stride_y_j2);
		void *src_11 = addr2_s(ib[j], 1, 1, stride_x_j2, stride_y_j2);

		void *dst_10 = addr2_s(ptr, 1, 0, stride_x_j2, stride_y_j2);
		void *dst_01 = addr2_s(ptr, 0, 1, stride_x_j2, stride_y_j2);
		void *dst_11 = addr2_s(ptr, 1, 1, stride_x_j2, stride_y_j2);

		dwt_util_copy3_s(src_10, dst_10, stride_x_j, stride_y_j, stride_x_j, stride_y_j, size_x_j, size_y_j);
		dwt_util_copy3_s(src_01, dst_01, stride_x_j, stride_y_j, stride_x_j, stride_y_j, size_x_j, size_y_j);
		dwt_util_copy3_s(src_11, dst_11, stride_x_j, stride_y_j, stride_x_j, stride_y_j, size_x_j, size_y_j);
	}
#endif
	// NOTE: for "J" copy whole LL: (0,0)*stride
	{
		int j = J;

		const int size_x_j = ceil_div_pow2(size_x,j+0);
		const int size_y_j = ceil_div_pow2(size_y,j+0);
		const int stride_x_j = mul_pow2(stride_x,j+0);
		const int stride_y_j = mul_pow2(stride_y,j+0);

#if 0
		dwt_util_copy3_s(ib[j], ptr, stride_x_j, stride_y_j, stride_x_j, stride_y_j, size_x_j, size_y_j);
#else
		dwt_util_copy3_s(ib[0], ptr, stride_x_j, stride_y_j, stride_x_j, stride_y_j, size_x_j, size_y_j);
#endif
	}
#endif
	// free image buffers
	for(int j = 1; j <= J; j++)
	{
		free(ib[j]);
	}
}

// TODO
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
)
{
	//FUNC_BEGIN;

	assert( M > 0 && N > 0 && fwd_secs && inv_secs );

// 	dwt_util_log(LOG_DBG, "perf. test (%i,%i) N=%i M=%i\n", size_x, size_y, N, M);

	int stride_y = sizeof(float);
	int stride_x = dwt_util_get_stride(stride_y * size_x, opt_stride);

	// pointer to M pointers to image data
	void *ptr[M];

	// template
	void *template;

	// allocate
	template = dwt_util_alloc_image2(
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	// fill with test pattern
	dwt_util_test_image_fill2_s(
		template,
		stride_x,
		stride_y,
		size_x,
		size_y,
		0,
		type
	);

	// allocate M images
	for(int m = 0; m < M; m++)
	{
		ptr[m] = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
		// copy template to ptr[m]
		for(int m = 0; m < M; m++)
		{
			dwt_util_copy_s(template, ptr[m], stride_x, stride_y, size_x, size_y);
		}
	
		// flush memory
		if(flush)
		{
			for(int m = 0; m < M; m++)
			{
				dwt_util_flush_cache(ptr[m], dwt_util_image_size(stride_x, stride_y, size_x, size_y));
			}
		}

		// start timer
		const dwt_clock_t time_fwd_start = dwt_util_get_clock(clock_type);
		// perform M fwd transforms
		for(int m = 0; m < M; m++)
		{
#if 0
			ms_cdf97_2f_dl_4x4_s(
#else
			ms_cdf97_2f_dl_4x4_buffers_s(
#endif
				size_x,
				size_y,
				ptr[m],
				stride_x,
				stride_y,
				J
			);
		}
		// stop timer
		const dwt_clock_t time_fwd_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_fwd_secs = (float)(time_fwd_stop - time_fwd_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_fwd_secs < *fwd_secs )
			*fwd_secs = time_fwd_secs;

		// flush memory
		if(flush)
		{
			for(int m = 0; m < M; m++)
			{
				dwt_util_flush_cache(ptr[m], dwt_util_image_size(stride_x, stride_y, size_x, size_y));
			}
		}

		// start timer
		const dwt_clock_t time_inv_start = dwt_util_get_clock(clock_type);
		// perform M inv transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2i_inplace_s(ptr[m], stride_x, stride_y, size_x, size_y, size_x, size_y, J, 0, 0);
		}
		// stop timer
		const dwt_clock_t time_inv_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_inv_secs = (float)(time_inv_stop - time_inv_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_inv_secs < *inv_secs )
			*inv_secs = time_inv_secs;

		// compare
		for(int m = 0; m < M; m++)
		{
			int err = dwt_util_compare2_destructive_s(
				ptr[m],
				template,
				stride_x,
				stride_y,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			if( err )
			{
				dwt_util_log(LOG_ERR, "perf: [%i] compare failed :(\n", m);
#if 1
				dwt_util_save_to_pgm_s("debug.pgm", 1.0, ptr[m], stride_x, stride_y, size_x, size_y);
#endif
			}
		}
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}
	dwt_util_free_image(&template);

	//FUNC_END;
}

extern const float g_growth_factor_s;

// TODO
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
)
{
	//FUNC_BEGIN;

	assert( min_x > 0 && min_x < max_x );
	assert( M > 0 && N > 0 );
	assert( fwd_plot_data && inv_plot_data );

	const float growth_factor = g_growth_factor_s;

	// for x = min_x to max_x
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
	{
		// y is equal to x
		const int y = x;

		int size_x = x;
		int size_y = y;

		dwt_util_log(LOG_DBG, "performance test for [%ix%i]...\n", size_x, size_y);

		float fwd_secs;
		float inv_secs;

		// call perf()
		dwt_util_perf_ms_cdf97_2f_dl_4x4_s(
			size_x,
			size_y,
			opt_stride,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs,
			1, // flush
			0, // type
			J
		);

#ifdef MEASURE_PER_PIXEL
		const int denominator = x*y;
#else
		const int denominator = 1;
#endif

		// printf into file
		fprintf(fwd_plot_data, "%i\t%.10f\n", x*y, fwd_secs/denominator);
		fprintf(inv_plot_data, "%i\t%.10f\n", x*y, inv_secs/denominator);

	}

	//FUNC_END;
}

// TODO: macro
#ifdef __SSE__
#define CORE_2X2_CALC(float_y0_x0, float_y0_x1, float_y1_x0, float_y1_x1, buff_h, buff_v) \
do { \
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s }; \
	const __m128 v_vertL = { CDF97_S1_INV_SQR_S, 1.f, 0.f, 0.f }; \
	const __m128 v_vertR = { 1.f, CDF97_S1_FWD_SQR_S, 0.f, 0.f }; \
	\
	__m128 t; \
	__m128 x, y, r, c; \
	\
	{ \
		float *l = ((float *)(buff_h)) + 0*(1*4); \
		\
		x[0] = (float_y0_x0); \
		x[1] = (float_y0_x1); \
		\
		y[0] = l[0]; \
		c[0] = l[1]; \
		c[1] = l[2]; \
		c[2] = l[3]; \
		c[3] = x[0]; \
		\
		r[3] = x[1]; \
		r[2] = c[3]+w[3]*(l[3]+r[3]); \
		r[1] = c[2]+w[2]*(l[2]+r[2]); \
		r[0] = c[1]+w[1]*(l[1]+r[1]); \
		y[1] = c[0]+w[0]*(l[0]+r[0]); \
		\
		t[0] = y[0]; \
		t[1] = y[1]; \
		\
		l[0] = r[0]; \
		l[1] = r[1]; \
		l[2] = r[2]; \
		l[3] = r[3]; \
	} \
	\
	{ \
		float *l = ((float *)(buff_h)) + 1*(1*4); \
		\
		x[0] = (float_y1_x0); \
		x[1] = (float_y1_x1); \
		\
		y[0] = l[0]; \
		c[0] = l[1]; \
		c[1] = l[2]; \
		c[2] = l[3]; \
		c[3] = x[0]; \
		\
		r[3] = x[1]; \
		r[2] = c[3]+w[3]*(l[3]+r[3]); \
		r[1] = c[2]+w[2]*(l[2]+r[2]); \
		r[0] = c[1]+w[1]*(l[1]+r[1]); \
		y[1] = c[0]+w[0]*(l[0]+r[0]); \
		\
		t[2] = y[0]; \
		t[3] = y[1]; \
		\
		l[0] = r[0]; \
		l[1] = r[1]; \
		l[2] = r[2]; \
		l[3] = r[3]; \
	} \
	\
	{ \
		float *l = ((float *)(buff_v)) + 0*(1*4); \
		\
		x[0] = t[0]; \
		x[1] = t[2]; \
		\
		y[0] = l[0]; \
		c[0] = l[1]; \
		c[1] = l[2]; \
		c[2] = l[3]; \
		c[3] = x[0]; \
		\
		r[3] = x[1]; \
		r[2] = c[3]+w[3]*(l[3]+r[3]); \
		r[1] = c[2]+w[2]*(l[2]+r[2]); \
		r[0] = c[1]+w[1]*(l[1]+r[1]); \
		y[1] = c[0]+w[0]*(l[0]+r[0]); \
		\
		y[0] *= v_vertL[0]; \
		y[1] *= v_vertL[1]; \
		\
		(float_y0_x0) = y[0]; \
		(float_y1_x0) = y[1]; \
		\
		l[0] = r[0]; \
		l[1] = r[1]; \
		l[2] = r[2]; \
		l[3] = r[3]; \
	} \
	\
	{ \
		float *l = ((float *)(buff_v)) + 1*(1*4); \
		\
		x[0] = t[1]; \
		x[1] = t[3]; \
		\
		y[2] = l[0]; \
		c[0] = l[1]; \
		c[1] = l[2]; \
		c[2] = l[3]; \
		c[3] = x[0]; \
		\
		r[3] = x[1]; \
		r[2] = c[3]+w[3]*(l[3]+r[3]); \
		r[1] = c[2]+w[2]*(l[2]+r[2]); \
		r[0] = c[1]+w[1]*(l[1]+r[1]); \
		y[3] = c[0]+w[0]*(l[0]+r[0]); \
		\
		y[2] *= v_vertR[0]; \
		y[3] *= v_vertR[1]; \
		\
		(float_y0_x1) = y[2]; \
		(float_y1_x1) = y[3]; \
		\
		l[0] = r[0]; \
		l[1] = r[1]; \
		l[2] = r[2]; \
		l[3] = r[3]; \
	} \
} while(0)
#endif

// TODO: function
#ifdef __SSE__
static
void core_2x2_calc(
	float in_y0_x0,
	float in_y0_x1,
	float in_y1_x0,
	float in_y1_x1,
	float *out_y0_x0,
	float *out_y0_x1,
	float *out_y1_x0,
	float *out_y1_x1,
	float *buff_h,
	float *buff_v
)
{
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v_vertL = { CDF97_S1_INV_SQR_S, 1.f, 0.f, 0.f };
	const __m128 v_vertR = { 1.f, CDF97_S1_FWD_SQR_S, 0.f, 0.f };

	__m128 t;
	__m128 x, y, r, c;

	{
		float *l = ((float *)(buff_h)) + 0*(1*4);

		x[0] = (in_y0_x0);
		x[1] = (in_y0_x1);
	
		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		t[0] = y[0];
		t[1] = y[1];

		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	{
		float *l = ((float *)(buff_h)) + 1*(1*4);

		x[0] = (in_y1_x0);
		x[1] = (in_y1_x1);

		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		t[2] = y[0];
		t[3] = y[1];

		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	{
		float *l = ((float *)(buff_v)) + 0*(1*4);

		x[0] = t[0];
		x[1] = t[2];

		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		y[0] *= v_vertL[0];
		y[1] *= v_vertL[1];

		(*out_y0_x0) = y[0];
		(*out_y1_x0) = y[1];

		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	{
		float *l = ((float *)(buff_v)) + 1*(1*4);

		x[0] = t[1];
		x[1] = t[3];

		y[2] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[3] = c[0]+w[0]*(l[0]+r[0]);

		y[2] *= v_vertR[0];
		y[3] *= v_vertR[1];

		(*out_y0_x1) = y[2];
		(*out_y1_x1) = y[3];

		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}
}
#endif

// TODO
static // BUG: uncomment this for SIGSEGV
void unified_2x2(
	int x, int y,
	int size_x, int size_y,
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	void *buffer_x,
	void *buffer_y
)
{
	// core size
	const int step_y = 2;
	const int step_x = 2;

	// vertical vectorization
	const int shift = 4;

	float T[4];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			const int pos_x = virt2real(x, xx, 0, size_x);
			const int pos_y = virt2real(y, yy, 0, size_y);

			T[step_x*yy+xx] = *addr2_const_s(src_ptr, pos_y, pos_x, src_stride_x, src_stride_y);
		}
	}

	// CALC
	//core_2x2_calc(T[0], T[1], T[2], T[3], (T+0), (T+1), (T+2), (T+3), buffer_y, buffer_x);
	CORE_2X2_CALC(T[0], T[1], T[2], T[3], buffer_y, buffer_x);
	//CORE_2X2_CALC(*(T+0), *(T+1), *(T+2), *(T+3), buffer_y, buffer_x);

	// STORE
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = T[step_x*yy+xx];
		}
	}
}

// TODO
static
void loop_unified_2x2(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int size_x, int size_y,
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	float *buffer_x,
	float *buffer_y
)
{
	// core size
	const int step_y = 2;
	const int step_x = 2;

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	// order=horizontal
	for(int y = base_y; y < stop_y; y += step_y)
	{
		for(int x = base_x; x < stop_x; x += step_x)
		{
			unified_2x2(
				x, y,
				size_x, size_y,
				src_ptr, src_stride_x, src_stride_y,
				dst_ptr, dst_stride_x, dst_stride_y,
				buffer_x + x*buff_elem_size,
				buffer_y + y*buff_elem_size
			);
		}
	}
}

// TODO
void cdf97_2f_dl_2x2_s(
	int size_x,
	int size_y,
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y
)
{
	const int words = 1;
	const int buff_elem_size = words*4;

	// offset_x, offset_y
	const int offset = 1;

	// step_x, step_y
	const int step_y = 2;
	const int step_x = 2;

	// shift_x, shift_y
// 	const int shift = 4; // 4 = vertical vectorization

	const int modulo_x_L = (offset) % step_x;
	const int overlap_x_L = step_x + !!modulo_x_L * (step_x-modulo_x_L); // (step_x-modulo_x_L) + (!!offset * step_x);
	const int modulo_x_R = (overlap_x_L+size_x) % step_x;
	const int overlap_x_R = step_x + !!modulo_x_R * (step_x-modulo_x_R); // step_x + ((modulo_x_R) ? (step_x-modulo_x_R) : (0));
	const int super_x = overlap_x_L + size_x + overlap_x_R;
// 	const int limit0_x = overlap_x_L + offset + shift;
// 	const int limit1_x = overlap_x_L + size_x - modulo_x_R - step_x*!modulo_x_R; // HACK: last term should not be here

	dwt_util_log(LOG_DBG, "mod_L=%i ovl_L=%i mod_R=%i ovl_R=%i => %i+%i+%i\n",
		modulo_x_L, overlap_x_L, modulo_x_R, overlap_x_R,
		overlap_x_L, size_x, overlap_x_R
    	);

	const int modulo_y_L = (offset) % step_y;
	const int overlap_y_L = step_y + !!modulo_y_L * (step_y-modulo_y_L); // (step_x-modulo_x_L) + (!!offset * step_x);
	const int modulo_y_R = (overlap_y_L+size_y) % step_y;
	const int overlap_y_R = step_y + !!modulo_y_R * (step_y-modulo_y_R); // step_x + ((modulo_x_R) ? (step_x-modulo_x_R) : (0));
	const int super_y = overlap_y_L + size_y + overlap_y_R;
// 	const int limit0_y = overlap_y_L + offset + shift;
// 	const int limit1_y = overlap_y_L + size_y - modulo_y_R - step_y*!modulo_y_R; // HACK: last term should not be here

	// alloc buffers
	float buffer_x[buff_elem_size*super_x] ALIGNED(16);
	float buffer_y[buff_elem_size*super_y] ALIGNED(16);

	// unified loop
	{
		loop_unified_2x2(
			/* base */ -overlap_x_L, -overlap_y_L,
			/* stop */ super_x, super_y,
			/* size */ size_x, size_y,
			src_ptr, src_stride_x, src_stride_y,
			dst_ptr, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y
		);
	}
}

// TODO
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
)
{
	UNUSED(zero_padding);

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int stride_y_j = mul_pow2(stride_y, j);
		const int stride_x_j = mul_pow2(stride_x, j);

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		if( size_x_j < 8 || size_y_j < 8 )
		{
			// FIXME
			dwt_util_error("unimplemented\n");
		}
		else
		{
// 			dwt_util_log(LOG_DBG, "j=%i: size=(%i,%i) stride=(%i,%i)\n", j, size_x_j, size_y_j, stride_x_j, stride_y_j);

			cdf97_2f_dl_2x2_s(
				size_x_j,
				size_y_j,
				ptr,
				stride_x_j,
				stride_y_j,
				ptr,
				stride_x_j,
				stride_y_j
			);
		}

		j++;
	}
}

// TODO
// FIXME: lag
static
void ms_loop_unified_2x2(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int size_x, int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	float *buffer_x,
	float *buffer_y,
	int J,
	int super_x,
	int super_y,
	int buffer_offset
)
{
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	// core size
	const int step_y = 2;
	const int step_x = 2;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

#if 1
	// order=horizontal
	for(int y = base_y; y < stop_y; y += step_y)
	{
		for(int x = base_x; x < stop_x; x += step_x)
		{
#else
	// order=vertical
	for(int x = base_x; x < stop_x; x += step_x)
	{
		for(int y = base_y; y < stop_y; y += step_y)
		{
#endif
			for(int j = 0; j < J; j++)
			{
				// mod == 0
#if 0
				if( ((x)%(step_x<<j)) == (0) && ((y)%(step_y<<j)) == (0) )
#else
				if( (x&((2<<j)-1)) == (0) && (y&((2<<j)-1)) == (0) )
#endif
				{
					// FIXME: what distance between read and write head is really needed?
					// 18 (ok), 4 (minimum), 8 (fixed left/top)
					const int lag = 8;

					// TODO: try to hardcode -(step_x-1) - lag as constant
					const int x_j = ceil_div_pow2(x,j) -(step_x-1) - lag;
					const int y_j = ceil_div_pow2(y,j) -(step_y-1) - lag;

					const int size_x_j = ceil_div_pow2(size_x,j);
					const int size_y_j = ceil_div_pow2(size_y,j);
					const int stride_x_j = mul_pow2(stride_x,j);
					const int stride_y_j = mul_pow2(stride_y,j);

					unified_2x2(
						x_j, y_j,
						size_x_j, size_y_j,
						ptr, stride_x_j, stride_y_j,
						ptr, stride_x_j, stride_y_j,
						// TODO: try to use incremental pointers
						buffer_x + j*buffer_x_elems + (buffer_offset+x_j)*buff_elem_size,
						buffer_y + j*buffer_y_elems + (buffer_offset+y_j)*buff_elem_size
					);
				}
			}
		}
	}
}

// TODO
void ms_cdf97_2f_dl_2x2_s(
	int size_x,
	int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	int J
)
{
	// TODO: assert

	// offset_x, offset_y
// 	const int offset = 1;

	// core size
// 	const int step_y = 4;
// 	const int step_x = 4;

	// vertical vectorization
// 	const int shift = 4;

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buff_guard = 32; // 32: maximal coordinate in negative value (currently -(step_x-1) - lag = -3 -8 = -11)
	const int overlap_L = 4; // 4
	// FIXME: what overlap is really needed?
	// 13<<J ... with lag = 18
	//  7<<J ... with lag =  8 (artifacts)
	const int overlap_R = 7<<J;

	const int super_x = buff_guard + overlap_L + size_x + overlap_R;
	const int super_y = buff_guard + overlap_L + size_y + overlap_R;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	// alloc buffers
	float buffer_x[J*buffer_x_elems] ALIGNED(16);
	float buffer_y[J*buffer_y_elems] ALIGNED(16);

	// zero buffers
#if 1
	dwt_util_zero_vec_s(buffer_x, J*buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, J*buffer_y_elems);
#endif

	const int buffer_offset = buff_guard+overlap_L;

	// unified loop
	{
		ms_loop_unified_2x2(
			/* base */ -overlap_L, -overlap_L, // x, y
			/* stop */ size_x+overlap_R, size_y+overlap_R, // x, y
			/* size */ size_x, size_y,
			ptr,
			stride_x, stride_y,
			buffer_x, buffer_y,
			J,
			super_x, super_y,
			buffer_offset
		);
	}
}

// TODO
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
)
{
	//FUNC_BEGIN;

	assert( M > 0 && N > 0 && fwd_secs && inv_secs );

// 	dwt_util_log(LOG_DBG, "perf. test (%i,%i) N=%i M=%i\n", size_x, size_y, N, M);

	int stride_y = sizeof(float);
	int stride_x = dwt_util_get_stride(stride_y * size_x, opt_stride);

	// pointer to M pointers to image data
	void *ptr[M];

	// template
	void *template;

	// allocate
	template = dwt_util_alloc_image2(
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	// fill with test pattern
	dwt_util_test_image_fill2_s(
		template,
		stride_x,
		stride_y,
		size_x,
		size_y,
		0,
		type
	);

	// allocate M images
	for(int m = 0; m < M; m++)
	{
		ptr[m] = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
		// copy template to ptr[m]
		for(int m = 0; m < M; m++)
		{
			dwt_util_copy_s(template, ptr[m], stride_x, stride_y, size_x, size_y);
		}
	
		// flush memory
		if(flush)
		{
			for(int m = 0; m < M; m++)
			{
				dwt_util_flush_cache(ptr[m], dwt_util_image_size(stride_x, stride_y, size_x, size_y));
			}
		}

		// start timer
		const dwt_clock_t time_fwd_start = dwt_util_get_clock(clock_type);
		// perform M fwd transforms
		for(int m = 0; m < M; m++)
		{
			ms_cdf97_2f_dl_2x2_s(
				size_x,
				size_y,
				ptr[m],
				stride_x,
				stride_y,
				J
			);
		}
		// stop timer
		const dwt_clock_t time_fwd_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_fwd_secs = (float)(time_fwd_stop - time_fwd_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_fwd_secs < *fwd_secs )
			*fwd_secs = time_fwd_secs;

		// flush memory
		if(flush)
		{
			for(int m = 0; m < M; m++)
			{
				dwt_util_flush_cache(ptr[m], dwt_util_image_size(stride_x, stride_y, size_x, size_y));
			}
		}

		// start timer
		const dwt_clock_t time_inv_start = dwt_util_get_clock(clock_type);
		// perform M inv transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2i_inplace_s(ptr[m], stride_x, stride_y, size_x, size_y, size_x, size_y, J, 0, 0);
		}
		// stop timer
		const dwt_clock_t time_inv_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_inv_secs = (float)(time_inv_stop - time_inv_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_inv_secs < *inv_secs )
			*inv_secs = time_inv_secs;

		// compare
		for(int m = 0; m < M; m++)
		{
			int err = dwt_util_compare2_destructive_s(
				ptr[m],
				template,
				stride_x,
				stride_y,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			if( err )
			{
				dwt_util_log(LOG_ERR, "perf: [%i] compare failed :(\n", m);
#if 1
				dwt_util_save_to_pgm_s("debug.pgm", 1.0, ptr[m], stride_x, stride_y, size_x, size_y);
#endif
			}
		}
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}
	dwt_util_free_image(&template);

	//FUNC_END;
}

// TODO
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
)
{
	//FUNC_BEGIN;

	assert( min_x > 0 && min_x < max_x );
	assert( M > 0 && N > 0 );
	assert( fwd_plot_data && inv_plot_data );

	const float growth_factor = g_growth_factor_s;

	// for x = min_x to max_x
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
	{
		// y is equal to x
		const int y = x;

		int size_x = x;
		int size_y = y;

		dwt_util_log(LOG_DBG, "performance test for [%ix%i]...\n", size_x, size_y);

		float fwd_secs;
		float inv_secs;

		// call perf()
		dwt_util_perf_ms_cdf97_2f_dl_2x2_s(
			size_x,
			size_y,
			opt_stride,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs,
			1, // flush
			0, // type
			J
		);

#ifdef MEASURE_PER_PIXEL
		const int denominator = x*y;
#else
		const int denominator = 1;
#endif

		// printf into file
		fprintf(fwd_plot_data, "%i\t%.10f\n", x*y, fwd_secs/denominator);
		fprintf(inv_plot_data, "%i\t%.10f\n", x*y, inv_secs/denominator);

	}

	//FUNC_END;
}
