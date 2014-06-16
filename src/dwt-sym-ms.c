#include "dwt-sym-ms.h"
#include "libdwt.h"
#include "inline.h"
#include <math.h>
#ifdef __SSE__
	#include <xmmintrin.h>
#endif

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
	const int overlap_R = 7<<J;

	const int super_x = buff_guard + overlap_L + size_x + overlap_R;
	const int super_y = buff_guard + overlap_L + size_y + overlap_R;

	// FIXME: try to use prime stride (this is number of elements, not the stride in bytes)
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
