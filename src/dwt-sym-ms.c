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

#if 0
	if( real > size-1 )
		dwt_util_log(LOG_DBG, "%s: real=%i > size-1=%i (diff=%i)\n", __FUNCTION__, real, size-1, real - (size-1));
#endif

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
#ifdef __SSE__
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
#endif /* __SSE__ */
}

static
void direct_4x4(
	int x,
	int y,
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
#ifdef __SSE__
	// core size
	const int step_y = 4;
	const int step_x = 4;

	__m128 t[4];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x, xx, 0, size_x);
			const int pos_y = virt2real_error(y, yy, 0, size_y);

#if 0
			if( pos_x < 0 || pos_y < 0 )
			{
				dwt_util_error("out of buffer\n");
			}
#endif
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
			const int pos_x = virt2real_error(x, xx, 0, size_x);
			const int pos_y = virt2real_error(y, yy, 0, size_y);

#if 0
			if( pos_x < 0 || pos_y < 0 )
			{
				dwt_util_error("out of buffer\n");
			}
#endif

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
#endif /* __SSE__ */
}

// store LL and HL/LH/HH separately
// NOTE: this is faster than original one :)
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
#ifdef __SSE__
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

	// STORE LL
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(low_ptr, pos_y, pos_x, low_stride_x, low_stride_y) = t[yy][xx];
		}
	}

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
#endif /* __SSE__ */
}

static
void unified_4x4_separately2(
	int size_x,
	int size_y,
	// input
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	int src_pos_x,
	int src_pos_y,
	// output LL
	void *low_ptr,
	int low_stride_x,
	int low_stride_y,
	int low_pos_x,
	int low_pos_y,
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	int dst_pos_x,
	int dst_pos_y,
	// buffers
	void *buffer_x,
	void *buffer_y
)
{
#ifdef __SSE__
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	__m128 t[4];

	// LOAD = src
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real(src_pos_x, xx, 0, size_x);
			const int pos_y = virt2real(src_pos_y, yy, 0, size_y);

			t[xx][yy] = *addr2_const_s(src_ptr, pos_y, pos_x, src_stride_x, src_stride_y);
		}
	}

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL = low
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(low_pos_x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(low_pos_y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(low_ptr, pos_y, pos_x, low_stride_x, low_stride_y) = t[yy][xx];
		}
	}

	// STORE HL/LH/HH = dst
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(dst_pos_x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(dst_pos_y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(dst_pos_x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(dst_pos_y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(dst_pos_x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(dst_pos_y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
#endif /* __SSE__ */
}

static
void unified_4x4_separately3(
	// input
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	int src_pos_x,
	int src_pos_y,
	int src_size_x,
	int src_size_y,
	// output LL
	void *low_ptr,
	int low_stride_x,
	int low_stride_y,
	int low_pos_x,
	int low_pos_y,
	int low_size_x,
	int low_size_y,
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	int dst_pos_x,
	int dst_pos_y,
	int dst_size_x,
	int dst_size_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// debug
	int j
)
{
#ifdef __SSE__
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	__m128 t[4];

	// LOAD = src
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real(src_pos_x, xx, 0, src_size_x);
			const int pos_y = virt2real(src_pos_y, yy, 0, src_size_y);

			t[xx][yy] = *addr2_const_s(src_ptr, pos_y, pos_x, src_stride_x, src_stride_y);
		}
	}

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL = low
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(low_pos_x-shift, xx, 0, low_size_x);
			const int pos_y = virt2real_error(low_pos_y-shift, yy, 0, low_size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(low_ptr, pos_y, pos_x, low_stride_x, low_stride_y) = t[yy][xx];
		}
	}

	// STORE HL/LH/HH = dst
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(dst_pos_x-shift, xx, 0, dst_size_x);
			const int pos_y = virt2real_error(dst_pos_y-shift, yy, 0, dst_size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(dst_pos_x-shift, xx, 0, dst_size_x);
			const int pos_y = virt2real_error(dst_pos_y-shift, yy, 0, dst_size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(dst_pos_x-shift, xx, 0, dst_size_x);
			const int pos_y = virt2real_error(dst_pos_y-shift, yy, 0, dst_size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
#endif /* __SSE__ */
}

#ifdef __SSE__
// image(LL) => buffer(LL)
static
void unified_4x4_separately_img2tmp(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	int segment_x,		// FIXME: segment of output buffer (0, 1)
	int segment_y,		// FIXME: segment of output buffer (0, 1)
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
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

	// STORE LL
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			LL_dst[/*x*/(segment_x<<1)+(xx>>1)][/*y*/(segment_y<<1)+(yy>>1)] = /*float*/t[yy][xx];
		}
	}

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segment): x=0 y=0
static
void unified_4x4_separately_img2tmp_segment00(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
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

	// STORE LL
	LL_dst[0][0] = t[1][1];
	LL_dst[1][0] = t[1][3];
	LL_dst[0][1] = t[3][1];
	LL_dst[1][1] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segment): x=0 y=1
static
void unified_4x4_separately_img2tmp_segment01(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
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

	// STORE LL
	LL_dst[0][2] = t[1][1];
	LL_dst[1][2] = t[1][3];
	LL_dst[0][3] = t[3][1];
	LL_dst[1][3] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segment): x=1 y=0
static
void unified_4x4_separately_img2tmp_segment10(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
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

	// STORE LL
	LL_dst[2][0] = t[1][1];
	LL_dst[3][0] = t[1][3];
	LL_dst[2][1] = t[3][1];
	LL_dst[3][1] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segment): x=1 y=1
static
void unified_4x4_separately_img2tmp_segment11(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
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

	// STORE LL
	LL_dst[2][2] = t[1][1];
	LL_dst[3][2] = t[1][3];
	LL_dst[2][3] = t[3][1];
	LL_dst[3][3] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// buffer(LL) => buffer(LL)
static
void unified_4x4_separately_tmp2tmp(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	int segment_x,		// FIXME: segment of output buffer (0, 1)
	int segment_y,		// FIXME: segment of output buffer (0, 1)
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	//__m128 t[4];

	// LOAD
	__m128 *t = LL_src;

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			LL_dst[/*x*/(segment_x<<1)+(xx>>1)][/*y*/(segment_y<<1)+(yy>>1)] = /*float*/t[yy][xx];
		}
	}

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segments): x=0, y=0
static
void unified_4x4_separately_tmp2tmp_segment00(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	//__m128 t[4];

	// LOAD
	__m128 *t = LL_src;

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL
	LL_dst[0][0] = t[1][1];
	LL_dst[1][0] = t[1][3];
	LL_dst[0][1] = t[3][1];
	LL_dst[1][1] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segments): x=0, y=1
static
void unified_4x4_separately_tmp2tmp_segment01(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	//__m128 t[4];

	// LOAD
	__m128 *t = LL_src;

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL
	LL_dst[0][2] = t[1][1];
	LL_dst[1][2] = t[1][3];
	LL_dst[0][3] = t[3][1];
	LL_dst[1][3] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segments): x=1, y=0
static
void unified_4x4_separately_tmp2tmp_segment10(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	//__m128 t[4];

	// LOAD
	__m128 *t = LL_src;

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL
	LL_dst[2][0] = t[1][1];
	LL_dst[3][0] = t[1][3];
	LL_dst[2][1] = t[3][1];
	LL_dst[3][1] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// (segments): x=1, y=1
static
void unified_4x4_separately_tmp2tmp_segment11(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	//__m128 t[4];

	// LOAD
	__m128 *t = LL_src;

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL
	LL_dst[2][2] = t[1][1];
	LL_dst[3][2] = t[1][3];
	LL_dst[2][3] = t[3][1];
	LL_dst[3][3] = t[3][3];

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

#ifdef __SSE__
// buffer(LL) => image(LL)
static
void unified_4x4_separately_tmp2img(
	int x,
	int y,
	int size_x,
	int size_y,
	// input
	const void *src_ptr,	// FIXME: possibly unused
	int src_stride_x,	// FIXME: possibly unused
	int src_stride_y,	// FIXME: possibly unused
	// output LL
	void *low_ptr,		// FIXME: possibly unused
	int low_stride_x,	// FIXME: possibly unused
	int low_stride_y,	// FIXME: possibly unused
	// output HL/LH/HH
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// core buffers
	void *buffer_x,
	void *buffer_y,
	// LL buffers
	__m128 *LL_src,		// FIXME: input from this buffer
	__m128 *LL_dst		// FIXME: output into 1/4 of this buffer
)
{
	// core size
	const int step_y = 4;
	const int step_x = 4;

	// vertical vectorization
	const int shift = 4;

	// LOAD
	__m128 *t = LL_src;

	// CALC
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// STORE LL
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(low_ptr, pos_y, pos_x, low_stride_x, low_stride_y) = t[yy][xx];
		}
	}

	// STORE HL/LH/HH
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 0; yy < step_y; yy+=2)
	{
		for(int xx = 1; xx < step_x; xx+=2)
		{
			// virtual => real coordinates
			const int pos_x = virt2real_error(x-shift, xx, 0, size_x);
			const int pos_y = virt2real_error(y-shift, yy, 0, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
	for(int yy = 1; yy < step_y; yy+=2)
	{
		for(int xx = 0; xx < step_x; xx+=2)
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
#endif /* __SSE__ */

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
						buffer_x + j*buffer_x_elems + (buffer_offset+x_j)*buff_elem_size,
						buffer_y + j*buffer_y_elems + (buffer_offset+y_j)*buff_elem_size
					);
				}
			}
		}
	}
}

// get_buffer_ptr(buffer, j, buffer_elems, pos, buff_elem_size)
static
float *get_buffer_ptr(
	float *buffer,
	int j,
	int buffer_elems,
	int pos,
	int buff_elem_size
)
{
	return buffer + j*buffer_elems + pos*buff_elem_size;
}

#ifdef __SSE__
static
void multiscale_j(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output,
	int segment_x,
	int segment_y
)
{
	// buffer[4x4] for LL in "j-1" => "j"
	__m128 temp_for_input[4]; // on the stack

	const int step_y = 4;
	const int step_x = 4;

	if( j > 0 )
	{
		// process j-1: four multiscale_j(j-1)

		const int step_y_j1 = mul_pow2(step_y, j-1);
		const int step_x_j1 = mul_pow2(step_x, j-1);

		// left-top (segment x=0 y=0)
		multiscale_j(
			j-1, x+0,         y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input, 0, 0
		);
		// right-top (segment x=1 y=0)
		multiscale_j(
			j-1, x+step_x_j1, y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input, 1, 0
		);
		// left-bottom (segment x=0 y=1)
		multiscale_j(
			j-1, x+0,         y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input, 0, 1
		);
		// right-bottom (segment x=1 y=1)
		multiscale_j(
			j-1, x+step_x_j1, y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input, 1, 1
		);
	}

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;
	
	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);
	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

#if 1
	if( j > 0 )
	{
		// process j: unified_4x4_separately(j)

		// buffer => buffer
		unified_4x4_separately_tmp2tmp(
			x_j, y_j,
			size_x_j, size_y_j,
			// input
			src, src_stride_x_j, src_stride_y_j,
			// output LL
			dst, dst_stride_x_j, dst_stride_y_j,
			// output HL/LH/HH
			dst, dst_stride_x_j, dst_stride_y_j,
			// core buffers
			get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
			get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
			// LL buffers
			segment_x,		// FIXME: segment of output buffer (0, 1)
			segment_y,		// FIXME: segment of output buffer (0, 1)
			temp_for_input,		// FIXME: input from this buffer
			temp_for_output		// FIXME: output into 1/4 of this buffer
		);
	}
	else
	{
		// bottom case
		{
			// bottom (input from src)
			unified_4x4_separately_img2tmp(
				x_j, y_j,
				size_x_j, size_y_j,
				// input
				src, src_stride_x_j, src_stride_y_j,
				// output LL
				dst, dst_stride_x_j, dst_stride_y_j,
				// output HL/LH/HH
				dst, dst_stride_x_j, dst_stride_y_j,
				// core buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
				// LL buffers
				segment_x,		// FIXME: segment of output buffer (0, 1)
				segment_y,		// FIXME: segment of output buffer (0, 1)
				temp_for_input,		// FIXME: input from this buffer
				temp_for_output		// FIXME: output into 1/4 of this buffer
			);
		}
	}
#else
	unified_4x4_separately(
		x_j, y_j,
		size_x_j, size_y_j,
		// use a single aux. buffer
		src, src_stride_x_j, src_stride_y_j, // input
		src, src_stride_x_j, src_stride_y_j, // output LL
		// output to a single destination
		dst, dst_stride_x_j, dst_stride_y_j, // output HL/LH/HH
		// buffers
		get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
		get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size)
	);
#endif
}
#endif /* __SSE__ */

#ifdef __SSE__
// decl
static
void multiscale_j_segment00(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
);
#endif

#ifdef __SSE__
// decl
static
void multiscale_j_segment01(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
);
#endif

#ifdef __SSE__
// decl
static
void multiscale_j_segment10(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
);
#endif

#ifdef __SSE__
// decl
static
void multiscale_j_segment11(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
);
#endif

#ifdef __SSE__
// (segments): x=0 y=0
static
void multiscale_j_segment00(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
)
{
	__m128 temp_for_input[4]; // on the stack

	const int step_y = 4;
	const int step_x = 4;

	if( j > 0 )
	{
		// process j-1: four multiscale_j(j-1)

		const int step_y_j1 = mul_pow2(step_y, j-1);
		const int step_x_j1 = mul_pow2(step_x, j-1);

		// left-top (segment x=0 y=0)
		multiscale_j_segment00(
			j-1, x+0,         y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-top (segment x=1 y=0)
		multiscale_j_segment10(
			j-1, x+step_x_j1, y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// left-bottom (segment x=0 y=1)
		multiscale_j_segment01(
			j-1, x+0,         y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-bottom (segment x=1 y=1)
		multiscale_j_segment11(
			j-1, x+step_x_j1, y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
	}

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;
	
	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);
	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

	if( j > 0 )
	{
		// process j: unified_4x4_separately(j)

		// buffer => buffer
		unified_4x4_separately_tmp2tmp_segment00(
			x_j, y_j,
			size_x_j, size_y_j,
			// input
			src, src_stride_x_j, src_stride_y_j,
			// output LL
			dst, dst_stride_x_j, dst_stride_y_j,
			// output HL/LH/HH
			dst, dst_stride_x_j, dst_stride_y_j,
			// core buffers
			get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
			get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
			// LL buffers
			temp_for_input,		// FIXME: input from this buffer
			temp_for_output		// FIXME: output into 1/4 of this buffer
		);
	}
	else
	{
		// bottom case
		{
			// bottom (input from src)
			unified_4x4_separately_img2tmp_segment00(
				x_j, y_j,
				size_x_j, size_y_j,
				// input
				src, src_stride_x_j, src_stride_y_j,
				// output LL
				dst, dst_stride_x_j, dst_stride_y_j,
				// output HL/LH/HH
				dst, dst_stride_x_j, dst_stride_y_j,
				// core buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
				// LL buffers
				temp_for_input,		// FIXME: input from this buffer
				temp_for_output		// FIXME: output into 1/4 of this buffer
			);
		}
	}
}

// (segments): x=0 y=1
static
void multiscale_j_segment01(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
)
{
	__m128 temp_for_input[4]; // on the stack

	const int step_y = 4;
	const int step_x = 4;

	if( j > 0 )
	{
		// process j-1: four multiscale_j(j-1)

		const int step_y_j1 = mul_pow2(step_y, j-1);
		const int step_x_j1 = mul_pow2(step_x, j-1);

		// left-top (segment x=0 y=0)
		multiscale_j_segment00(
			j-1, x+0,         y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-top (segment x=1 y=0)
		multiscale_j_segment10(
			j-1, x+step_x_j1, y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// left-bottom (segment x=0 y=1)
		multiscale_j_segment01(
			j-1, x+0,         y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-bottom (segment x=1 y=1)
		multiscale_j_segment11(
			j-1, x+step_x_j1, y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
	}

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;
	
	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);
	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

	if( j > 0 )
	{
		// process j: unified_4x4_separately(j)

		// buffer => buffer
		unified_4x4_separately_tmp2tmp_segment01(
			x_j, y_j,
			size_x_j, size_y_j,
			// input
			src, src_stride_x_j, src_stride_y_j,
			// output LL
			dst, dst_stride_x_j, dst_stride_y_j,
			// output HL/LH/HH
			dst, dst_stride_x_j, dst_stride_y_j,
			// core buffers
			get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
			get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
			// LL buffers
			temp_for_input,		// FIXME: input from this buffer
			temp_for_output		// FIXME: output into 1/4 of this buffer
		);
	}
	else
	{
		// bottom case
		{
			// bottom (input from src)
			unified_4x4_separately_img2tmp_segment01(
				x_j, y_j,
				size_x_j, size_y_j,
				// input
				src, src_stride_x_j, src_stride_y_j,
				// output LL
				dst, dst_stride_x_j, dst_stride_y_j,
				// output HL/LH/HH
				dst, dst_stride_x_j, dst_stride_y_j,
				// core buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
				// LL buffers
				temp_for_input,		// FIXME: input from this buffer
				temp_for_output		// FIXME: output into 1/4 of this buffer
			);
		}
	}
}

// (segments): x=1 y=0
static
void multiscale_j_segment10(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
)
{
	__m128 temp_for_input[4]; // on the stack

	const int step_y = 4;
	const int step_x = 4;

	if( j > 0 )
	{
		// process j-1: four multiscale_j(j-1)

		const int step_y_j1 = mul_pow2(step_y, j-1);
		const int step_x_j1 = mul_pow2(step_x, j-1);

		// left-top (segment x=0 y=0)
		multiscale_j_segment00(
			j-1, x+0,         y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-top (segment x=1 y=0)
		multiscale_j_segment10(
			j-1, x+step_x_j1, y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// left-bottom (segment x=0 y=1)
		multiscale_j_segment01(
			j-1, x+0,         y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-bottom (segment x=1 y=1)
		multiscale_j_segment11(
			j-1, x+step_x_j1, y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
	}

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;
	
	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);
	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

	if( j > 0 )
	{
		// process j: unified_4x4_separately(j)

		// buffer => buffer
		unified_4x4_separately_tmp2tmp_segment10(
			x_j, y_j,
			size_x_j, size_y_j,
			// input
			src, src_stride_x_j, src_stride_y_j,
			// output LL
			dst, dst_stride_x_j, dst_stride_y_j,
			// output HL/LH/HH
			dst, dst_stride_x_j, dst_stride_y_j,
			// core buffers
			get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
			get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
			// LL buffers
			temp_for_input,		// FIXME: input from this buffer
			temp_for_output		// FIXME: output into 1/4 of this buffer
		);
	}
	else
	{
		// bottom case
		{
			// bottom (input from src)
			unified_4x4_separately_img2tmp_segment10(
				x_j, y_j,
				size_x_j, size_y_j,
				// input
				src, src_stride_x_j, src_stride_y_j,
				// output LL
				dst, dst_stride_x_j, dst_stride_y_j,
				// output HL/LH/HH
				dst, dst_stride_x_j, dst_stride_y_j,
				// core buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
				// LL buffers
				temp_for_input,		// FIXME: input from this buffer
				temp_for_output		// FIXME: output into 1/4 of this buffer
			);
		}
	}
}

// (segments): x=1 y=1
static
void multiscale_j_segment11(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output
)
{
	__m128 temp_for_input[4]; // on the stack

	const int step_y = 4;
	const int step_x = 4;

	if( j > 0 )
	{
		// process j-1: four multiscale_j(j-1)

		const int step_y_j1 = mul_pow2(step_y, j-1);
		const int step_x_j1 = mul_pow2(step_x, j-1);

		// left-top (segment x=0 y=0)
		multiscale_j_segment00(
			j-1, x+0,         y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-top (segment x=1 y=0)
		multiscale_j_segment10(
			j-1, x+step_x_j1, y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// left-bottom (segment x=0 y=1)
		multiscale_j_segment01(
			j-1, x+0,         y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-bottom (segment x=1 y=1)
		multiscale_j_segment11(
			j-1, x+step_x_j1, y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
	}

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;

	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);
	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

	if( j > 0 )
	{
		// process j: unified_4x4_separately(j)

		// buffer => buffer
		unified_4x4_separately_tmp2tmp_segment11(
			x_j, y_j,
			size_x_j, size_y_j,
			// input
			src, src_stride_x_j, src_stride_y_j,
			// output LL
			dst, dst_stride_x_j, dst_stride_y_j,
			// output HL/LH/HH
			dst, dst_stride_x_j, dst_stride_y_j,
			// core buffers
			get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
			get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
			// LL buffers
			temp_for_input,		// FIXME: input from this buffer
			temp_for_output		// FIXME: output into 1/4 of this buffer
		);
	}
	else
	{
		// bottom case
		{
			// bottom (input from src)
			unified_4x4_separately_img2tmp_segment11(
				x_j, y_j,
				size_x_j, size_y_j,
				// input
				src, src_stride_x_j, src_stride_y_j,
				// output LL
				dst, dst_stride_x_j, dst_stride_y_j,
				// output HL/LH/HH
				dst, dst_stride_x_j, dst_stride_y_j,
				// core buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
				// LL buffers
				temp_for_input,		// FIXME: input from this buffer
				temp_for_output		// FIXME: output into 1/4 of this buffer
			);
		}
	}
}

static
void multiscale_j_top(
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// test
	int top,	// 1 = on top (maximal "j"), 0 otherwise
	__m128 *temp_for_output,
	int segment_x,
	int segment_y
)
{
	// buffer[4x4] for LL in "j-1" => "j"
	__m128 temp_for_input[4]; // on the stack

	const int step_y = 4;
	const int step_x = 4;

	// process j-1: four multiscale_j(j-1)

	{
		const int step_y_j1 = mul_pow2(step_y, j-1);
		const int step_x_j1 = mul_pow2(step_x, j-1);

		// left-top (segment x=0 y=0)
		multiscale_j_segment00(
			j-1, x+0,         y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-top (segment x=1 y=0)
		multiscale_j_segment10(
			j-1, x+step_x_j1, y+0,         size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// left-bottom (segment x=0 y=1)
		multiscale_j_segment01(
			j-1, x+0,         y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
		// right-bottom (segment x=1 y=1)
		multiscale_j_segment11(
			j-1, x+step_x_j1, y+step_y_j1, size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y, super_x, super_y, buffer_offset,
			0, temp_for_input
		);
	}

	// process j: unified_4x4_separately(j)

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;
	
	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);
	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

	{
		{
			// top (maximal "j", output to dst)
			unified_4x4_separately_tmp2img(
				x_j, y_j,
				size_x_j, size_y_j,
				// input
				src, src_stride_x_j, src_stride_y_j,
				// output LL
				dst, dst_stride_x_j, dst_stride_y_j,
				// output HL/LH/HH
				dst, dst_stride_x_j, dst_stride_y_j,
				// core buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size),
				// LL buffers
				temp_for_input,		// FIXME: input from this buffer
				temp_for_output		// FIXME: output into 1/4 of this buffer
			);
		}
	}
}
#endif /* __SSE__ */

static
void multiscale_4x4(
	// scale
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// data
	void *ptr,
	int stride_x,
	int stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset
)
{
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int step_y = 4;
	const int step_x = 4;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;
	
	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);
	const int stride_x_j = mul_pow2(stride_x, j);
	const int stride_y_j = mul_pow2(stride_y, j);

	unified_4x4_separately(
		x_j, y_j,
		size_x_j, size_y_j,
		// use a single aux. buffer
		ptr, stride_x_j, stride_y_j, // input
		// LL are intermediate results for j < j_max
		ptr, stride_x_j, stride_y_j, // output LL
		// output to a single destination
		ptr, stride_x_j, stride_y_j, // output HL/LH/HH
		// buffers
		get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
		get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size)
	);
}

static
void multiscale_4x4_separately(
	// scale
	int j,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// data
	void *src,
	int src_stride_x,
	int src_stride_y,
	void *low,
	int low_stride_x,
	int low_stride_y,
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset
)
{
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int step_y = 4;
	const int step_x = 4;

	const int lag = 0;

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;
	
	const int x_j = ceil_div_pow2(x, j) - lag_x;
	const int y_j = ceil_div_pow2(y, j) - lag_y;

	const int size_x_j = ceil_div_pow2(size_x, j);
	const int size_y_j = ceil_div_pow2(size_y, j);

	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int low_stride_x_j = mul_pow2(low_stride_x, j);
	const int low_stride_y_j = mul_pow2(low_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

	unified_4x4_separately(
		x_j, y_j,
		size_x_j, size_y_j,
		// load
		src, src_stride_x_j, src_stride_y_j, // input
		// store LL
		low, low_stride_x_j, low_stride_y_j, // output LL
		// store LH/HL/HH
		dst, dst_stride_x_j, dst_stride_y_j, // output HL/LH/HH
		// buffers
		get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
		get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size)
	);
}

static
void multiscale_4x4_separately3(
	// scale
	int j,
	// position
	int x,
	int y,
	// data
	void *src,
	int src_stride_x,
	int src_stride_y,
	int src_size_x,
	int src_size_y,
	int src_offset_x,
	int src_offset_y,
	void *low,
	int low_stride_x,
	int low_stride_y,
	int low_size_x,
	int low_size_y,
	int low_offset_x,
	int low_offset_y,
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	int dst_size_x,
	int dst_size_y,
	int dst_offset_x,
	int dst_offset_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset
)
{
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int step_y = 4;
	const int step_x = 4;

	const int lag = 16; // FIXME

	const int lag_x = (step_x-1) + lag;
	const int lag_y = (step_y-1) + lag;

	const int src_x_j = ceil_div_pow2(x+src_offset_x, j) - lag_x;
	const int src_y_j = ceil_div_pow2(y+src_offset_y, j) - lag_y;
	const int low_x_j = ceil_div_pow2(x+low_offset_x, j) - lag_x;
	const int low_y_j = ceil_div_pow2(y+low_offset_y, j) - lag_y;
	const int dst_x_j = ceil_div_pow2(x+dst_offset_x, j) - lag_x;
	const int dst_y_j = ceil_div_pow2(y+dst_offset_y, j) - lag_y;

	const int src_size_x_j = ceil_div_pow2(src_size_x, j);
	const int src_size_y_j = ceil_div_pow2(src_size_y, j);
	const int low_size_x_j = ceil_div_pow2(low_size_x, j);
	const int low_size_y_j = ceil_div_pow2(low_size_y, j);
	const int dst_size_x_j = ceil_div_pow2(dst_size_x, j);
	const int dst_size_y_j = ceil_div_pow2(dst_size_y, j);

	const int src_stride_x_j = mul_pow2(src_stride_x, j);
	const int src_stride_y_j = mul_pow2(src_stride_y, j);
	const int low_stride_x_j = mul_pow2(low_stride_x, j);
	const int low_stride_y_j = mul_pow2(low_stride_y, j);
	const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
	const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

	unified_4x4_separately3(
		// load
		src, src_stride_x_j, src_stride_y_j, src_x_j, src_y_j, src_size_x_j, src_size_y_j, // input
		// store LL
		low, low_stride_x_j, low_stride_y_j, low_x_j, low_y_j, low_size_x_j, low_size_y_j, // output LL
		// store LH/HL/HH
		dst, dst_stride_x_j, dst_stride_y_j, dst_x_j, dst_y_j, dst_size_x_j, dst_size_y_j, // output HL/LH/HH
		// buffers
		get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+src_x_j), buff_elem_size),
		get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+src_y_j), buff_elem_size),
		// debug
		j
	);
}

// typedef __m128 __m512[4] ALIGNED(16);

static
int select_i(
	int bool,
	int if_true,
	int if_zero
)
{
	return bool * if_true + !bool * if_zero;
}

static
void *select_p(
	int bool,
	void *if_true,
	void *if_zero
)
{
	return (intptr_t)bool * (intptr_t)if_true + (intptr_t)!bool * (intptr_t)if_zero;
}

static
void copy_src_to_buff(
	// input image
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	int src_size_x,
	int src_size_y,
	int src_pos_x,
	int src_pos_y,
	// buffer
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	// copy area of (0..size)
	int size_x,
	int size_y
)
{
	for(int virt_y = 0; virt_y < size_y; virt_y++)
	{
		for(int virt_x = 0; virt_x < size_x; virt_x++)
		{
			const int real_x = virt2real(src_pos_x, virt_x, 0, src_size_x);
			const int real_y = virt2real(src_pos_y, virt_y, 0, src_size_y);

			*addr2_s(dst_ptr, virt_y, virt_x, dst_stride_x, dst_stride_y) =
				*addr2_const_s(src_ptr, real_y, real_x, src_stride_x, src_stride_y);
		}
	}
}

static
void copy_buff_to_dst(
	// buffer
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	// output image
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	int dst_size_x,
	int dst_size_y,
	int dst_pos_x,
	int dst_pos_y,
	// copy area of (0..size)
	int size_x,
	int size_y,
	int J
)
{
#if 0
	const int delay = 0;

	for(int virt_y = 0; virt_y < size_y; virt_y++)
	{
		for(int virt_x = 0; virt_x < size_x; virt_x++)
		{
			const int real_x = virt2real_error(dst_pos_x, virt_x-delay, 0, dst_size_x);
			const int real_y = virt2real_error(dst_pos_y, virt_y-delay, 0, dst_size_y);

			if( real_x < 0 || real_y < 0 )
				continue;

			*addr2_s(dst_ptr, real_y, real_x, dst_stride_x, dst_stride_y) =
				*addr2_const_s(src_ptr, virt_y, virt_x, src_stride_x, src_stride_y);
		}
	}
#endif
#if 1
	const int step = 4;

	// H bands
	for(int j = 0; j < J; j++)
	{
		const int start_j = mul_pow2(1, j) - 1;
		const int short_step_j = mul_pow2(1, j);
		const int long_step_j = mul_pow2(1, j+1);

		const int delay_j = mul_pow2(step, j+1) - step;

		// segments
		for(int yy = 0; yy < 2; yy++)
		for(int xx = 0; xx < 2; xx++)
		{
			// except of J-1, break loop on next level
// 			if( 1 == xx && 1 == yy )
			if( j < J-1 && 1 == xx && 1 == yy )
			{
				break;
			}

// 			dwt_util_log(LOG_DBG, "copy-back: j=%i H(%i,%i) start=%i short-step=%i long-step=%i delay=%i\n",
// 				j, yy, xx, start_j, short_step_j, long_step_j, delay_j);

			for(int y = start_j+yy*short_step_j; y < size_y; y += long_step_j)
			for(int x = start_j+xx*short_step_j; x < size_x; x += long_step_j)
			{
				const int real_x = virt2real_error(dst_pos_x, x-delay_j, 0, dst_size_x);
				const int real_y = virt2real_error(dst_pos_y, y-delay_j, 0, dst_size_y);

				// out of the image
				if( real_x < 0 || real_y < 0 )
					continue;

				*addr2_s(dst_ptr, real_y, real_x, dst_stride_x, dst_stride_y) =
					*addr2_const_s(src_ptr, y, x, src_stride_x, src_stride_y);
			}
		}
	}

	// L band
// 	{
// 		const int start_j = mul_pow2(1, J) - 1;
// 		const int long_step_j = mul_pow2(1, J);
// 
// 		const int delay_j = mul_pow2(step, J) - step;
// 
// // 		dwt_util_log(LOG_DBG, "copy-back: J=%i L start=%i long-step=%i delay=%i\n",
// // 				J, start_j, long_step_j, delay_j);
// 
// 		for(int y = start_j; y < size_y; y += long_step_j)
// 		for(int x = start_j; x < size_x; x += long_step_j)
// 		{
// 				const int real_x = virt2real_error(dst_pos_x, x-delay_j, 0, dst_size_x);
// 				const int real_y = virt2real_error(dst_pos_y, y-delay_j, 0, dst_size_y);
// 
// 				// out of the image
// 				if( real_x < 0 || real_y < 0 )
// 					continue;
// 
// 				*addr2_s(dst_ptr, real_y, real_x, dst_stride_x, dst_stride_y) =
// 					*addr2_const_s(src_ptr, y, x, src_stride_x, src_stride_y);
// 		}
// 	}
#endif
}

static
void multiscale_4x4_j_loop_new(
	// scale
	int J,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// ...
	int super_x,
	int super_y,
	int buffer_offset,
	...
)
{
	const int step_y = 4;
	const int step_x = 4;

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int step_x_max = mul_pow2(step_x, J-1);
	const int step_y_max = mul_pow2(step_y, J-1);

// 	dwt_util_log(LOG_DBG, "ms-core: step-max=(%i,%i) @ (%i,%i) in (0,0)..(%i,%i)\n",
// 		step_y_max, step_x_max,
// 		y, x,
// 		size_y, size_x
// 	);

	// FIXME: use persistent buffer[]
	// create buffer
	const int buffer_size_x = step_x_max;
	const int buffer_size_y = step_y_max;
	const int buffer_stride_y = sizeof(float);
	const int buffer_stride_x = /*dwt_util_get_opt_stride*/(buffer_stride_y * step_x_max);
	float buffer_ptr[buffer_size_x*buffer_size_y] /*ALIGNED(16)*/;

	// copy area (x,y)..(+step_x_max,+step_y_max) into buffer[]
	copy_src_to_buff(
		// image
		src,
		src_stride_x,
		src_stride_y,
		size_x,
		size_y,
		x,
		y,
		// buffer
		buffer_ptr,
		buffer_stride_x,
		buffer_stride_y,
		// size of the tile
		buffer_size_x,
		buffer_size_y
	);

	// for each j in [0;J)
	for(int j = 0; j < J; j++)
	{
		// strides
		const int stride_x_j = mul_pow2(buffer_stride_x, j);
		const int stride_y_j = mul_pow2(buffer_stride_y, j);

		const int step_x_j = mul_pow2(step_x, j);
		const int step_y_j = mul_pow2(step_y, j);

		const int start_j = mul_pow2(1, j) - 1;

		const int len_x_j = step_x_j - start_j;
		const int len_y_j = step_y_j - start_j;

// 		dwt_util_log(LOG_DBG, "loop: j=%i start=(%i) limit=(%i,%i) len=(%i,%i) step=(%i,%i) stride=(%i,%i)\n",
// 			j, start_j, step_y_max, step_x_max, len_y_j, len_x_j, step_y_j, step_x_j, stride_y_j, stride_x_j);

		// for local_(x,y) in [0;step_max)
		for(int local_y = start_j; local_y+len_y_j-1 < step_y_max; local_y += step_y_j)
		for(int local_x = start_j; local_x+len_x_j-1 < step_x_max; local_x += step_x_j)
		{
			// buffers
			const int pos_x_j = ceil_div_pow2(x+local_x, j);
			const int pos_y_j = ceil_div_pow2(y+local_y, j);

			void *local_buffer_x = get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+pos_x_j), buff_elem_size);
			void *local_buffer_y = get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+pos_y_j), buff_elem_size);

// 			dwt_util_log(LOG_DBG, "\t core: local=(%i,%i) global=(%i,%i) global-normalized=(%i,%i) local-buffer=(%i,%i)\n",
// 				local_y, local_x, y+local_y, x+local_x, pos_y_j, pos_x_j,
// 				(int)(local_buffer_y-buffer_y)/buff_elem_size/sizeof(float),
// 				(int)(local_buffer_x-buffer_x)/buff_elem_size/sizeof(float)
// 			);

			void *block4x4 = addr2_s(buffer_ptr, local_y, local_x, buffer_stride_x, buffer_stride_y);

			// core4x4 (buffer=>buffer, no shift)
			direct_4x4(
				0,
				0,
				step_x_max,
				step_y_max,
				block4x4,
				stride_x_j,
				stride_y_j,
				block4x4,
				stride_x_j,
				stride_y_j,
				local_buffer_x,
				local_buffer_y
			);

			// FIXME: store HL/LH/HH coeffs directly here
		}
	}

	// copy coeffs from buffer[] into output (complicated pattern)
	copy_buff_to_dst(
		// buffer
		buffer_ptr,
		buffer_stride_x,
		buffer_stride_y,
		// image
		dst,
		dst_stride_x,
		dst_stride_y,
		size_x,
		size_y,
		x,
		y,
		// tile size
		buffer_size_x,
		buffer_size_y,
		// new
		J
	);
}

static
void multiscale_4x4_j_loop(
	// scale
	int J,
	// position
	int x,
	int y,
	// size
	int size_x,
	int size_y,
	// src
	void *src,
	int src_stride_x,
	int src_stride_y,
	// dst
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	// buffers
	void *buffer_x,
	void *buffer_y,
	// others
	int super_x,
	int super_y,
	int buffer_offset,
	// temp
	void *tmp,
	int tmp_stride_x,
	int tmp_stride_y
)
{
	const int step_y = 4;
	const int step_x = 4;

	const int step_x_max = mul_pow2(step_x, J-1);
	const int step_y_max = mul_pow2(step_y, J-1);

#if 0
	for(int j = 0; j < J; j++)
	{
		const int step_x_j = mul_pow2(step_x, j);
		const int step_y_j = mul_pow2(step_y, j);

		for(int yy = y; yy < y+step_y_max; yy += step_y_j)
		for(int xx = x; xx < x+step_x_max; xx += step_x_j)
		{
			multiscale_4x4_separately(
				// scale
				j,
				// position
				xx,
				yy,
				// size
				size_x,
				size_y,
				// data
				src, 		// input LL(j-1)
				src_stride_x,
				src_stride_y,
				src,		// FIXME: output LL(j)
				src_stride_x,
				src_stride_y,
				dst,		// output LH/HL/HH(j)
				dst_stride_x,
				dst_stride_y,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset
			);
		}
	}
#endif
#if 1
	for(int j = 0; j < 1; j++)
	{
		const int step_x_j = mul_pow2(step_x, j);
		const int step_y_j = mul_pow2(step_y, j);

// 		dwt_util_log(LOG_DBG, "j=%i step_j=(%i,%i)\n", j, step_y_j, step_x_j);

		for(int yy = y; yy < y+step_y_max; yy += step_y_j)
		for(int xx = x; xx < x+step_x_max; xx += step_x_j)
		{
// 			dwt_util_log(LOG_DBG, "multiscale_4x4...\n");
			multiscale_4x4_separately3(
				// scale
				j,
				// position
				xx,
				yy,
				// data
				src, 		// input LL(j-1)
				src_stride_x,
				src_stride_y,
				size_x,
				size_y,
				0,
				0,
				tmp,		// FIXME: output LL(j)
				tmp_stride_x,
				tmp_stride_y,
				size_x,
				size_y,
				0,
				0,
				dst,		// output LH/HL/HH(j)
				dst_stride_x,
				dst_stride_y,
				size_x,
				size_y,
				0,
				0,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset
			);
		}
	}
	for(int j = 1; j < J-1; j++)
	{
		const int step_x_j = mul_pow2(step_x, j);
		const int step_y_j = mul_pow2(step_y, j);

// 		dwt_util_log(LOG_DBG, "j=%i step_j=(%i,%i)\n", j, step_y_j, step_x_j);

		for(int yy = y; yy < y+step_y_max; yy += step_y_j)
		for(int xx = x; xx < x+step_x_max; xx += step_x_j)
		{
// 			dwt_util_log(LOG_DBG, "multiscale_4x4...\n");
			multiscale_4x4_separately3(
				// scale
				j,
				// position
				xx,
				yy,
				// data
				tmp, 		// input LL(j-1)
				tmp_stride_x,
				tmp_stride_y,
				size_x,
				size_y,
				0,
				0,
				tmp,		// FIXME: output LL(j)
				tmp_stride_x,
				tmp_stride_y,
				size_x,
				size_y,
				0,
				0,
				dst,		// output LH/HL/HH(j)
				dst_stride_x,
				dst_stride_y,
				size_x,
				size_y,
				0,
				0,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset
			);
		}
	}
	for(int j = J-1; j < J; j++)
	{
		const int step_x_j = mul_pow2(step_x, j);
		const int step_y_j = mul_pow2(step_y, j);

// 		dwt_util_log(LOG_DBG, "j=%i step_j=(%i,%i)\n", j, step_y_j, step_x_j);

		for(int yy = y; yy < y+step_y_max; yy += step_y_j)
		for(int xx = x; xx < x+step_x_max; xx += step_x_j)
		{
// 			dwt_util_log(LOG_DBG, "multiscale_4x4...\n");
			multiscale_4x4_separately3(
				// scale
				j,
				// position
				xx,
				yy,
				// data
				tmp, 		// input LL(j-1)
				tmp_stride_x,
				tmp_stride_y,
				size_x,
				size_y,
				0,
				0,
				dst,		// FIXME: output LL(j)
				dst_stride_x,
				dst_stride_y,
				size_x,
				size_y,
				0,
				0,
				dst,		// output LH/HL/HH(j)
				dst_stride_x,
				dst_stride_y,
				size_x,
				size_y,
				0,
				0,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset
			);
		}
	}
#endif
#if 0
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag_x = (step_x-1);
	const int lag_y = (step_y-1);

	// j = 0 (src => tmp)
	{
		const int j = 0;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int src_stride_x_j = mul_pow2(src_stride_x, j);
		const int src_stride_y_j = mul_pow2(src_stride_y, j);
		const int tmp_stride_x_j = mul_pow2(tmp_stride_x, j);
		const int tmp_stride_y_j = mul_pow2(tmp_stride_y, j);
		const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
		const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

		const int base_y_j = ceil_div_pow2(y, j) - lag_y;
		const int base_x_j = ceil_div_pow2(x, j) - lag_x;
		const int stop_y_j = ceil_div_pow2(y, j) - lag_y + ceil_div_pow2(step_y_max, j);
		const int stop_x_j = ceil_div_pow2(x, j) - lag_x + ceil_div_pow2(step_x_max, j);

		for(int y_j = base_y_j, local_y_j = 0; y_j < stop_y_j; y_j += step_y, local_y_j += step_y)
		for(int x_j = base_x_j, local_x_j = 0; x_j < stop_x_j; x_j += step_x, local_x_j += step_x)
		{
			unified_4x4_separately2(
				size_x_j, size_y_j,
				// load -- from (0,0)
				src, src_stride_x_j, src_stride_y_j, x_j, y_j, // input
				// store LL -- into (1-4,1-4)
				tmp, tmp_stride_x_j, tmp_stride_y_j, local_x_j+lag_x, local_y_j+lag_y,
				// store LH/HL/HH
				dst, dst_stride_x_j, dst_stride_y_j, x_j, y_j, // output HL/LH/HH
				// buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size)
			);
		}
	}

	// j = 1..J-2 (tmp => tmp)
	for(int j = 1; j < J-1; j++)
	{
		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int src_stride_x_j = mul_pow2(src_stride_x, j);
		const int src_stride_y_j = mul_pow2(src_stride_y, j);
		const int tmp_stride_x_j = mul_pow2(tmp_stride_x, j);
		const int tmp_stride_y_j = mul_pow2(tmp_stride_y, j);
		const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
		const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

		const int base_y_j = ceil_div_pow2(y, j) - lag_y;
		const int base_x_j = ceil_div_pow2(x, j) - lag_x;
		const int stop_y_j = ceil_div_pow2(y, j) - lag_y + ceil_div_pow2(step_y_max, j);
		const int stop_x_j = ceil_div_pow2(x, j) - lag_x + ceil_div_pow2(step_x_max, j);

		for(int y_j = base_y_j, local_y_j = 0; y_j < stop_y_j; y_j += step_y, local_y_j += step_y)
		for(int x_j = base_x_j, local_x_j = 0; x_j < stop_x_j; x_j += step_x, local_x_j += step_x)
		{

			unified_4x4_separately2(
				size_x_j, size_y_j,
				// load
				tmp, tmp_stride_x_j, tmp_stride_y_j, local_x_j, local_y_j,
				// store LL
				tmp, tmp_stride_x_j, tmp_stride_y_j, local_x_j+lag_x, local_y_j+lag_y,
				// store LH/HL/HH
				dst, dst_stride_x_j, dst_stride_y_j, x_j, y_j, // output HL/LH/HH
				// buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size)
			);
		}
	}

	// j = J-1 (tmp => dst)
	{
		const int j = J-1;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int src_stride_x_j = mul_pow2(src_stride_x, j);
		const int src_stride_y_j = mul_pow2(src_stride_y, j);
		const int tmp_stride_x_j = mul_pow2(tmp_stride_x, j);
		const int tmp_stride_y_j = mul_pow2(tmp_stride_y, j);
		const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
		const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

		const int x_j = ceil_div_pow2(x, j) - lag_x;
		const int y_j = ceil_div_pow2(y, j) - lag_y;

		const int local_x_j = 0;
		const int local_y_j = 0;

		unified_4x4_separately2(
			size_x_j, size_y_j,
			// load
			tmp, tmp_stride_x_j, tmp_stride_y_j, local_x_j, local_y_j,
			// store LL
			dst, dst_stride_x_j, dst_stride_y_j, x_j, y_j, // output LL
			// store LH/HL/HH
			dst, dst_stride_x_j, dst_stride_y_j, x_j, y_j, // output HL/LH/HH
			// buffers
			get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
			get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size)
		);
	}
#endif
#if 0
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	const int lag_x = (step_x-1);
	const int lag_y = (step_y-1);

// 	dwt_util_log(LOG_DBG, "pos=(%i,%i)\n",
// 		y, x
// 	);

	for(int j = 0; j < J; j++)
	{
		const int is_first = !j;
		const int not_first = !is_first;
		const int is_last  = !(j-(J-1));
		const int not_last = !is_last;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		void *src_j = select_p(is_first, src, tmp);
		const int src_stride_x_j = mul_pow2(select_i(is_first, src_stride_x, tmp_stride_x), j);
		const int src_stride_y_j = mul_pow2(select_i(is_first, src_stride_y, tmp_stride_y), j);

		void *low_j = select_p(is_last, dst, tmp);
		const int low_stride_x_j = mul_pow2(select_i(is_last, dst_stride_x, tmp_stride_x), j);
		const int low_stride_y_j = mul_pow2(select_i(is_last, dst_stride_y, tmp_stride_y), j);

		const int dst_stride_x_j = mul_pow2(dst_stride_x, j);
		const int dst_stride_y_j = mul_pow2(dst_stride_y, j);

		const int base_y_j = ceil_div_pow2(y, j) - lag_y;
		const int base_x_j = ceil_div_pow2(x, j) - lag_x;
		const int stop_y_j = base_y_j + ceil_div_pow2(step_y_max, j);
		const int stop_x_j = base_x_j + ceil_div_pow2(step_x_max, j);

		for(int y_j = base_y_j; y_j < stop_y_j; y_j += step_y)
		for(int x_j = base_x_j; x_j < stop_x_j; x_j += step_x)
		{
			// src: j=0: global; otherwise: local
			const int src_pos_x = x_j - not_first * base_x_j;
			const int src_pos_y = y_j - not_first * base_y_j;
			// low: j=J-1: global; otherwise: local
			const int low_pos_x = x_j - not_last * base_x_j + not_last * lag_x;
			const int low_pos_y = y_j - not_last * base_y_j + not_last * lag_y;

// 			dwt_util_log(LOG_DBG, "low-pos=(%i,%i) low-stride=(%i,%i)\n",
// 				low_pos_y, low_pos_x,  low_stride_x_j, low_stride_y_j
// 			);

			unified_4x4_separately2(
				size_x_j, size_y_j,
				// load
				src_j, src_stride_x_j, src_stride_y_j, src_pos_x, src_pos_y,
				// store LL
				low_j, low_stride_x_j, low_stride_y_j, low_pos_x, low_pos_y,
				// store LH/HL/HH
				dst, dst_stride_x_j, dst_stride_y_j, x_j, y_j,
				// buffers
				get_buffer_ptr(buffer_x, j, buffer_x_elems, (buffer_offset+x_j), buff_elem_size),
				get_buffer_ptr(buffer_y, j, buffer_y_elems, (buffer_offset+y_j), buff_elem_size)
			);
		}
	}
#endif
}

static
void ms_loop_unified_4x4_fused(
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
// 	const int words = 1; // vertical
// 	const int buff_elem_size = words*4;

	// core size
	const int step_y = 4;
	const int step_x = 4;

// 	const int buffer_x_elems = buff_elem_size*super_x;
// 	const int buffer_y_elems = buff_elem_size*super_y;

	// FIXME: what distance between read and write head is really needed?
// 	const int lag = 0;

// 	const int lag_x = (step_x-1) + lag;
// 	const int lag_y = (step_y-1) + lag;

	const int step_x_max = mul_pow2(step_x, J-1);
	const int step_y_max = mul_pow2(step_y, J-1);

	const int tmp_stride_y = sizeof(float);
	const int tmp_stride_x = dwt_util_get_opt_stride(tmp_stride_y * step_x_max);
	const int tmp_size = tmp_stride_x * step_y_max;
	char tmp[tmp_size];

// 	dwt_util_log(LOG_DBG, "step_max=(%i,%i) J=%i\n", step_y_max, step_x_max, J);

	// order=horizontal
	for(int y = base_y; y < stop_y; y += step_y_max)
	{
		for(int x = base_x; x < stop_x; x += step_x_max)
		{
#if 0
			multiscale_j_top(
				J-1,
				// position
				x,
				y,
				// size
				size_x,
				size_y,
				// src
				ptr,
				stride_x,
				stride_y,
				// dst
				ptr,
				stride_x,
				stride_y,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset,
				1, NULL, 0, 0
			);
#endif
#if 1
			multiscale_4x4_j_loop(
				// scale
				J,
				// position
				x,
				y,
				// size
				size_x,
				size_y,
				// src
				ptr,
				stride_x,
				stride_y,
				// dst
				ptr,
				stride_x,
				stride_y,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset,
				// temp
				tmp,
				tmp_stride_x,
				tmp_stride_y
			);
#endif
		}
	}
}

static
void ms_loop_unified_4x4_fused2(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int size_x, int size_y,
	void *src,
	int src_stride_x,
	int src_stride_y,
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	float *buffer_x,
	float *buffer_y,
	int J,
	int super_x,
	int super_y,
	int buffer_offset
)
{
// 	const int words = 1; // vertical
// 	const int buff_elem_size = words*4;

	// core size
	const int step_y = 4;
	const int step_x = 4;

// 	const int buffer_x_elems = buff_elem_size*super_x;
// 	const int buffer_y_elems = buff_elem_size*super_y;

	const int step_x_max = mul_pow2(step_x, J-1);
	const int step_y_max = mul_pow2(step_y, J-1);

#if 0
	// FIXME: persistent buffer
	const int tmp_stride_y = sizeof(float);
	const int tmp_stride_x = dwt_util_get_opt_stride(tmp_stride_y * step_x_max);
	const int tmp_size = tmp_stride_x * step_y_max;
	char tmp[tmp_size];
#endif

// 	dwt_util_log(LOG_DBG, "step_max=(%i,%i) J=%i\n", step_y_max, step_x_max, J);

	// order=horizontal
	for(int y = base_y; y < stop_y; y += step_y_max)
	{
		for(int x = base_x; x < stop_x; x += step_x_max)
		{
// 			dwt_util_log(LOG_DBG, "ms-core @ (%i,%i)\n", y, x);
#if 0
			multiscale_j_top(
				J-1,
				// position
				x,
				y,
				// size
				size_x,
				size_y,
				// src
				src,
				src_stride_x,
				src_stride_y,
				// dst
				dst,
				dst_stride_x,
				dst_stride_y,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset,
				1, NULL, 0, 0
			);
#endif
#if 0
			multiscale_4x4_j_loop(
				// scale
				J,
				// position
				x,
				y,
				// size
				size_x,
				size_y,
				// src
				src,
				src_stride_x,
				src_stride_y,
				// dst
				dst,
				dst_stride_x,
				dst_stride_y,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset,
				// temp
				tmp,
				tmp_stride_x,
				tmp_stride_y
			);
#endif
#if 1
			multiscale_4x4_j_loop_new(
				// scale
				J,
				// position
				x,
				y,
				// size
				size_x,
				size_y,
				// src
				src,
				src_stride_x,
				src_stride_y,
				// dst
				dst,
				dst_stride_x,
				dst_stride_y,
				// buffers
				buffer_x,
				buffer_y,
				// others
				super_x,
				super_y,
				buffer_offset
			);
#endif
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
void ms_cdf97_2f_dl_4x4_fused_s(
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

	// maximal coordinate in negative value (e.g., -(step_x-1) - lag = -3 -0 = -3)
	const int buff_guard = 32;
	// FIXME: what overlap is really needed?
	// 4(j=2), 8(j=3), 32(j=5), 0(j=5)
	const int overlap_L = 128;
	// 13<<J ... with lag = 18
	//  7<<J ... with lag =  8 (artifacts)
	//  5<<J ... with lag =  4 (buffers)
	//   128 ... with lag =  0 (j=5)
	const int overlap_R = 128;

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
		ms_loop_unified_4x4_fused(
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
)
{
	// TODO: assert

	// offset_x, offset_y
	const int offset = 1;

	// core size
// 	const int step_y = 4;
// 	const int step_x = 4;

	// vertical vectorization
// 	const int shift = 4;

	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	// maximal coordinate in negative value
	const int buff_guard = 64;
	const int overlap_L = -(offset-4-(4<<J));
	const int overlap_R = (4<<J)-(1<<J);

	const int super_x = buff_guard + overlap_L + size_x + overlap_R;
	const int super_y = buff_guard + overlap_L + size_y + overlap_R;

	const int buffer_x_elems = buff_elem_size*super_x;
	const int buffer_y_elems = buff_elem_size*super_y;

	// alloc buffers
	float buffer_x[J*buffer_x_elems] ALIGNED(16);
	float buffer_y[J*buffer_y_elems] ALIGNED(16);

#if 0
	dwt_util_zero_vec_s(buffer_x, J*buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, J*buffer_y_elems);
	dwt_util_test_image_zero_s(
		dst,
		dst_stride_x,
		dst_stride_y,
		size_x,
		size_y
	);
#endif

	const int buffer_offset = buff_guard+overlap_L;

	// unified loop
	{
		ms_loop_unified_4x4_fused2(
			/* base */ -overlap_L, -overlap_L, // x, y
			/* stop */ size_x+overlap_R, size_y+overlap_R, // x, y
			/* size */ size_x, size_y,
			src, src_stride_x, src_stride_y,
			dst, dst_stride_x, dst_stride_y,
			buffer_x, buffer_y,
			J,
			super_x, super_y,
			buffer_offset
		);
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
			ms_cdf97_2f_dl_4x4_fused_s(
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

// macro
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

// function
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
