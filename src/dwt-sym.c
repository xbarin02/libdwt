#include "dwt-sym.h"
#include "libdwt.h"
#define MEASURE_FACTOR 1
#define MEASURE_PER_PIXEL
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

// vert_2x4 inv.
#ifdef __SSE__
static
void vert_2x4_inv(
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
	const __m128 w0 = { +dwt_cdf97_p1_s, +dwt_cdf97_p1_s, +dwt_cdf97_p1_s, +dwt_cdf97_p1_s };
	const __m128 w1 = { -dwt_cdf97_u1_s, -dwt_cdf97_u1_s, -dwt_cdf97_u1_s, -dwt_cdf97_u1_s };
	const __m128 w2 = { +dwt_cdf97_p2_s, +dwt_cdf97_p2_s, +dwt_cdf97_p2_s, +dwt_cdf97_p2_s };
	const __m128 w3 = { -dwt_cdf97_u2_s, -dwt_cdf97_u2_s, -dwt_cdf97_u2_s, -dwt_cdf97_u2_s };

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

// ~ fdwt_cdf97_vert_cor4x4_sse_s
static
void vert_4x4(
	intptr_t src_y0_x0, // pointer to (0,0)
	ptrdiff_t src_stride_x, // +1 row
	ptrdiff_t src_stride_y, // +1 col
	intptr_t dst_y0_x0, // pointer to (0-shift,0-shift)
	ptrdiff_t dst_stride_x, // +1 row
	ptrdiff_t dst_stride_y, // +1 col
	float *buff_h0, // +(0..3)*(1*4) [ y down> ]
	float *buff_v0  // +(0..3)*(1*4) [ x right> ]
)
{
#ifdef __SSE__
	// this 4x4 core approach corresponds to "transpose-SIMD" in Figure 9 in Kutil2006 (the "line-SIMD" should be 8x2 core)
	__m128 t0, t1, t2, t3;

	// load 4x4
	t0 = (__m128){
		*(float *)(src_y0_x0 + 0*src_stride_x + 0*src_stride_y),
		*(float *)(src_y0_x0 + 1*src_stride_x + 0*src_stride_y),
		*(float *)(src_y0_x0 + 2*src_stride_x + 0*src_stride_y),
		*(float *)(src_y0_x0 + 3*src_stride_x + 0*src_stride_y)
	};
	t1 = (__m128){
		*(float *)(src_y0_x0 + 0*src_stride_x + 1*src_stride_y),
		*(float *)(src_y0_x0 + 1*src_stride_x + 1*src_stride_y),
		*(float *)(src_y0_x0 + 2*src_stride_x + 1*src_stride_y),
		*(float *)(src_y0_x0 + 3*src_stride_x + 1*src_stride_y)
	};
	t2 = (__m128){
		*(float *)(src_y0_x0 + 0*src_stride_x + 2*src_stride_y),
		*(float *)(src_y0_x0 + 1*src_stride_x + 2*src_stride_y),
		*(float *)(src_y0_x0 + 2*src_stride_x + 2*src_stride_y),
		*(float *)(src_y0_x0 + 3*src_stride_x + 2*src_stride_y)
	};
	t3 = (__m128){
		*(float *)(src_y0_x0 + 0*src_stride_x + 3*src_stride_y),
		*(float *)(src_y0_x0 + 1*src_stride_x + 3*src_stride_y),
		*(float *)(src_y0_x0 + 2*src_stride_x + 3*src_stride_y),
		*(float *)(src_y0_x0 + 3*src_stride_x + 3*src_stride_y)
	};

	// left horiz.
	vert_2x4(
		t0,
		t1,
		&t0,
		&t1,
		buff_h0
	);

	// right horiz
	vert_2x4(
		t2,
		t3,
		&t2,
		&t3,
		buff_h0
	);

	// shuffle t0..3
	_MM_TRANSPOSE4_PS(t0, t1, t2, t3);

	// top vert
	vert_2x4(
		t0,
		t1,
		&t0,
		&t1,
		buff_v0
	);
	// bottom vert
	vert_2x4(
		t2,
		t3,
		&t2,
		&t3,
		buff_v0
	);

	const float z = dwt_cdf97_s1_s;
	t0 *= (const __m128){ 1/(z*z),   1.f, 1/(z*z),   1.f };
	t1 *= (const __m128){     1.f, (z*z),     1.f, (z*z) };
	t2 *= (const __m128){ 1/(z*z),   1.f, 1/(z*z),   1.f };
	t3 *= (const __m128){     1.f, (z*z),     1.f, (z*z) };

	*(float *)(dst_y0_x0 + 0*dst_stride_x + 0*dst_stride_y) = t0[0];
	*(float *)(dst_y0_x0 + 1*dst_stride_x + 0*dst_stride_y) = t1[0];
	*(float *)(dst_y0_x0 + 2*dst_stride_x + 0*dst_stride_y) = t2[0];
	*(float *)(dst_y0_x0 + 3*dst_stride_x + 0*dst_stride_y) = t3[0];

	*(float *)(dst_y0_x0 + 0*dst_stride_x + 1*dst_stride_y) = t0[1];
	*(float *)(dst_y0_x0 + 1*dst_stride_x + 1*dst_stride_y) = t1[1];
	*(float *)(dst_y0_x0 + 2*dst_stride_x + 1*dst_stride_y) = t2[1];
	*(float *)(dst_y0_x0 + 3*dst_stride_x + 1*dst_stride_y) = t3[1];

	*(float *)(dst_y0_x0 + 0*dst_stride_x + 2*dst_stride_y) = t0[2];
	*(float *)(dst_y0_x0 + 1*dst_stride_x + 2*dst_stride_y) = t1[2];
	*(float *)(dst_y0_x0 + 2*dst_stride_x + 2*dst_stride_y) = t2[2];
	*(float *)(dst_y0_x0 + 3*dst_stride_x + 2*dst_stride_y) = t3[2];

	*(float *)(dst_y0_x0 + 0*dst_stride_x + 3*dst_stride_y) = t0[3];
	*(float *)(dst_y0_x0 + 1*dst_stride_x + 3*dst_stride_y) = t1[3];
	*(float *)(dst_y0_x0 + 2*dst_stride_x + 3*dst_stride_y) = t2[3];
	*(float *)(dst_y0_x0 + 3*dst_stride_x + 3*dst_stride_y) = t3[3];
#endif /* __SSE__ */
}

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

#define CORE_4X4_CALC_INV(t0, t1, t2, t3, buff_h, buff_v) \
do { \
	CORE_4X4_SCALE((t0), (t1), (t2), (t3)); \
	\
	vert_2x4_inv((t0), (t1), &(t0), &(t1), (buff_h)); \
	vert_2x4_inv((t2), (t3), &(t2), &(t3), (buff_h)); \
	\
	_MM_TRANSPOSE4_PS((t0), (t1), (t2), (t3)); \
	\
	vert_2x4_inv((t0), (t1), &(t0), &(t1), (buff_v)); \
	vert_2x4_inv((t2), (t3), &(t2), &(t3), (buff_v)); \
} while(0)

#ifdef __SSE__
static
void core_4x4_calc(
	__m128 *t0,
	__m128 *t1,
	__m128 *t2,
	__m128 *t3,
	float *buff_h0, // +(0..3)*(1*4) [ y down> ]
	float *buff_v0  // +(0..3)*(1*4) [ x right> ]
)
{
#if 0
	// left horiz.
	vert_2x4(
		*t0,
		*t1,
		t0,
		t1,
		buff_h0
	);

	// right horiz
	vert_2x4(
		*t2,
		*t3,
		t2,
		t3,
		buff_h0
	);

	// shuffle t0..3
	_MM_TRANSPOSE4_PS(*t0, *t1, *t2, *t3);

	// top vert
	vert_2x4(
		*t0,
		*t1,
		t0,
		t1,
		buff_v0
	);
	// bottom vert
	vert_2x4(
		*t2,
		*t3,
		t2,
		t3,
		buff_v0
	);

	CORE_4X4_SCALE(*t0, *t1, *t2, *t3);
#else
	CORE_4X4_CALC(*t0, *t1, *t2, *t3, buff_h0, buff_v0);
#endif
}
#endif

// ~ fdwt_vert_4x4_cor_HORIZ
static
void block_vert_4x4_cor(
	const void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int buff_elem_size = 1*4; // vert

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	const char *src_y0_x0 = (const void *)addr2_const_s(src_ptr, base_y, base_x, src_stride_x, src_stride_y);
	char       *dst_y0_x0 = (void *)      addr2_s      (dst_ptr, base_y, base_x, dst_stride_x, dst_stride_y);

	// increments
	const ptrdiff_t src_diff_y4 = (ptrdiff_t)addr1_s(0, +4, src_stride_x); // +4 cols
	const ptrdiff_t src_diff_x4 = (ptrdiff_t)addr1_s(0, +4, src_stride_y); // +4 rows
	const ptrdiff_t dst_diff_y4 = (ptrdiff_t)addr1_s(0, +4, dst_stride_x); // +4 cols
	const ptrdiff_t dst_diff_x4 = (ptrdiff_t)addr1_s(0, +4, dst_stride_y); // +4 rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+3 < stop_y; y += 4)
	{
		const char *src_y0_x0_i = src_y0_x0;
		char *dst_y0_x0_i = dst_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+3 < stop_x; x += 4)
		{
			vert_4x4(
				// src
				(intptr_t)src_y0_x0_i,
				src_stride_x,
				src_stride_y,
				// dst
				(intptr_t)dst_y0_x0_i,
				dst_stride_x,
				dst_stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			src_y0_x0_i += src_diff_x4;
			dst_y0_x0_i += dst_diff_x4;

			buffer_x0_i += 4*(buff_elem_size);
		}

		src_y0_x0 += src_diff_y4;
		dst_y0_x0 += dst_diff_y4;

		buffer_y0_i += 4*(buff_elem_size);
	}
}

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
	int overlap_x_L, int size_x,
	int overlap_y_L, int size_y,
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
	const int step_y = 4;
	const int step_x = 4;

	const int shift = 4; // vertical vectorization

	__m128 t[4];

	// CORE -- LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, size_y);

			t[xx][yy] = *addr2_const_s(src_ptr, pos_y, pos_x, src_stride_x, src_stride_y);
		}
	}

	// CORE -- CALC
#if 0
	core_4x4_calc(
		&t[0],
		&t[1],
		&t[2],
		&t[3],
		real_buffer_y,
		real_buffer_x
	);
#else
	CORE_4X4_CALC(t[0], t[1], t[2], t[3], buffer_y, buffer_x);
#endif

	// CORE -- STORE
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			// virtual => real coordinates
#if 0
			const int pos_x = virt2real(x-shift, xx, overlap_x_L, size_x);
			const int pos_y = virt2real(y-shift, yy, overlap_y_L, size_y);
#else
			const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
			const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;
#endif
			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
#endif /* __SSE__ */
}

static
void unified_4x4_inv(
	int x, int y,
	int overlap_x_L, int size_x,
	int overlap_y_L, int size_y,
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
	const int step_y = 4;
	const int step_x = 4;

	const int shift = 4; // vertical vectorization

	__m128 t[4];

	// CORE -- LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual => real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, size_y);

			t[xx][yy] = *addr2_const_s(src_ptr, pos_y, pos_x, src_stride_x, src_stride_y);
		}
	}

	// CORE -- CALC
	CORE_4X4_CALC_INV(t[0], t[1], t[2], t[3], buffer_y, buffer_x);

	// CORE -- STORE
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			// virtual => real coordinates
#if 0
			const int pos_x = virt2real(x-shift, xx, overlap_x_L, size_x);
			const int pos_y = virt2real(y-shift, yy, overlap_y_L, size_y);
#else
			const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
			const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;
#endif
			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) = t[yy][xx];
		}
	}
#endif /* __SSE__ */
}

static
void loop_shorted_4x4(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int step_x, int step_y,
	int overlap_x_L, int size_x,
	int overlap_y_L, int size_y,
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
	UNUSED(step_x);
	UNUSED(step_y);
	UNUSED(size_x);
	UNUSED(size_y);
	UNUSED(overlap_x_L);
	UNUSED(overlap_y_L);

	block_vert_4x4_cor(
		src_ptr,
		src_stride_x,
		src_stride_y,
		dst_ptr,
		dst_stride_x,
		dst_stride_y,
		base_x,
		base_y,
		stop_x,
		stop_y,
		buffer_y,
		buffer_x
	);
}

static
void loop_unified_4x4(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int step_x, int step_y,
	int overlap_x_L, int size_x,
	int overlap_y_L, int size_y,
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
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

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
			unified_4x4(
				x, y,
				overlap_x_L, size_x,
				overlap_y_L, size_y,
				src_ptr, src_stride_x, src_stride_y,
				dst_ptr, dst_stride_x, dst_stride_y,
				buffer_x + x*buff_elem_size,
				buffer_y + y*buff_elem_size
			);
		}
	}
}

static
void loop_unified_4x4_inv(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int step_x, int step_y,
	int overlap_x_L, int size_x,
	int overlap_y_L, int size_y,
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
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

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
			unified_4x4_inv(
				x, y,
				overlap_x_L, size_x,
				overlap_y_L, size_y,
				src_ptr, src_stride_x, src_stride_y,
				dst_ptr, dst_stride_x, dst_stride_y,
				buffer_x + x*buff_elem_size,
				buffer_y + y*buff_elem_size
			);
		}
	}
}

static
void loop_unified_4x4_V(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int step_x, int step_y,
	int overlap_x_L, int size_x,
	int overlap_y_L, int size_y,
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
	const int words = 1; // vertical
	const int buff_elem_size = words*4;

	// order=vertical
	for(int x = base_x; x < stop_x; x += step_x)
	{
		for(int y = base_y; y < stop_y; y += step_y)
		{
			unified_4x4(
				x, y,
				overlap_x_L, size_x,
				overlap_y_L, size_y,
				src_ptr, src_stride_x, src_stride_y,
				dst_ptr, dst_stride_x, dst_stride_y,
				buffer_x + x*buff_elem_size,
				buffer_y + y*buff_elem_size
			);
		}
	}
}

// like loop_unified_4x4 but buffered
static
void loop_unified_4x4_B(
	int base_x, int base_y,
	int stop_x, int stop_y,
	int step_x, int step_y,
	int overlap_x_L, int size_x,
	int overlap_y_L, int size_y,
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
	const int words = 1;
	const int buff_elem_size = words*4;

	const int shift = 4;

	const int blocks_x = (stop_x-base_x)/step_x;
	const int blocks_y = (stop_y-base_y)/step_y;

	const int tmp_size_x = step_x*blocks_x;
	const int tmp_size_y = step_y*blocks_y;
	const int tmp_stride_y = sizeof(float);
	const int tmp_stride_x = tmp_stride_y * tmp_size_x;
	char tmp[tmp_size_y*tmp_stride_x];

	float *tmp_ptr = addr2_s(tmp, shift-base_y+overlap_y_L, shift-base_x+overlap_x_L, tmp_stride_x, tmp_stride_y);

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
			unified_4x4(
				x, y,
				overlap_x_L, size_x,
				overlap_y_L, size_y,
				src_ptr, src_stride_x, src_stride_y,
				tmp_ptr, tmp_stride_x, tmp_stride_y,
				buffer_x + x*buff_elem_size,
				buffer_y + y*buff_elem_size
			);
		}
	}

	// copy "tmp" => "dst"
	for(int y = 0; y < tmp_size_y; y++)
	{
		for(int x = 0; x < tmp_size_x; x++)
		{
			const int pos_x = virt2real_error(base_x-shift, x, overlap_x_L, size_x);
			const int pos_y = virt2real_error(base_y-shift, y, overlap_y_L, size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst_ptr, pos_y, pos_x, dst_stride_x, dst_stride_y) =
			*addr2_s(tmp_ptr, pos_y, pos_x, tmp_stride_x, tmp_stride_y);
		}
	}
}

void cdf97_2f_dl_4x4_s(
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
	// TODO: assert

	const int words = 1; // 1 = vertical vectorization
	const int buff_elem_size = words*4;

	// offset_x, offset_y
	const int offset = 1;

	// step_x, step_y
	const int step_y = 4;
	const int step_x = 4;

	// shift_x, shift_y
	const int shift = 4; // 4 = vertical vectorization

	const int modulo_x_L = (offset) % step_x;
	const int overlap_x_L = step_x + !!modulo_x_L * (step_x-modulo_x_L); // (step_x-modulo_x_L) + (!!offset * step_x);
	const int modulo_x_R = (overlap_x_L+size_x) % step_x;
	const int overlap_x_R = step_x + !!modulo_x_R * (step_x-modulo_x_R); // step_x + ((modulo_x_R) ? (step_x-modulo_x_R) : (0));
	const int super_x = overlap_x_L + size_x + overlap_x_R;
	const int limit0_x = overlap_x_L + offset + shift;
	const int limit1_x = overlap_x_L + size_x - modulo_x_R - step_x*!modulo_x_R; // HACK: last term should not be here

// 	dwt_util_log(LOG_DBG, "mod_L=%i ovl_L=%i mod_R=%i ovl_R=%i => %i+%i+%i | limits: %i %i \n",
// 		modulo_x_L, overlap_x_L, modulo_x_R, overlap_x_R,
// 		overlap_x_L, size_x, overlap_x_R,
// 		limit0_x, limit1_x
//     		);

	const int modulo_y_L = (offset) % step_y;
	const int overlap_y_L = step_y + !!modulo_y_L * (step_y-modulo_y_L); // (step_x-modulo_x_L) + (!!offset * step_x);
	const int modulo_y_R = (overlap_y_L+size_y) % step_y;
	const int overlap_y_R = step_y + !!modulo_y_R * (step_y-modulo_y_R); // step_x + ((modulo_x_R) ? (step_x-modulo_x_R) : (0));
	const int super_y = overlap_y_L + size_y + overlap_y_R;
	const int limit0_y = overlap_y_L + offset + shift;
	const int limit1_y = overlap_y_L + size_y - modulo_y_R - step_y*!modulo_y_R; // HACK: last term should not be here

	// alloc buffers
	float buffer_x[buff_elem_size*super_x] ALIGNED(16);
	float buffer_y[buff_elem_size*super_y] ALIGNED(16);

	// zero buffers
#if 0
	dwt_util_zero_vec_s(buffer_x, buff_elem_size*super_x);
	dwt_util_zero_vec_s(buffer_y, buff_elem_size*super_y);
#endif

#if 0 /* one big loop */
	// unified loop
	{
		loop_unified_4x4(
			/* base */ 0, 0,
			/* stop */ super_x, super_y,
			/* step */ step_x, step_y,
			overlap_x_L, size_x,
			overlap_y_L, size_y,
			src_ptr, src_stride_x, src_stride_y,
			dst_ptr, dst_stride_x, dst_stride_y,
			buffer_x,
			buffer_y
		);
	}
#else /* one big loop */
	// top strip
	loop_unified_4x4(
		/* base */ 0, 0,
		/* stop */ super_x, limit0_y,
		/* step */ step_x, step_y,
		overlap_x_L, size_x,
		overlap_y_L, size_y,
		src_ptr, src_stride_x, src_stride_y,
		dst_ptr, dst_stride_x, dst_stride_y,
		buffer_x,
		buffer_y
	);

	// left strip
	loop_unified_4x4(
		/* base */ 0, limit0_y,
		/* stop */ limit0_x, limit1_y,
		/* step */ step_x, step_y,
		overlap_x_L, size_x,
		overlap_y_L, size_y,
		src_ptr, src_stride_x, src_stride_y,
		dst_ptr, dst_stride_x, dst_stride_y,
		buffer_x,
		buffer_y
	);

	// core strip
#if 0
	loop_unified_4x4(
		/* base */ limit0_x, limit0_y,
		/* stop */ limit1_x, limit1_y,
		/* step */ step_x, step_y,
		overlap_x_L, size_x,
		overlap_y_L, size_y,
		src_ptr, src_stride_x, src_stride_y,
		dst_ptr, dst_stride_x, dst_stride_y,
		buffer_x,
		buffer_y
	);
#else
	{
		const void *src_ptr_shifted = addr2_const_s(src_ptr,       -overlap_y_L,       -overlap_x_L, dst_stride_x, dst_stride_y);
		void       *dst_ptr_shifted =       addr2_s(dst_ptr, -shift-overlap_y_L, -shift-overlap_x_L, dst_stride_x, dst_stride_y);

		// TODO: parallelize this using threads
		loop_shorted_4x4(
			/* base */ limit0_x, limit0_y,
			/* stop */ limit1_x, limit1_y,
			/* step */ step_x, step_y,
			overlap_x_L, size_x,
			overlap_y_L, size_y,
			src_ptr_shifted, src_stride_x, src_stride_y,
			dst_ptr_shifted, dst_stride_x, dst_stride_y,
			buffer_x,
			buffer_y
		);
	}
#endif

	// right strip
	loop_unified_4x4(
		/* base */ limit1_x, limit0_y,
		/* stop */ super_x, limit1_y,
		/* step */ step_x, step_y,
		overlap_x_L, size_x,
		overlap_y_L, size_y,
		src_ptr, src_stride_x, src_stride_y,
		dst_ptr, dst_stride_x, dst_stride_y,
		buffer_x,
		buffer_y
	);

	// bottom strip
	loop_unified_4x4_V(
		/* base */ 0, limit1_y,
		/* stop */ limit1_x, super_y,
		/* step */ step_x, step_y,
		overlap_x_L, size_x,
		overlap_y_L, size_y,
		src_ptr, src_stride_x, src_stride_y,
		dst_ptr, dst_stride_x, dst_stride_y,
		buffer_x,
		buffer_y
	);

	// right-bottom corner
	loop_unified_4x4_B(
		/* base */ limit1_x, limit1_y,
		/* stop */ super_x, super_y,
		/* step */ step_x, step_y,
		overlap_x_L, size_x,
		overlap_y_L, size_y,
		src_ptr, src_stride_x, src_stride_y,
		dst_ptr, dst_stride_x, dst_stride_y,
		buffer_x,
		buffer_y
	);
#endif /* one big loop */
}

// TODO: in-place transforms is not treated
void cdf97_2i_dl_4x4_s(
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
	// TODO: assert

	const int words = 1; // 1 = vertical vectorization
	const int buff_elem_size = words*4;

	// offset_x, offset_y
	const int offset = 0; // FIXME: for FORWARD == 1

	// step_x, step_y
	const int step_y = 4;
	const int step_x = 4;

	// shift_x, shift_y
	const int shift = 4; // 4 = vertical vectorization

	const int modulo_x_L = (offset) % step_x;
	const int overlap_x_L = step_x + !!modulo_x_L * (step_x-modulo_x_L); // (step_x-modulo_x_L) + (!!offset * step_x);
	const int modulo_x_R = (overlap_x_L+size_x) % step_x;
	const int overlap_x_R = step_x + !!modulo_x_R * (step_x-modulo_x_R); // step_x + ((modulo_x_R) ? (step_x-modulo_x_R) : (0));
	const int super_x = overlap_x_L + size_x + overlap_x_R;
// 	const int limit0_x = overlap_x_L + offset + shift;
// 	const int limit1_x = overlap_x_L + size_x - modulo_x_R - step_x*!modulo_x_R; // HACK: last term should not be here

// 	dwt_util_log(LOG_DBG, "mod_L=%i ovl_L=%i mod_R=%i ovl_R=%i => %i+%i+%i | limits: %i %i \n",
// 		modulo_x_L, overlap_x_L, modulo_x_R, overlap_x_R,
// 		overlap_x_L, size_x, overlap_x_R,
// 		limit0_x, limit1_x
//     		);

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

	// zero buffers
#if 0
	dwt_util_zero_vec_s(buffer_x, buff_elem_size*super_x);
	dwt_util_zero_vec_s(buffer_y, buff_elem_size*super_y);
#endif

	// unified loop
	{
		loop_unified_4x4_inv(
			/* base */ 0, 0,
			/* stop */ super_x, super_y,
			/* step */ step_x, step_y,
			overlap_x_L, size_x,
			overlap_y_L, size_y,
			src_ptr, src_stride_x, src_stride_y,
			dst_ptr, dst_stride_x, dst_stride_y,
			buffer_x,
			buffer_y
		);
	}
}

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
			cdf97_2f_dl_4x4_s(
				size_x,
				size_y,
				ptr[m],
				stride_x,
				stride_y,
				ptr[m],
				stride_x,
				stride_y
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
			dwt_cdf97_2i_inplace_s(ptr[m], stride_x, stride_y, size_x, size_y, size_x, size_y, 1, 0, 0);
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

void dwt_util_measure_perf_cdf97_2f_dl_4x4_s(
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data
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
		dwt_util_perf_cdf97_2f_dl_4x4_s(
			size_x,
			size_y,
			opt_stride,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs,
			1, // flush
			0 // type
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

			cdf97_2f_dl_4x4_s(
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
)
{
	UNUSED(zero_padding);

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if( 0 == j )
			break;

		const int stride_y_j = mul_pow2(stride_y, j-1);
		const int stride_x_j = mul_pow2(stride_x, j-1);

		const int size_x_j = ceil_div_pow2(size_x, j-1);
		const int size_y_j = ceil_div_pow2(size_y, j-1);

		if( size_x_j < 8 || size_y_j < 8 )
		{
			// FIXME
			dwt_util_error("unimplemented\n");
		}
		else
		{
// 			dwt_util_log(LOG_DBG, "j=%i: size=(%i,%i) stride=(%i,%i)\n", j, size_x_j, size_y_j, stride_x_j, stride_y_j);

			cdf97_2i_dl_4x4_s(
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

		j--;
	}
}

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
			int j = J;

			dwt_cdf97_2f_dl_4x4_s(
				ptr[m],
				stride_x,
				stride_y,
				size_x,
				size_y,
				size_x,
				size_y,
				&j,
				0,
				0
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
		dwt_util_perf_dwt_cdf97_2f_dl_4x4_s(
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
)
{
	dwt_util_measure_perf_cdf97_2_inplace_sep_s(
		DWT_ARR_PACKED,
		min_x,
		max_x,
		opt_stride,
		J,
		0,
		0,
		M,
		N,
		clock_type,
		fwd_plot_data,
		inv_plot_data
	);
}
