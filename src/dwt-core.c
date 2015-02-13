#include "dwt-core.h"
#include "inline.h"
#include "libdwt.h"
#include "dwt.h"
#include <assert.h>
// #include "dwt-core-test.h"

#ifdef __SSE__
	#include <xmmintrin.h>
#endif
#include "inline-sdl.h"

void fdwt_cdf97_diag_pro2x2_sse_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outR0,
	float *outL1, float *outR1,
	float *lAL,
	float *lAR,
	float *lBL,
	float *lBR
)
{
#ifdef __SSE__
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	__m128 buff;
	__m128 z;

	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	// A/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lAL+4), *(__m128 *)(lAL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lAR+4), *(__m128 *)(lAR+8));

	// A/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAL+4), w, *(__m128 *)(lAL+0), *(__m128 *)(lAL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lAL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAL+4), *(__m128 *)(lAL+0), *(__m128 *)(lAL+8), z);

	// A/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAR+4), w, *(__m128 *)(lAR+0), *(__m128 *)(lAR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lAR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAR+4), *(__m128 *)(lAR+0), *(__m128 *)(lAR+8), z);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lBL+4), *(__m128 *)(lBL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lBR+4), *(__m128 *)(lBR+8));

	// B/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBL+4), w, *(__m128 *)(lBL+0), *(__m128 *)(lBL+8));
	op4s_sdl2_update_s_sse(*(__m128 *)(lBL+4), *(__m128 *)(lBL+0), *(__m128 *)(lBL+8), z);

	// B/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBR+4), w, *(__m128 *)(lBR+0), *(__m128 *)(lBR+8));
	op4s_sdl2_update_s_sse(*(__m128 *)(lBR+4), *(__m128 *)(lBR+0), *(__m128 *)(lBR+8), z);
#endif /* __SSE__ */
}

// full (with register rotation) diagonal core (horizontally only)
// returns in format [ y0x0 y0x1 y1x0 y1x1 ]
#ifdef __SSE__
static
__m128 diag_horizontally_2x2(
	__m128 in, // in format [ y0x0 y0x1 y1x0 y1x1 ]
	// top (y=0) horizontal buffer
	__m128 *buff_h0_0, // l
	__m128 *buff_h0_4, // c
	__m128 *buff_h0_8, // r
	// bottom (y=1) horizontal buffer
	__m128 *buff_h1_0, // l
	__m128 *buff_h1_4, // c
	__m128 *buff_h1_8  // r
)
{
	const __m128 w = {
		+dwt_cdf97_u2_s,
		-dwt_cdf97_p2_s,
		+dwt_cdf97_u1_s,
		-dwt_cdf97_p1_s
	};

	__m128 z;

	op4s_sdl2_shuffle_input_low_s_sse(in, *buff_h0_4, *buff_h0_8);
	op4s_sdl2_shuffle_input_high_s_sse(in, *buff_h1_4, *buff_h1_8);

	op4s_sdl2_op_s_sse(z, *buff_h0_4, w, *buff_h0_0, *buff_h0_8);
	op4s_sdl2_output_low_s_sse(in, *buff_h0_0, z);
	op4s_sdl2_update_s_sse(*buff_h0_4, *buff_h0_0, *buff_h0_8, z);

	op4s_sdl2_op_s_sse(z, *buff_h1_4, w, *buff_h1_0, *buff_h1_8);
	op4s_sdl2_output_high_s_sse(in, *buff_h1_0, z);
	op4s_sdl2_update_s_sse(*buff_h1_4, *buff_h1_0, *buff_h1_8, z);

	return in;
}
#endif /* __SSE__ */

#ifdef __SSE__
#define diag_horizontally_2x2_MACRO(in, buff_h0_0, buff_h0_4, buff_h0_8, buff_h1_0, buff_h1_4, buff_h1_8) \
	do { \
		const __m128 w = { \
			+dwt_cdf97_u2_s, \
			-dwt_cdf97_p2_s, \
			+dwt_cdf97_u1_s, \
			-dwt_cdf97_p1_s \
		}; \
	 \
		__m128 z; \
	 \
		op4s_sdl2_shuffle_input_low_s_sse(in, *buff_h0_4, *buff_h0_8); \
		op4s_sdl2_shuffle_input_high_s_sse(in, *buff_h1_4, *buff_h1_8); \
	 \
		op4s_sdl2_op_s_sse(z, *buff_h0_4, w, *buff_h0_0, *buff_h0_8); \
		op4s_sdl2_output_low_s_sse(in, *buff_h0_0, z); \
		op4s_sdl2_update_s_sse(*buff_h0_4, *buff_h0_0, *buff_h0_8, z); \
	 \
		op4s_sdl2_op_s_sse(z, *buff_h1_4, w, *buff_h1_0, *buff_h1_8); \
		op4s_sdl2_output_high_s_sse(in, *buff_h1_0, z); \
		op4s_sdl2_update_s_sse(*buff_h1_4, *buff_h1_0, *buff_h1_8, z); \
	} while(0)
#endif /* __SSE__ */

// fast (without register rotation) diagonal core (horizontally only)
// returns in format [ y0x0 y0x1 y1x0 y1x1 ]
#ifdef __SSE__
static
__m128 diag_horizontally_2x2_fast(
	__m128 in, // in format [ y0x0 y0x1 y1x0 y1x1 ]
	// top (y=0) horizontal buffer
	__m128 *buff_h0_0, // l
	__m128 *buff_h0_4, // c
	__m128 *buff_h0_8, // r
	// bottom (y=1) horizontal buffer
	__m128 *buff_h1_0, // l
	__m128 *buff_h1_4, // c
	__m128 *buff_h1_8  // r
)
{
	const __m128 w = {
		+dwt_cdf97_u2_s,
		-dwt_cdf97_p2_s,
		+dwt_cdf97_u1_s,
		-dwt_cdf97_p1_s
	};

	__m128 z;

	op4s_sdl2_shuffle_input_low_s_sse(in, *buff_h0_4, *buff_h0_8);
	op4s_sdl2_shuffle_input_high_s_sse(in, *buff_h1_4, *buff_h1_8);

	op4s_sdl2_op_s_sse(z, *buff_h0_4, w, *buff_h0_0, *buff_h0_8);
	op4s_sdl2_output_low_s_sse(in, *buff_h0_0, z);
	op4s_sdl2_update_s_sse_FAST(*buff_h0_4, *buff_h0_0, *buff_h0_8, z);

	op4s_sdl2_op_s_sse(z, *buff_h1_4, w, *buff_h1_0, *buff_h1_8);
	op4s_sdl2_output_high_s_sse(in, *buff_h1_0, z);
	op4s_sdl2_update_s_sse_FAST(*buff_h1_4, *buff_h1_0, *buff_h1_8, z);

	return in;
}
#endif /* __SSE__ */

#ifdef __SSE__
#define diag_horizontally_2x2_FAST_MACRO(in, buff_h0_0, buff_h0_4, buff_h0_8, buff_h1_0, buff_h1_4, buff_h1_8) \
	do { \
		const __m128 w = { \
			+dwt_cdf97_u2_s, \
			-dwt_cdf97_p2_s, \
			+dwt_cdf97_u1_s, \
			-dwt_cdf97_p1_s \
		}; \
	 \
		__m128 z; \
	 \
		op4s_sdl2_shuffle_input_low_s_sse(in, *buff_h0_4, *buff_h0_8); \
		op4s_sdl2_shuffle_input_high_s_sse(in, *buff_h1_4, *buff_h1_8); \
	 \
		op4s_sdl2_op_s_sse(z, *buff_h0_4, w, *buff_h0_0, *buff_h0_8); \
		op4s_sdl2_output_low_s_sse(in, *buff_h0_0, z); \
		op4s_sdl2_update_s_sse_FAST(*buff_h0_4, *buff_h0_0, *buff_h0_8, z); \
	 \
		op4s_sdl2_op_s_sse(z, *buff_h1_4, w, *buff_h1_0, *buff_h1_8); \
		op4s_sdl2_output_high_s_sse(in, *buff_h1_0, z); \
		op4s_sdl2_update_s_sse_FAST(*buff_h1_4, *buff_h1_0, *buff_h1_8, z); \
	} while(0)
#endif /* __SSE__ */

#ifdef __SSE__
static
void diag_6x2(
	// input by 2x2 blocks in format [ y0x0 y0x1 y1x0 y1x1 ]
	__m128 in0,
	__m128 in1,
	__m128 in2,
	// output by 2x2 blocks in format [ y0x0 y1x0 y0x1 y1x1 ]
	__m128 *out0,
	__m128 *out1,
	__m128 *out2,
	// buffers with stride = (3*4)*sizeof(float)
	float *buff_h0, // +(0..1)*(3*4) [ y down> ]
	float *buff_v0  // +(0..5)*(3*4) [ x right> ]
)
{
#if 0
	float *buff_h1 = buff_h0 + 1*(3*4);

	float *buff_v1 = buff_v0 + 1*(3*4);
	float *buff_v2 = buff_v0 + 2*(3*4);
	float *buff_v3 = buff_v0 + 3*(3*4);
	float *buff_v4 = buff_v0 + 4*(3*4);
	float *buff_v5 = buff_v0 + 5*(3*4);

	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f, 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s) };

	__m128 z;

	// ====== 2x2 y=0..1 x=0..1 ======

	// horizontally
	op4s_sdl2_shuffle_input_low_s_sse(in0, *(__m128 *)(buff_h0+4), *(__m128 *)(buff_h0+8));
	op4s_sdl2_shuffle_input_high_s_sse(in0, *(__m128 *)(buff_h1+4), *(__m128 *)(buff_h1+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h0+4), w, *(__m128 *)(buff_h0+0), *(__m128 *)(buff_h0+8));
	op4s_sdl2_output_low_s_sse(in0, *(__m128 *)(buff_h0+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_h0+4), *(__m128 *)(buff_h0+0), *(__m128 *)(buff_h0+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h1+4), w, *(__m128 *)(buff_h1+0), *(__m128 *)(buff_h1+8));
	op4s_sdl2_output_high_s_sse(in0, *(__m128 *)(buff_h1+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_h1+4), *(__m128 *)(buff_h1+0), *(__m128 *)(buff_h1+8), z);

	in0 = _mm_shuffle_ps(in0, in0, _MM_SHUFFLE(3,1,2,0));

	// vertically
	op4s_sdl2_shuffle_input_low_s_sse(in0, *(__m128 *)(buff_v0+4), *(__m128 *)(buff_v0+8));
	op4s_sdl2_shuffle_input_high_s_sse(in0, *(__m128 *)(buff_v1+4), *(__m128 *)(buff_v1+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v0+4), w, *(__m128 *)(buff_v0+0), *(__m128 *)(buff_v0+8));
	op4s_sdl2_output_low_s_sse(in0, *(__m128 *)(buff_v0+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v0+4), *(__m128 *)(buff_v0+0), *(__m128 *)(buff_v0+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v1+4), w, *(__m128 *)(buff_v1+0), *(__m128 *)(buff_v1+8));
	op4s_sdl2_output_high_s_sse(in0, *(__m128 *)(buff_v1+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v1+4), *(__m128 *)(buff_v1+0), *(__m128 *)(buff_v1+8), z);

	op4s_sdl2_scale_s_sse(in0, v);

	*out0 = in0;

	// ====== 2x2 y=0..1 x=2..3 ======

	// horizontally
	op4s_sdl2_shuffle_input_low_s_sse(in1, *(__m128 *)(buff_h0+4), *(__m128 *)(buff_h0+8));
	op4s_sdl2_shuffle_input_high_s_sse(in1, *(__m128 *)(buff_h1+4), *(__m128 *)(buff_h1+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h0+4), w, *(__m128 *)(buff_h0+0), *(__m128 *)(buff_h0+8));
	op4s_sdl2_output_low_s_sse(in1, *(__m128 *)(buff_h0+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_h0+4), *(__m128 *)(buff_h0+0), *(__m128 *)(buff_h0+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h1+4), w, *(__m128 *)(buff_h1+0), *(__m128 *)(buff_h1+8));
	op4s_sdl2_output_high_s_sse(in1, *(__m128 *)(buff_h1+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_h1+4), *(__m128 *)(buff_h1+0), *(__m128 *)(buff_h1+8), z);

	in1 = _mm_shuffle_ps(in1, in1, _MM_SHUFFLE(3,1,2,0));

	// vertically
	op4s_sdl2_shuffle_input_low_s_sse(in1, *(__m128 *)(buff_v2+4), *(__m128 *)(buff_v2+8));
	op4s_sdl2_shuffle_input_high_s_sse(in1, *(__m128 *)(buff_v3+4), *(__m128 *)(buff_v3+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v2+4), w, *(__m128 *)(buff_v2+0), *(__m128 *)(buff_v2+8));
	op4s_sdl2_output_low_s_sse(in1, *(__m128 *)(buff_v2+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v2+4), *(__m128 *)(buff_v2+0), *(__m128 *)(buff_v2+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v3+4), w, *(__m128 *)(buff_v3+0), *(__m128 *)(buff_v3+8));
	op4s_sdl2_output_high_s_sse(in1, *(__m128 *)(buff_v3+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v3+4), *(__m128 *)(buff_v3+0), *(__m128 *)(buff_v3+8), z);

	op4s_sdl2_scale_s_sse(in1, v);

	*out1 = in1;

	// ====== 2x2 y=0..1 x=4..5 ======

	// horizontally
	op4s_sdl2_shuffle_input_low_s_sse(in2, *(__m128 *)(buff_h0+4), *(__m128 *)(buff_h0+8));
	op4s_sdl2_shuffle_input_high_s_sse(in2, *(__m128 *)(buff_h1+4), *(__m128 *)(buff_h1+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h0+4), w, *(__m128 *)(buff_h0+0), *(__m128 *)(buff_h0+8));
	op4s_sdl2_output_low_s_sse(in2, *(__m128 *)(buff_h0+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_h0+4), *(__m128 *)(buff_h0+0), *(__m128 *)(buff_h0+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h1+4), w, *(__m128 *)(buff_h1+0), *(__m128 *)(buff_h1+8));
	op4s_sdl2_output_high_s_sse(in2, *(__m128 *)(buff_h1+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_h1+4), *(__m128 *)(buff_h1+0), *(__m128 *)(buff_h1+8), z);

	in2 = _mm_shuffle_ps(in2, in2, _MM_SHUFFLE(3,1,2,0));

	// vertically
	op4s_sdl2_shuffle_input_low_s_sse(in2, *(__m128 *)(buff_v4+4), *(__m128 *)(buff_v4+8));
	op4s_sdl2_shuffle_input_high_s_sse(in2, *(__m128 *)(buff_v5+4), *(__m128 *)(buff_v5+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v4+4), w, *(__m128 *)(buff_v4+0), *(__m128 *)(buff_v4+8));
	op4s_sdl2_output_low_s_sse(in2, *(__m128 *)(buff_v4+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v4+4), *(__m128 *)(buff_v4+0), *(__m128 *)(buff_v4+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v5+4), w, *(__m128 *)(buff_v5+0), *(__m128 *)(buff_v5+8));
	op4s_sdl2_output_high_s_sse(in2, *(__m128 *)(buff_v5+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v5+4), *(__m128 *)(buff_v5+0), *(__m128 *)(buff_v5+8), z);

	op4s_sdl2_scale_s_sse(in2, v);

	*out2 = in2;
#endif
#if 0
	// FAST :-)

	float *buff_h1 = buff_h0 + 1*(3*4);

	float *buff_v1 = buff_v0 + 1*(3*4);
	float *buff_v2 = buff_v0 + 2*(3*4);
	float *buff_v3 = buff_v0 + 3*(3*4);
	float *buff_v4 = buff_v0 + 4*(3*4);
	float *buff_v5 = buff_v0 + 5*(3*4);

	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f, 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s) };

	__m128 z;

	#define op4s_sdl2_update_s_sse_X(c, l, r, z) do { (c) = (z); } while(0)

	// ====== 2x2 y=0..1 x=0..1 ======
// 1 iter.
#define H0 0
#define H4 4
#define H8 8
	// horizontally
	op4s_sdl2_shuffle_input_low_s_sse(in0, *(__m128 *)(buff_h0+H4), *(__m128 *)(buff_h0+H8));
	op4s_sdl2_shuffle_input_high_s_sse(in0, *(__m128 *)(buff_h1+H4), *(__m128 *)(buff_h1+H8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h0+H4), w, *(__m128 *)(buff_h0+H0), *(__m128 *)(buff_h0+H8));
	op4s_sdl2_output_low_s_sse(in0, *(__m128 *)(buff_h0+H0), z);
	op4s_sdl2_update_s_sse_X(*(__m128 *)(buff_h0+H4), *(__m128 *)(buff_h0+H0), *(__m128 *)(buff_h0+H8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h1+H4), w, *(__m128 *)(buff_h1+H0), *(__m128 *)(buff_h1+H8));
	op4s_sdl2_output_high_s_sse(in0, *(__m128 *)(buff_h1+H0), z);
	op4s_sdl2_update_s_sse_X(*(__m128 *)(buff_h1+H4), *(__m128 *)(buff_h1+H0), *(__m128 *)(buff_h1+H8), z);
#undef H0
#undef H4
#undef H8
	in0 = _mm_shuffle_ps(in0, in0, _MM_SHUFFLE(3,1,2,0));

	// vertically
	op4s_sdl2_shuffle_input_low_s_sse(in0, *(__m128 *)(buff_v0+4), *(__m128 *)(buff_v0+8));
	op4s_sdl2_shuffle_input_high_s_sse(in0, *(__m128 *)(buff_v1+4), *(__m128 *)(buff_v1+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v0+4), w, *(__m128 *)(buff_v0+0), *(__m128 *)(buff_v0+8));
	op4s_sdl2_output_low_s_sse(in0, *(__m128 *)(buff_v0+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v0+4), *(__m128 *)(buff_v0+0), *(__m128 *)(buff_v0+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v1+4), w, *(__m128 *)(buff_v1+0), *(__m128 *)(buff_v1+8));
	op4s_sdl2_output_high_s_sse(in0, *(__m128 *)(buff_v1+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v1+4), *(__m128 *)(buff_v1+0), *(__m128 *)(buff_v1+8), z);

	op4s_sdl2_scale_s_sse(in0, v);

	*out0 = in0;

	// ====== 2x2 y=0..1 x=2..3 ======
// 2 iter.
#define H0 8
#define H4 0
#define H8 4
	// horizontally
	op4s_sdl2_shuffle_input_low_s_sse(in1, *(__m128 *)(buff_h0+H4), *(__m128 *)(buff_h0+H8));
	op4s_sdl2_shuffle_input_high_s_sse(in1, *(__m128 *)(buff_h1+H4), *(__m128 *)(buff_h1+H8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h0+H4), w, *(__m128 *)(buff_h0+H0), *(__m128 *)(buff_h0+H8));
	op4s_sdl2_output_low_s_sse(in1, *(__m128 *)(buff_h0+H0), z);
	op4s_sdl2_update_s_sse_X(*(__m128 *)(buff_h0+H4), *(__m128 *)(buff_h0+H0), *(__m128 *)(buff_h0+H8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h1+H4), w, *(__m128 *)(buff_h1+H0), *(__m128 *)(buff_h1+H8));
	op4s_sdl2_output_high_s_sse(in1, *(__m128 *)(buff_h1+H0), z);
	op4s_sdl2_update_s_sse_X(*(__m128 *)(buff_h1+H4), *(__m128 *)(buff_h1+H0), *(__m128 *)(buff_h1+H8), z);
#undef H0
#undef H4
#undef H8
	in1 = _mm_shuffle_ps(in1, in1, _MM_SHUFFLE(3,1,2,0));

	// vertically
	op4s_sdl2_shuffle_input_low_s_sse(in1, *(__m128 *)(buff_v2+4), *(__m128 *)(buff_v2+8));
	op4s_sdl2_shuffle_input_high_s_sse(in1, *(__m128 *)(buff_v3+4), *(__m128 *)(buff_v3+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v2+4), w, *(__m128 *)(buff_v2+0), *(__m128 *)(buff_v2+8));
	op4s_sdl2_output_low_s_sse(in1, *(__m128 *)(buff_v2+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v2+4), *(__m128 *)(buff_v2+0), *(__m128 *)(buff_v2+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v3+4), w, *(__m128 *)(buff_v3+0), *(__m128 *)(buff_v3+8));
	op4s_sdl2_output_high_s_sse(in1, *(__m128 *)(buff_v3+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v3+4), *(__m128 *)(buff_v3+0), *(__m128 *)(buff_v3+8), z);

	op4s_sdl2_scale_s_sse(in1, v);

	*out1 = in1;

	// ====== 2x2 y=0..1 x=4..5 ======
// 3 iter.
#define H0 4
#define H4 8
#define H8 0
	// horizontally
	op4s_sdl2_shuffle_input_low_s_sse(in2, *(__m128 *)(buff_h0+H4), *(__m128 *)(buff_h0+H8));
	op4s_sdl2_shuffle_input_high_s_sse(in2, *(__m128 *)(buff_h1+H4), *(__m128 *)(buff_h1+H8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h0+H4), w, *(__m128 *)(buff_h0+H0), *(__m128 *)(buff_h0+H8));
	op4s_sdl2_output_low_s_sse(in2, *(__m128 *)(buff_h0+H0), z);
	op4s_sdl2_update_s_sse_X(*(__m128 *)(buff_h0+H4), *(__m128 *)(buff_h0+H0), *(__m128 *)(buff_h0+H8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_h1+H4), w, *(__m128 *)(buff_h1+H0), *(__m128 *)(buff_h1+H8));
	op4s_sdl2_output_high_s_sse(in2, *(__m128 *)(buff_h1+H0), z);
	op4s_sdl2_update_s_sse_X(*(__m128 *)(buff_h1+H4), *(__m128 *)(buff_h1+H0), *(__m128 *)(buff_h1+H8), z);
#undef H0
#undef H4
#undef H8
	in2 = _mm_shuffle_ps(in2, in2, _MM_SHUFFLE(3,1,2,0));

	// vertically
	op4s_sdl2_shuffle_input_low_s_sse(in2, *(__m128 *)(buff_v4+4), *(__m128 *)(buff_v4+8));
	op4s_sdl2_shuffle_input_high_s_sse(in2, *(__m128 *)(buff_v5+4), *(__m128 *)(buff_v5+8));
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v4+4), w, *(__m128 *)(buff_v4+0), *(__m128 *)(buff_v4+8));
	op4s_sdl2_output_low_s_sse(in2, *(__m128 *)(buff_v4+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v4+4), *(__m128 *)(buff_v4+0), *(__m128 *)(buff_v4+8), z);
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buff_v5+4), w, *(__m128 *)(buff_v5+0), *(__m128 *)(buff_v5+8));
	op4s_sdl2_output_high_s_sse(in2, *(__m128 *)(buff_v5+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(buff_v5+4), *(__m128 *)(buff_v5+0), *(__m128 *)(buff_v5+8), z);

	op4s_sdl2_scale_s_sse(in2, v);

	*out2 = in2;
#endif
#if 1
	float *buff_h1 = buff_h0 + 1*(3*4);

	float *buff_v1 = buff_v0 + 1*(3*4);
	float *buff_v2 = buff_v0 + 2*(3*4);
	float *buff_v3 = buff_v0 + 3*(3*4);
	float *buff_v4 = buff_v0 + 4*(3*4);
	float *buff_v5 = buff_v0 + 5*(3*4);

	const __m128 w = { +dwt_cdf97_u2_s, -dwt_cdf97_p2_s, +dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f, 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s) };

	__m128 z;

	// ====== 2x2 y=0..1 x=0..1 ======
// iter. 1
#define H0 0
#define H4 4
#define H8 8
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in0,
		(__m128 *)(buff_h0+H0),
		(__m128 *)(buff_h0+H4),
		(__m128 *)(buff_h0+H8),
		(__m128 *)(buff_h1+H0),
		(__m128 *)(buff_h1+H4),
		(__m128 *)(buff_h1+H8)
	);

	_MM_TRANSPOSE1_PS(in0);

	// vertically
	diag_horizontally_2x2_MACRO(
		in0,
		(__m128 *)(buff_v0+0),
		(__m128 *)(buff_v0+4),
		(__m128 *)(buff_v0+8),
		(__m128 *)(buff_v1+0),
		(__m128 *)(buff_v1+4),
		(__m128 *)(buff_v1+8)
	);

	op4s_sdl2_scale_s_sse(in0, v);

	*out0 = in0;
#undef H0
#undef H4
#undef H8

	// ====== 2x2 y=0..1 x=2..3 ======
// iter. 2
#define H0 8
#define H4 0
#define H8 4
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in1,
		(__m128 *)(buff_h0+H0),
		(__m128 *)(buff_h0+H4),
		(__m128 *)(buff_h0+H8),
		(__m128 *)(buff_h1+H0),
		(__m128 *)(buff_h1+H4),
		(__m128 *)(buff_h1+H8)
	);

	_MM_TRANSPOSE1_PS(in1);

	// vertically
	diag_horizontally_2x2_MACRO(
		in1,
		(__m128 *)(buff_v2+0),
		(__m128 *)(buff_v2+4),
		(__m128 *)(buff_v2+8),
		(__m128 *)(buff_v3+0),
		(__m128 *)(buff_v3+4),
		(__m128 *)(buff_v3+8)
	);

	op4s_sdl2_scale_s_sse(in1, v);

	*out1 = in1;
#undef H0
#undef H4
#undef H8

	// ====== 2x2 y=0..1 x=4..5 ======
// iter. 3
#define H0 4
#define H4 8
#define H8 0
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in2,
		(__m128 *)(buff_h0+H0),
		(__m128 *)(buff_h0+H4),
		(__m128 *)(buff_h0+H8),
		(__m128 *)(buff_h1+H0),
		(__m128 *)(buff_h1+H4),
		(__m128 *)(buff_h1+H8)
	);

	_MM_TRANSPOSE1_PS(in2);

	// vertically
	diag_horizontally_2x2_MACRO(
		in2,
		(__m128 *)(buff_v4+0),
		(__m128 *)(buff_v4+4),
		(__m128 *)(buff_v4+8),
		(__m128 *)(buff_v5+0),
		(__m128 *)(buff_v5+4),
		(__m128 *)(buff_v5+8)
	);

	op4s_sdl2_scale_s_sse(in2, v);

	*out2 = in2;
#undef H0
#undef H4
#undef H8

#endif
}
#endif /* __SSE__ */

#ifdef __SSE__
static
void diag_2x6(
	// input by 2x2 blocks in format [ y0x0 y0x1 y1x0 y1x1 ]
	__m128 in0,
	__m128 in1,
	__m128 in2,
	// output by 2x2 blocks in format [ y0x0 y1x0 y0x1 y1x1 ]
	__m128 *out0,
	__m128 *out1,
	__m128 *out2,
	// buffers with stride = (3*4)*sizeof(float)
	float *buff_h0, // +(0..5)*(3*4) [ y down> ]
	float *buff_v0  // +(0..1)*(3*4) [ x right> ]
)
{
#if 1
	float *buff_h1 = buff_h0 + 1*(3*4);
	float *buff_h2 = buff_h0 + 2*(3*4);
	float *buff_h3 = buff_h0 + 3*(3*4);
	float *buff_h4 = buff_h0 + 4*(3*4);
	float *buff_h5 = buff_h0 + 5*(3*4);

	float *buff_v1 = buff_v0 + 1*(3*4);

	const __m128 w = {
		+dwt_cdf97_u2_s,
		-dwt_cdf97_p2_s,
		+dwt_cdf97_u1_s,
		-dwt_cdf97_p1_s
	};

	const float zeta2 = dwt_cdf97_s1_s*dwt_cdf97_s1_s;
	const __m128 v = {
		1/zeta2, 1.f,
		1.f,   zeta2
	};

	__m128 z;

	// ====== 2x2 y=0..1 x=0..1 ======
// iter. 1
#define H0 0
#define H4 4
#define H8 8
	// horizontally
	diag_horizontally_2x2_MACRO(
		in0,
		(__m128 *)(buff_h0+0),
		(__m128 *)(buff_h0+4),
		(__m128 *)(buff_h0+8),
		(__m128 *)(buff_h1+0),
		(__m128 *)(buff_h1+4),
		(__m128 *)(buff_h1+8)
	);

	_MM_TRANSPOSE1_PS(in0);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in0,
		(__m128 *)(buff_v0+H0),
		(__m128 *)(buff_v0+H4),
		(__m128 *)(buff_v0+H8),
		(__m128 *)(buff_v1+H0),
		(__m128 *)(buff_v1+H4),
		(__m128 *)(buff_v1+H8)
	);

	op4s_sdl2_scale_s_sse(in0, v);

	*out0 = in0;
#undef H0
#undef H4
#undef H8

	// ====== 2x2 y=2..3 x=0..1 ======
// iter. 2
#define H0 8
#define H4 0
#define H8 4
	// horizontally
	diag_horizontally_2x2_MACRO(
		in1,
		(__m128 *)(buff_h2+0),
		(__m128 *)(buff_h2+4),
		(__m128 *)(buff_h2+8),
		(__m128 *)(buff_h3+0),
		(__m128 *)(buff_h3+4),
		(__m128 *)(buff_h3+8)
	);

	_MM_TRANSPOSE1_PS(in1);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in1,
		(__m128 *)(buff_v0+H0),
		(__m128 *)(buff_v0+H4),
		(__m128 *)(buff_v0+H8),
		(__m128 *)(buff_v1+H0),
		(__m128 *)(buff_v1+H4),
		(__m128 *)(buff_v1+H8)
	);

	op4s_sdl2_scale_s_sse(in1, v);

	*out1 = in1;
#undef H0
#undef H4
#undef H8

	// ====== 2x2 y=4..5 x=0..1 ======
// iter. 3
#define H0 4
#define H4 8
#define H8 0
	// horizontally
	diag_horizontally_2x2_MACRO(
		in2,
		(__m128 *)(buff_h4+0),
		(__m128 *)(buff_h4+4),
		(__m128 *)(buff_h4+8),
		(__m128 *)(buff_h5+0),
		(__m128 *)(buff_h5+4),
		(__m128 *)(buff_h5+8)
	);

	_MM_TRANSPOSE1_PS(in2);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in2,
		(__m128 *)(buff_v0+H0),
		(__m128 *)(buff_v0+H4),
		(__m128 *)(buff_v0+H8),
		(__m128 *)(buff_v1+H0),
		(__m128 *)(buff_v1+H4),
		(__m128 *)(buff_v1+H8)
	);

	op4s_sdl2_scale_s_sse(in2, v);

	*out2 = in2;
#undef H0
#undef H4
#undef H8

#endif
}
#endif /* __SSE__ */

#ifdef __SSE__
static
void diag_6x6(
	// input by 2x2 blocks in format [ y0x0 y0x1 y1x0 y1x1 ]
	__m128 in00, // y=0 x=0
	__m128 in01,
	__m128 in02,
	__m128 in10, // y=2 x=0
	__m128 in11,
	__m128 in12,
	__m128 in20, // y=4 x=0
	__m128 in21,
	__m128 in22,
	// output by 2x2 blocks in format [ y0x0 y1x0 y0x1 y1x1 ]
	__m128 *out00, // y=0 x=0
	__m128 *out01,
	__m128 *out02,
	__m128 *out10, // y=2 x=0
	__m128 *out11,
	__m128 *out12,
	__m128 *out20, // y=4 x=0
	__m128 *out21,
	__m128 *out22,
	// buffers with stride = (3*4)*sizeof(float)
	float *buff_h0, // +(0..5)*(3*4) [ y down> ]
	float *buff_v0  // +(0..5)*(3*4) [ x right> ]
)
{
#if 1
	// TODO

	float *buff_h1 = buff_h0 + 1*(3*4);
	float *buff_h2 = buff_h0 + 2*(3*4);
	float *buff_h3 = buff_h0 + 3*(3*4);
	float *buff_h4 = buff_h0 + 4*(3*4);
	float *buff_h5 = buff_h0 + 5*(3*4);

	float *buff_v1 = buff_v0 + 1*(3*4);
	float *buff_v2 = buff_v0 + 2*(3*4);
	float *buff_v3 = buff_v0 + 3*(3*4);
	float *buff_v4 = buff_v0 + 4*(3*4);
	float *buff_v5 = buff_v0 + 5*(3*4);

	const __m128 w = {
		+dwt_cdf97_u2_s,
		-dwt_cdf97_p2_s,
		+dwt_cdf97_u1_s,
		-dwt_cdf97_p1_s
	};

	const float zeta2 = dwt_cdf97_s1_s*dwt_cdf97_s1_s;
	const __m128 v = {
		1/zeta2, 1.f,
		1.f,   zeta2
	};

	__m128 z;

	// legend: x0 x4 x8
	// iter 1:  0  4  8
	// iter 2:  8  0  4
	// iter 3:  4  8  0
#define NEW_ORDER
	// ====== 2x2 y=0..1 x=0..1 ======
	// iter: 1. H + 1. V
#ifdef NEW_ORDER
	#define V0 0
	#define V4 4
	#define V8 8
	#define H0 0
	#define H4 4
	#define H8 8
#else
	#define H0 0
	#define H4 4
	#define H8 8
	#define V0 0
	#define V4 4
	#define V8 8
#endif
#define buff_H0 buff_h0
#define buff_H1 buff_h1
#define buff_V0 buff_v0
#define buff_V1 buff_v1
#define in in00
#define out out00
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=0..1 x=2..3 ======
	// iter: 2. H + 1. V
#ifdef NEW_ORDER
	#define V0 8
	#define V4 0
	#define V8 4
	#define H0 0
	#define H4 4
	#define H8 8
#else
	#define H0 8
	#define H4 0
	#define H8 4
	#define V0 0
	#define V4 4
	#define V8 8
#endif
#define buff_H0 buff_h2
#define buff_H1 buff_h3
#define buff_V0 buff_v0
#define buff_V1 buff_v1
#define in in01
#define out out01
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=0..1 x=4..5 ======
	// iter: 3. H + 1. V
#ifdef NEW_ORDER
	#define V0 4
	#define V4 8
	#define V8 0
	#define H0 0
	#define H4 4
	#define H8 8
#else
	#define H0 4
	#define H4 8
	#define H8 0
	#define V0 0
	#define V4 4
	#define V8 8
#endif
#define buff_H0 buff_h4
#define buff_H1 buff_h5
#define buff_V0 buff_v0
#define buff_V1 buff_v1
#define in in02
#define out out02
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=2..3 x=0..1 ======
	// iter: 1. H + 2. V
#ifdef NEW_ORDER
	#define V0 0
	#define V4 4
	#define V8 8
	#define H0 8
	#define H4 0
	#define H8 4
#else
	#define H0 0
	#define H4 4
	#define H8 8
	#define V0 8
	#define V4 0
	#define V8 4
#endif
#define buff_H0 buff_h0
#define buff_H1 buff_h1
#define buff_V0 buff_v2
#define buff_V1 buff_v3
#define in in10
#define out out10
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=2..3 x=2..3 ======
	// iter: 2. H + 2. V
#ifdef NEW_ORDER
	#define V0 8
	#define V4 0
	#define V8 4
	#define H0 8
	#define H4 0
	#define H8 4
#else
	#define H0 8
	#define H4 0
	#define H8 4
	#define V0 8
	#define V4 0
	#define V8 4
#endif
#define buff_H0 buff_h2
#define buff_H1 buff_h3
#define buff_V0 buff_v2
#define buff_V1 buff_v3
#define in in11
#define out out11
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=2..3 x=4..5 ======
	// iter: 3. H + 2. V
#ifdef NEW_ORDER
	#define V0 4
	#define V4 8
	#define V8 0
	#define H0 8
	#define H4 0
	#define H8 4
#else
	#define H0 4
	#define H4 8
	#define H8 0
	#define V0 8
	#define V4 0
	#define V8 4
#endif
#define buff_H0 buff_h4
#define buff_H1 buff_h5
#define buff_V0 buff_v2
#define buff_V1 buff_v3
#define in in12
#define out out12
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=4..5 x=0..1 ======
	// iter: 1. H + 3. V
#ifdef NEW_ORDER
	#define V0 0
	#define V4 4
	#define V8 8
	#define H0 4
	#define H4 8
	#define H8 0
#else
	#define H0 0
	#define H4 4
	#define H8 8
	#define V0 4
	#define V4 8
	#define V8 0
#endif
#define buff_H0 buff_h0
#define buff_H1 buff_h1
#define buff_V0 buff_v4
#define buff_V1 buff_v5
#define in in20
#define out out20
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=4..5 x=2..3 ======
	// iter: 2. H + 3. V
#ifdef NEW_ORDER
	#define V0 8
	#define V4 0
	#define V8 4
	#define H0 4
	#define H4 8
	#define H8 0
#else
	#define H0 8
	#define H4 0
	#define H8 4
	#define V0 4
	#define V4 8
	#define V8 0
#endif
#define buff_H0 buff_h2
#define buff_H1 buff_h3
#define buff_V0 buff_v4
#define buff_V1 buff_v5
#define in in21
#define out out21
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// ====== 2x2 y=4..5 x=4..5 ======
	// iter: 3. H + 3. V
#ifdef NEW_ORDER
	#define V0 4
	#define V4 8
	#define V8 0
	#define H0 4
	#define H4 8
	#define H8 0
#else
	#define H0 4
	#define H4 8
	#define H8 0
	#define V0 4
	#define V4 8
	#define V8 0
#endif
#define buff_H0 buff_h4
#define buff_H1 buff_h5
#define buff_V0 buff_v4
#define buff_V1 buff_v5
#define in in22
#define out out22
	// horizontally
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_H0+H0),
		(__m128 *)(buff_H0+H4),
		(__m128 *)(buff_H0+H8),
		(__m128 *)(buff_H1+H0),
		(__m128 *)(buff_H1+H4),
		(__m128 *)(buff_H1+H8)
	);

	_MM_TRANSPOSE1_PS(in);

	// vertically
	diag_horizontally_2x2_FAST_MACRO(
		in,
		(__m128 *)(buff_V0+V0),
		(__m128 *)(buff_V0+V4),
		(__m128 *)(buff_V0+V8),
		(__m128 *)(buff_V1+V0),
		(__m128 *)(buff_V1+V4),
		(__m128 *)(buff_V1+V8)
	);

	op4s_sdl2_scale_s_sse(in, v);

	*out = in;
#undef buff_H0
#undef buff_H1
#undef buff_V0
#undef buff_V1
#undef in
#undef out
#undef H0
#undef H4
#undef H8
#undef V0
#undef V4
#undef V8
	// done
#endif
}
#endif /* __SSE__ */

/*
	buff_h0 + y0*buff_elem_size,
	buff_h0 + y1*buff_elem_size,
	buff_v0 + x0*buff_elem_size,
	buff_v0 + x1*buff_elem_size
 */
static
void fdwt_cdf97_diag_cor2x2_sse_s(
	float *ptrL0, float *ptrL1, // y=0
	float *ptrR0, float *ptrR1, // y=1
	float *outL0, float *outR0, // y=0
	float *outL1, float *outR1, // y=1
	float *lAL, // h0
	float *lAR, // h1
	float *lBL, // v0
	float *lBR  // v1
)
{
#ifdef __SSE__
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v_vert = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f, 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s) };

	__m128 buff;
	__m128 z;

	// [ y0x0 y0x1 y1x0 y1x1 ]
	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	// A/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lAL+4), *(__m128 *)(lAL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lAR+4), *(__m128 *)(lAR+8));

	// A/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAL+4), w, *(__m128 *)(lAL+0), *(__m128 *)(lAL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lAL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAL+4), *(__m128 *)(lAL+0), *(__m128 *)(lAL+8), z);

	// A/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAR+4), w, *(__m128 *)(lAR+0), *(__m128 *)(lAR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lAR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAR+4), *(__m128 *)(lAR+0), *(__m128 *)(lAR+8), z);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lBL+4), *(__m128 *)(lBL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lBR+4), *(__m128 *)(lBR+8));

	// B/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBL+4), w, *(__m128 *)(lBL+0), *(__m128 *)(lBL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lBL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lBL+4), *(__m128 *)(lBL+0), *(__m128 *)(lBL+8), z);

	// B/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBR+4), w, *(__m128 *)(lBR+0), *(__m128 *)(lBR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lBR+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(lBR+4), *(__m128 *)(lBR+0), *(__m128 *)(lBR+8), z);

	// B/L+R
	op4s_sdl2_scale_s_sse(buff, v_vert);

	// [ y0x0 y1x0 y0x1 y1x1 ]
	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
#endif /* __SSE__ */
}

static
void fdwt_cdf97_diag_cor6x2_sse_s(
	intptr_t ptr_y0_x0,
	intptr_t out_y0_x0,
	ptrdiff_t stride_x, // +1 row
	ptrdiff_t stride_y, // +1 col
	float *buff_h0, // +(0..?)*(3*4) [ y down> ]
	float *buff_v0  // +(0..?)*(3*4) [ x right> ]
)
{
#ifdef __SSE__
#if 0
	const int buff_elem_size = 3*4; // diagonal core

	// 2x2 y=0..1 x=0..1
	{
		const int y0 = 0;
		const int x0 = 0;

		const int y1 = y0+1;
		const int x1 = x0+1;
	
		fdwt_cdf97_diag_cor2x2_sse_s(
			// input pointers
			/* y=0 x=0 */ ptr_y0_x0 + y0*stride_x + x0*stride_y,
			/* y=0 x=1 */ ptr_y0_x0 + y0*stride_x + x1*stride_y,
			/* y=1 x=0 */ ptr_y0_x0 + y1*stride_x + x0*stride_y,
			/* y=1 x=1 */ ptr_y0_x0 + y1*stride_x + x1*stride_y,
			// output pointers
			/* y=0 x=0 */ out_y0_x0 + y0*stride_x + x0*stride_y,
			/* y=0 x=1 */ out_y0_x0 + y0*stride_x + x1*stride_y,
			/* y=1 x=0 */ out_y0_x0 + y1*stride_x + x0*stride_y,
			/* y=1 x=1 */ out_y0_x0 + y1*stride_x + x1*stride_y,
			// buffers
			buff_h0 + y0*buff_elem_size,
			buff_h0 + y1*buff_elem_size,
			buff_v0 + x0*buff_elem_size,
			buff_v0 + x1*buff_elem_size
		);
	}

	// 2x2 y=0..1 x=2..3
	{
		const int y0 = 0;
		const int x0 = 2;

		const int y1 = y0+1;
		const int x1 = x0+1;
	
		fdwt_cdf97_diag_cor2x2_sse_s(
			// input pointers
			/* y=0 x=0 */ ptr_y0_x0 + y0*stride_x + x0*stride_y,
			/* y=0 x=1 */ ptr_y0_x0 + y0*stride_x + x1*stride_y,
			/* y=1 x=0 */ ptr_y0_x0 + y1*stride_x + x0*stride_y,
			/* y=1 x=1 */ ptr_y0_x0 + y1*stride_x + x1*stride_y,
			// output pointers
			/* y=0 x=0 */ out_y0_x0 + y0*stride_x + x0*stride_y,
			/* y=0 x=1 */ out_y0_x0 + y0*stride_x + x1*stride_y,
			/* y=1 x=0 */ out_y0_x0 + y1*stride_x + x0*stride_y,
			/* y=1 x=1 */ out_y0_x0 + y1*stride_x + x1*stride_y,
			// buffers
			buff_h0 + y0*buff_elem_size,
			buff_h0 + y1*buff_elem_size,
			buff_v0 + x0*buff_elem_size,
			buff_v0 + x1*buff_elem_size
		);
	}

	// 2x2 y=0..1 x=4..5
	{
		const int y0 = 0;
		const int x0 = 4;

		const int y1 = y0+1;
		const int x1 = x0+1;
	
		fdwt_cdf97_diag_cor2x2_sse_s(
			// input pointers
			/* y=0 x=0 */ ptr_y0_x0 + y0*stride_x + x0*stride_y,
			/* y=0 x=1 */ ptr_y0_x0 + y0*stride_x + x1*stride_y,
			/* y=1 x=0 */ ptr_y0_x0 + y1*stride_x + x0*stride_y,
			/* y=1 x=1 */ ptr_y0_x0 + y1*stride_x + x1*stride_y,
			// output pointers
			/* y=0 x=0 */ out_y0_x0 + y0*stride_x + x0*stride_y,
			/* y=0 x=1 */ out_y0_x0 + y0*stride_x + x1*stride_y,
			/* y=1 x=0 */ out_y0_x0 + y1*stride_x + x0*stride_y,
			/* y=1 x=1 */ out_y0_x0 + y1*stride_x + x1*stride_y,
			// buffers
			buff_h0 + y0*buff_elem_size,
			buff_h0 + y1*buff_elem_size,
			buff_v0 + x0*buff_elem_size,
			buff_v0 + x1*buff_elem_size
		);
	}
#endif
#if 1
	__m128 t0, t1, t2;

	t0 = (__m128){
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 1*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 1*stride_y)
	};
	t1 = (__m128){
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 2*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 3*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 2*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 3*stride_y)
	};
	t2 = (__m128){
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 4*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 5*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 4*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 5*stride_y)
	};

	diag_6x2(
		// input by 2x2 blocks in format [ y0x0 y0x1 y1x0 y1x1 ]
		t0,
		t1,
		t2,
		// output by 2x2 blocks in format [ y0x0 y1x0 y0x1 y1x1 ]
		&t0,
		&t1,
		&t2,
		// buffers with stride = (3*4)*sizeof(float)
		buff_h0, // +(0..1)*(3*4) [ y down> ]
		buff_v0  // +(0..5)*(3*4) [ x right> ]
	);

	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 0*stride_y) = t0[0];
	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 1*stride_y) = t0[2];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 0*stride_y) = t0[1];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 1*stride_y) = t0[3];

	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 2*stride_y) = t1[0];
	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 3*stride_y) = t1[2];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 2*stride_y) = t1[1];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 3*stride_y) = t1[3];

	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 4*stride_y) = t2[0];
	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 5*stride_y) = t2[2];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 4*stride_y) = t2[1];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 5*stride_y) = t2[3];
#endif
#endif /* __SSE__ */
}

static
void fdwt_cdf97_diag_cor2x6_sse_s(
	intptr_t ptr_y0_x0,
	intptr_t out_y0_x0,
	ptrdiff_t stride_x, // +1 row
	ptrdiff_t stride_y, // +1 col
	float *buff_h0, // +(0..?)*(3*4) [ y down> ]
	float *buff_v0  // +(0..?)*(3*4) [ x right> ]
)
{
#ifdef __SSE__
#if 1
	__m128 t0, t1, t2;

	t0 = (__m128){
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 1*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 1*stride_y)
	};
	t1 = (__m128){
		/* y= x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 1*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 1*stride_y)
	};
	t2 = (__m128){
		/* y= x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 1*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 0*stride_y),
		/* y= x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 1*stride_y)
	};

	diag_2x6(
		// input by 2x2 blocks in format [ y0x0 y0x1 y1x0 y1x1 ]
		t0,
		t1,
		t2,
		// output by 2x2 blocks in format [ y0x0 y1x0 y0x1 y1x1 ]
		&t0,
		&t1,
		&t2,
		// buffers with stride = (3*4)*sizeof(float)
		buff_h0, // +(0..5)*(3*4) [ y down> ]
		buff_v0  // +(0..1)*(3*4) [ x right> ]
	);

	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 0*stride_y) = t0[0];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 0*stride_y) = t0[1];
	/* y= x= */ *(float *)(out_y0_x0 + 0*stride_x + 1*stride_y) = t0[2];
	/* y= x= */ *(float *)(out_y0_x0 + 1*stride_x + 1*stride_y) = t0[3];

	/* y= x= */ *(float *)(out_y0_x0 + 2*stride_x + 0*stride_y) = t1[0];
	/* y= x= */ *(float *)(out_y0_x0 + 3*stride_x + 0*stride_y) = t1[1];
	/* y= x= */ *(float *)(out_y0_x0 + 2*stride_x + 1*stride_y) = t1[2];
	/* y= x= */ *(float *)(out_y0_x0 + 3*stride_x + 1*stride_y) = t1[3];

	/* y= x= */ *(float *)(out_y0_x0 + 4*stride_x + 0*stride_y) = t2[0];
	/* y= x= */ *(float *)(out_y0_x0 + 5*stride_x + 0*stride_y) = t2[1];
	/* y= x= */ *(float *)(out_y0_x0 + 4*stride_x + 1*stride_y) = t2[2];
	/* y= x= */ *(float *)(out_y0_x0 + 5*stride_x + 1*stride_y) = t2[3];
#endif
#endif /* __SSE__ */
}

static
void fdwt_cdf97_diag_cor6x6_sse_s(
	intptr_t ptr_y0_x0,
	intptr_t out_y0_x0,
	ptrdiff_t stride_x, // +1 row
	ptrdiff_t stride_y, // +1 col
	float *buff_h0, // +(0..?)*(3*4) [ y down> ]
	float *buff_v0  // +(0..?)*(3*4) [ x right> ]
)
{
#ifdef __SSE__
#if 1
	__m128 t00, t01, t02;
	__m128 t10, t11, t12;
	__m128 t20, t21, t22;

	// x=0..1
	t00 = (__m128){
		/* y=0 x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 0*stride_y),
		/* y=0 x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 1*stride_y),
		/* y=1 x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 0*stride_y),
		/* y=1 x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 1*stride_y)
	};
	t10 = (__m128){
		/* y=2 x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 0*stride_y),
		/* y=2 x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 1*stride_y),
		/* y=3 x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 0*stride_y),
		/* y=3 x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 1*stride_y)
	};
	t20 = (__m128){
		/* y=4 x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 0*stride_y),
		/* y=4 x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 1*stride_y),
		/* y=5 x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 0*stride_y),
		/* y=5 x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 1*stride_y)
	};
	// x=2..3
	t01 = (__m128){
		/* y=0 x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 2*stride_y),
		/* y=0 x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 3*stride_y),
		/* y=1 x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 2*stride_y),
		/* y=1 x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 3*stride_y)
	};
	t11 = (__m128){
		/* y=2 x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 2*stride_y),
		/* y=2 x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 3*stride_y),
		/* y=3 x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 2*stride_y),
		/* y=3 x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 3*stride_y)
	};
	t21 = (__m128){
		/* y=4 x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 2*stride_y),
		/* y=4 x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 3*stride_y),
		/* y=5 x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 2*stride_y),
		/* y=5 x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 3*stride_y)
	};
	// x=4..5
	t02 = (__m128){
		/* y=0 x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 4*stride_y),
		/* y=0 x= */ *(float *)(ptr_y0_x0 + 0*stride_x + 5*stride_y),
		/* y=1 x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 4*stride_y),
		/* y=1 x= */ *(float *)(ptr_y0_x0 + 1*stride_x + 5*stride_y)
	};
	t12 = (__m128){
		/* y=2 x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 4*stride_y),
		/* y=2 x= */ *(float *)(ptr_y0_x0 + 2*stride_x + 5*stride_y),
		/* y=3 x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 4*stride_y),
		/* y=3 x= */ *(float *)(ptr_y0_x0 + 3*stride_x + 5*stride_y)
	};
	t22 = (__m128){
		/* y=4 x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 4*stride_y),
		/* y=4 x= */ *(float *)(ptr_y0_x0 + 4*stride_x + 5*stride_y),
		/* y=5 x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 4*stride_y),
		/* y=5 x= */ *(float *)(ptr_y0_x0 + 5*stride_x + 5*stride_y)
	};

	_MM_TRANSPOSE1_PS(t00);
	_MM_TRANSPOSE1_PS(t01);
	_MM_TRANSPOSE1_PS(t02);
	_MM_TRANSPOSE1_PS(t10);
	_MM_TRANSPOSE1_PS(t11);
	_MM_TRANSPOSE1_PS(t12);
	_MM_TRANSPOSE1_PS(t20);
	_MM_TRANSPOSE1_PS(t21);
	_MM_TRANSPOSE1_PS(t22);

	diag_6x6(
#if 1
		// input by 2x2 blocks in format [ y0x0 y0x1 y1x0 y1x1 ]
		t00, t01, t02,
		t10, t11, t12,
		t20, t21, t22,
		// output by 2x2 blocks in format [ y0x0 y1x0 y0x1 y1x1 ]
		&t00, &t01, &t02,
		&t10, &t11, &t12,
		&t20, &t21, &t22,
#else
		t00, t10, t20,
		t01, t11, t21,
		t02, t12, t22,
		&t00, &t10, &t20,
		&t01, &t11, &t21,
		&t02, &t12, &t22,
#endif
		// buffers with stride = (3*4)*sizeof(float)
		buff_v0, // h[] +(0..5)*(3*4) [ y down> ]
		buff_h0  // v[] +(0..5)*(3*4) [ x right> ]
	);

	_MM_TRANSPOSE1_PS(t00);
	_MM_TRANSPOSE1_PS(t01);
	_MM_TRANSPOSE1_PS(t02);
	_MM_TRANSPOSE1_PS(t10);
	_MM_TRANSPOSE1_PS(t11);
	_MM_TRANSPOSE1_PS(t12);
	_MM_TRANSPOSE1_PS(t20);
	_MM_TRANSPOSE1_PS(t21);
	_MM_TRANSPOSE1_PS(t22);

	// x = 0..1
	/* y=0 x= */ *(float *)(out_y0_x0 + 0*stride_x + 0*stride_y) = t00[0];
	/* y=1 x= */ *(float *)(out_y0_x0 + 1*stride_x + 0*stride_y) = t00[1];
	/* y=0 x= */ *(float *)(out_y0_x0 + 0*stride_x + 1*stride_y) = t00[2];
	/* y=1 x= */ *(float *)(out_y0_x0 + 1*stride_x + 1*stride_y) = t00[3];
	/* y=2 x= */ *(float *)(out_y0_x0 + 2*stride_x + 0*stride_y) = t10[0];
	/* y=3 x= */ *(float *)(out_y0_x0 + 3*stride_x + 0*stride_y) = t10[1];
	/* y=2 x= */ *(float *)(out_y0_x0 + 2*stride_x + 1*stride_y) = t10[2];
	/* y=3 x= */ *(float *)(out_y0_x0 + 3*stride_x + 1*stride_y) = t10[3];
	/* y=4 x= */ *(float *)(out_y0_x0 + 4*stride_x + 0*stride_y) = t20[0];
	/* y=5 x= */ *(float *)(out_y0_x0 + 5*stride_x + 0*stride_y) = t20[1];
	/* y=4 x= */ *(float *)(out_y0_x0 + 4*stride_x + 1*stride_y) = t20[2];
	/* y=5 x= */ *(float *)(out_y0_x0 + 5*stride_x + 1*stride_y) = t20[3];
	// x = 2..3
	/* y=0 x= */ *(float *)(out_y0_x0 + 0*stride_x + 2*stride_y) = t01[0];
	/* y=1 x= */ *(float *)(out_y0_x0 + 1*stride_x + 2*stride_y) = t01[1];
	/* y=0 x= */ *(float *)(out_y0_x0 + 0*stride_x + 3*stride_y) = t01[2];
	/* y=1 x= */ *(float *)(out_y0_x0 + 1*stride_x + 3*stride_y) = t01[3];
	/* y=2 x= */ *(float *)(out_y0_x0 + 2*stride_x + 2*stride_y) = t11[0];
	/* y=3 x= */ *(float *)(out_y0_x0 + 3*stride_x + 2*stride_y) = t11[1];
	/* y=2 x= */ *(float *)(out_y0_x0 + 2*stride_x + 3*stride_y) = t11[2];
	/* y=3 x= */ *(float *)(out_y0_x0 + 3*stride_x + 3*stride_y) = t11[3];
	/* y=4 x= */ *(float *)(out_y0_x0 + 4*stride_x + 2*stride_y) = t21[0];
	/* y=5 x= */ *(float *)(out_y0_x0 + 5*stride_x + 2*stride_y) = t21[1];
	/* y=4 x= */ *(float *)(out_y0_x0 + 4*stride_x + 3*stride_y) = t21[2];
	/* y=5 x= */ *(float *)(out_y0_x0 + 5*stride_x + 3*stride_y) = t21[3];
	// x = 4..5
	/* y=0 x= */ *(float *)(out_y0_x0 + 0*stride_x + 4*stride_y) = t02[0];
	/* y=1 x= */ *(float *)(out_y0_x0 + 1*stride_x + 4*stride_y) = t02[1];
	/* y=0 x= */ *(float *)(out_y0_x0 + 0*stride_x + 5*stride_y) = t02[2];
	/* y=1 x= */ *(float *)(out_y0_x0 + 1*stride_x + 5*stride_y) = t02[3];
	/* y=2 x= */ *(float *)(out_y0_x0 + 2*stride_x + 4*stride_y) = t12[0];
	/* y=3 x= */ *(float *)(out_y0_x0 + 3*stride_x + 4*stride_y) = t12[1];
	/* y=2 x= */ *(float *)(out_y0_x0 + 2*stride_x + 5*stride_y) = t12[2];
	/* y=3 x= */ *(float *)(out_y0_x0 + 3*stride_x + 5*stride_y) = t12[3];
	/* y=4 x= */ *(float *)(out_y0_x0 + 4*stride_x + 4*stride_y) = t22[0];
	/* y=5 x= */ *(float *)(out_y0_x0 + 5*stride_x + 4*stride_y) = t22[1];
	/* y=4 x= */ *(float *)(out_y0_x0 + 4*stride_x + 5*stride_y) = t22[2];
	/* y=5 x= */ *(float *)(out_y0_x0 + 5*stride_x + 5*stride_y) = t22[3];
#endif
#endif /* __SSE__ */
}

static
void fdwt_cdf97_vert_cor2x2_sse_s(
	float *ptr_y0_x0, // in
	float *ptr_y0_x1, // in
	float *ptr_y1_x0, // in
	float *ptr_y1_x1, // in
	float *out_y0_x0, // out
	float *out_y0_x1, // out
	float *out_y1_x0, // out
	float *out_y1_x1, // out
	float *buff_h0, // [4]
	float *buff_h1, // [4]
	float *buff_v0, // [4]
	float *buff_v1  // [4]
)
{
#ifdef __SSE__
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	const __m128 v_vertL = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f,
		0.f, 0.f };
	const __m128 v_vertR = { 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s),
		0.f, 0.f };

	// temp
	__m128 t;

	// aux. variables
	__m128 x, y, r, c;

	// horiz 1
	{
		float *l = buff_h0;

		// inputs
		x[0] = *ptr_y0_x0;
		x[1] = *ptr_y0_x1;

		// shuffles
		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		// outputs
		t[0] = y[0];
		t[1] = y[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	// horiz 2
	{
		float *l = buff_h1;

		// inputs
		x[0] = *ptr_y1_x0;
		x[1] = *ptr_y1_x1;

		// shuffles
		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		// outputs
		t[2] = y[0];
		t[3] = y[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	// vert 1
	{
		float *l = buff_v0;

		// inputs
		x[0] = t[0];
		x[1] = t[2];

		// shuffles
		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		// scaling
		y[0] *= v_vertL[0];
		y[1] *= v_vertL[1];

		// outputs
		*out_y0_x0 = y[0];
		*out_y1_x0 = y[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	// vert 2
	{
		float *l = buff_v1;

		// inputs
		x[0] = t[1];
		x[1] = t[3];

		// shuffles
		y[2] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[3] = c[0]+w[0]*(l[0]+r[0]);

		// scaling
		y[2] *= v_vertR[0];
		y[3] *= v_vertR[1];

		// outputs
		*out_y0_x1 = y[2];
		*out_y1_x1 = y[3];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}
#endif /* __SSE__ */
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_vert_2x2_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // vert
	const int buff_elem_size = 1*4; // vert

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x,       stride_x, stride_y);
	char *ptr_y1_x0 = (void *)addr2_s(ptr, base_y+1,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x-shift, stride_x, stride_y);
	char *out_y1_x0 = (void *)addr2_s(ptr, base_y+1-shift, base_x-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y0_x1 = (ptrdiff_t)addr1_s(0, +1, stride_y); // +1 rows

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *ptr_y1_x0_i = ptr_y1_x0;
		char *out_y0_x0_i = out_y0_x0;
		char *out_y1_x0_i = out_y1_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_vert_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i,
				(void *)ptr_y0_x0_i + diff_y0_x1,
				(void *)ptr_y1_x0_i,
				(void *)ptr_y1_x0_i + diff_y0_x1,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i,
				(void *)out_y0_x0_i + diff_y0_x1,
				(void *)out_y1_x0_i,
				(void *)out_y1_x0_i + diff_y0_x1,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_x2;
			ptr_y1_x0_i += diff_x2;
			out_y0_x0_i += diff_x2;
			out_y1_x0_i += diff_x2;

			buffer_x0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y2;
		ptr_y1_x0 += diff_y2;
		out_y0_x0 += diff_y2;
		out_y1_x0 += diff_y2;

		buffer_y0_i += 2*(buff_elem_size);
	}
}

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
#endif /* __SSE__ */

// vert 4x4 core
static
void fdwt_cdf97_vert_cor4x4_sse_s(
	intptr_t ptr_y0_x0, // pointer to (0,0)
	intptr_t out_y0_x0, // pointer to (0-shift,0-shift)
	ptrdiff_t stride_x, // +1 row
	ptrdiff_t stride_y, // +1 col
	float *buff_h0, // +(0..3)*(1*4) [ y down> ]
	float *buff_v0  // +(0..3)*(1*4) [ x right> ]
)
{
#ifdef __SSE__
#if 1
	// this 4x4 core approach corresponds to "transpose-SIMD" in Figure 9 in Kutil2006 (the "line-SIMD" should be 8x2 core)
	__m128 t0, t1, t2, t3;

	// load 4x4
	t0 = (__m128){
		*(float *)(ptr_y0_x0 + 0*stride_x + 0*stride_y),
		*(float *)(ptr_y0_x0 + 1*stride_x + 0*stride_y),
		*(float *)(ptr_y0_x0 + 2*stride_x + 0*stride_y),
		*(float *)(ptr_y0_x0 + 3*stride_x + 0*stride_y)
	};
	t1 = (__m128){
		*(float *)(ptr_y0_x0 + 0*stride_x + 1*stride_y),
		*(float *)(ptr_y0_x0 + 1*stride_x + 1*stride_y),
		*(float *)(ptr_y0_x0 + 2*stride_x + 1*stride_y),
		*(float *)(ptr_y0_x0 + 3*stride_x + 1*stride_y)
	};
	t2 = (__m128){
		*(float *)(ptr_y0_x0 + 0*stride_x + 2*stride_y),
		*(float *)(ptr_y0_x0 + 1*stride_x + 2*stride_y),
		*(float *)(ptr_y0_x0 + 2*stride_x + 2*stride_y),
		*(float *)(ptr_y0_x0 + 3*stride_x + 2*stride_y)
	};
	t3 = (__m128){
		*(float *)(ptr_y0_x0 + 0*stride_x + 3*stride_y),
		*(float *)(ptr_y0_x0 + 1*stride_x + 3*stride_y),
		*(float *)(ptr_y0_x0 + 2*stride_x + 3*stride_y),
		*(float *)(ptr_y0_x0 + 3*stride_x + 3*stride_y)
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

#if 0
	// TODO: this transposition can be discarded
	_MM_TRANSPOSE4_PS(t0, t1, t2, t3);

	// scaling 4x4
	const float z = dwt_cdf97_s1_s;
	t0 *= (__m128){ 1/(z*z),   1.f, 1/(z*z),   1.f };
	t1 *= (__m128){     1.f, (z*z),     1.f, (z*z) };
	t2 *= (__m128){ 1/(z*z),   1.f, 1/(z*z),   1.f };
	t3 *= (__m128){     1.f, (z*z),     1.f, (z*z) };

	// store 4x4
	//(__m128){ out_y0_x0 + 0*stride_x + 0*stride_y, out_y0_x0 + 1*stride_x + 0*stride_y, out_y0_x0 + 2*stride_x + 0*stride_y, out_y0_x0 + 3*stride_x + 0*stride_y } = t0;
	*(float *)(out_y0_x0 + 0*stride_x + 0*stride_y) = t0[0];
	*(float *)(out_y0_x0 + 1*stride_x + 0*stride_y) = t0[1];
	*(float *)(out_y0_x0 + 2*stride_x + 0*stride_y) = t0[2];
	*(float *)(out_y0_x0 + 3*stride_x + 0*stride_y) = t0[3];
	//(__m128){ out_y0_x0 + 0*stride_x + 1*stride_y, out_y0_x0 + 1*stride_x + 1*stride_y, out_y0_x0 + 2*stride_x + 1*stride_y, out_y0_x0 + 3*stride_x + 1*stride_y } = t1;
	*(float *)(out_y0_x0 + 0*stride_x + 1*stride_y) = t1[0];
	*(float *)(out_y0_x0 + 1*stride_x + 1*stride_y) = t1[1];
	*(float *)(out_y0_x0 + 2*stride_x + 1*stride_y) = t1[2];
	*(float *)(out_y0_x0 + 3*stride_x + 1*stride_y) = t1[3];
	//(__m128){ out_y0_x0 + 0*stride_x + 2*stride_y, out_y0_x0 + 1*stride_x + 2*stride_y, out_y0_x0 + 2*stride_x + 2*stride_y, out_y0_x0 + 3*stride_x + 2*stride_y } = t2;
	*(float *)(out_y0_x0 + 0*stride_x + 2*stride_y) = t2[0];
	*(float *)(out_y0_x0 + 1*stride_x + 2*stride_y) = t2[1];
	*(float *)(out_y0_x0 + 2*stride_x + 2*stride_y) = t2[2];
	*(float *)(out_y0_x0 + 3*stride_x + 2*stride_y) = t2[3];
	//(__m128){ out_y0_x0 + 0*stride_x + 3*stride_y, out_y0_x0 + 1*stride_x + 3*stride_y, out_y0_x0 + 2*stride_x + 3*stride_y, out_y0_x0 + 3*stride_x + 3*stride_y } = t3;
	*(float *)(out_y0_x0 + 0*stride_x + 3*stride_y) = t3[0];
	*(float *)(out_y0_x0 + 1*stride_x + 3*stride_y) = t3[1];
	*(float *)(out_y0_x0 + 2*stride_x + 3*stride_y) = t3[2];
	*(float *)(out_y0_x0 + 3*stride_x + 3*stride_y) = t3[3];
#else
	const float z = dwt_cdf97_s1_s;
	t0 *= (const __m128){ 1/(z*z),   1.f, 1/(z*z),   1.f };
	t1 *= (const __m128){     1.f, (z*z),     1.f, (z*z) };
	t2 *= (const __m128){ 1/(z*z),   1.f, 1/(z*z),   1.f };
	t3 *= (const __m128){     1.f, (z*z),     1.f, (z*z) };

	*(float *)(out_y0_x0 + 0*stride_x + 0*stride_y) = t0[0];
	*(float *)(out_y0_x0 + 1*stride_x + 0*stride_y) = t1[0];
	*(float *)(out_y0_x0 + 2*stride_x + 0*stride_y) = t2[0];
	*(float *)(out_y0_x0 + 3*stride_x + 0*stride_y) = t3[0];

	*(float *)(out_y0_x0 + 0*stride_x + 1*stride_y) = t0[1];
	*(float *)(out_y0_x0 + 1*stride_x + 1*stride_y) = t1[1];
	*(float *)(out_y0_x0 + 2*stride_x + 1*stride_y) = t2[1];
	*(float *)(out_y0_x0 + 3*stride_x + 1*stride_y) = t3[1];

	*(float *)(out_y0_x0 + 0*stride_x + 2*stride_y) = t0[2];
	*(float *)(out_y0_x0 + 1*stride_x + 2*stride_y) = t1[2];
	*(float *)(out_y0_x0 + 2*stride_x + 2*stride_y) = t2[2];
	*(float *)(out_y0_x0 + 3*stride_x + 2*stride_y) = t3[2];

	*(float *)(out_y0_x0 + 0*stride_x + 3*stride_y) = t0[3];
	*(float *)(out_y0_x0 + 1*stride_x + 3*stride_y) = t1[3];
	*(float *)(out_y0_x0 + 2*stride_x + 3*stride_y) = t2[3];
	*(float *)(out_y0_x0 + 3*stride_x + 3*stride_y) = t3[3];
#endif

#endif
#if 0
	const int buff_elem_size = 1*4; // diagonal core

	// 2x2 (y=0..1,x=0..1)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=0 */(char *)ptr_y0_x0 + 0*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)ptr_y0_x0 + 0*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)ptr_y0_x0 + 1*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)ptr_y0_x0 + 1*stride_x + 1*stride_y,
		// outputs
		/* y=0 x=0 */(char *)out_y0_x0 + 0*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)out_y0_x0 + 0*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)out_y0_x0 + 1*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)out_y0_x0 + 1*stride_x + 1*stride_y,
		// buffers
		buff_h0 + 0*buff_elem_size,
		buff_h0 + 1*buff_elem_size,
		buff_v0 + 0*buff_elem_size,
		buff_v0 + 1*buff_elem_size
	);

	// 2x2 (y=0..1,x=2..3)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=2 */(char *)ptr_y0_x0 + 0*stride_x + 2*stride_y,
		/* y=0 x=3 */(char *)ptr_y0_x0 + 0*stride_x + 3*stride_y,
		/* y=1 x=2 */(char *)ptr_y0_x0 + 1*stride_x + 2*stride_y,
		/* y=1 x=3 */(char *)ptr_y0_x0 + 1*stride_x + 3*stride_y,
		// outputs
		/* y=0 x=2 */(char *)out_y0_x0 + 0*stride_x + 2*stride_y,
		/* y=0 x=3 */(char *)out_y0_x0 + 0*stride_x + 3*stride_y,
		/* y=1 x=2 */(char *)out_y0_x0 + 1*stride_x + 2*stride_y,
		/* y=1 x=3 */(char *)out_y0_x0 + 1*stride_x + 3*stride_y,
		// buffers
		buff_h0 + 0*buff_elem_size,
		buff_h0 + 1*buff_elem_size,
		buff_v0 + 2*buff_elem_size,
		buff_v0 + 3*buff_elem_size
	);

	// 2x2 (y=2..3,x=0..1)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=0 */(char *)ptr_y0_x0 + 2*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)ptr_y0_x0 + 2*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)ptr_y0_x0 + 3*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)ptr_y0_x0 + 3*stride_x + 1*stride_y,
		// outputs
		/* y=0 x=0 */(char *)out_y0_x0 + 2*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)out_y0_x0 + 2*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)out_y0_x0 + 3*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)out_y0_x0 + 3*stride_x + 1*stride_y,
		// buffers
		buff_h0 + 2*buff_elem_size,
		buff_h0 + 3*buff_elem_size,
		buff_v0 + 0*buff_elem_size,
		buff_v0 + 1*buff_elem_size
	);

	// 2x2 (y=2..3,x=2..3)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=0 */(char *)ptr_y0_x0 + 2*stride_x + 2*stride_y,
		/* y=0 x=1 */(char *)ptr_y0_x0 + 2*stride_x + 3*stride_y,
		/* y=1 x=0 */(char *)ptr_y0_x0 + 3*stride_x + 2*stride_y,
		/* y=1 x=1 */(char *)ptr_y0_x0 + 3*stride_x + 3*stride_y,
		// outputs
		/* y=0 x=0 */(char *)out_y0_x0 + 2*stride_x + 2*stride_y,
		/* y=0 x=1 */(char *)out_y0_x0 + 2*stride_x + 3*stride_y,
		/* y=1 x=0 */(char *)out_y0_x0 + 3*stride_x + 2*stride_y,
		/* y=1 x=1 */(char *)out_y0_x0 + 3*stride_x + 3*stride_y,
		// buffers
		buff_h0 + 2*buff_elem_size,
		buff_h0 + 3*buff_elem_size,
		buff_v0 + 2*buff_elem_size,
		buff_v0 + 3*buff_elem_size
	);
#endif
#if 0
	const __m128 w = {
		+dwt_cdf97_u2_s,
		-dwt_cdf97_p2_s,
		+dwt_cdf97_u1_s,
		-dwt_cdf97_p1_s
	};

	const __m128 v_vertL = {
		1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f,
		0.f, 0.f };
	const __m128 v_vertR = {
		1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s),
		0.f, 0.f };

	const int buff_elem_size = 1*4; // diagonal core

	// 2x2 (y=0..1,x=0..1)
	{
		// inputs
		float *_ptr_y0_x0 = (char *)ptr_y0_x0 + 0*stride_x + 0*stride_y;
		float *_ptr_y0_x1 = (char *)ptr_y0_x0 + 0*stride_x + 1*stride_y;
		float *_ptr_y1_x0 = (char *)ptr_y0_x0 + 1*stride_x + 0*stride_y;
		float *_ptr_y1_x1 = (char *)ptr_y0_x0 + 1*stride_x + 1*stride_y;
		// output
		float *_out_y0_x0 = (char *)out_y0_x0 + 0*stride_x + 0*stride_y;
		float *_out_y0_x1 = (char *)out_y0_x0 + 0*stride_x + 1*stride_y;
		float *_out_y1_x0 = (char *)out_y0_x0 + 1*stride_x + 0*stride_y;
		float *_out_y1_x1 = (char *)out_y0_x0 + 1*stride_x + 1*stride_y;
		// buffers
		float *_buff_h0 = buff_h0 + 0*buff_elem_size;
		float *_buff_h1 = buff_h0 + 1*buff_elem_size;
		float *_buff_v0 = buff_v0 + 0*buff_elem_size;
		float *_buff_v1 = buff_v0 + 1*buff_elem_size;
		// variables
		float t[4];
		float x[2], y[2];
		float r[4], c[4];
		// horiz 1
		{
			float *l = _buff_h0;
			// inputs
			x[0] = *_ptr_y0_x0;
			x[1] = *_ptr_y0_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[0] = y[0];
			t[1] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// horiz 2
		{
			float *l = _buff_h1;
			// inputs
			x[0] = *_ptr_y1_x0;
			x[1] = *_ptr_y1_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[2] = y[0];
			t[3] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 1
		{
			float *l = _buff_v0;
			// inputs
			x[0] = t[0];
			x[1] = t[2];
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[0] *= v_vertL[0];
			y[1] *= v_vertL[1];
			// outputs
			*_out_y0_x0 = y[0];
			*_out_y1_x0 = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 2
		{
			float *l = _buff_v1;
			// inputs
			x[0] = t[1];
			x[1] = t[3];
			// shuffles
			y[2] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[3] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[2] *= v_vertR[0];
			y[3] *= v_vertR[1];
			// outputs
			*_out_y0_x1 = y[2];
			*_out_y1_x1 = y[3];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
	}

	// 2x2 (y=0..1,x=2..3)
	{
		// inputs
		float *_ptr_y0_x0 = (char *)ptr_y0_x0 + 0*stride_x + 2*stride_y;
		float *_ptr_y0_x1 = (char *)ptr_y0_x0 + 0*stride_x + 3*stride_y;
		float *_ptr_y1_x0 = (char *)ptr_y0_x0 + 1*stride_x + 2*stride_y;
		float *_ptr_y1_x1 = (char *)ptr_y0_x0 + 1*stride_x + 3*stride_y;
		// output
		float *_out_y0_x0 = (char *)out_y0_x0 + 0*stride_x + 2*stride_y;
		float *_out_y0_x1 = (char *)out_y0_x0 + 0*stride_x + 3*stride_y;
		float *_out_y1_x0 = (char *)out_y0_x0 + 1*stride_x + 2*stride_y;
		float *_out_y1_x1 = (char *)out_y0_x0 + 1*stride_x + 3*stride_y;
		// buffers
		float *_buff_h0 = buff_h0 + 0*buff_elem_size;
		float *_buff_h1 = buff_h0 + 1*buff_elem_size;
		float *_buff_v0 = buff_v0 + 2*buff_elem_size;
		float *_buff_v1 = buff_v0 + 3*buff_elem_size;
		// variables
		float t[4];
		float x[2], y[2];
		float r[4], c[4];
		// horiz 1
		{
			float *l = _buff_h0;
			// inputs
			x[0] = *_ptr_y0_x0;
			x[1] = *_ptr_y0_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[0] = y[0];
			t[1] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// horiz 2
		{
			float *l = _buff_h1;
			// inputs
			x[0] = *_ptr_y1_x0;
			x[1] = *_ptr_y1_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[2] = y[0];
			t[3] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 1
		{
			float *l = _buff_v0;
			// inputs
			x[0] = t[0];
			x[1] = t[2];
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[0] *= v_vertL[0];
			y[1] *= v_vertL[1];
			// outputs
			*_out_y0_x0 = y[0];
			*_out_y1_x0 = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 2
		{
			float *l = _buff_v1;
			// inputs
			x[0] = t[1];
			x[1] = t[3];
			// shuffles
			y[2] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[3] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[2] *= v_vertR[0];
			y[3] *= v_vertR[1];
			// outputs
			*_out_y0_x1 = y[2];
			*_out_y1_x1 = y[3];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
	}

	// 2x2 (y=2..3,x=0..1)
	{
		// inputs
		float *_ptr_y0_x0 = (char *)ptr_y0_x0 + 2*stride_x + 0*stride_y;
		float *_ptr_y0_x1 = (char *)ptr_y0_x0 + 2*stride_x + 1*stride_y;
		float *_ptr_y1_x0 = (char *)ptr_y0_x0 + 3*stride_x + 0*stride_y;
		float *_ptr_y1_x1 = (char *)ptr_y0_x0 + 3*stride_x + 1*stride_y;
		// output
		float *_out_y0_x0 = (char *)out_y0_x0 + 2*stride_x + 0*stride_y;
		float *_out_y0_x1 = (char *)out_y0_x0 + 2*stride_x + 1*stride_y;
		float *_out_y1_x0 = (char *)out_y0_x0 + 3*stride_x + 0*stride_y;
		float *_out_y1_x1 = (char *)out_y0_x0 + 3*stride_x + 1*stride_y;
		// buffers
		float *_buff_h0 = buff_h0 + 2*buff_elem_size;
		float *_buff_h1 = buff_h0 + 3*buff_elem_size;
		float *_buff_v0 = buff_v0 + 0*buff_elem_size;
		float *_buff_v1 = buff_v0 + 1*buff_elem_size;
		// variables
		float t[4];
		float x[2], y[2];
		float r[4], c[4];
		// horiz 1
		{
			float *l = _buff_h0;
			// inputs
			x[0] = *_ptr_y0_x0;
			x[1] = *_ptr_y0_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[0] = y[0];
			t[1] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// horiz 2
		{
			float *l = _buff_h1;
			// inputs
			x[0] = *_ptr_y1_x0;
			x[1] = *_ptr_y1_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[2] = y[0];
			t[3] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 1
		{
			float *l = _buff_v0;
			// inputs
			x[0] = t[0];
			x[1] = t[2];
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[0] *= v_vertL[0];
			y[1] *= v_vertL[1];
			// outputs
			*_out_y0_x0 = y[0];
			*_out_y1_x0 = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 2
		{
			float *l = _buff_v1;
			// inputs
			x[0] = t[1];
			x[1] = t[3];
			// shuffles
			y[2] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[3] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[2] *= v_vertR[0];
			y[3] *= v_vertR[1];
			// outputs
			*_out_y0_x1 = y[2];
			*_out_y1_x1 = y[3];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
	}

	// 2x2 (y=2..3,x=2..3)
	{
		// inputs
		float *_ptr_y0_x0 = (char *)ptr_y0_x0 + 2*stride_x + 2*stride_y;
		float *_ptr_y0_x1 = (char *)ptr_y0_x0 + 2*stride_x + 3*stride_y;
		float *_ptr_y1_x0 = (char *)ptr_y0_x0 + 3*stride_x + 2*stride_y;
		float *_ptr_y1_x1 = (char *)ptr_y0_x0 + 3*stride_x + 3*stride_y;
		// output
		float *_out_y0_x0 = (char *)out_y0_x0 + 2*stride_x + 2*stride_y;
		float *_out_y0_x1 = (char *)out_y0_x0 + 2*stride_x + 3*stride_y;
		float *_out_y1_x0 = (char *)out_y0_x0 + 3*stride_x + 2*stride_y;
		float *_out_y1_x1 = (char *)out_y0_x0 + 3*stride_x + 3*stride_y;
		// buffers
		float *_buff_h0 = buff_h0 + 2*buff_elem_size;
		float *_buff_h1 = buff_h0 + 3*buff_elem_size;
		float *_buff_v0 = buff_v0 + 2*buff_elem_size;
		float *_buff_v1 = buff_v0 + 3*buff_elem_size;
		// variables
		float t[4];
		float x[2], y[2];
		float r[4], c[4];
		// horiz 1
		{
			float *l = _buff_h0;
			// inputs
			x[0] = *_ptr_y0_x0;
			x[1] = *_ptr_y0_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[0] = y[0];
			t[1] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// horiz 2
		{
			float *l = _buff_h1;
			// inputs
			x[0] = *_ptr_y1_x0;
			x[1] = *_ptr_y1_x1;
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// outputs
			t[2] = y[0];
			t[3] = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 1
		{
			float *l = _buff_v0;
			// inputs
			x[0] = t[0];
			x[1] = t[2];
			// shuffles
			y[0] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[1] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[0] *= v_vertL[0];
			y[1] *= v_vertL[1];
			// outputs
			*_out_y0_x0 = y[0];
			*_out_y1_x0 = y[1];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
		// vert 2
		{
			float *l = _buff_v1;
			// inputs
			x[0] = t[1];
			x[1] = t[3];
			// shuffles
			y[2] = l[0];
			c[0] = l[1];
			c[1] = l[2];
			c[2] = l[3];
			c[3] = x[0];
			// operation
			r[3] = x[1];
			r[2] = c[3]+w[3]*(l[3]+r[3]);
			r[1] = c[2]+w[2]*(l[2]+r[2]);
			r[0] = c[1]+w[1]*(l[1]+r[1]);
			y[3] = c[0]+w[0]*(l[0]+r[0]);
			// scaling
			y[2] *= v_vertR[0];
			y[3] *= v_vertR[1];
			// outputs
			*_out_y0_x1 = y[2];
			*_out_y1_x1 = y[3];
			// update l[]
			l[0] = r[0];
			l[1] = r[1];
			l[2] = r[2];
			l[3] = r[3];
		}
	}
#endif
#endif /* __SSE__ */
}

#ifdef __SSE__
/*
L[0] = *ptr;
L[1] = R[0];
L[2] = R[1];
L[3] = R[2];
*ptr = R[3];
 */
static
__m128 shuff(__m128 r, float *ptr)
{
#if 0
	const float tmp = *ptr;

	// permutation
	r = (__m128){ r[3], r[0], r[1], r[2] };

	*ptr = r[0];
	r[0] = tmp;

	return r;
#endif
#if 0
	const float tmp = *ptr;

	// permutation
	r = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2,1,0,3));

	*ptr = r[0];
	r[0] = tmp;

	return r;
#endif
#if 1
	const float tmp = *ptr;

	*ptr = r[3];

	// permutation
	r = _mm_shuffle_ps(r, r, _MM_SHUFFLE(2,1,0,3));

	r[0] = tmp;

	return r;
#endif
}
#endif /* __SSE__ */

#ifdef __SSE__
static
void vert_8x1(
	// input even samples
	__m128 in0,
	// input odd samples
	__m128 in1,
	// output even samples
	__m128 *out0,
	// output odd samples
	__m128 *out1,
	// 1x buffer "L" of size (1*4) * sizeof(float)
	float *buff
)
{
	__m128 x0, x1;
	__m128 y0, y1;

	const __m128 w0 = { +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s };
	const __m128 w1 = { -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s };
	const __m128 w2 = { +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s };
	const __m128 w3 = { -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s };

	__m128 l0, l1, l2, l3;
	__m128 c0, c1, c2, c3;
	__m128 r0, r1, r2, r3;

	float *l = buff;

	// inputs
	x0 = in0;
	x1 = in1;
#if 1
	// 3 C-R <= X0 X1
	c3 = x0; // movaps
	r3 = x1; // movaps

	// 3 import + export
	l3 = shuff(r3, l+3);

	// 2 C-R
	c2 = l3; // movaps
	r2 = c3 + w3 * ( l3 + r3 ); // op

	// 2 import + export
	l2 = shuff(r2, l+2);

	// 1 C-R
	c1 = l2; // movaps
	r1 = c2 + w2 * ( l2 + r2 ); // op

	// 1 import + export
	l1 = shuff(r1, l+1);

	// 0 C-R
	c0 = l1; // movaps
	r0 = c1 + w1 * ( l1 + r1 ); // op

	// 0 import + export
	l0 = shuff(r0, l+0);

	// 1- C-R => Y0 Y1
	y0 = l0; // movaps
	y1 = c0 + w0 * ( l0 + r0 ); // op
#endif
#if 0
	// TODO TEST
	y1 = x1;
	c2 = shuff(y1, l+3);
	y1 += c2;
	y1 *= w3;
	y1 += x0;
	c1 = shuff(y1, l+2);
	y1 += c1;
	y1 *= w2;
	y1 += c2;
	c0 = shuff(y1, l+1);
	y1 += c0;
	y1 *= w1;
	y1 += c1;
	y0 = shuff(y1, l+0);
	y1 += y0;
	y1 *= w0;
	y1 += c0;
#endif
#if 0
	// TODO TEST
	__m128 L, R;

	L = shuff(x1, l+3);
	c2 = L;
	L += x1;
	L *= w3;
	L += x0;
	R = L;
	L = shuff(L, l+2);
	c1 = L;
	L += R;
	L *= w2;
	L += c2;
	R = L;
	L = shuff(R, l+1);
	c0 = L;
	L += R;
	L *= w1;
	L += c1;
	R = L;
	y1 = shuff(R, l+0);
	y0 = y1;
	y1 += R;
	y1 *= w0;
	y1 += c0;
#endif
	// outputs
	*out0 = y0;
	*out1 = y1;
}
#endif /* __SSE__ */

static
void fdwt_cdf97_vert_cor8x2_sse_s(
	intptr_t ptr_y0_x0, // pointer to (0,0)
	intptr_t out_y0_x0, // pointer to (0-shift,0-shift)
	ptrdiff_t stride_x, // +1 row
	ptrdiff_t stride_y, // +1 col
	float *buff_h0, // +(0..1)*(1*4) [ y down> ]
	float *buff_v0  // +(0..7)*(1*4) [ x right> ]
)
{
#ifdef __SSE__
#if 0
	const int buff_elem_size = 1*4; // vertical core

	// 2x2 (y=0..1,x=0..1)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=0 */(char *)ptr_y0_x0 + 0*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)ptr_y0_x0 + 0*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)ptr_y0_x0 + 1*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)ptr_y0_x0 + 1*stride_x + 1*stride_y,
		// outputs
		/* y=0 x=0 */(char *)out_y0_x0 + 0*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)out_y0_x0 + 0*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)out_y0_x0 + 1*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)out_y0_x0 + 1*stride_x + 1*stride_y,
		// buffers
		buff_h0 + 0*buff_elem_size,
		buff_h0 + 1*buff_elem_size,
		buff_v0 + 0*buff_elem_size,
		buff_v0 + 1*buff_elem_size
	);

	// 2x2 (y=0..1,x=2..3)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=2 */(char *)ptr_y0_x0 + 0*stride_x + 2*stride_y,
		/* y=0 x=3 */(char *)ptr_y0_x0 + 0*stride_x + 3*stride_y,
		/* y=1 x=2 */(char *)ptr_y0_x0 + 1*stride_x + 2*stride_y,
		/* y=1 x=3 */(char *)ptr_y0_x0 + 1*stride_x + 3*stride_y,
		// outputs
		/* y=0 x=2 */(char *)out_y0_x0 + 0*stride_x + 2*stride_y,
		/* y=0 x=3 */(char *)out_y0_x0 + 0*stride_x + 3*stride_y,
		/* y=1 x=2 */(char *)out_y0_x0 + 1*stride_x + 2*stride_y,
		/* y=1 x=3 */(char *)out_y0_x0 + 1*stride_x + 3*stride_y,
		// buffers
		buff_h0 + 0*buff_elem_size,
		buff_h0 + 1*buff_elem_size,
		buff_v0 + 2*buff_elem_size,
		buff_v0 + 3*buff_elem_size
	);

	// 2x2 (y=0..1,x=4..5)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=2 */(char *)ptr_y0_x0 + 0*stride_x + 4*stride_y,
		/* y=0 x=3 */(char *)ptr_y0_x0 + 0*stride_x + 5*stride_y,
		/* y=1 x=2 */(char *)ptr_y0_x0 + 1*stride_x + 4*stride_y,
		/* y=1 x=3 */(char *)ptr_y0_x0 + 1*stride_x + 5*stride_y,
		// outputs
		/* y=0 x=2 */(char *)out_y0_x0 + 0*stride_x + 4*stride_y,
		/* y=0 x=3 */(char *)out_y0_x0 + 0*stride_x + 5*stride_y,
		/* y=1 x=2 */(char *)out_y0_x0 + 1*stride_x + 4*stride_y,
		/* y=1 x=3 */(char *)out_y0_x0 + 1*stride_x + 5*stride_y,
		// buffers
		buff_h0 + 0*buff_elem_size,
		buff_h0 + 1*buff_elem_size,
		buff_v0 + 4*buff_elem_size,
		buff_v0 + 5*buff_elem_size
	);

	// 2x2 (y=0..1,x=6..7)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=2 */(char *)ptr_y0_x0 + 0*stride_x + 6*stride_y,
		/* y=0 x=3 */(char *)ptr_y0_x0 + 0*stride_x + 7*stride_y,
		/* y=1 x=2 */(char *)ptr_y0_x0 + 1*stride_x + 6*stride_y,
		/* y=1 x=3 */(char *)ptr_y0_x0 + 1*stride_x + 7*stride_y,
		// outputs
		/* y=0 x=2 */(char *)out_y0_x0 + 0*stride_x + 6*stride_y,
		/* y=0 x=3 */(char *)out_y0_x0 + 0*stride_x + 7*stride_y,
		/* y=1 x=2 */(char *)out_y0_x0 + 1*stride_x + 6*stride_y,
		/* y=1 x=3 */(char *)out_y0_x0 + 1*stride_x + 7*stride_y,
		// buffers
		buff_h0 + 0*buff_elem_size,
		buff_h0 + 1*buff_elem_size,
		buff_v0 + 6*buff_elem_size,
		buff_v0 + 7*buff_elem_size
	);
#endif
#if 1
	const int buff_elem_size = 1*4; // vertical core

	// load top row
	__m128 t0e, t0o;
	t0e[0] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/0*stride_y);
	t0e[1] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/2*stride_y);
	t0e[2] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/4*stride_y);
	t0e[3] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/6*stride_y);
	t0o[0] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/1*stride_y);
	t0o[1] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/3*stride_y);
	t0o[2] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/5*stride_y);
	t0o[3] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/7*stride_y);

	// load bottom row
	__m128 t1e, t1o;
	t1e[0] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/0*stride_y);
	t1e[1] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/2*stride_y);
	t1e[2] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/4*stride_y);
	t1e[3] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/6*stride_y);
	t1o[0] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/1*stride_y);
	t1o[1] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/3*stride_y);
	t1o[2] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/5*stride_y);
	t1o[3] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/7*stride_y);

	// horiz top 8x1
	vert_8x1(
		// input even samples
		t0e,
		// input odd samples
		t0o,
		// output even samples
		&t0e,
		// output odd samples
		&t0o,
		// 1x buffer "L" of size (1*4) * sizeof(float)
		buff_h0 + 0*buff_elem_size
	);

	// horiz bottom 8x1
	vert_8x1(
		// input even samples
		t1e,
		// input odd samples
		t1o,
		// output even samples
		&t1e,
		// output odd samples
		&t1o,
		// 1x buffer "L" of size (1*4) * sizeof(float)
		buff_h0 + 1*buff_elem_size
	);

	// vertical 8x2 in parallel on even, then on odd samples
	vert_2x4(
		// left input column [4]
		t0e,
		// right input column [4]
		t1e,
		// output 0 [4]
		&t0e,
		// output 1 [4]
		&t1e,
		// 4x buffer "L" with stride = (1*4) * sizeof(float)
		buff_v0 + 0*4*buff_elem_size
	);
	vert_2x4(
		// left input column [4]
		t0o,
		// right input column [4]
		t1o,
		// output 0 [4]
		&t0o,
		// output 1 [4]
		&t1o,
		// 4x buffer "L" with stride = (1*4) * sizeof(float)
		buff_v0 + 1*4*buff_elem_size
	);

	// scaling
	// [ t0e t0o t0e t0o t0e t0o t0e t0o ] *= [ 1/z^2   1 1/z^2   1 1/z^2   1 1/z^2   1 ]
	// [ t1e e1o t1e e1o t1e e1o t1e e1o ] *= [     1 z^2     1 z^2     1 z^2     1 z^2 ]
	// [ t0e t0e t0e t0e ] *= [ 1/z^2 1/z^2 1/z^2 1/z^2 ]
	// [ t0o t0o t0o t0o ] *= [ 1 1 1 1 ]
	// [ t1e t1e t1e t1e ] *= [ 1 1 1 1 ]
	// [ t1o t1o t1o t1o ] *= [ z^2 z^2 z^2 z^2 ];
	const float z = dwt_cdf97_s1_s;
	t0e *= (const __m128){ 1/(z*z), 1/(z*z), 1/(z*z), 1/(z*z) };
	t1o *= (const __m128){   (z*z),   (z*z),   (z*z),   (z*z) };

	// store top row
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/0*stride_y) = t0e[0];
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/2*stride_y) = t0e[1];
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/4*stride_y) = t0e[2];
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/6*stride_y) = t0e[3];
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/1*stride_y) = t0o[0];
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/3*stride_y) = t0o[1];
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/5*stride_y) = t0o[2];
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/7*stride_y) = t0o[3];

	// store bottom row
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/0*stride_y) = t1e[0];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/2*stride_y) = t1e[1];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/4*stride_y) = t1e[2];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/6*stride_y) = t1e[3];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/1*stride_y) = t1o[0];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/3*stride_y) = t1o[1];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/5*stride_y) = t1o[2];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/7*stride_y) = t1o[3];
#endif
#endif /* __SSE__ */
}

static
void fdwt_cdf97_vert_cor2x8_sse_s(
	intptr_t ptr_y0_x0, // pointer to (0,0)
	intptr_t out_y0_x0, // pointer to (0-shift,0-shift)
	ptrdiff_t stride_x, // +1 row
	ptrdiff_t stride_y, // +1 col
	float *buff_h0, // +(0..1)*(1*4) [ y down> ]
	float *buff_v0  // +(0..7)*(1*4) [ x right> ]
)
{
#ifdef __SSE__
#if 0
	const int buff_elem_size = 1*4; // vertical core

	// 2x2 (y=0..1,x=0..1)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=0 */(char *)ptr_y0_x0 + 0*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)ptr_y0_x0 + 0*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)ptr_y0_x0 + 1*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)ptr_y0_x0 + 1*stride_x + 1*stride_y,
		// outputs
		/* y=0 x=0 */(char *)out_y0_x0 + 0*stride_x + 0*stride_y,
		/* y=0 x=1 */(char *)out_y0_x0 + 0*stride_x + 1*stride_y,
		/* y=1 x=0 */(char *)out_y0_x0 + 1*stride_x + 0*stride_y,
		/* y=1 x=1 */(char *)out_y0_x0 + 1*stride_x + 1*stride_y,
		// buffers
		buff_h0 + 0*buff_elem_size,
		buff_h0 + 1*buff_elem_size,
		buff_v0 + 0*buff_elem_size,
		buff_v0 + 1*buff_elem_size
	);

	// 2x2 (y=2..3,x=0..1)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=2 */(char *)ptr_y0_x0 + 2*stride_x + 0*stride_y,
		/* y=0 x=3 */(char *)ptr_y0_x0 + 2*stride_x + 1*stride_y,
		/* y=1 x=2 */(char *)ptr_y0_x0 + 3*stride_x + 0*stride_y,
		/* y=1 x=3 */(char *)ptr_y0_x0 + 3*stride_x + 1*stride_y,
		// outputs
		/* y=0 x=2 */(char *)out_y0_x0 + 2*stride_x + 0*stride_y,
		/* y=0 x=3 */(char *)out_y0_x0 + 2*stride_x + 1*stride_y,
		/* y=1 x=2 */(char *)out_y0_x0 + 3*stride_x + 0*stride_y,
		/* y=1 x=3 */(char *)out_y0_x0 + 3*stride_x + 1*stride_y,
		// buffers
		buff_h0 + 2*buff_elem_size,
		buff_h0 + 3*buff_elem_size,
		buff_v0 + 0*buff_elem_size,
		buff_v0 + 1*buff_elem_size
	);

	// 2x2 (y=4..5,x=0..1)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=2 */(char *)ptr_y0_x0 + 4*stride_x + 0*stride_y,
		/* y=0 x=3 */(char *)ptr_y0_x0 + 4*stride_x + 1*stride_y,
		/* y=1 x=2 */(char *)ptr_y0_x0 + 5*stride_x + 0*stride_y,
		/* y=1 x=3 */(char *)ptr_y0_x0 + 5*stride_x + 1*stride_y,
		// outputs
		/* y=0 x=2 */(char *)out_y0_x0 + 4*stride_x + 0*stride_y,
		/* y=0 x=3 */(char *)out_y0_x0 + 4*stride_x + 1*stride_y,
		/* y=1 x=2 */(char *)out_y0_x0 + 5*stride_x + 0*stride_y,
		/* y=1 x=3 */(char *)out_y0_x0 + 5*stride_x + 1*stride_y,
		// buffers
		buff_h0 + 4*buff_elem_size,
		buff_h0 + 5*buff_elem_size,
		buff_v0 + 0*buff_elem_size,
		buff_v0 + 1*buff_elem_size
	);

	// 2x2 (y=6..7,x=0..1)
	fdwt_cdf97_vert_cor2x2_sse_s(
		// inputs
		/* y=0 x=2 */(char *)ptr_y0_x0 + 6*stride_x + 0*stride_y,
		/* y=0 x=3 */(char *)ptr_y0_x0 + 6*stride_x + 1*stride_y,
		/* y=1 x=2 */(char *)ptr_y0_x0 + 7*stride_x + 0*stride_y,
		/* y=1 x=3 */(char *)ptr_y0_x0 + 7*stride_x + 1*stride_y,
		// outputs
		/* y=0 x=2 */(char *)out_y0_x0 + 6*stride_x + 0*stride_y,
		/* y=0 x=3 */(char *)out_y0_x0 + 6*stride_x + 1*stride_y,
		/* y=1 x=2 */(char *)out_y0_x0 + 7*stride_x + 0*stride_y,
		/* y=1 x=3 */(char *)out_y0_x0 + 7*stride_x + 1*stride_y,
		// buffers
		buff_h0 + 6*buff_elem_size,
		buff_h0 + 7*buff_elem_size,
		buff_v0 + 0*buff_elem_size,
		buff_v0 + 1*buff_elem_size
	);
#endif
#if 1
	const int buff_elem_size = 1*4; // vertical core

	// load top row
	__m128 t0e, t0o;
	t0e[0] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/0*stride_y);
	t0e[1] = *(float *)(ptr_y0_x0 + /*y=*/2*stride_x + /*x=*/0*stride_y);
	t0e[2] = *(float *)(ptr_y0_x0 + /*y=*/4*stride_x + /*x=*/0*stride_y);
	t0e[3] = *(float *)(ptr_y0_x0 + /*y=*/6*stride_x + /*x=*/0*stride_y);
	t0o[0] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/0*stride_y);
	t0o[1] = *(float *)(ptr_y0_x0 + /*y=*/3*stride_x + /*x=*/0*stride_y);
	t0o[2] = *(float *)(ptr_y0_x0 + /*y=*/5*stride_x + /*x=*/0*stride_y);
	t0o[3] = *(float *)(ptr_y0_x0 + /*y=*/7*stride_x + /*x=*/0*stride_y);

	// load bottom row
	__m128 t1e, t1o;
	t1e[0] = *(float *)(ptr_y0_x0 + /*y=*/0*stride_x + /*x=*/1*stride_y);
	t1e[1] = *(float *)(ptr_y0_x0 + /*y=*/2*stride_x + /*x=*/1*stride_y);
	t1e[2] = *(float *)(ptr_y0_x0 + /*y=*/4*stride_x + /*x=*/1*stride_y);
	t1e[3] = *(float *)(ptr_y0_x0 + /*y=*/6*stride_x + /*x=*/1*stride_y);
	t1o[0] = *(float *)(ptr_y0_x0 + /*y=*/1*stride_x + /*x=*/1*stride_y);
	t1o[1] = *(float *)(ptr_y0_x0 + /*y=*/3*stride_x + /*x=*/1*stride_y);
	t1o[2] = *(float *)(ptr_y0_x0 + /*y=*/5*stride_x + /*x=*/1*stride_y);
	t1o[3] = *(float *)(ptr_y0_x0 + /*y=*/7*stride_x + /*x=*/1*stride_y);

	// horiz top 8x1
	vert_8x1(
		// input even samples
		t0e,
		// input odd samples
		t0o,
		// output even samples
		&t0e,
		// output odd samples
		&t0o,
		// 1x buffer "L" of size (1*4) * sizeof(float)
		buff_v0 + 0*buff_elem_size
	);

	// horiz bottom 8x1
	vert_8x1(
		// input even samples
		t1e,
		// input odd samples
		t1o,
		// output even samples
		&t1e,
		// output odd samples
		&t1o,
		// 1x buffer "L" of size (1*4) * sizeof(float)
		buff_v0 + 1*buff_elem_size
	);

	// vertical 8x2 in parallel on even, then on odd samples
	vert_2x4(
		// left input column [4]
		t0e,
		// right input column [4]
		t1e,
		// output 0 [4]
		&t0e,
		// output 1 [4]
		&t1e,
		// 4x buffer "L" with stride = (1*4) * sizeof(float)
		buff_h0 + 0*4*buff_elem_size
	);
	vert_2x4(
		// left input column [4]
		t0o,
		// right input column [4]
		t1o,
		// output 0 [4]
		&t0o,
		// output 1 [4]
		&t1o,
		// 4x buffer "L" with stride = (1*4) * sizeof(float)
		buff_h0 + 1*4*buff_elem_size
	);

	// scaling
	// [ t0e t0o t0e t0o t0e t0o t0e t0o ] *= [ 1/z^2   1 1/z^2   1 1/z^2   1 1/z^2   1 ]
	// [ t1e e1o t1e e1o t1e e1o t1e e1o ] *= [     1 z^2     1 z^2     1 z^2     1 z^2 ]
	// [ t0e t0e t0e t0e ] *= [ 1/z^2 1/z^2 1/z^2 1/z^2 ]
	// [ t0o t0o t0o t0o ] *= [ 1 1 1 1 ]
	// [ t1e t1e t1e t1e ] *= [ 1 1 1 1 ]
	// [ t1o t1o t1o t1o ] *= [ z^2 z^2 z^2 z^2 ];
	const float z = dwt_cdf97_s1_s;
	t0e *= (const __m128){ 1/(z*z), 1/(z*z), 1/(z*z), 1/(z*z) };
	t1o *= (const __m128){   (z*z),   (z*z),   (z*z),   (z*z) };

	// store top row
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/0*stride_y) = t0e[0];
	*(float *)(out_y0_x0 + /*y=*/2*stride_x + /*x=*/0*stride_y) = t0e[1];
	*(float *)(out_y0_x0 + /*y=*/4*stride_x + /*x=*/0*stride_y) = t0e[2];
	*(float *)(out_y0_x0 + /*y=*/6*stride_x + /*x=*/0*stride_y) = t0e[3];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/0*stride_y) = t0o[0];
	*(float *)(out_y0_x0 + /*y=*/3*stride_x + /*x=*/0*stride_y) = t0o[1];
	*(float *)(out_y0_x0 + /*y=*/5*stride_x + /*x=*/0*stride_y) = t0o[2];
	*(float *)(out_y0_x0 + /*y=*/7*stride_x + /*x=*/0*stride_y) = t0o[3];

	// store bottom row
	*(float *)(out_y0_x0 + /*y=*/0*stride_x + /*x=*/1*stride_y) = t1e[0];
	*(float *)(out_y0_x0 + /*y=*/2*stride_x + /*x=*/1*stride_y) = t1e[1];
	*(float *)(out_y0_x0 + /*y=*/4*stride_x + /*x=*/1*stride_y) = t1e[2];
	*(float *)(out_y0_x0 + /*y=*/6*stride_x + /*x=*/1*stride_y) = t1e[3];
	*(float *)(out_y0_x0 + /*y=*/1*stride_x + /*x=*/1*stride_y) = t1o[0];
	*(float *)(out_y0_x0 + /*y=*/3*stride_x + /*x=*/1*stride_y) = t1o[1];
	*(float *)(out_y0_x0 + /*y=*/5*stride_x + /*x=*/1*stride_y) = t1o[2];
	*(float *)(out_y0_x0 + /*y=*/7*stride_x + /*x=*/1*stride_y) = t1o[3];
#endif
#endif /* __SSE__ */
}

static
void fdwt_cdf97_vert_cor8x8_sse_s(
	intptr_t ptr_y0_x0, // pointer to (0,0)
	intptr_t out_y0_x0, // pointer to (0-shift,0-shift)
	ptrdiff_t stride_x, // +1 row
	ptrdiff_t stride_y, // +1 col
	float *buff_h0, // +(0..1)*(1*4) [ y down> ]
	float *buff_v0  // +(0..7)*(1*4) [ x right> ]
)
{
#ifdef __SSE__
#if 0
	const int buff_elem_size = 1*4; // vertical core

	for(int y0 = 0, y1 = 1; y0 < 8; y0 +=2, y1 +=2)
	{
		for(int x0 = 0, x1 = 1; x0 < 8; x0 +=2, x1 +=2)
		{
			fdwt_cdf97_vert_cor2x2_sse_s(
				// inputs
				/* y= x= */(char *)ptr_y0_x0 + y0*stride_x + x0*stride_y,
				/* y= x= */(char *)ptr_y0_x0 + y0*stride_x + x1*stride_y,
				/* y= x= */(char *)ptr_y0_x0 + y1*stride_x + x0*stride_y,
				/* y= x= */(char *)ptr_y0_x0 + y1*stride_x + x1*stride_y,
				// outputs
				/* y= x= */(char *)out_y0_x0 + y0*stride_x + x0*stride_y,
				/* y= x= */(char *)out_y0_x0 + y0*stride_x + x1*stride_y,
				/* y= x= */(char *)out_y0_x0 + y1*stride_x + x0*stride_y,
				/* y= x= */(char *)out_y0_x0 + y1*stride_x + x1*stride_y,
				// buffers
				buff_h0 + y0*buff_elem_size,
				buff_h0 + y1*buff_elem_size,
				buff_v0 + x0*buff_elem_size,
				buff_v0 + x1*buff_elem_size
			);
		}
	}
#endif
#if 1
	// BUG is here

	const int buff_elem_size = 1*4; // vertical core

	__m128 y0e, y0o;
	__m128 y1e, y1o;
	__m128 y2e, y2o;
	__m128 y3e, y3o;
	__m128 y4e, y4o;
	__m128 y5e, y5o;
	__m128 y6e, y6o;
	__m128 y7e, y7o;

	// load
#define get(y, x, stride_x, stride_y) *(float *)(ptr_y0_x0 + /*y=*/(y)*(stride_x) + /*x=*/(x)*(stride_y))
	for(int x = 0; x < 4; x++)
	{
		y0e[x] = get(2*x+0, 0, stride_x, stride_y);
		y0o[x] = get(2*x+1, 0, stride_x, stride_y);
		y1e[x] = get(2*x+0, 1, stride_x, stride_y);
		y1o[x] = get(2*x+1, 1, stride_x, stride_y);
		y2e[x] = get(2*x+0, 2, stride_x, stride_y);
		y2o[x] = get(2*x+1, 2, stride_x, stride_y);
		y3e[x] = get(2*x+0, 3, stride_x, stride_y);
		y3o[x] = get(2*x+1, 3, stride_x, stride_y);
		y4e[x] = get(2*x+0, 4, stride_x, stride_y);
		y4o[x] = get(2*x+1, 4, stride_x, stride_y);
		y5e[x] = get(2*x+0, 5, stride_x, stride_y);
		y5o[x] = get(2*x+1, 5, stride_x, stride_y);
		y6e[x] = get(2*x+0, 6, stride_x, stride_y);
		y6o[x] = get(2*x+1, 6, stride_x, stride_y);
		y7e[x] = get(2*x+0, 7, stride_x, stride_y);
		y7o[x] = get(2*x+1, 7, stride_x, stride_y);
	}
#undef get

	// 8x1 horizontally for y
	vert_8x1(y0e, y0o, &y0e, &y0o, buff_h0 + 0*buff_elem_size);
	vert_8x1(y1e, y1o, &y1e, &y1o, buff_h0 + 1*buff_elem_size);
	vert_8x1(y2e, y2o, &y2e, &y2o, buff_h0 + 2*buff_elem_size);
	vert_8x1(y3e, y3o, &y3e, &y3o, buff_h0 + 3*buff_elem_size);
	vert_8x1(y4e, y4o, &y4e, &y4o, buff_h0 + 4*buff_elem_size);
	vert_8x1(y5e, y5o, &y5e, &y5o, buff_h0 + 5*buff_elem_size);
	vert_8x1(y6e, y6o, &y6e, &y6o, buff_h0 + 6*buff_elem_size);
	vert_8x1(y7e, y7o, &y7e, &y7o, buff_h0 + 7*buff_elem_size);

	//  transpose
	_MM_TRANSPOSE4_PS(y0e, y1e, y2e, y3e);
	_MM_TRANSPOSE4_PS(y0o, y1o, y2o, y3o);
	_MM_TRANSPOSE4_PS(y4e, y5e, y6e, y7e);
	_MM_TRANSPOSE4_PS(y4o, y5o, y6o, y7o);

	__m128 x0e, x0o;
	__m128 x1e, x1o;
	__m128 x2e, x2o;
	__m128 x3e, x3o;
	__m128 x4e, x4o;
	__m128 x5e, x5o;
	__m128 x6e, x6o;
	__m128 x7e, x7o;

	// TODO: interleave y0e+y4e => x0e, x0o
	x0e = _mm_unpacklo_ps(y0e, y4e);
	x0o = _mm_unpackhi_ps(y0e, y4e);
	x1e = _mm_unpacklo_ps(y0o, y4o);
	x1o = _mm_unpackhi_ps(y0o, y4o);
	x2e = _mm_unpacklo_ps(y1e, y5e);
	x2o = _mm_unpackhi_ps(y1e, y5e);
	x3e = _mm_unpacklo_ps(y1o, y5o);
	x3o = _mm_unpackhi_ps(y1o, y5o);

	x4e = _mm_unpacklo_ps(y2e, y6e);
	x4o = _mm_unpackhi_ps(y2e, y6e);
	x5e = _mm_unpacklo_ps(y2o, y6o);
	x5o = _mm_unpackhi_ps(y2o, y6o);
	x6e = _mm_unpacklo_ps(y3e, y7e);
	x6o = _mm_unpackhi_ps(y3e, y7e);
	x7e = _mm_unpacklo_ps(y3o, y7o);
	x7o = _mm_unpackhi_ps(y3o, y7o);

	// 8x1 vertically for x
	vert_8x1(x0e, x0o, &x0e, &x0o, buff_v0 + 0*buff_elem_size);
	vert_8x1(x1e, x1o, &x1e, &x1o, buff_v0 + 1*buff_elem_size);
	vert_8x1(x2e, x2o, &x2e, &x2o, buff_v0 + 2*buff_elem_size);
	vert_8x1(x3e, x3o, &x3e, &x3o, buff_v0 + 3*buff_elem_size);
	vert_8x1(x4e, x4o, &x4e, &x4o, buff_v0 + 4*buff_elem_size);
	vert_8x1(x5e, x5o, &x5e, &x5o, buff_v0 + 5*buff_elem_size);
	vert_8x1(x6e, x6o, &x6e, &x6o, buff_v0 + 6*buff_elem_size);
	vert_8x1(x7e, x7o, &x7e, &x7o, buff_v0 + 7*buff_elem_size);

	// TODO: scalling

	// store
#define set(y, x, stride_x, stride_y) *(float *)(out_y0_x0 + /*y=*/(y)*(stride_x) + /*x=*/(x)*(stride_y))
	for(int y = 0; y < 4; y++)
	{
		set(2*y+0, 0, stride_x, stride_y) = x0e[y];
		set(2*y+1, 0, stride_x, stride_y) = x0o[y];
		set(2*y+0, 1, stride_x, stride_y) = x1e[y];
		set(2*y+1, 1, stride_x, stride_y) = x1o[y];
		set(2*y+0, 2, stride_x, stride_y) = x2e[y];
		set(2*y+1, 2, stride_x, stride_y) = x2o[y];
		set(2*y+0, 3, stride_x, stride_y) = x3e[y];
		set(2*y+1, 3, stride_x, stride_y) = x3o[y];
		set(2*y+0, 4, stride_x, stride_y) = x4e[y];
		set(2*y+1, 4, stride_x, stride_y) = x4o[y];
		set(2*y+0, 5, stride_x, stride_y) = x5e[y];
		set(2*y+1, 5, stride_x, stride_y) = x5o[y];
		set(2*y+0, 6, stride_x, stride_y) = x6e[y];
		set(2*y+1, 6, stride_x, stride_y) = x6o[y];
		set(2*y+0, 7, stride_x, stride_y) = x7e[y];
		set(2*y+1, 7, stride_x, stride_y) = x7o[y];
	}
#undef set

#endif
#endif /* __SSE__ */
}

void fdwt_cdf97_diag_epi2x2_sse_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outR0,
	float *outL1, float *outR1,
	float *lAL,
	float *lAR,
	float *lBL,
	float *lBR
)
{
#ifdef __SSE__
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v_vert = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f, 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s) };

	__m128 buff = { 0.f, 0.f, 0.f, 0.f };
	__m128 z;

	// A/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lAL+4), *(__m128 *)(lAL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lAR+4), *(__m128 *)(lAR+8));

	// A/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAL+4), w, *(__m128 *)(lAL+0), *(__m128 *)(lAL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lAL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAL+4), *(__m128 *)(lAL+0), *(__m128 *)(lAL+8), z);

	// A/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAR+4), w, *(__m128 *)(lAR+0), *(__m128 *)(lAR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lAR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAR+4), *(__m128 *)(lAR+0), *(__m128 *)(lAR+8), z);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lBL+4), *(__m128 *)(lBL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lBR+4), *(__m128 *)(lBR+8));

	// B/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBL+4), w, *(__m128 *)(lBL+0), *(__m128 *)(lBL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lBL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lBL+4), *(__m128 *)(lBL+0), *(__m128 *)(lBL+8), z);

	// B/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBR+4), w, *(__m128 *)(lBR+0), *(__m128 *)(lBR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lBR+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(lBR+4), *(__m128 *)(lBR+0), *(__m128 *)(lBR+8), z);

	// B/L+R
	op4s_sdl2_scale_s_sse(buff, v_vert);

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
#endif /* __SSE__ */
}

void fdwt_cdf97_diag_nul2x2_sse_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outR0,
	float *outL1, float *outR1,
	float *lAL,
	float *lAR,
	float *lBL,
	float *lBR
)
{
#ifdef __SSE__
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	__m128 buff = { 0.f, 0.f, 0.f, 0.f };
	__m128 z;

	// A/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lAL+4), *(__m128 *)(lAL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lAR+4), *(__m128 *)(lAR+8));

	// A/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAL+4), w, *(__m128 *)(lAL+0), *(__m128 *)(lAL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lAL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAL+4), *(__m128 *)(lAL+0), *(__m128 *)(lAL+8), z);

	// A/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAR+4), w, *(__m128 *)(lAR+0), *(__m128 *)(lAR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lAR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAR+4), *(__m128 *)(lAR+0), *(__m128 *)(lAR+8), z);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lBL+4), *(__m128 *)(lBL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lBR+4), *(__m128 *)(lBR+8));

	// B/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBL+4), w, *(__m128 *)(lBL+0), *(__m128 *)(lBL+8));
	op4s_sdl2_update_s_sse(*(__m128 *)(lBL+4), *(__m128 *)(lBL+0), *(__m128 *)(lBL+8), z);

	// B/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBR+4), w, *(__m128 *)(lBR+0), *(__m128 *)(lBR+8));
	op4s_sdl2_update_s_sse(*(__m128 *)(lBR+4), *(__m128 *)(lBR+0), *(__m128 *)(lBR+8), z);
#endif /* __SSE__ */
}

int check_sink(float *sink)
{
	if(
		*(sink+0) != 0.f ||
		*(sink+1) != 0.f ||
		*(sink+2) != 0.f ||
		*(sink+3) != 0.f )
	{
		*(sink+0) =
		*(sink+1) =
		*(sink+2) =
		*(sink+3) = 0.f;
		return 1;
	}

	return 0;
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_diag_2x2_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 10; // diag
	const int buff_elem_size = 3*4; // diag

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x+0,       stride_x, stride_y);
	char *ptr_y1_x0 = (void *)addr2_s(ptr, base_y+1,       base_x+0,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x+0-shift, stride_x, stride_y);
	char *out_y1_x0 = (void *)addr2_s(ptr, base_y+1-shift, base_x+0-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y0_x1 = (ptrdiff_t)addr1_s(0, +1, stride_y); // +1 rows

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *ptr_y1_x0_i = ptr_y1_x0;
		char *out_y0_x0_i = out_y0_x0;
		char *out_y1_x0_i = out_y1_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_diag_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i,
				(void *)ptr_y0_x0_i + diff_y0_x1,
				(void *)ptr_y1_x0_i,
				(void *)ptr_y1_x0_i + diff_y0_x1,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i,
				(void *)out_y0_x0_i + diff_y0_x1,
				(void *)out_y1_x0_i,
				(void *)out_y1_x0_i + diff_y0_x1,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_x2;
			ptr_y1_x0_i += diff_x2;
			out_y0_x0_i += diff_x2;
			out_y1_x0_i += diff_x2;

			buffer_x0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y2;
		ptr_y1_x0 += diff_y2;
		out_y0_x0 += diff_y2;
		out_y1_x0 += diff_y2;

		buffer_y0_i += 2*(buff_elem_size);
	}
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_diag_6x2_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 10; // diagonal core
	const int buff_elem_size = 3*4; // diagonal core

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x+0,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x+0-shift, stride_x, stride_y);

	// core size
	const int step_x = 6;
	const int step_y = 2;

	// increments
	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +step_y, stride_x); // +step_y cols
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +step_x, stride_y); // +step_x rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+(step_y-1) < stop_y; y += step_y)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+(step_x-1) < stop_x; x += step_x)
		{
			fdwt_cdf97_diag_cor6x2_sse_s(
				// ptr
				(void *)ptr_y0_x0_i,
				// out
				(void *)out_y0_x0_i,
				// strides
				stride_x,
				stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			ptr_y0_x0_i += diff_x;
			out_y0_x0_i += diff_x;

			buffer_x0_i += step_x*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y;
		out_y0_x0 += diff_y;

		buffer_y0_i += step_y*(buff_elem_size);
	}
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_diag_2x6_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
// 	dwt_util_log(LOG_DBG, "diag_2x6 needs image to be multiple of (2,6) blocks: image size (%i,%i) has modulo (%i,%i)\n",
// 		     (stop_x-base_x), (stop_y-base_y),
// 		     (stop_x-base_x)%6, (stop_y-base_y)%6
// 	);

	// characteristic constants
	const int shift = 10; // diagonal core
	const int buff_elem_size = 3*4; // diagonal core

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x+0,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x+0-shift, stride_x, stride_y);

	// core size
	const int step_x = 2;
	const int step_y = 6;

	// increments
	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +step_y, stride_x); // +step_y cols
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +step_x, stride_y); // +step_x rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+(step_y-1) < stop_y; y += step_y)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+(step_x-1) < stop_x; x += step_x)
		{
			fdwt_cdf97_diag_cor2x6_sse_s(
				// ptr
				(void *)ptr_y0_x0_i,
				// out
				(void *)out_y0_x0_i,
				// strides
				stride_x,
				stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			ptr_y0_x0_i += diff_x;
			out_y0_x0_i += diff_x;

			buffer_x0_i += step_x*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y;
		out_y0_x0 += diff_y;

		buffer_y0_i += step_y*(buff_elem_size);
	}
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_diag_6x6_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 10; // diagonal core
	const int buff_elem_size = 3*4; // diagonal core

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x+0,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x+0-shift, stride_x, stride_y);

	// core size
	const int step_x = 6;
	const int step_y = 6;

	// increments
	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +step_y, stride_x); // +step_y cols
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +step_x, stride_y); // +step_x rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+(step_y-1) < stop_y; y += step_y)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+(step_x-1) < stop_x; x += step_x)
		{
			fdwt_cdf97_diag_cor6x6_sse_s(
				// ptr
				(void *)ptr_y0_x0_i,
				// out
				(void *)out_y0_x0_i,
				// strides
				stride_x,
				stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			ptr_y0_x0_i += diff_x;
			out_y0_x0_i += diff_x;

			buffer_x0_i += step_x*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y;
		out_y0_x0 += diff_y;

		buffer_y0_i += step_y*(buff_elem_size);
	}
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_diag_2x2_cor_HORIZ_nice(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 10; // diag
	const int buff_elem_size = 3*4; // diag

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y-shift, base_x-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y0_x0 = 0; // +0 col +0 row
	const ptrdiff_t diff_y1_x0 = (ptrdiff_t)addr1_s(0, +1, stride_x); // +1 cols
	const ptrdiff_t diff_y0_x1 = (ptrdiff_t)addr1_s(0, +1, stride_y); // +1 rows
	const ptrdiff_t diff_y1_x1 = diff_y0_x1 + diff_y1_x0; // +1 col +1 row

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_diag_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i + diff_y0_x0,
				(void *)ptr_y0_x0_i + diff_y0_x1,
				(void *)ptr_y0_x0_i + diff_y1_x0,
				(void *)ptr_y0_x0_i + diff_y1_x1,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i + diff_y0_x0,
				(void *)out_y0_x0_i + diff_y0_x1,
				(void *)out_y0_x0_i + diff_y1_x0,
				(void *)out_y0_x0_i + diff_y1_x1,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_x2;
			out_y0_x0_i += diff_x2;

			buffer_x0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y2;
		out_y0_x0 += diff_y2;

		buffer_y0_i += 2*(buff_elem_size);
	}
}

// TODO
void fdwt_vert_2x2_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
#if 0
	dwt_util_log(LOG_DBG, "performing: core=vertical order=horizontal size=2x2\n");
#endif
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 1*4 for DL aka vertical

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	//fdwt_vert_2x2_cor_HORIZ_4pointers( // 4 pointers
	fdwt_vert_2x2_cor_HORIZ( // 2 pointers
	//fdwt_vert_2x2_cor_HORIZ_nice( // 1 pointer
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_vert_2x2_cor_HORIZ_4pointers(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // vert
	const int buff_elem_size = 1*4; // vert

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x+0,       stride_x, stride_y);
	char *ptr_y0_x1 = (void *)addr2_s(ptr, base_y+0,       base_x+1,       stride_x, stride_y);
	char *ptr_y1_x0 = (void *)addr2_s(ptr, base_y+1,       base_x+0,       stride_x, stride_y);
	char *ptr_y1_x1 = (void *)addr2_s(ptr, base_y+1,       base_x+1,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x+0-shift, stride_x, stride_y);
	char *out_y0_x1 = (void *)addr2_s(ptr, base_y+0-shift, base_x+1-shift, stride_x, stride_y);
	char *out_y1_x0 = (void *)addr2_s(ptr, base_y+1-shift, base_x+0-shift, stride_x, stride_y);
	char *out_y1_x1 = (void *)addr2_s(ptr, base_y+1-shift, base_x+1-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *ptr_y0_x1_i = ptr_y0_x1;
		char *ptr_y1_x0_i = ptr_y1_x0;
		char *ptr_y1_x1_i = ptr_y1_x1;
		char *out_y0_x0_i = out_y0_x0;
		char *out_y0_x1_i = out_y0_x1;
		char *out_y1_x0_i = out_y1_x0;
		char *out_y1_x1_i = out_y1_x1;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_vert_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i,
				(void *)ptr_y0_x1_i,
				(void *)ptr_y1_x0_i,
				(void *)ptr_y1_x1_i,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i,
				(void *)out_y0_x1_i,
				(void *)out_y1_x0_i,
				(void *)out_y1_x1_i,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_x2;
			ptr_y0_x1_i += diff_x2;
			ptr_y1_x0_i += diff_x2;
			ptr_y1_x1_i += diff_x2;
			out_y0_x0_i += diff_x2;
			out_y0_x1_i += diff_x2;
			out_y1_x0_i += diff_x2;
			out_y1_x1_i += diff_x2;

			buffer_x0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y2;
		ptr_y0_x1 += diff_y2;
		ptr_y1_x0 += diff_y2;
		ptr_y1_x1 += diff_y2;
		out_y0_x0 += diff_y2;
		out_y0_x1 += diff_y2;
		out_y1_x0 += diff_y2;
		out_y1_x1 += diff_y2;

		buffer_y0_i += 2*(buff_elem_size);
	}
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_vert_4x4_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // diag
	const int buff_elem_size = 1*4; // diag

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y4 = (ptrdiff_t)addr1_s(0, +4, stride_x); // +2 cols
	const ptrdiff_t diff_x4 = (ptrdiff_t)addr1_s(0, +4, stride_y); // +2 rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+3 < stop_y; y += 4)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+3 < stop_x; x += 4)
		{
			fdwt_cdf97_vert_cor4x4_sse_s(
				// ptr
				(void *)ptr_y0_x0_i,
				// out
				(void *)out_y0_x0_i,
				stride_x,
				stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			ptr_y0_x0_i += diff_x4;
			out_y0_x0_i += diff_x4;

			buffer_x0_i += 4*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y4;
		out_y0_x0 += diff_y4;

		buffer_y0_i += 4*(buff_elem_size);
	}
}

void fdwt_vert_8x2_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // diag
	const int buff_elem_size = 1*4; // diag

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x-shift, stride_x, stride_y);

	const int step_x = 8;
	const int step_y = 2;

	// increments
	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +step_y, stride_x); // +step_y cols
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +step_x, stride_y); // +step_x rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+(step_y-1) < stop_y; y += step_y)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+(step_x-1) < stop_x; x += step_x)
		{
			fdwt_cdf97_vert_cor8x2_sse_s(
				// input pointer
				(void *)ptr_y0_x0_i,
				// output pointer
				(void *)out_y0_x0_i,
				// strides
				stride_x,
				stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			ptr_y0_x0_i += diff_x;
			out_y0_x0_i += diff_x;

			buffer_x0_i += step_x*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y;
		out_y0_x0 += diff_y;

		buffer_y0_i += step_y*(buff_elem_size);
	}
}

void fdwt_vert_2x8_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // diag
	const int buff_elem_size = 1*4; // diag

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x-shift, stride_x, stride_y);

	const int step_x = 2;
	const int step_y = 8;

	// increments
	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +step_y, stride_x); // +step_y cols
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +step_x, stride_y); // +step_x rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+(step_y-1) < stop_y; y += step_y)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+(step_x-1) < stop_x; x += step_x)
		{
			fdwt_cdf97_vert_cor2x8_sse_s(
				// input pointer
				(void *)ptr_y0_x0_i,
				// output pointer
				(void *)out_y0_x0_i,
				// strides
				stride_x,
				stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			ptr_y0_x0_i += diff_x;
			out_y0_x0_i += diff_x;

			buffer_x0_i += step_x*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y;
		out_y0_x0 += diff_y;

		buffer_y0_i += step_y*(buff_elem_size);
	}
}

void fdwt_vert_8x8_cor_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // diag
	const int buff_elem_size = 1*4; // diag

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x-shift, stride_x, stride_y);

	const int step_x = 8;
	const int step_y = 8;

	// increments
	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +step_y, stride_x); // +step_y cols
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +step_x, stride_y); // +step_x rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+(step_y-1) < stop_y; y += step_y)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+(step_x-1) < stop_x; x += step_x)
		{
			fdwt_cdf97_vert_cor8x8_sse_s(
				// input pointer
				(void *)ptr_y0_x0_i,
				// output pointer
				(void *)out_y0_x0_i,
				// strides
				stride_x,
				stride_y,
				// buffers
				buffer_y0_i,
				buffer_x0_i
			);

			ptr_y0_x0_i += diff_x;
			out_y0_x0_i += diff_x;

			buffer_x0_i += step_x*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y;
		out_y0_x0 += diff_y;

		buffer_y0_i += step_y*(buff_elem_size);
	}
}

// NOTE: read from (x,y) write to (x-shift,y-shift) for (base_x,base_y) <= (x,y) < (stop_x, stop_y)
void fdwt_vert_2x2_cor_HORIZ_nice(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // diag
	const int buff_elem_size = 1*4; // diag

	// initially
	float *const buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *const buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y-shift, base_x-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y0_x0 = 0; // +0 col +0 row
	const ptrdiff_t diff_y1_x0 = (ptrdiff_t)addr1_s(0, +1, stride_x); // +1 cols
	const ptrdiff_t diff_y0_x1 = (ptrdiff_t)addr1_s(0, +1, stride_y); // +1 rows
	const ptrdiff_t diff_y1_x1 = diff_y0_x1 + diff_y1_x0; // +1 col +1 row

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_y0_i = buffer_y0;

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_x0_i = buffer_x0;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_vert_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i + diff_y0_x0,
				(void *)ptr_y0_x0_i + diff_y0_x1,
				(void *)ptr_y0_x0_i + diff_y1_x0,
				(void *)ptr_y0_x0_i + diff_y1_x1,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i + diff_y0_x0,
				(void *)out_y0_x0_i + diff_y0_x1,
				(void *)out_y0_x0_i + diff_y1_x0,
				(void *)out_y0_x0_i + diff_y1_x1,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_x2;
			out_y0_x0_i += diff_x2;

			buffer_x0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_y2;
		out_y0_x0 += diff_y2;

		buffer_y0_i += 2*(buff_elem_size);
	}
}

void fdwt_diag_2x2_cor_VERT_4pointers(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 10; // diag
	const int buff_elem_size = 3*4; // diag

	// initially
	float *buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x+0,       stride_x, stride_y);
	char *ptr_y0_x1 = (void *)addr2_s(ptr, base_y+0,       base_x+1,       stride_x, stride_y);
	char *ptr_y1_x0 = (void *)addr2_s(ptr, base_y+1,       base_x+0,       stride_x, stride_y);
	char *ptr_y1_x1 = (void *)addr2_s(ptr, base_y+1,       base_x+1,       stride_x, stride_y);

	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x+0-shift, stride_x, stride_y);
	char *out_y0_x1 = (void *)addr2_s(ptr, base_y+0-shift, base_x+1-shift, stride_x, stride_y);
	char *out_y1_x0 = (void *)addr2_s(ptr, base_y+1-shift, base_x+0-shift, stride_x, stride_y);
	char *out_y1_x1 = (void *)addr2_s(ptr, base_y+1-shift, base_x+1-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_x0_i = buffer_x0;

	for(int x = base_x; x+1 < stop_x; x += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *ptr_y0_x1_i = ptr_y0_x1;
		char *ptr_y1_x0_i = ptr_y1_x0;
		char *ptr_y1_x1_i = ptr_y1_x1;

		char *out_y0_x0_i = out_y0_x0;
		char *out_y0_x1_i = out_y0_x1;
		char *out_y1_x0_i = out_y1_x0;
		char *out_y1_x1_i = out_y1_x1;

		float *buffer_y0_i = buffer_y0;

		for(int y = base_y; y+1 < stop_y; y += 2)
		{
			fdwt_cdf97_diag_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i,
				(void *)ptr_y0_x1_i,
				(void *)ptr_y1_x0_i,
				(void *)ptr_y1_x1_i ,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i,
				(void *)out_y0_x1_i,
				(void *)out_y1_x0_i,
				(void *)out_y1_x1_i,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_y2;
			ptr_y0_x1_i += diff_y2;
			ptr_y1_x0_i += diff_y2;
			ptr_y1_x1_i += diff_y2;

			out_y0_x0_i += diff_y2;
			out_y0_x1_i += diff_y2;
			out_y1_x0_i += diff_y2;
			out_y1_x1_i += diff_y2;

			buffer_y0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_x2;
		ptr_y0_x1 += diff_x2;
		ptr_y1_x0 += diff_x2;
		ptr_y1_x1 += diff_x2;

		out_y0_x0 += diff_x2;
		out_y0_x1 += diff_x2;
		out_y1_x0 += diff_x2;
		out_y1_x1 += diff_x2;

		buffer_x0_i += 2*(buff_elem_size);
	}
}

void fdwt_diag_2x2_cor_VERT(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 10; // diag
	const int buff_elem_size = 3*4; // diag

	// initially
	float *buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y-shift, base_x-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y0_x0 = 0; // +0 col +0 row
	const ptrdiff_t diff_y1_x0 = (ptrdiff_t)addr1_s(0, +1, stride_x); // +1 cols
	const ptrdiff_t diff_y0_x1 = (ptrdiff_t)addr1_s(0, +1, stride_y); // +1 rows
	const ptrdiff_t diff_y1_x1 = diff_y0_x1 + diff_y1_x0; // +1 col +1 row

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_x0_i = buffer_x0;

	for(int x = base_x; x+1 < stop_x; x += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_y0_i = buffer_y0;

		for(int y = base_y; y+1 < stop_y; y += 2)
		{
			fdwt_cdf97_diag_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i + diff_y0_x0,
				(void *)ptr_y0_x0_i + diff_y0_x1,
				(void *)ptr_y0_x0_i + diff_y1_x0,
				(void *)ptr_y0_x0_i + diff_y1_x1,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i + diff_y0_x0,
				(void *)out_y0_x0_i + diff_y0_x1,
				(void *)out_y0_x0_i + diff_y1_x0,
				(void *)out_y0_x0_i + diff_y1_x1,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_y2;
			out_y0_x0_i += diff_y2;

			buffer_y0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_x2;
		out_y0_x0 += diff_x2;

		buffer_x0_i += 2*(buff_elem_size);
	}
}

// TODO: not by rows; by columns
void fdwt_vert_2x2_cor_VERT(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	// characteristic constants
	const int shift = 4; // vert
	const int buff_elem_size = 1*4; // vert

	// initially
	float *buffer_y0 = &buffer_y[(base_y)*(buff_elem_size)];
	float *buffer_x0 = &buffer_x[(base_x)*(buff_elem_size)];

	// initially
	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y,       base_x,       stride_x, stride_y);
	char *out_y0_x0 = (void *)addr2_s(ptr, base_y-shift, base_x-shift, stride_x, stride_y);

	// increments
	const ptrdiff_t diff_y0_x0 = 0; // +0 col +0 row
	const ptrdiff_t diff_y1_x0 = (ptrdiff_t)addr1_s(0, +1, stride_x); // +1 cols
	const ptrdiff_t diff_y0_x1 = (ptrdiff_t)addr1_s(0, +1, stride_y); // +1 rows
	const ptrdiff_t diff_y1_x1 = diff_y0_x1 + diff_y1_x0; // +1 col +1 row

	// increments
	const ptrdiff_t diff_y2 = (ptrdiff_t)addr1_s(0, +2, stride_x); // +2 cols
	const ptrdiff_t diff_x2 = (ptrdiff_t)addr1_s(0, +2, stride_y); // +2 rows

	float *buffer_x0_i = buffer_x0;

	for(int x = base_x; x+1 < stop_x; x += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *out_y0_x0_i = out_y0_x0;

		float *buffer_y0_i = buffer_y0;

		for(int y = base_y; y+1 < stop_y; y += 2)
		{
			fdwt_cdf97_vert_cor2x2_sse_s(
				// ptr
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)ptr_y0_x0_i + diff_y0_x0,
				(void *)ptr_y0_x0_i + diff_y0_x1,
				(void *)ptr_y0_x0_i + diff_y1_x0,
				(void *)ptr_y0_x0_i + diff_y1_x1,
				// out
				// TODO: use gather/scatter for indirect memory access (e.g., _mm256_i64gather_ps)
				(void *)out_y0_x0_i + diff_y0_x0,
				(void *)out_y0_x0_i + diff_y0_x1,
				(void *)out_y0_x0_i + diff_y1_x0,
				(void *)out_y0_x0_i + diff_y1_x1,
				// buffers
				buffer_y0_i+0*(buff_elem_size),
				buffer_y0_i+1*(buff_elem_size),
				buffer_x0_i+0*(buff_elem_size),
				buffer_x0_i+1*(buff_elem_size)
			);

			ptr_y0_x0_i += diff_y2;
			out_y0_x0_i += diff_y2;

			buffer_y0_i += 2*(buff_elem_size);
		}

		ptr_y0_x0 += diff_x2;
		out_y0_x0 += diff_x2;

		buffer_x0_i += 2*(buff_elem_size);
	}
}

void fdwt_diag_2x2_pro(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	const int shift = 10; // diag

	//float sink[4] = { 0.f, 0.f, 0.f, 0.f };

	float *buffer_y0 = &buffer_y[(base_y+0)*(3*4)];
	float *buffer_y1 = &buffer_y[(base_y+1)*(3*4)];
	float *buffer_x0 = &buffer_x[(base_x+0)*(3*4)];
	float *buffer_x1 = &buffer_x[(base_x+1)*(3*4)];

	char *ptr_y0_x0 = (void *)addr2_s(ptr, base_y+0,       base_x+0,       stride_x, stride_y);
	char *ptr_y0_x1 = (void *)addr2_s(ptr, base_y+0,       base_x+1,       stride_x, stride_y);
	char *ptr_y1_x0 = (void *)addr2_s(ptr, base_y+1,       base_x+0,       stride_x, stride_y);
	char *ptr_y1_x1 = (void *)addr2_s(ptr, base_y+1,       base_x+1,       stride_x, stride_y);

	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +2, stride_x);
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +2, stride_y);

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		char *ptr_y0_x0_i = ptr_y0_x0;
		char *ptr_y0_x1_i = ptr_y0_x1;
		char *ptr_y1_x0_i = ptr_y1_x0;
		char *ptr_y1_x1_i = ptr_y1_x1;

		float *buffer_x0_i = buffer_x0;
		float *buffer_x1_i = buffer_x1;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_diag_pro2x2_sse_s(
				(void *)ptr_y0_x0_i,
				(void *)ptr_y0_x1_i,
				(void *)ptr_y1_x0_i,
				(void *)ptr_y1_x1_i,
				0,
				0,
				0,
				0,
				buffer_y0,
				buffer_y1,
				buffer_x0_i,
				buffer_x1_i
			);

			ptr_y0_x0_i += diff_x;
			ptr_y0_x1_i += diff_x;
			ptr_y1_x0_i += diff_x;
			ptr_y1_x1_i += diff_x;

			buffer_x0_i += 2*(3*4);
			buffer_x1_i += 2*(3*4);
		}

		ptr_y0_x0 += diff_y;
		ptr_y0_x1 += diff_y;
		ptr_y1_x0 += diff_y;
		ptr_y1_x1 += diff_y;

		buffer_y0 += 2*(3*4);
		buffer_y1 += 2*(3*4);
	}
}

void fdwt_diag_2x2_epi(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	const int shift = 10; // diag

	//float sink[4] = { 0.f, 0.f, 0.f, 0.f };

	float *buffer_y0 = &buffer_y[(base_y+0)*(3*4)];
	float *buffer_y1 = &buffer_y[(base_y+1)*(3*4)];
	float *buffer_x0 = &buffer_x[(base_x+0)*(3*4)];
	float *buffer_x1 = &buffer_x[(base_x+1)*(3*4)];

	char *out_y0_x0 = (void *)addr2_s(ptr, base_y+0-shift, base_x+0-shift, stride_x, stride_y);
	char *out_y0_x1 = (void *)addr2_s(ptr, base_y+0-shift, base_x+1-shift, stride_x, stride_y);
	char *out_y1_x0 = (void *)addr2_s(ptr, base_y+1-shift, base_x+0-shift, stride_x, stride_y);
	char *out_y1_x1 = (void *)addr2_s(ptr, base_y+1-shift, base_x+1-shift, stride_x, stride_y);

	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +2, stride_x);
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +2, stride_y);

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		char *out_y0_x0_i = out_y0_x0;
		char *out_y0_x1_i = out_y0_x1;
		char *out_y1_x0_i = out_y1_x0;
		char *out_y1_x1_i = out_y1_x1;

		float *buffer_x0_i = buffer_x0;
		float *buffer_x1_i = buffer_x1;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_diag_epi2x2_sse_s(
				0,
				0,
				0,
				0,
				(void *)out_y0_x0_i,
				(void *)out_y0_x1_i,
				(void *)out_y1_x0_i,
				(void *)out_y1_x1_i,
				buffer_y0,
				buffer_y1,
				buffer_x0_i,
				buffer_x1_i
			);

			out_y0_x0_i += diff_x;
			out_y0_x1_i += diff_x;
			out_y1_x0_i += diff_x;
			out_y1_x1_i += diff_x;

			buffer_x0_i += 2*(3*4);
			buffer_x1_i += 2*(3*4);
		}

		out_y0_x0 += diff_y;
		out_y0_x1 += diff_y;
		out_y1_x0 += diff_y;
		out_y1_x1 += diff_y;

		buffer_y0 += 2*(3*4);
		buffer_y1 += 2*(3*4);
	}
}

void fdwt_diag_2x2_nul(
	void *ptr,
	int stride_x,
	int stride_y,
	int base_x, // start at ...
	int base_y,
	int stop_x, // stop at ...
	int stop_y,
	float *buffer_y, // short_buffer
	float *buffer_x  // long_buffer
)
{
	const int shift = 10; // diag

	//float sink[4] = { 0.f, 0.f, 0.f, 0.f };

	float *buffer_y0 = &buffer_y[(base_y+0)*(3*4)];
	float *buffer_y1 = &buffer_y[(base_y+1)*(3*4)];
	float *buffer_x0 = &buffer_x[(base_x+0)*(3*4)];
	float *buffer_x1 = &buffer_x[(base_x+1)*(3*4)];

	const ptrdiff_t diff_y = (ptrdiff_t)addr1_s(0, +2, stride_x);
	const ptrdiff_t diff_x = (ptrdiff_t)addr1_s(0, +2, stride_y);

	for(int y = base_y; y+1 < stop_y; y += 2)
	{
		float *buffer_x0_i = buffer_x0;
		float *buffer_x1_i = buffer_x1;

		for(int x = base_x; x+1 < stop_x; x += 2)
		{
			fdwt_cdf97_diag_nul2x2_sse_s(
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				buffer_y0,
				buffer_y1,
				buffer_x0_i,
				buffer_x1_i
			);

			buffer_x0_i += 2*(3*4);
			buffer_x1_i += 2*(3*4);
		}

		buffer_y0 += 2*(3*4);
		buffer_y1 += 2*(3*4);
	}
}

void fdwt_diag_2x2(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// shift
	int shift = 10; // 10 for SDL aka diagonal
	//int border = 4; // 4 for 4 lifting steps

	int prolog_x = shift;
	int prolog_y = shift;
	int epilog_x = shift;
	int epilog_y = shift;

	float buffer_x[(3*4)*(size_x+shift)] ALIGNED(16);
	float buffer_y[(3*4)*(size_y+shift)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (3*4)*(size_x+shift));
	dwt_util_zero_vec_s(buffer_y, (3*4)*(size_y+shift));

	// NOTE: loops iterate over source image (read head)

	// [1a = prolog on top]
	fdwt_diag_2x2_pro(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		prolog_y, // stop_y
		buffer_y,
		buffer_x
	);

	// [1.1 = nullog on top]
	fdwt_diag_2x2_nul(
		ptr,
		stride_x,
		stride_y,
		size_x, // base_x
		0, // base_y
		size_x+epilog_x, // stop_x
		prolog_y, // stop_y
		buffer_y,
		buffer_x
	);

	// [2 = prolog on left]
	fdwt_diag_2x2_pro(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		prolog_y, // base_y
		prolog_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x 
	);

	// [2.1 = nullog on left]
	// FIXME: no effect
	fdwt_diag_2x2_nul(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		size_y, // base_y
		prolog_x, // stop_x
		size_y+epilog_y, // stop_y
		buffer_y,
		buffer_x 
	);

	// [3 = core]
	fdwt_diag_2x2_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		prolog_x, // base_x
		prolog_y, // base_y
		size_y, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);

	// [4 = epilog on right]
	fdwt_diag_2x2_epi(
		ptr,
		stride_x,
		stride_y,
		size_x, // base_x
		prolog_y, // base_y
		size_x+epilog_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);

	// [5 = epilog on bottom]
	fdwt_diag_2x2_epi(
		ptr,
		stride_x,
		stride_y,
		prolog_x, // base_x
		size_y, // base_y
		size_x+epilog_x, // stop_x
		size_y+epilog_y, // stop_y
		buffer_y,
		buffer_x
	);
}

// HACK: core only, no borders
void fdwt_diag_2x2_full(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// shift
	const int shift = 10; // 10 for SDL aka diagonal
	const int buff_elem = 3*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x+shift)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y+shift)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x+shift));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y+shift));

	const int strip_size_x = 512; // 154 elements
	const int strip_size_y = 16; // 154
	// NOTE: loops iterate over source image (read head)
	//dwt_util_log(LOG_DBG, "2x2_full: input: (x=%i,y=%i)\n", size_x, size_y);

	int base_x = 0;
	for( ; base_x+strip_size_x <= size_x+shift; base_x+=strip_size_x)
	{
		int base_y = 0;
		for( ; base_y+strip_size_y <= size_y+shift; base_y+=strip_size_y)
		{
			//dwt_util_log(LOG_DBG, "2x2_full: strip: base_x=%i stop_x=%i\n", base_x, base_x+strip_size_x);
			fdwt_diag_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				base_y+strip_size_y,//size_y+shift, // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last y
		{
			fdwt_diag_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				size_y+shift, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
	// last x
	{
		//dwt_util_log(LOG_DBG, "last: %i..%i\n", base_x, size_x+shift);
		fdwt_diag_2x2_cor_HORIZ(
			ptr,
			stride_x,
			stride_y,
			base_x, // base_x
			0, // base_y
			size_x+shift, // stop_x
			size_y+shift, // stop_y
			buffer_y,
			buffer_x
		);
	}
}

int g_strip_x = 2;
int g_strip_y = 2;

void fdwt_diag_2x2_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_diag_2x2_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_diag_6x2_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for diagonal core

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_diag_6x2_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_diag_2x6_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for diagonal core

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_diag_2x6_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_diag_6x6_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for diagonal core

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_diag_6x6_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_vert_4x4_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_vert_4x4_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_vert_8x2_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_vert_8x2_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_vert_2x8_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_vert_2x8_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_vert_8x8_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_vert_8x8_cor_HORIZ(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_diag_2x2_VERT(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_diag_2x2_cor_VERT(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_vert_2x2_VERT(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	// NOTE: loops iterate over source image (read head)

	fdwt_vert_2x2_cor_VERT(
		ptr,
		stride_x,
		stride_y,
		0, // base_x
		0, // base_y
		size_x, // stop_x
		size_y, // stop_y
		buffer_y,
		buffer_x
	);
}

void fdwt_diag_2x2_HORIZ_STRIPS(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_x = g_strip_x; // 128

	// NOTE: loops iterate over source image (read head)

	int base_x = 0;
	// strips
	for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
	{
		fdwt_diag_2x2_cor_HORIZ(
			ptr,
			stride_x,
			stride_y,
			base_x, // base_x
			0,      // base_y
			base_x+strip_size_x, // stop_x
			size_y,              // stop_y
			buffer_y,
			buffer_x
		);
	}
	// last x
	{
		fdwt_diag_2x2_cor_HORIZ(
			ptr,
			stride_x,
			stride_y,
			base_x, // base_x
			0,      // base_y
			size_x, // stop_x
			size_y, // stop_y
			buffer_y,
			buffer_x
		);
	}
}

void fdwt_vert_2x2_HORIZ_STRIPS(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_x = g_strip_x; // 128

	// NOTE: loops iterate over source image (read head)

	int base_x = 0;
	// strips
	for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
	{
		fdwt_vert_2x2_cor_HORIZ(
			ptr,
			stride_x,
			stride_y,
			base_x, // base_x
			0,      // base_y
			base_x+strip_size_x, // stop_x
			size_y,              // stop_y
			buffer_y,
			buffer_x
		);
	}
	// last x
	{
		fdwt_vert_2x2_cor_HORIZ(
			ptr,
			stride_x,
			stride_y,
			base_x, // base_x
			0,      // base_y
			size_x, // stop_x
			size_y, // stop_y
			buffer_y,
			buffer_x
		);
	}
}

void fdwt_diag_2x2_VERT_STRIPS(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_y = g_strip_y; // 8

	// NOTE: loops iterate over source image (read head)

	int base_y = 0;
	// strips
	for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
	{
		fdwt_diag_2x2_cor_VERT(
			ptr,
			stride_x,
			stride_y,
			0,           // base_x
			base_y,      // base_y
			size_x,              // stop_y
			base_y+strip_size_y, // stop_y
			buffer_y,
			buffer_x
		);
	}
	// last y
	{
		fdwt_diag_2x2_cor_VERT(
			ptr,
			stride_x,
			stride_y,
			0,      // base_x
			base_y, // base_y
			size_x, // stop_x
			size_y, // stop_y
			buffer_y,
			buffer_x
		);
	}
}

void fdwt_vert_2x2_VERT_STRIPS(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_y = g_strip_y; // 8

	// NOTE: loops iterate over source image (read head)

	int base_y = 0;
	// strips
	for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
	{
		fdwt_vert_2x2_cor_VERT(
			ptr,
			stride_x,
			stride_y,
			0,           // base_x
			base_y,      // base_y
			size_x,              // stop_y
			base_y+strip_size_y, // stop_y
			buffer_y,
			buffer_x
		);
	}
	// last y
	{
		fdwt_vert_2x2_cor_VERT(
			ptr,
			stride_x,
			stride_y,
			0,      // base_x
			base_y, // base_y
			size_x, // stop_x
			size_y, // stop_y
			buffer_y,
			buffer_x
		);
	}
}

void fdwt_diag_2x2_VERT_BLOCK(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_x = g_strip_x; // 32
	const int strip_size_y = g_strip_y; // 32

	// NOTE: loops iterate over source image (read head)

	int base_x = 0;
	// strips x
	for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
	{
		int base_y = 0;
		// strips y
		for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
		{
			fdwt_diag_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last y
		{
			fdwt_diag_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
	// last x
	{
		int base_y = 0;
		// strips y
		for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
		{
			fdwt_diag_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				size_x,              // stop_x
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last y
		{
			fdwt_diag_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				size_x, // stop_x
				size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
}

void fdwt_vert_2x2_VERT_BLOCK(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_x = g_strip_x; // 32
	const int strip_size_y = g_strip_y; // 32

	// NOTE: loops iterate over source image (read head)

	int base_x = 0;
	// strips x
	for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
	{
		int base_y = 0;
		// strips y
		for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
		{
			fdwt_vert_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last y
		{
			fdwt_vert_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
	// last x
	{
		int base_y = 0;
		// strips y
		for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
		{
			fdwt_vert_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				size_x,              // stop_x
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last y
		{
			fdwt_vert_2x2_cor_VERT(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				size_x, // stop_x
				size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
}

void fdwt_diag_2x2_HORIZ_BLOCK(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 3*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_x = g_strip_x; // 128
	const int strip_size_y = g_strip_y; // 128

	// NOTE: loops iterate over source image (read head)

	int base_y = 0;
	// strips y
	for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
	{
		int base_x = 0;
		// strips x
		for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
		{
			fdwt_diag_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x,       // base_x
				base_y,       // base_y
				base_x+strip_size_x, // stop_y
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last x
		{
			fdwt_diag_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x,       // base_x
				base_y,      // base_y
				size_x,              // stop_y
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
	// last y
	{
		int base_x = 0;
		// strips x
		for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
		{
			fdwt_diag_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				size_y,              // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last x
		{
			fdwt_diag_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				size_x, // stop_x
				size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
}

void fdwt_vert_2x2_HORIZ_BLOCK(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( is_even(size_x) && is_even(size_y) );
	assert( size_x >= 0 && size_y >= 0 );

	// aux.
	const int buff_elem = 1*4; // 3*4 for SDL aka diagonal

	float buffer_x[(buff_elem)*(size_x)] ALIGNED(16);
	float buffer_y[(buff_elem)*(size_y)] ALIGNED(16);

	dwt_util_zero_vec_s(buffer_x, (buff_elem)*(size_x));
	dwt_util_zero_vec_s(buffer_y, (buff_elem)*(size_y));

	const int strip_size_x = g_strip_x; // 128
	const int strip_size_y = g_strip_y; // 128

	// NOTE: loops iterate over source image (read head)

	int base_y = 0;
	// strips y
	for( ; base_y+strip_size_y <= size_y; base_y+=strip_size_y)
	{
		int base_x = 0;
		// strips x
		for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
		{
			fdwt_vert_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x,       // base_x
				base_y,       // base_y
				base_x+strip_size_x, // stop_y
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last x
		{
			fdwt_vert_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x,       // base_x
				base_y,      // base_y
				size_x,              // stop_y
				base_y+strip_size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
	// last y
	{
		int base_x = 0;
		// strips x
		for( ; base_x+strip_size_x <= size_x; base_x+=strip_size_x)
		{
			fdwt_vert_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				base_x+strip_size_x, // stop_x
				size_y,              // stop_y
				buffer_y,
				buffer_x
			);
		}
		// last x
		{
			fdwt_vert_2x2_cor_HORIZ(
				ptr,
				stride_x,
				stride_y,
				base_x, // base_x
				base_y, // base_y
				size_x, // stop_x
				size_y, // stop_y
				buffer_y,
				buffer_x
			);
		}
	}
}

fdwt_diag_2x2_func_t get_fdwt_diag_2x2_func(enum order order)
{
	assert( order < ORDER_LAST );

	fdwt_diag_2x2_func_t fdwt_diag_2x2_func[ORDER_LAST] = {
		// 2x2
		[ORDER_HORIZ]        = fdwt_diag_2x2_HORIZ,
		[ORDER_VERT]         = fdwt_diag_2x2_VERT,
		[ORDER_HORIZ_STRIPS] = fdwt_diag_2x2_HORIZ_STRIPS,
		[ORDER_VERT_STRIPS]  = fdwt_diag_2x2_VERT_STRIPS,
		[ORDER_HORIZ_BLOCKS] = fdwt_diag_2x2_HORIZ_BLOCK,
		[ORDER_VERT_BLOCKS]  = fdwt_diag_2x2_VERT_BLOCK,
		// fused
		[ORDER_HORIZ_6X2]    = fdwt_diag_6x2_HORIZ,
		[ORDER_HORIZ_2X6]    = fdwt_diag_2x6_HORIZ,
		[ORDER_HORIZ_6X6]    = fdwt_diag_6x6_HORIZ,
	};

	return fdwt_diag_2x2_func[order];
}

fdwt_vert_2x2_func_t get_fdwt_vert_2x2_func(enum order order)
{
	assert( order < ORDER_LAST );

	fdwt_vert_2x2_func_t fdwt_vert_2x2_func[ORDER_LAST] = {
		// 2x2
		[ORDER_HORIZ]        = fdwt_vert_2x2_HORIZ,
		[ORDER_VERT]         = fdwt_vert_2x2_VERT,
		[ORDER_HORIZ_STRIPS] = fdwt_vert_2x2_HORIZ_STRIPS,
		[ORDER_VERT_STRIPS]  = fdwt_vert_2x2_VERT_STRIPS,
		[ORDER_HORIZ_BLOCKS] = fdwt_vert_2x2_HORIZ_BLOCK,
		[ORDER_VERT_BLOCKS]  = fdwt_vert_2x2_VERT_BLOCK,
		// fused
		[ORDER_HORIZ_4X4]    = fdwt_vert_4x4_HORIZ,
		[ORDER_HORIZ_8X2]    = fdwt_vert_8x2_HORIZ,
		[ORDER_HORIZ_2X8]    = fdwt_vert_2x8_HORIZ,
		[ORDER_HORIZ_8X8]    = fdwt_vert_8x8_HORIZ,
	};

	return fdwt_vert_2x2_func[order];
}
