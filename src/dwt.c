#include "dwt.h"

#include "libdwt.h"

#ifdef __SSE__
	#include <xmmintrin.h>
#endif

#ifdef _OPENMP
	#include <omp.h>
#endif

#include <assert.h>
#include <math.h>
#define MEASURE_FACTOR 1
#define MEASURE_PER_PIXEL

#include "inline.h"

#ifdef __SSE__
#define op4s_sdl2_update_s_sse(c, l, r, z) \
do { \
	(c) = (l); \
	(l) = (r); \
	(r) = (z); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_sdl2_shuffle_input_low_s_sse(in, c, r) \
do { \
	__m128 t; \
	(t) = (in); \
	(t) = _mm_shuffle_ps((t), (c), _MM_SHUFFLE(3,2,1,0)); \
	(c) = _mm_shuffle_ps((c), (t), _MM_SHUFFLE(0,3,2,1)); \
	(t) = _mm_shuffle_ps((t), (r), _MM_SHUFFLE(3,2,1,0)); \
	(r) = _mm_shuffle_ps((r), (t), _MM_SHUFFLE(1,3,2,1)); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_sdl2_shuffle_input_high_s_sse(in, c, r) \
do { \
	(in) = _mm_shuffle_ps( (in), (c), _MM_SHUFFLE(3,2,3,2) ); \
	(c)  = _mm_shuffle_ps( (c), (in), _MM_SHUFFLE(0,3,2,1) ); \
	(in) = _mm_shuffle_ps( (in), (r), _MM_SHUFFLE(3,2,1,0) ); \
	(r)  = _mm_shuffle_ps( (r), (in), _MM_SHUFFLE(1,3,2,1) ); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_sdl2_op_s_sse(z, c, w, l, r) \
do { \
	(z) = (l); \
	(z) = _mm_add_ps((z), (r)); \
	(z) = _mm_mul_ps((z), (w)); \
	(z) = _mm_add_ps((z), (c)); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_sdl2_output_low_s_sse(out, l, z) \
do { \
	(out) = (l); \
	(out) = _mm_unpacklo_ps((out), (z)); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_sdl2_output_high_s_sse(out, l, z) \
do { \
	__m128 t; \
	(t) = (l); \
	(t) = _mm_unpacklo_ps((t), (z)); \
	(out) = _mm_shuffle_ps((out), t, _MM_SHUFFLE(1,0,1,0)); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_sdl2_scale_s_sse(out, v) \
do { \
	(out) = _mm_mul_ps((out), (v)); \
} while(0)
#endif

static
void op4s_sdl2_update_s_ref(float *c, float *l, float *r, const float *z)
{
	c[0] = l[0];
	c[1] = l[1];
	c[2] = l[2];
	c[3] = l[3];

	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	r[0] = z[0];
	r[1] = z[1];
	r[2] = z[2];
	r[3] = z[3];
}

static
void op4s_sdl2_shuffle_s_ref(float *c, float *r)
{
	c[0]=c[1]; c[1]=c[2]; c[2]=c[3];
	r[0]=r[1]; r[1]=r[2]; r[2]=r[3];
}

static
void op4s_sdl2_input_low_s_ref(const float *in, float *c, float *r)
{
	c[3] = in[0];
	r[3] = in[1];
}

static
void op4s_sdl2_input_high_s_ref(const float *in, float *c, float *r)
{
	c[3] = in[2];
	r[3] = in[3];
}

static
void op4s_sdl2_shuffle_input_low_s_ref(const float *in, float *c, float *r)
{
	op4s_sdl2_shuffle_s_ref(c, r);
	op4s_sdl2_input_low_s_ref(in, c, r);
}

static
void op4s_sdl2_shuffle_input_high_s_ref(const float *in, float *c, float *r)
{
	op4s_sdl2_shuffle_s_ref(c, r);
	op4s_sdl2_input_high_s_ref(in, c, r);
}

static
void op4s_sdl2_op_s_ref(float *z, const float *c, const float *w, const float *l, const float *r)
{
	z[3] = c[3] + w[3] * ( l[3] + r[3] );
	z[2] = c[2] + w[2] * ( l[2] + r[2] );
	z[1] = c[1] + w[1] * ( l[1] + r[1] );
	z[0] = c[0] + w[0] * ( l[0] + r[0] );
}

static
void op4s_sdl2_output_low_s_ref(float *out, const float *l, const float *z)
{
	out[0] = l[0];
	out[1] = z[0];
}

static
void op4s_sdl2_output_high_s_ref(float *out, const float *l, const float *z)
{
	out[2] = l[0];
	out[3] = z[0];
}

static
void op4s_sdl2_scale_s_ref(float *out, const float *v)
{
	out[0] *= v[0];
	out[1] *= v[1];
	out[2] *= v[2];
	out[3] *= v[3];
}


int dwt_alg_shift[DWT_ALG_LAST] = {
	[DWT_ALG_SL_CORE_DL] = 4,
	[DWT_ALG_SL_CORE_DL_SSE] = 4,
	[DWT_ALG_SL_CORE_DL_SC] = 4,
	[DWT_ALG_SL_CORE_DL_SC_SSE] = 4,
	[DWT_ALG_SL_CORE_SDL] = 10,
	[DWT_ALG_SL_CORE_SDL_SSE] = 10,
	[DWT_ALG_SL_CORE_SDL_SC] = 10,
	[DWT_ALG_SL_CORE_SDL_SC_SSE] = 10,
	[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0] = 10,
	[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1] = 10,
	[DWT_ALG_SL_CORE_DL_SC_SSE_OFF1] = 4,
	[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0_OVL1] = 10,
	[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1_OVL1] = 10,
};

int dwt_alg_get_shift(
	enum dwt_alg alg
)
{
	return dwt_alg_shift[alg];
}

static
void cdf97_fwd_core2_sdl_2x2_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outL1,
	float *outR0, float *outR1,
	float *lAL, float *cAL, float *rAL,
	float *lAR, float *cAR, float *rAR,
	float *lBL, float *cBL, float *rBL,
	float *lBR, float *cBR, float *rBR
)
{
	UNUSED(cAL);
	UNUSED(rAL);
	UNUSED(cAR);
	UNUSED(rAR);
	UNUSED(cBL);
	UNUSED(rBL);
	UNUSED(cBR);
	UNUSED(rBR);

	const float w[4] = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const float v[4] = { 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s, 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s };

	float buff[4];
	float z[4];

	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	// A/L+R
	op4s_sdl2_shuffle_input_low_s_ref(buff, (lAL+4), (lAL+8));
	op4s_sdl2_shuffle_input_high_s_ref(buff, (lAR+4), (lAR+8));

	// A/L
	op4s_sdl2_op_s_ref(z, (lAL+4), w, (lAL+0), (lAL+8));
	op4s_sdl2_output_low_s_ref(buff, (lAL+0), z);
	op4s_sdl2_update_s_ref((lAL+4), (lAL+0), (lAL+8), z);

	// A/R
	op4s_sdl2_op_s_ref(z, (lAR+4), w, (lAR+0), (lAR+8));
	op4s_sdl2_output_high_s_ref(buff, (lAR+0), z);
	op4s_sdl2_update_s_ref((lAR+4), (lAR+0), (lAR+8), z);

	// A/L+R
	op4s_sdl2_scale_s_ref(buff, v);

	// swap, this should by done by single shuffle instruction
	float tmp = buff[1]; buff[1] = buff[2]; buff[2] = tmp;

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_ref(buff, (lBL+4), (lBL+8));
	op4s_sdl2_shuffle_input_high_s_ref(buff, (lBR+4), (lBR+8));

	// B/L
	op4s_sdl2_op_s_ref(z, (lBL+4), w, (lBL+0), (lBL+8));
	op4s_sdl2_output_low_s_ref(buff, (lBL+0), z);
	op4s_sdl2_update_s_ref((lBL+4), (lBL+0), (lBL+8), z);

	// B/R
	op4s_sdl2_op_s_ref(z, (lBR+4), w, (lBR+0), (lBR+8));
	op4s_sdl2_output_high_s_ref(buff, (lBR+0), z); 
	op4s_sdl2_update_s_ref((lBR+4), (lBR+0), (lBR+8), z);

	// B/L+R
	op4s_sdl2_scale_s_ref(buff, v);

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
}

#ifdef __SSE__
static
void cdf97_fwd_core2_sdl_2x2_sse_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outL1,
	float *outR0, float *outR1,
	float *lAL, float *cAL, float *rAL,
	float *lAR, float *cAR, float *rAR,
	float *lBL, float *cBL, float *rBL,
	float *lBR, float *cBR, float *rBR
)
{
	UNUSED(cAL);
	UNUSED(rAL);
	UNUSED(cAR);
	UNUSED(rAR);
	UNUSED(cBL);
	UNUSED(rBL);
	UNUSED(cBR);
	UNUSED(rBR);

	const float w[4] __attribute__ ((aligned (16))) = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const float v[4] __attribute__ ((aligned (16))) = { 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s, 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s };

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
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAL+4), *(__m128 *)w, *(__m128 *)(lAL+0), *(__m128 *)(lAL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lAL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAL+4), *(__m128 *)(lAL+0), *(__m128 *)(lAL+8), z);

	// A/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAR+4), *(__m128 *)w, *(__m128 *)(lAR+0), *(__m128 *)(lAR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lAR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAR+4), *(__m128 *)(lAR+0), *(__m128 *)(lAR+8), z);

	// A/L+R
	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lBL+4), *(__m128 *)(lBL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lBR+4), *(__m128 *)(lBR+8));

	// B/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBL+4), *(__m128 *)w, *(__m128 *)(lBL+0), *(__m128 *)(lBL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lBL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lBL+4), *(__m128 *)(lBL+0), *(__m128 *)(lBL+8), z);

	// B/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBR+4), *(__m128 *)w, *(__m128 *)(lBR+0), *(__m128 *)(lBR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lBR+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(lBR+4), *(__m128 *)(lBR+0), *(__m128 *)(lBR+8), z);

	// B/L+R
	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core2_sdl_2x2_sc_sse_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outL1,
	float *outR0, float *outR1,
	float *lAL, float *cAL, float *rAL,
	float *lAR, float *cAR, float *rAR,
	float *lBL, float *cBL, float *rBL,
	float *lBR, float *cBR, float *rBR
)
{
	UNUSED(cAL);
	UNUSED(rAL);
	UNUSED(cAR);
	UNUSED(rAR);
	UNUSED(cBL);
	UNUSED(rBL);
	UNUSED(cBR);
	UNUSED(rBR);

	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v_vert = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f, 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s) };

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
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core2prolog_sdl_2x2_sc_sse_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outL1,
	float *outR0, float *outR1,
	float *lAL, float *cAL, float *rAL,
	float *lAR, float *cAR, float *rAR,
	float *lBL, float *cBL, float *rBL,
	float *lBR, float *cBR, float *rBR
)
{
	UNUSED(outL0);
	UNUSED(outL1);
	UNUSED(outR0);
	UNUSED(outR1);
	UNUSED(cAL);
	UNUSED(rAL);
	UNUSED(cAR);
	UNUSED(rAR);
	UNUSED(cBL);
	UNUSED(rBL);
	UNUSED(cBR);
	UNUSED(rBR);

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
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core_sdl_2x2_sse_s(
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
)
{
	cdf97_fwd_core2_sdl_2x2_sse_s(
		ptr_y0_x0, ptr_y0_x1,
		ptr_y1_x0, ptr_y1_x1,
		out_y0_x0, out_y1_x0,
		out_y0_x1, out_y1_x1,
		buff_y0+0, buff_y0+4, buff_y0+8,
		buff_y1+0, buff_y1+4, buff_y1+8,
		buff_x0+0, buff_x0+4, buff_x0+8,
		buff_x1+0, buff_x1+4, buff_x1+8
	);
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core_sdl_2x2_sc_sse_s(
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
)
{
	cdf97_fwd_core2_sdl_2x2_sc_sse_s(
		ptr_y0_x0, ptr_y0_x1,
		ptr_y1_x0, ptr_y1_x1,
		out_y0_x0, out_y1_x0,
		out_y0_x1, out_y1_x1,
		buff_y0+0, buff_y0+4, buff_y0+8,
		buff_y1+0, buff_y1+4, buff_y1+8,
		buff_x0+0, buff_x0+4, buff_x0+8,
		buff_x1+0, buff_x1+4, buff_x1+8
	);
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core_prolog_sdl_2x2_sc_sse_s(
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
)
{
	cdf97_fwd_core2prolog_sdl_2x2_sc_sse_s(
		ptr_y0_x0, ptr_y0_x1,
		ptr_y1_x0, ptr_y1_x1,
		out_y0_x0, out_y1_x0,
		out_y0_x1, out_y1_x1,
		buff_y0+0, buff_y0+4, buff_y0+8,
		buff_y1+0, buff_y1+4, buff_y1+8,
		buff_x0+0, buff_x0+4, buff_x0+8,
		buff_x1+0, buff_x1+4, buff_x1+8
	);
}
#endif

static
void cdf97_fwd_core_sdl_2x2_s(
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
)
{
	cdf97_fwd_core2_sdl_2x2_s(
		ptr_y0_x0, ptr_y0_x1,
		ptr_y1_x0, ptr_y1_x1,
		out_y0_x0, out_y1_x0,
		out_y0_x1, out_y1_x1,
		buff_y0+0, buff_y0+4, buff_y0+8,
		buff_y1+0, buff_y1+4, buff_y1+8,
		buff_x0+0, buff_x0+4, buff_x0+8,
		buff_x1+0, buff_x1+4, buff_x1+8
	);
}

// SL SDL WITH-MERGED-SCALING SSE
#ifdef __SSE__
void cdf97_fwd_core_sdl_sc_sse_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	const int offset = 1;

	float short_buffer_0[3*4] __attribute__ ((aligned (16)));
	float short_buffer_1[3*4] __attribute__ ((aligned (16)));

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{

		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sc_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
}
#endif

// SL SDL WITH*MERGED-SCALING SSE W/O OFFSET
#ifdef __SSE__
void cdf97_fwd_core_sdl_sc_sse_off0_inner_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	const int offset = 0;

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		float short_buffer_0[3*4] __attribute__ ((aligned (16)));
		float short_buffer_1[3*4] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sc_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
#pragma omp barrier
	}
}
#endif

// SL SDL WITH*MERGED-SCALING SSE OFFSET=1
#ifdef __SSE__
void cdf97_fwd_core_sdl_sc_sse_off1_inner_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	const int offset = 1;

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		float short_buffer_0[3*4] __attribute__ ((aligned (16)));
		float short_buffer_1[3*4] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sc_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
#pragma omp barrier
	}
}
#endif

// SL SDL WITH*MERGED-SCALING SSE W/O OFFSET
#ifdef __SSE__
void cdf97_fwd_core_sdl_sc_sse_off0_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
#if 0
	const int offset = 0;

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		float short_buffer_0[3*4] __attribute__ ((aligned (16)));
		float short_buffer_1[3*4] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sc_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
#else
	const int offset = 0;

	int threads = dwt_util_get_num_threads(); // 8, 4

	// assert( is_even(size_y) );

	// FIXME: really needs to be even?
	int thread_size_y = dwt_util_up_to_even( ceil_div(size_y, threads) );

	const int prolog_y = 10+4;
	const int overlay_y = 4;

	//#pragma omp parallel for
	//for(int thread = 0; thread < threads; thread++)
	#pragma omp parallel num_threads(threads)
	{
		int thread = omp_get_thread_num();

		void *thread_src = addr2_s(src, /*y*/thread*thread_size_y, offset, src_stride_x, src_stride_y);
		void *thread_dst = addr2_s(dst, /*y*/thread*thread_size_y, offset, dst_stride_x, dst_stride_y);
		void *thread_dst_null = NULL;
		char thread_dst_buff[(offset+overlay_y)*dst_stride_x]; // FIXME: unnecesarry when no overlay used

#if 0
		cdf97_fwd_core_sdl_sc_sse_off0_inner_s(
			thread_src,
			thread_dst,
			src_stride_x,
			src_stride_y,
			dst_stride_x,
			dst_stride_y,
			size_x,
			thread_size_y+overlay_y
		);
#else
		float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

		#pragma omp barrier

		for(int y = offset+0-prolog_y; y+1 < offset+0; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_null, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_null, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_prolog_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+overlay_y; y+1 < offset+thread_size_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *ptr1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				*addr1_s(out0_x, 0, dst_stride_y) = *addr1_s(ptr0_x, 0, src_stride_y);
				*addr1_s(out0_x, 1, dst_stride_y) = *addr1_s(ptr0_x, 1, src_stride_y);
				*addr1_s(out1_x, 0, dst_stride_y) = *addr1_s(ptr1_x, 0, src_stride_y);
				*addr1_s(out1_x, 1, dst_stride_y) = *addr1_s(ptr1_x, 1, src_stride_y);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}
#endif
	}
#endif
}
#endif

// SL SDL WITH*MERGED-SCALING SSE OFFSET=0 OVERLAY=1
#ifdef __SSE__
void cdf97_fwd_core_sdl_sc_sse_off0_ovl1_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
#if 0
	const int offset = 0;

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		float short_buffer_0[3*4] __attribute__ ((aligned (16)));
		float short_buffer_1[3*4] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sc_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
#else
	const int offset = 0;

	int threads = dwt_util_get_num_threads(); // 8, 4

	// assert( is_even(size_y) );

	// FIXME: really needs to be even?
	int thread_size_y = dwt_util_up_to_even( ceil_div(size_y, threads) );

	const int prolog_y = 10+4;
	const int overlay_y = 0;

	//#pragma omp parallel for
	//for(int thread = 0; thread < threads; thread++)
	#pragma omp parallel num_threads(threads)
	{
		int thread = omp_get_thread_num();

		void *thread_src = addr2_s(src, /*y*/thread*thread_size_y, offset, src_stride_x, src_stride_y);
		void *thread_dst = addr2_s(dst, /*y*/thread*thread_size_y, offset, dst_stride_x, dst_stride_y);
		void *thread_dst_null = NULL;
		char thread_dst_buff[(offset+overlay_y)*dst_stride_x];

#if 0
		cdf97_fwd_core_sdl_sc_sse_off0_inner_s(
			thread_src,
			thread_dst,
			src_stride_x,
			src_stride_y,
			dst_stride_x,
			dst_stride_y,
			size_x,
			thread_size_y+overlay_y
		);
#else
		float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

		#pragma omp barrier

		for(int y = offset+0-prolog_y; y+1 < offset+0; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_null, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_null, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_prolog_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+overlay_y; y+1 < offset+thread_size_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *ptr1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				*addr1_s(out0_x, 0, dst_stride_y) = *addr1_s(ptr0_x, 0, src_stride_y);
				*addr1_s(out0_x, 1, dst_stride_y) = *addr1_s(ptr0_x, 1, src_stride_y);
				*addr1_s(out1_x, 0, dst_stride_y) = *addr1_s(ptr1_x, 0, src_stride_y);
				*addr1_s(out1_x, 1, dst_stride_y) = *addr1_s(ptr1_x, 1, src_stride_y);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}
#endif
	}
#endif
}
#endif

// SL SDL WITH*MERGED-SCALING SSE OFFSET=1 THREADS
#ifdef __SSE__
void cdf97_fwd_core_sdl_sc_sse_off1_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
#if 0
	// NOTE: this works well

	const int offset = 1;

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		float short_buffer_0[3*4] __attribute__ ((aligned (16)));
		float short_buffer_1[3*4] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sc_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
#else
	const int offset = 1;

	int threads = dwt_util_get_num_threads(); // 8, 4

	// assert( is_even(size_y) );

	int thread_size_y = dwt_util_up_to_even( ceil_div(size_y, threads) );

	const int prolog_y = 10+4;
	const int overlay_y = 4;

	//#pragma omp parallel for
	//for(int thread = 0; thread < threads; thread++)
	#pragma omp parallel num_threads(threads)
	{
		int thread = omp_get_thread_num();

		void *thread_src = addr2_s(src, /*y*/thread*thread_size_y, 0/*offset*/, src_stride_x, src_stride_y);
		void *thread_dst = addr2_s(dst, /*y*/thread*thread_size_y, 0/*offset*/, dst_stride_x, dst_stride_y);
		void *thread_dst_null = NULL;
		char thread_dst_buff[(offset+overlay_y)*dst_stride_x];

#if 0
		cdf97_fwd_core_sdl_sc_sse_off1_inner_s(
			thread_src,
			thread_dst,
			src_stride_x,
			src_stride_y,
			dst_stride_x,
			dst_stride_y,
			size_x,
			thread_size_y+overlay_y
		);
#else
		float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

		#pragma omp barrier

		for(int y = offset+0-prolog_y; y+1 < offset+0; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_null, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_null, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_prolog_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+overlay_y; y+1 < offset+thread_size_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *ptr1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				*addr1_s(out0_x, 0, dst_stride_y) = *addr1_s(ptr0_x, 0, src_stride_y);
				*addr1_s(out0_x, 1, dst_stride_y) = *addr1_s(ptr0_x, 1, src_stride_y);
				*addr1_s(out1_x, 0, dst_stride_y) = *addr1_s(ptr1_x, 0, src_stride_y);
				*addr1_s(out1_x, 1, dst_stride_y) = *addr1_s(ptr1_x, 1, src_stride_y);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}
#endif
	}
#endif
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core_dl_sc_sse_2x2_s(
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
);
#endif

#ifdef __SSE__
static
void cdf97_fwd_core_prolog_dl_sc_sse_2x2_s(
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
);
#endif

// SL SDL WITH*MERGED-SCALING SSE OFFSET=1 THREADS
#ifdef __SSE__
void cdf97_fwd_core_dl_sc_sse_off1_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
#if 0
	// NOTE: this works well
	const int offset = 1;

	const int words = 1;

	float long_buffer[4*words*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 4*words*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(words*4)*(offset+0);

		float short_buffer_0[words*4] __attribute__ ((aligned (16)));
		float short_buffer_1[words*4] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(short_buffer_0, words*4);
		dwt_util_zero_vec_s(short_buffer_1, words*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_dl_sc_sse_2x2_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(words*4),
				long_buffer_ptr+1*(words*4)
			);

			long_buffer_ptr += 2*(words*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
#else
	const int offset = 1;

	const int words = 1;

	int threads = dwt_util_get_num_threads(); // 8, 4

	// assert( is_even(size_y) );

	int thread_size_y = dwt_util_up_to_even( ceil_div(size_y, threads) );

	const int prolog_y = 4+4;
	const int overlay_y = 4;

	//#pragma omp parallel for
	//for(int thread = 0; thread < threads; thread++)
	#pragma omp parallel num_threads(threads)
	{
		int thread = omp_get_thread_num();

		void *thread_src = addr2_s(src, /*y*/thread*thread_size_y, 0/*offset*/, src_stride_x, src_stride_y);
		void *thread_dst = addr2_s(dst, /*y*/thread*thread_size_y, 0/*offset*/, dst_stride_x, dst_stride_y);
		void *thread_dst_null = NULL;
		char thread_dst_buff[(offset+overlay_y)*dst_stride_x];

#if 0
		cdf97_fwd_core_sdl_sc_sse_off1_inner_s(
			thread_src,
			thread_dst,
			src_stride_x,
			src_stride_y,
			dst_stride_x,
			dst_stride_y,
			size_x,
			thread_size_y+overlay_y
		);
#else
		float long_buffer[4*words*size_x] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(long_buffer, words*4*size_x);

		#pragma omp barrier

		for(int y = offset+0-prolog_y; y+1 < offset+0; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_null, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_null, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(words*4)*(offset+0);

			float short_buffer_0[words*4] __attribute__ ((aligned (16)));
			float short_buffer_1[words*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, words*4);
			dwt_util_zero_vec_s(short_buffer_1, words*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_prolog_dl_sc_sse_2x2_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(words*4),
					long_buffer_ptr+1*(words*4)
				);

				long_buffer_ptr += 2*(words*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(words*4)*(offset+0);

			float short_buffer_0[words*4] __attribute__ ((aligned (16)));
			float short_buffer_1[words*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, words*4);
			dwt_util_zero_vec_s(short_buffer_1, words*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_dl_sc_sse_2x2_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(words*4),
					long_buffer_ptr+1*(words*4)
				);

				long_buffer_ptr += 2*(words*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+overlay_y; y+1 < offset+thread_size_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(words*4)*(offset+0);

			float short_buffer_0[words*4] __attribute__ ((aligned (16)));
			float short_buffer_1[words*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, words*4);
			dwt_util_zero_vec_s(short_buffer_1, words*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_dl_sc_sse_2x2_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(words*4),
					long_buffer_ptr+1*(words*4)
				);

				long_buffer_ptr += 2*(words*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *ptr1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				*addr1_s(out0_x, 0, dst_stride_y) = *addr1_s(ptr0_x, 0, src_stride_y);
				*addr1_s(out0_x, 1, dst_stride_y) = *addr1_s(ptr0_x, 1, src_stride_y);
				*addr1_s(out1_x, 0, dst_stride_y) = *addr1_s(ptr1_x, 0, src_stride_y);
				*addr1_s(out1_x, 1, dst_stride_y) = *addr1_s(ptr1_x, 1, src_stride_y);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}
#endif
	}
#endif
}
#endif

#ifdef __SSE__
void cdf97_fwd_core_sdl_sc_sse_off1_ovl1_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
#if 0
	// NOTE: this works well
	const int offset = 1;

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		float short_buffer_0[3*4] __attribute__ ((aligned (16)));
		float short_buffer_1[3*4] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sc_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
#else
	const int offset = 1;

	int threads = dwt_util_get_num_threads(); // 8, 4

	// assert( is_even(size_y) );

	int thread_size_y = dwt_util_up_to_even( ceil_div(size_y, threads) );

	const int prolog_y = 10+4;
	const int overlay_y = 0;

	//#pragma omp parallel for
	//for(int thread = 0; thread < threads; thread++)
	#pragma omp parallel num_threads(threads)
	{
		int thread = omp_get_thread_num();

		void *thread_src = addr2_s(src, /*y*/thread*thread_size_y, 0/*offset*/, src_stride_x, src_stride_y);
		void *thread_dst = addr2_s(dst, /*y*/thread*thread_size_y, 0/*offset*/, dst_stride_x, dst_stride_y);
		void *thread_dst_null = NULL;
		char thread_dst_buff[(offset+overlay_y)*dst_stride_x];

#if 0
		cdf97_fwd_core_sdl_sc_sse_off1_inner_s(
			thread_src,
			thread_dst,
			src_stride_x,
			src_stride_y,
			dst_stride_x,
			dst_stride_y,
			size_x,
			thread_size_y+overlay_y
		);
#else
		float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

		dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

		#pragma omp barrier

		for(int y = offset+0-prolog_y; y+1 < offset+0; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_null, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_null, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_prolog_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+overlay_y; y+1 < offset+thread_size_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_src, y+0, offset, src_stride_x, src_stride_y);
			float *ptr1_x = addr2_s(thread_src, y+1, offset, src_stride_x, src_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

			float short_buffer_0[3*4] __attribute__ ((aligned (16)));
			float short_buffer_1[3*4] __attribute__ ((aligned (16)));

			dwt_util_zero_vec_s(short_buffer_0, 3*4);
			dwt_util_zero_vec_s(short_buffer_1, 3*4);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				cdf97_fwd_core_sdl_2x2_sc_sse_s(
					addr1_s(ptr0_x, 0, src_stride_y),
					addr1_s(ptr0_x, 1, src_stride_y),
					addr1_s(ptr1_x, 0, src_stride_y),
					addr1_s(ptr1_x, 1, src_stride_y),
					addr1_s(out0_x, 0, dst_stride_y),
					addr1_s(out0_x, 1, dst_stride_y),
					addr1_s(out1_x, 0, dst_stride_y),
					addr1_s(out1_x, 1, dst_stride_y),
					short_buffer_0,
					short_buffer_1,
					long_buffer_ptr+0*(3*4),
					long_buffer_ptr+1*(3*4)
				);

				long_buffer_ptr += 2*(3*4);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}

		#pragma omp barrier

		for(int y = offset+0; y+1 < offset+overlay_y; y += 2)
		{
			float *ptr0_x = addr2_s(thread_dst_buff, y+0, offset, dst_stride_x, dst_stride_y);
			float *ptr1_x = addr2_s(thread_dst_buff, y+1, offset, dst_stride_x, dst_stride_y);
			float *out0_x = addr2_s(thread_dst, y+0, offset, dst_stride_x, dst_stride_y);
			float *out1_x = addr2_s(thread_dst, y+1, offset, dst_stride_x, dst_stride_y);

			for(int x = 0+offset; x+1 < size_x; x += 2)
			{
				*addr1_s(out0_x, 0, dst_stride_y) = *addr1_s(ptr0_x, 0, src_stride_y);
				*addr1_s(out0_x, 1, dst_stride_y) = *addr1_s(ptr0_x, 1, src_stride_y);
				*addr1_s(out1_x, 0, dst_stride_y) = *addr1_s(ptr1_x, 0, src_stride_y);
				*addr1_s(out1_x, 1, dst_stride_y) = *addr1_s(ptr1_x, 1, src_stride_y);

				ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
				ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
				out0_x = addr1_s(out0_x, +2, dst_stride_y);
				out1_x = addr1_s(out1_x, +2, dst_stride_y);
			}
		}
#endif
	}
#endif
}
#endif

// SL SDL NO-MERGED-SCALING SSE
#ifdef __SSE__
void cdf97_fwd_core_sdl_sse_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	const int offset = 1;

	float short_buffer_0[3*4] __attribute__ ((aligned (16)));
	float short_buffer_1[3*4] __attribute__ ((aligned (16)));

	float long_buffer[4*3*size_x] __attribute__ ((aligned (16)));

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{

		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_sse_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
}
#endif

// SL SDL NO-MERGED-SCALING
void cdf97_fwd_core_sdl_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	// forward transform with offset = 1
	const int offset = 1;

	float short_buffer_0[3*4];
	float short_buffer_1[3*4];

	float long_buffer[4*3*size_x];

	dwt_util_zero_vec_s(long_buffer, 3*4*size_x);

	// filter src into dst using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr0_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr1_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out0_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out1_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer+(3*4)*(offset+0);

		dwt_util_zero_vec_s(short_buffer_0, 3*4);
		dwt_util_zero_vec_s(short_buffer_1, 3*4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_sdl_2x2_s(
				addr1_s(ptr0_x, 0, src_stride_y),
				addr1_s(ptr0_x, 1, src_stride_y),
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(out0_x, 0, dst_stride_y),
				addr1_s(out0_x, 1, dst_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				short_buffer_0,
				short_buffer_1,
				long_buffer_ptr+0*(3*4),
				long_buffer_ptr+1*(3*4)
			);

			long_buffer_ptr += 2*(3*4);

			ptr0_x = addr1_s(ptr0_x, +2, src_stride_y);
			ptr1_x = addr1_s(ptr1_x, +2, src_stride_y);
			out0_x = addr1_s(out0_x, +2, dst_stride_y);
			out1_x = addr1_s(out1_x, +2, dst_stride_y);
		}
	}
}

static
void accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
	const float *ptr0,
	const float *ptr1,
	float *out0,
	float *out1,
	const float *w,
	const float *v,
	float *l // [4]
)
{
	// aux. variables
	float x[2];
	float y[2];
	float r[4];
	float c[4];

	// inputs
	x[0] = *ptr0;
	x[1] = *ptr1;

	// shuffles
	y[0] = l[0];
	c[0] = l[1];
	c[1] = l[2];
	c[2] = l[3];
	c[3] = x[0];

	// operation z[] = c[] + w[] * ( l[] + r[] )
	// by sequential computation from top/right to bottom/left
	r[3] = x[1];
	r[2] = c[3]+w[3]*(l[3]+r[3]);
	r[1] = c[2]+w[2]*(l[2]+r[2]);
	r[0] = c[1]+w[1]*(l[1]+r[1]);
	y[1] = c[0]+w[0]*(l[0]+r[0]);

	// scales
	y[0] = y[0] * v[0];
	y[1] = y[1] * v[1];

	// outputs
	*out0 = y[0];
	*out1 = y[1];

	// update l[]
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];
}

static
void accel_lift_op4s_uni_main_dl_stride_pair_core_s(
	const float *ptr0,
	const float *ptr1,
	float *out0,
	float *out1,
	float *w,
	float *v,
	float *l // [4]
)
{
	UNUSED(v);

	// aux. variables
	float x[2];
	float y[2];
	float r[4];
	float c[4];

	// inputs
	x[0] = *ptr0;
	x[1] = *ptr1;

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
	*out0 = y[0];
	*out1 = y[1];

	// update l[]
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];
}

static
void cdf97_fwd_core_dl_2x2_s(
	const float *ptr_y0_x0, // in
	const float *ptr_y0_x1, // in
	const float *ptr_y1_x0, // in
	const float *ptr_y1_x1, // in
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
#if 1
	const float w[4] = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const float v[4] = { 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s, 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s };

	float tmp[4];
	float *tmp_y0_x0 = tmp+0;
	float *tmp_y0_x1 = tmp+1;
	float *tmp_y1_x0 = tmp+2;
	float *tmp_y1_x1 = tmp+3;

	// horizontal 0
	// [y+0, x+0], [y+0, x+1] => [y+0, x+0-4], [y+0, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		ptr_y0_x0,
		ptr_y0_x1,
		tmp_y0_x0,
		tmp_y0_x1,
		w,
		v,
		buff_h0
	);

	// horizontal 1
	// [y+1, x+0], [y+1, x+1] => [y+1, x+0-4], [y+1, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		ptr_y1_x0,
		ptr_y1_x1,
		tmp_y1_x0,
		tmp_y1_x1,
		w,
		v,
		buff_h1
	);

	// vertical 0
	// [y+0, x+0-4] [y+1, x+0-4] => [y+0-4, x+0-4] [y+1-4, x+0-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x0,
		tmp_y1_x0,
		out_y0_x0,
		out_y1_x0,
		w,
		v,
		buff_v0
	);

	// vertical 1
	// [y+0, x+1-4] [y+1, x+1-4] => [y+0-4, x+1-4] [y+1-4, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x1,
		tmp_y1_x1,
		out_y0_x1,
		out_y1_x1,
		w,
		v,
		buff_v1
	);
#else
	const float w[4] = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const float v[4] = { 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s, 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s };

	// temp
	float t[4];

	// aux. variables
	float x[4], y[4], r[4], c[4];

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

		// scaling
		y[0] *= v[0];
		y[1] *= v[1];

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

		// scaling
		y[0] *= v[0];
		y[1] *= v[1];

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
		y[0] *= v[0];
		y[1] *= v[1];

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
		y[0] *= v[0];
		y[1] *= v[1];

		// outputs
		*out_y0_x1 = y[0];
		*out_y1_x1 = y[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}
#endif
}

static
void cdf97_fwd_core_dl_2x2_sse_s(
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
#if 0
	const float w[4] = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const float v[4] = { 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s, 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s };

	float tmp[4];
	float *tmp_y0_x0 = tmp+0;
	float *tmp_y0_x1 = tmp+1;
	float *tmp_y1_x0 = tmp+2;
	float *tmp_y1_x1 = tmp+3;

	// horizontal 0
	// [y+0, x+0], [y+0, x+1] => [y+0, x+0-4], [y+0, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		ptr_y0_x0,
		ptr_y0_x1,
		tmp_y0_x0,
		tmp_y0_x1,
		w,
		v,
		buff_h0
	);

	// horizontal 1
	// [y+1, x+0], [y+1, x+1] => [y+1, x+0-4], [y+1, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		ptr_y1_x0,
		ptr_y1_x1,
		tmp_y1_x0,
		tmp_y1_x1,
		w,
		v,
		buff_h1
	);

	// vertical 0
	// [y+0, x+0-4] [y+1, x+0-4] => [y+0-4, x+0-4] [y+1-4, x+0-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x0,
		tmp_y1_x0,
		out_y0_x0,
		out_y1_x0,
		w,
		v,
		buff_v0
	);

	// vertical 1
	// [y+0, x+1-4] [y+1, x+1-4] => [y+0-4, x+1-4] [y+1-4, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x1,
		tmp_y1_x1,
		out_y0_x1,
		out_y1_x1,
		w,
		v,
		buff_v1
	);
#else
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v = { 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s, 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s };

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

		// scaling
		y[0] *= v[0];
		y[1] *= v[1];

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
		y[2] *= v[2];
		y[3] *= v[3];

		// outputs
		t[2] = y[2];
		t[3] = y[3];

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
		y[0] *= v[0];
		y[1] *= v[1];

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
		y[2] *= v[2];
		y[3] *= v[3];

		// outputs
		*out_y0_x1 = y[2];
		*out_y1_x1 = y[3];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}
#endif
}

static
void cdf97_fwd_core_dl_sc_2x2_s(
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
#if 0
	const float w[4] = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	// v_horiz can be elliminated
	const float v_horiz[4] = { 1.f, 1.f,
		0.f, 0.f };
	const float v_vertL[4] = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f,
		0.f, 0.f };
	const float v_vertR[4] = { 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s),
		0.f, 0.f };

	float tmp[4];
	float *tmp_y0_x0 = tmp+0;
	float *tmp_y0_x1 = tmp+1;
	float *tmp_y1_x0 = tmp+2;
	float *tmp_y1_x1 = tmp+3;

	// horizontal 0
	// [y+0, x+0], [y+0, x+1] => [y+0, x+0-4], [y+0, x+1-4]
	accel_lift_op4s_uni_main_dl_stride_pair_core_s(
		ptr_y0_x0,
		ptr_y0_x1,
		tmp_y0_x0,
		tmp_y0_x1,
		w,
		v_horiz,
		buff_h0
	);

	// horizontal 1
	// [y+1, x+0], [y+1, x+1] => [y+1, x+0-4], [y+1, x+1-4]
	accel_lift_op4s_uni_main_dl_stride_pair_core_s(
		ptr_y1_x0,
		ptr_y1_x1,
		tmp_y1_x0,
		tmp_y1_x1,
		w,
		v_horiz,
		buff_h1
	);

	// vertical 0
	// [y+0, x+0-4] [y+1, x+0-4] => [y+0-4, x+0-4] [y+1-4, x+0-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x0,
		tmp_y1_x0,
		out_y0_x0,
		out_y1_x0,
		w,
		v_vertL,
		buff_v0
	);

	// vertical 1
	// [y+0, x+1-4] [y+1, x+1-4] => [y+0-4, x+1-4] [y+1-4, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x1,
		tmp_y1_x1,
		out_y0_x1,
		out_y1_x1,
		w,
		v_vertR,
		buff_v1
	);
#else
	const float w[4] = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	const float v_vertL[4] = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f,
		0.f, 0.f };
	const float v_vertR[4] = { 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s),
		0.f, 0.f };

	// temp
	float t[4];

	// aux. variables
	float x[4], y[4], r[4], c[4];

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
#endif
}

#ifdef __SSE__
static
void cdf97_fwd_core_dl_sc_sse_2x2_s(
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
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core_prolog_dl_sc_sse_2x2_s(
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
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

// 	const __m128 v_vertL = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f,
// 		0.f, 0.f };
// 	const __m128 v_vertR = { 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s),
// 		0.f, 0.f };

	UNUSED(out_y0_x0);
	UNUSED(out_y0_x1);
	UNUSED(out_y1_x0);
	UNUSED(out_y1_x1);

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
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);

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
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}
}
#endif

// SL DL NO-MERGED-SCALING
void cdf97_fwd_core_dl_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	// forward transform with offset = 1
	int offset = 1;

	float short_buffer_0[4];
	float short_buffer_1[4];

	float long_buffer[4*size_x];

	dwt_util_zero_vec_s(long_buffer, 4 * size_x);

	// filter big into output using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr1_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr2_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out1_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out2_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer;

		dwt_util_zero_vec_s(short_buffer_0, 4);
		dwt_util_zero_vec_s(short_buffer_1, 4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_dl_2x2_s(
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(ptr2_x, 0, src_stride_y),
				addr1_s(ptr2_x, 1, src_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				addr1_s(out2_x, 0, dst_stride_y),
				addr1_s(out2_x, 1, dst_stride_y),
				short_buffer_0, // [4]
				short_buffer_1, // [4]
				long_buffer_ptr+0*4, // [4]
				long_buffer_ptr+1*4  // [4]
			);

			long_buffer_ptr += 2*4;

			ptr1_x = addr1_s(ptr1_x, 2, src_stride_y);
			ptr2_x = addr1_s(ptr2_x, 2, src_stride_y);
			out1_x = addr1_s(out1_x, 2, dst_stride_y);
			out2_x = addr1_s(out2_x, 2, dst_stride_y);
		}
	}
}

// SL DL NO-MERGED-SCALING SSE
void cdf97_fwd_core_dl_sse_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	// forward transform with offset = 1
	int offset = 1;

	float short_buffer_0[4];
	float short_buffer_1[4];

	float long_buffer[4*size_x];

	dwt_util_zero_vec_s(long_buffer, 4 * size_x);

	// filter big into output using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr1_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr2_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out1_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out2_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer;

		dwt_util_zero_vec_s(short_buffer_0, 4);
		dwt_util_zero_vec_s(short_buffer_1, 4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_dl_2x2_sse_s(
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(ptr2_x, 0, src_stride_y),
				addr1_s(ptr2_x, 1, src_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				addr1_s(out2_x, 0, dst_stride_y),
				addr1_s(out2_x, 1, dst_stride_y),
				short_buffer_0, // [4]
				short_buffer_1, // [4]
				long_buffer_ptr+0*4, // [4]
				long_buffer_ptr+1*4  // [4]
			);

			long_buffer_ptr += 2*4;

			ptr1_x = addr1_s(ptr1_x, 2, src_stride_y);
			ptr2_x = addr1_s(ptr2_x, 2, src_stride_y);
			out1_x = addr1_s(out1_x, 2, dst_stride_y);
			out2_x = addr1_s(out2_x, 2, dst_stride_y);
		}
	}
}

// SL DL WITH-MERGED-SCALING
void cdf97_fwd_core_dl_sc_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	// forward transform with offset = 1
	int offset = 1;

	float short_buffer_0[4];
	float short_buffer_1[4];

	float long_buffer[4*size_x];

	dwt_util_zero_vec_s(long_buffer, 4 * size_x);

	// filter big into output using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr1_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr2_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out1_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out2_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer;

		dwt_util_zero_vec_s(short_buffer_0, 4);
		dwt_util_zero_vec_s(short_buffer_1, 4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_dl_sc_2x2_s(
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(ptr2_x, 0, src_stride_y),
				addr1_s(ptr2_x, 1, src_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				addr1_s(out2_x, 0, dst_stride_y),
				addr1_s(out2_x, 1, dst_stride_y),
				short_buffer_0, // [4]
				short_buffer_1, // [4]
				long_buffer_ptr+0*4, // [4]
				long_buffer_ptr+1*4  // [4]
			);

			long_buffer_ptr += 2*4;

			ptr1_x = addr1_s(ptr1_x, 2, src_stride_y);
			ptr2_x = addr1_s(ptr2_x, 2, src_stride_y);
			out1_x = addr1_s(out1_x, 2, dst_stride_y);
			out2_x = addr1_s(out2_x, 2, dst_stride_y);
		}
	}
}

// SL DL WITH-MERGED-SCALING SSE
void cdf97_fwd_core_dl_sc_sse_s(
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	// forward transform with offset = 1
	int offset = 1;

	float short_buffer_0[4];
	float short_buffer_1[4];

	float long_buffer[4*size_x];

	dwt_util_zero_vec_s(long_buffer, 4 * size_x);

	// filter big into output using 2x2 core
	for(int y = 0+offset; y+1 < size_y; y += 2)
	{
		float *ptr1_x = addr2_s(src, y+0, offset, src_stride_x, src_stride_y);
		float *ptr2_x = addr2_s(src, y+1, offset, src_stride_x, src_stride_y);
		float *out1_x = addr2_s(dst, y+0, offset, dst_stride_x, dst_stride_y);
		float *out2_x = addr2_s(dst, y+1, offset, dst_stride_x, dst_stride_y);

		float *long_buffer_ptr = long_buffer;

		dwt_util_zero_vec_s(short_buffer_0, 4);
		dwt_util_zero_vec_s(short_buffer_1, 4);

		for(int x = 0+offset; x+1 < size_x; x += 2)
		{
			cdf97_fwd_core_dl_sc_sse_2x2_s(
				addr1_s(ptr1_x, 0, src_stride_y),
				addr1_s(ptr1_x, 1, src_stride_y),
				addr1_s(ptr2_x, 0, src_stride_y),
				addr1_s(ptr2_x, 1, src_stride_y),
				addr1_s(out1_x, 0, dst_stride_y),
				addr1_s(out1_x, 1, dst_stride_y),
				addr1_s(out2_x, 0, dst_stride_y),
				addr1_s(out2_x, 1, dst_stride_y),
				short_buffer_0, // [4]
				short_buffer_1, // [4]
				long_buffer_ptr+0*4, // [4]
				long_buffer_ptr+1*4  // [4]
			);

			long_buffer_ptr += 2*4;

			ptr1_x = addr1_s(ptr1_x, 2, src_stride_y);
			ptr2_x = addr1_s(ptr2_x, 2, src_stride_y);
			out1_x = addr1_s(out1_x, 2, dst_stride_y);
			out2_x = addr1_s(out2_x, 2, dst_stride_y);
		}
	}
}

void dwt_util_alloc_zero(
	void **ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( ptr );

	// allocate image
	dwt_util_alloc_image(
		ptr,
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	if( !ptr )
		dwt_util_error("unable to allocate memory!\n");

	// fill dst with zeros
	dwt_util_test_image_zero_s(
		*ptr,
		stride_x,
		stride_y,
		size_x,
		size_y
	);
}

void dwt_util_copy2_s(
	const void *src,
	int src_stride_x,
	int src_stride_y,
	void *dst,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	assert( src && dst );

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			*dwt_util_addr_coeff_s(
				dst,
				y,
				x,
				dst_stride_x,
				dst_stride_y
			) = *dwt_util_addr_coeff_const_s(
				src,
				y,
				x,
				src_stride_x,
				src_stride_y
			);
		}
	}
}

enum dwt_frame {
	DWT_FRAME_SEPARATE,	// two separated memory areas
	DWT_FRAME_OVERLAP,	// same memory area, image is transformed into same coordinates
	DWT_FRAME_INPLACE,	// same memory area, image is shifted
};

#if 0
void *dwt_util_frame_ptr_input_image(
	enum dwt_alg alg,
	enum dwt_frame frame,
	void *input,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	int shift = dwt_alg_shift[alg];
	int offset = 1;

	return dwt_util_viewport(input, size_x, size_y, stride_x, stride_y, offset+shift, offset+shift);
}
#endif

#if 0
// allocate input and output frame image according to "frame", "alg" and "opt_stride"
void dwt_util_alloc_frames(
	// algorithm implies shift
	enum dwt_alg alg,
	// frame type implies memory layout
	enum dwt_frame frame,
	// stride
	int opt_stride,
	// source image
	const void *src,
	int src_stride_x,
	int src_stride_y,
	int src_size_x,
	int src_size_y,
	// input frame
	void **input,
	int *input_stride_x,
	int *input_stride_y,
	int *input_size_x,
	int *input_size_y,
	// output frame
	void **output,
	int *output_stride_x,
	int *output_stride_y,
	int *output_size_x,
	int *output_size_y
)
{
	int shift = dwt_alg_shift[alg];

	if( DWT_FRAME_SEPARATE == frame )
	{
		int size_x = 1+src_size_x+shift+4;
		int size_y = 1+src_size_y+shift+4;
		int stride_y = sizeof(float);
		int stride_x = dwt_util_get_stride(stride_y*size_x, opt_stride);

		// allocate and zero input
		*input = NULL;
		*input_stride_x = stride_x;
		*input_stride_y = stride_y;
		*input_size_x = size_x;
		*input_size_y = size_y;
		dwt_util_alloc_zero(
			input,
			*input_stride_x,
			*input_stride_y,
			*input_size_x,
			*input_size_y
		);

		// allocate and zero output
		*output = NULL;
		*output_stride_x = stride_x;
		*output_stride_y = stride_y;
		*output_size_x = size_x;
		*output_size_y = size_y;
		dwt_util_alloc_zero(
			output,
			*output_stride_x,
			*output_stride_y,
			*output_size_x,
			*output_size_y
		);

		// get pointer to input.image

		// copy image into input.image
	}
}
#endif

// TODO: should return pointer to src1, src, dst
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
)
{
	// TODO: assert

	int shift = dwt_alg_shift[alg];

	*dst_size_x = src_size_x+1+shift+2+2;
	*dst_size_y = src_size_y+1+shift+2+2;
	*dst_stride_y = sizeof(float);
	*dst_stride_x = dwt_util_get_stride(*dst_stride_y * *dst_size_x, opt_stride);

	// allocate and zero dst
	dwt_util_alloc_zero(
		dst_ptr,
		*dst_stride_x,
		*dst_stride_y,
		*dst_size_x,
		*dst_size_y
	);

	*offset_x = 1;
	*offset_y = 1;
	
	// viewport to src inside of dst
	int small_size_x = src_size_x;
	int small_size_y = src_size_y;
	int small_offset_x = *offset_x;
	int small_offset_y = *offset_y;
	int small_stride_x = *dst_stride_x;
	int small_stride_y = *dst_stride_y;

	*view_ptr = dwt_util_viewport(*dst_ptr, *dst_size_x, *dst_size_y, *dst_stride_x, *dst_stride_y, small_offset_x, small_offset_y);

	// place src inside of dst
	dwt_util_copy2_s(
		src_ptr,
		src_stride_x,
		src_stride_y,
		*view_ptr,
		small_stride_x,
		small_stride_y,
		small_size_x,
		small_size_y
	);
}

typedef void (*core_func_t)(void *, void *, int, int, int, int, int, int);

void dwt_cdf97_2f_inplace_alg_s(
	enum dwt_alg alg,
	void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	// outer frame
	// assert( is_even(size_x) && is_even(size_y) && "Odd sizes are not implemented yet!" );

	core_func_t core_func[DWT_ALG_LAST] = {
		[DWT_ALG_SL_CORE_SDL] = cdf97_fwd_core_sdl_s,			// 0.057469
		[DWT_ALG_SL_CORE_SDL_SSE] = cdf97_fwd_core_sdl_sse_s,		// 0.043640
		[DWT_ALG_SL_CORE_SDL_SC_SSE] = cdf97_fwd_core_sdl_sc_sse_s,	// 0.041391
		[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0] = cdf97_fwd_core_sdl_sc_sse_off0_s,
		[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1] = cdf97_fwd_core_sdl_sc_sse_off1_s,
		[DWT_ALG_SL_CORE_DL_SC_SSE_OFF1] = cdf97_fwd_core_dl_sc_sse_off1_s,
		[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0_OVL1] = cdf97_fwd_core_sdl_sc_sse_off0_ovl1_s,
		[DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1_OVL1] = cdf97_fwd_core_sdl_sc_sse_off1_ovl1_s,
		[DWT_ALG_SL_CORE_DL] = cdf97_fwd_core_dl_s,			// 0.042970
		[DWT_ALG_SL_CORE_DL_SSE] = cdf97_fwd_core_dl_sse_s, 		// 0.043767
		[DWT_ALG_SL_CORE_DL_SC] = cdf97_fwd_core_dl_sc_s,		// 0.042255
		[DWT_ALG_SL_CORE_DL_SC_SSE] = cdf97_fwd_core_dl_sc_sse_s,	// 0.041465
	};

	core_func[alg](
		src,
		dst,
		src_stride_x,
		src_stride_y,
		dst_stride_x,
		dst_stride_y,
		size_x,
		size_y
	);
}

// TODO: propagate "flush"
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
)
{
	//FUNC_BEGIN;

	assert( M > 0 && N > 0 && fwd_secs && inv_secs );

	// pointer to M pointers to image data
	void *ptr[M];
	void *big_ptr[M];
	void *out_ptr[M];

	int big_size_x;
	int big_size_y;
	int big_stride_y;
	int big_stride_x;

	int offset_x;
	int offset_y;

	// template
	void *template; // size_x, size_y
	int stride_y = sizeof(float);
	int stride_x = dwt_util_get_stride(stride_y * size_x, opt_stride);

	// allocate
	dwt_util_alloc_image(
		&template,
		stride_x,
		stride_y,
		size_x,
		size_y);

	// fill with test pattern
	dwt_util_test_image_fill_s(
		template,
		stride_x,
		stride_y,
		size_x,
		size_y,
		0);

	// allocate M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_wrap_image(
			template,
			size_x,
			size_y,
			stride_x,
			stride_y,
			alg,
			&big_ptr[m],
			&big_size_x,
			&big_size_y,
			&big_stride_x,
			&big_stride_y,
			&offset_x,
			&offset_y,
			&ptr[m],
			opt_stride
		);
	}

	int out_size_x = big_size_x;
	int out_size_y = big_size_y;
	int out_stride_y = sizeof(float);
	int out_stride_x = dwt_util_get_stride(out_stride_y * out_size_x, opt_stride);

	for(int m = 0; m < M; m++)
	{
		// allocate big output
		dwt_util_alloc_image(
			&out_ptr[m],
			out_stride_x,
			out_stride_y,
			out_size_x,
			out_size_y
		);

		// fill
		dwt_util_test_image_zero_s(
			out_ptr[m],
			out_stride_x,
			out_stride_y,
			out_size_x,
			out_size_y
		);
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
		{
			dwt_util_flush_cache(big_ptr[m], dwt_util_image_size(big_stride_x, big_stride_y, big_size_x, big_size_y) );
			dwt_util_flush_cache(out_ptr[m], dwt_util_image_size(out_stride_x, out_stride_y, out_size_x, out_size_y) );
		}
#endif

		// start timer
		const dwt_clock_t time_fwd_start = dwt_util_get_clock(clock_type);
		// perform M fwd transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2f_inplace_alg_s(
				alg,
				big_ptr[m],
				out_ptr[m],
				big_stride_x,
				big_stride_y,
				out_stride_x,
				out_stride_y,
				big_size_x,
				big_size_y
			);
		}
		// stop timer
		const dwt_clock_t time_fwd_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_fwd_secs = (float)(time_fwd_stop - time_fwd_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_fwd_secs < *fwd_secs )
			*fwd_secs = time_fwd_secs;

#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
		{
			dwt_util_flush_cache(out_ptr[m], dwt_util_image_size(out_stride_x, out_stride_y, out_size_x, out_size_y) );
		}
#endif

		// start timer
		const dwt_clock_t time_inv_start = dwt_util_get_clock(clock_type);
		// perform M inv transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2i_s(
				out_ptr[m],
				out_stride_x,
				out_stride_y,
				out_size_x,
				out_size_y,
				out_size_x,
				out_size_y,
				1,
				0,
				0);
		}
		// stop timer
		const dwt_clock_t time_inv_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_inv_secs = (float)(time_inv_stop - time_inv_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_inv_secs < *inv_secs )
			*inv_secs = time_inv_secs;
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&big_ptr[m]);
		dwt_util_free_image(&out_ptr[m]);
	}
	dwt_util_free_image(&template);

	//FUNC_END;
}

extern const float g_growth_factor_s;

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
		dwt_util_perf_cdf97_2_inplace_alg_s(
			alg,
			size_x,
			size_y,
			opt_stride, // FIXME
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs
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

// core transform with overlapping src/dst areas
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
	int overlap // 0 = B, 1 = C
)
{
	//FUNC_BEGIN;

	assert( M > 0 && N > 0 && fwd_secs && inv_secs );

	// template
	void *template_ptr;
	int template_size_x = size_x;
	int template_size_y = size_y;
	int template_stride_y = sizeof(float);
	int template_stride_x = dwt_util_get_stride(template_stride_y * template_size_x, opt_stride);

	// allocate template
	dwt_util_alloc_image(
		&template_ptr,
		template_stride_x,
		template_stride_y,
		template_size_x,
		template_size_y);

	// fill template with test pattern
	dwt_util_test_image_fill_s(
		template_ptr,
		template_stride_x,
		template_stride_y,
		template_size_x,
		template_size_y,
		0);

	// shift
	int shift = dwt_alg_get_shift(alg);

	// offset
	//int offset = 1;
	int offset = ( DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0!=alg && DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0_OVL1!=alg ) ? 1 : 0;

	// frame
	void *out_ptr[M];
	int out_size_x = offset+shift+size_x+shift+4;
	int out_size_y = offset+shift+size_y+shift+4;
	int out_stride_y = sizeof(float);
	int out_stride_x = dwt_util_get_stride(out_stride_y * out_size_x, opt_stride);

	// pointers
	void *src_img[M]; // image size_x x size_y
	void *src_ptr[M]; // transform 1+img_size_x+shift+2+2 x 1+img_size_y+shift+2+2
	void *dst_img[M]; // image size_x x size_y
	void *dst_ptr[M]; // transform 1+img_size_x+shift+2+2 x 1+img_size_y+shift+2+2

	// allocate M images
	for(int m = 0; m < M; m++)
	{
		// allocate
		dwt_util_alloc_image(
			&out_ptr[m],
			out_stride_x,
			out_stride_y,
			out_size_x,
			out_size_y*2+21 // HACK: splitting among threads causes access up to up_to_even( Y/(Y-1) ) ~= Y*2 rows
		);
		// HACK: read area have to be filled with zeros
		dwt_util_test_image_zero_s(
			out_ptr[m],
			out_stride_x,
			out_stride_y,
			out_size_x,
			out_size_y*2+21
		);
		out_ptr[m] += out_stride_x*20; // HACK: need to read some area before image beginning (threads)

		src_ptr[m] = dwt_util_viewport(out_ptr[m], out_size_x, out_size_y, out_stride_x, out_stride_y, shift, shift);
		src_img[m] = dwt_util_viewport(out_ptr[m], out_size_x, out_size_y, out_stride_x, out_stride_y, offset+shift, offset+shift);
		if( !overlap )
		{
			dst_ptr[m] = dwt_util_viewport(out_ptr[m], out_size_x, out_size_y, out_stride_x, out_stride_y, 0, 0);
			dst_img[m] = dwt_util_viewport(out_ptr[m], out_size_x, out_size_y, out_stride_x, out_stride_y, offset+shift, offset+shift);
		}
		else
		{
			dst_ptr[m] = dwt_util_viewport(out_ptr[m], out_size_x, out_size_y, out_stride_x, out_stride_y, shift, shift);
			dst_img[m] = dwt_util_viewport(out_ptr[m], out_size_x, out_size_y, out_stride_x, out_stride_y, offset+2*shift, offset+2*shift);
		}
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
		// copy template into frame.src_img
		for(int m = 0; m < M; m++)
		{
			// TODO: this is necessary
			dwt_util_test_image_zero_s(
				out_ptr[m],
				out_stride_x,
				out_stride_y,
				out_size_x,
				out_size_y
			);

			dwt_util_copy2_s(
				template_ptr,
				template_stride_x,
				template_stride_y,
				src_img[m],
				out_stride_x,
				out_stride_y,
				template_size_x,
				template_size_y
			);
		}

		// flush memory
		if( flush )
		{
			for(int m = 0; m < M; m++)
			{
				dwt_util_flush_cache(out_ptr[m], dwt_util_image_size(out_stride_x, out_stride_y, out_size_x, out_size_y) );
			}
		}

		// start timer
		const dwt_clock_t time_fwd_start = dwt_util_get_clock(clock_type);
		// perform M fwd transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2f_inplace_alg_s(
				alg,
				src_ptr[m],
				dst_ptr[m],
				out_stride_x,
				out_stride_y,
				out_stride_x,
				out_stride_y,
				offset+size_x+shift+4,
				offset+size_y+shift+4
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
		if( flush )
		{
			for(int m = 0; m < M; m++)
			{
				dwt_util_flush_cache(out_ptr[m], dwt_util_image_size(out_stride_x, out_stride_y, out_size_x, out_size_y) );
			}
		}

		int inv_offset = 1-offset;

		// start timer
		const dwt_clock_t time_inv_start = dwt_util_get_clock(clock_type);
		// perform M inv transforms
		for(int m = 0; m < M; m++)
		{
// 			dwt_cdf97_2i_inplace_s(
// 				out_ptr[m],
// 				out_stride_x,
// 				out_stride_y,
// 				out_size_x,
// 				out_size_y,
// 				out_size_x,
// 				out_size_y,
// 				1,
// 				0,
// 				0
// 			);
			dwt_cdf97_2i_inplace_s(
				dwt_util_addr_coeff_s(out_ptr[m],inv_offset,inv_offset,out_stride_x,out_stride_y),
				out_stride_x,
				out_stride_y,
				out_size_x-inv_offset,
				out_size_y-inv_offset,
				out_size_x-inv_offset,
				out_size_y-inv_offset,
				1,
				0,
				0
			);
		}
		// stop timer
		const dwt_clock_t time_inv_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_inv_secs = (float)(time_inv_stop - time_inv_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_inv_secs < *inv_secs )
			*inv_secs = time_inv_secs;

		// compare template and dst_img
		for(int m = 0; m < M; m++)
		{
			if( dwt_util_compare2_s(dst_img[m], template_ptr, out_stride_x, out_stride_y, template_stride_x, template_stride_y, size_x, size_y) )
			{
				dwt_util_log(LOG_INFO, "images differs (%i,%i) overlap=%i alg=%i\n", size_x, size_y, overlap, (int)alg);
				dwt_util_save_to_pgm_s("debug.pgm", 1.0, dst_img[m], out_stride_x, out_stride_y, size_x, size_y);
				dwt_util_save_to_pgm_s("ref.pgm", 1.0, template_ptr, template_stride_x, template_stride_y, size_x, size_y);
			}
		}
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		out_ptr[m] -= out_stride_x*20; // HACK
		dwt_util_free_image(&out_ptr[m]);
	}
	dwt_util_free_image(&template_ptr);

	//FUNC_END;
}

void dwt_util_measure_perf_cdf97_2_inplaceB_alg_s(
	enum dwt_alg alg,
	int min_x,
	int max_x,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data,
	int flush,
	int overlap
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
		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			alg,
			size_x,
			size_y,
			opt_stride,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs,
			flush,
			overlap
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

void dwt_util_perf_cdf97_2_inplaceABC_alg_s(
	enum dwt_alg alg,
	int size_x,
	int size_y,
	int opt_stride,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs,
	int overlap // 0 = B, 1 = C, -1 = A
)
{
	// HACK
	size_x = dwt_util_to_even(size_x);
	size_y = dwt_util_to_even(size_y);

	int flush = 1;

	if( -1 == overlap )
		dwt_util_perf_cdf97_2_inplace_alg_s(
			alg,
			size_x,
			size_y,
			opt_stride,
			M,
			N,
			clock_type,
			fwd_secs,
			inv_secs
		);
	else
		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			alg,
			size_x,
			size_y,
			opt_stride,
			M,
			N,
			clock_type,
			fwd_secs,
			inv_secs,
			flush,
			overlap // 0 = B, 1 = C
		);
}

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
		dwt_util_perf_cdf97_2_inplaceABC_alg_s(
			alg,
			size_x,
			size_y,
			opt_stride,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs,
			overlap
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
