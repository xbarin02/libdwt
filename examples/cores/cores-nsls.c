#include "cores-nsls.h"

#include "libdwt.h"
#include <assert.h>
#include "inline.h"

#ifdef __SSE__
	#include <xmmintrin.h>
#endif
#ifdef __SSE3__
	#include <pmmintrin.h>
#endif
#ifdef __SSE4_1__
	#include <smmintrin.h>
#endif

static
int virt2real(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		real *= -1;
	if( real > size-1 )
		real = 2*(size-1) - real;

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

// H1 = -1.58613434342059f
#define H1 (-1.58613434342059f)
// H2 = -0.0529801185729f
#define H2 (-0.0529801185729f)
// H3 = +0.8829110755309f
#define H3 (+0.8829110755309f)
// H4 = +0.4435068520439f
#define H4 (+0.4435068520439f)
// H11 = +H1*H1 = +(-1.58613434342059f * -1.58613434342059f) = +2.51582215538f
#define H11 (+2.51582215538f)
// H22 = -H2*H2 = -(-0.0529801185729f  * -0.0529801185729f ) = -0.00280689296f
#define H22 (-0.00280689296f)
// H33 = +H3*H3 = +(+0.8829110755309f  * +0.8829110755309f ) = +0.77953196729f
#define H33 (+0.77953196729f)
// H44 = -H4*H4 = -(+0.4435068520439f  * +0.4435068520439f ) = -0.19669832781f
#define H44 (-0.19669832781f)
// SZ = 1.f*(1.1496043988602f*1.1496043988602f)
#define SZ (+1.32159027388f)
// SR = 1.f/(1.1496043988602f*1.1496043988602f)
#define SR (+0.7566641642f)

static
void copy4(float *dst, const float *src)
{
	for(int i = 0; i < 4; i++)
		dst[i] = src[i];
}

#ifdef __SSE__
static
void copy4_sse(__m128 *dst, const __m128 *src)
{
	*dst = *src;
}
#endif

#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
static
void cdf53_core(
	float *red, // 4 = input
	float *green, // 4 = output
	float *blue_left, // 4
	float *blue_top // 4
)
{
	// P
	{
		// square += +1/4*circle -1/2*cross -1/2*triangle
		red[0] +=
			+0.25f * ( green[3] + blue_top[3] + blue_left[3] + red[3] )
			-0.5f  * ( blue_top[2] + red[2] )
			-0.5f  * ( blue_left[1] + red[1] );
	}
	// PP
	{
		// cross += -1/2*circle +1/4*square
		blue_top[2] +=
			-0.5f  * ( green[3] + blue_top[3] )
			+0.25f * ( blue_top[0] + red[0] );

		// triangle += -1/2*circle +1/4*square
		blue_left[1] +=
			-0.5f  * ( green[3] + blue_left[3] )
			+0.25f * ( blue_left[0] + red[0] );
	}
	// U
	{
		// circle += +1/4*cross +1/4*triangle -1/16*square
		green[3] +=
			+0.25f   * ( green[2] + blue_top[2] )
			+0.25f   * ( green[1] + blue_left[1] )
			-0.0625f * ( green[0] + blue_top[0] + blue_left[0] + red[0] );
	}
	// S
	{
		green[0] *= 0.5f;
		green[1] *= 1.0f;
		green[2] *= 1.0f;
		green[3] *= 2.0f;
	}
}
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"

#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
static
void core(
	float *red, // 4 = input
	float *green, // 4 = output
	float *blue0, // 8
	float *blue1, // 8
	float *blue2, // 8
	float *yellow // 4
)
{
	// 1 P
	{
		// HH = star(y+0,x+0) (d/LL*H11, v/LH*H1, h/HL*H1)
		red[0] +=
			+ H1  * blue2[6] // N  = v
			+ H11 * blue2[7] // NE = d
			+ H1  * red[1]   // E  = h
			+ H11 * red[3]   // SE = d
			+ H1  * red[2]   // S  = v
			+ H11 * blue1[7] // SW = d
			+ H1  * blue1[5] // W  = h
			+ H11 * blue1[3] // NW = d
			;
	}
	// 1 PP
	{
		// LH = cross(y+1,x+0) (h/LL*H1, v/HH*H2)
		blue2[6] +=
			+ H2 * blue2[4] // N = v
			+ H1 * blue2[7] // E = h
			+ H2 * red[0]   // S = v
			+ H1 * blue1[3] // W = h
			;
		// HL = cross(y+0,x+1) (v/LL*H1, h/HH*H2)
		blue1[5] +=
			+ H1 * blue1[3] // N = v
			+ H2 * red[0]   // E = h
			+ H1 * blue1[7] // S = v
			+ H2 * blue1[4] // W = h
			;
	}
	// 1 U
	{
		// LL = op_star(y=1,x=1) (h/LH*H2, v/HL*H2, d/HH*H22)
		blue1[3] +=
			+ H2  * blue1[1] // N  = v
			+ H22 * blue2[4] // NE = d
			+ H2  * blue2[6] // E  = h
			+ H22 * red[0]   // SE = d
			+ H2  * blue1[5] // S  = v
			+ H22 * blue1[4] // SW = d
			+ H2  * blue1[2] // W  = h
			+ H22 * blue1[0] // NW = d
			;
	}
	// 2 P
	{
		// HH = star(y+0,x+0) (d/LL*H33, v/LH*H3, h/HL*H3)
		blue1[0] +=
			+ H3  * yellow[2] // N  = v
			+ H33 * yellow[3] // NE = d
			+ H3  * blue1[1]  // E  = h
			+ H33 * blue1[3]  // SE = d
			+ H3  * blue1[2]  // S  = v
			+ H33 * blue0[3]  // SW = d
			+ H3  * blue0[1]  // W  = h
			+ H33 * green[3]  // NW = d
			;
	}
	// 2 PP
	{
		// LH = cross(y+1,x+0) (h/LL*H3, v/HH*H4)
		yellow[2] +=
			+ H4 * yellow[0] // N  = v
			+ H3 * yellow[3] // E  = h
			+ H4 * blue1 [0] // S  = v
			+ H3 * green [3] // W  = h
			;
		// HL = cross(y+0,x+1) (v/LL*H3, h/HH*H4)
		blue0[1] +=
			+ H3 * green[3] // N  = v
			+ H4 * blue1[0] // E  = h
			+ H3 * blue0[3] // S  = v
			+ H4 * blue0[0] // W  = h
			;
	}
	// 2 U
	{
		// LL = op_star(y=1,x=1) (h/LH*H4, v/HL*H4, d/HH*H44)
		green[3] +=
			+ H4  * green[1]  // N  = v
			+ H44 * yellow[0] // NE = d
			+ H4  * yellow[2] // E  = h
			+ H44 * blue1[0]  // SE = d
			+ H4  * blue0[1]  // S  = v
			+ H44 * blue0[0]  // SW = d
			+ H4  * green[2]  // W  = h
			+ H44 * green[0]  // NW = d
			;
	}
	// S
	{
		green[0] *= SR;
		green[1] *= 1.f;
		green[2] *= 1.f;
		green[3] *= SZ;
	}
}
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"

#ifdef __SSE__
static
float dot_sse(__m128 a, __m128 b)
{
#ifdef __SSE4_1__
	return _mm_dp_ps(a, b, 0xff)[0];
#endif
#ifdef __SSE3__
	a *= b;

	a = _mm_hadd_ps(a, a);
	a = _mm_hadd_ps(a, a);

	return a[0];
#endif
	a *= b;

	return a[0] + a[1] + a[2] + a[3];
}
#endif

#ifdef __SSE__
static
void core_sse(
	__m128 *red,   // 4 = input
	__m128 *green, // 4 = output
	__m128 *blue0, // 8
	__m128 *blue1, // 8
	__m128 *blue2, // 8
	__m128 *yellow // 4
)
{
	// 1 P
	{
#if 0
		// HH = star(y+0,x+0) (d/LL*H11, v/LH*H1, h/HL*H1)
		(*red)[0] +=
			+ H1  * (blue2[1])[6-4] // N  = v
			+ H11 * (blue2[1])[7-4] // NE = d
			+ H1  * (*red)[1]   // E  = h
			+ H11 * (*red)[3]   // SE = d
			+ H1  * (*red)[2]   // S  = v
			+ H11 * (blue1[1])[7-4] // SW = d
			+ H1  * (blue1[1])[5-4] // W  = h
			+ H11 * (blue1[0])[3-0] // NW = d
			;
#else
		red[0][0] +=
			+ H1  * blue2[1][2] // N  = v
			+ H11 * blue2[1][3] // NE = d

			+ dot_sse(red[0], (__m128){ 0.f, H1, H1, H11 })

			+ H11 * blue1[1][3] // SW = d
			+ H1  * blue1[1][1] // W  = h

			+ H11 * blue1[0][3] // NW = d
		;
#endif
	}
	// 1 PP
	{
		// LH = cross(y+1,x+0) (h/LL*H1, v/HH*H2)
		(blue2[1])[6-4] +=
			+ H2 * (blue2[1])[4-4] // N = v
			+ H1 * (blue2[1])[7-4] // E = h
			+ H2 * (*red)[0]   // S = v
			+ H1 * (blue1[0])[3-0] // W = h
			;
		// HL = cross(y+0,x+1) (v/LL*H1, h/HH*H2)
		(blue1[1])[5-4] +=
			+ H1 * (blue1[0])[3] // N = v
			+ H2 * (*red)[0]   // E = h
			+ H1 * (blue1[1])[7-4] // S = v
			+ H2 * (blue1[1])[4-4] // W = h
			;
	}
	// 1 U
	{
#if 0
		// LL = op_star(y=1,x=1) (h/LH*H2, v/HL*H2, d/HH*H22)
		(blue1[0])[3-0] +=
			+ H2  * (blue1[0])[1-0] // N  = v
			+ H22 * (blue2[1])[4-4] // NE = d
			+ H2  * (blue2[1])[6-4] // E  = h
			+ H22 * (*red)[0]   // SE = d
			+ H2  * (blue1[1])[5-4] // S  = v
			+ H22 * (blue1[1])[4-4] // SW = d
			+ H2  * (blue1[0])[2-0] // W  = h
			+ H22 * (blue1[0])[0-0] // NW = d
			;
#else
		blue1[0][3] +=
			+ dot_sse(blue1[0], (__m128){ H22, H2, H2, 0.f })

			+ H22 * blue2[1][0] // NE = d
			+ H2  * blue2[1][2] // E  = h

			+ H22 * red[0][0]   // SE = d

			+ H2  * blue1[1][1] // S  = v
			+ H22 * blue1[1][0] // SW = d

		;
#endif
	}
	// 2 P
	{
#if 0
		// HH = star(y+0,x+0) (d/LL*H33, v/LH*H3, h/HL*H3)
		(blue1[0])[0-0] +=
			+ H3  * (*yellow)[2] // N  = v
			+ H33 * (*yellow)[3] // NE = d
			+ H3  * (blue1[0])[1-0]  // E  = h
			+ H33 * (blue1[0])[3-0]  // SE = d
			+ H3  * (blue1[0])[2-0]  // S  = v
			+ H33 * (blue0[0])[3-0]  // SW = d
			+ H3  * (blue0[0])[1-0]  // W  = h
			+ H33 * (*green)[3]  // NW = d
			;
#else
		blue1[0][0] +=
			+ H3  * yellow[0][2] // N  = v
			+ H33 * yellow[0][3] // NE = d

			+ dot_sse(blue1[0], (__m128){ 0.f, H3, H3, H33 })

			+ H33 * blue0[0][3]  // SW = d
			+ H3  * blue0[0][1]  // W  = h

			+ H33 * green[0][3]  // NW = d
		;
#endif
	}
	// 2 PP
	{
		// LH = cross(y+1,x+0) (h/LL*H3, v/HH*H4)
		(*yellow)[2] +=
			+ H4 * (*yellow)[0] // N  = v
			+ H3 * (*yellow)[3] // E  = h
			+ H4 * (blue1[0])[0-0] // S  = v
			+ H3 * (*green)[3] // W  = h
			;
		// HL = cross(y+0,x+1) (v/LL*H3, h/HH*H4)
		(blue0[0])[1] +=
			+ H3 * (*green)[3] // N  = v
			+ H4 * (blue1[0])[0-0] // E  = h
			+ H3 * (blue0[0])[3-0] // S  = v
			+ H4 * (blue0[0])[0-0] // W  = h
			;
	}
	// 2 U
	{
#if 0
		// LL = op_star(y=1,x=1) (h/LH*H4, v/HL*H4, d/HH*H44)
		(*green)[3] +=
			+ H4  * (*green)[1]  // N  = v
			+ H44 * (*yellow)[0] // NE = d
			+ H4  * (*yellow)[2] // E  = h
			+ H44 * (blue1[0])[0-0]  // SE = d
			+ H4  * (blue0[0])[1-0]  // S  = v
			+ H44 * (blue0[0])[0-0]  // SW = d
			+ H4  * (*green)[2]  // W  = h
			+ H44 * (*green)[0]  // NW = d
			;
#else
		(*green)[3] +=
			+ dot_sse(green[0], (__m128){ H44, H4, H4, 0.f })

			+ H44 * yellow[0][0] // NE = d
			+ H4  * yellow[0][2] // E  = h

			+ H44 * blue0[0][0]  // SW = d
			+ H4  * blue0[0][1]  // S  = v

			+ H44 * blue1[0][0]  // SE = d
		;
#endif
	}
	// S
	{
#if 0
		(*green)[0] *= SR;
		(*green)[1] *= 1.f;
		(*green)[2] *= 1.f;
		(*green)[3] *= SZ;
#else
		(*green) *= (__m128){ SR, 1.f, 1.f, SZ };
#endif
	}
}
#endif

static
void cores2f_cdf97_v2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	float *yellow, // [4]
	float *green, // [4]
	float *blue0, // [8]
	float *blue1, // [8]
	float *blue2 // [8]
)
{
	const int overlap_x_L = 5;
	const int overlap_y_L = 5;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

	// 2x2
	float red[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			red[yy*step_x+xx] = *addr2_s(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// core
	core(
		red,   // 4 = input
		green, // 4 = output
		blue0, // 8
		blue1, // 8
		blue2, // 8
		yellow // 4
	);

	// store
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, src->size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = green[yy*step_x+xx];
		}
	}

	// shuffle
	{

		copy4(green, yellow);
		copy4(yellow, blue2+0);
		copy4(blue2+0, blue2+4);
		copy4(blue2+4, red);
	}
}

static
void cores2f_cdf53_v2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	float *green, // [4]
	float *blue_left, // [4]
	float *blue_top // [4]
)
{
	const int overlap_x_L = 3;
	const int overlap_y_L = 3;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 2;

	// 2x2
	float red[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			red[yy*step_x+xx] = *addr2_s(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// core
	cdf53_core(
		red,   // 4 = input
		green, // 4 = output
		blue_left, // 4
		blue_top // 4
	);

	// store
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, src->size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = green[yy*step_x+xx];
		}
	}

	// shuffle
	{
		copy4(green, blue_top);
		copy4(blue_top, red);
	}
}

#ifdef __SSE__
static
void cores2f_cdf97_v2x2_f32_core_sse(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	__m128 *yellow, // [4]
	__m128 *green,  // [4]
	__m128 *blue0,  // [8]
	__m128 *blue1,  // [8]
	__m128 *blue2   // [8]
)
{
	const int overlap_x_L = 5;
	const int overlap_y_L = 5;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

	// 2x2
	__m128 red; // [4]

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			red[yy*step_x+xx] = *addr2_s(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// core
	core_sse(
		&red,  // 4 = input
		green, // 4 = output
		blue0, // 8
		blue1, // 8
		blue2, // 8
		yellow // 4
	);

	// store
	for(int yy = 0; yy < step_y; yy++)
	{
		for(int xx = 0; xx < step_x; xx++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, src->size_y);
			if( pos_x < 0 || pos_y < 0 )
				continue;

			*addr2_s(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = (*green)[yy*step_x+xx];
		}
	}

	// shuffle
	{

		copy4_sse(green, yellow);
		copy4_sse(yellow, blue2+0);
		copy4_sse(blue2+0, blue2+1);
		copy4_sse(blue2+1, &red);
	}
}
#endif

void cores2f_cdf97_n2x2_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int buff_elem_size = 4;

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 5;
	const int overlap_y_L = 5;
	const int overlap_x_R = 5;
	const int overlap_y_R = 5;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	float blue[buff_elem_size*super_x];
	float yellow[4];
	float green[4];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf97_v2x2_f32_core(
				src,
				dst,
				x,
				y,
				yellow, // yellow
				green, // green
				blue+buff_elem_size*(x-4), // blue0
				blue+buff_elem_size*(x-2), // blue1
				blue+buff_elem_size*(x-0)  // blue2
			);
}

void cores2f_cdf53_n2x2_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int buff_elem_size = 2;

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 3;
	const int overlap_y_L = 3;
	const int overlap_x_R = 3;
	const int overlap_y_R = 3;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	float blue[buff_elem_size*super_x];
	float green[4];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf53_v2x2_f32_core(
				src,
				dst,
				x,
				y,
				green, // green
				blue+buff_elem_size*(x-2), // blue_left
				blue+buff_elem_size*(x-0)  // blue_top
			);
}

void cores2f_cdf97_n2x2_f32_sse(
	struct image_t *src,
	struct image_t *dst
)
{
#ifdef __SSE__
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 5;
	const int overlap_y_L = 5;
	const int overlap_x_R = 5;
	const int overlap_y_R = 5;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	__m128 blue[super_x];
	__m128 yellow;
	__m128 green;

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf97_v2x2_f32_core_sse(
				src,
				dst,
				x,
				y,
				&yellow, // yellow
				&green, // green
				blue+(x-4), // blue0
				blue+(x-2), // blue1
				blue+(x-0)  // blue2
			);
#endif
}
