#include "cores-haar.h"
#include <assert.h>
#include "coords.h"
#include "inline.h"
#ifdef __SSE__
	#include <xmmintrin.h>
#endif

static
void core_fwd_haar_v2x1_f32(
        float *data0, // left [1]
        float *data1 // right [1]
)
{
	const float w[2] = { +0.5f, -1.f }; 

	// variables
	float x0, x1;

	// inputs
	x0 = *data0;
	x1 = *data1;

	x1 += w[1] * x0; // -1
	x0 += w[0] * x1; // +1/2

	// outputs
	*data0 = x0;
	*data1 = x1;
}

static
void core_fwd_haar_n2x2_f32(
        float *data
)
{
	// variables
	float a, h, v, d;

	// inputs
	a = data[0];
	h = data[1];
	v = data[2];
	d = data[3];

	// \alpha = +1/2
	// \beta = -1

	// P
	// D += \alpha^2 * A + \alpha * V + \alpha * H
	d += a - v - h;

	// PP
	// H += \beta * D + \alpha * A
	h += 0.5f*d - a;
	// V += \beta * D + \alpha * A
	v += 0.5f*d - a;

	// U
	// A += \beta * H + \beta * V + \alpha*\beta^2 * D
	a += 0.5f*h + 0.5f*v - 0.25f*d;

	// outputs
	data[0] = a;
	data[1] = h;
	data[2] = v;
	data[3] = d;
}

#ifdef __SSE__
static
void core_fwd_haar_v4x2_f32(
        __m128 *data0, // left [1]
        __m128 *data1 // right [1]
)
{
	const __m128 w[2] = {
		{ +0.5f, +0.5f, +0.5f, +0.5f, },
		{ -1.f, -1.f, -1.f, -1.f, }
	}; 

	// variables
	__m128 x0, x1;

	// inputs
	x0 = *data0;
	x1 = *data1;

	x1 += w[1] * x0; // -1
	x0 += w[0] * x1; // +1/2

	// outputs
	*data0 = x0;
	*data1 = x1;
}
#endif

static
void core_inv_haar_v2x1_f32(
        float *data0, // left [1]
        float *data1 // right [1]
)
{
	const float w[2] = { +0.5f, -1.f }; 

	// variables
	float x0, x1;

	// inputs
	x0 = *data0;
	x1 = *data1;

	x0 -= w[0] * x1; // +1/2
	x1 -= w[1] * x0; // -1

	// outputs
	*data0 = x0;
	*data1 = x1;
}

static
void cores2f_haar_v2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y
)
{
	const int overlap_x_L = 0;
	const int overlap_y_L = 0;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 0;

	// 2x2
	float t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			t[yy*step_x+xx] = *addr2_s(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// calc
	core_fwd_haar_v2x1_f32(t+0, t+1);
	core_fwd_haar_v2x1_f32(t+2, t+3);
	core_fwd_haar_v2x1_f32(t+0, t+2);
	core_fwd_haar_v2x1_f32(t+1, t+3);

	// scaling

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

			*addr2_s(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = t[yy*step_x+xx];
		}
	}
}

static
void cores2f_haar_n2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y
)
{
	const int overlap_x_L = 0;
	const int overlap_y_L = 0;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 0;

	// 2x2
	float t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			t[yy*step_x+xx] = *addr2_s(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// calc
	core_fwd_haar_n2x2_f32(t);

	// scaling

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

			*addr2_s(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = t[yy*step_x+xx];
		}
	}
}

static
void cores2f_haar_v4x4_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y
)
{
#ifdef __SSE__
	const int overlap_x_L = 0;
	const int overlap_y_L = 0;

	const int step_y = 4;
	const int step_x = 4;

	const int shift = 0;

	// 4x4
	__m128 t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			t[xx][yy] = *addr2_s(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// horizontal
	core_fwd_haar_v4x2_f32(t+0, t+1);
	core_fwd_haar_v4x2_f32(t+2, t+3);

	// transpose
	_MM_TRANSPOSE4_PS(t[0], t[1], t[2], t[3]);
	
	// vertical
	core_fwd_haar_v4x2_f32(t+0, t+1);
	core_fwd_haar_v4x2_f32(t+2, t+3);

	// scaling

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

			*addr2_s(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = t[yy][xx];
		}
	}
#endif
}

static
void cores2i_haar_v2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y
)
{
	const int overlap_x_L = 0;
	const int overlap_y_L = 0;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 0;

	// 2x2
	float t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			t[yy*step_x+xx] = *addr2_s(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// calc
	core_inv_haar_v2x1_f32(t+0, t+1);
	core_inv_haar_v2x1_f32(t+2, t+3);
	core_inv_haar_v2x1_f32(t+0, t+2);
	core_inv_haar_v2x1_f32(t+1, t+3);

	// scaling

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

			*addr2_s(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = t[yy*step_x+xx];
		}
	}
}

void cores2f_haar_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 0;
	const int overlap_y_L = 0;
	const int overlap_x_R = 0;
	const int overlap_y_R = 0;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_haar_v2x2_f32_core(src, dst, x, y);
}

void cores2f_haar_n2x2_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 0;
	const int overlap_y_L = 0;
	const int overlap_x_R = 0;
	const int overlap_y_R = 0;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_haar_n2x2_f32_core(src, dst, x, y);
}

void cores2f_haar_v4x4_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int step_x = 4;
	const int step_y = 4;

	const int overlap_x_L = 0;
	const int overlap_y_L = 0;
	const int overlap_x_R = 0;
	const int overlap_y_R = 0;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_haar_v4x4_f32_core(src, dst, x, y);
}

void cores2i_haar_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 0;
	const int overlap_y_L = 0;
	const int overlap_x_R = 0;
	const int overlap_y_R = 0;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2i_haar_v2x2_f32_core(src, dst, x, y);
}
