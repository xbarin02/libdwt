#include "cores.h"
#include "libdwt.h"
#include <assert.h>
#include "inline.h"
#include <stdint.h>
#include "fix.h"

// symmetric border extension (whole point symmetry)
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

// constant padding with first/last value
static
int virt2real_copy(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		real = 0;
	if( real > size-1 )
		real = size-1;

	return real;
}

// error outside of image area
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
float op_f32(float l, float r, float w)
{
	return w*(l+r);
}

static
int32_t op_i32(int32_t l, int32_t r, int32_t w, int s)
{
	const int k = 1<<(s-1);

	return ( w*(l+r) + k ) >> s;
}

static
FIX32_T op_x32(FIX32_T l, FIX32_T r, FIX32_T w)
{

	return fix32_mul(w, l+r);
}

static
FIX16_T op_x16(FIX16_T l, FIX16_T r, FIX16_T w)
{

	return fix16_mul(w, l+r);
}

static
void vert_2x1_f32(
        float *data0, // left [1]
        float *data1, // right [1]
        float *buff // [4]
)
{
	const float w[4] = { +dwt_cdf97_u2_s, -dwt_cdf97_p2_s, +dwt_cdf97_u1_s, -dwt_cdf97_p1_s }; // [ u2 p2 u1 p1 ]

	// variables
	float c[4];
	float r[4];
	float x0, x1;
	float y0, y1;

	// load
	float *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0   = l[0];
	c[0] = l[1];
	c[1] = l[2];
	c[2] = l[3];
	c[3] = x0;

	// operation
	r[3] = x1;
	r[2] = c[3] + op_f32(l[3], r[3], w[3]);
	r[1] = c[2] + op_f32(l[2], r[2], w[2]);
	r[0] = c[1] + op_f32(l[1], r[1], w[1]);
	y1   = c[0] + op_f32(l[0], r[0], w[0]);

	// update
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	// outputs
	*data0 = y0;
	*data1 = y1;
}

static
void cdf53_vert_2x1_f32(
        float *data0, // left [1]
        float *data1, // right [1]
        float *buff // [2]
)
{
	const float w[2] = { +dwt_cdf53_u1_s, -dwt_cdf53_p1_s }; 

	// variables
	float c[2];
	float r[2];
	float x0, x1;
	float y0, y1;

	// load
	float *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0   = l[0];
	c[0] = l[1];
	c[1] = x0;

	// operation
	r[1] = x1;
	r[0] = c[1] + op_f32(l[1], r[1], w[1]);
	y1   = c[0] + op_f32(l[0], r[0], w[0]);

	// update
	l[0] = r[0];
	l[1] = r[1];

	// outputs
	*data0 = y0;
	*data1 = y1;
}

static
void vert_2x1_inv_f32(
        float *data0, // left [1]
        float *data1, // right [1]
        float *buff // [4]
)
{
	const float w[4] = { +dwt_cdf97_p1_s, -dwt_cdf97_u1_s, +dwt_cdf97_p2_s, -dwt_cdf97_u2_s };

	// variables
	float c[4];
	float r[4];
	float x0, x1;
	float y0, y1;

	// load
	float *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0   = l[0];
	c[0] = l[1];
	c[1] = l[2];
	c[2] = l[3];
	c[3] = x0;

	// operation
	r[3] = x1;
	r[2] = c[3] + op_f32(l[3], r[3], w[3]);
	r[1] = c[2] + op_f32(l[2], r[2], w[2]);
	r[0] = c[1] + op_f32(l[1], r[1], w[1]);
	y1   = c[0] + op_f32(l[0], r[0], w[0]);

	// update
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	// outputs
	*data0 = y0;
	*data1 = y1;
}

static
void vert_2x1_i32(
        int32_t *data0, // left [1]
        int32_t *data1, // right [1]
        int32_t *buff // [4]
)
{
	// weights
	const int32_t w[4] = { +1817, +113, -217, -203 }; // [ u2 p2 u1 p1 ]
	// shifts
	const int s[4] = { 12, 7, 12, 7 }; // [ u2 p2 u1 p1 ]

	// variables
	int32_t c[4];
	int32_t r[4];
	int32_t x0, x1;
	int32_t y0, y1;

	// load
	int32_t *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0   = l[0];
	c[0] = l[1];
	c[1] = l[2];
	c[2] = l[3];
	c[3] = x0;

	// operation
	r[3] = x1;
	r[2] = c[3] + op_i32(l[3], r[3], w[3], s[3]);
	r[1] = c[2] + op_i32(l[2], r[2], w[2], s[2]);
	r[0] = c[1] + op_i32(l[1], r[1], w[1], s[1]);
	y1   = c[0] + op_i32(l[0], r[0], w[0], s[0]);

	// update
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	// outputs
	*data0 = y0;
	*data1 = y1;
}

static
void vert_2x1_x32(
        FIX32_T *data0, // left [1]
        FIX32_T *data1, // right [1]
        FIX32_T *buff // [4]
)
{
// 	const FIX32_T w[4] = {
// 		conv_float32_to_fix32(+dwt_cdf97_u2_s),
// 		conv_float32_to_fix32(-dwt_cdf97_p2_s),
// 		conv_float32_to_fix32(+dwt_cdf97_u1_s),
// 		conv_float32_to_fix32(-dwt_cdf97_p1_s)
// 	}; // [ u2 p2 u1 p1 ]

	const FIX32_T w[4] = { 29066, 57862, -3472, -103949 };

	// variables
	FIX32_T c[4];
	FIX32_T r[4];
	FIX32_T x0, x1;
	FIX32_T y0, y1;

	// load
	FIX32_T *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0   = l[0];
	c[0] = l[1];
	c[1] = l[2];
	c[2] = l[3];
	c[3] = x0;

	// operation
	r[3] = x1;
	r[2] = c[3] + op_x32(l[3], r[3], w[3]);
	r[1] = c[2] + op_x32(l[2], r[2], w[2]);
	r[0] = c[1] + op_x32(l[1], r[1], w[1]);
	y1   = c[0] + op_x32(l[0], r[0], w[0]);

	// update
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	// outputs
	*data0 = y0;
	*data1 = y1;
}

static
void vert_2x1_x16(
        FIX16_T *data0, // left [1]
        FIX16_T *data1, // right [1]
        FIX16_T *buff // [4]
)
{
// 	const FIX16_T w[4] = {
// 		conv_float32_to_fix16(+dwt_cdf97_u2_s),
// 		conv_float32_to_fix16(-dwt_cdf97_p2_s),
// 		conv_float32_to_fix16(+dwt_cdf97_u1_s),
// 		conv_float32_to_fix16(-dwt_cdf97_p1_s)
// 	}; // [ u2 p2 u1 p1 ]

// 	printf("%i, %i, %i, %i\n",
// 		conv_float32_to_fix16(+dwt_cdf97_u2_s),
// 		conv_float32_to_fix16(-dwt_cdf97_p2_s),
// 		conv_float32_to_fix16(+dwt_cdf97_u1_s),
// 		conv_float32_to_fix16(-dwt_cdf97_p1_s)
// 	);
	const FIX16_T w[4] = { 227, 452, -27, -812 };

	// variables
	FIX16_T c[4];
	FIX16_T r[4];
	FIX16_T x0, x1;
	FIX16_T y0, y1;

	// load
	FIX16_T *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0   = l[0];
	c[0] = l[1];
	c[1] = l[2];
	c[2] = l[3];
	c[3] = x0;

	// operation
	r[3] = x1;
	r[2] = c[3] + op_x16(l[3], r[3], w[3]);
	r[1] = c[2] + op_x16(l[2], r[2], w[2]);
	r[0] = c[1] + op_x16(l[1], r[1], w[1]);
	y1   = c[0] + op_x16(l[0], r[0], w[0]);

	// update
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	// outputs
	*data0 = y0;
	*data1 = y1;
}

static
void scale_2x2_f32(float *t)
{
	const float zeta2 = dwt_cdf97_s1_s*dwt_cdf97_s1_s;
	const float v[4] = {
		1/zeta2, 1.f,
		1.f,   zeta2
	};
	for(int i = 0; i < 4; i++)
		t[i] *= v[i];
}

static
void cdf53_scale_2x2_f32(float *t)
{
	const float zeta2 = dwt_cdf53_s1_s*dwt_cdf53_s1_s;
	const float v[4] = {
		1/zeta2, 1.f,
		1.f,   zeta2
	};
	for(int i = 0; i < 4; i++)
		t[i] *= v[i];
}

static
void scale_2x2_x32(FIX32_T *t)
{
	//const FIX32_T z2_x32 = conv_float32_to_fix32( dwt_cdf97_s1_s*dwt_cdf97_s1_s );
	//const FIX32_T r2_x32 = conv_float32_to_fix32( 1.f / (dwt_cdf97_s1_s*dwt_cdf97_s1_s) );
	const FIX32_T z2_x32 = 86612;
	const FIX32_T r2_x32 = 49589;

	const FIX32_T v[4] = {
		r2_x32, FIX32_ONE,
		FIX32_ONE, z2_x32
	};
	for(int i = 0; i < 4; i++)
		t[i] = fix32_mul(t[i], v[i]);
}

static
void scale_2x2_x16(FIX16_T *t)
{
// 	const FIX16_T z2_x16 = conv_float32_to_fix16( dwt_cdf97_s1_s*dwt_cdf97_s1_s );
// 	const FIX16_T r2_x16 = conv_float32_to_fix16( 1.f / (dwt_cdf97_s1_s*dwt_cdf97_s1_s) );
// 	printf("%i %i\n", z2_x16, r2_x16);
	const FIX16_T z2_x16 = 677;
	const FIX16_T r2_x16 = 387;

	const FIX16_T v[4] = {
		r2_x16, FIX16_ONE,
		FIX16_ONE, z2_x16
	};
	for(int i = 0; i < 4; i++)
		t[i] = fix16_mul(t[i], v[i]);
}

static
void cores2f_cdf97_v2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	float *buffer_x_ptr,
	float *buffer_y_ptr
)
{
	const int overlap_x_L = 5;
	const int overlap_y_L = 5;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

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
	vert_2x1_f32(t+0, t+1, buffer_y_ptr+0);
	vert_2x1_f32(t+2, t+3, buffer_y_ptr+4);
	vert_2x1_f32(t+0, t+2, buffer_x_ptr+0);
	vert_2x1_f32(t+1, t+3, buffer_x_ptr+4);

	// scaling
	scale_2x2_f32(t);

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
void cores2f_cdf53_v2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	float *buffer_x_ptr,
	float *buffer_y_ptr
)
{
	const int overlap_x_L = 3;
	const int overlap_y_L = 3;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 2;

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
	cdf53_vert_2x1_f32(t+0, t+1, buffer_y_ptr+0);
	cdf53_vert_2x1_f32(t+2, t+3, buffer_y_ptr+2);
	cdf53_vert_2x1_f32(t+0, t+2, buffer_x_ptr+0);
	cdf53_vert_2x1_f32(t+1, t+3, buffer_x_ptr+2);

	// scaling
	cdf53_scale_2x2_f32(t);

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
void cores2i_cdf97_v2x2_f32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	float *buffer_x_ptr,
	float *buffer_y_ptr
)
{
	const int overlap_x_L = 4;
	const int overlap_y_L = 4;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

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

	// scaling
	scale_2x2_f32(t);

	// calc
	vert_2x1_inv_f32(t+0, t+1, buffer_y_ptr+0);
	vert_2x1_inv_f32(t+2, t+3, buffer_y_ptr+4);
	vert_2x1_inv_f32(t+0, t+2, buffer_x_ptr+0);
	vert_2x1_inv_f32(t+1, t+3, buffer_x_ptr+4);

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
void cores2f_cdf97_v2x2_i32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	int32_t *buffer_x_ptr,
	int32_t *buffer_y_ptr
)
{
	const int overlap_x_L = 5;
	const int overlap_y_L = 5;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

	// 2x2
	int32_t t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			t[yy*step_x+xx] = *addr2_i(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// calc
	vert_2x1_i32(t+0, t+1, buffer_y_ptr+0);
	vert_2x1_i32(t+2, t+3, buffer_y_ptr+4);
	vert_2x1_i32(t+0, t+2, buffer_x_ptr+0);
	vert_2x1_i32(t+1, t+3, buffer_x_ptr+4);

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

			*addr2_i(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = t[yy*step_x+xx];
		}
	}
}

static
void cores2f_cdf97_v2x2_x32_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	FIX32_T *buffer_x_ptr,
	FIX32_T *buffer_y_ptr
)
{
	const int overlap_x_L = 5;
	const int overlap_y_L = 5;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

	// 2x2
	FIX32_T t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			t[yy*step_x+xx] = *addr2_i(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// calc
	vert_2x1_x32(t+0, t+1, buffer_y_ptr+0);
	vert_2x1_x32(t+2, t+3, buffer_y_ptr+4);
	vert_2x1_x32(t+0, t+2, buffer_x_ptr+0);
	vert_2x1_x32(t+1, t+3, buffer_x_ptr+4);

	// scaling
	scale_2x2_x32(t);

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

			*addr2_i(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = t[yy*step_x+xx];
		}
	}
}

static
void cores2f_cdf97_v2x2_x16_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	FIX16_T *buffer_x_ptr,
	FIX16_T *buffer_y_ptr
)
{
	const int overlap_x_L = 5;
	const int overlap_y_L = 5;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

	// 2x2
	FIX16_T t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			// virtual to real coordinates
			const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
			const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

			t[yy*step_x+xx] = *addr2_i16(src->ptr, pos_y, pos_x, src->stride_y, src->stride_x);
		}
	}

	// calc
	vert_2x1_x16(t+0, t+1, buffer_y_ptr+0);
	vert_2x1_x16(t+2, t+3, buffer_y_ptr+4);
	vert_2x1_x16(t+0, t+2, buffer_x_ptr+0);
	vert_2x1_x16(t+1, t+3, buffer_x_ptr+4);

	// scaling
	scale_2x2_x16(t);

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

			*addr2_i16(dst->ptr, pos_y, pos_x, dst->stride_y, dst->stride_x) = t[yy*step_x+xx];
		}
	}
}

void dump_buffer_f32(float *buffer, int super)
{
	const int buff_elem_size = 4;

	for(int i = 0; i < super; i++)
	{
		printf("[ ");
		for(int j = 0; j < buff_elem_size; j++)
		{
			printf("%+f ", buffer[ i * buff_elem_size + j ]);
		}
		printf("]\n");
	}
}

void cores2f_cdf97_v2x2_f32(
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

	float buffer_x[buff_elem_size*super_x];
	float buffer_y[buff_elem_size*super_y];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf97_v2x2_f32_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}

void cores2f_cdf53_v2x2_f32(
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

	float buffer_x[buff_elem_size*super_x];
	float buffer_y[buff_elem_size*super_y];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf53_v2x2_f32_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}

void cores2i_cdf97_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int buff_elem_size = 4;

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 4;
	const int overlap_y_L = 4;
	const int overlap_x_R = 4;
	const int overlap_y_R = 4;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	float buffer_x[buff_elem_size*super_x];
	float buffer_y[buff_elem_size*super_y];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2i_cdf97_v2x2_f32_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}

void cores2f_cdf97_v2x2_i32(
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

	int32_t buffer_x[buff_elem_size*super_x];
	int32_t buffer_y[buff_elem_size*super_y];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf97_v2x2_i32_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}

void cores2f_cdf97_v2x2_x32(
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

	FIX32_T buffer_x[buff_elem_size*super_x];
	FIX32_T buffer_y[buff_elem_size*super_y];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf97_v2x2_x32_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}

void cores2f_cdf97_v2x2_x16(
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

	FIX16_T buffer_x[buff_elem_size*super_x];
	FIX16_T buffer_y[buff_elem_size*super_y];

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			cores2f_cdf97_v2x2_x16_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}
