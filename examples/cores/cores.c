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
UNUSED_FUNC
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

// JPEG 2000 compatible
// ITU-T Rec. T.800 (2000 FCDV1.0), p. 126
// F.3.8.1 Reversible 1D filtering
// returns the output of the 1D_FILT_R procedure
static
void cdf53_vert_2x1_i16(
        int16_t *data0, // left [1]
        int16_t *data1, // right [1]
        int16_t *buff // [2]
)
{
	int16_t d1_x1 = buff[0]; // 16 bits
	int16_t s0_x1 = buff[1]; // 16 bits

	int16_t d0_x0 = *data0; // 16 bits
	int16_t s0_x0 = *data1; // 16 bits

	int16_t d1_x0 = d0_x0 - ( ( s0_x1 + s0_x0 + 0 ) >> 1 ); // (1)
	int16_t s2_x1 = s0_x1 + ( ( d1_x1 + d1_x0 + 2 ) >> 2 ); // (2)

	buff[0] = d1_x0;
	buff[1] = s0_x0;

	*data0 = s2_x1;
	*data1 = d1_x0;
}

// JPEG 2000 compatible
// ITU-T Rec. T.800 (2000 FCDV1.0), p. 126
// F.3.8.1 Reversible 1D filtering
// returns the output of the 1D_FILT_R procedure
static
void cdf53_vert_2x1B_i16(
        int16_t *data0, // left
        int16_t *data1, // right
        int16_t *buff // []
)
{
	int16_t d0_x0 = *data0; // 16 bits
	int16_t s0_x0 = *data1; // 16 bits

	int16_t s0_x1 = buff[0]; // 15 bits
	int16_t s1_x1 = buff[1]; // 16 bits
	int16_t b0_x1 = buff[2]; // 1 bit

	int16_t c1 = 1 & s0_x0 & s0_x1; // 1 bit AND
	int16_t c2 = 2 - b0_x1 - c1; // 2 bit LUT

	int16_t s1_x0 = (s0_x0<<2) - (s0_x0&~1) + d0_x0 - (s0_x1>>1); // (10)
	int16_t b0_x0 = c1; // (11)
	int16_t s2_x1 = ( s1_x1 + d0_x0 - (s0_x0>>1) + c2 ) >> 2; // (12)
	int16_t d1_x0 = d0_x0 - ( ( s0_x1 + s0_x0 ) >> 1 ); // (5)

	*data0 = s2_x1;
	*data1 = d1_x0;

	buff[0] = s0_x0;
	buff[1] = s1_x0;
	buff[2] = b0_x0;
}

static
void cdf53_vert_2x1_inv_i16(
        int16_t *data0, // left [1]
        int16_t *data1, // right [1]
        int16_t *buff // [2]
)
{
	int16_t c[2];
	int16_t r[2];
	int16_t x0, x1;
	int16_t y0, y1;

	int16_t *l = buff;

	x0 = *data0;
	x1 = *data1;

	y0   = l[0];
	c[0] = l[1];
	c[1] = x0;

	r[1] = x1;
	r[0] = c[1] - ( ( l[1] + r[1] + 2 ) >> 2 );
	y1   = c[0] + ( ( l[0] + r[0] + 0 ) >> 1 );

	l[0] = r[0];
	l[1] = r[1];

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

// lag=1 steps=2
static
void cdf53_vert_2x1A_f32(
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
	float y1;

	// load
	float *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
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
	*data0 = y1;
	*data1 = r[0];
}

// not a lifting scheme
static
void cdf53_vert_2x1B_f32(
        float *data0, // left [1]
        float *data1, // right [1]
        float *buff // [2]
)
{
	const float alpha = -dwt_cdf53_p1_s;
	const float beta  = +dwt_cdf53_u1_s;
	const float gamma = 1.f + 2*alpha*beta;

	float d0_x0 = *data0;
	float s0_x0 = *data1;
	float s0_x1 = buff[0];
	float s1_x1 = buff[1];

	float s1_x0 =
		+ alpha*beta * s0_x1
		+ beta * d0_x0
		+ gamma * s0_x0;

	float s2_x1 = s1_x1
		+ beta * d0_x0
		+ alpha*beta * s0_x0;

	float d1_x0 = d0_x0
		+ alpha * s0_x1
		+ alpha * s0_x0;

	*data0 = s2_x1;
	*data1 = d1_x0;
	buff[0] = s0_x0;
	buff[1] = s1_x0;
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

// M. D. Adams. Reversible Integer-to-Integer Wavelet Transforms for Image Coding. PhD thesis, 2002.
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
void cdf53_scale2_2x2_f32(float *t)
{
	const float zeta2 = dwt_cdf53_s1_s*dwt_cdf53_s1_s;
	const float v[4] = {
		zeta2, 1.f,
		1.f,   1/zeta2
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

// JPEG 2000 compatible
// F.3.2 The 2D_SD procedure
// ITU-T Rec. T.800 (2000 FCDV1.0), p. 121
// returns outputs of the 2D_SD procedure
static
void cores2f_cdf53_v2x2_i16_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	int16_t *buffer_x_ptr,
	int16_t *buffer_y_ptr
)
{
	const int overlap_x_L = 3;
	const int overlap_y_L = 3;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 1;
	const int buff_elem_size = 4; // FIXME: 2

	// 2x2
	int16_t t[4];

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

#if 0
	// vertical
	cdf53_vert_2x1_i16(t+0, t+2, buffer_x_ptr+0*buff_elem_size);
	cdf53_vert_2x1_i16(t+1, t+3, buffer_x_ptr+1*buff_elem_size);

	// horizontal
	cdf53_vert_2x1_i16(t+0, t+1, buffer_y_ptr+0*buff_elem_size);
	cdf53_vert_2x1_i16(t+2, t+3, buffer_y_ptr+1*buff_elem_size);
#endif
#if 1
	// vertical
	cdf53_vert_2x1B_i16(t+0, t+2, buffer_x_ptr+0*buff_elem_size);
	cdf53_vert_2x1B_i16(t+1, t+3, buffer_x_ptr+1*buff_elem_size);

	// horizontal
	cdf53_vert_2x1B_i16(t+0, t+1, buffer_y_ptr+0*buff_elem_size);
	cdf53_vert_2x1B_i16(t+2, t+3, buffer_y_ptr+1*buff_elem_size);
#endif

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

static
void cores2i_cdf53_v2x2_i16_core(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	int16_t *buffer_x_ptr,
	int16_t *buffer_y_ptr
)
{
	const int overlap_x_L = 2;
	const int overlap_y_L = 2;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 2;

	// 2x2
	int16_t t[4];

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

	// horizontal
	cdf53_vert_2x1_inv_i16(t+0, t+1, buffer_y_ptr+0);
	cdf53_vert_2x1_inv_i16(t+2, t+3, buffer_y_ptr+2);

	// vertical
	cdf53_vert_2x1_inv_i16(t+0, t+2, buffer_x_ptr+0);
	cdf53_vert_2x1_inv_i16(t+1, t+3, buffer_x_ptr+2);

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

static
void cores2f_cdf53_v2x2B_f32_core(
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

	const int shift = 1;

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

#if 0
	// separable reference
	cdf53_vert_2x1A_f32(t+0, t+1, buffer_y_ptr+0);
	cdf53_vert_2x1A_f32(t+2, t+3, buffer_y_ptr+2);
	cdf53_vert_2x1A_f32(t+0, t+2, buffer_x_ptr+0);
	cdf53_vert_2x1A_f32(t+1, t+3, buffer_x_ptr+2);
#endif
#if 1
	// separable

	const float alpha = -dwt_cdf53_p1_s;
	const float beta  = +dwt_cdf53_u1_s;

	float *buff_y = buffer_y_ptr;
	float *buff_x = buffer_x_ptr;

	float d0_x0y0 = t[0];
	float v0_x0y0 = t[1];
	float h0_x0y0 = t[2];
	float a0_x0y0 = t[3];

	float d1_x1y0 = buff_y[0];
	float v0_x1y0 = buff_y[1];
	float h1_x1y0 = buff_y[2];
	float a0_x1y0 = buff_y[3];

	float v2_x1y1 = buff_x[0];
	float a1_x1y1 = buff_x[1];
	float d2_x0y1 = buff_x[2];
	float h1_x0y1 = buff_x[3];

	float d1_x0y0 = d0_x0y0 + alpha * ( v0_x1y0 + v0_x0y0 );
	float v1_x1y0 = v0_x1y0 + beta  * ( d1_x1y0 + d1_x0y0 );

	float h1_x0y0 = h0_x0y0 + alpha * ( a0_x1y0 + a0_x0y0 );
	float a1_x1y0 = a0_x1y0 + beta  * ( h1_x1y0 + h1_x0y0 );

	float v2_x1y0 = v1_x1y0 + alpha * ( a1_x1y1 + a1_x1y0 );
	float a2_x1y1 = a1_x1y1 + beta  * ( v2_x1y1 + v2_x1y0 );

	float d2_x0y0 = d1_x0y0 + alpha * ( h1_x0y1 + h1_x0y0 );
	float h2_x0y1 = h1_x0y1 + beta  * ( d2_x0y1 + d2_x0y0 );

	buff_y[0] = d1_x0y0;
	buff_y[1] = v0_x0y0;
	buff_y[2] = h1_x0y0;
	buff_y[3] = a0_x0y0;

	buff_x[0] = v2_x1y0;
	buff_x[1] = a1_x1y0;
	buff_x[2] = d2_x0y0;
	buff_x[3] = h1_x0y0;

	t[0] = a2_x1y1;
	t[1] = h2_x0y1;
	t[2] = v2_x1y0;
	t[3] = d2_x0y0;
#endif
#if 0
	// separable with reduced latency (reference)

	cdf53_vert_2x1B_f32(t+0, t+1, buffer_y_ptr+0);
	cdf53_vert_2x1B_f32(t+2, t+3, buffer_y_ptr+2);
	cdf53_vert_2x1B_f32(t+0, t+2, buffer_x_ptr+0);
	cdf53_vert_2x1B_f32(t+1, t+3, buffer_x_ptr+2);
#endif
#if 0
	// separable with reduced latency

	float *buff_y = buffer_y_ptr;
	float *buff_x = buffer_x_ptr;

	const float alpha = -dwt_cdf53_p1_s;
	const float beta  = +dwt_cdf53_u1_s;

	float d0_x0y0 = t[0];
	float v0_x0y0 = t[1];
	float h0_x0y0 = t[2];
	float a0_x0y0 = t[3];

	float v0_x1y0 = buff_y[0];
	float v1_x1y0 = buff_y[1];
	float a0_x1y0 = buff_y[2];
	float a1_x1y0 = buff_y[3];

	float a2_x1y1 = buff_x[0];
	float a3_x1y1 = buff_x[1];
	float h1_x0y1 = buff_x[2];
	float h2_x0y1 = buff_x[3];

	float v1_x0y0 = v0_x0y0
		+ alpha*beta * v0_x1y0
		+ beta * d0_x0y0
		+ 2*alpha*beta * v0_x0y0;

	float v2_x1y0 = v1_x1y0
		+ beta * d0_x0y0
		+ alpha*beta * v0_x0y0;

	float d1_x0y0 = d0_x0y0
		+ alpha * v0_x1y0
		+ alpha * v0_x0y0;

	float a1_x0y0 = a0_x0y0
		+ alpha*beta * a0_x1y0
		+ beta * h0_x0y0
		+ 2*alpha*beta * a0_x0y0;

	float a2_x1y0 = a1_x1y0
		+ beta * h0_x0y0
		+ alpha*beta * a0_x0y0;

	float h1_x0y0 = h0_x0y0
		+ alpha * a0_x1y0
		+ alpha * a0_x0y0;

	// ---

	float a3_x1y0 = a2_x1y0
		+ alpha*beta * a2_x1y1
		+ beta * v2_x1y0
		+ 2*alpha*beta * a2_x1y0;

	float a4_x1y1 = a3_x1y1
		+ beta * v2_x1y0
		+ alpha*beta * a2_x1y0;

	float v3_x1y0 = v2_x1y0
		+ alpha * a2_x1y1
		+ alpha * a2_x1y0;

	float h2_x0y0 = h1_x0y0
		+ alpha*beta * h1_x0y1
		+ beta * d1_x0y0
		+ 2*alpha*beta * h1_x0y0;

	float h3_x0y1 = h2_x0y1
		+ beta * d1_x0y0
		+ alpha*beta * h1_x0y0;

	float d2_x0y0 = d1_x0y0
		+ alpha * h1_x0y1
		+ alpha * h1_x0y0;

	buff_y[0] = v0_x0y0;
	buff_y[1] = v1_x0y0;
	buff_y[2] = a0_x0y0;
	buff_y[3] = a1_x0y0;

	buff_x[0] = a2_x1y0;
	buff_x[1] = a3_x1y0;
	buff_x[2] = h1_x0y0;
	buff_x[3] = h2_x0y0;

	t[0] = a4_x1y1;
	t[1] = h3_x0y1;
	t[2] = v3_x1y0;
	t[3] = d2_x0y0;
#endif
#if 0
	// NSLS Iwahashi2013

	float *buff_y = buffer_y_ptr;
	float *buff_x = buffer_x_ptr;

	const float alpha = -dwt_cdf53_p1_s;
	const float beta  = +dwt_cdf53_u1_s;

	float a0_x0y0 = t[3];
	float v0_x0y0 = t[1];
	float h0_x0y0 = t[2];
	float d0_x0y0 = t[0];

	// pop from buffers
	float h0_x0y1 = buff_x[0];
	float a0_x0y1 = buff_x[1];
	float v0_x1y0 = buff_y[0];
	float a0_x1y0 = buff_y[1];
	float a0_x1y1 = buff_y[2];
	float d1_x0y1 = buff_x[2];
	float d1_x1y0 = buff_y[3];
	float d1_x1y1 = buff_x[3];
	float h1_x1y1 = buff_y[4];
	float v1_x1y1 = buff_x[4];

	float d1_x0y0 = d0_x0y0
		+ alpha*alpha * ( a0_x0y0 + a0_x0y1 + a0_x1y0 + a0_x1y1 )
		+ alpha       * ( h0_x0y0 + h0_x0y1 )
		+ alpha       * ( v0_x0y0 + v0_x1y0 )
	;

	float h1_x0y1 = h0_x0y1
		+ alpha * ( a0_x1y1 + a0_x0y1 )
		+ beta  * ( d1_x0y1 + d1_x0y0 )
	;

	float v1_x1y0 = v0_x1y0
		+ alpha * ( a0_x1y1 + a0_x1y0 )
		+ beta  * ( d1_x1y0 + d1_x0y0 )
	;

	float a1_x1y1 = a0_x1y1
		- beta*beta * ( d1_x0y0 + d1_x0y1 + d1_x1y0 + d1_x1y1 )
		+ beta      * ( h1_x1y1 + h1_x0y1 )
		+ beta      * ( v1_x1y1 + v1_x1y0 )
	;

	// push into buffers
	buff_x[0] = h0_x0y0;
	buff_x[1] = a0_x0y0;
	buff_x[2] = d1_x0y0;
	buff_x[3] = d1_x1y0;
	buff_y[4] = h1_x0y1;
	buff_y[0] = v0_x0y0;
	buff_y[1] = a0_x0y0;
	buff_y[2] = a0_x0y1;
	buff_y[3] = d1_x0y0;
	buff_x[4] = v1_x1y0;

	t[0] = a1_x1y1;
	t[1] = h1_x0y1;
	t[2] = v1_x1y0;
	t[3] = d1_x0y0;
#endif
#if 0
	// 2 steps, 22 MACs

	float *buff_y = buffer_y_ptr;
	float *buff_x = buffer_x_ptr;

	const float alpha = -dwt_cdf53_p1_s;
	const float beta  = +dwt_cdf53_u1_s;

	float a0_x0y0 = t[3];
	float v0_x0y0 = t[1];
	float h0_x0y0 = t[2];
	float d0_x0y0 = t[0];

	// pop from buffers
	float h0_x0y1 = buff_x[0]; // h0_x0y0->h0_x0y1 -- vertical
	float a0_x0y1 = buff_x[1]; // a0_x0y0->a0_x0y1 -- vertical (+diagonal a0_x0y0->a0_x1y1 part 1)
	float v0_x1y0 = buff_y[0]; // v0_x0y0->v0_x1y0 -- horizontal
	float a0_x1y0 = buff_y[1]; // a0_x0y0->a0_x1y0 -- horizontal
	float a0_x1y1 = buff_y[2]; // a0_x0y0->a0_x1y1 -- diagonal (part 2)
	float v1_x1y0 = buff_y[5]; // v1_x0y0->v1_x1y0 -- horizontal
	float h1_x0y1 = buff_x[5]; // h1_x0y0->h1_x0y1 -- vertical
	float a1_x0y1 = buff_x[6]; // a1_x0y0->a1_x1y1 -- diagonal (part 1)
	float a1_x1y1 = buff_y[7]; // a1_x0y0->a1_x1y1 -- diagonal (part 2)

	// D-*
	float d1_x0y0 = d0_x0y0
		+ alpha*alpha * ( a0_x0y0 + a0_x0y1 + a0_x1y0 + a0_x1y1 )
		+ alpha       * ( h0_x0y0 + h0_x0y1 )
		+ alpha       * ( v0_x0y0 + v0_x1y0 )
	;

	float v1a_x0y0 = v0_x0y0 + alpha * ( a0_x0y0 + a0_x0y1 );
	float h1a_x0y0 = h0_x0y0 + alpha * ( a0_x0y0 + a0_x1y0 );

	// ---

	float a3_x1y1 = a1_x1y1
		+ beta*beta * d1_x0y0
		+ beta * ( h1_x0y1 + v1_x1y0 );
	float h2_x0y1 = h1_x0y1
		+ beta * d1_x0y0;
	float v2_x1y0 = v1_x1y0
		+ beta * d1_x0y0;

	float a1_x0y0 = a0_x0y0
		+ beta*beta * d1_x0y0
		+ beta * ( v1a_x0y0 + h1a_x0y0 );
	float v1_x0y0 = v1a_x0y0
		+ beta * d1_x0y0;
	float h1_x0y0 = h1a_x0y0
		+ beta * d1_x0y0;

	// push into buffers
	buff_x[0] = h0_x0y0;
	buff_x[1] = a0_x0y0;
	buff_y[0] = v0_x0y0;
	buff_y[1] = a0_x0y0;
	buff_y[2] = a0_x0y1;
	buff_y[5] = v1_x0y0;
	buff_x[5] = h1_x0y0;
	buff_x[6] = a1_x0y0;
	buff_y[7] = a1_x0y1;

	t[0] = a3_x1y1;
	t[1] = h2_x0y1;
	t[2] = v2_x1y0;
	t[3] = d1_x0y0;
#endif

	// scaling
	cdf53_scale2_2x2_f32(t);

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

void cores2f_cdf53_v2x2_i16(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int buff_elem_size = 4; // FIXME: 2

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 3;
	const int overlap_y_L = 3;
	const int overlap_x_R = 3;
	const int overlap_y_R = 3;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

#if 0
	int16_t buffer_x[buff_elem_size*super_x];
	int16_t buffer_y[buff_elem_size*super_y];
#else
	int16_t *buffer_x = dwt_util_alloc(buff_elem_size*super_x, sizeof(int16_t));
	int16_t *buffer_y = dwt_util_alloc(buff_elem_size*super_y, sizeof(int16_t));

	assert( buffer_x && buffer_y );
#endif

	for(int y = 0; y+step_y-1 < super_y; y += step_y)
		for(int x = 0; x+step_x-1 < super_x; x += step_x)
			cores2f_cdf53_v2x2_i16_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}

void cores2i_cdf53_v2x2_i16(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int buff_elem_size = 2;

	const int step_x = 2;
	const int step_y = 2;

	const int overlap_x_L = 2;
	const int overlap_y_L = 2;
	const int overlap_x_R = 2;
	const int overlap_y_R = 2;

	const int super_x = overlap_x_L + src->size_x + overlap_x_R;
	const int super_y = overlap_y_L + src->size_y + overlap_y_R;

	int16_t buffer_x[buff_elem_size*super_x];
	int16_t buffer_y[buff_elem_size*super_y];

	for(int y = 0; y+step_y-1 < super_y; y += step_y)
		for(int x = 0; x+step_x-1 < super_x; x += step_x)
			cores2i_cdf53_v2x2_i16_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}

void cores2f_cdf53_v2x2B_f32(
	struct image_t *src,
	struct image_t *dst
)
{
	assert( src->size_x == dst->size_x && src->size_y == dst->size_y );

	const int buff_elem_size = 8; // FIXME: 2

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

	for(int y = 0; y+step_y-1 < super_y; y += step_y)
		for(int x = 0; x+step_x-1 < super_x; x += step_x)
			cores2f_cdf53_v2x2B_f32_core(src, dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
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
