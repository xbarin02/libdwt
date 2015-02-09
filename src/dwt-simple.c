#include "dwt-simple.h"
#include "libdwt.h"
#include "inline.h"
#include <math.h>

static
void op4s_sdl_shuffle_s_ref(float *c, float *r)
{
	c[0] = c[1];
	c[1] = c[2];
	c[2] = c[3];

	r[0] = r[1];
	r[1] = r[2];
	r[2] = r[3];
}

static
void op2s_sdl_shuffle_s_ref(float *c, float *r)
{
	c[0] = c[1];
	r[0] = r[1];
}

static
void op4s_sdl_load_stride_s_ref(float *x, const float *addr, int stride)
{
	x[0] = *addr1_const_s(addr, 0, stride);
	x[1] = *addr1_const_s(addr, 1, stride);
}

static
void op2s_sdl_load_stride_s_ref(float *x, const float *addr, int stride)
{
	x[0] = *addr1_const_s(addr, 0, stride);
	x[1] = *addr1_const_s(addr, 1, stride);
}

static
void op4s_sdl_save_stride_s_ref(float *y, float *addr, int stride)
{
	*addr1_s(addr, 0, stride) = y[0];
	*addr1_s(addr, 1, stride) = y[1];
}

static
void op2s_sdl_save_stride_s_ref(float *y, float *addr, int stride)
{
	*addr1_s(addr, 0, stride) = y[0];
	*addr1_s(addr, 1, stride) = y[1];
}

static
void op4s_sdl_input_s_ref(const float *x, float *c, float *r)
{
	c[3] = x[0];
	r[3] = x[1];
}

static
void op2s_sdl_input_s_ref(const float *x, float *c, float *r)
{
	c[1] = x[0];
	r[1] = x[1];
}

static
void op4s_sdl_output_s_ref(float *y, const float *l, const float *z)
{
	y[0] = l[0];
	y[1] = z[0];
}

static
void op2s_sdl_output_s_ref(float *y, const float *l, const float *z)
{
	y[0] = l[0];
	y[1] = z[0];
}

static
void op4s_sdl_scale_s_ref(float *y, const float *v)
{
	y[0] *= v[0];
	y[1] *= v[1];
}

static
void op2s_sdl_scale_s_ref(float *y, const float *v)
{
	y[0] *= v[0];
	y[1] *= v[1];
}

static
void op4s_sdl_op_s_ref(float *z, const float *c, const float *w, const float *l, const float *r)
{
	z[3] = c[3] + w[3] * ( l[3] + r[3] );
	z[2] = c[2] + w[2] * ( l[2] + r[2] );
	z[1] = c[1] + w[1] * ( l[1] + r[1] );
	z[0] = c[0] + w[0] * ( l[0] + r[0] );
}

static
void op2s_sdl_op_s_ref(float *z, const float *c, const float *w, const float *l, const float *r)
{
	z[1] = c[1] + w[1] * ( l[1] + r[1] );
	z[0] = c[0] + w[0] * ( l[0] + r[0] );
}

static
void op2s_sdl_op_s_ref_eaw(
	float *z,
	const float *c,
	const float *w, // [beta] [alpha]
	const float *l,
	const float *r,
	const float *eaw_w // [betaL] [betaR] [-] [alphaL] [alphaR]
)
{
	z[1] = c[1] + ( eaw_w[3]*l[1] + eaw_w[4]*r[1] ) / (eaw_w[3]+eaw_w[4]) * (2.f*w[1]); // alpha
	z[0] = c[0] + ( eaw_w[0]*l[0] + eaw_w[1]*r[0] ) / (eaw_w[0]+eaw_w[1]) * (2.f*w[0]); // beta
}

static
void op4s_sdl_update_s_ref(float *c, float *l, float *r, const float *z)
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
void op2s_sdl_update_s_ref(float *c, float *l, float *r, const float *z)
{
	c[0] = l[0];
	c[1] = l[1];

	l[0] = r[0];
	l[1] = r[1];

	r[0] = z[0];
	r[1] = z[1];
}

static
void fdwt_cdf97_diagonal_prolog_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride
)
{
	UNUSED(v);
	UNUSED(y);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_stride_s_ref(x, addr1_s(*addr, +4, stride), stride);

	// (descale)

	// input
	op4s_sdl_input_s_ref(x, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// (output)

	// (scale)

	// (save)

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr, 2, stride);
}

static
void fdwt_cdf53_diagonal_prolog_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride
)
{
	UNUSED(v);
	UNUSED(y);

	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// load
	op2s_sdl_load_stride_s_ref(x, addr1_s(*addr, +2, stride), stride);

	// (descale)

	// input
	op2s_sdl_input_s_ref(x, c, r);

	// operation
	op2s_sdl_op_s_ref(z, c, w, l, r);

	// (output)

	// (scale)

	// (save)

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr, 2, stride);
}

static
void fdwt_eaw53_diagonal_prolog_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride,
	const float *eaw_w
)
{
	UNUSED(v);
	UNUSED(y);

	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// load
	op2s_sdl_load_stride_s_ref(x, addr1_s(*addr, +2, stride), stride);

	// (descale)

	// input
	op2s_sdl_input_s_ref(x, c, r);

	// operation
	op2s_sdl_op_s_ref_eaw(z, c, w, l, r, eaw_w);

	// (output)

	// (scale)

	// (save)

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr, 2, stride);
}

static
void fdwt_cdf97_diagonal_epilog_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride
)
{
	UNUSED(x);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// (load)

	// (descale)

	// (input)

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op4s_sdl_output_s_ref(y, l, z);

	// scale
	op4s_sdl_scale_s_ref(y, v);

	// save
	op4s_sdl_save_stride_s_ref(y, addr1_s(*addr, -6, stride), stride);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) = addr1_s(*addr, 2, stride);
}

static
void fdwt_cdf53_diagonal_epilog_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride
)
{
	UNUSED(x);

	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// (load)

	// (descale)

	// (input)

	// operation
	op2s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op2s_sdl_output_s_ref(y, l, z);

	// scale
	op2s_sdl_scale_s_ref(y, v);

	// save
	op2s_sdl_save_stride_s_ref(y, addr1_s(*addr, -2, stride), stride);

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) = addr1_s(*addr, 2, stride);
}

static
void fdwt_eaw53_diagonal_epilog_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride,
	const float *eaw_w
)
{
	UNUSED(x);

	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// (load)

	// (descale)

	// (input)

	// operation
	op2s_sdl_op_s_ref_eaw(z, c, w, l, r, eaw_w);

	// output
	op2s_sdl_output_s_ref(y, l, z);

	// scale
	op2s_sdl_scale_s_ref(y, v);

	// save
	op2s_sdl_save_stride_s_ref(y, addr1_s(*addr, -2, stride), stride);

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) = addr1_s(*addr, 2, stride);
}

static
void fdwt_cdf97_short_s(
	float *arr,
	int N,
	int stride
)
{
	assert( arr );

	float alpha = -dwt_cdf97_p1_s;
	float beta  = +dwt_cdf97_u1_s;
	float gamma = -dwt_cdf97_p2_s;
	float delta = +dwt_cdf97_u2_s;
	float zeta  = +dwt_cdf97_s1_s;

	if( 1 == N )
	{
		*addr1_s(arr, 0, stride) *= zeta;
	}

	if( 2 == N )
	{

		// alpha
		*addr1_s(arr, 1, stride) += 2*alpha * (*addr1_s(arr, 0, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));

		// gamma
		*addr1_s(arr, 1, stride) += 2*gamma * (*addr1_s(arr, 0, stride));

		// delta
		*addr1_s(arr, 0, stride) += 2*delta * (*addr1_s(arr, 1, stride));

		// scaling
		*addr1_s(arr, 0, stride) *= zeta;
		*addr1_s(arr, 1, stride) *= 1/zeta;
	}

	if( 3 == N )
	{
		// alpha
		*addr1_s(arr, 1, stride) += alpha * (*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += 2*beta  * (*addr1_s(arr, 1, stride));

		// gamma
		*addr1_s(arr, 1, stride) += gamma * (*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

		// delta
		*addr1_s(arr, 0, stride) += 2*delta * (*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += 2*delta * (*addr1_s(arr, 1, stride));

		// scaling
		*addr1_s(arr, 0, stride) *= zeta;
		*addr1_s(arr, 1, stride) *= 1/zeta;
		*addr1_s(arr, 2, stride) *= zeta;
	}

	if( 4 == N )
	{
		// alpha
		*addr1_s(arr, 1, stride) += alpha * (*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
		*addr1_s(arr, 3, stride) += 2*alpha * (*addr1_s(arr, 2, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += beta  * (*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

		// gamma
		*addr1_s(arr, 1, stride) += gamma * (*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
		*addr1_s(arr, 3, stride) += 2*gamma * (*addr1_s(arr, 2, stride));

		// delta
		*addr1_s(arr, 0, stride) += 2*delta * (*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += delta * (*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

		// scaling
		*addr1_s(arr, 0, stride) *= zeta;
		*addr1_s(arr, 1, stride) *= 1/zeta;
		*addr1_s(arr, 2, stride) *= zeta;
		*addr1_s(arr, 3, stride) *= 1/zeta;
	}
}

static
void fdwt_cdf53_short_s(
	float *arr,
	int N,
	int stride
)
{
	assert( arr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	if( 1 == N )
	{
		*addr1_s(arr, 0, stride) *= zeta;
	}

	if( 2 == N )
	{

		// alpha
		*addr1_s(arr, 1, stride) += 2*alpha * (*addr1_s(arr, 0, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));

		// scaling
		*addr1_s(arr, 0, stride) *= zeta;
		*addr1_s(arr, 1, stride) *= 1/zeta;
	}
}

static
void fdwt_eaw53_short_s(
	float *arr,
	int N,
	int stride,
	const float *eaw_w
)
{
	assert( arr );
	UNUSED( eaw_w );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	if( 1 == N )
	{
		*addr1_s(arr, 0, stride) *= zeta;
	}

	if( 2 == N )
	{

		// alpha
		*addr1_s(arr, 1, stride) += 2*alpha * (*addr1_s(arr, 0, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));

		// scaling
		*addr1_s(arr, 0, stride) *= zeta;
		*addr1_s(arr, 1, stride) *= 1/zeta;
	}
}

static
void fdwt_cdf97_prolog_s(
	float *arr,
	int N,
	int stride
)
{
	assert( N >= 5 );
	assert( arr );

	float alpha = -dwt_cdf97_p1_s;
	float beta  = +dwt_cdf97_u1_s;
	float gamma = -dwt_cdf97_p2_s;
	float delta = +dwt_cdf97_u2_s;
	float zeta  = +dwt_cdf97_s1_s;

	// alpha
	*addr1_s(arr, 1, stride) += alpha * (*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
	*addr1_s(arr, 3, stride) += alpha * (*addr1_s(arr, 2, stride) + *addr1_s(arr, 4, stride));

	// beta
	*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));
	*addr1_s(arr, 2, stride) += beta * (*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

	// gamma
	*addr1_s(arr, 1, stride) += gamma * (*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

	// delta
	*addr1_s(arr, 0, stride) += 2*delta * (*addr1_s(arr, 1, stride));

	// scaling
	*addr1_s(arr, 0, stride) *= zeta;
}

static
void fdwt_cdf53_prolog_s(
	float *arr,
	int N,
	int stride
)
{
	assert( N >= 3 );
	assert( arr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	// alpha
	*addr1_s(arr, 1, stride) += alpha * (*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

	// beta
	*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));

	// scaling
	*addr1_s(arr, 0, stride) *= zeta;
}

static
void fdwt_eaw53_prolog_s(
	float *arr,
	int N,
	int stride,
	const float *eaw_w
)
{
	assert( N >= 3 );
	assert( arr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	// alpha
	*addr1_s(arr, 1, stride) += ( eaw_w[1-1] * *addr1_s(arr, 0, stride) + eaw_w[1+0] * *addr1_s(arr, 2, stride) ) / (eaw_w[1-1]+eaw_w[1+0]) * (2.f * alpha);

	// beta
	*addr1_s(arr, 0, stride) += 2*beta  * (*addr1_s(arr, 1, stride));

	// scaling
	*addr1_s(arr, 0, stride) *= zeta;
}

static
void fdwt_cdf97_vertical_core_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	float *l // [4]
)
{
	// constants
	const float w[4] = { delta, gamma, beta, alpha };
	const float v[2] = { 1/zeta, zeta };

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
	r[2] = c[3] + w[3] * (l[3] + r[3]);
	r[1] = c[2] + w[2] * (l[2] + r[2]);
	r[0] = c[1] + w[1] * (l[1] + r[1]);
	y[1] = c[0] + w[0] * (l[0] + r[0]);

	// scales
	y[0] *= v[0];
	y[1] *= v[1];

	// outputs
	*out0 = y[0];
	*out1 = y[1];

	// update
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];
}

static
void fdwt_cdf53_vertical_core_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float zeta,
	float *l // [2]
)
{
	// constants
	const float w[2] = { beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// aux. variables
	float x[2];
	float y[2];
	float r[2];
	float c[2];

	// inputs
	x[0] = *ptr0;
	x[1] = *ptr1;

	// shuffles
	y[0] = l[0];
	c[0] = l[1];
	c[1] = x[0];

	// operation
	r[1] = x[1];
	r[0] = c[1] + w[1] * (l[1] + r[1]);
	y[1] = c[0] + w[0] * (l[0] + r[0]);

	// scales
	y[0] *= v[0];
	y[1] *= v[1];

	// outputs
	*out0 = y[0];
	*out1 = y[1];

	// update
	l[0] = r[0];
	l[1] = r[1];
}

static
void fdwt_eaw53_vertical_core_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float zeta,
	float *l, // [2]
	const float *eaw_w // [3]: [0] = wL/beta, [1] = wR/beta = wL/alpha, [2] = wR/alpha
)
{
	// constants
	const float w[2] = { beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// aux. variables
	float x[2];
	float y[2];
	float r[2];
	float c[2];

	// inputs
	x[0] = *ptr0;
	x[1] = *ptr1;

	// shuffles
	y[0] = l[0];
	c[0] = l[1];
	c[1] = x[0];

	// operation
	r[1] = x[1];
	r[0] = c[1] + (eaw_w[1]*l[1]+eaw_w[2]*r[1]) / (eaw_w[1]+eaw_w[2]) * (2.f*w[1]); // alpha
	y[1] = c[0] + (eaw_w[0]*l[0]+eaw_w[1]*r[0]) / (eaw_w[0]+eaw_w[1]) * (2.f*w[0]); // beta

	// scales
	y[0] *= v[0];
	y[1] *= v[1];

	// outputs
	*out0 = y[0];
	*out1 = y[1];

	// update
	l[0] = r[0];
	l[1] = r[1];
}

void fdwt_cdf97_vertical_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( ptr );

	float alpha = -dwt_cdf97_p1_s;
	float beta  = +dwt_cdf97_u1_s;
	float gamma = -dwt_cdf97_p2_s;
	float delta = +dwt_cdf97_u2_s;
	float zeta  = +dwt_cdf97_s1_s;

	int pairs = (to_even(size)-4)/2;

	float *begin = addr1_s(ptr, 0, stride);

	assert( pairs >= 0 );

	// buffer
	float l[4];

	// prolog-vertical
	l[0] = *addr1_s(begin, 0, stride);
	l[1] = *addr1_s(begin, 1, stride);
	l[2] = *addr1_s(begin, 2, stride);
	l[3] = *addr1_s(begin, 3, stride);

	// init
	float *addr = addr1_s(begin, 4, stride);

	// loop by pairs from left to right
	for(int s = 0; s < pairs; s++)
	{
		fdwt_cdf97_vertical_core_s(
			addr1_s(addr, 0, stride),
			addr1_s(addr, 1, stride),
			addr1_s(addr, 0-4, stride),
			addr1_s(addr, 1-4, stride),
			alpha,
			beta,
			gamma,
			delta,
			zeta,
			l
		);

		// pointers
		addr = addr1_s(addr, 2, stride);
	}

	// epilog-vertical
	*addr1_s(addr, 0-4, stride) = l[0];
	*addr1_s(addr, 1-4, stride) = l[1];
	*addr1_s(addr, 2-4, stride) = l[2];
	*addr1_s(addr, 3-4, stride) = l[3];
}

void fdwt_cdf53_vertical_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( ptr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	int pairs = (to_even(size)-2)/2;

	float *begin = addr1_s(ptr, 0, stride);

	assert( pairs >= 0 );

	// buffer
	float l[2];

	// prolog-vertical
	l[0] = *addr1_s(begin, 0, stride);
	l[1] = *addr1_s(begin, 1, stride);

	// init
	float *addr = addr1_s(begin, 2, stride);

	// loop by pairs from left to right
	for(int s = 0; s < pairs; s++)
	{
		fdwt_cdf53_vertical_core_s(
			addr1_s(addr, 0, stride),
			addr1_s(addr, 1, stride),
			addr1_s(addr, 0-2, stride),
			addr1_s(addr, 1-2, stride),
			alpha,
			beta,
			zeta,
			l
		);

		// pointers
		addr = addr1_s(addr, 2, stride);
	}

	// epilog-vertical
	*addr1_s(addr, 0-2, stride) = l[0];
	*addr1_s(addr, 1-2, stride) = l[1];
}

void fdwt_eaw53_vertical_s(
	void *ptr,
	int size,
	int stride,
	const float *eaw_w
)
{
	assert( ptr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	int pairs = (to_even(size)-2)/2;

	float *begin = addr1_s(ptr, 0, stride);

	assert( pairs >= 0 );

	// buffer
	float l[2];

	// prolog-vertical
	l[0] = *addr1_s(begin, 0, stride);
	l[1] = *addr1_s(begin, 1, stride);

	// init
	float *addr = addr1_s(begin, 2, stride);

	// loop by pairs from left to right
	for(int s = 0; s < pairs; s++)
	{
		fdwt_eaw53_vertical_core_s(
			addr1_s(addr, 0, stride),
			addr1_s(addr, 1, stride),
			addr1_s(addr, 0-2, stride),
			addr1_s(addr, 1-2, stride),
			alpha,
			beta,
			zeta,
			l,
			&eaw_w[2*s]
		);

		// pointers
		addr = addr1_s(addr, 2, stride);
	}

	// epilog-vertical
	*addr1_s(addr, 0-2, stride) = l[0];
	*addr1_s(addr, 1-2, stride) = l[1];
}

void fdwt_cdf97_horizontal_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( ptr );

	float alpha = -dwt_cdf97_p1_s;
	float beta  = +dwt_cdf97_u1_s;
	float gamma = -dwt_cdf97_p2_s;
	float delta = +dwt_cdf97_u2_s;
	float zeta  = +dwt_cdf97_s1_s;

	int pairs = (to_even(size)-4)/2;

	float *begin = addr1_s(ptr, 0, stride);

	assert( pairs >= 0 );

	// constants
	const float w[4] = { delta, gamma, beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// operations
	for(int off = 4; off >= 1; off--)
	{
		float *out = addr1_s(begin, off, stride);
		const float c = w[off-1];

		for(int s = 0; s < pairs; s++)
		{
			*addr1_s(out, 0, stride) += c * (*addr1_s(out, -1, stride) + *addr1_s(out, +1, stride));

			out = addr1_s(out, 2, stride);
		}
	}

	// scale
	float *out = addr1_s(begin, 0, stride);

	for(int s = 0; s < pairs; s++)
	{
		*addr1_s(out, 0, stride) *= v[0];
		*addr1_s(out, 1, stride) *= v[1];

		out = addr1_s(out, 2, stride);
	}
}

void fdwt_cdf53_horizontal_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( ptr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	int pairs = (to_even(size)-2)/2;

	float *begin = addr1_s(ptr, 0, stride);

	assert( pairs >= 0 );

	// constants
	const float w[2] = { beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// operations
	for(int off = 2; off >= 1; off--)
	{
		float *out = addr1_s(begin, off, stride);
		const float c = w[off-1];

		for(int s = 0; s < pairs; s++)
		{
			*addr1_s(out, 0, stride) += c * (*addr1_s(out, -1, stride) + *addr1_s(out, +1, stride));

			out = addr1_s(out, 2, stride);
		}
	}

	// scale
	float *out = addr1_s(begin, 0, stride);

	for(int s = 0; s < pairs; s++)
	{
		*addr1_s(out, 0, stride) *= v[0];
		*addr1_s(out, 1, stride) *= v[1];

		out = addr1_s(out, 2, stride);
	}
}

static
float dwt_eaw_w(float n, float m, float alpha)
{
	const float eps = 1.0e-5f;

	return 1.f / (powf(fabsf(n-m), alpha) + eps);
}

static
void dwt_calc_eaw_w_stride_s(
	float *w,
	float *arr, int N, int stride,
	float alpha
)
{
	assert( w );
	assert( arr );

	for(int i = 0; i < N-1; i++)
	{
		w[i] = dwt_eaw_w(
			*addr1_s(arr, i+0, stride),
			*addr1_s(arr, i+1, stride),
			alpha);
	}
	w[N-1] = 0.f; // not necessary
}

void fdwt_eaw53_horizontal_s(
	void *ptr,
	int size,
	int stride,
	const float *eaw_w
)
{
	assert( ptr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	int pairs = (to_even(size)-2)/2;

	float *begin = addr1_s(ptr, 0, stride);

	assert( pairs >= 0 );

	// constants
	const float c[2] = { beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// operations
	for(int off = 2; off >= 1; off--)
	{
		float *out = addr1_s(begin, off, stride);
		const float coeff = c[off-1];

		for(int s = 0; s < pairs; s++)
		{
			float wL = eaw_w[off+2*s-1];
			float wR = eaw_w[off+2*s+0];

			*addr1_s(out, 0, stride) += ( wL * *addr1_s(out, -1, stride) + wR * *addr1_s(out, +1, stride) ) / (wL+wR) * (2.f * coeff);

			out = addr1_s(out, 2, stride);
		}
	}

	// scale
	float *out = addr1_s(begin, 0, stride);

	for(int s = 0; s < pairs; s++)
	{
		*addr1_s(out, 0, stride) *= v[0];
		*addr1_s(out, 1, stride) *= v[1];

		out = addr1_s(out, 2, stride);
	}
}

static
void fdwt_cdf97_diagonal_core_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride
)
{
	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_stride_s_ref(x, addr1_s(*addr, +4, stride), stride);

	// (descale)

	// input
	op4s_sdl_input_s_ref(x, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op4s_sdl_output_s_ref(y, l, z);

	// scale
	op4s_sdl_scale_s_ref(y, v);

	// save
	op4s_sdl_save_stride_s_ref(y, addr1_s(*addr, -6, stride), stride);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr, 2, stride);
}

static
void fdwt_cdf53_diagonal_core_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride
)
{
	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// load
	op2s_sdl_load_stride_s_ref(x, addr1_s(*addr, +2, stride), stride);

	// (descale)

	// input
	op2s_sdl_input_s_ref(x, c, r);

	// operation
	op2s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op2s_sdl_output_s_ref(y, l, z);

	// scale
	op2s_sdl_scale_s_ref(y, v);

	// save
	op2s_sdl_save_stride_s_ref(y, addr1_s(*addr, -2, stride), stride);

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr, 2, stride);
}

static
void fdwt_eaw53_diagonal_core_s(
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *x,
	float *y,
	float **addr,
	int stride,
	const float *eaw_w
)
{
	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// load
	op2s_sdl_load_stride_s_ref(x, addr1_s(*addr, +2, stride), stride);

	// (descale)

	// input
	op2s_sdl_input_s_ref(x, c, r);

	// operation
	op2s_sdl_op_s_ref_eaw(z, c, w, l, r, eaw_w);

	// output
	op2s_sdl_output_s_ref(y, l, z);

	// scale
	op2s_sdl_scale_s_ref(y, v);

	// save
	op2s_sdl_save_stride_s_ref(y, addr1_s(*addr, -2, stride), stride);

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr, 2, stride);
}

void fdwt_cdf97_diagonal_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( ptr );

	float alpha = -dwt_cdf97_p1_s;
	float beta  = +dwt_cdf97_u1_s;
	float gamma = -dwt_cdf97_p2_s;
	float delta = +dwt_cdf97_u2_s;
	float zeta  = +dwt_cdf97_s1_s;

	int pairs = (to_even(size)-4)/2;

	float *begin = addr1_s(ptr, 0,       stride);
	float *end   = addr1_s(ptr, 2*pairs, stride);

	if( pairs < 3 )
	{
		// NOTE: unfornunately, the diagonal vectorisation cannot handle less than 3 pairs of coefficients
		fdwt_cdf97_vertical_s(ptr, size, stride);
	}

	if( pairs >= 3 )
	{
		const float w[4] = { delta, gamma, beta, alpha };
		const float v[4] = { 1/zeta, zeta, 1/zeta, zeta };

		float l[4];
		float c[4];
		float r[4];
		float z[4];
		float x[4];
		float y[4];

		float *addr = begin;

		// prolog-diagonal
		l[3] = *addr1_const_s(begin, 3, stride);
		fdwt_cdf97_diagonal_prolog_s(w, v, l, c, r, z, x, y, &addr, stride);
		l[2] = *addr1_const_s(begin, 2, stride);
		fdwt_cdf97_diagonal_prolog_s(w, v, l, c, r, z, x, y, &addr, stride);
		l[1] = *addr1_const_s(begin, 1, stride);
		fdwt_cdf97_diagonal_prolog_s(w, v, l, c, r, z, x, y, &addr, stride);
		l[0] = *addr1_const_s(begin, 0, stride);

		// core
		for(int s = 0; s < pairs-3; s++)
		{
			fdwt_cdf97_diagonal_core_s(w, v, l, c, r, z, x, y, &addr, stride);
		}

		// epilog-diagonal
		*addr1_s(end, 3, stride) = l[3];
		fdwt_cdf97_diagonal_epilog_s(w, v, l, c, r, z, x, y, &addr, stride);
		*addr1_s(end, 2, stride) = l[2];
		fdwt_cdf97_diagonal_epilog_s(w, v, l, c, r, z, x, y, &addr, stride);
		*addr1_s(end, 1, stride) = l[1];
		fdwt_cdf97_diagonal_epilog_s(w, v, l, c, r, z, x, y, &addr, stride);
		*addr1_s(end, 0, stride) = l[0];
	}
}

void fdwt_cdf53_diagonal_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( ptr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	int pairs = (to_even(size)-2)/2;

	float *begin = addr1_s(ptr, 0,       stride);
	float *end   = addr1_s(ptr, 2*pairs, stride);

	if( pairs < 1 )
	{
		// NOTE: unfornunately, the diagonal vectorisation cannot handle less than 1 pair of coefficients
		fdwt_cdf53_vertical_s(ptr, size, stride);
	}

	if( pairs >= 1 )
	{
		const float w[2] = { beta, alpha };
		const float v[2] = { 1/zeta, zeta };

		float l[2];
		float c[2];
		float r[2];
		float z[2];
		float x[2];
		float y[2];

		float *addr = begin;

		// prolog-diagonal
		l[1] = *addr1_const_s(begin, 1, stride);
		fdwt_cdf53_diagonal_prolog_s(w, v, l, c, r, z, x, y, &addr, stride);
		l[0] = *addr1_const_s(begin, 0, stride);

		// core
		for(int s = 0; s < pairs-1; s++)
		{
			fdwt_cdf53_diagonal_core_s(w, v, l, c, r, z, x, y, &addr, stride);
		}

		// epilog-diagonal
		*addr1_s(end, 1, stride) = l[1];
		fdwt_cdf53_diagonal_epilog_s(w, v, l, c, r, z, x, y, &addr, stride);
		*addr1_s(end, 0, stride) = l[0];
	}
}

void fdwt_eaw53_diagonal_s(
	void *ptr,
	int size,
	int stride,
	const float *eaw_w
)
{
	assert( ptr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	int pairs = (to_even(size)-2)/2;

	float *begin = addr1_s(ptr, 0,       stride);
	float *end   = addr1_s(ptr, 2*pairs, stride);

	if( pairs < 1 )
	{
		// NOTE: unfornunately, the diagonal vectorisation cannot handle less than 1 pair of coefficients
		fdwt_eaw53_vertical_s(ptr, size, stride, eaw_w);
	}

	if( pairs >= 1 )
	{
		const float w[2] = { beta, alpha };
		const float v[2] = { 1/zeta, zeta };

		float l[2];
		float c[2];
		float r[2];
		float z[2];
		float x[2];
		float y[2];

		float *addr = begin;

		// prolog-diagonal
		l[1] = *addr1_const_s(begin, 1, stride);
		fdwt_eaw53_diagonal_prolog_s(w, v, l, c, r, z, x, y, &addr, stride, &eaw_w[-2]);
		l[0] = *addr1_const_s(begin, 0, stride);

		// core
		for(int s = 0; s < pairs-1; s++)
		{
			fdwt_eaw53_diagonal_core_s(w, v, l, c, r, z, x, y, &addr, stride, &eaw_w[2*s]);
		}

		// epilog-diagonal
		*addr1_s(end, 1, stride) = l[1];
		fdwt_eaw53_diagonal_epilog_s(w, v, l, c, r, z, x, y, &addr, stride, &eaw_w[2*pairs-2]);
		*addr1_s(end, 0, stride) = l[0];
	}
}

static
void fdwt_cdf97_epilog_s(
	float *arr,
	int N,
	int stride
)
{
	assert( N >= 4 );
	assert( arr );

	float alpha = -dwt_cdf97_p1_s;
	float beta  = +dwt_cdf97_u1_s;
	float gamma = -dwt_cdf97_p2_s;
	float delta = +dwt_cdf97_u2_s;
	float zeta  = +dwt_cdf97_s1_s;

	if( is_even(N) )
	{
		// alpha
		// none

		// beta
		*addr1_s(arr, N-1, stride) += 2*beta*(*addr1_s(arr, N-2, stride));

		// gamma
		*addr1_s(arr, N-2, stride) += gamma*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

		// delta
		*addr1_s(arr, N-1, stride) += 2*delta*(*addr1_s(arr, N-2, stride));
		*addr1_s(arr, N-3, stride) += delta*(*addr1_s(arr, N-4, stride) + *addr1_s(arr, N-2, stride));

		// scaling
		*addr1_s(arr, N-4, stride) *= 1/zeta;
		*addr1_s(arr, N-3, stride) *= zeta;
		*addr1_s(arr, N-2, stride) *= 1/zeta;
		*addr1_s(arr, N-1, stride) *= zeta;
	}
	else /* is_odd(N) */
	{
		// alpha
		*addr1_s(arr, N-1, stride) += 2*alpha*(*addr1_s(arr, N-2, stride));

		// beta
		*addr1_s(arr, N-2, stride) += beta*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

		// gamma
		*addr1_s(arr, N-1, stride) += 2*gamma*(*addr1_s(arr, N-2, stride));
		*addr1_s(arr, N-3, stride) += gamma*(*addr1_s(arr, N-2, stride) + *addr1_s(arr, N-4, stride));

		// delta
		*addr1_s(arr, N-2, stride) += delta*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));
		*addr1_s(arr, N-4, stride) += delta*(*addr1_s(arr, N-5, stride) + *addr1_s(arr, N-3, stride));

		// scaling
		*addr1_s(arr, N-5, stride) *= 1/zeta;
		*addr1_s(arr, N-4, stride) *= zeta;
		*addr1_s(arr, N-3, stride) *= 1/zeta;
		*addr1_s(arr, N-2, stride) *= zeta;
		*addr1_s(arr, N-1, stride) *= 1/zeta;
	}
}

static
void fdwt_cdf53_epilog_s(
	float *arr,
	int N,
	int stride
)
{
	assert( N >= 2 );
	assert( arr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	if( is_even(N) )
	{
		// alpha
		// none

		// beta
		*addr1_s(arr, N-1, stride) += 2*beta*(*addr1_s(arr, N-2, stride));

		// scaling
		*addr1_s(arr, N-2, stride) *= 1/zeta;
		*addr1_s(arr, N-1, stride) *= zeta;
	}
	else /* is_odd(N) */
	{
		// alpha
		*addr1_s(arr, N-1, stride) += 2*alpha*(*addr1_s(arr, N-2, stride));

		// beta
		*addr1_s(arr, N-2, stride) += beta*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

		// scaling
		*addr1_s(arr, N-3, stride) *= 1/zeta;
		*addr1_s(arr, N-2, stride) *= zeta;
		*addr1_s(arr, N-1, stride) *= 1/zeta;
	}
}

static
void fdwt_eaw53_epilog_s(
	float *arr,
	int N,
	int stride,
	const float *eaw_w
)
{
	assert( N >= 2 );
	assert( arr );

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	if( is_even(N) )
	{
		// alpha
		// none

		// beta
		*addr1_s(arr, N-1, stride) += 2*beta*(*addr1_s(arr, N-2, stride));

		// scaling
		*addr1_s(arr, N-2, stride) *= 1/zeta;
		*addr1_s(arr, N-1, stride) *= zeta;
	}
	else /* is_odd(N) */
	{
		// alpha
		*addr1_s(arr, N-1, stride) += 2*alpha*(*addr1_s(arr, N-2, stride));

		// beta
		float wL = eaw_w[(N-2)-1]; 
		float wR = eaw_w[(N-2)+0];
		*addr1_s(arr, N-2, stride) += ( wL * *addr1_s(arr, N-3, stride) + wR * *addr1_s(arr, N-1, stride) ) / (wL+wR) * (2.f * beta);

		// scaling
		*addr1_s(arr, N-3, stride) *= 1/zeta;
		*addr1_s(arr, N-2, stride) *= zeta;
		*addr1_s(arr, N-1, stride) *= 1/zeta;
	}
}

void fdwt2_cdf97_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		if( size_x_j > 1 && size_x_j < 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j < 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_vertical_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_vertical_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		j++;
	}
}

void fdwt2h1_cdf97_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
#endif

		if( size_x_j > 1 && size_x_j < 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_vertical_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}

		j++;
	}
}

void fdwt2v1_cdf97_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		if( size_y_j > 1 && size_y_j < 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_y_j > 1 && size_y_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_vertical_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		j++;
	}
}

void fdwt2_cdf53_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		if( size_x_j > 1 && size_x_j < 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j < 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_vertical_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_vertical_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		j++;
	}
}

void fdwt1_cdf97_horizontal_s(
	void *ptr,
	int size,
	int stride,
	int *j_max_ptr
)
{
	const int offset = 1;

	int j = 0;

	const int j_limit = ceil_log2(size);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size, j);

		const int stride_y_j = stride * (1 << j);

		if( size_x_j > 1 && size_x_j < 5 )
		{
			fdwt_cdf97_short_s(
				ptr,
				size_x_j,
				stride_y_j
			);
		}

		if( size_x_j >= 5 )
		{
			fdwt_cdf97_prolog_s(
				ptr,
				size_x_j,
				stride_y_j
			);

			fdwt_cdf97_horizontal_s(
				addr1_s(ptr, 0+offset, stride_y_j),
				size_x_j-offset,
				stride_y_j
			);

			fdwt_cdf97_epilog_s(
				addr1_s(ptr, 0+offset, stride_y_j),
				size_x_j-offset,
				stride_y_j
			);
		}

		j++;
	}
}

void fdwt1_single_cdf97_horizontal_s(
	void *ptr,
	int size,
	int stride
)
{
	const int offset = 1;

	const int j_limit = ceil_log2(size);

	if( j_limit < 1 )
		return;

	const int size_x_j = size;

	const int stride_y_j = stride;

	if( size_x_j > 1 && size_x_j < 5 )
	{
		fdwt_cdf97_short_s(
			ptr,
			size_x_j,
			stride_y_j
		);
	}

	if( size_x_j >= 5 )
	{
		fdwt_cdf97_prolog_s(
			ptr,
			size_x_j,
			stride_y_j
		);

		fdwt_cdf97_horizontal_s(
			addr1_s(ptr, 0+offset, stride_y_j),
			size_x_j-offset,
			stride_y_j
		);

		fdwt_cdf97_epilog_s(
			addr1_s(ptr, 0+offset, stride_y_j),
			size_x_j-offset,
			stride_y_j
		);
	}
}

void fdwt1_single_cdf97_horizontal_min5_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( size >= 5 );

	const int offset = 1;

	fdwt_cdf97_prolog_s(
		ptr,
		size,
		stride
	);

	fdwt_cdf97_horizontal_s(
		addr1_s(ptr, 0+offset, stride),
		size-offset,
		stride
	);

	fdwt_cdf97_epilog_s(
		addr1_s(ptr, 0+offset, stride),
		size-offset,
		stride
	);
}

void fdwt1_single_cdf97_vertical_min5_s(
	void *ptr,
	int size,
	int stride
)
{
	assert( size >= 5 );

	const int offset = 1;

	fdwt_cdf97_prolog_s(
		ptr,
		size,
		stride
	);

	fdwt_cdf97_vertical_s(
		addr1_s(ptr, 0+offset, stride),
		size-offset,
		stride
	);

	fdwt_cdf97_epilog_s(
		addr1_s(ptr, 0+offset, stride),
		size-offset,
		stride
	);
}

void fdwt2_cdf97_horizontal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		if( size_x_j > 1 && size_x_j < 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j < 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_horizontal_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_horizontal_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		j++;
	}
}

void fdwt2_cdf53_horizontal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		if( size_x_j > 1 && size_x_j < 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j < 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_horizontal_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_horizontal_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		j++;
	}
}

/**
 * @warning The EAW decomposition is different if we compute weights "w" for second (vertical) filtering before xor after first (horizontal) filtering.
 */
void fdwt2_eaw53_horizontal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one,
	float *wH[],
	float *wV[],
	float alpha
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		wH[j] = dwt_util_alloc(size_y_j * size_x_j, sizeof(float));
		wV[j] = dwt_util_alloc(size_x_j * size_y_j, sizeof(float));

		if( size_x_j > 1 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				dwt_calc_eaw_w_stride_s(
					&wH[j][y*size_x_j],
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					alpha
				);
			}
		}
		if( size_y_j > 1 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				dwt_calc_eaw_w_stride_s(
					&wV[j][x*size_y_j],
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					alpha
				);
			}
		}

		if( size_x_j > 1 && size_x_j < 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					&wH[j][y*size_x_j]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					&wH[j][y*size_x_j]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_horizontal_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j,
					&wH[j][y*size_x_j+offset]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j,
					&wH[j][y*size_x_j+offset]
				);
			}
		}

		if( size_y_j > 1 && size_y_j < 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					&wV[j][x*size_y_j]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					&wV[j][x*size_y_j]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_horizontal_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j,
					&wV[j][x*size_y_j+offset]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j,
					&wV[j][x*size_y_j+offset]
				);
			}
		}

		j++;
	}
}

void fdwt2_eaw53_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one,
	float *wH[],
	float *wV[],
	float alpha
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		wH[j] = dwt_util_alloc(size_y_j * size_x_j, sizeof(float));
		wV[j] = dwt_util_alloc(size_x_j * size_y_j, sizeof(float));

		if( size_x_j > 1 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				dwt_calc_eaw_w_stride_s(
					&wH[j][y*size_x_j],
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					alpha
				);
			}
		}
		if( size_y_j > 1 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				dwt_calc_eaw_w_stride_s(
					&wV[j][x*size_y_j],
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					alpha
				);
			}
		}

		if( size_x_j > 1 && size_x_j < 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					&wH[j][y*size_x_j]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					&wH[j][y*size_x_j]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_vertical_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j,
					&wH[j][y*size_x_j+offset]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j,
					&wH[j][y*size_x_j+offset]
				);
			}
		}

		if( size_y_j > 1 && size_y_j < 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					&wV[j][x*size_y_j]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					&wV[j][x*size_y_j]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_vertical_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j,
					&wV[j][x*size_y_j+offset]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j,
					&wV[j][x*size_y_j+offset]
				);
			}
		}

		j++;
	}
}

void fdwt2_eaw53_diagonal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one,
	float *wH[],
	float *wV[],
	float alpha
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		wH[j] = dwt_util_alloc(size_y_j * size_x_j, sizeof(float));
		wV[j] = dwt_util_alloc(size_x_j * size_y_j, sizeof(float));

		if( size_x_j > 1 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				dwt_calc_eaw_w_stride_s(
					&wH[j][y*size_x_j],
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					alpha
				);
			}
		}
		if( size_y_j > 1 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				dwt_calc_eaw_w_stride_s(
					&wV[j][x*size_y_j],
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					alpha
				);
			}
		}

		if( size_x_j > 1 && size_x_j < 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					&wH[j][y*size_x_j]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j,
					&wH[j][y*size_x_j]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_diagonal_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j,
					&wH[j][y*size_x_j+offset]
				);
			}
		}
		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_eaw53_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j,
					&wH[j][y*size_x_j+offset]
				);
			}
		}

		if( size_y_j > 1 && size_y_j < 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					&wV[j][x*size_y_j]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j,
					&wV[j][x*size_y_j]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_diagonal_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j,
					&wV[j][x*size_y_j+offset]
				);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_eaw53_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j,
					&wV[j][x*size_y_j+offset]
				);
			}
		}

		j++;
	}
}

void fdwt2_cdf97_diagonal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		if( size_x_j > 1 && size_x_j < 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j < 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_diagonal_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_diagonal_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 5 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 5 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf97_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		j++;
	}
}

void fdwt2_cdf53_diagonal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int offset = 1;

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_min = min(size_x, size_y);
	const int size_max = max(size_x, size_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_max : size_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_x, j);
		const int size_y_j = ceil_div_pow2(size_y, j);

		const int stride_y_j = stride_y * (1 << j);
		const int stride_x_j = stride_x * (1 << j);

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y_j, threads);
		const int threads_segment_x = ceil_div(size_x_j, threads);
#endif

		if( size_x_j > 1 && size_x_j < 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_short_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j < 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_short_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x_j,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y_j,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_diagonal_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_diagonal_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		if( size_x_j > 1 && size_x_j >= 3 )
		{
			for(int y = 0; y < size_y_j; y++)
			{
				fdwt_cdf53_epilog_s(
					addr2_s(ptr, y, 0+offset, stride_x_j, stride_y_j),
					size_x_j-offset,
					stride_y_j);
			}
		}
		if( size_y_j > 1 && size_y_j >= 3 )
		{
			for(int x = 0; x < size_x_j; x++)
			{
				fdwt_cdf53_epilog_s(
					addr2_s(ptr, 0+offset, x, stride_x_j, stride_y_j),
					size_y_j-offset,
					stride_x_j);
			}
		}

		j++;
	}
}
