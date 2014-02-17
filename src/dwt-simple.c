#include "dwt-simple.h"
#include "libdwt.h"
#include "inline.h"

static
void op4s_sdl_import_stride_s_ref(float *l, const float *addr, int idx, int stride)
{
	l[idx] = *addr1_const_s(addr, idx, stride);
}

static
void op2s_sdl_import_stride_s_ref(float *l, const float *addr, int idx, int stride)
{
	l[idx] = *addr1_const_s(addr, idx, stride);
}

static
void op4s_sdl_export_stride_s_ref(const float *l, float *addr, int idx, int stride)
{
	*addr1_s(addr,idx,stride) = l[idx];
}

static
void op2s_sdl_export_stride_s_ref(const float *l, float *addr, int idx, int stride)
{
	*addr1_s(addr,idx,stride) = l[idx];
}

static
void op4s_sdl_shuffle_s_ref(float *c, float *r)
{
	c[0]=c[1]; c[1]=c[2]; c[2]=c[3];
	r[0]=r[1]; r[1]=r[2]; r[2]=r[3];
}

static
void op2s_sdl_shuffle_s_ref(float *c, float *r)
{
	c[0] = c[1];
	r[0] = r[1];
}

static
void op4s_sdl_load_stride_s_ref(float *in, const float *addr, int stride)
{
	in[0] = *addr1_const_s(addr,0,stride);
	in[1] = *addr1_const_s(addr,1,stride);
}

static
void op2s_sdl_load_stride_s_ref(float *in, const float *addr, int stride)
{
	in[0] = *addr1_const_s(addr,0,stride);
	in[1] = *addr1_const_s(addr,1,stride);
}

static
void op4s_sdl_save_stride_s_ref(float *out, float *addr, int stride)
{
	*addr1_s(addr,0,stride) = out[0];
	*addr1_s(addr,1,stride) = out[1];
}

static
void op2s_sdl_save_stride_s_ref(float *out, float *addr, int stride)
{
	*addr1_s(addr,0,stride) = out[0];
	*addr1_s(addr,1,stride) = out[1];
}

static
void op4s_sdl_input_s_ref(const float *in, float *c, float *r)
{
	c[3] = in[0];
	r[3] = in[1];
}

static
void op2s_sdl_input_s_ref(const float *in, float *c, float *r)
{
	c[1] = in[0];
	r[1] = in[1];
}

static
void op4s_sdl_output_s_ref(float *out, const float *l, const float *z)
{
	out[0] = l[0];
	out[1] = z[0];
}

static
void op2s_sdl_output_s_ref(float *out, const float *l, const float *z)
{
	out[0] = l[0];
	out[1] = z[0];
}

static
void op4s_sdl_scale_s_ref(float *out, const float *v)
{
	out[0] *= v[0];
	out[1] *= v[1];
}

static
void op2s_sdl_scale_s_ref(float *out, const float *v)
{
	out[0] *= v[0];
	out[1] *= v[1];
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
void op4s_sdl_pass_fwd_prolog_stride_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	UNUSED(v);
	UNUSED(out);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_stride_s_ref(in, addr1_s(*addr,+4,stride), stride);

	// (descale)

	// input
	op4s_sdl_input_s_ref(in, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// (output)

	// (scale)

	// (save)

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr,2,stride);
}

static
void op2s_sdl_pass_fwd_prolog_stride_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	UNUSED(v);
	UNUSED(out);

	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// load
	op2s_sdl_load_stride_s_ref(in, addr1_s(*addr,+2,stride), stride);

	// (descale)

	// input
	op2s_sdl_input_s_ref(in, c, r);

	// operation
	op2s_sdl_op_s_ref(z, c, w, l, r);

	// (output)

	// (scale)

	// (save)

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr,2,stride);
}

static
void op4s_sdl_pass_fwd_epilog_stride_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	UNUSED(in);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// (load)

	// (descale)

	// (input)

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op4s_sdl_output_s_ref(out, l, z);

	// scale
	op4s_sdl_scale_s_ref(out, v);

	// save
	op4s_sdl_save_stride_s_ref(out, addr1_s(*addr,-6,stride), stride);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) = addr1_s(*addr,2,stride);
}

static
void op2s_sdl_pass_fwd_epilog_stride_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	UNUSED(in);

	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// (load)

	// (descale)

	// (input)

	// operation
	op2s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op2s_sdl_output_s_ref(out, l, z);

	// scale
	op2s_sdl_scale_s_ref(out, v);

	// save
	op2s_sdl_save_stride_s_ref(out, addr1_s(*addr,-2,stride), stride);

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) = addr1_s(*addr,2,stride);
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
		*addr1_s(arr, 1, stride) += 2*alpha*(*addr1_s(arr, 0, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));

		// gamma
		*addr1_s(arr, 1, stride) += 2*gamma*(*addr1_s(arr, 0, stride));

		// delta
		*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));

		// scaling
		*addr1_s(arr, 0, stride) *= zeta;
		*addr1_s(arr, 1, stride) *= 1/zeta;
	}

	if( 3 == N )
	{
		// alpha
		*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += 2*beta*(*addr1_s(arr, 1, stride));

		// gamma
		*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

		// delta
		*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += 2*delta*(*addr1_s(arr, 1, stride));

		// scaling
		*addr1_s(arr, 0, stride) *= zeta;
		*addr1_s(arr, 1, stride) *= 1/zeta;
		*addr1_s(arr, 2, stride) *= zeta;
	}

	if( 4 == N )
	{
		// alpha
		*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
		*addr1_s(arr, 3, stride) += 2*alpha*(*addr1_s(arr, 2, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += beta*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

		// gamma
		*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
		*addr1_s(arr, 3, stride) += 2*gamma*(*addr1_s(arr, 2, stride));

		// delta
		*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));
		*addr1_s(arr, 2, stride) += delta*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

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
		*addr1_s(arr, 1, stride) += 2*alpha*(*addr1_s(arr, 0, stride));

		// beta
		*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));

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

	float alpha = -dwt_cdf97_p1_s;
	float beta  = +dwt_cdf97_u1_s;
	float gamma = -dwt_cdf97_p2_s;
	float delta = +dwt_cdf97_u2_s;
	float zeta  = +dwt_cdf97_s1_s;

	// alpha
	*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
	*addr1_s(arr, 3, stride) += alpha*(*addr1_s(arr, 2, stride) + *addr1_s(arr, 4, stride));

	// beta
	*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));
	*addr1_s(arr, 2, stride) += beta*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

	// gamma
	*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

	// delta
	*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));

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

	float alpha = -dwt_cdf53_p1_s;
	float beta  = +dwt_cdf53_u1_s;
	float zeta  = +dwt_cdf53_s1_s;

	// alpha
	*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

	// beta
	*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));

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
	float in[2];
	float out[2];
	float r[4];
	float c[4];

	// inputs
	in[0] = *ptr0;
	in[1] = *ptr1;

	// shuffles
	out[0] = l[0];
	c[0]   = l[1];
	c[1]   = l[2];
	c[2]   = l[3];
	c[3]   = in[0];

	// operation
	r[3]   = in[1];
	r[2]   = c[3]+w[3]*(l[3]+r[3]);
	r[1]   = c[2]+w[2]*(l[2]+r[2]);
	r[0]   = c[1]+w[1]*(l[1]+r[1]);
	out[1] = c[0]+w[0]*(l[0]+r[0]);

	// scales
	out[0] = out[0] * v[0];
	out[1] = out[1] * v[1];

	// outputs
	*out0 = out[0];
	*out1 = out[1];

	// update l[]
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
	float in[2];
	float out[2];
	float r[2];
	float c[2];

	// inputs
	in[0] = *ptr0;
	in[1] = *ptr1;

	// shuffles
	out[0] = l[0];
	c[0]   = l[1];
	c[1]   = in[0];

	// operation
	r[1]   = in[1];
	r[0]   = c[1]+w[1]*(l[1]+r[1]);
	out[1] = c[0]+w[0]*(l[0]+r[0]);

	// scales
	out[0] = out[0] * v[0];
	out[1] = out[1] * v[1];

	// outputs
	*out0 = out[0];
	*out1 = out[1];

	// update l[]
	l[0] = r[0];
	l[1] = r[1];
}

void fdwt_cdf97_vertical_s(
	void *ptr,
	int size,
	int stride
)
{
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

void fdwt_cdf97_horizontal_s(
	void *ptr,
	int size,
	int stride
)
{
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
void fdwt_cdf97_diagonal_prolog_s(
	float *base,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *in,
	float *out,
	float **addr,
	int stride
)
{
	// prolog2: import(3)
	op4s_sdl_import_stride_s_ref(l, base, 3, stride);

	// prolog2: pass-prolog
	op4s_sdl_pass_fwd_prolog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// prolog2: import(2)
	op4s_sdl_import_stride_s_ref(l, base, 2, stride);

	// prolog2: pass-prolog
	op4s_sdl_pass_fwd_prolog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// prolog2: import(1)
	op4s_sdl_import_stride_s_ref(l, base, 1, stride);

	// prolog2: pass-prolog
	op4s_sdl_pass_fwd_prolog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// prolog2: import(0)
	op4s_sdl_import_stride_s_ref(l, base, 0, stride);
}

static
void fdwt_cdf53_diagonal_prolog_s(
	float *base,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *in,
	float *out,
	float **addr,
	int stride
)
{
	// prolog2: import(1)
	op2s_sdl_import_stride_s_ref(l, base, 1, stride);

	// prolog2: pass-prolog
	op2s_sdl_pass_fwd_prolog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// prolog2: import(0)
	op2s_sdl_import_stride_s_ref(l, base, 0, stride);
}

static
void fdwt_cdf97_diagonal_epilog_s(
	float *base,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *in,
	float *out,
	float **addr,
	int stride
)
{
	// epilog2: export(3)
	op4s_sdl_export_stride_s_ref(l, base, 3, stride);

	// epilog2: pass-epilog
	op4s_sdl_pass_fwd_epilog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// epilog2: export(2)
	op4s_sdl_export_stride_s_ref(l, base, 2, stride);

	// epilog2: pass-epilog
	op4s_sdl_pass_fwd_epilog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// epilog2: export(1)
	op4s_sdl_export_stride_s_ref(l, base, 1, stride);

	// epilog2: pass-epilog
	op4s_sdl_pass_fwd_epilog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// epilog2: export(0)
	op4s_sdl_export_stride_s_ref(l, base, 0, stride);
}

static
void fdwt_cdf53_diagonal_epilog_s(
	float *base,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *in,
	float *out,
	float **addr,
	int stride
)
{
	// epilog2: export(1)
	op2s_sdl_export_stride_s_ref(l, base, 1, stride);

	// epilog2: pass-epilog
	op2s_sdl_pass_fwd_epilog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// epilog2: export(0)
	op2s_sdl_export_stride_s_ref(l, base, 0, stride);
}

static
void fdwt_cdf97_diagonal_core_s(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_stride_s_ref(in, addr1_s(*addr,4,stride), stride);

	// (descale)

	// input
	op4s_sdl_input_s_ref(in, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op4s_sdl_output_s_ref(out, l, z);

	// scale
	op4s_sdl_scale_s_ref(out, v);

	// save
	op4s_sdl_save_stride_s_ref(out, addr1_s(*addr,-6,stride), stride);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr,2,stride);
}

static
void fdwt_cdf53_diagonal_core_s(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	// shuffle
	op2s_sdl_shuffle_s_ref(c, r);

	// load
	op2s_sdl_load_stride_s_ref(in, addr1_s(*addr,2,stride), stride);

	// (descale)

	// input
	op2s_sdl_input_s_ref(in, c, r);

	// operation
	op2s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op2s_sdl_output_s_ref(out, l, z);

	// scale
	op2s_sdl_scale_s_ref(out, v);

	// save
	op2s_sdl_save_stride_s_ref(out, addr1_s(*addr,-2,stride), stride);

	// update
	op2s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr,2,stride);
}

void fdwt_cdf97_diagonal_s(
	void *ptr,
	int size,
	int stride
)
{
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
		float in[4];
		float out[4];

		float *addr = begin;

		// prolog-diagonal
		fdwt_cdf97_diagonal_prolog_s(begin, w, v, l, c, r, z, in, out, &addr, stride);

		// core
		for(int s = 0; s < pairs-3; s++)
		{
			fdwt_cdf97_diagonal_core_s(w, v, l, c, r, z, in, out, &addr, stride);
		}

		// epilog-diagonal
		fdwt_cdf97_diagonal_epilog_s(end, w, v, l, c, r, z, in, out, &addr, stride);
	}
}

void fdwt_cdf53_diagonal_s(
	void *ptr,
	int size,
	int stride
)
{
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
		float in[2];
		float out[2];

		float *addr = begin;

		// prolog-diagonal
		fdwt_cdf53_diagonal_prolog_s(begin, w, v, l, c, r, z, in, out, &addr, stride);

		// core
		for(int s = 0; s < pairs-1; s++)
		{
			fdwt_cdf53_diagonal_core_s(w, v, l, c, r, z, in, out, &addr, stride);
		}

		// epilog-diagonal
		fdwt_cdf53_diagonal_epilog_s(end, w, v, l, c, r, z, in, out, &addr, stride);
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
