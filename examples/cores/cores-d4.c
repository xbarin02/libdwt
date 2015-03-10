#include "cores-d4.h"
#include <assert.h>
#include "system.h"
#include <math.h>
#include "inline.h"
#include "coords.h"

static
float *image_pix_s(image_t *image, int x, int y)
{
	return addr2_s(image->ptr, y, x, image->stride_y, image->stride_x);
}

// naive / horizontal / multi-loop
// 1-D forward interleaved-subbands
// Daubechies1998, D4, 3rd implementation
// cite: One can also obtain an entirely different lifting factorization of D4 by shifting the filter pair corresponding to...
static
void fdwt1_d4_horiz(
	float *arr,
	int size,
	int stride
)
{
	assert( arr );

	const float alpha = -1./sqrtf(3.f);
	const float beta  = (6.f-3.f*sqrtf(3.f))/4;
	const float gamma = sqrtf(3.f)/4;
	const float delta = -1.f/3.f;

	const float zeta_e = (3.f+sqrtf(3.f))/(3.f*sqrtf(2.f)); // even = s = L
	const float zeta_o = (3.f-sqrtf(3.f))/(3.f*sqrtf(2.f)); // odd  = d = H

	// P1: odd += f(even)
	for(int x = 1; x < size-1; x += 2)
		*addr1_s(arr, x, stride) +=
			+alpha * *addr1_s(arr, x+1, stride);

	// U1: even += f(odd)
	for(int x = 2; x < size-1; x += 2)
		*addr1_s(arr, x, stride) +=
			+beta  * *addr1_s(arr, x-1, stride)
			+gamma * *addr1_s(arr, x+1, stride);

	// P2: odd += f(even)
	for(int x = 1; x < size-0; x += 2)
		*addr1_s(arr, x, stride) +=
			+delta * *addr1_s(arr, x-1, stride);

	// S
	for(int x = 0; x < size-1; x += 2)
	{
		*addr1_s(arr, x+0, stride) *= zeta_e;
		*addr1_s(arr, x+1, stride) *= zeta_o;
	}
}

static
void idwt1_d4_horiz(
	float *arr,
	int size,
	int stride
)
{
	assert( arr );

	const float alpha = -1./sqrtf(3.f);
	const float beta  = (6.f-3.f*sqrtf(3.f))/4;
	const float gamma = sqrtf(3.f)/4;
	const float delta = -1.f/3.f;

	const float zeta_e = (3.f*sqrtf(2.f))/(3.f+sqrtf(3.f)); // even = s = L
	const float zeta_o = (3.f*sqrtf(2.f))/(3.f-sqrtf(3.f)); // odd  = d = H

	// S
	for(int x = 0; x < size-1; x += 2)
	{
		*addr1_s(arr, x+0, stride) *= zeta_e;
		*addr1_s(arr, x+1, stride) *= zeta_o;
	}

	// P2: odd -= f(even)
	for(int x = 1; x < size-0; x += 2)
		*addr1_s(arr, x, stride) -=
			+delta * *addr1_s(arr, x-1, stride);

	// U1: even -= f(odd)
	for(int x = 2; x < size-1; x += 2)
		*addr1_s(arr, x, stride) -=
			+beta  * *addr1_s(arr, x-1, stride)
			+gamma * *addr1_s(arr, x+1, stride);

	// P1: odd -= f(even)
	for(int x = 1; x < size-1; x += 2)
		*addr1_s(arr, x, stride) -=
			+alpha * *addr1_s(arr, x+1, stride);
}

// naive / separable horizontal / multi-loop
// 2-D forward interleaved-subbands
void fdwt2_d4_sep_horiz(
	struct image_t *source,
	struct image_t *target
)
{
	// assert
	assert( source && target && source->size_x == target->size_x && source->size_y == target->size_y );

	// for each row
	for(int y = 0; y < source->size_y; y++)
	{
		float *target_row = image_pix_s(target, 0, y);
		float *source_row = image_pix_s(source, 0, y);

		// copy row from source into target
		dwt_util_memcpy_stride_s(
			target_row,
			target->stride_x,
			source_row,
			source->stride_x,
			source->size_x
		);

		fdwt1_d4_horiz(target_row, source->size_x, target->stride_x);
	}

	// for each column
	for(int x = 0; x < source->size_y; x++)
	{
		float *target_col = image_pix_s(target, x, 0);

		fdwt1_d4_horiz(target_col, source->size_y, target->stride_y);
	}
}

void idwt2_d4_sep_horiz(
	struct image_t *source,
	struct image_t *target
)
{
	// assert
	assert( source && target && source->size_x == target->size_x && source->size_y == target->size_y );

	// for each row
	for(int y = 0; y < source->size_y; y++)
	{
		float *target_row = image_pix_s(target, 0, y);
		float *source_row = image_pix_s(source, 0, y);

		// copy row from source into target
		dwt_util_memcpy_stride_s(
			target_row,
			target->stride_x,
			source_row,
			source->stride_x,
			source->size_x
		);

		idwt1_d4_horiz(target_row, source->size_x, target->stride_x);
	}

	// for each column
	for(int x = 0; x < source->size_y; x++)
	{
		float *target_col = image_pix_s(target, x, 0);

		idwt1_d4_horiz(target_col, source->size_y, target->stride_y);
	}
}

static
void core1_d4_2x1(
	float *data0, // [1]
	float *data1, // [1]
	float *buff // [2]
)
{
	const float alpha = -1./sqrtf(3.f);
	const float beta  = (6.f-3.f*sqrtf(3.f))/4;
	const float gamma = sqrtf(3.f)/4;
	const float delta = -1.f/3.f;

	const float zeta_e = (3.f+sqrtf(3.f))/(3.f*sqrtf(2.f)); // even = s = L
	const float zeta_o = (3.f-sqrtf(3.f))/(3.f*sqrtf(2.f)); // odd  = d = H

	// input
	float x0 = *data0;
	float x1 = *data1;

	x0 += alpha * x1;

	// load
	float l0 = buff[0];
	float l1 = buff[1];

	// store
	buff[0] = x0;
	buff[1] = x1;

	x1 = x0;
	x0 = beta * l0 + l1 + gamma * x0;

	x1 += delta * x0;

	x0 *= zeta_e;
	x1 *= zeta_o;

	// output
	*data0 = x0;
	*data1 = x1;
}

static
void core1_d4_2x1_alpha(
	float *data0, // [1]
	float *data1, // [1]
	float *buff // [2]
)
{
	UNUSED(buff);

	const float alpha = -1./sqrtf(3.f);

	// input
	float x0 = *data0;
	float x1 = *data1;

	x0 += alpha * x1;

	// output
	*data0 = x0;
	*data1 = x1;
}

static
void core1_d4_2x1_beta_gamma(
	float *data0, // [1]
	float *data1, // [1]
	float *buff // [2]
)
{
	const float beta  = (6.f-3.f*sqrtf(3.f))/4;
	const float gamma = sqrtf(3.f)/4;

	// input
	float x0 = *data0;
	float x1 = *data1;

	// load
	float l0 = buff[0];
	float l1 = buff[1];

	// store
	buff[0] = x0;
	buff[1] = x1;

	x1 = x0;
	x0 = beta * l0 + l1 + gamma * x0;

	// output
	*data0 = x0;
	*data1 = x1;
}

static
void core1_d4_2x1_delta(
	float *data0, // [1]
	float *data1, // [1]
	float *buff // [2]
)
{
	UNUSED(buff);

	const float delta = -1.f/3.f;

	// input
	float x0 = *data0;
	float x1 = *data1;

	x1 += delta * x0;

	// output
	*data0 = x0;
	*data1 = x1;
}

static
void core1_d4_2x1_zeta(
	float *data0, // [1]
	float *data1, // [1]
	float *buff // [2]
)
{
	UNUSED(buff);

	const float zeta_e = (3.f+sqrtf(3.f))/(3.f*sqrtf(2.f)); // even = s = L
	const float zeta_o = (3.f-sqrtf(3.f))/(3.f*sqrtf(2.f)); // odd  = d = H

	// input
	float x0 = *data0;
	float x1 = *data1;

	x0 *= zeta_e;
	x1 *= zeta_o;

	// output
	*data0 = x0;
	*data1 = x1;
}

static
void fdwt1_d4_vert(
	float *src,
	int src_stride,
	float *dst,
	int dst_stride,
	int size
)
{
	float buff[2] = {0};

	for(int x = 1; x < size-1; x += 2)
	{
		float t[2];

		// load
		t[0] = *addr1_s(src, x+0, src_stride);
		t[1] = *addr1_s(src, x+1, src_stride);

		// calc
		core1_d4_2x1(t+0, t+1, buff);

		// store
		*addr1_s(dst, x-1, dst_stride) = t[0];
		*addr1_s(dst, x+0, dst_stride) = t[1];
	}
}

// symmetric extension
// UNUSED_FUNC
static
float *addr2_sym_s(
	void *ptr,
	int y,
	int x,
	int stride_y,
	int stride_x,
	int overlap_y,
	int overlap_x,
	int size_y,
	int size_x
)
{
	const int pos_x = virt2real(x, 0, overlap_x, size_x);
	const int pos_y = virt2real(y, 0, overlap_y, size_y);

	return (float *)( (char *)ptr + pos_x*stride_x + pos_y*stride_y );
}

// UNUSED_FUNC
static
float *addr2_err_s(
	void *ptr,
	int y,
	int x,
	int stride_y,
	int stride_x,
	int overlap_y,
	int overlap_x,
	int size_y,
	int size_x
)
{
	const int pos_x = virt2real_error(x, 0, overlap_x, size_x);
	const int pos_y = virt2real_error(y, 0, overlap_y, size_y);

	if( pos_x < 0 || pos_y < 0 )
		return NULL;

	return (float *)( (char *)ptr + pos_x*stride_x + pos_y*stride_y );
}

static
void fcore2_d4_v2x2(
	int x,
	int y,
	float *src_ptr,
	int src_stride_x,
	int src_stride_y,
	float *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y,
	float *buff_x, // [3]
	float *buff_y  // [3]
)
{
	const int buff_elem_size = 3; // FIXME

	const int overlap_x_L = 3;
	const int overlap_y_L = 3;

	const int shift = 1;

	float t[4];

	// load
	for(int yy = 0; yy < 2; yy++)
	{
		for(int xx = 0; xx < 2; xx++)
		{
			t[2*yy+xx] = *addr2_sym_s(src_ptr, y+yy, x+xx, src_stride_y, src_stride_x, overlap_y_L, overlap_x_L, size_y, size_x);
		}
	}

	// calc
#if 0
	core1_d4_2x1(t+0, t+1, buff_y+0);
	core1_d4_2x1(t+2, t+3, buff_y+2);
	core1_d4_2x1(t+0, t+2, buff_x+0);
	core1_d4_2x1(t+1, t+3, buff_x+2);
#endif
#if 0
	core1_d4_2x1_alpha(t+0, t+1, buff_y+0);
	core1_d4_2x1_alpha(t+2, t+3, buff_y+2);
	core1_d4_2x1_alpha(t+0, t+2, buff_x+0);
	core1_d4_2x1_alpha(t+1, t+3, buff_x+2);

	core1_d4_2x1_beta_gamma(t+0, t+1, buff_y+0);
	core1_d4_2x1_beta_gamma(t+2, t+3, buff_y+2);
	core1_d4_2x1_beta_gamma(t+0, t+2, buff_x+0);
	core1_d4_2x1_beta_gamma(t+1, t+3, buff_x+2);

	core1_d4_2x1_delta(t+0, t+1, buff_y+0);
	core1_d4_2x1_delta(t+2, t+3, buff_y+2);
	core1_d4_2x1_delta(t+0, t+2, buff_x+0);
	core1_d4_2x1_delta(t+1, t+3, buff_x+2);

	core1_d4_2x1_zeta(t+0, t+1, buff_y+0);
	core1_d4_2x1_zeta(t+2, t+3, buff_y+2);
	core1_d4_2x1_zeta(t+0, t+2, buff_x+0);
	core1_d4_2x1_zeta(t+1, t+3, buff_x+2);
#endif
#if 1
	const float alpha = -1./sqrtf(3.f);
	const float beta  = (6.f-3.f*sqrtf(3.f))/4;
	const float gamma = sqrtf(3.f)/4;
	const float delta = -1.f/3.f;

	const float zeta_e = (3.f+sqrtf(3.f))/(3.f*sqrtf(2.f)); // even = s = L
	const float zeta_o = (3.f-sqrtf(3.f))/(3.f*sqrtf(2.f)); // odd  = d = H

	{
		// t[0] .. D * z^0
		// t[1] .. V * z^0
		// t[2] .. H * z^0
		// t[3] .. A * z^0

		// predict D: D += \alpha*V + \alpha*H + \alpha^2*A
		t[0] += alpha*t[1] + alpha*t[2] + alpha*alpha*t[3];

		// load (A,H,V,D) * z_x^{-1}
		float y0l0 = (buff_y+0*buff_elem_size)[0]; // y0l0 <= z_x^{-1}*D
		float y0l1 = (buff_y+0*buff_elem_size)[1]; // y0l1 <= z_x^{-1}*V
		//float y2l0 = (buff_y+1*buff_elem_size)[0]; // y2l0 <= z_x^{-1}*H
		float y2l1 = (buff_y+1*buff_elem_size)[1]; // y2l1 <= z_x^{-1}*A
		float y0l2 = (buff_y+0*buff_elem_size)[2]; // y0l2 <= z_x^{-1}*P(z_y^{-1}*H)
		float y1l2 = (buff_y+1*buff_elem_size)[2]; // y1l2 <= z_x^{-1}*z_y^{-1}*D

		// load (A,H,V,D) * z_y^{-1}
		float x0l0 = (buff_x+0*buff_elem_size)[0]; // x0l0 <= z_y^{-1}*P(z_x^{-1}*V)
		float x0l1 = (buff_x+0*buff_elem_size)[1]; // x0l1 <= z_y^{-1}*z_x^{-1}*A
		float x2l0 = (buff_x+1*buff_elem_size)[0]; // x2l0 <= z_y^{-1}*D
		float x2l1 = (buff_x+1*buff_elem_size)[1]; // x2l1 <= z_y^{-1}*H
		float x0l2 = (buff_x+0*buff_elem_size)[2]; // x0l2 <= z_y^{-1}*A
		//float x1l2 = (buff_x+1*buff_elem_size)[2]; // x1l2 <= z_y^{-1}*z_x^{-1}*H

		// save (A,H,V,D) * z_x^{0}
		(buff_y+0*buff_elem_size)[0] = t[0]; // buff_y[0][0] <= D
		(buff_y+0*buff_elem_size)[1] = t[1]; // buff_y[0][1] <= V
		//(buff_y+1*buff_elem_size)[0] = t[2]; // buff_y[1][0] <= H
		(buff_y+1*buff_elem_size)[1] = t[3]; // buff_y[1][1] <= A

		// save (A,H,V,D) * z_y^{0}
		(buff_x+1*buff_elem_size)[0] = t[0]; // buff_x[1][0] <= D
		(buff_x+1*buff_elem_size)[1] = t[2]; // buff_x[1][1] <= H
		(buff_x+0*buff_elem_size)[2] = t[3]; // buff_x[0][2] <= A
		//(buff_x+1*buff_elem_size)[2] = y2l0; // buff_x[1][2] <= z_x^{-1}*H

		// predict V: V*z_x^{-1} += \alpha*A*z_x^{-1} + \beta*D*z_x^{-1} + \gamma*D
		t[2] = y0l1;          // t[2] <= z_x^{-1}*V
		t[2] += alpha * y2l1; // z_x^{-1}*V += \alpha * z_x^{-1}*A
		t[2] += beta  * y0l0; // z_x^{-1}*V += \beta  * z_x^{-1}*D
		t[2] += gamma * t[0]; // z_x^{-1}*V += \gamma * D
		// t[2] .. P(z_x^{-1}*V)
		(buff_x+0*buff_elem_size)[0] = t[2]; // buff_x[0][0] <= P(z_x^{-1}*V)

		// D
		t[3] = t[0];

		// predict H: H*z_y^{-1} += \alpha*A*z_y^{-1} + \beta*D*z_y^{-1} + \gamma*D
		t[1] = x2l1;          // t[1] <= z_y^{-1}*H
		t[1] += alpha * x0l2; // z_y^{-1}*H += \alpha * z_y^{-1}*A
		t[1] += beta  * x2l0; // z_y^{-1}*H += \beta  * z_y^{-1}*D
		t[1] += gamma * t[0]; // z_y^{-1}*H += \gamma * D
		// t[1] .. P(z_y^{-1}*H)
		(buff_y+0*buff_elem_size)[2] = t[1]; // buff_y[0][2] <= P(z_y^{-1}*H)

		(buff_y+1*buff_elem_size)[2] = x2l0; // buff_y[1][2] <= z_y^{-1}*D

		// A
		(buff_x+0*buff_elem_size)[1] = y2l1; // buff_x[0][1] <= z_x^{-1}*A

		// legend
		// ------
		// x0l0 : z_y^{-1}*P(z_x^{-1}*V) (updated)
		// x0l0 : z_y^{-1}*z_x^{-1}*V    (non-updated)
		// y0l2 : z_x^{-1}*P(z_y^{-1}*H)
		// x1l2 : z_y^{-1}*z_x^{-1}*H
		// y0l0 : z_x^{-1}*D
		// x2l0 : z_y^{-1}*D

		// update A
		t[0] = x0l1;                     // t[0] <= z_y^{-1}*z_x^{-1}*A
		t[0] += beta * x0l0;             // += \beta *         z_y^{-1}*P(z_x^{-1}*V)
		t[0] += beta * y0l2;             // += \beta *         z_x^{-1}*P(z_y^{-1}*H)
		t[0] += -beta * beta * y1l2;     // += -\beta^2 *      z_x^{-1}*z_y^{-1}*D
		t[0] += gamma * t[2];            // += \gamma * P(z_x^{-1}*V)
		t[0] += gamma * t[1];            // += \gamma * P(z_y^{-1}*H)
		t[0] += -gamma * beta * y0l0;    // += -\gamma*\beta * z_x^{-1}*D
		t[0] += -gamma * beta * x2l0;    // += -\gamma*\beta * z_y^{-1}*D
		t[0] += -gamma*gamma * t[3];     // += -\gamma^2 * D
		// TeX:
		// a = a + U(z)
		// U(z) = +\beta z_y^{-1}*z_x^{-1}*V(z)
		// ...

		// t[0] .. A*z_x^{-1}*z_y^{-1}
		// t[1] .. H*z_y^{-1}
		// t[2] .. V*z_x^{-1}
		// t[3] .. D
	}

	{
		// t[0] .. A*z_x^{-1}*z_y^{-1}
		// t[1] .. H*z_y^{-1}
		// t[2] .. V*z_x^{-1}
		// t[3] .. D

		t[3] += delta*t[2] + delta*t[1] + delta*delta*t[0]; // D += \delta*V*z_x^{-1} + \delta*H*z_y^{-1} + \delta^2*A*z_x^{-1}*z_y^{-1}
		t[1] += delta*t[0]; // H*z_y^{-1} += \delta*A*z_x^{-1}*z_y^{-1}
		t[2] += delta*t[0]; // V*z_x^{-1} += \delta*A*z_x^{-1}*z_y^{-1}
	}

	{
		t[0] *= zeta_e*zeta_e; // A: e = s, o = d
		t[1] *= zeta_o*zeta_e; // H
		t[2] *= zeta_e*zeta_o; // V
		t[3] *= zeta_o*zeta_o; // D
	}
#endif

	// store
	for(int yy = 0; yy < 2; yy++)
	{
		for(int xx = 0; xx < 2; xx++)
		{
			float *target = addr2_err_s(dst_ptr, y+yy-shift, x+xx-shift, dst_stride_y, dst_stride_x, overlap_y_L, overlap_x_L, size_y, size_x);

			if( target )
				*target = t[2*yy+xx];
		}
	}
}

void fdwt2_d4_v2x2(
	struct image_t *source,
	struct image_t *target
)
{
	const int overlap_x_L = 3;
	const int overlap_y_L = 3;

	const int super_x = overlap_x_L + source->size_x + 2;
	const int super_y = overlap_y_L + source->size_y + 2;

	const int buff_elem_size = 3; // FIXME

	const int step_x = 2;
	const int step_y = 2;

	float *buff_x = dwt_util_reliably_alloc2(super_x*buff_elem_size, sizeof(float));
	float *buff_y = dwt_util_reliably_alloc2(super_y*buff_elem_size, sizeof(float));

	assert( buff_x && buff_y );

	for(int y = 0; y+step_y-1 < super_y; y += step_y)
		for(int x = 0; x+step_x-1 < super_x; x += step_x)
			fcore2_d4_v2x2(
				x,
				y,
				source->ptr,
				source->stride_x,
				source->stride_y,
				target->ptr,
				target->stride_x,
				target->stride_y,
				source->size_x,
				source->size_y,
				buff_x + x*buff_elem_size, // [2]
				buff_y + y*buff_elem_size  // [2]
			);

	dwt_util_free(buff_x);
	dwt_util_free(buff_y);
}

void fdwt2_d4_sep_vert(
	struct image_t *source,
	struct image_t *target
)
{
	// assert
	assert( source && target && source->size_x == target->size_x && source->size_y == target->size_y );

	// for each row
	for(int y = 0; y < source->size_y; y++)
	{
		float *target_row = image_pix_s(target, 0, y);
		float *source_row = image_pix_s(source, 0, y);

		fdwt1_d4_vert(source_row, source->stride_x, target_row, target->stride_x, source->size_x);
	}

	// for each column
	for(int x = 0; x < source->size_y; x++)
	{
		float *target_col = image_pix_s(target, x, 0);

		fdwt1_d4_vert(target_col, target->stride_y, target_col, target->stride_y, source->size_y);
	}
}
