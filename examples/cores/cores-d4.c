#include "cores-d4.h"
#include <assert.h>
#include "system.h"
#include <math.h>
#include "inline.h"

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
