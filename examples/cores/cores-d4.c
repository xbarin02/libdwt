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
