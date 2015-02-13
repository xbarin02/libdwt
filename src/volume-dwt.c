#include "volume-dwt.h"
#include <assert.h> // assert
#include "dwt-simple.h"
#include "libdwt.h"
#include "system.h" // dwt_util_memcpy_stride_s, dwt_util_alloc_aligned_ex_reliably
#include "dwt-sym.h" // cdf97_2f_dl_4x4_s
#include "inline.h" // dwt_cdf97_*_s coefficients, is_pow2
#include <stdint.h> // intptr_t
#include <malloc.h> // memalign
#include <math.h> // INFINITY
#include <stdio.h> // fopen, fclose
#ifdef __SSE__
	#include <xmmintrin.h>
#endif
#include "inline-sdl.h"
#include <limits.h> // ULONG_MAX

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

#ifdef __SSE__
static
void vert2x1(
	float *x0,
	float *x1,
	float *y0,
	float *y1,
	float * ALIGNED(16) buff4
)
{
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	__m128 x, y, r, c;

	float * ALIGNED(16) l = buff4;

	// inputs
	x[0] = *x0;
	x[1] = *x1;

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
	*y0 = y[0];
	*y1 = y[1];

	// update l[]
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];
}
#endif

/**
 * vertical 2^3 core
 * inspired by: unified_4x4
 */
static
void cube_2x2x2(
	// coordinates
	int x,
	int y,
	int z,
	// image size
	int size_x,
	int size_y,
	int size_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *src_ptr,
	// strides (to address another pixels)
	int src_stride_x,
	int src_stride_y,
	int src_stride_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *dst_ptr,
	// strides (to address another pixels)
	int dst_stride_x,
	int dst_stride_y,
	int dst_stride_z,
	// pointers to three buffers (aligned to 16 bytes = 4 floats)
	float * ALIGNED(16) buffer_x,
	float * ALIGNED(16) buffer_y,
	float * ALIGNED(16) buffer_z,
	// super sizes in order to access elements in buffers
	int super_x,
	int super_y,
	int super_z
)
{
	UNUSED(super_z);

	const int overlap_x_L = 5;
	const int overlap_y_L = 5;
	const int overlap_z_L = 5;

	const int shift = 4;

	const int step_x = 2;
	const int step_y = 2;
	const int step_z = 2;

#ifdef __SSE__
	// 2^3 pixels = 8x float = 2x __m128 (z=0..1)
	__m128 t[2];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
				const int pos_y = virt2real(y, yy, overlap_y_L, size_y);
				const int pos_z = virt2real(z, zz, overlap_z_L, size_z);

				// [0] : yy=0 xx=0
				// [1] : yy=0 xx=1
				// [2] : yy=1 xx=0
				// [3] : yy=1 xx=1
				t[zz][yy*step_x+xx] = *( (float *)( (char *)src_ptr + pos_x*src_stride_x + pos_y*src_stride_y + pos_z*src_stride_z ) );
			}
		}
	}

	// CALC
#if 1
	// NOTE: along x-axis (4 times vert2x1)
	vert2x1(
		&t[0][0],
		&t[0][1],
		&t[0][0],
		&t[0][1],
		buffer_x + (y+0)*4 + (z+0)*(super_y*4)
	);
	vert2x1(
		&t[0][2],
		&t[0][3],
		&t[0][2],
		&t[0][3],
		buffer_x + (y+1)*4 + (z+0)*(super_y*4)
	);
	vert2x1(
		&t[1][0],
		&t[1][1],
		&t[1][0],
		&t[1][1],
		buffer_x + (y+0)*4 + (z+1)*(super_y*4)
	);
	vert2x1(
		&t[1][2],
		&t[1][3],
		&t[1][2],
		&t[1][3],
		buffer_x + (y+1)*4 + (z+1)*(super_y*4)
	);
#endif
#if 1
	// NOTE: along y-axis (4 times vert2x1)
	vert2x1(
		&t[0][0],
		&t[0][2],
		&t[0][0],
		&t[0][2],
		buffer_y + (x+0)*4 + (z+0)*(super_x*4)
	);
	vert2x1(
		&t[0][1],
		&t[0][3],
		&t[0][1],
		&t[0][3],
		buffer_y + (x+1)*4 + (z+0)*(super_x*4)
	);
	vert2x1(
		&t[1][0],
		&t[1][2],
		&t[1][0],
		&t[1][2],
		buffer_y + (x+0)*4 + (z+1)*(super_x*4)
	);
	vert2x1(
		&t[1][1],
		&t[1][3],
		&t[1][1],
		&t[1][3],
		buffer_y + (x+1)*4 + (z+1)*(super_x*4)
	);
#endif
#if 1
	// NOTE: along z-axis (4 times vert2x1)
	vert2x1(
		&t[0][0],
		&t[1][0],
		&t[0][0],
		&t[1][0],
		buffer_z + (x+0)*4 + (y+0)*(super_x*4)
	);
	vert2x1(
		&t[0][1],
		&t[1][1],
		&t[0][1],
		&t[1][1],
		buffer_z + (x+1)*4 + (y+0)*(super_x*4)
	);
	vert2x1(
		&t[0][2],
		&t[1][2],
		&t[0][2],
		&t[1][2],
		buffer_z + (x+0)*4 + (y+1)*(super_x*4)
	);
	vert2x1(
		&t[0][3],
		&t[1][3],
		&t[0][3],
		&t[1][3],
		buffer_z + (x+1)*4 + (y+1)*(super_x*4)
	);
#endif
#if 1
	// NOTE: scaling
	const float z1 = dwt_cdf97_s1_s;
	const float z3 = dwt_cdf97_s1_s*dwt_cdf97_s1_s*dwt_cdf97_s1_s;

	t[0] *= (__m128){ 1.f/z3, 1.f/z1, 1.f/z1, z1 };
	t[1] *= (__m128){ 1.f/z1, z1, z1, z3 };
#endif

	// STORE
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
				const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
				const int pos_z = virt2real_error(z-shift, zz, overlap_z_L, size_z);

				if( pos_x < 0 || pos_y < 0 || pos_z < 0 )
					continue;

				*( (float *)( (char *)dst_ptr + pos_x*dst_stride_x + pos_y*dst_stride_y + pos_z*dst_stride_z ) ) = t[zz][yy*step_x+xx];
			}
		}
	}
#endif
}

#define ALLOC_ON_HEAP

void cdf97_3f_op_baseline_vert2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// width of buffers: 4 floats
	const int buff_elem_size = 4;

	// step_x, step_y, step_z
	const int step = 2;

	// super sizes
	const int super_x = /*overlap_x_L*/5 + volume_src->size_x + /*overlap_x_R*/5;
	const int super_y = /*overlap_y_L*/5 + volume_src->size_y + /*overlap_y_R*/5;
	const int super_z = /*overlap_z_L*/5 + volume_src->size_z + /*overlap_z_R*/5;

	assert( 0 == super_x % step );
	assert( 0 == super_y % step );
	assert( 0 == super_z % step );

	// alloc buffers
	const int buffer_x_elems = super_y*super_z * buff_elem_size;
	const int buffer_y_elems = super_x*super_z * buff_elem_size;
	const int buffer_z_elems = super_x*super_y * buff_elem_size;

#ifndef ALLOC_ON_HEAP
	float buffer_x[buffer_x_elems] ALIGNED(16); // NOTE: huge array on stack
	float buffer_y[buffer_y_elems] ALIGNED(16); // NOTE: huge array on stack
	float buffer_z[buffer_z_elems] ALIGNED(16); // NOTE: huge array on stack
#else
	float * ALIGNED(16) buffer_x = dwt_util_alloc_aligned_ex_reliably(buffer_x_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_y = dwt_util_alloc_aligned_ex_reliably(buffer_y_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_z = dwt_util_alloc_aligned_ex_reliably(buffer_z_elems, sizeof(float), 16);
#endif

#if 0
	// zero buffers
	dwt_util_zero_vec_s(buffer_x, buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, buffer_y_elems);
	dwt_util_zero_vec_s(buffer_z, buffer_z_elems);
#endif

	// transform per 2x2x2 cubes

	// for each z
	for(int z = 0; z < super_z; z += step)
		// for each y
		for(int y = 0; y < super_y; y += step)
			// for each x (CPU cache friendly)
			for(int x = 0; x < super_x; x += step)
			{
				// call cube core
				cube_2x2x2(
					// coordinates
					x,
					y,
					z,
					// image size
					volume_src->size_x,
					volume_src->size_y,
					volume_src->size_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_src->data,
					// strides (to address another pixels)
					volume_src->stride_x,
					volume_src->stride_y,
					volume_src->stride_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_dst->data,
					// strides (to address another pixels)
					volume_dst->stride_x,
					volume_dst->stride_y,
					volume_dst->stride_z,
					// pointers to three buffers (aligned to 16 bytes = 4 floats)
					buffer_x,
					buffer_y,
					buffer_z,
					// super sizes in order to access elements in buffers
					super_x,
					super_y,
					super_z
				);
			}

#ifdef ALLOC_ON_HEAP
	free(buffer_x);
	free(buffer_y);
	free(buffer_z);
#endif
}

/**
 * vertical 2^3 core
 * inspired by: unified_4x4
 */
static
void cube_2x2x2_HORIZ(
	// coordinates
	int x,
	int y,
	int z,
	// image size
	int size_x,
	int size_y,
	int size_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *src_ptr,
	// strides (to address another pixels)
	int src_stride_x,
	int src_stride_y,
	int src_stride_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *dst_ptr,
	// strides (to address another pixels)
	int dst_stride_x,
	int dst_stride_y,
	int dst_stride_z,
	// pointers to three buffers (aligned to 16 bytes = 4 floats)
	float * ALIGNED(16) buffer_x,
	float * ALIGNED(16) buffer_y,
	float * ALIGNED(16) buffer_z,
	// super sizes in order to access elements in buffers
	int super_x,
	int super_y,
	int super_z
)
{
	UNUSED(super_y);
	UNUSED(super_z);

	const int overlap_x_L = 5;
	const int overlap_y_L = 5;
	const int overlap_z_L = 5;

	const int shift = 4;

	const int step_x = 2;
	const int step_y = 2;
	const int step_z = 2;

#ifdef __SSE__
	// 2^3 pixels = 8x float = 2x __m128 (z=0..1)
	__m128 t[2];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
				const int pos_y = virt2real(y, yy, overlap_y_L, size_y);
				const int pos_z = virt2real(z, zz, overlap_z_L, size_z);

				// [0] : yy=0 xx=0
				// [1] : yy=0 xx=1
				// [2] : yy=1 xx=0
				// [3] : yy=1 xx=1
				t[zz][yy*step_x+xx] = *( (float *)( (char *)src_ptr + pos_x*src_stride_x + pos_y*src_stride_y + pos_z*src_stride_z ) );
			}
		}
	}

	// CALC
#if 1
	// NOTE: along x-axis (4 times vert2x1)
	vert2x1(
		&t[0][0],
		&t[0][1],
		&t[0][0],
		&t[0][1],
		buffer_x + 0
	);
	vert2x1(
		&t[0][2],
		&t[0][3],
		&t[0][2],
		&t[0][3],
		buffer_x + 4
	);
	vert2x1(
		&t[1][0],
		&t[1][1],
		&t[1][0],
		&t[1][1],
		buffer_x + 8
	);
	vert2x1(
		&t[1][2],
		&t[1][3],
		&t[1][2],
		&t[1][3],
		buffer_x + 12
	);
#endif
#if 1
	// NOTE: along y-axis (4 times vert2x1)
	vert2x1(
		&t[0][0],
		&t[0][2],
		&t[0][0],
		&t[0][2],
		buffer_y + (x+0)*4 + (0+0)*(super_x*4)
	);
	vert2x1(
		&t[0][1],
		&t[0][3],
		&t[0][1],
		&t[0][3],
		buffer_y + (x+1)*4 + (0+0)*(super_x*4)
	);
	vert2x1(
		&t[1][0],
		&t[1][2],
		&t[1][0],
		&t[1][2],
		buffer_y + (x+0)*4 + (0+1)*(super_x*4)
	);
	vert2x1(
		&t[1][1],
		&t[1][3],
		&t[1][1],
		&t[1][3],
		buffer_y + (x+1)*4 + (0+1)*(super_x*4)
	);
#endif
#if 1
	// NOTE: along z-axis (4 times vert2x1)
	vert2x1(
		&t[0][0],
		&t[1][0],
		&t[0][0],
		&t[1][0],
		buffer_z + (x+0)*4 + (y+0)*(super_x*4)
	);
	vert2x1(
		&t[0][1],
		&t[1][1],
		&t[0][1],
		&t[1][1],
		buffer_z + (x+1)*4 + (y+0)*(super_x*4)
	);
	vert2x1(
		&t[0][2],
		&t[1][2],
		&t[0][2],
		&t[1][2],
		buffer_z + (x+0)*4 + (y+1)*(super_x*4)
	);
	vert2x1(
		&t[0][3],
		&t[1][3],
		&t[0][3],
		&t[1][3],
		buffer_z + (x+1)*4 + (y+1)*(super_x*4)
	);
#endif
#if 1
	// NOTE: scaling
	const float z1 = dwt_cdf97_s1_s;
	const float z3 = dwt_cdf97_s1_s*dwt_cdf97_s1_s*dwt_cdf97_s1_s;

	t[0] *= (__m128){ 1.f/z3, 1.f/z1, 1.f/z1, z1 };
	t[1] *= (__m128){ 1.f/z1, z1, z1, z3 };
#endif

	// STORE
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
				const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
				const int pos_z = virt2real_error(z-shift, zz, overlap_z_L, size_z);

				if( pos_x < 0 || pos_y < 0 || pos_z < 0 )
					continue;

				*( (float *)( (char *)dst_ptr + pos_x*dst_stride_x + pos_y*dst_stride_y + pos_z*dst_stride_z ) ) = t[zz][yy*step_x+xx];
			}
		}
	}
#endif
}

void cdf97_3f_op_HORIZ_vert2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// width of buffers: 4 floats
	const int buff_elem_size = 4;

	// step_x, step_y, step_z
	const int step = 2;

	// super sizes
	const int super_x = /*overlap_x_L*/5 + volume_src->size_x + /*overlap_x_R*/5;
	const int super_y = /*overlap_y_L*/5 + volume_src->size_y + /*overlap_y_R*/5;
	const int super_z = /*overlap_z_L*/5 + volume_src->size_z + /*overlap_z_R*/5;

	assert( 0 == super_x % step );
	assert( 0 == super_y % step );
	assert( 0 == super_z % step );

	// alloc buffers
	const int buffer_x_elems = 2*2 * buff_elem_size;
	const int buffer_y_elems = super_x*2 * buff_elem_size;
	const int buffer_z_elems = super_x*super_y * buff_elem_size;

#ifndef ALLOC_ON_HEAP
	float buffer_x[buffer_x_elems] ALIGNED(16); // NOTE: huge array on stack
	float buffer_y[buffer_y_elems] ALIGNED(16); // NOTE: huge array on stack
	float buffer_z[buffer_z_elems] ALIGNED(16); // NOTE: huge array on stack
#else
	float * ALIGNED(16) buffer_x = dwt_util_alloc_aligned_ex_reliably(buffer_x_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_y = dwt_util_alloc_aligned_ex_reliably(buffer_y_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_z = dwt_util_alloc_aligned_ex_reliably(buffer_z_elems, sizeof(float), 16);
#endif

#if 0
	// zero buffers
	dwt_util_zero_vec_s(buffer_x, buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, buffer_y_elems);
	dwt_util_zero_vec_s(buffer_z, buffer_z_elems);
#endif

	// transform per 2x2x2 cubes

	// for each z
	for(int z = 0; z < super_z; z += step)
		// for each y
		for(int y = 0; y < super_y; y += step)
			// for each x (CPU cache friendly)
			for(int x = 0; x < super_x; x += step)
			{
				// call cube core
				cube_2x2x2_HORIZ(
					// coordinates
					x,
					y,
					z,
					// image size
					volume_src->size_x,
					volume_src->size_y,
					volume_src->size_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_src->data,
					// strides (to address another pixels)
					volume_src->stride_x,
					volume_src->stride_y,
					volume_src->stride_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_dst->data,
					// strides (to address another pixels)
					volume_dst->stride_x,
					volume_dst->stride_y,
					volume_dst->stride_z,
					// pointers to three buffers (aligned to 16 bytes = 4 floats)
					buffer_x,
					buffer_y,
					buffer_z,
					// super sizes in order to access elements in buffers
					super_x,
					super_y,
					super_z
				);
			}

#ifdef ALLOC_ON_HEAP
	free(buffer_x);
	free(buffer_y);
	free(buffer_z);
#endif
}

void cdf97_3f_ip_sep_horizontal_s(struct volume_t *volume)
{
	assert( volume );

	// NOTE: over x-axis (x=0..size_x-1)
	// for each y
	for(int y = 0; y < volume->size_y; y++)
	{
		// for each z
		for(int z = 0; z < volume->size_z; z++)
		{
			// get pointer to (x=0,y,z)
			void *ptr = volume_get_pix(volume, 0, y, z);

			// transform vector (ptr, size_x, stride_x)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume->size_x, volume->stride_x);
		}
	}

	// NOTE: over y-axis (y=0..size_y-1)
	// for each x
	for(int x = 0; x < volume->size_x; x++)
	{
		// for each z
		for(int z = 0; z < volume->size_z; z++)
		{
			// get pointer to (x,y=0,z)
			void *ptr = volume_get_pix(volume, x, 0, z);

			// transform vector (ptr, size_y, stride_y)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume->size_y, volume->stride_y);
		}
	}

	// NOTE: over z-axis (z=0..size_z-1)
	// for each x
	for(int x = 0; x < volume->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume->size_z, volume->stride_z);
		}
	}
}

void cdf97_3f_op_sep_horizontal_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_src );
	assert( volume_dst );

	// NOTE: over x-axis (x=0..size_x-1)
	// for each y
	for(int y = 0; y < volume_dst->size_y; y++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// copy vector here
			dwt_util_memcpy_stride_s(
				volume_get_pix(volume_dst, 0, y, z),
				volume_dst->stride_x,
				volume_get_pix(volume_src, 0, y, z),
				volume_src->stride_x,
				volume_dst->size_x
			);

			// get pointer to (x=0,y,z)
			void *ptr = volume_get_pix(volume_dst, 0, y, z);

			// transform vector (ptr, size_x, stride_x)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_x, volume_dst->stride_x);
		}
	}

	// NOTE: over y-axis (y=0..size_y-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// get pointer to (x,y=0,z)
			void *ptr = volume_get_pix(volume_dst, x, 0, z);

			// transform vector (ptr, size_y, stride_y)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_y, volume_dst->stride_y);
		}
	}

	// NOTE: over z-axis (z=0..size_z-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume_dst->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume_dst, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_z, volume_dst->stride_z);
		}
	}
}

// only x-axes (+copying)
void cdf97_3f_op_sep_horizontal_x_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_src );
	assert( volume_dst );
#if 1
	// NOTE: over x-axis (x=0..size_x-1)
	// for each y
	for(int y = 0; y < volume_dst->size_y; y++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// copy vector here
			dwt_util_memcpy_stride_s(
				volume_get_pix(volume_dst, 0, y, z),
				volume_dst->stride_x,
				volume_get_pix(volume_src, 0, y, z),
				volume_src->stride_x,
				volume_dst->size_x
			);

			// get pointer to (x=0,y,z)
			void *ptr = volume_get_pix(volume_dst, 0, y, z);

			// transform vector (ptr, size_x, stride_x)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_x, volume_dst->stride_x);
		}
	}
#endif
#if 0
	// NOTE: over y-axis (y=0..size_y-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// get pointer to (x,y=0,z)
			void *ptr = volume_get_pix(volume_dst, x, 0, z);

			// transform vector (ptr, size_y, stride_y)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_y, volume_dst->stride_y);
		}
	}
#endif
#if 0
	// NOTE: over z-axis (z=0..size_z-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume_dst->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume_dst, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_z, volume_dst->stride_z);
		}
	}
#endif
}

// only y-axes
void cdf97_3f_op_sep_horizontal_y_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	UNUSED(volume_src);

	assert( volume_src );
	assert( volume_dst );
#if 0
	// NOTE: over x-axis (x=0..size_x-1)
	// for each y
	for(int y = 0; y < volume_dst->size_y; y++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// copy vector here
			dwt_util_memcpy_stride_s(
				volume_get_pix(volume_dst, 0, y, z),
				volume_dst->stride_x,
				volume_get_pix(volume_src, 0, y, z),
				volume_src->stride_x,
				volume_dst->size_x
			);

			// get pointer to (x=0,y,z)
			void *ptr = volume_get_pix(volume_dst, 0, y, z);

			// transform vector (ptr, size_x, stride_x)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_x, volume_dst->stride_x);
		}
	}
#endif
#if 1
	// NOTE: over y-axis (y=0..size_y-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// get pointer to (x,y=0,z)
			void *ptr = volume_get_pix(volume_dst, x, 0, z);

			// transform vector (ptr, size_y, stride_y)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_y, volume_dst->stride_y);
		}
	}
#endif
#if 0
	// NOTE: over z-axis (z=0..size_z-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume_dst->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume_dst, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_z, volume_dst->stride_z);
		}
	}
#endif
}

// only z-axes
void cdf97_3f_op_sep_horizontal_z_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	UNUSED(volume_src);

	assert( volume_src );
	assert( volume_dst );
#if 0
	// NOTE: over x-axis (x=0..size_x-1)
	// for each y
	for(int y = 0; y < volume_dst->size_y; y++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// copy vector here
			dwt_util_memcpy_stride_s(
				volume_get_pix(volume_dst, 0, y, z),
				volume_dst->stride_x,
				volume_get_pix(volume_src, 0, y, z),
				volume_src->stride_x,
				volume_dst->size_x
			);

			// get pointer to (x=0,y,z)
			void *ptr = volume_get_pix(volume_dst, 0, y, z);

			// transform vector (ptr, size_x, stride_x)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_x, volume_dst->stride_x);
		}
	}
#endif
#if 0
	// NOTE: over y-axis (y=0..size_y-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// get pointer to (x,y=0,z)
			void *ptr = volume_get_pix(volume_dst, x, 0, z);

			// transform vector (ptr, size_y, stride_y)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_y, volume_dst->stride_y);
		}
	}
#endif
#if 1
	// NOTE: over z-axis (z=0..size_z-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume_dst->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume_dst, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			fdwt1_single_cdf97_horizontal_min5_s(ptr, volume_dst->size_z, volume_dst->stride_z);
		}
	}
#endif
}

void cdf97_3f_op_sep_vertical_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_src );
	assert( volume_dst );

	// NOTE: over x-axis (x=0..size_x-1)
	// for each y
	for(int y = 0; y < volume_dst->size_y; y++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// copy vector here
			dwt_util_memcpy_stride_s(
				volume_get_pix(volume_dst, 0, y, z),
				volume_dst->stride_x,
				volume_get_pix(volume_src, 0, y, z),
				volume_src->stride_x,
				volume_dst->size_x
			);

			// get pointer to (x=0,y,z)
			void *ptr = volume_get_pix(volume_dst, 0, y, z);

			// transform vector (ptr, size_x, stride_x)
			fdwt1_single_cdf97_vertical_min5_s(ptr, volume_dst->size_x, volume_dst->stride_x);
		}
	}

	// NOTE: over y-axis (y=0..size_y-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each z
		for(int z = 0; z < volume_dst->size_z; z++)
		{
			// get pointer to (x,y=0,z)
			void *ptr = volume_get_pix(volume_dst, x, 0, z);

			// transform vector (ptr, size_y, stride_y)
			fdwt1_single_cdf97_vertical_min5_s(ptr, volume_dst->size_y, volume_dst->stride_y);
		}
	}

	// NOTE: over z-axis (z=0..size_z-1)
	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume_dst->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume_dst, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			fdwt1_single_cdf97_vertical_min5_s(ptr, volume_dst->size_z, volume_dst->stride_z);
		}
	}
}

void cdf97_3f_op_slices_vert4x4_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// for each slice
	for(int z = 0; z < volume_src->size_z; z++)
	{
		// get slices
		void *slice_dst = volume_get_slice(volume_dst, z);
		void *slice_src = volume_get_slice(volume_src, z);

#if 0
		// copy the slice
		dwt_util_copy3_s(
			slice_src,
			slice_dst,
			volume_src->stride_y,
			volume_src->stride_x,
			volume_dst->stride_y,
			volume_dst->stride_x,
			volume_src->size_x,
			volume_src->size_y
		);

		// transform the slice in-place
		cdf97_2f_dl_4x4_s(
			volume_dst->size_x,
			volume_dst->size_y,
			slice_dst,
			volume_dst->stride_y,
			volume_dst->stride_x,
			slice_dst,
			volume_dst->stride_y,
			volume_dst->stride_x
		);
#else
		// transform the slice out-of-place
		cdf97_2f_dl_4x4_s(
			volume_dst->size_x,
			volume_dst->size_y,
			slice_src,
			volume_src->stride_y,
			volume_src->stride_x,
			slice_dst,
			volume_dst->stride_y,
			volume_dst->stride_x
		);
#endif
	}

	// now, a transform over z-axe is missing
	// lets perform it as separable-vertical

	// for each x
	for(int x = 0; x < volume_dst->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume_dst->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume_dst, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			fdwt1_single_cdf97_vertical_min5_s(ptr, volume_dst->size_z, volume_dst->stride_z);
		}
	}
}

void cdf97_3i_ip_sep_horizontal_s(struct volume_t *volume)
{
	assert( volume );

	// NOTE: over x-axis (x=0..size_x-1)
	// for each y
	for(int y = 0; y < volume->size_y; y++)
	{
		// for each z
		for(int z = 0; z < volume->size_z; z++)
		{
			// get pointer to (x=0,y,z)
			void *ptr = volume_get_pix(volume, 0, y, z);

			// transform vector (ptr, size_x, stride_x)
			dwt_cdf97_1i_inplace_s(ptr, volume->stride_x, volume->size_x, 1);
		}
	}

	// NOTE: over y-axis (y=0..size_y-1)
	// for each x
	for(int x = 0; x < volume->size_x; x++)
	{
		// for each z
		for(int z = 0; z < volume->size_z; z++)
		{
			// get pointer to (x,y=0,z)
			void *ptr = volume_get_pix(volume, x, 0, z);

			// transform vector (ptr, size_y, stride_y)
			dwt_cdf97_1i_inplace_s(ptr, volume->stride_y, volume->size_y, 1);
		}
	}

	// NOTE: over z-axis (z=0..size_z-1)
	// for each x
	for(int x = 0; x < volume->size_x; x++)
	{
		// for each y
		for(int y = 0; y < volume->size_y; y++)
		{
			// get pointer to (x,y,z=0)
			void *ptr = volume_get_pix(volume, x, y, 0);

			// transform vector (ptr, size_z, stride_z)
			dwt_cdf97_1i_inplace_s(ptr, volume->stride_z, volume->size_z, 1);
		}
	}
}

// vert_2x4
#ifdef __SSE__
static
void vert_2x4(
	// left input column [4]
	__m128 in0,
	// right input column [4]
	__m128 in1,
	// output 0 [4]
	__m128 *out0,
	// output 1 [4]
	__m128 *out1,
	// 4x buffer "L" with stride = (1*4) * sizeof(float)
	float *buff
)
{
	// weights
	const __m128 w0 = { +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s };
	const __m128 w1 = { -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s };
	const __m128 w2 = { +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s };
	const __m128 w3 = { -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s };

	// variables
	__m128 l0, l1, l2, l3;
	__m128 c0, c1, c2, c3;
	__m128 r0, r1, r2, r3;
	__m128 x0, x1;
	__m128 y0, y1;

	// load "L"
	l0 = _mm_load_ps(&buff[0*(1*4)]);
	l1 = _mm_load_ps(&buff[1*(1*4)]);
	l2 = _mm_load_ps(&buff[2*(1*4)]);
	l3 = _mm_load_ps(&buff[3*(1*4)]);

	// inputs
	x0 = in0;
	x1 = in1;

	// shuffles
	y0 = l0;
	c0 = l1;
	c1 = l2;
	c2 = l3;
	c3 = x0;

	// operation
	r3 = x1;
	r2 = c3 + w3 * (l3 + r3);
	r1 = c2 + w2 * (l2 + r2);
	r0 = c1 + w1 * (l1 + r1);
	y1 = c0 + w0 * (l0 + r0);

	// update
	l0 = r0;
	l1 = r1;
	l2 = r2;
	l3 = r3;

	// outputs
	*out0 = y0;
	*out1 = y1;

	// store "L"
	_mm_store_ps(&buff[0*(1*4)], l0);
	_mm_store_ps(&buff[1*(1*4)], l1);
	_mm_store_ps(&buff[2*(1*4)], l2);
	_mm_store_ps(&buff[3*(1*4)], l3);
}
#endif

/**
 * vertical 4x4x2 core
 * inspired by: unified_4x4
 */
static
void cube_4x4x2(
	// coordinates
	int x,
	int y,
	int z,
	// image size
	int size_x,
	int size_y,
	int size_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *src_ptr,
	// strides (to address another pixels)
	int src_stride_x,
	int src_stride_y,
	int src_stride_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *dst_ptr,
	// strides (to address another pixels)
	int dst_stride_x,
	int dst_stride_y,
	int dst_stride_z,
	// pointers to three buffers (aligned to 16 bytes = 4 floats)
	float * ALIGNED(16) buffer_x,
	float * ALIGNED(16) buffer_y,
	float * ALIGNED(16) buffer_z,
	// super sizes in order to access elements in buffers
	int super_x,
	int super_y,
	int super_z
)
{
	UNUSED(super_y);
	UNUSED(super_z);

	const int overlap_x_L = 7;
	const int overlap_y_L = 7;
	const int overlap_z_L = 5;

	const int shift = 4;

	const int step_x = 4;
	const int step_y = 4;
	const int step_z = 2;

#ifdef __SSE__
	// 4^3 / 2 pixels = 32x float = 8x __m128 (z=0..1, y=0..3)
	__m128 t[2][4]; // NOTE: [z][y][x]

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
				const int pos_y = virt2real(y, yy, overlap_y_L, size_y);
				const int pos_z = virt2real(z, zz, overlap_z_L, size_z);

				// NOTE: transposed x-y
				t[zz][xx][yy] = *( (float *)( (char *)src_ptr + pos_x*src_stride_x + pos_y*src_stride_y + pos_z*src_stride_z ) );
			}
		}
	}

	// CALC
#if 1
	// NOTE: along x-axis (4 times vert2x1)
	// front 4x4 slice
	vert_2x4(
		t[0][0],
		t[0][1],
		&t[0][0],
		&t[0][1],
		buffer_x + (y+0)*4 + (z+0)*(super_x*4)
	);
	vert_2x4(
		t[0][2],
		t[0][3],
		&t[0][2],
		&t[0][3],
		buffer_x + (y+0)*4 + (z+0)*(super_x*4)
	);
	// back 4x4 slice
	vert_2x4(
		t[1][0],
		t[1][1],
		&t[1][0],
		&t[1][1],
		buffer_x + (y+0)*4 + (z+1)*(super_x*4)
	);
	vert_2x4(
		t[1][2],
		t[1][3],
		&t[1][2],
		&t[1][3],
		buffer_x + (y+0)*4 + (z+1)*(super_x*4)
	);
#endif
	// transpose
	_MM_TRANSPOSE4_PS(t[0][0], t[0][1], t[0][2], t[0][3]);
	_MM_TRANSPOSE4_PS(t[1][0], t[1][1], t[1][2], t[1][3]);
#if 1
	// NOTE: along y-axis (4 times vert2x1)
	// front 4x4 slice
	vert_2x4(
		t[0][0],
		t[0][1],
		&t[0][0],
		&t[0][1],
		buffer_y + (x+0)*4 + (z+0)*(super_x*4)
	);
	vert_2x4(
		t[0][2],
		t[0][3],
		&t[0][2],
		&t[0][3],
		buffer_y + (x+0)*4 + (z+0)*(super_x*4)
	);
	// back 4x4 slice
	vert_2x4(
		t[1][0],
		t[1][1],
		&t[1][0],
		&t[1][1],
		buffer_y + (x+0)*4 + (z+1)*(super_x*4)
	);
	vert_2x4(
		t[1][2],
		t[1][3],
		&t[1][2],
		&t[1][3],
		buffer_y + (x+0)*4 + (z+1)*(super_x*4)
	);
#endif
#if 1
	// NOTE: along z-axis (4 times vert2x1)
	vert_2x4(
		t[0][0],
		t[1][0],
		&t[0][0],
		&t[1][0],
		buffer_z + (x+0)*4 + (y+0)*(super_x*4)
	);
	vert_2x4(
		t[0][1],
		t[1][1],
		&t[0][1],
		&t[1][1],
		buffer_z + (x+0)*4 + (y+1)*(super_x*4)
	);
	vert_2x4(
		t[0][2],
		t[1][2],
		&t[0][2],
		&t[1][2],
		buffer_z + (x+0)*4 + (y+2)*(super_x*4)
	);
	vert_2x4(
		t[0][3],
		t[1][3],
		&t[0][3],
		&t[1][3],
		buffer_z + (x+0)*4 + (y+3)*(super_x*4)
	);
#endif
#if 1
	// NOTE: scaling
	const float z1 = dwt_cdf97_s1_s;
	const float z3 = dwt_cdf97_s1_s*dwt_cdf97_s1_s*dwt_cdf97_s1_s;
	const float r1 = 1.f/z1;
	const float r3 = 1.f/z3;

	t[0][0] *= (__m128){ r3, r1, r3, r1 };
	t[0][1] *= (__m128){ r1, z1, r1, z1 };
	t[0][2] *= (__m128){ r3, r1, r3, r1 };
	t[0][3] *= (__m128){ r1, z1, r1, z1 };

	t[1][0] *= (__m128){ r1, z1, r1, z1 };
	t[1][1] *= (__m128){ z1, z3, z1, z3 };
	t[1][2] *= (__m128){ r1, z1, r1, z1 };
	t[1][3] *= (__m128){ z1, z3, z1, z3 };
#endif

	// STORE
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
				const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
				const int pos_z = virt2real_error(z-shift, zz, overlap_z_L, size_z);

				if( pos_x < 0 || pos_y < 0 || pos_z < 0 )
					continue;

				*( (float *)( (char *)dst_ptr + pos_x*dst_stride_x + pos_y*dst_stride_y + pos_z*dst_stride_z ) ) = t[zz][yy][xx];
			}
		}
	}
#endif
}

void cdf97_3f_op_cube_vert4x4x2_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// width of buffers: 4 floats
	const int buff_elem_size = 4;

	// step_x, step_y, step_z
	const int step_x = 4;
	const int step_y = 4;
	const int step_z = 2;

	// super sizes
	const int super_x = /*overlap_x_L*/7 + volume_src->size_x + /*overlap_x_R*/5;
	const int super_y = /*overlap_y_L*/7 + volume_src->size_y + /*overlap_y_R*/5;
	const int super_z = /*overlap_z_L*/5 + volume_src->size_z + /*overlap_z_R*/5;

	assert( 0 == super_x % step_x );
	assert( 0 == super_y % step_y );
	assert( 0 == super_z % step_z );

	// alloc buffers
	const int buffer_x_elems = super_y*super_z * buff_elem_size;
	const int buffer_y_elems = super_x*super_z * buff_elem_size;
	const int buffer_z_elems = super_x*super_y * buff_elem_size;

	float * ALIGNED(16) buffer_x = dwt_util_alloc_aligned_ex_reliably(buffer_x_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_y = dwt_util_alloc_aligned_ex_reliably(buffer_y_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_z = dwt_util_alloc_aligned_ex_reliably(buffer_z_elems, sizeof(float), 16);

#if 0
	// zero buffers
	dwt_util_zero_vec_s(buffer_x, buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, buffer_y_elems);
	dwt_util_zero_vec_s(buffer_z, buffer_z_elems);
#endif

	// transform per 4x4x2 cubes

	// for each z
	for(int z = 0; z < super_z; z += step_z)
		// for each y
		for(int y = 0; y < super_y; y += step_y)
			// for each x (CPU cache friendly)
			for(int x = 0; x < super_x; x += step_x)
			{
				// call cube core
				cube_4x4x2(
					// coordinates
					x,
					y,
					z,
					// image size
					volume_src->size_x,
					volume_src->size_y,
					volume_src->size_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_src->data,
					// strides (to address another pixels)
					volume_src->stride_x,
					volume_src->stride_y,
					volume_src->stride_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_dst->data,
					// strides (to address another pixels)
					volume_dst->stride_x,
					volume_dst->stride_y,
					volume_dst->stride_z,
					// pointers to three buffers (aligned to 16 bytes = 4 floats)
					buffer_x,
					buffer_y,
					buffer_z,
					// super sizes in order to access elements in buffers
					super_x,
					super_y,
					super_z
				);
			}

	free(buffer_x);
	free(buffer_y);
	free(buffer_z);
}

/**
 * vertical 4x4x2 core
 * inspired by: unified_4x4
 */
static
void cube_4x4x2_HORIZ(
	// coordinates
	int x,
	int y,
	int z,
	// image size
	int size_x,
	int size_y,
	int size_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *src_ptr,
	// strides (to address another pixels)
	int src_stride_x,
	int src_stride_y,
	int src_stride_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *dst_ptr,
	// strides (to address another pixels)
	int dst_stride_x,
	int dst_stride_y,
	int dst_stride_z,
	// pointers to three buffers (aligned to 16 bytes = 4 floats)
	float * ALIGNED(16) buffer_x,
	float * ALIGNED(16) buffer_y,
	float * ALIGNED(16) buffer_z,
	// super sizes in order to access elements in buffers
	int super_x,
	int super_y,
	int super_z
)
{
	UNUSED(super_y);
	UNUSED(super_z);

	const int overlap_x_L = 7;
	const int overlap_y_L = 7;
	const int overlap_z_L = 5;

	const int shift = 4;

	const int step_x = 4;
	const int step_y = 4;
	const int step_z = 2;

#ifdef __SSE__
	// 4^3 / 2 pixels = 32x float = 8x __m128 (z=0..1, y=0..3)
	__m128 t[2][4]; // NOTE: [z][y][x]

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
				const int pos_y = virt2real(y, yy, overlap_y_L, size_y);
				const int pos_z = virt2real(z, zz, overlap_z_L, size_z);

				// NOTE: transposed x-y
				t[zz][xx][yy] = *( (float *)( (char *)src_ptr + pos_x*src_stride_x + pos_y*src_stride_y + pos_z*src_stride_z ) );
			}
		}
	}

	// CALC
#if 1
	// NOTE: along x-axis (4 times vert2x1)
	// front 4x4 slice
	vert_2x4(
		t[0][0],
		t[0][1],
		&t[0][0],
		&t[0][1],
		buffer_x + 0
	);
	vert_2x4(
		t[0][2],
		t[0][3],
		&t[0][2],
		&t[0][3],
		buffer_x + 0
	);
	// back 4x4 slice
	vert_2x4(
		t[1][0],
		t[1][1],
		&t[1][0],
		&t[1][1],
		buffer_x + 16
	);
	vert_2x4(
		t[1][2],
		t[1][3],
		&t[1][2],
		&t[1][3],
		buffer_x + 16
	);
#endif
	// transpose
	_MM_TRANSPOSE4_PS(t[0][0], t[0][1], t[0][2], t[0][3]);
	_MM_TRANSPOSE4_PS(t[1][0], t[1][1], t[1][2], t[1][3]);
#if 1
	// NOTE: along y-axis (4 times vert2x1)
	// front 4x4 slice
	vert_2x4(
		t[0][0],
		t[0][1],
		&t[0][0],
		&t[0][1],
		buffer_y + (x+0)*4 + (0+0)*(super_x*4)
	);
	vert_2x4(
		t[0][2],
		t[0][3],
		&t[0][2],
		&t[0][3],
		buffer_y + (x+0)*4 + (0+0)*(super_x*4)
	);
	// back 4x4 slice
	vert_2x4(
		t[1][0],
		t[1][1],
		&t[1][0],
		&t[1][1],
		buffer_y + (x+0)*4 + (0+1)*(super_x*4)
	);
	vert_2x4(
		t[1][2],
		t[1][3],
		&t[1][2],
		&t[1][3],
		buffer_y + (x+0)*4 + (0+1)*(super_x*4)
	);
#endif
#if 1
	// NOTE: along z-axis (4 times vert2x1)
	vert_2x4(
		t[0][0],
		t[1][0],
		&t[0][0],
		&t[1][0],
		buffer_z + (x+0)*4 + (y+0)*(super_x*4)
	);
	vert_2x4(
		t[0][1],
		t[1][1],
		&t[0][1],
		&t[1][1],
		buffer_z + (x+0)*4 + (y+1)*(super_x*4)
	);
	vert_2x4(
		t[0][2],
		t[1][2],
		&t[0][2],
		&t[1][2],
		buffer_z + (x+0)*4 + (y+2)*(super_x*4)
	);
	vert_2x4(
		t[0][3],
		t[1][3],
		&t[0][3],
		&t[1][3],
		buffer_z + (x+0)*4 + (y+3)*(super_x*4)
	);
#endif
#if 1
	// NOTE: scaling
	const float z1 = dwt_cdf97_s1_s;
	const float z3 = dwt_cdf97_s1_s*dwt_cdf97_s1_s*dwt_cdf97_s1_s;
	const float r1 = 1.f/z1;
	const float r3 = 1.f/z3;

	t[0][0] *= (__m128){ r3, r1, r3, r1 };
	t[0][1] *= (__m128){ r1, z1, r1, z1 };
	t[0][2] *= (__m128){ r3, r1, r3, r1 };
	t[0][3] *= (__m128){ r1, z1, r1, z1 };

	t[1][0] *= (__m128){ r1, z1, r1, z1 };
	t[1][1] *= (__m128){ z1, z3, z1, z3 };
	t[1][2] *= (__m128){ r1, z1, r1, z1 };
	t[1][3] *= (__m128){ z1, z3, z1, z3 };
#endif

	// STORE
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
				const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
				const int pos_z = virt2real_error(z-shift, zz, overlap_z_L, size_z);

				if( pos_x < 0 || pos_y < 0 || pos_z < 0 )
					continue;

				*( (float *)( (char *)dst_ptr + pos_x*dst_stride_x + pos_y*dst_stride_y + pos_z*dst_stride_z ) ) = t[zz][yy][xx];
			}
		}
	}
#endif
}

void cdf97_3f_op_HORIZ_vert4x4x2_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// width of buffers: 4 floats
	const int buff_elem_size = 4;

	// step_x, step_y, step_z
	const int step_x = 4;
	const int step_y = 4;
	const int step_z = 2;

	// super sizes
	const int super_x = /*overlap_x_L*/7 + volume_src->size_x + /*overlap_x_R*/5;
	const int super_y = /*overlap_y_L*/7 + volume_src->size_y + /*overlap_y_R*/5;
	const int super_z = /*overlap_z_L*/5 + volume_src->size_z + /*overlap_z_R*/5;

	assert( 0 == super_x % step_x );
	assert( 0 == super_y % step_y );
	assert( 0 == super_z % step_z );

	// alloc buffers
	const int buffer_x_elems = 2*4 * buff_elem_size;
	const int buffer_y_elems = super_x*2 * buff_elem_size;
	const int buffer_z_elems = super_x*super_y * buff_elem_size;

	float * ALIGNED(16) buffer_x = dwt_util_alloc_aligned_ex_reliably(buffer_x_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_y = dwt_util_alloc_aligned_ex_reliably(buffer_y_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_z = dwt_util_alloc_aligned_ex_reliably(buffer_z_elems, sizeof(float), 16);

#if 0
	// zero buffers
	dwt_util_zero_vec_s(buffer_x, buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, buffer_y_elems);
	dwt_util_zero_vec_s(buffer_z, buffer_z_elems);
#endif

	// transform per 4x4x2 cubes

	// for each z
	for(int z = 0; z < super_z; z += step_z)
		// for each y
		for(int y = 0; y < super_y; y += step_y)
			// for each x (CPU cache friendly)
			for(int x = 0; x < super_x; x += step_x)
			{
				// call cube core
				cube_4x4x2_HORIZ(
					// coordinates
					x,
					y,
					z,
					// image size
					volume_src->size_x,
					volume_src->size_y,
					volume_src->size_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_src->data,
					// strides (to address another pixels)
					volume_src->stride_x,
					volume_src->stride_y,
					volume_src->stride_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_dst->data,
					// strides (to address another pixels)
					volume_dst->stride_x,
					volume_dst->stride_y,
					volume_dst->stride_z,
					// pointers to three buffers (aligned to 16 bytes = 4 floats)
					buffer_x,
					buffer_y,
					buffer_z,
					// super sizes in order to access elements in buffers
					super_x,
					super_y,
					super_z
				);
			}

	free(buffer_x);
	free(buffer_y);
	free(buffer_z);
}

#ifdef __SSE__
static
__m128 diag2x2_elem_op(
	__m128 input, // in [ y0x0 y0x1 y1x0 y1x1 ]
	float * ALIGNED(16) buffL, // l.L [3*4*float]
	float * ALIGNED(16) buffR  // l.R [3*4*float]
)
{
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	__m128 z;

	// L+R
	op4s_sdl2_shuffle_input_low_s_sse(input, *(__m128 *)(buffL+4), *(__m128 *)(buffL+8));
	op4s_sdl2_shuffle_input_high_s_sse(input, *(__m128 *)(buffR+4), *(__m128 *)(buffR+8));

	// L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buffL+4), w, *(__m128 *)(buffL+0), *(__m128 *)(buffL+8));
	op4s_sdl2_output_low_s_sse(input, *(__m128 *)(buffL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buffL+4), *(__m128 *)(buffL+0), *(__m128 *)(buffL+8), z);

	// R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(buffR+4), w, *(__m128 *)(buffR+0), *(__m128 *)(buffR+8));
	op4s_sdl2_output_high_s_sse(input, *(__m128 *)(buffR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(buffR+4), *(__m128 *)(buffR+0), *(__m128 *)(buffR+8), z);

	return input;
}
#endif

/**
 * diagonal 2x2x2 core
 */
static
void cube_diag2x2x2(
	// coordinates
	int x,
	int y,
	int z,
	// image size
	int size_x,
	int size_y,
	int size_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *src_ptr,
	// strides (to address another pixels)
	int src_stride_x,
	int src_stride_y,
	int src_stride_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *dst_ptr,
	// strides (to address another pixels)
	int dst_stride_x,
	int dst_stride_y,
	int dst_stride_z,
	// pointers to three buffers (aligned to 16 bytes = 4 floats)
	float * ALIGNED(16) buffer_x,
	float * ALIGNED(16) buffer_y,
	float * ALIGNED(16) buffer_z,
	// super sizes in order to access elements in buffers
	int super_x,
	int super_y,
	int super_z
)
{
	UNUSED(super_z);

	const int overlap_x_L = 5;
	const int overlap_y_L = 5;
	const int overlap_z_L = 5;

	const int shift = 10;

	const int step_x = 2;
	const int step_y = 2;
	const int step_z = 2;

#ifdef __SSE__
	// 2^3 pixels = 8x float = 2x __m128 (z=0..1)
	__m128 t[2];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
				const int pos_y = virt2real(y, yy, overlap_y_L, size_y);
				const int pos_z = virt2real(z, zz, overlap_z_L, size_z);

				// [0] : yy=0 xx=0
				// [1] : yy=0 xx=1
				// [2] : yy=1 xx=0
				// [3] : yy=1 xx=1
				t[zz][yy*step_x+xx] = *( (float *)( (char *)src_ptr + pos_x*src_stride_x + pos_y*src_stride_y + pos_z*src_stride_z ) );
			}
		}
	}

	// CALC
#if 1
	// NOTE: along x-axis
	t[0] = diag2x2_elem_op(
		t[0],
		buffer_x + (y+0)*12 + (z+0)*(super_y*12),
		buffer_x + (y+1)*12 + (z+0)*(super_y*12)
	);
	t[1] = diag2x2_elem_op(
		t[1],
		buffer_x + (y+0)*12 + (z+1)*(super_y*12),
		buffer_x + (y+1)*12 + (z+1)*(super_y*12)
	);
#endif
	// transpose
	_MM_TRANSPOSE1_PS(t[0]);
	_MM_TRANSPOSE1_PS(t[1]);
#if 1
	// NOTE: along y-axis
	t[0] = diag2x2_elem_op(
		t[0],
		buffer_y + (x+0)*12 + (z+0)*(super_x*12),
		buffer_y + (x+1)*12 + (z+0)*(super_x*12)
	);
	t[1] = diag2x2_elem_op(
		t[1],
		buffer_y + (x+0)*12 + (z+1)*(super_x*12),
		buffer_y + (x+1)*12 + (z+1)*(super_x*12)
	);
#endif
	// transpose
	_MM_TRANSPOSE2_PS(t[0], t[1]);
#if 1
	// NOTE: along z-axis
	t[0] = diag2x2_elem_op(
		t[0],
		buffer_z + (x+0)*12 + (y+0)*(super_x*12),
		buffer_z + (x+1)*12 + (y+0)*(super_x*12)
	);
	t[1] = diag2x2_elem_op(
		t[1],
		buffer_z + (x+0)*12 + (y+1)*(super_x*12),
		buffer_z + (x+1)*12 + (y+1)*(super_x*12)
	);
#endif
#if 1
	// NOTE: scaling
	const float z1 = dwt_cdf97_s1_s;
	const float z3 = dwt_cdf97_s1_s*dwt_cdf97_s1_s*dwt_cdf97_s1_s;

	t[0] *= (__m128){
		1.f/z3, 1.f/z1,
		1.f/z1, z1
	};
	t[1] *= (__m128){
		1.f/z1, z1,
		z1, z3
	};
#endif

	// STORE
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
				const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
				const int pos_z = virt2real_error(z-shift, zz, overlap_z_L, size_z);

				if( pos_x < 0 || pos_y < 0 || pos_z < 0 )
					continue;

				// NOTE: transposed x<->y, then y<->z
				*( (float *)( (char *)dst_ptr + pos_x*dst_stride_x + pos_y*dst_stride_y + pos_z*dst_stride_z ) ) = t[xx][yy*step_x+zz];
			}
		}
	}
#endif
}

void cdf97_3f_op_baseline_diag2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// width of buffers
	const int buff_elem_size = 12;

	// step_x, step_y, step_z
	const int step = 2;

	// super sizes
	const int super_x = /*overlap_x_L*/5 + volume_src->size_x + /*overlap_x_R*/11;
	const int super_y = /*overlap_y_L*/5 + volume_src->size_y + /*overlap_y_R*/11;
	const int super_z = /*overlap_z_L*/5 + volume_src->size_z + /*overlap_z_R*/11;

	assert( 0 == super_x % step );
	assert( 0 == super_y % step );
	assert( 0 == super_z % step );

	// alloc buffers
	const int buffer_x_elems = super_y*super_z * buff_elem_size;
	const int buffer_y_elems = super_x*super_z * buff_elem_size;
	const int buffer_z_elems = super_x*super_y * buff_elem_size;

	float * ALIGNED(16) buffer_x = dwt_util_alloc_aligned_ex_reliably(buffer_x_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_y = dwt_util_alloc_aligned_ex_reliably(buffer_y_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_z = dwt_util_alloc_aligned_ex_reliably(buffer_z_elems, sizeof(float), 16);

#if 0
	// zero buffers
	dwt_util_zero_vec_s(buffer_x, buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, buffer_y_elems);
	dwt_util_zero_vec_s(buffer_z, buffer_z_elems);
#endif

	// transform per 2x2x2 cubes

	// for each z
	for(int z = 0; z < super_z; z += step)
		// for each y
		for(int y = 0; y < super_y; y += step)
			// for each x (CPU cache friendly)
			for(int x = 0; x < super_x; x += step)
			{
				// call cube core
				cube_diag2x2x2(
					// coordinates
					x,
					y,
					z,
					// image size
					volume_src->size_x,
					volume_src->size_y,
					volume_src->size_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_src->data,
					// strides (to address another pixels)
					volume_src->stride_x,
					volume_src->stride_y,
					volume_src->stride_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_dst->data,
					// strides (to address another pixels)
					volume_dst->stride_x,
					volume_dst->stride_y,
					volume_dst->stride_z,
					// pointers to three buffers (aligned to 16 bytes = 4 floats)
					buffer_x,
					buffer_y,
					buffer_z,
					// super sizes in order to access elements in buffers
					super_x,
					super_y,
					super_z
				);
			}

	free(buffer_x);
	free(buffer_y);
	free(buffer_z);
}

#ifdef __SSE__
static
__m128 diag2x2x1_elem_op(
	__m128 input, // input [ y0x0 y0x1 y1x0 y1x1 ]
	__m128 *buff // [24*float]
)
{
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	__m128 z;

	// L+R
	op4s_sdl2_shuffle_input_low_s_sse(input, buff[1], buff[2]);
	op4s_sdl2_shuffle_input_high_s_sse(input, buff[4], buff[5]);

	// L
	op4s_sdl2_op_s_sse(z, buff[1], w, buff[0], buff[2]);
	op4s_sdl2_output_low_s_sse(input, buff[0], z);
	op4s_sdl2_update_s_sse(buff[1], buff[0], buff[2], z);

	// R
	op4s_sdl2_op_s_sse(z, buff[4], w, buff[3], buff[5]);
	op4s_sdl2_output_high_s_sse(input, buff[3], z);
	op4s_sdl2_update_s_sse(buff[4], buff[3], buff[5], z);

	return input;
}
#endif

/**
 * diagonal 2x2x2 core
 */
static
void cube_diag2x2x2_HORIZ(
	// coordinates
	int virt_x,
	int virt_y,
	int virt_z,
	// image size
	int size_x,
	int size_y,
	int size_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *src_ptr,
	// strides (to address another pixels)
	int src_stride_x,
	int src_stride_y,
	int src_stride_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *dst_ptr,
	// strides (to address another pixels)
	int dst_stride_x,
	int dst_stride_y,
	int dst_stride_z,
	// pointers to three buffers (aligned to 16 bytes = 4 floats)
	float * ALIGNED(16) buffer_x,
	float * ALIGNED(16) buffer_y,
	float * ALIGNED(16) buffer_z,
	// super sizes in order to access elements in buffers
	int super_x,
	int super_y,
	int super_z
)
{
	UNUSED(super_y);
	UNUSED(super_z);

	const int overlap_x_L = 5;
	const int overlap_y_L = 5;
	const int overlap_z_L = 5;

	const int shift = 10;

	const int step_x = 2;
	const int step_y = 2;
	const int step_z = 2;

#ifdef __SSE__
	// 2^3 pixels = 8x float = 2x __m128 (z=0..1)
	__m128 data[2];

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real(virt_x, xx, overlap_x_L, size_x);
				const int pos_y = virt2real(virt_y, yy, overlap_y_L, size_y);
				const int pos_z = virt2real(virt_z, zz, overlap_z_L, size_z);

				// [0] : yy=0 xx=0
				// [1] : yy=0 xx=1
				// [2] : yy=1 xx=0
				// [3] : yy=1 xx=1
				data[zz][yy*step_x+xx] = *( (float *)( (char *)src_ptr + pos_x*src_stride_x + pos_y*src_stride_y + pos_z*src_stride_z ) );
			}
		}
	}

	// CALC
#if 1
	// NOTE: along x-axis
	data[0] = diag2x2x1_elem_op(
		data[0],
		(__m128 *)( buffer_x + 0 )
	);
	data[1] = diag2x2x1_elem_op(
		data[1],
		(__m128 *)( buffer_x + 24 )
	);
#endif
	// transpose x-y
	_MM_TRANSPOSE1_PS(data[0]);
	_MM_TRANSPOSE1_PS(data[1]);
#if 1
	// NOTE: along y-axis
	data[0] = diag2x2x1_elem_op(
		data[0],
		(__m128 *)( buffer_y + (virt_x)*24 + 0 )
	);
	data[1] = diag2x2x1_elem_op(
		data[1],
		(__m128 *)( buffer_y + (virt_x)*24 + 24 )
	);
#endif
	// transpose y-z
	_MM_TRANSPOSE2_PS(data[0], data[1]);
#if 1
	// NOTE: along z-axis
	data[0] = diag2x2x1_elem_op(
		data[0],
		(__m128 *)( buffer_z + (virt_x)*24 + (virt_y)*(super_x*24) )
	);
	data[1] = diag2x2x1_elem_op(
		data[1],
		(__m128 *)( buffer_z + (virt_x)*24 + (virt_y)*(super_x*24) + 24 )
	);
#endif
#if 1
	// NOTE: scaling
	const float z1 = dwt_cdf97_s1_s;
	const float z3 = dwt_cdf97_s1_s*dwt_cdf97_s1_s*dwt_cdf97_s1_s;

	data[0] *= (const __m128){
		1.f/z3, 1.f/z1,
		1.f/z1, z1
	};
	data[1] *= (const __m128){
		1.f/z1, z1,
		z1, z3
	};
#endif

	// STORE
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real_error(virt_x-shift, xx, overlap_x_L, size_x);
				const int pos_y = virt2real_error(virt_y-shift, yy, overlap_y_L, size_y);
				const int pos_z = virt2real_error(virt_z-shift, zz, overlap_z_L, size_z);

				if( pos_x < 0 || pos_y < 0 || pos_z < 0 )
					continue;

				// NOTE: transposed x<->y, then y<->z
				*( (float *)( (char *)dst_ptr + pos_x*dst_stride_x + pos_y*dst_stride_y + pos_z*dst_stride_z ) ) = data[xx][yy*step_x+zz];
			}
		}
	}
#endif
}

void cdf97_3f_op_HORIZ_diag2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// width of buffers
	const int buff_elem_size = 2*12;

	// step_x, step_y, step_z
	const int step = 2;

	// super sizes
	const int super_x = /*overlap_x_L*/5 + volume_src->size_x + /*overlap_x_R*/11;
	const int super_y = /*overlap_y_L*/5 + volume_src->size_y + /*overlap_y_R*/11;
	const int super_z = /*overlap_z_L*/5 + volume_src->size_z + /*overlap_z_R*/11;

	assert( 0 == super_x % step );
	assert( 0 == super_y % step );
	assert( 0 == super_z % step );

	// alloc buffers
	const int buffer_x_elems = /*super_y*super_z*/ 2*2 * buff_elem_size;
	const int buffer_y_elems = super_x*/*super_z*/2 * buff_elem_size;
	const int buffer_z_elems = super_x*super_y * buff_elem_size;

	float * ALIGNED(16) buffer_x = dwt_util_alloc_aligned_ex_reliably(buffer_x_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_y = dwt_util_alloc_aligned_ex_reliably(buffer_y_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_z = dwt_util_alloc_aligned_ex_reliably(buffer_z_elems, sizeof(float), 16);

#if 0
	// zero buffers
	dwt_util_zero_vec_s(buffer_x, buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, buffer_y_elems);
	dwt_util_zero_vec_s(buffer_z, buffer_z_elems);
#endif

	// transform per 2x2x2 cubes

	// for each z
	for(int z = 0; z < super_z; z += step)
		// for each y
		for(int y = 0; y < super_y; y += step)
			// for each x (CPU cache friendly)
			for(int x = 0; x < super_x; x += step)
			{
				// call cube core
				cube_diag2x2x2_HORIZ(
					// coordinates
					x,
					y,
					z,
					// image size
					volume_src->size_x,
					volume_src->size_y,
					volume_src->size_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_src->data,
					// strides (to address another pixels)
					volume_src->stride_x,
					volume_src->stride_y,
					volume_src->stride_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_dst->data,
					// strides (to address another pixels)
					volume_dst->stride_x,
					volume_dst->stride_y,
					volume_dst->stride_z,
					// pointers to three buffers (aligned to 16 bytes = 4 floats)
					buffer_x,
					buffer_y,
					buffer_z,
					// super sizes in order to access elements in buffers
					super_x,
					super_y,
					super_z
				);
			}

	free(buffer_x);
	free(buffer_y);
	free(buffer_z);
}

#ifdef __SSE__
static
void vert_2x4x1(
	__m128 *data0,
	__m128 *data1,
	float *buff	// 16 * float
)
{
	// weights
	const __m128 w0 = { +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s, +dwt_cdf97_u2_s };
	const __m128 w1 = { -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s, -dwt_cdf97_p2_s };
	const __m128 w2 = { +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s, +dwt_cdf97_u1_s };
	const __m128 w3 = { -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s, -dwt_cdf97_p1_s };

	// variables
	__m128 l0, l1, l2, l3;
	__m128 c0, c1, c2, c3;
	__m128 r0, r1, r2, r3;
	__m128 x0, x1;
	__m128 y0, y1;

	// load "L"
	l0 = _mm_load_ps(&buff[0*(1*4)]);
	l1 = _mm_load_ps(&buff[1*(1*4)]);
	l2 = _mm_load_ps(&buff[2*(1*4)]);
	l3 = _mm_load_ps(&buff[3*(1*4)]);

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0 = l0;
	c0 = l1;
	c1 = l2;
	c2 = l3;
	c3 = x0;

	// operation
	r3 = x1;
	r2 = c3 + w3 * (l3 + r3);
	r1 = c2 + w2 * (l2 + r2);
	r0 = c1 + w1 * (l1 + r1);
	y1 = c0 + w0 * (l0 + r0);

	// update
	l0 = r0;
	l1 = r1;
	l2 = r2;
	l3 = r3;

	// outputs
	*data0 = y0;
	*data1 = y1;

	// store "L"
	_mm_store_ps(&buff[0*(1*4)], l0);
	_mm_store_ps(&buff[1*(1*4)], l1);
	_mm_store_ps(&buff[2*(1*4)], l2);
	_mm_store_ps(&buff[3*(1*4)], l3);
}
#endif

static
void cube_4x4x4(
	// coordinates
	int x,
	int y,
	int z,
	// image size
	int size_x,
	int size_y,
	int size_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *src_ptr,
	// strides (to address another pixels)
	int src_stride_x,
	int src_stride_y,
	int src_stride_z,
	// poiter to pixel at (0,0,0) in struct volume_t
	float *dst_ptr,
	// strides (to address another pixels)
	int dst_stride_x,
	int dst_stride_y,
	int dst_stride_z,
	// pointers to three buffers (aligned to 16 bytes = 4 floats)
	float * ALIGNED(16) buffer_x,
	float * ALIGNED(16) buffer_y,
	float * ALIGNED(16) buffer_z,
	// super sizes in order to access elements in buffers
	int super_x,
	int super_y,
	int super_z
)
{
	UNUSED(super_y);
	UNUSED(super_z);

	const int overlap_x_L = 7;
	const int overlap_y_L = 7;
	const int overlap_z_L = 7;

	const int shift = 4;

	const int step_x = 4;
	const int step_y = 4;
	const int step_z = 4;

#ifdef __SSE__
	// 4^3 pixels = 32x float = 8x __m128 (z=0..3, y=0..3)
	__m128 t[4][4]; // NOTE: [z][y][x]

	// LOAD
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real(x, xx, overlap_x_L, size_x);
				const int pos_y = virt2real(y, yy, overlap_y_L, size_y);
				const int pos_z = virt2real(z, zz, overlap_z_L, size_z);

				// NOTE: transposed x<->y
				t[zz][xx][yy] = *( (float *)( (char *)src_ptr + pos_x*src_stride_x + pos_y*src_stride_y + pos_z*src_stride_z ) );
			}
		}
	}

	// CALC
#if 1
	// NOTE: along x-axis
	// front 4x4 slice
	vert_2x4x1(
		&t[0][0],
		&t[0][1],
		buffer_x + 0*32
	);
	vert_2x4x1(
		&t[0][2],
		&t[0][3],
		buffer_x + 0*32
	);
	// second 4x4 slice
	vert_2x4x1(
		&t[1][0],
		&t[1][1],
		buffer_x + 1*32
	);
	vert_2x4x1(
		&t[1][2],
		&t[1][3],
		buffer_x + 1*32
	);
	// third 4x4 slice
	vert_2x4x1(
		&t[2][0],
		&t[2][1],
		buffer_x + 2*32
	);
	vert_2x4x1(
		&t[2][2],
		&t[2][3],
		buffer_x + 2*32
	);
	// back 4x4 slice
	vert_2x4x1(
		&t[3][0],
		&t[3][1],
		buffer_x + 3*32
	);
	vert_2x4x1(
		&t[3][2],
		&t[3][3],
		buffer_x + 3*32
	);
#endif
	// transpose
	_MM_TRANSPOSE4_PS(t[0][0], t[0][1], t[0][2], t[0][3]);
	_MM_TRANSPOSE4_PS(t[1][0], t[1][1], t[1][2], t[1][3]);
	_MM_TRANSPOSE4_PS(t[2][0], t[2][1], t[2][2], t[2][3]);
	_MM_TRANSPOSE4_PS(t[3][0], t[3][1], t[3][2], t[3][3]);
#if 1
	// NOTE: along y-axis
	// front 4x4 slice
	vert_2x4x1(
		&t[0][0],
		&t[0][1],
		buffer_y + 32*(x+0)
	);
	vert_2x4x1(
		&t[0][2],
		&t[0][3],
		buffer_y + 32*(x+0)
	);
	// second 4x4 slice
	vert_2x4x1(
		&t[1][0],
		&t[1][1],
		buffer_y + 32*(x+1)
	);
	vert_2x4x1(
		&t[1][2],
		&t[1][3],
		buffer_y + 32*(x+1)
	);
	// third 4x4 slice
	vert_2x4x1(
		&t[2][0],
		&t[2][1],
		buffer_y + 32*(x+2)
	);
	vert_2x4x1(
		&t[2][2],
		&t[2][3],
		buffer_y + 32*(x+2)
	);
	// back 4x4 slice
	vert_2x4x1(
		&t[3][0],
		&t[3][1],
		buffer_y + 32*(x+3)
	);
	vert_2x4x1(
		&t[3][2],
		&t[3][3],
		buffer_y + 32*(x+3)
	);
#endif
#if 1
	// NOTE: along z-axis
	// front 4x4x2
	vert_2x4x1(
		&t[0][0],
		&t[1][0],
		buffer_z + 32*(y*super_x+x+0)
	);
	vert_2x4x1(
		&t[0][1],
		&t[1][1],
		buffer_z + 32*(y*super_x+x+1)
	);
	vert_2x4x1(
		&t[0][2],
		&t[1][2],
		buffer_z + 32*(y*super_x+x+2)
	);
	vert_2x4x1(
		&t[0][3],
		&t[1][3],
		buffer_z + 32*(y*super_x+x+3)
	);
	// back 4x4x2
	vert_2x4x1(
		&t[2][0],
		&t[3][0],
		buffer_z + 32*(y*super_x+x+0)
	);
	vert_2x4x1(
		&t[2][1],
		&t[3][1],
		buffer_z + 32*(y*super_x+x+1)
	);
	vert_2x4x1(
		&t[2][2],
		&t[3][2],
		buffer_z + 32*(y*super_x+x+2)
	);
	vert_2x4x1(
		&t[2][3],
		&t[3][3],
		buffer_z + 32*(y*super_x+x+3)
	);
#endif
#if 1
	// NOTE: scaling
	const float z1 = dwt_cdf97_s1_s;
	const float z3 = dwt_cdf97_s1_s*dwt_cdf97_s1_s*dwt_cdf97_s1_s;
	const float r1 = 1.f/z1;
	const float r3 = 1.f/z3;

	t[0][0] *= (const __m128){ r3, r1, r3, r1 };
	t[0][1] *= (const __m128){ r1, z1, r1, z1 };
	t[0][2] *= (const __m128){ r3, r1, r3, r1 };
	t[0][3] *= (const __m128){ r1, z1, r1, z1 };

	t[1][0] *= (const __m128){ r1, z1, r1, z1 };
	t[1][1] *= (const __m128){ z1, z3, z1, z3 };
	t[1][2] *= (const __m128){ r1, z1, r1, z1 };
	t[1][3] *= (const __m128){ z1, z3, z1, z3 };

	t[2][0] *= (const __m128){ r3, r1, r3, r1 };
	t[2][1] *= (const __m128){ r1, z1, r1, z1 };
	t[2][2] *= (const __m128){ r3, r1, r3, r1 };
	t[2][3] *= (const __m128){ r1, z1, r1, z1 };

	t[3][0] *= (const __m128){ r1, z1, r1, z1 };
	t[3][1] *= (const __m128){ z1, z3, z1, z3 };
	t[3][2] *= (const __m128){ r1, z1, r1, z1 };
	t[3][3] *= (const __m128){ z1, z3, z1, z3 };
#endif

	// STORE
	for(int xx = 0; xx < step_x; xx++)
	{
		for(int yy = 0; yy < step_y; yy++)
		{
			for(int zz = 0; zz < step_z; zz++)
			{
				// virtual => real coordinates
				const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, size_x);
				const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, size_y);
				const int pos_z = virt2real_error(z-shift, zz, overlap_z_L, size_z);

				if( pos_x < 0 || pos_y < 0 || pos_z < 0 )
					continue;

				*( (float *)( (char *)dst_ptr + pos_x*dst_stride_x + pos_y*dst_stride_y + pos_z*dst_stride_z ) ) = t[zz][yy][xx];
			}
		}
	}
#endif
}

void cdf97_3f_op_HORIZ_vert4x4x4_s(struct volume_t *volume_src, struct volume_t *volume_dst)
{
	assert( volume_dst );
	assert( volume_src );
	assert( volume_dst->size_x == volume_src->size_x );
	assert( volume_dst->size_y == volume_src->size_y );
	assert( volume_dst->size_z == volume_src->size_z );

	// width of buffers
	const int buff_elem_size = 32;

	// step_x, step_y, step_z
	const int step_x = 4;
	const int step_y = 4;
	const int step_z = 4;

	// super sizes
	const int super_x = /*overlap_x_L*/7 + volume_src->size_x + /*overlap_x_R*/5;
	const int super_y = /*overlap_y_L*/7 + volume_src->size_y + /*overlap_y_R*/5;
	const int super_z = /*overlap_z_L*/7 + volume_src->size_z + /*overlap_z_R*/5;

	assert( 0 == super_x % step_x );
	assert( 0 == super_y % step_y );
	assert( 0 == super_z % step_z );

	// alloc buffers
	const int buffer_x_elems = 2*2 * buff_elem_size;
	const int buffer_y_elems = super_x*2 * buff_elem_size;
	const int buffer_z_elems = super_x*super_y * buff_elem_size;

	float * ALIGNED(16) buffer_x = dwt_util_alloc_aligned_ex_reliably(buffer_x_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_y = dwt_util_alloc_aligned_ex_reliably(buffer_y_elems, sizeof(float), 16);
	float * ALIGNED(16) buffer_z = dwt_util_alloc_aligned_ex_reliably(buffer_z_elems, sizeof(float), 16);

#if 0
	// zero buffers
	dwt_util_zero_vec_s(buffer_x, buffer_x_elems);
	dwt_util_zero_vec_s(buffer_y, buffer_y_elems);
	dwt_util_zero_vec_s(buffer_z, buffer_z_elems);
#endif

	// transform per 4x4x2 cubes

	// for each z
	for(int z = 0; z < super_z; z += step_z)
		// for each y
		for(int y = 0; y < super_y; y += step_y)
			// for each x (CPU cache friendly)
			for(int x = 0; x < super_x; x += step_x)
			{
				// call cube core
				cube_4x4x4(
					// coordinates
					x,
					y,
					z,
					// image size
					volume_src->size_x,
					volume_src->size_y,
					volume_src->size_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_src->data,
					// strides (to address another pixels)
					volume_src->stride_x,
					volume_src->stride_y,
					volume_src->stride_z,
					// poiter to pixel at (0,0,0) in struct volume_t
					volume_dst->data,
					// strides (to address another pixels)
					volume_dst->stride_x,
					volume_dst->stride_y,
					volume_dst->stride_z,
					// pointers to three buffers (aligned to 16 bytes = 4 floats)
					buffer_x,
					buffer_y,
					buffer_z,
					// super sizes in order to access elements in buffers
					super_x,
					super_y,
					super_z
				);
			}

	free(buffer_x);
	free(buffer_y);
	free(buffer_z);
}

typedef void (*volume_func_t)(struct volume_t *, struct volume_t *);

void cdf97_3f_op_wrapper_s(struct volume_t *volume_src, struct volume_t *volume_dst, enum volume_approach approach)
{
	assert( approach >= 0 && approach < VOL_LAST );

	volume_func_t volume_func[VOL_LAST] = {
		[VOL_SEP_HORIZONTAL] = cdf97_3f_op_sep_horizontal_s,
		[VOL_SEP_VERTICAL] = cdf97_3f_op_sep_vertical_s,
		[VOL_SLICES_VERT4X4] = cdf97_3f_op_slices_vert4x4_s,
		[VOL_BASELINE_VERT2X2X2] = cdf97_3f_op_baseline_vert2x2x2_s,
		[VOL_HORIZ_VERT2X2X2] = cdf97_3f_op_HORIZ_vert2x2x2_s,
		[VOL_BASELINE_VERT4X4X2] = cdf97_3f_op_cube_vert4x4x2_s,
		[VOL_HORIZ_VERT4X4X2] = cdf97_3f_op_HORIZ_vert4x4x2_s,
		[VOL_HORIZ_VERT4X4X4] = cdf97_3f_op_HORIZ_vert4x4x4_s,
		[VOL_BASELINE_DIAG2X2X2] = cdf97_3f_op_baseline_diag2x2x2_s,
		[VOL_HORIZ_DIAG2X2X2] = cdf97_3f_op_HORIZ_diag2x2x2_s,
		[VOL_SEP_HORIZONTAL_X] = cdf97_3f_op_sep_horizontal_x_s,
		[VOL_SEP_HORIZONTAL_Y] = cdf97_3f_op_sep_horizontal_y_s,
		[VOL_SEP_HORIZONTAL_Z] = cdf97_3f_op_sep_horizontal_z_s,
	};

	volume_func[approach](volume_src, volume_dst);
}

int volume_perftest_fwd97op_s(
	int size, // size_x, size_y_ size_z
	int opt_stride,
	enum volume_approach approach,
	int N, // tests, select minimum
	double *secs, // seconds per pixel
	long unsigned *faults // page faults
)
{
	const int pixels = size*size*size;
	const int clock_type = dwt_util_clock_autoselect();

	int return_code = 0;
	*secs = +INFINITY;
	*faults = 0 /*ULONG_MAX*/;

	// allocate
	struct volume_t *data1 = volume_alloc_realiably_locked(sizeof(float), size, size, size, opt_stride);
	struct volume_t *data2 = volume_alloc_realiably_locked(sizeof(float), size, size, size, opt_stride);

	for(int n = 0; n < N; n++)
	{
		// fill with test pattern
		volume_fill_s(data1);

		// invalidate CPU cache
		volume_invalidate_cache(data1);
		volume_invalidate_cache(data2);

		// start measurement
		long unsigned page_faults_start = dwt_util_get_page_fault();
		dwt_clock_t start = dwt_util_get_clock(clock_type);

		// forward transform
		cdf97_3f_op_wrapper_s(data1, data2, approach);

		// stop measurement
		dwt_clock_t stop = dwt_util_get_clock(clock_type);
		long unsigned page_faults_stop = dwt_util_get_page_fault();

		// elapsed time
		const double elapsed_time_in_seconds = (stop - start)/(double)dwt_util_get_frequency(clock_type);
		const double secs_per_pel = elapsed_time_in_seconds/pixels;

		// new page faults
		long unsigned page_faults = page_faults_stop - page_faults_start;
#if 0
		dwt_util_log(LOG_DBG, "page faults: %lu - %lu = %lu\n", page_faults_stop, page_faults_start, page_faults);
#endif

		// select minimum time
		if( secs_per_pel < *secs )
			*secs = secs_per_pel;

		// select minimum faults
		if( page_faults > /*<*/ *faults )
			*faults = page_faults;

		// inverse transform
		cdf97_3i_ip_sep_horizontal_s(data2);

		// compare volumes
		const int error = volume_compare_s(data1, data2);
		return_code += error;
	}

	// free data
	volume_free(data1);
	volume_free(data2);

	return return_code;
}

static
int size_grow(int size, int align)
{
	assert( is_pow2(align) );

	extern float g_growth_factor_s;

	size *= g_growth_factor_s;
	size += 1;
	size +=  (align-1);
	size &= ~(align-1);

	return size;
}

int volume_measure_fwd97op_s(int size_min, int size_max, int size_step, int N, int opt_stride, enum volume_approach approach)
{
	char path[4096];

	sprintf(path, "data/perftest/time-stride=%i-approach=%i.txt", opt_stride, approach);

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "file=%s\n", path);
#endif

	FILE *file_time = fopen(path, "w");

	if( !file_time )
	{
		dwt_util_error("unable to open file: %s\n", path);
	}

	sprintf(path, "data/perftest/faults-stride=%i-approach=%i.txt", opt_stride, approach);
	FILE *file_faults = fopen(path, "w");
	if( !file_faults )
	{
		dwt_util_error("unable to open file: %s\n", path);
	}

	fprintf(file_time, "# voxels secs/pel\n");
	fprintf(file_faults, "# voxels page_faults\n");

	int total_errors = 0;

	for(int size = size_min; size < size_max; size = size_grow(size, size_step))
	{
		double secs;
		long unsigned faults;

		int errors = volume_perftest_fwd97op_s(
			size,
			opt_stride,
			approach,
			N,
			&secs,
			&faults
		);

		const int voxels = size*size*size;

		dwt_util_log(LOG_INFO,
			"perftest: size=%4i opt_stride=%i approach=%2i (N=%2i): time=%f [nsecs/pel]; errors=%i; faults=%lu\n",
			size, opt_stride, approach, N, secs*1e9, errors, faults
		);

		fprintf(file_time, "%i\t%.20f\n", voxels, secs);
		fprintf(file_faults, "%i\t%lu\n", voxels, faults);

		total_errors += errors;
	}

	fclose(file_time);
	fclose(file_faults);

	return total_errors;
}
