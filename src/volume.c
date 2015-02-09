/**
 * @brief 3-D data routines.
 */

#include "volume.h"
#include "system.h" // dwt_util_reliably_alloc1
#include "libdwt.h" // dwt_util_get_stride
#include <assert.h> // assert

struct volume_t *volume_alloc_realiably(size_t pix_size, int size_x, int size_y, int size_z, int opt_stride)
{
	struct volume_t *volume = dwt_util_reliably_alloc1(sizeof(struct volume_t));

	volume->size_x = size_x;
	volume->size_y = size_y;
	volume->size_z = size_z;

	volume->stride_x = pix_size;
	volume->stride_y = dwt_util_get_stride(volume->stride_x * volume->size_x, opt_stride);
	volume->stride_z = dwt_util_get_stride(volume->stride_y * volume->size_y, opt_stride);

	const size_t total_size = volume->stride_z * volume->size_z;

	volume->data = dwt_util_reliably_alloc1(total_size);

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "allocated %zu MiB\n", total_size>>20);
#endif

	return volume;
}

void volume_free(struct volume_t *volume)
{
	assert( volume );

	dwt_util_free(volume->data);
	dwt_util_free(volume);
}

void volume_fill_s(struct volume_t *volume)
{
	// for each slice
	for(int z = 0; z < volume->size_z; z++)
	{
		// get slice
		void *slice = volume_get_slice(volume, z);

		int type = 0;

		int rand = z&11;
		if(rand > 11/2)
			rand = 11 - rand;

		// fill
		dwt_util_test_image_fill2_s(
			slice,
			volume->stride_y, ///< difference between rows (in bytes)
			volume->stride_x, ///< difference between columns (in bytes)
			volume->size_x,
			volume->size_y,
			rand,
			type
		);
	}
}

int volume_copy_s(struct volume_t *volume_dst, struct volume_t *volume_src)
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

		// copy
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
	}

	return 0;
}

int volume_compare_s(struct volume_t *volume_l, struct volume_t *volume_r)
{
	assert( volume_l );
	assert( volume_r );
	assert( volume_l->size_x == volume_r->size_x );
	assert( volume_l->size_y == volume_r->size_y );
	assert( volume_l->size_z == volume_r->size_z );

	// for each slice
	for(int z = 0; z < volume_r->size_z; z++)
	{
		// get slices
		void *slice_l = volume_get_slice(volume_l, z);
		void *slice_r = volume_get_slice(volume_r, z);

		// compare slices
		int code = dwt_util_compare2_s(
			slice_l,
			slice_r,
			volume_l->stride_y,
			volume_l->stride_x,
			volume_r->stride_y,
			volume_r->stride_x,
			volume_r->size_x,
			volume_r->size_y
		);

		if(code)
			return code;
	}

	return 0;
}

void volume_save_to_pgm_s(struct volume_t *volume, const char *path)
{
	assert( volume );

	// for each slice
	for(int z = 0; z < volume->size_z; z++)
	{
		// get slice
		void *slice = volume_get_slice(volume, z);

		char file_name[4096];

		// sprintf(file_name, path, slice_no)
		sprintf(file_name, path, z);

#ifdef DEBUG
// 		dwt_util_log(LOG_DBG, "saving into %s...\n", file_name);
#endif

		// save to (file_name)
		dwt_util_save_to_pgm_s(
			file_name,
			1.f,
			slice,
			volume->stride_y, ///< difference between rows (in bytes)
			volume->stride_x, ///< difference between columns (in bytes)
			volume->size_x,
			volume->size_y
		);
	}
}

void volume_save_log_to_pgm_s(struct volume_t *volume, const char *path)
{
	// for each slice
	for(int z = 0; z < volume->size_z; z++)
	{
		// get slice
		void *slice = volume_get_slice(volume, z);

		char file_name[4096];

		// sprintf(file_name, path, slice_no)
		sprintf(file_name, path, z);

#ifdef DEBUG
// 		dwt_util_log(LOG_DBG, "saving into %s...\n", file_name);
#endif

		// save to (file_name)
		dwt_util_save_log_to_pgm_s(
			file_name,
			slice,
			volume->stride_y, ///< difference between rows (in bytes)
			volume->stride_x, ///< difference between columns (in bytes)
			volume->size_x,
			volume->size_y
		);
	}
}

struct volume_t *volume_alloc_realiably_locked(size_t pix_size, int size_x, int size_y, int size_z, int opt_stride)
{
	struct volume_t *volume = dwt_util_reliably_alloc1(sizeof(struct volume_t));

	volume->size_x = size_x;
	volume->size_y = size_y;
	volume->size_z = size_z;

	volume->stride_x = pix_size;
	volume->stride_y = dwt_util_get_stride(volume->stride_x * volume->size_x, opt_stride);
	volume->stride_z = dwt_util_get_stride(volume->stride_y * volume->size_y, opt_stride);

	const size_t total_size = volume->stride_z * volume->size_z;

#if 0
	volume->data = dwt_util_alloc_locked(total_size);
#else
	volume->data = dwt_util_reliably_alloc_locked(total_size);
#endif

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "allocated %zu MiB\n", total_size>>20);
#endif

	return volume;
}

void volume_invalidate_cache(struct volume_t *volume)
{
	const size_t total_size = volume->stride_z * volume->size_z;
	dwt_util_flush_cache(volume->data, total_size);
}
