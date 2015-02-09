/**
 * @brief 3-D data routines.
 */

#ifndef VOLUME_H
#define VOLUME_H

#include "inline.h" // UNUSED_FUNC
#include <stddef.h> // size_t

/**
 * Volumetric 3-D data.
 */
struct volume_t {
	int size_x; // columns
	int size_y; // rows
	int size_z; // slices

	size_t stride_x; // sizeof(pixel)
	size_t stride_y; // sizeof(row)
	size_t stride_z; // sizeof(slice)

	void *data; // pixel(x,y,z) = base + pos_x*stride_x + pos_y*stride_y + pos_z*stride_z;
};

/**
 * @brief Allocate 3-D data.
 */
struct volume_t *volume_alloc_realiably(size_t pix_size, int size_x, int size_y, int size_z, int opt_stride);

/**
 * @brief Free 3-D data.
 */
void volume_free(struct volume_t *volume);

UNUSED_FUNC
static
void *volume_get_slice(struct volume_t *volume, int pos_z)
{
	assert( volume && volume->data );

	return (char *)volume->data + pos_z * volume->stride_z;
}

/**
 * @brief Fill 3-D data with test pattern.
 */
void volume_fill_s(struct volume_t *volume);

/**
 * @brief Copy volumetric data.
 *
 * @p volume_src have to be already allocated
 */
int volume_copy_s(struct volume_t *volume_dst, struct volume_t *volume_src);

/**
 * @brief Compare volumetric data.
 */
int volume_compare_s(struct volume_t *volume_l, struct volume_t *volume_r);

/**
 * @brief Save 3-D data as sequence of PGM images.
 */
void volume_save_to_pgm_s(struct volume_t *volume, const char *path);

/**
 * @brief Save 3-D data in a logarithmic scale as sequence of PGM images.
 */
void volume_save_log_to_pgm_s(struct volume_t *volume, const char *path);

UNUSED_FUNC
static
void *volume_get_pix(struct volume_t *volume, int x, int y, int z)
{
	assert( volume && volume->data );

	return (char *)volume->data + x*volume->stride_x + y*volume->stride_y + z*volume->stride_z;
}

struct volume_t *volume_alloc_realiably_locked(
	size_t pix_size,
	int size_x,
	int size_y,
	int size_z,
	int opt_stride
);

void volume_invalidate_cache(struct volume_t *volume);

#endif
