#ifndef SYSTEM_H
#define SYSTEM_H

// size_t
#include <stddef.h>

/**
 * @brief Maximum number of bytes in a pathname, including the terminating null character.
 * @warning experimental
 */
size_t dwt_util_get_path_max();

/**
 * @brief Allocate dynamic memory.
 * @warning experimental
 */
void *dwt_util_alloc1(
	size_t size
);

/**
 * @brief Allocate dynamic memory for an array of @e elems elements of size bytes each.
 * @warning experimental
 */
void *dwt_util_alloc2(
	unsigned long elems,
	size_t size
);

/**
 * @brief Reliably allocate dynamic memory.
 * @warning experimental
 */
void *dwt_util_reliably_alloc1(
	size_t size
);

/**
 * @brief Reliably allocate dynamic memory for an array of @e elems elements of size bytes each.
 * @warning experimental
 */
void *dwt_util_reliably_alloc2(
	unsigned long elems,
	size_t size
);

/**
 * @brief Free dynamic memory.
 * @warning experimental
 */
void dwt_util_free(
	void *ptr
);

/**
 * @brief Copy memory area.
 *
 * This function copies @p n floats from memory area @p src to memory area
 * @p dst. Memory areas can be sparse. The strides (in bytes) are determined by
 * @p stride_dst and @p stride_src arguments. The memory areas must not overlap.
 *
 * @returns The function returns a pointer to @p dst.
 */
void *dwt_util_memcpy_stride_s(
	void *restrict dst,
	size_t stride_dst,
	const void *restrict src,
	size_t stride_src,
	int elements		///< Number of floats to be copied, not number of bytes.
);

/**
 * @brief Copy memory area.
 *
 * This function copies @p n ints from memory area @p src to memory area
 * @p dst. Memory areas can be sparse. The strides (in bytes) are determined by
 * @p stride_dst and @p stride_src arguments. The memory areas must not overlap.
 *
 * @returns The function returns a pointer to @p dst.
 */
void *dwt_util_memcpy_stride_i(
	void *restrict dst,
	size_t stride_dst,
	const void *restrict src,
	size_t stride_src,
	int elements		///< Number of ints to be copied, not number of bytes.
);

/**
 * @brief Copy memory area.
 *
 * This function copies @p n doubles from memory area @p src to memory area
 * @p dst. Memory areas can be sparse. The strides (in bytes) are determined by
 * @p stride_dst and @p stride_src arguments. The memory areas must not overlap.
 *
 * @returns The function returns a pointer to @p dst.
 */
void *dwt_util_memcpy_stride_d(
	void *restrict dst,
	size_t stride_dst,
	const void *restrict src,
	size_t stride_src,
	int elements		///< Number of doubles to be copied, not number of bytes.
);

#include "inline.h" // UNUSED_FUNC
#include <stddef.h> // size_t
#include <stdint.h> // intptr_t

UNUSED_FUNC
static
int is_aligned(
	void *ptr,
	size_t alignment
)
{
	assert( is_pow2(alignment) );

	return ( (intptr_t)ptr & (intptr_t)(alignment-1) ) ? 0 : 1;
}

void *dwt_util_alloc_aligned_ex(
	unsigned elements,
	size_t elem_size,
	size_t align
);

void *dwt_util_alloc_aligned_ex_reliably(
	unsigned elements,
	size_t elem_size,
	size_t align
);

void *dwt_util_alloc_locked(
	size_t size
);

void *dwt_util_reliably_alloc_locked(
	size_t size
);

void dwt_util_set_realtime_scheduler();

void dwt_util_set_affinity();

/**
 * @brief Gets major+minor page faults from /proc/self/stat file.
 */
long unsigned dwt_util_get_page_fault();

void dwt_util_set_cpufreq();

void dwt_util_env_single_threading();

#endif
