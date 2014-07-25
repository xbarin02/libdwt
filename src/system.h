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

#endif
