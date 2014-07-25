#include "system.h"

// dwt_util_error
#include "libdwt.h"

// malloc, free
#include <stdlib.h> 

// size_t
#include <stddef.h>

// PATH_MAX
#include <limits.h>
#ifndef microblaze
#include <linux/limits.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static
size_t get_path_max()
{
	return PATH_MAX;
}

size_t dwt_util_get_path_max()
{
	return get_path_max();
}

void *dwt_util_alloc1(
	size_t size
)
{
	void *ptr = malloc(size);

	return ptr;
}

void *dwt_util_reliably_alloc1(
	size_t size
)
{
	void *ptr = malloc(size);

	if( !ptr )
	{
		dwt_util_error("Unable to allocate memory.\n");
	}

	return ptr;
}

void *dwt_util_alloc2(
	unsigned long elems,
	size_t size
)
{
	return dwt_util_alloc1(elems*size);
}

void *dwt_util_reliably_alloc2(
	unsigned long elems,
	size_t size
)
{
	return dwt_util_reliably_alloc1(elems*size);
}

void dwt_util_free(
	void *ptr
)
{
	free(ptr);
}
