#include "spectra.h"
#include "libdwt.h"
// assert
#include <assert.h>
// open
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
// mmap
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>
// lseek
#include <sys/types.h>
#include <unistd.h>
// memcpy
#include <string.h>
// PATH_MAX
#include <limits.h>
#ifndef microblaze
#include <linux/limits.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
// free
#include <stdlib.h>

void *spectra_load(
	const char *path,
	int *stride_x,
	int *stride_y,
	int *size_x,
	int *size_y
)
{
	assert( path );
	assert( stride_x && stride_y && size_x && size_y );

	char path_cache[PATH_MAX];
	char path_meta[PATH_MAX];

	sprintf(path_cache, "%s.%s", path, "cache");
	sprintf(path_meta, "%s.%s", path, "meta");

	dwt_util_log(LOG_DBG, "spectra_load: path='%s' cache='%s' meta='%s'\n", path, path_cache, path_meta);

	size_t length = 0;
	void *ptr = NULL;

	int fd = open(path_cache, O_RDONLY);
	if( fd < 0 )
	{
		dwt_util_log(LOG_WARN, "Creating new cache...\n");

		dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", path);
		dwt_util_load_from_mat_s(path, &ptr, size_x, size_y, stride_x, stride_y);

		if( !ptr )
		{
			dwt_util_log(LOG_ERR, "Unable to load spectra.\n");
		}
		else
		{
			length = dwt_util_image_size(
				*stride_x,
				*stride_y,
				*size_x,
				*size_y
			);

#if 0
			dwt_util_log(LOG_INFO, "Shifting base-line by median...\n");
			dwt_util_shift21_med_s(
				ptr,
				*size_x,
				*size_y,
				*stride_x,
				*stride_y
			);
#endif

			fd = open(path_cache, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
			if( fd < 0 )
			{
				dwt_util_log(LOG_ERR, "open() fails\n");
				return NULL;
			}

			if( -1 == lseek(fd, length-1, SEEK_SET) )
			{
				dwt_util_log(LOG_ERR, "lseek() fails\n");
				close(fd);
				return NULL;
			}

			// An empty string is actually a single '\0' character
			if( 1 != write(fd, "", 1) )
			{
				dwt_util_log(LOG_ERR, "write() fails\n");
				close(fd);
				return NULL;
			}

			void *tmp_ptr = mmap(0, length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
			if( MAP_FAILED == tmp_ptr )
			{
				dwt_util_log(LOG_ERR, "mmap() fails\n");
				close(fd);
				return NULL;
			}
			memcpy(tmp_ptr, ptr, length);
			free(ptr);
			ptr = tmp_ptr;

			int res = 0;

			int fd_meta = open(path_meta, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
			if( fd_meta < 0 )
			{
				dwt_util_log(LOG_ERR, "open() fails\n");
				close(fd);
				munmap(ptr, length);
				return NULL;
			}
			res += write(fd_meta, &length, sizeof(length));
			res += write(fd_meta, stride_x, sizeof(*stride_x));
			res += write(fd_meta, stride_y, sizeof(*stride_y));
			res += write(fd_meta, size_x, sizeof(*size_x));
			res += write(fd_meta, size_y, sizeof(*size_y));
			close(fd_meta);
		}
	}
	else
	{
		dwt_util_log(LOG_WARN, "Loading spectra from the cache...\n");

		int res = 0;

		int fd_meta = open(path_meta, O_RDONLY);
		if( fd_meta < 0 )
		{
			dwt_util_log(LOG_ERR, "open() fails\n");
			close(fd);
			return NULL;
		}
		res += read(fd_meta, &length, sizeof(length));
		res += read(fd_meta, stride_x, sizeof(*stride_x));
		res += read(fd_meta, stride_y, sizeof(*stride_y));
		res += read(fd_meta, size_x, sizeof(*size_x));
		res += read(fd_meta, size_y, sizeof(*size_y));
		close(fd_meta);

		ptr = mmap(0, length, PROT_READ, MAP_PRIVATE, fd, 0);
		if( MAP_FAILED == ptr )
		{
			dwt_util_log(LOG_ERR, "mmap() fails\n");
			return NULL;
		}
	}

	close(fd);

	return ptr;
}

void spectra_unload(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	int length = dwt_util_image_size(
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	if( ptr )
		munmap(ptr, length);
}

float *dwt_util_addr_row_s(
	void *ptr,
	int y,
	int stride_x
)
{
	return dwt_util_addr_coeff_s(
		ptr,
		y,
		0,
		stride_x,
		0
	);
}

int *dwt_util_addr_row_i(
	void *ptr,
	int y,
	int stride_x
)
{
	return dwt_util_addr_coeff_i(
		ptr,
		y,
		0,
		stride_x,
		0
	);
}

int dwt_util_save1_to_mat_s(
	const char *path,
	const void *ptr,
	int size_x,
	int stride_y
)
{
	return dwt_util_save_to_mat_s(
		path,
		ptr,
		size_x,
		1,
		0,
		stride_y
	);
}
