/**
 * @brief image_t infrastructure
 */

#include "image.h"

#include "libdwt.h"
#include "inline.h"
#include "system.h"

void image_init(image_t *image, void *ptr, int size_x, int size_y, int stride_x, int stride_y)
{
	image->ptr = ptr;
	image->size_x = size_x;
	image->size_y = size_y;
	image->stride_x = stride_x;
	image->stride_y = stride_y;
}

int image_alloc(image_t *image)
{
	dwt_util_alloc_image(
		&image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y
	);

	return image->ptr != NULL;
}

void image_free(
	image_t *image
)
{
	dwt_util_free_image(&image->ptr);
}

void image_destroy(
	image_t *image
)
{
	image_free(image); // free image->ptr

	dwt_util_free(image); // free image
}

image_t *image_create_s(int size_x, int size_y)
{
	image_t *image = dwt_util_alloc(1, sizeof(image_t));

	image_init(image, NULL, size_x, size_y, sizeof(float)*size_x, sizeof(float));

	image_alloc(image);

	return image;
}

image_t *image_create_opt_s(int size_x, int size_y)
{
	image_t *image = dwt_util_alloc(1, sizeof(image_t));

	image_init(image, NULL, size_x, size_y, dwt_util_get_opt_stride(sizeof(float)*size_x), sizeof(float));

	image_alloc(image);

	return image;
}

int image_load_from_mat_s(image_t *image, const char *path)
{
	dwt_util_load_from_mat_s(
		path,
		&image->ptr,
		&image->size_x,
		&image->size_y,
		&image->stride_x,
		&image->stride_y
	);

	return NULL != image->ptr;
}

image_t *image_create_from_mat_s(
	const char *path
)
{
	image_t *image = dwt_util_alloc(1, sizeof(image_t));

	if( !image_load_from_mat_s(image, path) )
	{
		dwt_util_free(image);

		return NULL;
	}

	return image;
}

int image_save_to_mat_s(image_t *image, const char *path)
{
	return dwt_util_save_to_mat_s(
		path,
		image->ptr,
		image->size_x,
		image->size_y,
		image->stride_x,
		image->stride_y
	);
}

int image_save_to_svm_s(image_t *classes, image_t *features, const char *path)
{
	return dwt_util_save_to_svm_s(
		path,
		features->ptr,
		features->size_x,
		features->size_y,
		features->stride_x,
		features->stride_y,
		classes->ptr,
		classes->size_x,
		classes->size_y,
		classes->stride_x,
		classes->stride_y
	);
}

void image_save_to_pgm_s(image_t *image, const char *path)
{
	image_t s;

	image_init(&s, NULL, image->size_x, image->size_y, image->stride_x, image->stride_y);

	float minv, maxv;

	image_alloc(&s);

	dwt_util_find_min_max_s(
		image->ptr,
		image->size_x,
		image->size_y,
		image->stride_x,
		image->stride_y,
		&minv,
		&maxv
	);

	dwt_util_copy_s(
		image->ptr,
		s.ptr,
		s.stride_x,
		s.stride_y,
		s.size_x,
		s.size_y
	);

	dwt_util_shift_s(
		s.ptr,
		s.size_x,
		s.size_y,
		s.stride_x,
		s.stride_y,
		-minv
	);

	dwt_util_save_to_pgm_s(
		path,
		(-minv + maxv),
		s.ptr,
		s.stride_x,
		s.stride_y,
		s.size_x,
		s.size_y
	);

	image_free(&s);
}

void image_save_to_pgm2_s(image_t *image, const char *path)
{
	dwt_util_save_to_pgm_s(
		path,
		1.f,
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y
	);
}

void image_save_log_to_pgm_s(image_t *image, const char *path)
{
	dwt_util_save_log_to_pgm_s(
		path,
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y
	);
}

void image_save_to_pgm_format_s(
	image_t *image,
	const char *format,
	...
)
{
	const size_t path_max = dwt_util_get_path_max();

	char *path = dwt_util_reliably_alloc1(path_max);

	va_list ap;

	va_start(ap, format);
	int n = vsnprintf(path, path_max, format, ap);
	va_end(ap);

	if( n < 0 )
	{
		dwt_util_error("vsnprintf returned negative value!\n");
	}

	image_save_to_pgm_s(image, path);

	dwt_util_free(path);
}

void image_save_to_mat_format_s(
	image_t *image,
	const char *format,
	...
)
{
	const size_t path_max = dwt_util_get_path_max();

	char *path = dwt_util_reliably_alloc1(path_max);

	va_list ap;

	va_start(ap, format);
	int n = vsnprintf(path, path_max, format, ap);
	va_end(ap);

	if( n < 0 )
	{
		dwt_util_error("vsnprintf returned negative value!\n");
	}

	image_save_to_mat_s(image, path);

	dwt_util_free(path);
}

void image_zero(image_t *image)
{
	dwt_util_test_image_zero_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y
	);
}

void image_subband(image_t *src, image_t *dst, int j, enum dwt_subbands band)
{
	dst->stride_x = src->stride_x;
	dst->stride_y = src->stride_y;

	dwt_util_subband_s(
		src->ptr,
		src->stride_x,
		src->stride_y,
		src->size_x,
		src->size_y,
		src->size_x,
		src->size_y,
		j,
		band,
		&dst->ptr,
		&dst->size_x,
		&dst->size_y);
}

float *image_coeff_s(image_t *image, int y, int x)
{
	return dwt_util_addr_coeff_s(image->ptr, y, x, image->stride_x, image->stride_y);
}

int *image_coeff_i(image_t *image, int y, int x)
{
	return dwt_util_addr_coeff_i(image->ptr, y, x, image->stride_x, image->stride_y);
}

float *image_row_s(image_t *image, int y)
{
	return image_coeff_s(image, y, 0);
}

int *image_row_i(image_t *image, int y)
{
	return image_coeff_i(image, y, 0);
}

void image_copy(image_t *src, image_t *dst)
{
	for(int y = 0; y < src->size_y; y++)
		for(int x = 0; x < src->size_x; x++)
		{
			*image_coeff_s(dst, y, x) =
				*image_coeff_s(src, y, x);
		}
}

void image_diff(image_t *dst, image_t *src0, image_t *src1)
{
	for(int y = 0; y < dst->size_y; y++)
		for(int x = 0; x < dst->size_x; x++)
		{
			*image_coeff_s(dst, y, x) =
				*image_coeff_s(src1, y, x) - *image_coeff_s(src0, y, x);
		}
}

void image_idwt_interp53_s(image_t *image, int levels)
{
	dwt_interp53_2i_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		levels,
		0,	///< should be row or column of size one pixel decomposed? zero value if not
		0	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	);
}

void image_idwt_cdf53_s(image_t *image, int levels)
{
	dwt_cdf53_2i_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		levels,
		0,
		0
	);
}

void image_idwt_cdf97_s(image_t *image, int levels)
{
	dwt_cdf97_2i_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		levels,
		0,
		0
	);
}

typedef
void (*image_idwt_func_t)(image_t *, int);

void image_idwt_s(
	image_t *image,
	int levels,
	enum wavelet_t wavelet
)
{
	image_idwt_func_t image_idwt_func[WAVELET_LAST] = {
		[WAVELET_CDF97] = image_idwt_cdf97_s,
		[WAVELET_CDF53] = image_idwt_cdf53_s,
		[WAVELET_INTERP53] = image_idwt_interp53_s,
	};

	image_idwt_func[wavelet](image, levels);
}

typedef
int (*image_fdwt_func_t)(image_t *, int);

int image_fdwt_s(
	image_t *image,
	int levels,
	enum wavelet_t wavelet
)
{
	image_fdwt_func_t image_fdwt_func[WAVELET_LAST] = {
		[WAVELET_CDF97] = image_fdwt_cdf97_s,
		[WAVELET_CDF53] = image_fdwt_cdf53_s,
		[WAVELET_INTERP53] = image_fdwt_interp53_s,
	};

	return image_fdwt_func[wavelet](image, levels);
}

int image_fdwt_interp53_s(image_t *image, int levels)
{
	dwt_interp53_2f_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		&levels,
		0,	///< should be row or column of size one pixel decomposed? zero value if not
		0	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	);

	return levels;
}

int image_fdwt_cdf53_s(image_t *image, int levels)
{
	dwt_cdf53_2f_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		&levels,
		0,	///< should be row or column of size one pixel decomposed? zero value if not
		0	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	);

	return levels;
}

int image_fdwt_cdf97_s(image_t *image, int levels)
{
	dwt_cdf97_2f_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		&levels,
		0,	///< should be row or column of size one pixel decomposed? zero value if not
		0	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	);

	return levels;
}

image_t *image_extend_s(
	const image_t *src,
	int pixels
)
{
	image_t *dst = image_create_s(2*pixels+src->size_x, 2*pixels+src->size_y);

	dwt_util_copy3_s(
		src->ptr,
		addr2_s(dst->ptr, pixels, pixels, dst->stride_x, dst->stride_y),
		src->stride_x,
		src->stride_y,
		dst->stride_x,
		dst->stride_y,
		src->size_x,
		src->size_y
	);

	// T
	for(int y = 0; y < pixels; y++)
		for(int x = 0; x < src->size_x; x++)
			*addr2_s(dst->ptr, y, pixels+x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, 0, x, src->stride_x, src->stride_y);

	// B
	for(int y = 0; y < pixels; y++)
		for(int x = 0; x < src->size_x; x++)
			*addr2_s(dst->ptr, pixels+src->size_y+y, pixels+x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, src->size_y-1, x, src->stride_x, src->stride_y);

	// L
	for(int y = 0; y < src->size_y; y++)
		for(int x = 0; x < pixels; x++)
			*addr2_s(dst->ptr, pixels+y, x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, y, 0, src->stride_x, src->stride_y);
	
	// R
	for(int y = 0; y < src->size_y; y++)
		for(int x = 0; x < pixels; x++)
			*addr2_s(dst->ptr, pixels+y, pixels+src->size_x+x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, y, src->size_x-1, src->stride_x, src->stride_y);

	// TL
	for(int y = 0; y < pixels; y++)
		for(int x = 0; x < pixels; x++)
			*addr2_s(dst->ptr, y, x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, 0, 0, src->stride_x, src->stride_y);

	// TR
	for(int y = 0; y < pixels; y++)
		for(int x = 0; x < pixels; x++)
			*addr2_s(dst->ptr, y, pixels+src->size_x+x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, 0, src->size_x-1, src->stride_x, src->stride_y);

	// BL
	for(int y = 0; y < pixels; y++)
		for(int x = 0; x < pixels; x++)
			*addr2_s(dst->ptr, pixels+src->size_y+y, x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, src->size_y-1, 0, src->stride_x, src->stride_y);

	// BR
	for(int y = 0; y < pixels; y++)
		for(int x = 0; x < pixels; x++)
			*addr2_s(dst->ptr, pixels+src->size_y+y, pixels+src->size_x+x, dst->stride_x, dst->stride_y) =
				*addr2_const_s(src->ptr, src->size_y-1, src->size_x-1, src->stride_x, src->stride_y);

	return dst;
}
