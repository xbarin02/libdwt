/**
 * @brief Integer-to-integer 2-D transform using CDF 9/7 and core approach.
 */

#include <stdlib.h> // malloc, abort
#include <stdio.h> // fprintf
#include "libdwt.h"
#include "core-int.h"
#include "image.h" // image_t

void ftransform(struct image_t *src, struct image_t *dst)
{
#if 0
	dwt_util_copy3_s(
		src->ptr,
		dst->ptr,
		src->stride_y,
		src->stride_x,
		dst->stride_y,
		dst->stride_x,
		src->size_x,
		src->size_y
	);

	int j = 1;

	dwt_cdf97_2f_inplace_i(
		dst->ptr,
		dst->stride_y,
		dst->stride_x,
		dst->size_x,
		dst->size_y,
		dst->size_x,
		dst->size_y,
		&j,
		1,
		0
	);
#else
	dwt_cdf97_2f_vert2x2_i(
		src->ptr,
		src->stride_x,
		src->stride_y,
		dst->ptr,
		dst->stride_x,
		dst->stride_y,
		src->size_x,
		src->size_y
	);
#endif
}

void itransform(struct image_t *image)
{
	dwt_cdf97_2i_inplace_i(
		image->ptr,
		image->stride_y,
		image->stride_x,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		1,
		1,
		0
	);
}

void fill_with_test_pattern(struct image_t *image)
{
	int type = 2;

	dwt_util_test_image_fill2_i(
		image->ptr,
		image->stride_y,
		image->stride_x,
		image->size_x,
		image->size_y,
		0,
		type
	);
}

int dump(struct image_t *image, const char *path)
{
	return dwt_util_save_to_pgm_i(
		path,
		0xff,
		image->ptr,
		image->stride_y,
		image->stride_x,
		image->size_x,
		image->size_y
	);
}

int dump_log(struct image_t *image, const char *path)
{
	void *copy = malloc(image->size);

	dwt_util_conv_show_i(image->ptr, copy,  image->stride_y,  image->stride_x, image->size_x, image->size_y);

	int r = dwt_util_save_to_pgm_i(
		path,
		0xff,
		copy,
		image->stride_y,
		image->stride_x,
		image->size_x,
		image->size_y
	);

	free(copy);

	return r;
}

int compare(struct image_t *src, struct image_t *dst)
{
	return dwt_util_compare_i(src->ptr, dst->ptr, src->stride_y, src->stride_x, src->size_x, src->size_y);
}

int main()
{
	struct image_t source = {
		.size_x   = 512, // width
		.size_y   = 512, // height
		.stride_x = sizeof(int32_t), // size of pixel
	};

	source.stride_y = source.stride_x * source.size_x; // size of row
	source.size     = source.stride_y * source.size_y; // size of image

	struct image_t target = source; // copy a struct

	source.ptr = malloc(source.size);
	target.ptr = malloc(target.size);

	if( !source.ptr || !target.ptr )
	{
		fprintf(stderr, "Not enough memory\n");
		return 1;
	}

	// fill the source
	fill_with_test_pattern(&source);

	dump(&source, "source.pgm");

	// process the image
	ftransform(&source, &target);

	int min, max;
	dwt_util_find_min_max_i(
		target.ptr,
		target.size_x,
		target.size_y,
		target.stride_y,
		target.stride_x,
		&min,
		&max
	);
	dwt_util_log(LOG_DBG, "min=%i max=%i\n", min, max);

	dump_log(&target, "subbands.pgm");

#if 1
	itransform(&target);

	if( compare(&source, &target) )
		fprintf(stderr, "Something is wrong\n");
#endif

	dump(&target, "target.pgm");

	free(source.ptr);
	free(target.ptr);

	return 0;
}
