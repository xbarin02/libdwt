#include "image2.h"

#include "libdwt.h"
#include "system.h"
#include "inline.h"
#include "fix.h"
#include "cores.h"

struct image_t *image2_create_ex(size_t size_pel, int size_x, int size_y, int opt_stride)
{
	struct image_t *image = dwt_util_reliably_alloc1(sizeof(struct image_t));

	image->size_x = size_x;
	image->size_y = size_y;

	image->stride_x = size_pel;
	image->stride_y = dwt_util_get_stride(size_x * size_pel, opt_stride);

	image->size = size_y * image->stride_y;

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "%s: allocate %u MiB\n", __FUNCTION__, image->size >> 20);
#endif

	image->ptr = dwt_util_reliably_alloc1(image->size);

	return image;
}

void image2_conv_float32_to_fix32(struct image_t *source, struct image_t *target)
{
	for(int y = 0; y < source->size_y; y++)
	{
		for(int x = 0; x < source->size_x; x++)
		{
			*addr2_i(target->ptr, y, x, target->stride_y, target->stride_x) =
				conv_float32_to_fix32( *addr2_s(source->ptr, y, x, source->stride_y, source->stride_x) );
		}
	}
}

void image2_conv_float32_to_fix16(struct image_t *source, struct image_t *target)
{
	for(int y = 0; y < source->size_y; y++)
	{
		for(int x = 0; x < source->size_x; x++)
		{
			*addr2_i16(target->ptr, y, x, target->stride_y, target->stride_x) =
				conv_float32_to_fix16( *addr2_s(source->ptr, y, x, source->stride_y, source->stride_x) );
		}
	}
}

void image2_conv_fix32_to_float32(struct image_t *source, struct image_t *target)
{
	for(int y = 0; y < source->size_y; y++)
	{
		for(int x = 0; x < source->size_x; x++)
		{
			*addr2_s(target->ptr, y, x, target->stride_y, target->stride_x) =
				conv_fix32_to_float32( *addr2_i(source->ptr, y, x, source->stride_y, source->stride_x) );
		}
	}
}

void image2_conv_fix16_to_float32(struct image_t *source, struct image_t *target)
{
	for(int y = 0; y < source->size_y; y++)
	{
		for(int x = 0; x < source->size_x; x++)
		{
			*addr2_s(target->ptr, y, x, target->stride_y, target->stride_x) =
				conv_fix16_to_float32( *addr2_i16(source->ptr, y, x, source->stride_y, source->stride_x) );
		}
	}
}

size_t sizeof_type(enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32: return sizeof(float);
		case TYPE_INT32: return sizeof(int32_t);
		case TYPE_FIX32: return sizeof(FIX32_T);
		case TYPE_FIX16: return sizeof(FIX16_T);
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}

	return 0;
}

// mean squared error
float image2_mse(struct image_t *source, struct image_t *target, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_INT32:
		{
			float mse = 0.f;

			for(int y = 0; y < source->size_y; y++)
			{
				for(int x = 0; x < source->size_x; x++)
				{
					const int diff =
						*addr2_i(source->ptr, y, x, source->stride_y, source->stride_x) -
						*addr2_i(target->ptr, y, x, target->stride_y, target->stride_x);
					mse += (float)diff*diff;
				}
			}

			return mse/source->size_y/source->size_x;
		}
		case TYPE_FLOAT32:
		{
			float mse = 0.f;

			for(int y = 0; y < source->size_y; y++)
			{
				for(int x = 0; x < source->size_x; x++)
				{
					const float diff =
						*addr2_s(source->ptr, y, x, source->stride_y, source->stride_x) -
						*addr2_s(target->ptr, y, x, target->stride_y, target->stride_x);
					mse += diff*diff;
				}
			}

			return mse/source->size_y/source->size_x;
		}
		case TYPE_FIX32:
		{
			struct image_t *float32_source = image2_create_ex(sizeof_type(TYPE_FLOAT32), source->size_x, source->size_y, 2);
			struct image_t *float32_target = image2_create_ex(sizeof_type(TYPE_FLOAT32), target->size_x, target->size_y, 2);

			image2_conv_fix32_to_float32(source, float32_source);
			image2_conv_fix32_to_float32(target, float32_target);

			float mse = image2_mse(float32_source, float32_target, TYPE_FLOAT32);
	
			image_destroy(float32_source);
			image_destroy(float32_target);

			return mse;
		}
		case TYPE_FIX16:
		{
			struct image_t *float32_source = image2_create_ex(sizeof_type(TYPE_FLOAT32), source->size_x, source->size_y, 2);
			struct image_t *float32_target = image2_create_ex(sizeof_type(TYPE_FLOAT32), target->size_x, target->size_y, 2);

			image2_conv_fix16_to_float32(source, float32_source);
			image2_conv_fix16_to_float32(target, float32_target);

			float mse = image2_mse(float32_source, float32_target, TYPE_FLOAT32);
	
			image_destroy(float32_source);
			image_destroy(float32_target);

			return mse;
		}
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}

	// fallback
	return +INFINITY;
}

void image2_fill(struct image_t *image, enum dwt_types data_type)
{
	int pattern_type = 2;

	switch(data_type)
	{
		case TYPE_FLOAT32:
		{
			dwt_util_test_image_fill2_s(
				image->ptr,
				image->stride_y,
				image->stride_x,
				image->size_x,
				image->size_y,
				0,
				pattern_type
			);
			break;
		}
		case TYPE_INT32:
		{
			dwt_util_test_image_fill2_i(
				image->ptr,
				image->stride_y,
				image->stride_x,
				image->size_x,
				image->size_y,
				0,
				pattern_type
			);
			break;
		}
		case TYPE_FIX32:
		{
			image2_fill(image, TYPE_FLOAT32);
			image2_conv_float32_to_fix32(image, image);
			break;
		}
		case TYPE_FIX16:
		{
			// allocate TYPE_FLOAT32
			struct image_t *float32_image = image2_create_ex(4, image->size_x, image->size_y, 2);
			// fill TYPE_FLOAT32
			image2_fill(float32_image, TYPE_FLOAT32);
			// conv TYPE_FLOAT32 to TYPE_FIX16
			image2_conv_float32_to_fix16(float32_image, image);
			// free TYPE_FLOAT32
			image_destroy(float32_image);
			break;
		}
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

void image2_save_to_pgm_s(struct image_t *image, const char *path)
{
	struct image_t swapped = (struct image_t){
		.ptr      = image->ptr,
		.size     = image->size,
		.size_x   = image->size_x,
		.size_y   = image->size_y,
		.stride_x = image->stride_y,
		.stride_y = image->stride_x,
	};

	// FIXME: image_save_to_pgm_s(&swapped, path);
	image_save_to_pgm2_s(&swapped, path);
}

void image2_save_to_pgm(struct image_t *image, const char *path, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
		{
			image2_save_to_pgm_s(image, path);
			break;
		}
		case TYPE_INT32:
		{
			dwt_util_save_to_pgm_i(
				path,
				0xff,
				image->ptr,
				image->stride_y,
				image->stride_x,
				image->size_x,
				image->size_y
			);
			break;
		}
		case TYPE_FIX32:
		{
			// allocate
			struct image_t image_copy = *image;
			image_copy.ptr = dwt_util_reliably_alloc1(image->size);
			// conv
			image2_conv_fix32_to_float32(image, &image_copy);
			// save
			image2_save_to_pgm(&image_copy, path, TYPE_FLOAT32);
			// free
			dwt_util_free(image_copy.ptr);
			break;
		}
		case TYPE_FIX16:
		{
			// allocate TYPE_FLOAT32
			struct image_t *float32_image = image2_create_ex(4, image->size_x, image->size_y, 2);
			// conv TYPE_FIX16 to TYPE_FLOAT32
			image2_conv_fix16_to_float32(image, float32_image);
			// save TYPE_FLOAT32
			image2_save_to_pgm(float32_image, path, TYPE_FLOAT32);
			// free TYPE_FLOAT32
			image_destroy(float32_image);
			break;
		}
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

void image2_save_log_to_pgm_s(struct image_t *image, const char *path)
{
	struct image_t swapped = (struct image_t){
		.ptr      = image->ptr,
		.size     = image->size,
		.size_x   = image->size_x,
		.size_y   = image->size_y,
		.stride_x = image->stride_y,
		.stride_y = image->stride_x,
	};

	image_save_log_to_pgm_s(&swapped, path);
}

int image2_save_log_to_pgm_i(struct image_t *image, const char *path)
{
	void *copy = dwt_util_reliably_alloc1(image->size);

	dwt_util_conv_show_i(image->ptr, copy, image->stride_y,  image->stride_x, image->size_x, image->size_y);

	int r = dwt_util_save_to_pgm_i(
		path,
		0xff,
		copy,
		image->stride_y,
		image->stride_x,
		image->size_x,
		image->size_y
	);

	dwt_util_free(copy);

	return r;
}

void image2_save_log_to_pgm(struct image_t *image, const char *path, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
			image2_save_log_to_pgm_s(image, path);
			break;
		case TYPE_INT32:
			image2_save_log_to_pgm_i(image, path);
			break;
		case TYPE_FIX32:
		{
			struct image_t image_copy = *image;
			image_copy.ptr = dwt_util_reliably_alloc1(image->size);
			image2_conv_fix32_to_float32(image, &image_copy);
			image2_save_log_to_pgm(&image_copy, path, TYPE_FLOAT32);
			dwt_util_free(image_copy.ptr);
			break;
		}
		case TYPE_FIX16:
		{
			struct image_t *float32_image = image2_create_ex(sizeof_type(TYPE_FLOAT32), image->size_x, image->size_y, 2);
			image2_conv_fix16_to_float32(image, float32_image);
			image2_save_log_to_pgm(float32_image, path, TYPE_FLOAT32);
			image_destroy(float32_image);
			break;
		}
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

void image2_idwt_cdf53_ip(struct image_t *image, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
		{
			dwt_cdf53_2i_inplace_s(
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
			break;
		}
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

void image2_idwt_cdf97_ip(struct image_t *image, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
		{
			dwt_cdf97_2i_inplace_s(
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
			break;
		}
		case TYPE_INT32:
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
			break;
		}
		case TYPE_FIX32:
		{
			image2_conv_fix32_to_float32(image, image);
			image2_idwt_cdf97_ip(image, TYPE_FLOAT32);
			image2_conv_float32_to_fix32(image, image);
			break;
		}
		case TYPE_FIX16:
		{
			struct image_t *float32_image = image2_create_ex(sizeof_type(TYPE_FLOAT32), image->size_x, image->size_y, 2);
			image2_conv_fix16_to_float32(image, float32_image);
			image2_idwt_cdf97_ip(float32_image, TYPE_FLOAT32);
			image2_conv_float32_to_fix16(float32_image, image);
			image_destroy(float32_image);
			break;
		}
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

void image2_idwt_cdf97_op(struct image_t *source, struct image_t *target, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
			cores2i_cdf97_v2x2_f32(source, target);
			break;
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

void image2_fdwt_cdf53_op(struct image_t *source, struct image_t *target, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
			cores2f_cdf53_v2x2_f32(source, target);
			break;
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

void image2_fdwt_cdf97_op(struct image_t *source, struct image_t *target, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
			cores2f_cdf97_v2x2_f32(source, target);
			break;
		case TYPE_INT32:
			cores2f_cdf97_v2x2_i32(source, target);
			break;
		case TYPE_FIX32:
			cores2f_cdf97_v2x2_x32(source, target);
			break;
		case TYPE_FIX16:
			cores2f_cdf97_v2x2_x16(source, target);
			break;
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}
}

int image2_compare_map(struct image_t *source, struct image_t *target, struct image_t *map, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
			return dwt_util_compare2_destructive2_s(
				source->ptr,
				target->ptr,
				map->ptr,
				source->stride_y,
				source->stride_x,
				target->stride_y,
				target->stride_x,
				map->stride_y,
				map->stride_x,
				source->size_x,
				source->size_y
			);
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}

	return 1;
}
int image2_compare(struct image_t *source, struct image_t *target, enum dwt_types data_type)
{
	switch(data_type)
	{
		case TYPE_FLOAT32:
			return dwt_util_compare2_s(
				source->ptr,
				target->ptr,
				source->stride_y,
				source->stride_x,
				target->stride_y,
				target->stride_x,
				source->size_x,
				source->size_y
			);
		case TYPE_INT32:
			return dwt_util_compare2_i(
				source->ptr,
				target->ptr,
				source->stride_y,
				source->stride_x,
				target->stride_y,
				target->stride_x,
				source->size_x,
				source->size_y
			);
		case TYPE_FIX32:
		{
			struct image_t source_copy = *source;
			struct image_t target_copy = *target;

			source_copy.ptr = dwt_util_reliably_alloc1(source->size);
			target_copy.ptr = dwt_util_reliably_alloc1(target->size);

			image2_conv_fix32_to_float32(source, &source_copy);
			image2_conv_fix32_to_float32(target, &target_copy);
			
			int return_code = image2_compare(&source_copy, &target_copy, TYPE_FLOAT32);
			
			dwt_util_free(target_copy.ptr);
			dwt_util_free(source_copy.ptr);

			return return_code;
		}
		case TYPE_FIX16:
		{
			struct image_t *float32_source = image2_create_ex(sizeof_type(TYPE_FLOAT32), source->size_x, source->size_y, 2);
			struct image_t *float32_target = image2_create_ex(sizeof_type(TYPE_FLOAT32), target->size_x, target->size_y, 2);

			image2_conv_fix16_to_float32(source, float32_source);
			image2_conv_fix16_to_float32(target, float32_target);

			int return_code = image2_compare(float32_source, float32_target, TYPE_FLOAT32);
	
			image_destroy(float32_source);
			image_destroy(float32_target);

			return return_code;
		}
		default:
			dwt_util_error("%s: unsupported data type (%i)\n", __FUNCTION__, data_type);
	}

	return 1;
}

void image2_flush_cache(struct image_t *image)
{
	assert( image->stride_y > 0 );

	dwt_util_flush_cache(
		image->ptr,
		image->size
	);
}
