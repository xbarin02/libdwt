#include "libdwt.h"
#include "exr.h"
#include "image.h"
#include <math.h>
#include "eaw-experimental.h"

//#define CDF97

int image_load_from_exr_s(
	image_t *image,
	const char *path,
	const char *channel
)
{
	dwt_util_load_from_exr_s(
		path,
		channel,
		&image->ptr,
		&image->stride_x,
		&image->stride_y,
		&image->size_x,
		&image->size_y
	);

	return NULL != image->ptr;
}

void image_info_s(image_t *image, const char *desc)
{
	float minv, maxv;

	dwt_util_find_min_max_s(
		image->ptr,
		image->size_x,
		image->size_y,
		image->stride_x,
		image->stride_y,
		&minv,
		&maxv
	);

	const int size_x = image->size_x;
	const int size_y = image->size_y;

	dwt_util_log(LOG_INFO, "%s [%ix%i]: min=%f max=%f\n", desc, size_x, size_y, minv, maxv);
}

float image_min_s(image_t *image)
{
	float minv, maxv;

	dwt_util_find_min_max_s(
		image->ptr,
		image->size_x,
		image->size_y,
		image->stride_x,
		image->stride_y,
		&minv,
		&maxv
	);

	return minv;
}

void image_scale_subband_s(image_t *image, int j, enum dwt_subbands band, float a)
{
	void *subband_ptr;
	int subband_size_x;
	int subband_size_y;

	dwt_util_subband_s(
		image->ptr, image->stride_x, image->stride_y, image->size_x, image->size_y, image->size_x, image->size_y,
		j,
		band,
		&subband_ptr, &subband_size_x, &subband_size_y);

	dwt_util_scale_s(subband_ptr, subband_size_x, subband_size_y, image->stride_x, image->stride_y, a);
}

int dwt_util_compress_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float beta
)
{
	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float *c = dwt_util_addr_coeff_s(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			);

			float sign = (*c > 0) ? +1 : -1;
			float mag = fabsf(*c);

			*c = sign * powf(mag, beta);
		}
	}

	return 0;
}

void image_compress_subband_s(image_t *image, int j, enum dwt_subbands band, float beta)
{
	void *subband_ptr;
	int subband_size_x;
	int subband_size_y;

	dwt_util_subband_s(
		image->ptr, image->stride_x, image->stride_y, image->size_x, image->size_y, image->size_x, image->size_y,
		j,
		band,
		&subband_ptr, &subband_size_x, &subband_size_y);

	dwt_util_compress_s(subband_ptr, subband_size_x, subband_size_y, image->stride_x, image->stride_y, beta);
}

void image_save_transform_to_pgm_s(image_t *image, const char *path)
{
	image_t *show = image_create_s(image->size_x, image->size_y);

	if( show->stride_x != image->stride_x || show->stride_y != image->stride_y )
		dwt_util_error("TODO: unimplemented\n");

	dwt_util_conv_show_s(image->ptr, show->ptr, image->stride_x, image->stride_y, image->size_x, image->size_y);

	image_save_to_pgm_s(show, path);
}

void image_fdwt_eaw53_s(
	image_t *image,
	int *levels,
	float *wH[],
	float *wV[],
	float alpha
)
{
	dwt_eaw53_2f_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		levels,
		0,
		0,
		wH,
		wV,
		alpha
	);
}

void image_fdwt_eaw97_s(
	image_t *image,
	int *levels,
	float *wH[],
	float *wV[],
	float alpha
)
{
	dwt_eaw97_2f_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		levels,
		0,
		0,
		wH,
		wV,
		alpha
	);
}

void image_idwt_eaw53_s(
	image_t *image,
	int levels,
	float *wH[],
	float *wV[]
)
{
	dwt_eaw53_2i_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		levels,
		0,
		0,
		wH,
		wV
	);
}

void image_idwt_eaw97_s(
	image_t *image,
	int levels,
	float *wH[],
	float *wV[]
)
{
	dwt_eaw97_2i_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		levels,
		0,
		0,
		wH,
		wV
	);
}

void image_shift_s(image_t *image, float a)
{
	dwt_util_shift_s(
		image->ptr,
		image->size_x,
		image->size_y,
		image->stride_x,
		image->stride_y,
		a
	);
}

void image_log_s(image_t *image, float eps)
{
	for(int y = 0; y < image->size_y; y++)
	{
		for(int x = 0; x < image->size_x; x++)
		{
			float *c = image_coeff_s(image, y, x);

			*c = logf(*c + eps);
		}
	}
}

void image_exp_s(image_t *image, float eps)
{
	for(int y = 0; y < image->size_y; y++)
	{
		for(int x = 0; x < image->size_x; x++)
		{
			float *c = image_coeff_s(image, y, x);

			*c = expf(*c) - eps;
		}
	}
}

int image_levels_eaw53_s(image_t *image)
{
	int j = -1;

	dwt_eaw53_2f_dummy_s(image->ptr, image->stride_x, image->stride_y, image->size_x, image->size_y, image->size_x, image->size_y, &j, 0);

	return j;
}

void image_compress_details_s(image_t *image, int levels, float beta)
{
	for(int jj = 1; jj <= levels; jj++)
	{
		image_compress_subband_s(image, jj, DWT_LH, beta);
		image_compress_subband_s(image, jj, DWT_HL, beta);
		image_compress_subband_s(image, jj, DWT_HH, beta);
	}
}

int main(int argc, char *argv[])
{
	const char *path = argc>1 ? argv[1] : "./data/img_light1_lamp250_pos0.exr";

	image_t rgb[3];

	image_load_from_exr_s(&rgb[0], path, "R");
	image_load_from_exr_s(&rgb[1], path, "G");
	image_load_from_exr_s(&rgb[2], path, "B");

	image_info_s(&rgb[0], "R");
	image_info_s(&rgb[1], "G");
	image_info_s(&rgb[2], "B");

	const int size_x = rgb[0].size_x;
	const int size_y = rgb[0].size_y;

	image_t yuv[3];
	yuv[0] = *image_create_s(size_x, size_y);
	yuv[1] = *image_create_s(size_x, size_y);
	yuv[2] = *image_create_s(size_x, size_y);

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float R = *image_coeff_s(&rgb[0], y, x);
			const float G = *image_coeff_s(&rgb[1], y, x);
			const float B = *image_coeff_s(&rgb[2], y, x);

			float Y, U, V;

			Y = 0.299*R + 0.587*G + 0.114*B;
			U = 0.492*(B-Y);
			V = 0.877*(R-Y);

			*image_coeff_s(&yuv[0], y, x) = Y;
			*image_coeff_s(&yuv[1], y, x) = U;
			*image_coeff_s(&yuv[2], y, x) = V;
		}
	}

	image_info_s(&yuv[0], "Y");
	image_info_s(&yuv[1], "U");
	image_info_s(&yuv[2], "V");

	image_save_to_pgm_s(&yuv[0], "luma.pgm");

	const float low = image_min_s(&yuv[0]);
	const float eps = 1e-5;

	image_shift_s(&yuv[0], -low);

	dwt_util_log(LOG_INFO, "Y_low = %f\n", low);

	image_log_s(&yuv[0], eps);

	image_save_to_pgm_s(&yuv[0], "logluma.pgm");

	int j = image_levels_eaw53_s(&yuv[0]);

	dwt_util_log(LOG_INFO, "j = %i\n", j);

	float *wH[j];
	float *wV[j];

	float alpha = 0.8f;

#ifdef CDF97
	image_fdwt_eaw97_s(&yuv[0], &j, wH, wV, alpha);
#else
	image_fdwt_eaw53_s(&yuv[0], &j, wH, wV, alpha);
#endif

	image_save_transform_to_pgm_s(&yuv[0], "eaw.pgm");

#if 0
	float beta = 0.15f;
	float gamma = 0.7f;
	for(int jj = 1; jj <= j; jj++)
	{
		image_scale_subband_s(&yuv[0], jj, DWT_LH, powf(gamma, (float)j));
		image_scale_subband_s(&yuv[0], jj, DWT_HL, powf(gamma, (float)j));
		image_scale_subband_s(&yuv[0], jj, DWT_HH, powf(gamma, (float)j));
	}
	image_scale_subband_s(&yuv[0], j, DWT_LL, beta);
#else
	float beta = 0.70f;

	image_compress_details_s(&yuv[0], j, beta);
#endif

	image_save_transform_to_pgm_s(&yuv[0], "eaw_scaled.pgm");

#ifdef CDF97
	image_idwt_eaw97_s(&yuv[0], j, wH, wV);
#else
	image_idwt_eaw53_s(&yuv[0], j, wH, wV);
#endif

	image_save_to_pgm_s(&yuv[0], "logluma_scaled.pgm");

	image_exp_s(&yuv[0], eps);

	image_shift_s(&yuv[0], +low);

	image_save_to_pgm_s(&yuv[0], "luma_scaled.pgm");

	image_info_s(&yuv[0], "Y");
	image_info_s(&yuv[1], "U");
	image_info_s(&yuv[2], "V");

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float Y = *image_coeff_s(&yuv[0], y, x);
			const float U = *image_coeff_s(&yuv[1], y, x);
			const float V = *image_coeff_s(&yuv[2], y, x);

			float R, G, B;

			R = Y +1.13983*V;
			G = Y -0.39465*U -0.58060*V;
			B = Y +2.03211*U;

			*image_coeff_s(&rgb[0], y, x) = R;
			*image_coeff_s(&rgb[1], y, x) = G;
			*image_coeff_s(&rgb[2], y, x) = B;
		}
	}

	image_info_s(&rgb[0], "R");
	image_info_s(&rgb[1], "G");
	image_info_s(&rgb[2], "B");

	dwt_util_log(LOG_INFO, "done\n");

	return 0;
}
