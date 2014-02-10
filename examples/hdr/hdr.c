#include "libdwt.h"
#include "exr.h"
#include "image.h"
#include <math.h>

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

	//const float low = 1.0f - image_min_s(&yuv[0]);
	const float low = 1e-5 - image_min_s(&yuv[0]);

	dwt_util_log(LOG_INFO, "Y_low = %f\n", low);

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float *pY = image_coeff_s(&yuv[0], y, x);

			if( logf(*pY + low) < 0.f )
				dwt_util_log(LOG_DBG, "oups: log(%f + %f)\n", *pY, low);

			*pY = logf(*pY + low);
		}
	}

	image_save_to_pgm_s(&yuv[0], "logluma.pgm");

	int j = -1;
	dwt_eaw53_2f_dummy_s(yuv[0].ptr, yuv[0].stride_x, yuv[0].stride_y, yuv[0].size_x, yuv[0].size_y, yuv[0].size_x, yuv[0].size_y, &j, 0);
	dwt_util_log(LOG_INFO, "j = %i\n", j);
	float *wH[j];
	float *wV[j];

	dwt_eaw53_2f_s(
		yuv[0].ptr,
		yuv[0].stride_x,
		yuv[0].stride_y,
		yuv[0].size_x,
		yuv[0].size_y,
		yuv[0].size_x,
		yuv[0].size_y,
		&j,
		0,
		0,
		wH,
		wV
	);

	image_save_transform_to_pgm_s(&yuv[0], "eaw.pgm");

	extern float g_alpha;
	g_alpha = 0.8f;

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
	float beta = 0.75f; // FIXME
	for(int jj = 1; jj <= j; jj++)
	{
		image_compress_subband_s(&yuv[0], jj, DWT_LH, beta);
		image_compress_subband_s(&yuv[0], jj, DWT_HL, beta);
		image_compress_subband_s(&yuv[0], jj, DWT_HH, beta);
	}
#endif

	image_save_transform_to_pgm_s(&yuv[0], "eaw_scaled.pgm");

	dwt_eaw53_2i_s(
		yuv[0].ptr,
		yuv[0].stride_x,
		yuv[0].stride_y,
		yuv[0].size_x,
		yuv[0].size_y,
		yuv[0].size_x,
		yuv[0].size_y,
		j, 0, 0, wH, wV);

	image_save_to_pgm_s(&yuv[0], "logluma_scaled.pgm");

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float *pY = image_coeff_s(&yuv[0], y, x);

			*pY = expf(*pY) - low;
		}
	}

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
