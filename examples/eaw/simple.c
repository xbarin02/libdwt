/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Simple example on Edge-Avoiding Wavelets.
 */

#include "libdwt.h"
#include "eaw-experimental.h"

#define EAW
#define CDF97

int main(int argc, char *argv[])
{
	const char *path = argc > 1 ? argv[1] : "Lenna.pgm";

	// init platform
	dwt_util_init();

	// image size
	int x, y;

	// image strides
	int stride_x, stride_y;

	// image data
	void *data1, *data2, *data3;

	// some log messages
	dwt_util_log(LOG_INFO, "Library version is \"%s\".\n", dwt_util_version());
	dwt_util_log(LOG_INFO, "We are running on \"%s\" architecture.\n", dwt_util_arch());
	dwt_util_log(LOG_INFO, "Node name is \"%s\".\n", dwt_util_node());
	dwt_util_log(LOG_INFO, "Application name is \"%s\".\n", dwt_util_appname());
	dwt_util_log(LOG_INFO, "Acceleration type is %i.\n", dwt_util_get_accel());
	dwt_util_log(LOG_INFO, "Using %i threads.\n", dwt_util_get_num_threads());
	dwt_util_log(LOG_INFO, "Using %i workers.\n", dwt_util_get_num_workers());

	dwt_util_log(LOG_INFO, "Loading a file \"%s\"...\n", path);

	if( dwt_util_load_from_pgm_s(path, 1.0, &data1, &stride_x, &stride_y, &x, &y) )
	{
		// error occurred, use default test image
		dwt_util_log(LOG_ERR, "Unable to load an image, using the default one.\n");

		// default size
		x = 512;
		y = 512;

		stride_y = sizeof(float);
		stride_x = dwt_util_get_opt_stride(stride_y * x);

		dwt_util_alloc_image(&data1, stride_x, stride_y, x, y);
		const int fill_type = 1;
		dwt_util_test_image_fill2_s(data1, stride_x, stride_y, x, y, 0, fill_type);
	}

	dwt_util_log(LOG_INFO, "generating test images...\n");

	// create test images
	dwt_util_alloc_image(&data2, stride_x, stride_y, x, y);
	dwt_util_copy_s(data1, data2, stride_x, stride_y, x, y);
	dwt_util_alloc_image(&data3, stride_x, stride_y, x, y);

#ifdef EAW
	float alpha = 1.0f;
#endif

	// full decomposition
	int j = -1;

#ifdef EAW
	dwt_eaw53_2f_dummy_s(data1, stride_x, stride_y, x, y, x, y, &j, 0);
#else
	dwt_cdf53_2f_dummy_s(data1, stride_x, stride_y, x, y, x, y, &j, 0);
#endif

	// HACK
	j = 5;

	dwt_util_log(LOG_INFO, "j = %i\n", j);

#ifdef EAW
	float *wH[j];
	float *wV[j];
#endif

	// init timer
	dwt_clock_t time_start;
	dwt_clock_t time_stop;
	const int type = dwt_util_clock_autoselect();

	dwt_util_log(LOG_INFO, "forward transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// forward transform
#ifndef EAW
	#ifdef CDF97
	dwt_cdf97_2f_s(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0);
	#else
	dwt_cdf53_2f_s(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0);
	#endif
#else
	#ifdef CDF97
	dwt_eaw97_2f_s(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0, wH, wV, alpha);
	#else
	dwt_eaw53_2f_s(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0, wH, wV, alpha);
	#endif
#endif

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

#if 0
	// reset transform
	dwt_util_test_image_zero_s(data1, stride_x, stride_y, x, y);
	int jj = j;
	enum dwt_subbands subband = DWT_LL;
	void *subband_ptr;
	int subband_size_x;
	int subband_size_y;
	dwt_util_subband_s(data1, stride_x, stride_y, x, y, x, y, jj, subband, &subband_ptr, &subband_size_x, &subband_size_y);

	for(int off_x=-3; off_x<=+3; off_x+=2)
		for(int off_y=-3; off_y<=+3; off_y+=2)
			*dwt_util_addr_coeff_s(subband_ptr, off_y+subband_size_y/2, off_x+subband_size_x/2, stride_x, stride_y) = 20.f;
#endif

	// convert transform into viewable format
	dwt_util_conv_show_s(data1, data3, stride_x, stride_y, x, y);
	
	dwt_util_log(LOG_INFO, "inverse transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// inverse transform
#ifndef EAW
	#ifdef CDF97
	dwt_cdf97_2i_s(data1, stride_x, stride_y, x, y, x, y, j, 0, 0);
	#else
	dwt_cdf53_2i_s(data1, stride_x, stride_y, x, y, x, y, j, 0, 0);
	#endif
#else
	#ifdef CDF97
	dwt_eaw97_2i_s(data1, stride_x, stride_y, x, y, x, y, j, 0, 0, wH, wV);
	#else
	dwt_eaw53_2i_s(data1, stride_x, stride_y, x, y, x, y, j, 0, 0, wH, wV);
	#endif
#endif

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

	// compare
	if( dwt_util_compare_s(data1, data2, stride_x, stride_y, x, y) )
		dwt_util_log(LOG_INFO, "images differs\n");
	else
		dwt_util_log(LOG_INFO, "success\n");
	
	// release platform resources
	dwt_util_finish();

	// save images into files
	dwt_util_log(LOG_INFO, "saving...\n");
	dwt_util_save_to_pgm_s("data1.pgm", 1.0, data1, stride_x, stride_y, x, y);
	dwt_util_save_to_pgm_s("data2.pgm", 1.0, data2, stride_x, stride_y, x, y);
	dwt_util_save_to_pgm_s("data3.pgm", 1.0, data3, stride_x, stride_y, x, y);

	// free allocated memory
	dwt_util_free_image(&data1);
	dwt_util_free_image(&data2);
	dwt_util_free_image(&data3);

	return 0;
}
