/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Simple example application showing usage of libdwt functions.
 */

#include "libdwt.h"

int main()
{
	// init platform
	dwt_util_init();

	// set threads, workes, accel_type
	dwt_util_set_num_threads(1);
	dwt_util_set_num_workers(1);
	dwt_util_set_accel(0);

	// image size
	const int x = 512, y = 512;

	// compute optimal stride
	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(stride_y * x);

	// image data
	void *data1, *data2, *data3;

	// some log messages
	dwt_util_log(LOG_INFO, "Acceleration type is %i.\n", dwt_util_get_accel());
	dwt_util_log(LOG_INFO, "Using %i threads.\n", dwt_util_get_num_threads());
	dwt_util_log(LOG_INFO, "Using %i workers.\n", dwt_util_get_num_workers());

	// decomposition
	int j = 1;

	dwt_util_log(LOG_INFO, "generating test images...\n");

	// create test images
	dwt_util_alloc_image(&data1, stride_x, stride_y, x, y);
	dwt_util_test_image_fill_s(data1, stride_x, stride_y, x, y, 0);
	dwt_util_alloc_image(&data2, stride_x, stride_y, x, y);
	dwt_util_copy_s(data1, data2, stride_x, stride_y, x, y);
	dwt_util_alloc_image(&data3, stride_x, stride_y, x, y);

	// init timer
	dwt_clock_t time_start;
	dwt_clock_t time_stop;
	const int type = dwt_util_clock_autoselect();

	dwt_util_log(LOG_INFO, "forward transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// forward transform
#if 0
	dwt_cdf97_2f_inplace_s(data1, stride_x, stride_y, x, y, x, y, &j, 1, 0);
#else
	dwt_cdf97_2f_inplace_sdl_s(data1, stride_x, stride_y, x, y, x, y, &j, 1, 0);
#endif

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

	// convert transform into viewable format
	dwt_util_conv_show_s(data1, data3, stride_x, stride_y, x, y);

	dwt_util_log(LOG_INFO, "inverse transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// inverse transform
	dwt_cdf97_2i_inplace_s(data1, stride_x, stride_y, x, y, x, y, j, 1, 0);

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
	dwt_util_save_to_pgm_s("data.pgm", 1.0, data1, stride_x, stride_y, x, y);
	dwt_util_save_to_pgm_s("orig.pgm", 1.0, data2, stride_x, stride_y, x, y);
	dwt_util_save_to_pgm_s("tran.pgm", 1.0, data3, stride_x, stride_y, x, y);

	// free allocated memory
	dwt_util_free_image(&data1);
	dwt_util_free_image(&data2);
	dwt_util_free_image(&data3);

	return 0;
}
