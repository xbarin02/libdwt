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

	// image size
	const int x = 512, y = 512;

	// compute optimal stride
	const int stride_x = dwt_util_get_opt_stride(x * sizeof(double)), stride_y = sizeof(double);

	// some log messages
	dwt_util_log(LOG_INFO, "Library version is \"%s\".\n", dwt_util_version());
	dwt_util_log(LOG_INFO, "We are running on \"%s\" architecture.\n", dwt_util_arch());
	dwt_util_log(LOG_INFO, "Node name is \"%s\".\n", dwt_util_node());
	dwt_util_log(LOG_INFO, "Application name is \"%s\".\n", dwt_util_appname());
	dwt_util_log(LOG_INFO, "Using %i threads.\n", dwt_util_get_num_threads());
	dwt_util_log(LOG_INFO, "Using %i workers.\n", dwt_util_get_num_workers());
	dwt_util_log(LOG_INFO, "Using image of size of %ix%i pixels.\n", x, y);
	dwt_util_log(LOG_INFO, "Using stride of %i bytes.\n", stride_x);

	// image data
	void *data1, *data2, *data3;

	// full decomposition
	int j = -1;

	dwt_util_log(LOG_INFO, "generating test images...\n");

	// create test images
	dwt_util_alloc_image(&data1, stride_x, stride_y, x, y);
	dwt_util_test_image_fill_d(data1, stride_x, stride_y, x, y, 0);
	dwt_util_alloc_image(&data2, stride_x, stride_y, x, y);
	dwt_util_test_image_fill_d(data2, stride_x, stride_y, x, y, 0);
	dwt_util_alloc_image(&data3, stride_x, stride_y, x, y);

	// init timer
	dwt_clock_t time_start;
	dwt_clock_t time_stop;
	const int type = dwt_util_clock_autoselect();

	dwt_util_log(LOG_INFO, "forward transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// forward transform
	dwt_cdf97_2f_d(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0);

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

	// convert transform into viewable format
	dwt_util_conv_show_d(data1, data3, stride_x, stride_y, x, y);
	
	dwt_util_log(LOG_INFO, "inverse transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// inverse transform
	dwt_cdf97_2i_d(data1, stride_x, stride_y, x, y, x, y, j, 0, 0);

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

	// compare
	if( dwt_util_compare_d(data1, data2, stride_x, stride_y, x, y) )
		dwt_util_log(LOG_INFO, "images differs\n");
	else
		dwt_util_log(LOG_INFO, "success\n");
	
	// release platform resources
	dwt_util_finish();

	// save images into files
	dwt_util_log(LOG_INFO, "saving...\n");
	dwt_util_save_to_pgm_d("data1.pgm", 1.0, data1, stride_x, stride_y, x, y);
	dwt_util_save_to_pgm_d("data2.pgm", 1.0, data2, stride_x, stride_y, x, y);
	dwt_util_save_to_pgm_d("data3.pgm", 1.0, data3, stride_x, stride_y, x, y);

	// free allocated memory
	dwt_util_free_image(&data1);
	dwt_util_free_image(&data2);

	return 0;
}
