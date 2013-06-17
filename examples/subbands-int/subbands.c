/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Example application showing usage of subband access functions.
 */

#include "libdwt.h"

int main()
{
	// init platform
	dwt_util_init();

	// image size
	const int x = 512, y = 512;

	// compute optimal stride
	const int stride_y = sizeof(int), stride_x = dwt_util_get_opt_stride(stride_y * x);

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
	void *data1, *data2;

	// full decomposition
	int j = -1;

	dwt_util_log(LOG_INFO, "generating test images...\n");

	// create test images
	dwt_util_alloc_image(&data1, stride_x, stride_y, x, y);
	dwt_util_test_image_fill_i(data1, stride_x, stride_y, x, y, 0);
	dwt_util_alloc_image(&data2, stride_x, stride_y, x, y);
	dwt_util_test_image_fill_i(data2, stride_x, stride_y, x, y, 0);

	// init timer
	dwt_clock_t time_start;
	dwt_clock_t time_stop;
	const int type = dwt_util_clock_autoselect();

	dwt_util_log(LOG_INFO, "forward transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// forward transform
	dwt_cdf53_2f_i(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0);

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

	for(int jj = 1; jj <= j; jj++)
	{
		void *subband_ptr;
		int subband_size_x;
		int subband_size_y;

		dwt_util_subband_i(data1, stride_x, stride_y, x, y, x, y, jj, DWT_LH, &subband_ptr, &subband_size_x, &subband_size_y);

		dwt_util_log(LOG_INFO, "erasing LH_%i subband of size of %ix%i coefficients...\n", jj, subband_size_x, subband_size_y);

		for(int yy = 0; yy < subband_size_y; yy++)
		{
			for(int xx = 0; xx < subband_size_x; xx++)
			{
				int *coeff = dwt_util_addr_coeff_i(subband_ptr, yy, xx, stride_x, stride_y);

				*coeff = 0;
			}
		}
	}

	dwt_util_log(LOG_INFO, "inverse transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// inverse transform
	dwt_cdf53_2i_i(data1, stride_x, stride_y, x, y, x, y, j, 0, 0);

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

	// compare
	if( dwt_util_compare_i(data1, data2, stride_x, stride_y, x, y) )
		dwt_util_log(LOG_INFO, "images differs\n");
	else
		dwt_util_log(LOG_INFO, "success\n");
	
	// release platform resources
	dwt_util_finish();

	// save images into files
	dwt_util_log(LOG_INFO, "saving...\n");
	dwt_util_save_to_pgm_i("data1.pgm", 255, data1, stride_x, stride_y, x, y);
	dwt_util_save_to_pgm_i("data2.pgm", 255, data2, stride_x, stride_y, x, y);

	// free allocated memory
	dwt_util_free_image(&data1);
	dwt_util_free_image(&data2);

	return 0;
}
