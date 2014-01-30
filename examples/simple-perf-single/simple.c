/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Example application measuring transform performance.
 */

#include "libdwt.h"

int main()
{
	// init platform
	dwt_util_init();

	// use fast SSE implementation
	dwt_util_set_num_threads(1);
	dwt_util_set_num_workers(1);
	dwt_util_set_accel(0);

	// image size
	const int x = 1920, y = 1080;

	// size of picture element
	const int stride_y = sizeof(float);

	// compute optimal stride
	const int stride_x = dwt_util_get_opt_stride(stride_y * x);

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

	// level of decomposition
	int j = 1;

	dwt_util_log(LOG_INFO, "generating test images...\n");

	// create test images
	dwt_util_alloc_image(&data1, stride_x, stride_y, x, y);
	dwt_util_test_image_fill_s(data1, stride_x, stride_y, x, y, 0);
	dwt_util_alloc_image(&data2, stride_x, stride_y, x, y);
	dwt_util_test_image_fill_s(data2, stride_x, stride_y, x, y, 0);

	// init timer
	dwt_clock_t time_start;
	dwt_clock_t time_stop;
	const int type = dwt_util_clock_autoselect();

	dwt_util_log(LOG_INFO, "forward transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// forward transform
	dwt_cdf97_2f_inplace_s(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0);

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));


	dwt_util_log(LOG_INFO, "inverse transform...\n");

	// start timer
	time_start = dwt_util_get_clock(type);

	// inverse transform
	dwt_cdf97_2i_inplace_s(data1, stride_x, stride_y, x, y, x, y, j, 0, 0);

	// stop timer
	time_stop = dwt_util_get_clock(type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(time_stop - time_start) / dwt_util_get_frequency(type));

	// compare
	if( dwt_util_compare_s(data1, data2, stride_x, stride_y, x, y) )
		dwt_util_log(LOG_INFO, "images differs\n");
	else
		dwt_util_log(LOG_INFO, "success\n");
	
	// free allocated memory
	dwt_util_free_image(&data1);
	dwt_util_free_image(&data2);

	float fwd_secs, inv_secs;

	// more objective performance test
	dwt_util_perf_cdf97_2_inplace_s(
		stride_x, stride_y,
		x, y, x, y,
		j, 0, 0,
		4, 8,		// 4 transforms per test loop, 8 test loops
		type,
		&fwd_secs,
		&inv_secs);

	dwt_util_log(LOG_INFO, "performance test: fwd=%f secs (%f fps), inv=%f secs (%f fps)\n", fwd_secs, 1/fwd_secs, inv_secs, 1/inv_secs);

	// release platform resources
	dwt_util_finish();

	return 0;
}
