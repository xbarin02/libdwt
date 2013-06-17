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

	// level of decomposition (here -1 for full decomposition)
	int j = -1;

	// compute strides
	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(x * stride_y);

	// pointer to image data
	void *data1;

	// allocate image
	dwt_util_alloc_image(&data1, stride_x, stride_y, x, y);

	// create test image
	dwt_util_test_image_fill_s(data1, stride_x, stride_y, x, y, 0);

	// forward transform
	dwt_cdf97_2f_s(data1, stride_x, stride_y, x, y, x, y, &j, 0, 0);

	// inverse transform
	dwt_cdf97_2i_s(data1, stride_x, stride_y, x, y, x, y, j, 0, 0);

	// free allocated memory
	dwt_util_free_image(&data1);

	// release platform resources
	dwt_util_finish();

	return 0;
}
