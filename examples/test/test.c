/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Test correct function of DWT transform functions.
 */

#include "libdwt.h"

int main()
{
	dwt_util_init();

	dwt_util_set_num_workers(4);

	const int x = 256, y = 256;

	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(stride_y * x);

	dwt_util_log(LOG_INFO, "Using %i threads.\n", dwt_util_get_num_threads());
	dwt_util_log(LOG_INFO, "Using %i workers.\n", dwt_util_get_num_workers());
	dwt_util_log(LOG_INFO, "Using image of size of %ix%i pixels.\n", x, y);

	for(int accel = 0; accel <= 16; accel++)
	{
		if( 2 == accel )
			continue;

		dwt_util_set_accel(accel);

		dwt_util_log(LOG_INFO, "Acceleration type is %i.\n", dwt_util_get_accel());

		int j = -1;

		int ret = dwt_util_test_cdf97_2_s(
			stride_x,
			stride_y,
			x,
			y,
			x,
			y,
			j,
			0,
			0
		);

		if( ret )
			dwt_util_log(LOG_INFO, "fail\n");
		else
			dwt_util_log(LOG_INFO, "success\n");
	}

	dwt_util_finish();

	return 0;
}
