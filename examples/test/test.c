/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Test correct function of DWT transform functions.
 */

#include "libdwt.h"

int main()
{
	dwt_util_init();

#ifdef microblaze
	dwt_util_set_num_workers(2);
#else
	dwt_util_set_num_workers(4);
#endif

	const int x = 256, y = 256;

	dwt_util_log(LOG_INFO, "Using %i threads.\n", dwt_util_get_num_threads());
	dwt_util_log(LOG_INFO, "Using %i workers.\n", dwt_util_get_num_workers());
	dwt_util_log(LOG_INFO, "Using image of size of %ix%i pixels.\n", x, y);

	int j = -1;

	dwt_util_log(LOG_INFO, "Testing: CDF 9/7, 2D, float, in-place...\n");

	for(int accel = 0; accel <= 16; accel++)
	{
		if( 2 == accel )
			continue;

		dwt_util_set_accel(accel);

		dwt_util_log(LOG_INFO, "Acceleration type is %i.\n", dwt_util_get_accel());

		if( dwt_util_test2_cdf97_2_s(DWT_ARR_SIMPLE, x, y, 1, j, 1) )
			dwt_util_log(LOG_INFO, "fail\n");
		else
			dwt_util_log(LOG_INFO, "success\n");
	}

	dwt_util_log(LOG_INFO, "Testing: CDF 9/7, 2D, float, out-of-place...\n");

	for(int accel = 0; accel <= 16; accel++)
	{
		if( 2 == accel )
			continue;

		dwt_util_set_accel(accel);

		dwt_util_log(LOG_INFO, "Acceleration type is %i.\n", dwt_util_get_accel());

		if( dwt_util_test2_cdf97_2_s2(DWT_ARR_SIMPLE, x, y, 1, j, 1) )
			dwt_util_log(LOG_INFO, "fail\n");
		else
			dwt_util_log(LOG_INFO, "success\n");
	}

	dwt_util_log(LOG_INFO, "Testing: CDF 9/7, 2D, double, in-place...\n");

	if( dwt_util_test2_cdf97_2_d(DWT_ARR_SIMPLE, x, y, 1, j, 1) )
		dwt_util_log(LOG_INFO, "fail\n");
	else
		dwt_util_log(LOG_INFO, "success\n");

	dwt_util_log(LOG_INFO, "Testing: CDF 9/7, 2D, int, in-place...\n");

	if( dwt_util_test2_cdf97_2_i(DWT_ARR_SIMPLE, x, y, 1, j, 1) )
		dwt_util_log(LOG_INFO, "fail\n");
	else
		dwt_util_log(LOG_INFO, "success\n");


	dwt_util_finish();

	return 0;
}
