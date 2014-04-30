/**
 * @brief Plot 2-D wavelet/scaling functions.
 */

#include "libdwt.h"
#include <assert.h>
#include <math.h>

int main()
{
	// settings
	const int log2_size = 9;
	const enum dwt_subbands subband = DWT_HL;

	const int size_x = 1<<log2_size;
	const int size_y = 1<<log2_size;
	dwt_util_log(LOG_INFO, "size=(%i,%i)\n", size_x, size_y);

	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(stride_y * size_x);

	// alloc transform
	void *ptr;
	dwt_util_alloc_image(&ptr, stride_x, stride_y, size_x, size_y);
	assert( ptr );

	// how many levels can be performed?
	int max_j = -1;
	dwt_cdf53_2f_dummy_s(ptr, stride_x, stride_y, size_x, size_y, size_x, size_y, &max_j, 0);
	dwt_util_log(LOG_INFO, "max_j = %i\n", max_j);

	// reset the transform
	dwt_util_test_image_zero_s(ptr, stride_x, stride_y, size_x, size_y);

	int j = max_j - 3;

	assert( j > 0 );

	// the subband
	void *subband_ptr;
	int subband_size_x;
	int subband_size_y;
	dwt_util_subband_s(ptr, stride_x, stride_y, size_x, size_y, size_x, size_y, j, subband, &subband_ptr, &subband_size_x, &subband_size_y);

	// one non-zero coeff.
	*dwt_util_addr_coeff_s(subband_ptr, subband_size_y/2, subband_size_x/2, stride_x, stride_y) = 1.f;

	// show transform
	void *show;
	dwt_util_alloc_image(&show, stride_x, stride_y, size_x, size_y);
	assert( show );
	dwt_util_conv_show_s(ptr, show, stride_x, stride_y, size_x, size_y);

	// save transform
	dwt_util_save_to_pgm_s("show.pgm", 1.0, show, stride_x, stride_y, size_x, size_y);

	// perform inverse transform
	dwt_cdf97_2i_s(ptr, stride_x, stride_y, size_x, size_y, size_x, size_y, j, 0, 0);

	// find extreme value
	float min, max;
	dwt_util_find_min_max_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		&min,
		&max
	);
	dwt_util_log(LOG_INFO, "min=%f max=%f\n", min, max);
	min = fabsf(min);
	max = fabsf(max);
	max = (max > min) ? max : min;
	dwt_util_log(LOG_INFO, "abs(max)=%f\n", max);

	// normalize into -0.5..0.5
	dwt_util_scale_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		1.f/2.f/max
	);

	// save in MATLAB format
	dwt_util_save_to_mat_s(
		"func.mat",
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y
	);

	dwt_util_find_min_max_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		&min,
		&max
	);
	dwt_util_log(LOG_INFO, "min=%f max=%f\n", min, max);

	// shift into 0..1
	dwt_util_shift_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		+1.f/2.f
	);

	// save to file
	dwt_util_save_to_pgm_s("func.pgm", 1.0, ptr, stride_x, stride_y, size_x, size_y);

	dwt_util_find_min_max_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		&min,
		&max
	);
	dwt_util_log(LOG_INFO, "min=%f max=%f\n", min, max);

	return 0;
}
