/**
 * @brief The single-loop core approach using the 4x4 core with the vertical vectorization.
 */
#include "libdwt.h"
#include "dwt-sym.h"

int main()
{
	dwt_util_set_num_threads(1);

	int size_x = 512;
	int size_y = 512;

	int clock_type = dwt_util_clock_autoselect();

	int stride_y = sizeof(float);
	int stride_x = dwt_util_get_opt_stride(stride_y * size_x);

	void *src = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);
	void *dst = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);

	dwt_util_test_image_fill2_s(src, stride_x, stride_y, size_x, size_y, 0, 1);
	dwt_util_save_to_pgm_s("src.pgm", 1.0, src, stride_x, stride_y, size_x, size_y);

	// in case of in-place transform
	dwt_util_test_image_fill2_s(dst, stride_x, stride_y, size_x, size_y, 0, 1);

	int j = 5;

	// forward
	dwt_clock_t start = dwt_util_get_clock(clock_type);
#if 0
	// reference, copy + in-place, single loop
	dwt_util_copy_s(src, dst, stride_x, stride_y, size_x, size_y);
	dwt_cdf97_2f_inplace_s(dst, stride_x, stride_y, size_x, size_y, size_x, size_y, &j, 1, 0);
#endif
#if 0
	// reference, in-place, single loop
	dwt_cdf97_2f_inplace_s(dst, stride_x, stride_y, size_x, size_y, size_x, size_y, &j, 1, 0);
#endif
#if 0
	// reference, in-place, separated loops
	dwt_cdf97_2f_inplace_sep_s(dst, stride_x, stride_y, size_x, size_y, size_x, size_y, &j, 1, 0);
#endif
#if 1
	// TODO: fix for small image sizes
	// in-place, core approach
	dwt_cdf97_2f_dl_4x4_s(dst, stride_x, stride_y, size_x, size_y, size_x, size_y, &j, 1, 0);
#endif
	dwt_clock_t stop = dwt_util_get_clock(clock_type);
	double secs = (stop - start)/(double)dwt_util_get_frequency(clock_type);
	dwt_util_log(LOG_INFO, "forward @ levels: %i; image: %f Mpel; total: %f secs; pel: %f nsecs\n", j, (float)size_x*size_y/1024/1024, secs, secs/size_x/size_y*1024*1024*1024);
	dwt_util_save_log_to_pgm_s("dwt.pgm", dst, stride_x, stride_y, size_x, size_y);

	// inverse (inplace)
	dwt_cdf97_2i_inplace_s(dst, stride_x, stride_y, size_x, size_y, size_x, size_y, j, 1, 0);
	dwt_util_save_to_pgm_s("dst.pgm", 1.0, dst, stride_x, stride_y, size_x, size_y);

	// compare
	if( dwt_util_compare2_destructive_s(
		dst,
		src,
		stride_x,
		stride_y,
		stride_x,
		stride_y,
		size_x,
		size_y
	) )
	{
		dwt_util_log(LOG_ERR, "images differ!\n");
	}
	else
	{
		dwt_util_log(LOG_INFO, "images equal :)\n");
	}

	dwt_util_save_to_pgm_s("err.pgm", 1.0, dst, stride_x, stride_y, size_x, size_y);

	dwt_util_free_image(&src);
	dwt_util_free_image(&dst);

	return 0;
}
