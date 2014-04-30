/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief More complex application measuring performance.
 */

#include "libdwt.h"
#include "dwt.h"
#include "dwt-core.h"

// PATH_MAX
#include <limits.h>
#ifndef microblaze
#include <linux/limits.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

FILE *fopen_data(int dir, int opt_stride, int j, int core, enum order order, int strip_x, int strip_y)
{
	char file_name[PATH_MAX];

	snprintf(file_name, PATH_MAX,
		"data/"
		"dir=%s."
		"opt-stride=%i."
		"j=%i."
		"core=%c."
		"order=%i."
		"stripX=%i."
		"stripY=%i."
		"txt",
		dir>0 ? "f" : "i",
		opt_stride,
		j,
		(char)core,
		(int)order,
		strip_x,
		strip_y
       	);

	dwt_util_log(LOG_DBG, "%s\n", file_name);

	FILE *file;

	file = fopen(file_name, "w");

	if( NULL == file )
		dwt_util_abort();

	return file;
}

typedef void(*func_t)(enum dwt_array, int, int, int, int, int, int, int, int, int, FILE *, FILE *);

void do_test(int min_size, int max_size, int opt_stride, int j, int core, enum order order, int strip_x, int strip_y)
{
	FILE *file_fwd = fopen_data(+1, opt_stride, j, core, order, strip_x, strip_y);
	FILE *file_inv = fopen_data(-1, opt_stride, j, core, order, strip_x, strip_y);

	dwt_util_set_accel(0);
	dwt_util_set_num_threads(1);
	dwt_util_set_num_workers(1);

	int N = 10; // the number of test loops, minimum is selected
	int M = 1; // the number of transforms in each test loop, avg. is calculated

	switch(core)
	{
		case 'D':
			dwt_util_measure_perf_cdf97_2_inplace_new_s(
				min_size,
				max_size,
				opt_stride,
				M,
				N,
				dwt_util_clock_autoselect(),
				file_fwd,
				file_inv,
				order,
				strip_x,
				strip_y
			);
			break;
		case 'V':
			dwt_util_measure_perf_cdf97_2_inplace_new_VERT_s(
				min_size,
				max_size,
				opt_stride,
				M,
				N,
				dwt_util_clock_autoselect(),
				file_fwd,
				file_inv,
				order,
				strip_x,
				strip_y
			);
			break;
		default:
			dwt_util_abort();
	}

	fclose(file_fwd);
	fclose(file_inv);
}

int main()
{
	dwt_util_init();

	// measure from min_size up to max_size
	const int min_size = 1<<5;
	const int max_size = 1<<13; // FIXME: 13, 12, 11, 10, 9, 8

	int j = 1;
#if 0
	for(int opt_stride = 0; opt_stride <= 7; opt_stride++)
#else
	int opt_stride = 1;
#endif
	{
		// full
		{
			do_test(min_size, max_size, opt_stride, j, 'D', ORDER_HORIZ, 0, 0);
			do_test(min_size, max_size, opt_stride, j, 'D', ORDER_VERT,  0, 0);
			do_test(min_size, max_size, opt_stride, j, 'V', ORDER_HORIZ, 0, 0);
			do_test(min_size, max_size, opt_stride, j, 'V', ORDER_VERT,  0, 0);
		}

		// strips
		for(int strip = 2; strip <= max_size>>1; strip *= 2)
		{
			do_test(min_size, max_size, opt_stride, j, 'D', ORDER_HORIZ_STRIPS, strip, strip);
			do_test(min_size, max_size, opt_stride, j, 'D', ORDER_VERT_STRIPS,  strip, strip);
			do_test(min_size, max_size, opt_stride, j, 'V', ORDER_HORIZ_STRIPS, strip, strip);
			do_test(min_size, max_size, opt_stride, j, 'V', ORDER_VERT_STRIPS,  strip, strip);
		}

		// blocks
#if 1
		for(int strip_x = 2; strip_x <= max_size>>3; strip_x *= 2)
		for(int strip_y = 2; strip_y <= max_size>>3; strip_y *= 2)
		{
#else
		for(int strip = 2; strip <= max_size>>1; strip *= 2)
		{
			int strip_x = strip;
			int strip_y = strip;
#endif
			do_test(min_size, max_size, opt_stride, j, 'D', ORDER_HORIZ_BLOCKS, strip_x, strip_y);
			do_test(min_size, max_size, opt_stride, j, 'D', ORDER_VERT_BLOCKS,  strip_x, strip_y);
			do_test(min_size, max_size, opt_stride, j, 'V', ORDER_HORIZ_BLOCKS, strip_x, strip_y);
			do_test(min_size, max_size, opt_stride, j, 'V', ORDER_VERT_BLOCKS,  strip_x, strip_y);
		}
	}

	dwt_util_finish();

	return 0;
}
