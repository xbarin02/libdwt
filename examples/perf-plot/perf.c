/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief More complex application measuring performance.
 */

#include "libdwt.h"

// PATH_MAX
#include <limits.h>
#ifndef microblaze
#include <linux/limits.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

FILE *fopen_data(int dir, int threads, int accel, int opt_stride, int j, int arr, int workers)
{
	const char *arr_name[] =
	{
		[DWT_ARR_SIMPLE] = "simple",
		[DWT_ARR_SPARSE] = "sparse",
		[DWT_ARR_PACKED] = "packed"
	};

	char file_name[PATH_MAX];

	snprintf(file_name, PATH_MAX,
		"data/"
		"dir=%s."
		"threads=%i."
		"accel=%i."
		"opt-stride=%i."
		"j=%i."
		"arr=%s."
		"workers=%i."
		"type=%s."
		"txt",
		dir>0 ? "fwd" : "inv",
		threads,
		accel,
		opt_stride,
		j,
		arr_name[arr],
		workers,
		"float"
       	);

	dwt_util_log(LOG_DBG, "%s\n", file_name);

	FILE *file;

	file = fopen(file_name, "w");

	if( NULL == file )
		dwt_util_abort();

	return file;
}

void do_test(int threads, int accel, int opt_stride, int j, int arr, int workers, int min_size, int max_size)
{
	FILE *file_fwd = fopen_data(+1, threads, accel, opt_stride, j, arr, workers);
	FILE *file_inv = fopen_data(-1, threads, accel, opt_stride, j, arr, workers);

	dwt_util_set_accel(accel);
	dwt_util_set_num_threads(threads);
	dwt_util_set_num_workers(workers);

	dwt_util_measure_perf_cdf97_2_s(
		arr, // arr type
		min_size, // min
		max_size, // max
		opt_stride, // opt_stride
		j, // decomposition level
		1, // 1
		0, // 0
		1, // the number of transforms in each test loop, avg. is calculated
#ifdef microblaze
		2,
#else
		5, // the number of test loops, minimum is selected
#endif
		dwt_util_clock_autoselect(), // timer type
		file_fwd,
		file_inv
	);

	fclose(file_fwd);
	fclose(file_inv);
}

int main()
{
	dwt_util_init();

	// measure from min_size up to max_size
	const int min_size = 1<<5;
	const int max_size = 1<<10;

	// how many threads are available
	const int max_threads = dwt_util_get_max_threads();

	// iterate over all these parameters
	int j, arr, opt_stride, accel, threads, workers;

#if 1
	// compare array types (2-D)
	j = -1;
	opt_stride = 2;
	accel = 9;
	workers = 1;
	threads = 1;
	for(arr = DWT_ARR_SIMPLE; arr <= DWT_ARR_PACKED; arr++)
	{
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size);
	}
#endif

#if 1
	// compare (non)optimal stride (2-D)
	j = 1;
	accel = 9;
	workers = 1;
	threads = 1;
	arr = DWT_ARR_PACKED;
	for(opt_stride = 0; opt_stride <= 7; opt_stride++)
	{
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size);
	}
#endif

#if 1
	// compare threads (2-D)
	j = 1;
	accel = 9;
	workers = 1;
	opt_stride = 2;
	arr = DWT_ARR_PACKED;
	for(threads = 1; threads <= max_threads; threads++)
	{
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size);
	}
#endif

#if 1
	// compare all acceleration types (2-D)
	opt_stride = 7;
	arr = DWT_ARR_PACKED;
	j = 1;
	threads = 1;
#ifdef microblaze
	/* microblaze */
	do_test(threads,  0, opt_stride, j, arr, 1, min_size, max_size); // ML
	do_test(threads,  1, opt_stride, j, arr, 1, min_size, max_size); // ML/BCEx1
	do_test(threads,  1, opt_stride, j, arr, 2, min_size, max_size); // ML/BCEx2
	do_test(threads,  4, opt_stride, j, arr, 1, min_size, max_size); // DL
	do_test(threads,  5, opt_stride, j, arr, 1, min_size, max_size); // SDL
#endif
#ifdef __x86_64__
	/* x86-64 */
	do_test(threads,  0, opt_stride, j, arr, 1, min_size, max_size); // ML
	do_test(threads,  9, opt_stride, j, arr, 1, min_size, max_size); // SDL6/SSE
	do_test(threads, 11, opt_stride, j, arr, 4, min_size, max_size); // DL4/SSE
	do_test(threads, 12, opt_stride, j, arr, 4, min_size, max_size); // ML4/SSE
	do_test(threads, 13, opt_stride, j, arr, 1, min_size, max_size); // ML/NOSSE
	do_test(threads, 16, opt_stride, j, arr, 1, min_size, max_size); // DL4LINE/SSE
#endif /* __x86_64__ */
#ifdef __arm__
	do_test(threads,  0, opt_stride, j, arr, 1, min_size, max_size); // ML
	do_test(threads,  4, opt_stride, j, arr, 1, min_size, max_size); // DL
	do_test(threads,  5, opt_stride, j, arr, 1, min_size, max_size); // SDL
	do_test(threads,  6, opt_stride, j, arr, 1, min_size, max_size); // SDL2
	do_test(threads,  7, opt_stride, j, arr, 1, min_size, max_size); // SDL6
	do_test(threads, 15, opt_stride, j, arr, 1, min_size, max_size); // DL4LINE
#endif
#endif

	dwt_util_finish();

	return 0;
}
