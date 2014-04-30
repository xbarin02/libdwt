/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief More complex application measuring performance.
 */

#include "libdwt.h"
#include "dwt.h"

// mkdir
#include <sys/stat.h>
#include <sys/types.h>

// PATH_MAX
#include <limits.h>
#ifndef microblaze
#include <linux/limits.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

FILE *fopen_data(int dir, int threads, int accel, int opt_stride, int j, int arr, int workers, int inplace)
{
	mkdir("data", S_IRWXU);

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
		"inplace=%i."
		"txt",
		dir>0 ? "fwd" : "inv",
		threads,
		accel,
		opt_stride,
		j,
		arr_name[arr],
		workers,
		"float",
		inplace
       	);

	dwt_util_log(LOG_DBG, "%s\n", file_name);

	FILE *file;

	file = fopen(file_name, "w");

	if( NULL == file )
		dwt_util_abort();

	return file;
}

typedef void(*func_t)(enum dwt_array, int, int, int, int, int, int, int, int, int, FILE *, FILE *);

void do_test(int threads, int accel, int opt_stride, int j, int arr, int workers, int min_size, int max_size, int inplace)
{
	FILE *file_fwd = fopen_data(+1, threads, accel, opt_stride, j, arr, workers, inplace);
	FILE *file_inv = fopen_data(-1, threads, accel, opt_stride, j, arr, workers, inplace);

	dwt_util_set_accel(accel);
	dwt_util_set_num_threads(threads);
	dwt_util_set_num_workers(workers);

	func_t func[5] = {
		[0] = dwt_util_measure_perf_cdf97_2_s, // using accel_type
		[1] = dwt_util_measure_perf_cdf97_2_inplace_s, // SL.DL+SSE
		[2] = dwt_util_measure_perf_cdf97_2_inplace_sdl_s, // SL.SDL+SSE
		[3] = dwt_util_measure_perf_cdf97_2_inplace_sep_s, // DL
		[4] = dwt_util_measure_perf_cdf97_2_inplace_sep_sdl_s // SDL+SSE
	};

	int N = 10; // 10, 5, 1

	if( inplace < 5 )
	{
		func[inplace](
			arr, // arr type
			min_size, // min
			max_size, // max
			opt_stride, // opt_stride
			j, // decomposition level
			1, // 1
			0, // 0
			1, // the number of transforms in each test loop, avg. is calculated
			N, // the number of test loops, minimum is selected
			dwt_util_clock_autoselect(), // timer type
			file_fwd,
			file_inv
		);
	}
	else
	{
		// TODO: 0/B should be best choice
		int overlap = 0;

		enum dwt_alg alg;

		switch(inplace)
		{
			case 5: // SL/CORE SDL SSE
				alg = DWT_ALG_SL_CORE_SDL_SSE;
				break;
			case 6: // SL/CORE SDL SC/SSE
				alg = DWT_ALG_SL_CORE_SDL_SC_SSE;
				break;
			case 7: // SL/CORE DL
				alg =  DWT_ALG_SL_CORE_DL;
				break;
			case 8: // SL/CORE DL SC/SSE
				alg =  DWT_ALG_SL_CORE_DL_SC_SSE;
				break;
			case 9: // SL/CORE SDL SC/SSE THREADS
				alg = DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1;
				break;
			case 10: // SL/CORE DL ??? THREADS
				alg = DWT_ALG_SL_CORE_DL_SC_SSE_OFF1;
				break;
			case 11:
				alg = DWT_ALG_SL_CORE_DL_SC_SSE_OFF1_4X4;
				break;
			case 12:
				alg = DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1_6X2;
				break;
			default:
				dwt_util_abort();
		}

		dwt_util_measure_perf_cdf97_2_inplaceABC_alg_s(
			alg,
			min_size,
			max_size,
			opt_stride,
			1,
			N,
			dwt_util_clock_autoselect(),
			file_fwd,
			file_inv,
			overlap // -1=A 0=B 1=C
		);
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

	const int max_threads = 4; // 2, 4, 8

	dwt_util_log(LOG_INFO, "Testing %i..%i kpel using up to %i threads...\n", (min_size*min_size)>>10, (max_size*max_size)>>10, max_threads);

	// iterate over all these parameters
	int j, arr, opt_stride, threads, accel, workers;

	// compare all acceleration types (2-D)
	opt_stride = 1; // FIXME: 7, 1
	arr = DWT_ARR_PACKED;
	j = 1;
	accel = 0;
	workers = 1;

	for(threads = 1; threads <= max_threads; threads++)
	{
#if 0
		// accel_type
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 0);
#endif
		// naive + single-loop
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 1);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 2);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 3);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 4);
		// core
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 5);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 6);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 7);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 8);
		// threads
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 9);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 10);
		// threads + super-cores
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 11);
		do_test(threads, accel, opt_stride, j, arr, workers, min_size, max_size, 12);
	}

	dwt_util_finish();

	return 0;
}
