#include "libdwt.h"
#include "dwt.h"

int main()
{
	dwt_util_init();

	{
		const int N = 10;

		// DWT_ALG_SL_CORE_SDL_SSE DWT_ALG_SL_CORE_SDL DWT_ALG_SL_CORE_SDL_SC_SSE
		// DWT_ALG_SL_CORE_DL DWT_ALG_SL_CORE_DL_SSE DWT_ALG_SL_CORE_DL_SC DWT_ALG_SL_CORE_DL_SC_SSE
		enum dwt_alg alg = DWT_ALG_SL_CORE_SDL_SC_SSE;

		float fwd, inv;

		int edge_size = 128; // 128, 142, 502, 1024, 2024

		int img_size_x = dwt_util_to_even(edge_size);
		int img_size_y = dwt_util_to_even(edge_size);

		/*
		 * 1 = all OK, only C is slower, differences are significant
		 * 7 = all faster, C is correct, differences are small
		 */
		int opt_stride = 1;

		dwt_util_log(LOG_INFO, "edge=%i stride=%i alg=%i\n\n", edge_size, opt_stride, (int)alg);
#if 1
		dwt_util_log(LOG_INFO, "perf. old-inplace DL:\n");

		dwt_util_perf_cdf97_2_inplace_sep_s(
			dwt_util_get_stride(sizeof(float) * img_size_x, opt_stride),
			sizeof(float),
			img_size_x,
			img_size_y,
			img_size_x,
			img_size_y,
			1,
			0,
			0,
			1,
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. old-inplace SDL:\n");

		dwt_util_perf_cdf97_2_inplace_sep_sdl_s(
			dwt_util_get_stride(sizeof(float) * img_size_x, opt_stride),
			sizeof(float),
			img_size_x,
			img_size_y,
			img_size_x,
			img_size_y,
			1,
			0,
			0,
			1,
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. old-inplace SL.DL:\n");

		dwt_util_perf_cdf97_2_inplace_s(
			dwt_util_get_stride(sizeof(float) * img_size_x, opt_stride),
			sizeof(float),
			img_size_x,
			img_size_y,
			img_size_x,
			img_size_y,
			1,
			0,
			0,
			1,
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. old-inplace SL.SDL:\n");

		dwt_util_perf_cdf97_2_inplace_sdl_s(
			dwt_util_get_stride(sizeof(float) * img_size_x, opt_stride),
			sizeof(float),
			img_size_x,
			img_size_y,
			img_size_x,
			img_size_y,
			1,
			0,
			0,
			1,
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);
#endif
		dwt_util_log(LOG_INFO, "perf. inplace A:\n");

		dwt_util_perf_cdf97_2_inplace_alg_s(
			alg,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. inplace B:\n");

		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			alg,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv,
			1, // flush
			0 // overlap
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. inplace C:\n");

		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			alg,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv,
			1, // flush
			1 // overlap
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. inplace B (offset 0 + %i threads) SDL/SSE:\n", dwt_util_get_num_threads());

		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv,
			1, // flush
			0 // overlap
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. inplace B (offset 1 + %i threads) SDL/SSE:\n", dwt_util_get_num_threads());

		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv,
			1, // flush
			0 // overlap
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);


		dwt_util_log(LOG_INFO, "perf. inplace C (offset 0 + %i threads) SDL/SSE:\n", dwt_util_get_num_threads());

		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			DWT_ALG_SL_CORE_SDL_SC_SSE_OFF0_OVL1,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv,
			1, // flush
			1 // overlap
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. inplace C (offset 1 + %i threads) SDL/SSE:\n", dwt_util_get_num_threads());

		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			DWT_ALG_SL_CORE_SDL_SC_SSE_OFF1_OVL1,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv,
			1, // flush
			1 // overlap
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);

		dwt_util_log(LOG_INFO, "perf. inplace B (offset 1 + %i threads) DL/SSE:\n", dwt_util_get_num_threads());

		dwt_util_perf_cdf97_2_inplaceB_alg_s(
			DWT_ALG_SL_CORE_DL_SC_SSE_OFF1,
			img_size_x,
			img_size_y,
			opt_stride,
			1, // M
			N,
			dwt_util_clock_autoselect(),
			&fwd,
			&inv,
			1, // flush
			0 // overlap
		);

		dwt_util_log(LOG_INFO, "%f\n", fwd);
	}

	dwt_util_finish();

	return 0;
}
