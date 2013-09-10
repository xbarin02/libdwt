/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Calculate approximation of scaling/wavelet functions.
 */

#include "libdwt.h"
#include <math.h> // sqrtf
#include "util.h" // dwt_util_convolve1_s

int main()
{
	dwt_util_init();

	float dec_phi[9+2] = { +0.000000, +0.037828, -0.023849, -0.110624, +0.377403, +0.852699, +0.377403, -0.110624, -0.023849, +0.037828, +0.000000 };
	float dec_psi[7+2] =            { +0.000000, +0.064539, -0.040689, -0.418092, +0.788486, -0.418092, -0.040689, +0.064539, +0.000000 };
	float rec_phi[7+2] =            { +0.000000, -0.064539, -0.040689, +0.418092, +0.788486, +0.418092, -0.040689, -0.064539, +0.000000 };
	float rec_psi[9+2] = { -0.000000, +0.037828, +0.023849, -0.110624, -0.377403, +0.852699, -0.377403, -0.110624, +0.023849, +0.037828, -0.000000 };

#define FILTER rec_phi

	float *fir = FILTER;
	int size = sizeof(FILTER)/sizeof(*FILTER) - 2;

#if 0
	// scale coeffs
	dwt_util_scale_s(fir, size, 1, 0, sizeof(float), 1.f/sqrtf(2.f));
#endif

#if 1
	// another scale
	dwt_util_scale_s(fir, size, 1, 0, sizeof(float), sqrtf(2.f));
#endif

	dwt_util_log(LOG_INFO, "Using filter = %s with center=%i, 1-norm=%f, 2-norm=%f\n",
		dwt_util_str_vec_s(fir, size),
		size/2,
		dwt_util_band_lpnorm_s(fir, 0, sizeof(float), size, 1, /*p=*/1),
		dwt_util_band_lpnorm_s(fir, 0, sizeof(float), size, 1, /*p=*/2)
	);

	int l_size = 32; // long size, 32, 256
	int passes = 50; // 20, 3, 100, 200
	float a[passes][l_size];

	// unit impulse => a[0]
	dwt_util_unit_vec_s(a[0], l_size, 0);

	// print result
	dwt_util_log(LOG_DBG, "impulse: %s\n", dwt_util_str_vec_s(a[0], l_size));

	// a[p] <= a[p-1] (*) kernel
	for(int p = 1; p < passes; p++)
	{
		// convolve
		dwt_util_convolve1_s(
			// output response
			a[p],
			sizeof(float),
			l_size,
			l_size/2,
			// input signal
			a[p-1],
			sizeof(float),
			l_size,
			l_size/2,
			// kernel
			fir,
			sizeof(float),
			size,
			size/2+1,
			// parameters
			1, // downsample output
			2 // upsample kernel
		);

		// print result
		dwt_util_log(LOG_DBG, "phi(pass=%i): %s\n", p, dwt_util_str_vec_s(a[p], l_size));
	}

	dwt_util_save_to_mat_s(
		"func.mat",
		a[passes-1],
		l_size,
		1,
		0,
		sizeof(float)
	);

	dwt_util_finish();

	return 0;
}
