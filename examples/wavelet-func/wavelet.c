/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Calculate approximation of scaling/wavelet functions.
 */

#include "libdwt.h"
#include <math.h> // sqrtf
#include "util.h" // dwt_util_convolve1_s

// gnuplot 1-D
int dwt_util_save1_to_gp_s(
	const char *path,	///< target file name, e.g. "output.dat"
	const void *data,	///< pointer to beginning of signal data
	int size_x,		///< width of nested/inner signal (in elements)
	int stride_y		///< difference between columns (in bytes)
)
{
	// FIXME: assert

	FILE *file = fopen(path, "w");

	if( NULL == file )
		return 1;

	int symb_delim[] = { ',', ';', '\t', ' ' };
	int symb_newline[] = { '\n', '\r' };

	for(int x = 0; x < size_x; x++)
	{
		const float val_x = (float)x;
		const float val_y = *dwt_util_addr_coeff_const_s(
			data,
			0,
			x,
			0,
			stride_y
		);

		fprintf(file, "%f%c%f%c", val_x, symb_delim[2], val_y, symb_newline[0]);
	}

	fclose(file);

	return 0;
}

int main()
{
	dwt_util_init();

	float cdf97_dec_phi[9] = { +0.037828, -0.023849, -0.110624, +0.377403, +0.852699, +0.377403, -0.110624, -0.023849, +0.037828 };
	float cdf97_dec_psi[7] =            { +0.064539, -0.040689, -0.418092, +0.788486, -0.418092, -0.040689, +0.064539 };
	float cdf97_rec_phi[7] =            { -0.064539, -0.040689, +0.418092, +0.788486, +0.418092, -0.040689, -0.064539 };
	float cdf97_rec_psi[9] = { +0.037828, +0.023849, -0.110624, -0.377403, +0.852699, -0.377403, -0.110624, +0.023849, +0.037828 };

	float cdf53_dec_phi[5] = { -0.17677669, +0.35355338, +1.06066012, +0.35355338, -0.17677669 };
	float cdf53_dec_psi[3] =              { -0.35355338, +0.70710677, -0.35355338 };
	float cdf53_rec_phi[3] =              { +0.35355338, +0.70710677, +0.35355338 };
	float cdf53_rec_psi[5] = { -0.17677669, -0.35355338, +1.06066012, -0.35355338, -0.17677669 };

	float interp53_dec_phi[1] =              { +1.41421354 };
	float interp53_dec_psi[3] = { -0.35355338, +0.70710677, -0.35355338 };
	float interp53_rec_phi[3] = { +0.35355338, +0.70710677, +0.35355338 };
	float interp53_rec_psi[1] =              { +1.41421354 };

	float interp2_dec_phi[1] = { +1.41421354 };
	float interp2_dec_psi[7] = { -0.07071068, +0.00000000, -0.28284273, +0.70710677, -0.28284273, +0.00000000, -0.07071068 };
	float interp2_rec_phi[7] = { +0.07071068, +0.00000000, +0.28284273, +0.70710677, +0.28284273, +0.00000000, +0.07071068 };
	float interp2_rec_psi[1] = { +1.41421354 };

#define FILTER_PHI cdf97_dec_phi
#define FILTER_PSI cdf97_dec_psi

	// scale coefficients
	dwt_util_scale_s(FILTER_PHI, sizeof_arr(FILTER_PHI), 1, 0, sizeof(float), sqrtf(2.f));
	dwt_util_scale_s(FILTER_PSI, sizeof_arr(FILTER_PSI), 1, 0, sizeof(float), sqrtf(2.f));

	dwt_util_log(LOG_INFO, "Using phi filter = %s with center=%i, 1-norm=%f, 2-norm=%f\n",
		dwt_util_str_vec_s(FILTER_PHI, sizeof_arr(FILTER_PHI)),
		sizeof_arr(FILTER_PHI)/2,
		dwt_util_band_lpnorm_s(FILTER_PHI, 0, sizeof(float), sizeof_arr(FILTER_PHI), 1, /*p=*/1),
		dwt_util_band_lpnorm_s(FILTER_PHI, 0, sizeof(float), sizeof_arr(FILTER_PHI), 1, /*p=*/2)
	);

	// configuration
	int density = 128;
	int size = 3*sizeof_arr(FILTER_PHI)*density;
	int passes = 12;

	// store approximations here
	float a[passes][size];

	// unit impulse as initial iteration
	dwt_util_unit_vec_s(a[0], size, 0);

	// a[p] <= a[p-1] (*) kernel
	for(int p = 1; p < passes-1; p++)
	{
		// convolve
		dwt_util_convolve1_s(
			// output response
			a[p],
			sizeof(float),
			size,
			size/2,
			// input signal
			a[p-1],
			sizeof(float),
			size,
			size/2,
			// kernel
			FILTER_PHI,
			sizeof(float),
			sizeof_arr(FILTER_PHI),
			sizeof_arr(FILTER_PHI)/2,
			// parameters
			2, // downsample output
			density // upsample kernel
		);
	}

	// last pass
	int p = passes-1;
	{
		dwt_util_convolve1_s(
			// output response
			a[p],
			sizeof(float),
			size,
			size/2,
			// input signal
			a[p-1],
			sizeof(float),
			size,
			size/2,
			// kernel
#if 1
			FILTER_PSI,
			sizeof(float),
			sizeof_arr(FILTER_PSI),
			sizeof_arr(FILTER_PSI)/2,
#else
			FILTER_PHI,
			sizeof(float),
			sizeof_arr(FILTER_PHI),
			sizeof_arr(FILTER_PHI)/2,
#endif
			// parameters
			2, // downsample output
			density // upsample kernel
		);
	}

	dwt_util_save_to_mat_s(
		"func.mat",
		a[passes-1],
		size,
		1,
		0,
		sizeof(float)
	);

	dwt_util_save1_to_gp_s(
		"func.dat",
		a[passes-1],
		size,
		sizeof(float)
	);

	dwt_util_finish();

	return 0;
}
