/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Print h/g filters coefficients.
 */

#include "libdwt.h"
#include <stdlib.h> // free
#include <math.h> // sqrtf

#if 1
void (*dwt_wt_1f_s)(void *, int, int, int, int *, int)  = dwt_cdf97_1f_s;
#endif
#if 0
void (*dwt_wt_1f_s)(void *, int, int, int, int *, int)  = dwt_cdf53_1f_s;
#endif
#if 0
void (*dwt_wt_1f_s)(void *, int, int, int, int *, int)  = dwt_interp53_1f_s;
#endif
#if 0
void (*dwt_wt_1f_s)(void *, int, int, int, int *, int)  = dwt_interp2_1f_s;
#endif

// synthesis (reconstruction)
//#define PRIM

int main()
{
	dwt_util_init();

	// for CDF 9/7
	int size = 12; // FIXME: up to 20

	float *vec = dwt_util_allocate_vec_s(size);

	// analysis (decomposition) coefficients
	float lo[size];
	float hi[size];

	int j = 1;

	// even coefficients

	dwt_util_unit_vec_s(vec, size, 0);

	dwt_wt_1f_s(vec, sizeof(float), size, size, &j, 0);

	for(int i=0; i<size/2; i++)
	{
		lo[2*i+0] = (vec+0)[i];
		hi[2*i+0] = (vec+size/2)[i];
#ifdef PRIM
		hi[2*i+0] *= -1;
#endif
	}

	// odd coefficients

	dwt_util_unit_vec_s(vec, size, -1);

	dwt_wt_1f_s(vec, sizeof(float), size, size, &j, 0);

	for(int i=0; i<size/2; i++)
	{
		lo[2*i+1] = (vec+0)[i];
		hi[2*i+1] = (vec+size/2)[i];
#ifdef PRIM
		lo[2*i+1] *= -1;
#endif
	}

#if 0
	// scale coeffs
	dwt_util_scale_s(lo, size, 1, 0, sizeof(float), 1.f/sqrtf(2.f));
	dwt_util_scale_s(hi, size, 1, 0, sizeof(float), 1.f/sqrtf(2.f));
#endif

	dwt_util_log(LOG_INFO, "lo = %s, center=%i, norm=%f\n", dwt_util_str_vec_s(lo, size), size/2,
		dwt_util_band_lpnorm_s(lo, 0, sizeof(float), size, 1, /*p=*/2) );
	dwt_util_log(LOG_INFO, "hi = %s, center=%i, norm=%f\n", dwt_util_str_vec_s(hi, size), size/2-1,
		dwt_util_band_lpnorm_s(hi, 0, sizeof(float), size, 1, /*p=*/2) );

	dwt_util_save_to_mat_s(
		"lo.mat",
		lo,
		size,
		1,
		0,
		sizeof(float)
	);

	dwt_util_save_to_mat_s(
		"hi.mat",
		hi,
		size,
		1,
		0,
		sizeof(float)
	);

	free(vec);
	
	dwt_util_finish();

	return 0;
}
