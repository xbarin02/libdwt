/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Plot reconstruction scaling/wavelet functions using iterating the filter bank.
 */

#include "libdwt.h"

#if 1
void (*dwt_wt_1i_s)(void *, int, int, int, int, int) = dwt_cdf97_1i_s;
void (*dwt_wt_1f_s)(void *, int, int, int, int *, int)  = dwt_cdf97_1f_s;
#endif
#if 0
void (*dwt_wt_1i_s)(void *, int, int, int, int, int) = dwt_cdf53_1i_s;
void (*dwt_wt_1f_s)(void *, int, int, int, int *, int)  = dwt_cdf53_1f_s;
#endif
#if 0
void (*dwt_wt_1i_s)(void *, int, int, int, int, int) = dwt_interp53_1i_s;
void (*dwt_wt_1f_s)(void *, int, int, int, int *, int)  = dwt_interp53_1f_s;
#endif

int main()
{
	dwt_util_init();

	// length of function approximation
	int size = 4096;

	// put the function here
	float vec[size];

	int j = -1;

	// find max. j
	dwt_wt_1f_s(vec, sizeof(float), size, size, &j, 0);

	// target scale
	int jj = j - 3;

	// single nonzero coefficient at the scale jj
	float energy = 100.f;

	dwt_util_log(LOG_INFO, "for size=%i: max. j=%i, using jj=%i\n", size, j, jj);

	// for subband access
	void *subband_ptr;
	int subband_size_x;
	int subband_size_y;

	// L subband
	dwt_util_subband_s(vec, 0, sizeof(float), size, 1, size, 1, jj, DWT_LL, &subband_ptr, &subband_size_x, &subband_size_y);

	dwt_util_log(LOG_INFO, "using L_%i subband of size of %i coefficients...\n", jj, subband_size_x);

	// zero the transform
	dwt_util_zero_vec_s(vec, size);

	// put one non-zero value in center of subband
	((float *)subband_ptr)[subband_size_x/2] = energy;

	// iterate the filter bank
	dwt_wt_1i_s(vec, sizeof(float), size, size, jj, 0);

	// save scaling function
	dwt_util_save_to_mat_s("phi.mat", vec, size, 1, 0, sizeof(float));

	// H subband
	dwt_util_subband_s(vec, 0, sizeof(float), size, 1, size, 1, jj, DWT_HL, &subband_ptr, &subband_size_x, &subband_size_y);

	dwt_util_log(LOG_INFO, "using H_%i subband of size of %i coefficients...\n", jj, subband_size_x);

	// zero the transform
	dwt_util_zero_vec_s(vec, size);

	// put one non-zero value in center of subband
	((float *)subband_ptr)[subband_size_x/2] = energy;

	// iterate the filter bank
	dwt_wt_1i_s(vec, sizeof(float), size, size, jj, 0);

	// save scaling function
	dwt_util_save_to_mat_s("psi.mat", vec, size, 1, 0, sizeof(float));

	dwt_util_finish();

	return 0;
}
