/**
 * @brief Stationary wavelet transform using convolution.
 */

#include "libdwt.h"
#include "util.h"

// low-pass
float g[9] = { +0.037828, -0.023849, -0.110624, +0.377403, +0.852699, +0.377403, -0.110624, -0.023849, +0.037828 };

// high-pass
float h[7] = { +0.064539, -0.040689, -0.418092, +0.788486, -0.418092, -0.040689, +0.064539 };

int main(int argc, char *argv[])
{
	const char *vec_path = (argc > 1) ? argv[1]
		: "data/spectra.dat";

	dwt_util_init();

	void *ptr;
	int size_x, size_y;
	int stride_x, stride_y;

	dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", vec_path);

	// load the data from a MAT file
	dwt_util_load_from_mat_s(vec_path, &ptr, &size_x, &size_y, &stride_x, &stride_y);

	if( ptr )
		dwt_util_log(LOG_INFO, "Loaded %i spectra of length of %i samples.\n", size_y, size_x);
	else
		dwt_util_error("Unable to load spectra.\n");

	// testing signal
	int size = size_x;
	int y = 10;
	float *x = dwt_util_addr_coeff_s(
		ptr,
		y,
		0,
		stride_x,
		stride_y
	);

#if 0
	dwt_util_log(LOG_DBG, "stride_y=%i stride_x=%i ptr=%p (is_aligned_16=%i) x=%p (is_aligned_16=%i)\n", stride_y, stride_x, ptr, dwt_util_is_aligned_16(ptr), x, dwt_util_is_aligned_16(x));

	// FIXME: cause SIGSEGV; function expect aligned memory pointer; probably the stride should be multiply by sizeof(xmm)
	dwt_util_unit_vec_s(x, size, 0);
#endif

	int levels = 7; // 5

	// store SWT here
	float xL[levels+1][size];
	float xH[levels+1][size];

	// copy x => xL[0]
	dwt_util_copy_vec_s(x, xL[0], size);

	// for
	for(int l = 0; l < levels; l++)
	{
		// filter xL[l] => xL[l+1] with g upsampled by factor 2^l
		dwt_util_convolve1_s(
			// output response
			xL[l+1],
			sizeof(float),
			size,
			size/2,
			// input signal
			xL[l],
			sizeof(float),
			size,
			size/2,
			// kernel
			g,
			sizeof(float),
			9,
			9/2,
			// parameters
			1,
			1<<l
		);

		// filter xL[l] => xH[l+1] with h upsampled by factor 2^l
		dwt_util_convolve1_s(
			// output response
			xH[l+1],
			sizeof(float),
			size,
			size/2,
			// input signal
			xL[l],
			sizeof(float),
			size,
			size/2,
			// kernel
			h,
			sizeof(float),
			7,
			7/2,
			// parameters
			1,
			1<<l
		);
	}

	dwt_util_save_to_mat_s(
		"x.mat",
		x,
		size,
		1,
		0,
		sizeof(float)
	);

	dwt_util_save_to_mat_s(
		"lo.mat",
		xL[levels],
		size,
		1,
		0,
		sizeof(float)
	);

	dwt_util_save_to_mat_s(
		"hi.mat",
		xH[levels],
		size,
		1,
		0,
		sizeof(float)
	);

	dwt_util_free_image(&ptr);

	dwt_util_finish();

	return 0;
}
