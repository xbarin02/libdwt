/**
 * @brief Stationary wavelet transform using convolution.
 */

#include "libdwt.h"
#include "util.h"

// low-pass
float g[9] = { +0.037828, -0.023849, -0.110624, +0.377403, +0.852699, +0.377403, -0.110624, -0.023849, +0.037828 };

// high-pass
float h[7] = { +0.064539, -0.040689, -0.418092, +0.788486, -0.418092, -0.040689, +0.064539 };

int main()
{
	dwt_util_init();

	// testing signal
	int size = 256;
	float x[size];

	// generate test signal
	dwt_util_unit_vec_s(x, size, 0);

	int levels = 5;

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
		"lo.mat",
		xL[levels],
		size,
		1,
		0,
		sizeof(float)
	);

	dwt_util_finish();

	return 0;
}
