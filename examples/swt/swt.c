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
	const char *cls_path = (argc > 2) ? argv[2]
		: "data/target_classes.dat";

	const char *fv_svm_path = "data/fv.svm";

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

	int levels = 10;

	float mat_fvH[size_y*levels];
	float mat_fvL[size_y*levels];

	// TODO: for each spectrum
	for(int y = 0; y < size_y; y++)
	{
		// testing signal
		int size = size_x;
		float *x = dwt_util_addr_coeff_s(
			ptr,
			y,
			0,
			stride_x,
			stride_y
		);

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

#if 0
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
// 		);
#endif
		dwt_util_log(LOG_INFO, "y=%i: Extracting feature vectors...\n", y);

		float *fvH = &mat_fvH[y*levels];
		float *fvL = &mat_fvL[y*levels];

		for(int l = 0; l < levels; l++)
		{
			// extract from x?[l+1] => fv?[l]
			fvH[l] = dwt_util_band_med_s(
				xH[l+1],
				0,
				sizeof(float),
				size,
				1
			);

			fvL[l] = dwt_util_band_med_s(
				xL[l+1],
				0,
				sizeof(float),
				size,
				1
			);
		}

		dwt_util_log(LOG_INFO, "Extraction done\n");
	}

	// classes
	void *cls_ptr;
	int cls_size_x, cls_size_y;
	int cls_stride_x, cls_stride_y;

	// load classes as "int" matrix
	dwt_util_load_from_mat_i(cls_path, &cls_ptr, &cls_size_x, &cls_size_y, &cls_stride_x, &cls_stride_y);

	if( cls_ptr )
		dwt_util_log(LOG_INFO, "Loaded %i classes.\n", cls_size_y);
	else
		dwt_util_error("Unable to load classes.\n");

	// merge with classes to LIBSVM
	dwt_util_save_to_svm_s(
		fv_svm_path,
		// matrix of vectors
		mat_fvH,
		levels,
		size_y,
		sizeof(float)*levels,
		sizeof(float),
		// matrix of labels
		cls_ptr,
		cls_size_x,
		cls_size_y,
		cls_stride_x,
		cls_stride_y
	);

	dwt_util_free_image(&cls_ptr);
	dwt_util_free_image(&ptr);

	dwt_util_finish();

	return 0;
}