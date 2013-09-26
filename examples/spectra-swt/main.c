/**
 * @brief Stationary wavelet transform using convolution.
 */

#include "libdwt.h"
#include "swt.h"

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

#if 1
	dwt_util_log(LOG_INFO, "Shifting base-line by median...\n");
	dwt_util_shift21_med_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y
	);
#endif

#if 1
	dwt_util_log(LOG_INFO, "Centering...\n");
	// center vectors
	dwt_util_center21_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		20
	);
#endif

	int levels = 10;

	float mat_fvH[size_y*levels];
	float mat_fvL[size_y*levels];

	// for each spectrum
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
#if 1
			swt_cdf97_f_ex_stride_s(
#else
			swt_cdf53_f_ex_stride_s(
#endif
				xL[l],
				xL[l+1],
				xH[l+1],
				size,
				sizeof(float),
				l
			);
		}

#if 0
		if( 1000 == y )
		{
			int l = 6; // levels
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
				xL[l],
				size,
				1,
				0,
				sizeof(float)
			);

			dwt_util_save_to_mat_s(
				"hi.mat",
				xH[l],
				size,
				1,
				0,
				sizeof(float)
			);
		}
#endif
		float *fvH = &mat_fvH[y*levels];
		float *fvL = &mat_fvL[y*levels];

#define _FE_FNC(FE) dwt_util_band_##FE##_s
#define FE_FNC _FE_FNC(med)

		for(int l = 0; l < levels; l++)
		{
			// extract from x?[l+1] => fv?[l]
			fvH[l] = FE_FNC(
				xH[l+1],
				0,
				sizeof(float),
				size,
				1
			);

			fvL[l] = FE_FNC(
				xL[l+1],
				0,
				sizeof(float),
				size,
				1
			);
		}
	}

	dwt_util_log(LOG_INFO, "Extraction done\n");

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
