/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Wavelet power spectrum for stellar classification.
 */

#include "libdwt.h"

int main(int argc, char *argv[])
{
	dwt_util_init();

	const char *vec_path = (argc > 1) ? argv[1]
		: "data/spectra.dat";
	const char *cls_path = (argc > 2) ? argv[2]
		: "data/target_classes.dat";

	const char *vec_svm_path = "data/spectra.svm";
	const char *vec_pgm_path = "data/spectra.pgm";
	const char *vec_mat_path = "data/spectra.mat";

	const char *dwt_svm_path = "data/dwt.svm";
	const char *dwt_mat_path = "data/dwt.mat";
	const char *dwt_pgm_path = "data/dwt.pgm";

	const char *fv_svm_path = "data/fv.svm";
	const char *fv_mat_path = "data/fv.mat";

	// input/transformed data
	void *ptr;
	int size_x, size_y;
	int stride_x, stride_y;

	dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", vec_path);

	// load the data from a MAT file
	dwt_util_load_from_mat_s(vec_path, &ptr, &size_x, &size_y, &stride_x, &stride_y);

	if( ptr )
		dwt_util_log(LOG_INFO, "Loaded %i spectra of length of %i samples\n", size_y, size_x);
	else
		dwt_util_error("Unable to load spectra.\n");

	void *free_ptr = ptr;

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

#if 0
	dwt_util_log(LOG_INFO, "Shifting base-line by 1...\n");
	// shift base-line from "1" to "0"
	dwt_util_shift_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		-1.0f
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

#if 0
	// crop 256 samples around the center of each vector
	int crop_size_x = 256;

	dwt_util_log(LOG_INFO, "Using crop of %i samples around the center...\n", crop_size_x);

	ptr = dwt_util_crop21(ptr, size_x, size_y, stride_x, stride_y, crop_size_x);
	size_x = crop_size_x;
#endif

#if 0
	dwt_util_log(LOG_INFO, "Scaling...\n");
	// scale to <0;1> interval
	dwt_util_scale21_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		0.0f,
		1.0f
	);
#endif

	// classes
	void *cls_ptr;
	int cls_size_x, cls_size_y;
	int cls_stride_x, cls_stride_y;

	dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", cls_path);

	// load classes as "int" matrix
	dwt_util_load_from_mat_i(cls_path, &cls_ptr, &cls_size_x, &cls_size_y, &cls_stride_x, &cls_stride_y);

	if( cls_ptr )
		dwt_util_log(LOG_INFO, "Loaded %i classes\n", cls_size_y);
	else
		dwt_util_error("Unable to load classes.\n");

	dwt_util_log(LOG_INFO, "Saving into SVM-file...\n");

	// combine input_dwt and target_classes into LIBSVM format
	dwt_util_save_to_svm_s(
		vec_svm_path,
		// matrix of vectors
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		// matrix of labels
		cls_ptr,
		cls_size_x,
		cls_size_y,
		cls_stride_x,
		cls_stride_y
	);

	dwt_util_log(LOG_INFO, "Saving into PGM-file...\n");

	// save the loaded data as a PGM image (should be in 0..1 interval)
	dwt_util_save_to_pgm_s(vec_pgm_path, 1.0, ptr, stride_x, stride_y, size_x, size_y);

	dwt_util_log(LOG_INFO, "Saving into MAT-file...\n");

	// save centered spectra
	dwt_util_save_to_mat_s(vec_mat_path, ptr, size_x, size_y, stride_x, stride_y);

	// maximum levels of decomposition
	int j = -1;

	// series of 1-D transforms
#if 1
	dwt_cdf97_2f1_s(ptr, stride_x, stride_y, size_x, size_y, size_x, size_y, &j, 0);
#else
	dwt_cdf53_2f1_s(ptr, stride_x, stride_y, size_x, size_y, size_x, size_y, &j, 0);
#endif

	dwt_util_log(LOG_INFO, "DWT done, reached j=%i\n", j);

	dwt_util_log(LOG_INFO, "Saving DWT into SVM-file...\n");

	// save the DWT into LIBSVM format
	dwt_util_save_to_svm_s(
		dwt_svm_path,
		// matrix of vectors
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		// matrix of labels
		cls_ptr,
		cls_size_x,
		cls_size_y,
		cls_stride_x,
		cls_stride_y
	);

	dwt_util_log(LOG_INFO, "Saving DWT into MAT-file...\n");

	// save the transformed vectors into a MAT file
	dwt_util_save_to_mat_s(dwt_mat_path, ptr, size_x, size_y, stride_x, stride_y);

	dwt_util_log(LOG_INFO, "Saving DWT into PGM-file...\n");

	// save the transformed vectors as a PGM image
	void *show;
	dwt_util_alloc_image(&show, stride_x, stride_y, size_x, size_y);
	dwt_util_conv_show_s(ptr, show, stride_x, stride_y, size_x, size_y);
	dwt_util_save_to_pgm_s(dwt_pgm_path, 1.0, show, stride_x, stride_y, size_x, size_y);
	dwt_util_free_image(&show);

	// feature vectors
	void *fv;
	int fv_size_y = size_y;
	int fv_size_x = dwt_util_count_subbands_s(ptr, stride_x, stride_y, size_x, 1, size_x, 1, j);
	int fv_stride_y = sizeof(float);
	int fv_stride_x = dwt_util_get_opt_stride(fv_stride_y * fv_size_x);

	// allocate memory for the feature vectors
	dwt_util_alloc_image(&fv, fv_stride_x, fv_stride_y, fv_size_x, fv_size_y);

	dwt_util_log(LOG_INFO, "Computing feature vectors...\n");

	// for each transformed vector
	for(int y = 0; y < size_y; y++)
	{
		// single transformed vector
		const void *src = dwt_util_addr_coeff_const_s(ptr, y, 0, stride_x, stride_y);
		int src_x = size_x;
		int src_y = 1;

#if 0
		// optional: operate on magnitudes instead of original coefficient values
		dwt_util_abs_s(
			src,
			stride_x,
			stride_y,
			src_x,
			src_y
		);
#endif

		// extract the feature vector
		// available functions: wps, maxidx, mean, med, var, stdev, skew, kurt, norm, maxnorm
		dwt_util_wps_s(
			src,
			stride_x,
			stride_y,
			src_x,
			src_y,
			src_x,
			src_y,
			j,
			dwt_util_addr_coeff_s(fv, y, 0, fv_stride_x, fv_stride_y)
		);
	}

#if 0
	// scale to <0;1> interval
	dwt_util_scale21_s(
		fv,
		fv_size_x,
		fv_size_y,
		fv_stride_x,
		fv_stride_y,
		0.0f,
		1.0f
	);
#endif

	dwt_util_log(LOG_INFO, "Saving FV into MAT-file...\n");

	// save feature vectors into MAT-file
	dwt_util_save_to_mat_s(fv_mat_path, fv, fv_size_x, fv_size_y, fv_stride_x, fv_stride_y);

	dwt_util_log(LOG_INFO, "Saving FV into SVM-file...\n");

	// save these vectors into LIBSVM format
	dwt_util_save_to_svm_s(
		fv_svm_path,
		// matrix of vectors
		fv,
		fv_size_x,
		fv_size_y,
		fv_stride_x,
		fv_stride_y,
		// matrix of labels
		cls_ptr,
		cls_size_x,
		cls_size_y,
		cls_stride_x,
		cls_stride_y
	);

	// release resources
	dwt_util_free_image(&fv);
	dwt_util_free_image(&free_ptr);
	dwt_util_free_image(&cls_ptr);
	
	dwt_util_finish();

	dwt_util_log(LOG_INFO, "Done\n");

	return 0;
}
