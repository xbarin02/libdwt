/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Wavelet power spectrum for stellar classification.
 */

#include "libdwt.h"

int main(int argc, char *argv[])
{
	dwt_util_init();

	const char *path = (argc > 1) ? argv[1]
		: "data/input_dwt.dat";
	const char *cls_path = (argc > 2) ? argv[2]
		: "data/target_classes.dat";
	const char *svm_path = (argc > 3) ? argv[3]
		: "data/spectra.svm";

	// input/transformed data
	void *ptr;
	int size_x, size_y;
	int stride_x, stride_y;

	dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", path);

	// load the data from a MAT file
	dwt_util_load_from_mat_s(path, &ptr, &size_x, &size_y, &stride_x, &stride_y);

#if 1
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
	// center vectors
	for(int y = 0; y < size_y; y++)
	{
		dwt_util_center1_s(
			dwt_util_addr_coeff_s(
				ptr,
				y, // y
				0, // x
				stride_x,
				stride_y
			),
			size_x,
			stride_y,
			20
		);
	}
#endif

#if 0
	// scale to <0;1> interval
	dwt_util_scale2_s(
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

	// load classes as "int" matrix
	dwt_util_load_from_mat_i(cls_path, &cls_ptr, &cls_size_x, &cls_size_y, &cls_stride_x, &cls_stride_y);

	// combine input_dwt and target_classes into LIBSVM format
	dwt_util_save_to_svm_s(
		svm_path,
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

	// save the loaded data as a PGM image (should be in 0..1 interval)
	dwt_util_save_to_pgm_s("data/input_dwt.pgm", 1.0, ptr, stride_x, stride_y, size_x, size_y);

	// maximum levels of decomposition
	int j = -1;

	// series of 1-D transforms
	dwt_cdf97_2f1_s(ptr, stride_x, stride_y, size_x, size_y, size_x, size_y, &j, 0);

	// save the transformed vectors into a MAT file
	dwt_util_save_to_mat_s("data/dwt.dat", ptr, size_x, size_y, stride_x, stride_y);

	// save the transformed vectors as a PGM image
	void *show;
	dwt_util_alloc_image(&show, stride_x, stride_y, size_x, size_y);
	dwt_util_conv_show_s(ptr, show, stride_x, stride_y, size_x, size_y);
	dwt_util_save_to_pgm_s("data/dwt.pgm", 1.0, show, stride_x, stride_y, size_x, size_y);
	dwt_util_free_image(&show);

	// feature vectors
	void *fv;
	int fv_size_y = size_y;
	int fv_size_x = dwt_util_count_subbands_s(ptr, stride_x, stride_y, size_x, 1, size_x, 1, j);
	int fv_stride_y = sizeof(float);
	int fv_stride_x = dwt_util_get_opt_stride(fv_stride_y * fv_size_x);

	// allocate memory for the feature vectors
	dwt_util_alloc_image(&fv, fv_stride_x, fv_stride_y, fv_size_x, fv_size_y);

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
	dwt_util_scale2_s(
		fv,
		fv_size_x,
		fv_size_y,
		fv_stride_x,
		fv_stride_y,
		0.0f,
		1.0f
	);
#endif

	// save feature vectors into MAT-file
	dwt_util_save_to_mat_s("data/fv.dat", fv, fv_size_x, fv_size_y, fv_stride_x, fv_stride_y);

	// save these vectors into LIBSVM format
	dwt_util_save_to_svm_s(
		"data/fv.svm",
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
	dwt_util_free_image(&ptr);
	dwt_util_free_image(&cls_ptr);
	
	dwt_util_finish();

	return 0;
}
