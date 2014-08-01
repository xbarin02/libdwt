/**
 * @brief Calculate averages of S transforms in each class.
 */
#include "libdwt.h"
#include "gabor.h" // gabor_st_s
#include "spectra.h" // spectra_load
#include "image.h" // struct image_t
#include "../spectra-blobs/spectra-experimental.h"

static
const char *simple_sprintf(const char *format, ...)
{
	static char buffer[4096];

	va_list ap;

	va_start(ap, format);
	int n = vsnprintf(buffer, 4096, format, ap);
	va_end(ap);

	if( n < 0 )
		dwt_util_error("vsnprintf returned negative value!\n");

	return buffer;
}

int main(int argc, char *argv[])
{
	dwt_util_init();

	const char *vec_path = (argc > 1) ? argv[1]
		: "data/spectra.dat";

	struct image_t spectra;

	spectra.ptr = spectra_load(
		vec_path,
		&spectra.stride_x,
		&spectra.stride_y,
		&spectra.size_x,
		&spectra.size_y
	);

	if( spectra.ptr )
		dwt_util_log(LOG_INFO, "Loaded %i spectra of length of %i samples.\n", spectra.size_y, spectra.size_x);
	else
		dwt_util_error("Unable to load spectra\n");

	const char *cls_path = (argc > 2) ? argv[2]
		: "data/classes.dat";

	struct image_t cls;

	// load classes
	dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", cls_path);

	dwt_util_load_from_mat_i(cls_path, &cls.ptr, &cls.size_x, &cls.size_y, &cls.stride_x, &cls.stride_y);

	if( cls.ptr )
		dwt_util_log(LOG_INFO, "Loaded %i classes\n", cls.size_y);
	else
		dwt_util_error("Unable to load classes.\n");

	// TF plane
	const int bins = 256;

	const int size_x = spectra.size_x;
	const int size_y = bins;
	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(stride_y * size_x);

	void *magnitude = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);

	const int no_classes = 4; // 1..4

	void *st_sum[no_classes];
	int count[no_classes];

	for(int i = 0; i < no_classes; i++)
	{
		st_sum[i] = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);

		dwt_util_test_image_zero_s(st_sum[i], stride_x, stride_y, size_x, size_y);

		count[i] = 0;
	}

	const int ridges_no = 40;

	image_t *fv = image_create_s(ridges_no, spectra.size_y);

	image_t *ref_points = image_create_from_mat_s("../spectra-blobs/points.mat");

	for(int y = 0; y < spectra.size_y; y++)
	{
#if 1
		const int class = *dwt_util_addr_row_i(
			cls.ptr,
			y,
			cls.stride_x
		);
#else
		// everything in class 1
		const int class = 1;
#endif
		dwt_util_log(LOG_INFO, "Processing %i of %i (class %i)...\n", y, spectra.size_y, class);

		// get spectrum
		const void *row = dwt_util_addr_row_s(
			spectra.ptr,
			y,
			spectra.stride_x
		);

		if( (class-1) < 0 || (class-1) > no_classes )
		{
			dwt_util_error("invalid class %i\n", class);
		}

		// magnitude = ST(spectrum)
		gabor_st_s(
			// input
			row,
			spectra.stride_y,
			spectra.size_x,
			// output
			magnitude,
			stride_x,
			stride_y,
			bins
		);

		// sum[class] += magnitude
		dwt_util_add_s(
			st_sum[class-1],
			size_x,
			size_y,
			stride_x,
			stride_y,
			0,
			0,
			magnitude,
			size_x,
			size_y,
			stride_x,
			stride_y
		);

		count[class-1]++;

#if 1
		// plane
		image_t plane = (image_t){ .ptr = magnitude, .size_x = size_x, .size_y = size_y, .stride_x = stride_x, .stride_y = stride_y };
		// size_x = 2, size_y = ridges_no
		image_t *points = image_create_s(2, ridges_no);
		// fill points
		spectra_st_get_strongest_ridges(&plane, points, ridges_no);
		// diff maxima points
		image_t *diff = image_create_s(1/*x*/, ridges_no/*y*/);
		spectra_diff_points(diff, ref_points, points);
		// copy diff => fv
		for(int i = 0; i < ridges_no; i++)
			*image_coeff_s(fv, y, i) = *image_coeff_s(diff, i/*y*/, 0/*x*/);
		// free
		image_destroy(diff);
		// free
		image_destroy(points);
#endif
	}

	dwt_util_log(LOG_DBG, "done\n");

	dwt_util_save_to_svm_s(
		"data/maxima.svm",
		fv->ptr,
		fv->size_x,
		fv->size_y,
		fv->stride_x,
		fv->stride_y,
		cls.ptr,
		cls.size_x,
		cls.size_y,
		cls.stride_x,
		cls.stride_y
	);

	for(int i = 0; i < no_classes; i++)
	{
		dwt_util_log(LOG_INFO, "count(class %i) = %i\n", i+1, count[i]);

		dwt_util_scale_s(
			st_sum[i],
			size_x,
			size_y,
			stride_x,
			stride_y,
			1.f/count[i]
		);

		dwt_util_save_log_to_pgm_s(
			simple_sprintf("st-log-sum-class-%i.pgm", i+1),
			st_sum[i],
			stride_x,
			stride_y,
			size_x,
			size_y
		);

		image_t image_st_sum = (image_t){ st_sum[i], size_x, size_y, stride_x, stride_y };
		image_save_to_pgm_s(&image_st_sum, simple_sprintf("st-sum-class-%i.pgm", i+1));
		image_save_to_mat_s(&image_st_sum, simple_sprintf("st-sum-class-%i.mat", i+1));
	}

	dwt_util_finish();
}
