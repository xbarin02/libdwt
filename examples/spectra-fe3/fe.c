/**
 * @brief Spectra feature extraction using TF transform and DWT.
 */

#include "libdwt.h"
#include "gabor.h"
#include "image.h"
#include "spectra.h"
#include "system.h"
#include <math.h>

image_t *image_spectra_load(
	const char *path
)
{
	image_t *image = dwt_util_alloc(1, sizeof(image_t));

	image->ptr = spectra_load(
		path,
		&image->stride_x,
		&image->stride_y,
		&image->size_x,
		&image->size_y
	);

	return image;
}

void image_spectra_unload(
	image_t *image
)
{
	spectra_unload(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y
	);

	dwt_util_free(image);
}

enum t_aggregation {
	AGGREGATION_MED = 0,
	AGGREGATION_MAXNORM,
	AGGREGATION_NORM,
	AGGREGATION_STDEV,
	AGGREGATION_MEAN,
	AGGREGATION_WPS,
	AGGREGATION_VAR,
	AGGREGATION_SKEW,
	AGGREGATION_KURT,
	AGGREGATION_LAST
};

void (*dwt_util_aggregation_s[AGGREGATION_LAST])(const void *, int, int, int, int, int, int, int, float *) = {
	[AGGREGATION_MED] = dwt_util_med_s,
	[AGGREGATION_MAXNORM] = dwt_util_maxnorm_s,
	[AGGREGATION_NORM] = dwt_util_norm_s,
	[AGGREGATION_STDEV] = dwt_util_stdev_s,
	[AGGREGATION_MEAN] = dwt_util_mean_s,
	[AGGREGATION_WPS] = dwt_util_wps_s,
	[AGGREGATION_VAR] = dwt_util_var_s,
	[AGGREGATION_SKEW] = dwt_util_skew_s,
	[AGGREGATION_KURT] = dwt_util_kurt_s,
};

void image_aggregation_s(image_t *image, enum t_aggregation aggregation, int j, float *fv)
{
	dwt_util_aggregation_s[aggregation](
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		j,
		fv
	);
}

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

	// paths
	const char *spectra_path = (argc > 1) ? argv[1]
		: "data/spectra.dat";
	const char *classes_path = (argc > 2) ? argv[2]
		: "data/classes.dat";

	// load spectra
	image_t *spectra = image_spectra_load(spectra_path);

	if( spectra->ptr )
		dwt_util_log(LOG_INFO, "Loaded %i spectra of length of %i samples.\n", spectra->size_y, spectra->size_x);
	else
		dwt_util_error("Unable to load spectra from '%s'.\n", spectra_path);

	// load classes
	image_t *classes = dwt_util_alloc(1, sizeof(image_t));

	dwt_util_load_from_mat_i(classes_path, &classes->ptr, &classes->size_x, &classes->size_y, &classes->stride_x, &classes->stride_y);

	if( classes->ptr )
		dwt_util_log(LOG_INFO, "Loaded %i classes.\n", classes->size_y);
	else
		dwt_util_error("Unable to load classes from '%s'.\n", classes_path);

	// TF plane
	const int bins = 16; // 16, 32, 64, 128, 256, 512

	image_t *plane = image_create_opt_s(spectra->size_x, bins);

	// DWT
	const int levels = 3;

	// features
	int features_no = dwt_util_count_subbands_s(plane->ptr, plane->stride_x, plane->stride_y, plane->size_x, plane->size_y, plane->size_x, plane->size_y, levels+1);

	dwt_util_log(LOG_DBG, "%i features for %i levels\n", features_no, levels);

	image_t *features = image_create_opt_s(features_no, spectra->size_y);

	int wavelet = WAVELET_CDF97;

	for(int agg = 0; agg < AGGREGATION_LAST; agg++)
	{
		// for each spectrum
		for(int y = 0; y < spectra->size_y; y++)
		{
			// get class
			const int class = *image_row_i(classes, y);

			// get spectrum
			const void *spectrum = image_row_s(spectra, y);

			dwt_util_log(LOG_INFO, "Processing %i of %i (class %i)...\n", y, spectra->size_y, class);

			// compute TF transform
#if 0
			gabor_st_s(
				// input
				spectrum,
				spectra->stride_y,
				spectra->size_x,
				// output
				plane->ptr,
				plane->stride_x,
				plane->stride_y,
				bins
			);
#endif
#if 0
			gabor_wt_s(
				// input
				spectrum,
				spectra->stride_y,
				spectra->size_x,
				// output
				plane->ptr,
				plane->stride_x,
				plane->stride_y,
				bins,
				1.0f,
				(float)M_PI
			);
#endif
#if 1
			gabor_ft_s(
				// input
				spectrum,
				spectra->stride_y,
				spectra->size_x,
				// output
				plane->ptr,
				plane->stride_x,
				plane->stride_y,
				bins,
				480.f // 20, 40, 60, 80, 120, 240, 480, 960, 1920
			);
#endif

#if 0
			image_save_log_to_pgm_s(plane, "plane.pgm");
#endif

			// compute 2-D DWT of the TF plane
			if( image_fdwt_s(plane, levels, wavelet) != levels )
				dwt_util_error("too small plane\n");

#if 0
			image_save_log_to_pgm_s(plane, "dwt.pgm");
#endif

			// aggregation
			image_aggregation_s(plane, agg, levels+1, image_coeff_s(features, y, 0));
		}

		// save to SVM
		image_save_to_svm_s(classes, features, simple_sprintf("data/fv_dwt_%i.svm", agg));
	}

	image_destroy(plane);
	image_destroy(features);
	image_destroy(classes);

	// unload
	image_spectra_unload(spectra);

	dwt_util_finish();

	return 0;
}
