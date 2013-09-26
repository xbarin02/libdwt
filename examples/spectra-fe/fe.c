/**
 * @brief Stellar spectra feature extraction using DWT and SWT.
 */

#include "libdwt.h"
#include "swt.h"

#include <assert.h>
#include <stdlib.h>

enum t_wavelet {
	WAVELET_CDF97 = 0,
	WAVELET_CDF53,
	WAVELET_LAST
};

const char *wavelet_to_cstr(enum t_wavelet wavelet)
{
	switch(wavelet)
	{
		case WAVELET_CDF97: return "cdf97";
		case WAVELET_CDF53: return "cdf53";
		default: dwt_util_abort();
	}

	return NULL;
}

enum t_transform {
	TRANSFORM_DWT = 0,
	TRANSFORM_SWT,
	TRANSFORM_LAST
};

const char *transform_to_cstr(enum t_transform transform)
{
	switch(transform)
	{
		case TRANSFORM_DWT: return "dwt";
		case TRANSFORM_SWT: return "swt";
		default: dwt_util_abort();
	}

	return NULL;
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

const char *aggregation_to_cstr(enum t_aggregation aggregation)
{
	switch(aggregation)
	{
		case AGGREGATION_MED: return "med";
		case AGGREGATION_MAXNORM: return "maxnorm";
		case AGGREGATION_NORM: return "norm";
		case AGGREGATION_STDEV: return "stdev";
		case AGGREGATION_MEAN: return "mean";
		case AGGREGATION_WPS: return "wps";
		case AGGREGATION_VAR: return "var";
		case AGGREGATION_SKEW: return "skew";
		case AGGREGATION_KURT: return "kurt";
		default: dwt_util_abort();
	}

	return NULL;
}

void do_transform(
	// input image
	const void *src,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	// type of transform
	enum t_transform transform,
	enum t_wavelet wavelet,
	// output
	void **dst,
	// number of levels
	int levels
)
{
	size_t image_size = size_y * stride_x; // FIXME: use image_size() from libdwt.c

	if( TRANSFORM_DWT == transform )
	{
		// alloc dst
		dwt_util_alloc_image(
			dst,
			stride_x,
			stride_y,
			size_x,
			size_y);

		// copy src to dst
		// FIXME: _s
		dwt_util_copy_i(
			src,
			*dst,
			stride_x,
			stride_y,
			size_x,
			size_y);

		// dwt
		int j = levels;

		void (*dwt_wavelet_2f_s)(void *, int, int, int, int, int, int, int *, int, int) = 0;
		switch(wavelet)
		{
			case WAVELET_CDF97: dwt_wavelet_2f_s = dwt_cdf97_2f_s; break;
			case WAVELET_CDF53: dwt_wavelet_2f_s = dwt_cdf53_2f_s; break;
			default: dwt_util_abort();
		}

		dwt_wavelet_2f_s(
			*dst,
			stride_x,
			stride_y,
			size_x,
			size_y,
			size_x,
			size_y,
			&j,
			1,
			0);
	}

	if( TRANSFORM_SWT == transform )
	{
		// levels + input = levels+1
		int j = levels+1;

		int level_stride = 4 * image_size;
		int transform_size = j * level_stride;

		// FIXME: support 2D transform
		assert( 1 == size_y );

		// alloc dst
		*dst = dwt_util_allocate_vec_s(transform_size);

		// copy src to dst(level=0,subband=LL)
		// FIXME: _s
		dwt_util_copy_i(
			src,
			*dst,
			stride_x,
			stride_y,
			size_x,
			size_y);

		// swt
		for(int l = 0; l < j-1; l++)
		{
			// NOTE: SWT from LL(l) to LL(l+1)

			// horizontal
			void *srcL = (char *)*dst + ((l+0) * level_stride + 0);
			void *dstL = (char *)*dst + ((l+1) * level_stride + 0);
			void *dstH = (char *)*dst + ((l+1) * level_stride + image_size);

			// FIXME
			assert( sizeof(float) == stride_y );

			void (*swt_wavelet_f_ex_stride_s)(const void *, void *, void *, int, int, int) = 0;
			switch(wavelet)
			{
				case WAVELET_CDF97: swt_wavelet_f_ex_stride_s = swt_cdf97_f_ex_stride_s; break;
				case WAVELET_CDF53: swt_wavelet_f_ex_stride_s = swt_cdf53_f_ex_stride_s; break;
				default: dwt_util_abort();
			}

			swt_wavelet_f_ex_stride_s(
				srcL,
				dstL,
				dstH,
				size_x,
				stride_y,
				l
			);

			// vertical
			// FIXME: not implemented yet
		}
	}
}

void do_aggregation(
	// input transform
	void *t,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int levels,
	// type of transform
	enum t_transform transform,
	// aggregation
	enum t_aggregation aggregation,
	// store FV here
	float *fv
)
{
	if( TRANSFORM_DWT == transform )
	{
		void (*dwt_util_aggregation_s)(const void *, int, int, int, int, int, int, int, float *) = 0;
		switch(aggregation)
		{
			case AGGREGATION_MED: dwt_util_aggregation_s = dwt_util_med_s; break;
			case AGGREGATION_MAXNORM: dwt_util_aggregation_s = dwt_util_maxnorm_s; break;
			case AGGREGATION_NORM: dwt_util_aggregation_s = dwt_util_norm_s; break;
			case AGGREGATION_STDEV: dwt_util_aggregation_s = dwt_util_stdev_s; break;
			case AGGREGATION_MEAN: dwt_util_aggregation_s = dwt_util_mean_s; break;
			case AGGREGATION_WPS: dwt_util_aggregation_s = dwt_util_wps_s; break;
			case AGGREGATION_VAR: dwt_util_aggregation_s = dwt_util_var_s; break;
			case AGGREGATION_SKEW: dwt_util_aggregation_s = dwt_util_skew_s; break;
			case AGGREGATION_KURT: dwt_util_aggregation_s = dwt_util_kurt_s; break;
			default: dwt_util_abort();
		}

		dwt_util_aggregation_s(
			t,
			stride_x,
			stride_y,
			size_x,
			size_y,
			size_x,
			size_y,
			levels+1,
			fv
		);
	}

	if( TRANSFORM_SWT == transform )
	{
		size_t image_size = size_y * stride_x; // FIXME: use image_size() from libdwt.c
		int level_stride = 4 * image_size;

		for(int l = 0; l < levels; l++)
		{
			void *ptrH = (char *)t + ((l+1) * level_stride + image_size);

			float (*dwt_util_band_aggregation_s)(const void *, int, int, int, int) = 0;
			switch(aggregation)
			{
				case AGGREGATION_MED: dwt_util_band_aggregation_s = dwt_util_band_med_s; break;
				case AGGREGATION_MAXNORM: dwt_util_band_aggregation_s = dwt_util_band_maxnorm_s; break;
				case AGGREGATION_NORM: dwt_util_band_aggregation_s = dwt_util_band_norm_s; break;
				case AGGREGATION_STDEV: dwt_util_band_aggregation_s = dwt_util_band_stdev_s; break;
				case AGGREGATION_MEAN: dwt_util_band_aggregation_s = dwt_util_band_mean_s; break;
				case AGGREGATION_WPS: break;
				case AGGREGATION_VAR: dwt_util_band_aggregation_s = dwt_util_band_var_s; break;
				case AGGREGATION_SKEW: dwt_util_band_aggregation_s = dwt_util_band_skew_s; break;
				case AGGREGATION_KURT: dwt_util_band_aggregation_s = dwt_util_band_kurt_s; break;
				default: dwt_util_abort();
			}

			if( AGGREGATION_WPS == aggregation )
				fv[l] = dwt_util_band_wps_s(
					ptrH,
					0,
					sizeof(float),
					size_x,
					1,
					l
				);
			else
				fv[l] = dwt_util_band_aggregation_s(
					ptrH,
					0,
					sizeof(float),
					size_x,
					1
				);
		}
	}
}

int main(int argc, char *argv[])
{
	const char *vec_path = (argc > 1) ? argv[1]
		: "data/spectra.dat";
	const char *cls_path = (argc > 2) ? argv[2]
		: "data/target_classes.dat";

	dwt_util_init();

	dwt_util_log(LOG_INFO, "Started\n");

	void *ptr;
	int size_x, size_y;
	int stride_x, stride_y;

	// load spectra
	dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", vec_path);
	dwt_util_load_from_mat_s(vec_path, &ptr, &size_x, &size_y, &stride_x, &stride_y);
	if( ptr )
		dwt_util_log(LOG_INFO, "Loaded %i spectra of length of %i samples.\n", size_y, size_x);
	else
		dwt_util_error("Unable to load spectra.\n");

	// preprocessing
	dwt_util_log(LOG_INFO, "Shifting base-line by median...\n");
	dwt_util_shift21_med_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y
	);
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

	void *cls_ptr;
	int cls_size_x, cls_size_y;
	int cls_stride_x, cls_stride_y;

	// load classes
	dwt_util_log(LOG_INFO, "Loading data from ASCII MAT-file '%s'...\n", cls_path);
	dwt_util_load_from_mat_i(cls_path, &cls_ptr, &cls_size_x, &cls_size_y, &cls_stride_x, &cls_stride_y);
	if( cls_ptr )
		dwt_util_log(LOG_INFO, "Loaded %i classes\n", cls_size_y);
	else
		dwt_util_error("Unable to load classes.\n");

	// the tranforms
	void *t[size_y];

	int levels = 10;
	for(int transform = 0; transform < TRANSFORM_LAST; transform++)
	for(int wavelet = 0; wavelet < WAVELET_LAST; wavelet++)
	for(int aggregation = 0; aggregation < AGGREGATION_LAST; aggregation++)
	{
		dwt_util_log(LOG_INFO, "levels=%i transform=%s wavelet=%s aggregation=%s\n",
			levels, transform_to_cstr(transform), wavelet_to_cstr(wavelet), aggregation_to_cstr(aggregation));

		// the feature vectors
		float fv[size_y*levels];

		// for each(spectrum) do transform
		for(int y = 0; y < size_y; y++)
		{
			const void *src = dwt_util_addr_coeff_s(
				ptr,
				y,
				0,
				stride_x,
				stride_y
			);

			void **dst = &t[y];

			do_transform(
				// input image
				src,
				size_x,
				1,
				stride_x,
				stride_y,
				// type of transform
				transform,
				wavelet,
				// output
				dst,
				// number of levels
				levels
			);
		}

		// for each(spectrum) do aggregation
		for(int y = 0; y < size_y; y++)
		{
			do_aggregation(
				// input transform
				t[y],
				size_x,
				1,
				stride_x,
				stride_y,
				levels,
				// type of transform
				transform,
				// aggregation
				aggregation,
				// store FV here
				&fv[y*levels]
			);
		}

		// store fv
		char fv_svm_path[4096];
		sprintf(fv_svm_path, "data/fv_%s_%s_%i_%s.svm", transform_to_cstr(transform), wavelet_to_cstr(wavelet), levels, aggregation_to_cstr(aggregation));
		dwt_util_save_to_svm_s(
			fv_svm_path,
			// matrix of vectors
			fv,
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

		// free memory
		for(int y = 0; y < size_y; y++)
		{
			free(t[y]);
		}
	}

	dwt_util_free_image(&cls_ptr);
	dwt_util_free_image(&ptr);

	dwt_util_finish();

	return 0;
}
