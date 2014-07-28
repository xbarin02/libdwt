#include "libdwt.h"
#include "gabor.h"
#include <math.h> // M_PI
// assert
#include <assert.h>
// spectra_*
#include "spectra.h"

int main(int argc, char *argv[])
{
	const char *vec_path = (argc > 1) ? argv[1]
		: "data/spectra.dat";

	dwt_util_init();

	int clock_type = dwt_util_clock_autoselect();

	dwt_util_log(LOG_INFO, "Started\n");

	// load once, use mmap file
	void *spectra_ptr;
	int spectra_size_x, spectra_size_y;
	int spectra_stride_x, spectra_stride_y;

	dwt_clock_t clock_start = dwt_util_get_clock(clock_type);

	spectra_ptr = spectra_load(
		vec_path,
		&spectra_stride_x,
		&spectra_stride_y,
		&spectra_size_x,
		&spectra_size_y
	);

	dwt_clock_t clock_stop = dwt_util_get_clock(clock_type);
	double clock_secs = (clock_stop - clock_start)/(double)dwt_util_get_frequency(clock_type);
	dwt_util_log(LOG_INFO, "done in %f seconds\n", clock_secs);

	if( spectra_ptr )
		dwt_util_log(LOG_INFO, "Loaded %i spectra of length of %i samples.\n", spectra_size_y, spectra_size_x);

	const float unwrap_threshold = 2.f*(float)M_PI;

	// TF plane
	int bins = 256;

	const int size_x = spectra_size_x;
	const int size_y = bins;
	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(stride_y * size_x);

	void *magnitude = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);
	void *argument = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);
	void *frequency = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);
	void *ridges = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);

	// for each(spectrum) do ...
	//for(int y = 0; y < spectra_size_y; y++)
	int y = 1062;
	{
		// process spectrum
		assert( y < spectra_size_y );
		dwt_util_log(LOG_INFO, "Processing y=%i...\n", y);

		// get spectrum
		const void *row = dwt_util_addr_row_s(
			spectra_ptr,
			y,
			spectra_stride_x
		);

#if 1
		// save spectrum
		dwt_util_save1_to_mat_s(
			"spectrum.mat",
			row,
			spectra_size_x,
			spectra_stride_y
		);
#endif

#if 1
		// NOTE: ST
		{
			dwt_util_log(LOG_INFO, "ST...\n");
			// complex magnitude
			gabor_st_s(
				// input
				row,
				spectra_stride_y,
				spectra_size_x,
				// output
				magnitude,
				stride_x,
				stride_y,
				bins
				// params
			);
			// save magnitude
			dwt_util_save_log_to_pgm_s(
				"st-magnitude.pgm",
				magnitude,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// complex argument
			gabor_st_arg_s(
				// input
				row,
				spectra_stride_y,
				spectra_size_x,
				// output
				argument,
				stride_x,
				stride_y,
				bins
				// params
			);
			// inst. frequency
			phase_derivative_s(
				argument,
				frequency,
				stride_x,
				stride_y,
				size_x,
				size_y,
				unwrap_threshold
			);
			// ridges - 1
			detect_ridges1_s(
				magnitude,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_log_to_pgm_s(
				"st-ridges-1.pgm",
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// ridges - 2
			detect_ridges2_s(
				frequency,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_to_pgm_s(
				"st-ridges-2.pgm",
				1.f,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// ridges - 3
			detect_ridges3_s(
				magnitude,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_log_to_pgm_s(
				"st-ridges-3.pgm",
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// save inst. freq.
			dwt_util_save_sym_to_pgm_s(
				"st-frequency.pgm",
				unwrap_threshold,
				frequency,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
		}

		// NOTE: FT
		{
			dwt_util_log(LOG_INFO, "FT...\n");
			gabor_ft_s(
				// input
				row,
				spectra_stride_y,
				spectra_size_x,
				// output
				magnitude,
				stride_x,
				stride_y,
				bins,
				// params
				/* sigma = */ 40.0f // 20
			);
			dwt_util_save_log_to_pgm_s(
				"ft-magnitude.pgm",
				magnitude,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// complex argument
			gabor_ft_arg_s(
				// input
				row,
				spectra_stride_y,
				spectra_size_x,
				// output
				argument,
				stride_x,
				stride_y,
				bins,
				// params
				/* sigma = */ 40.0f // 20
			);
			// inst. frequency
			phase_derivative_s(
				argument,
				frequency,
				stride_x,
				stride_y,
				size_x,
				size_y,
				unwrap_threshold
			);
			// ridges - 1
			detect_ridges1_s(
				magnitude,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_log_to_pgm_s(
				"ft-ridges-1.pgm",
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// ridges - 2
			detect_ridges2_s(
				frequency,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_to_pgm_s(
				"ft-ridges-2.pgm",
				1.f,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// ridges - 3
			detect_ridges3_s(
				magnitude,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_log_to_pgm_s(
				"ft-ridges-3.pgm",
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// save inst. freq.
			dwt_util_save_sym_to_pgm_s(
				"ft-frequency.pgm",
				unwrap_threshold,
				frequency,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
		}

		// NOTE: WT
		{
			dwt_util_log(LOG_INFO, "WT...\n");
			gabor_wt_s(
				// input
				row,
				spectra_stride_y,
				spectra_size_x,
				// output
				magnitude,
				stride_x,
				stride_y,
				bins,
				// params
				/* sigma = */        1.0f, // 10; 2; 0.3; 1
				/* freq = */         0.999f*(float)M_PI // pi; pi; pi/2; pi/4
			);
			dwt_util_save_log_to_pgm_s(
				"wt-magnitude.pgm",
				magnitude,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			gabor_wt_arg_s(
				// input
				row,
				spectra_stride_y,
				spectra_size_x,
				// output
				argument,
				stride_x,
				stride_y,
				bins,
				// params
				/* sigma = */        1.0f, // 10; 2; 0.3; 1
				/* freq = */         0.999f*(float)M_PI // pi; pi; pi/2; pi/4
			);
			// inst. frequency
			phase_derivative_s(
				argument,
				frequency,
				stride_x,
				stride_y,
				size_x,
				size_y,
				unwrap_threshold
			);
			// ridges - 1
			detect_ridges1_s(
				magnitude,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_log_to_pgm_s(
				"wt-ridges-1.pgm",
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// ridges - 2
			detect_ridges2_s(
				frequency,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_to_pgm_s(
				"wt-ridges-2.pgm",
				1.f,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// ridges - 3
			detect_ridges3_s(
				magnitude,
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y,
				0.f
			);
			// save ridges
			dwt_util_save_log_to_pgm_s(
				"wt-ridges-3.pgm",
				ridges,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
			// save inst. freq.
			dwt_util_save_sym_to_pgm_s(
				"wt-frequency.pgm",
				unwrap_threshold,
				frequency,
				stride_x,
				stride_y,
				size_x,
				size_y
			);
		}
#endif
	}

	dwt_util_log(LOG_DBG, "done\n");

	spectra_unload(
		spectra_ptr,
		spectra_stride_x,
		spectra_stride_y,
		spectra_size_x,
		spectra_size_y
	);

	dwt_util_free_image(&magnitude);
	dwt_util_free_image(&argument);
	dwt_util_free_image(&frequency);
	dwt_util_free_image(&ridges);

	dwt_util_finish();

	return 0;
}
