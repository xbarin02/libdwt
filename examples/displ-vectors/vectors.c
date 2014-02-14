/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief The program for processing displacement vectors representing
 * a distortion of an optical system.
 */

#include "libdwt.h"
#include "image.h"

#include <math.h>
#include <stdlib.h>

//#define LINEAR

int mat_get_int_off(image_t *image, int col)
{
	if( col < 0 || col > image->size_x-1 )
		dwt_util_abort();

	int y = 0;
	int x = col;

	float *coeff = dwt_util_addr_coeff_s(image->ptr, y, x, image->stride_x, image->stride_y);

	return (int)*coeff;
}

int mat_get_int_step(image_t *image, int col)
{
	if( col < 0 || col > image->size_x-1 )
		dwt_util_abort();

	int x = col;

	int last, curr;

	for(int y = 0; y < image->size_y; y++)
	{
		float *coeff = dwt_util_addr_coeff_s(image->ptr, y, x, image->stride_x, image->stride_y);

		curr = (int)*coeff;

		if( 0 == y )
			last = curr;
		else
			if( last != curr )
				return curr - last;
	}

	return 0;
}

int mat_get_int_max(image_t *image, int col)
{
	if( col < 0 || col > image->size_x-1 )
		dwt_util_abort();

	int max = -1, curr;

	int x = col;

	for(int y = 0; y < image->size_y; y++)
	{
		float *coeff = dwt_util_addr_coeff_s(image->ptr, y, x, image->stride_x, image->stride_y);

		curr = (int)*coeff;

		if( 0 == y || curr > max )
			max = curr;
	}

	return max;
}

int mat_get_int_cnt(image_t *image, int col)
{
	if( col < 0 || col > image->size_x-1 )
		dwt_util_abort();

	return (mat_get_int_max(image, col) - mat_get_int_off(image, col)) / mat_get_int_step(image, col) + 1;
}

static
int icmp_s(
	const void *p1,
	const void *p2
)
{
	if( *(const float *)p1 < *(const float *)p2 )
		return +1;
	if( *(const float *)p1 > *(const float *)p2 )
		return -1;
	return 0;
}

int count_levels(int x)
{
	int levels = 0;

	while( x && !(x&1) )
	{
		x /= 2;
		levels++;
	}

	return levels;
}

int main(int argc, char *argv[])
{
	const char *input_path = argc>1 ? argv[1] : "data/coordinates.csv";
	const char *output_path = argc>2 ? argv[2] : "data/result.mat";
	const int keep_N_largest = argc>3 ? atoi(argv[3]) : -1;
	const int perform_idwt_levels =  argc>4 ? atoi(argv[4]) : -1;
	enum wavelet_t wavelet = argc>5 ? atoi(argv[5]): 0;

	if( wavelet < 0 || wavelet > WAVELET_LAST-1 )
		wavelet = 0;

	dwt_util_log(LOG_INFO, "Usage: %s <input.csv> <output.mat> <N> <j> <wavelet>\n", argv[0]);
	dwt_util_log(LOG_INFO, "\tN       ... keep N largest wavelet coefficients (default -1)\n");
	dwt_util_log(LOG_INFO, "\tlevels  ... perform j levels of inverse wavelet transform (default -1)\n");
	dwt_util_log(LOG_INFO, "\twavelet ... choose wavelet (0..%i)\n", WAVELET_LAST-1);

	image_t input;

	dwt_util_log(LOG_DBG, "loading %s\n", input_path);
	// open path as matrix of floats
	if( !image_load_from_mat_s(&input, input_path) )
		dwt_util_error("could not load the input file :(\n");

	if( input.size_x != 4 )
		dwt_util_error("wrong format of the input file\n");

	int scale_x = 1;
	int scale_y = 1;

	if( !mat_get_int_step(&input, 0) )
		scale_x = 800*2;
	if( !mat_get_int_step(&input, 1) )
		scale_y = 600*2;

	dwt_util_log(LOG_DBG, "scaling input coordinates by (%i,%i)...\n", scale_x, scale_y);

	void *col_x = dwt_util_viewport(input.ptr, input.size_x, input.size_y, input.stride_x, input.stride_y, /*offset_x*/0, /*offset_y*/0);
	void *col_y = dwt_util_viewport(input.ptr, input.size_x, input.size_y, input.stride_x, input.stride_y, /*offset_x*/1, /*offset_y*/0);

	dwt_util_scale_s(col_x, 1, input.size_y, input.stride_x, input.stride_y, scale_x);
	dwt_util_scale_s(col_y, 1, input.size_y, input.stride_x, input.stride_y, scale_y);

	int off_x = mat_get_int_off(&input, 0);
	int off_y = mat_get_int_off(&input, 1);
	int step_x = mat_get_int_step(&input, 0);
	int step_y = mat_get_int_step(&input, 1);
	int count_x = mat_get_int_cnt(&input, 0);
	int count_y = mat_get_int_cnt(&input, 1);
	int int_max_x = mat_get_int_max(&input, 0);
	int int_max_y = mat_get_int_max(&input, 1);

	int levels_x = count_levels(step_x);
	int levels_y = count_levels(step_y);
	int max_levels = levels_x < levels_y ? levels_x : levels_y;

	int levels = max_levels;
	if( perform_idwt_levels >= 0 && perform_idwt_levels <= max_levels )
		levels = perform_idwt_levels;

	dwt_util_log(LOG_INFO, "performing %i levels of inverse transform (min. 0, max. %i)...\n", levels, max_levels);

	int step2_x = step_x/(1<<levels);
	int step2_y = step_y/(1<<levels);
	int count2_x = (1<<levels)*count_x;
	int count2_y = (1<<levels)*count_y;

	dwt_util_log(LOG_INFO, "input = matrix (%i,%i) with sampling interval (%i,%i) starting at (%i,%i)\n", count_x, count_y, step_x, step_y, off_x, off_y);
	dwt_util_log(LOG_INFO, "output = matrix (%i,%i) with sampling interval (%i,%i) starting at (%i,%i)\n", count2_x, count2_y, step2_x, step2_y, off_x, off_y);

	image_t *dx = image_create_s(count_x, count_y);
	image_t *dy = image_create_s(count_x, count_y);

	// shuffle matrix "input" to two matrices of displacement vectors "dx" and "dy"
	for(int y = 0; y < input.size_y; y++)
	{
		float *orig_x = image_coeff_s(&input, y, 0);
		float *orig_y = image_coeff_s(&input, y, 1);
		float *diff_x = image_coeff_s(&input, y, 2);
		float *diff_y = image_coeff_s(&input, y, 3);

		int orig_x_i = (int)*orig_x;
		int orig_y_i = (int)*orig_y;

		int coord_x = (orig_x_i - off_x) / step_x;
		int coord_y = (orig_y_i - off_y) / step_y;

		if( coord_x > count_x || coord_y > count_y )
			dwt_util_abort();

		// dwt_util_log(LOG_DBG, "(row=%i) orig_i=(%i,%i) diff=(%f,%f) coord=(%i,%i)\n", y, orig_x_i, orig_y_i, *diff_x, *diff_y, coord_x, coord_y);

		// put "diff" into "dx" and "dy" at "coord"
		float *displ_x_coeff = image_coeff_s(dx, coord_y, coord_x);
		float *displ_y_coeff = image_coeff_s(dy, coord_y, coord_x);

		*displ_x_coeff = *diff_x;
		*displ_y_coeff = *diff_y;
	}

	// save "dx" and "dy" into PGM
	dwt_util_log(LOG_DBG, "saving input to PGM...\n");
	image_save_to_pgm_s(dx, "displ_x.pgm");
	image_save_to_pgm_s(dy, "displ_y.pgm");

#if 1
	// calc norm
	image_t *mag = image_create_s(count_x, count_y);

	for(int y = 0; y < mag->size_y; y++)
		for(int x = 0; x < mag->size_x; x++)
		{
			*image_coeff_s(mag, y, x) = sqrtf(
				*image_coeff_s(dx, y, x) * *image_coeff_s(dx, y, x) +
				*image_coeff_s(dy, y, x) * *image_coeff_s(dy, y, x)
			);
		}

	float p = 2.f;

	float norm = dwt_util_band_lpnorm_s(
		mag->ptr,
		mag->stride_x,
		mag->stride_y,
		mag->size_x,
		mag->size_y,
		p
	);

	dwt_util_log(LOG_INFO, "norm = %f\n", norm);
#endif

#if 1
	image_t *copy_of_dx = image_create_s(dx->size_x, dx->size_y);
	image_t *copy_of_dy = image_create_s(dy->size_x, dy->size_y);
	// copy
	image_copy(dx, copy_of_dx);
	image_copy(dy, copy_of_dy);

	int j;
	// forward transform
	j = image_fdwt_s(dx, -1, wavelet);
	j = image_fdwt_s(dy, -1, wavelet);
	dwt_util_log(LOG_DBG, "Achieved %i levels of DWT\n", j);

	// magnitudes
	image_t *map = image_create_s(count_x, count_y);

	// calc magnitudes
	for(int y = 0; y < map->size_y; y++)
		for(int x = 0; x < map->size_x; x++)
		{
			*image_coeff_s(map, y, x) = sqrtf(
				*image_coeff_s(dx, y, x) * *image_coeff_s(dx, y, x) +
				*image_coeff_s(dy, y, x) * *image_coeff_s(dy, y, x)
			);
		}

	// save map
	image_save_to_pgm_s(map, "map.pgm");

#ifndef LINEAR
	dwt_util_log(LOG_INFO, "non-linear approximation\n");

	float array[count_x * count_y];
	// fill array
	for(int y = 0; y < map->size_y; y++)
		for(int x = 0; x < map->size_x; x++)
			array[y*count_x+x] = *image_coeff_s(map, y, x);
	// qsort
	qsort(array, count_x * count_y, sizeof(float), icmp_s);
	// find thresh
	int N = keep_N_largest;
	if( N < 1 || N > count_x * count_y )
		N = count_x * count_y;
	dwt_util_log(LOG_INFO, "keeping %i largest coefficients (min. 1, max. %i)...\n", N, count_x * count_y);
	const float thr = array[N-1];
	dwt_util_log(LOG_DBG, "threshold = %f\n", thr);

	// thresholding: reset all coefficienty below the threshold
	for(int y = 0; y < map->size_y; y++)
		for(int x = 0; x < map->size_x; x++)
		{
			if( *image_coeff_s(map, y, x) < thr )
			{
				*image_coeff_s(dx, y, x) = 0.f;
				*image_coeff_s(dy, y, x) = 0.f;
			}
		}
#else
	dwt_util_log(LOG_INFO, "linear approximation\n");

	int N = keep_N_largest;
	if( N < 1 || N > j )
		N = j;
	dwt_util_log(LOG_INFO, "keeping %i levels (min. 1, max. %i)...\n", N, j);
	
	for(int jj = 1; jj <= j; jj++)
	{
		dwt_util_log(LOG_DBG, "j=%i\n", jj);

		if( jj > N )
		{
			image_t tmp, *subband = &tmp;

			// dx LH
			image_subband(dx, subband, jj, DWT_LH);
			image_zero(subband);
			// dx HL
			image_subband(dx, subband, jj, DWT_HL);
			image_zero(subband);
			// dx HH
			image_subband(dx, subband, jj, DWT_HH);
			image_zero(subband);
			// dy LH
			image_subband(dy, subband, jj, DWT_LH);
			image_zero(subband);
			// dy HL
			image_subband(dy, subband, jj, DWT_HL);
			image_zero(subband);
			// dy HH
			image_subband(dy, subband, jj, DWT_HH);
			image_zero(subband);
		}
	}
#endif

	// inverse transform
	image_idwt_s(dx, -1, wavelet);
	image_idwt_s(dy, -1, wavelet);

	// substract the reconstructed image from the input one

	image_t *diff_of_dx = image_create_s(dx->size_x, dx->size_y);
	image_t *diff_of_dy = image_create_s(dy->size_x, dy->size_y);

	image_diff(diff_of_dx, copy_of_dx, dx);
	image_diff(diff_of_dy, copy_of_dy, dy);

	image_t *diff_mag = image_create_s(dx->size_x, dx->size_y);

	for(int y = 0; y < diff_mag->size_y; y++)
		for(int x = 0; x < diff_mag->size_x; x++)
		{
			*image_coeff_s(diff_mag, y, x) = sqrtf(
				*image_coeff_s(diff_of_dx, y, x) * *image_coeff_s(diff_of_dx, y, x) +
				*image_coeff_s(diff_of_dy, y, x) * *image_coeff_s(diff_of_dy, y, x)
			);
		}

	float diff_norm = dwt_util_band_lpnorm_s(
		diff_mag->ptr,
		diff_mag->stride_x,
		diff_mag->stride_y,
		diff_mag->size_x,
		diff_mag->size_y,
		p
	);

	dwt_util_log(LOG_INFO, "diff_norm = %f\n", diff_norm);
#endif

	const int border = 4; // FIXME: 0, 4
	dwt_util_log(LOG_INFO, "Extending by %i pixels...\n", border);

	image_t *edx = image_extend_s(dx, border);
	image_t *edy = image_extend_s(dy, border);

	const int ecount2_x = count2_x + (1<<levels)*(2*border);
	const int ecount2_y = count2_y + (1<<levels)*(2*border);

	// (2^...)times bigger matrices "d2?"
	image_t *d2x = image_create_s(count2_x, count2_y);
	image_t *d2y = image_create_s(count2_x, count2_y);
	image_t *ed2x = image_create_s(ecount2_x, ecount2_y);
	image_t *ed2y = image_create_s(ecount2_x, ecount2_y);

	// fill "d2?" transform with zeros (HL, LH and HH subband)
	image_zero(d2x);
	image_zero(d2y);
	image_zero(ed2x);
	image_zero(ed2y);

	// copy "d?" into LL subband of "d2?"
	image_copy(dx, d2x);
	image_copy(dy, d2y);
	image_copy(edx, ed2x);
	image_copy(edy, ed2y);

	// save "d2?" as PGM
	dwt_util_log(LOG_DBG, "saving transform to PGM...\n");
	image_save_to_pgm_s(d2x, "displ2_x.pgm");
	image_save_to_pgm_s(d2y, "displ2_y.pgm");
	image_save_to_pgm_s(ed2x, "edispl2_x.pgm");
	image_save_to_pgm_s(ed2y, "edispl2_y.pgm");

	// inverse DWT
	dwt_util_log(LOG_INFO, "Using wavelet %i...\n", wavelet);
	image_idwt_s(d2x, levels, wavelet);
	image_idwt_s(d2y, levels, wavelet);
	image_idwt_s(ed2x, levels, wavelet);
	image_idwt_s(ed2y, levels, wavelet);

	// save transformed images again
	dwt_util_log(LOG_DBG, "saving result to PGM...\n");
	image_save_to_pgm_s(d2x, "displ2dwt_x.pgm");
	image_save_to_pgm_s(d2y, "displ2dwt_y.pgm");
	image_save_to_pgm_s(ed2x, "edispl2dwt_x.pgm");
	image_save_to_pgm_s(ed2y, "edispl2dwt_y.pgm");
	image_save_to_mat_s(ed2x, "edispl2dwt_x.mat");
	image_save_to_mat_s(ed2y, "edispl2dwt_y.mat");

	image_t *output = image_create_s(4, count2_y * count2_x);

	int row = 0;
	int skip = 0;
	// shuffle two matrices "d2?" into single matrix "output" with 4 columns
	for(int y = 0; y < count2_y; y++)
		for(int x = 0; x < count2_x; x++)
		{
			const int border_shift = (border*(1<<levels));
#if 0
			float *coeff_x = image_coeff_s(d2x, y, x);
			float *coeff_y = image_coeff_s(d2y, y, x);
#else
			float *coeff_x = image_coeff_s(ed2x, y+border_shift, x+border_shift);
			float *coeff_y = image_coeff_s(ed2y, y+border_shift, x+border_shift);
#endif
			int orig_x = off_x + x*step2_x;
			int orig_y = off_y + y*step2_y;

			float target_x = (float)orig_x / scale_x;
			float target_y = (float)orig_y / scale_y;

			if( orig_x > int_max_x || orig_y > int_max_y )
			{
				skip++;
				continue;
			}

			*image_coeff_s(output, row, 0) = target_x;
			*image_coeff_s(output, row, 1) = target_y;
			*image_coeff_s(output, row, 2) = *coeff_x * (1<<levels);
			*image_coeff_s(output, row, 3) = *coeff_y * (1<<levels);

			row++;
		}

	output->size_y = row;

	dwt_util_log(LOG_DBG, "stored %i points, skipped %i points\n", row, skip);

	// save transformed images into MAT-file
	dwt_util_log(LOG_DBG, "saving result into %s...\n", output_path);
	if( image_save_to_mat_s(output, output_path) )
		dwt_util_error("error during saving the output file :(\n");

	return 0;
}
