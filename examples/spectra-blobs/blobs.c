/**
 * @brief Find strongest ridges of the S transform plane.
 */

#include "libdwt.h"
#include <stddef.h> // size_t
#include "image.h" // image_t, ...
#include <complex.h> // complex
#include "gabor.h" // gaussian_center
#include "system.h" // dwt_util_free
#include "util.h" // dwt_util_find_max_pos_s
#include <assert.h> // assert
#include <math.h> // sqrtf
#include "spectra-experimental.h"

int main(int argc, char *argv[])
{
	dwt_util_init();

	const char *path = (argc > 1) ? argv[1] :
		"data/st-log/st-sum-class-all.mat";

	image_t *plane = image_create_from_mat_s(path);

	if( !plane )
		dwt_util_error("unable to load from %s\n", path);

	dwt_util_log(LOG_INFO, "plane size=(%i,%i)\n", plane->size_y, plane->size_x);

	image_save_to_pgm_s(plane, "plane.pgm");

	const int ridges_no = 40;

	// strongest points: size_x=2, size_y=ridges_no
	image_t *points = image_create_s(2, ridges_no);

	// strongest ridges
	spectra_st_get_strongest_ridges(plane, points, ridges_no);

	image_save_to_mat_s(points, "points.mat");

	image_destroy(points);

	image_destroy(plane);

	dwt_util_finish();

	return 0;
}
