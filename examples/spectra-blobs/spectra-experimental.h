#ifndef SPECTRA_EXPERIMENTAL_H
#define SPECTRA_EXPERIMENTAL_H

#include "image.h"

void spectra_st_get_strongest_ridges(
	image_t *plane,		// input
	image_t *points,	// output
	int ridges_no		// number of the strongest points
);

void spectra_diff_points(
	image_t *result,	// output
	image_t *reference,	// input
	image_t *tested		// input
);

#endif
