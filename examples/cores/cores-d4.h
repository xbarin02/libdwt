#ifndef CORES_D4_H
#define CORES_D4_H

#include "image.h"

void fdwt2_d4_sep_horiz(
	struct image_t *source,
	struct image_t *target
);

void fdwt2_d4_sep_vert(
	struct image_t *source,
	struct image_t *target
);

void idwt2_d4_sep_horiz(
	struct image_t *source,
	struct image_t *target
);

#endif
