#ifndef CORES_HAAR_H
#define CORES_HAAR_H

#include "image.h"

void cores2f_haar_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
);

void cores2i_haar_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
);

#endif
