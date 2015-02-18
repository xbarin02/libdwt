#ifndef CORES_NSLS_H
#define CORES_NSLS_H

#include "image2.h"

void cores2f_cdf97_n2x2_f32(
	struct image_t *src,
	struct image_t *dst
);

void cores2f_cdf53_n2x2_f32(
	struct image_t *src,
	struct image_t *dst
);

void cores2f_cdf97_n2x2_f32_sse(
	struct image_t *src,
	struct image_t *dst
);

#endif
