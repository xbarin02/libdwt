#ifndef CORES_H
#define CORES_H

#include "image.h"

/**
 * @brief Single-loop core.
 *
 * * levels: 1
 * * dims: 2-D
 * * dir: forward
 * * core: vert2x2
 * * type: int32_t
 * * order: horizontal
 */
void cores2f_cdf97_v2x2_i32(
	struct image_t *src,
	struct image_t *dst
);

/**
 * @brief Single-loop core.
 *
 * * levels: 1
 * * dims: 2-D
 * * dir: forward
 * * core: vert2x2
 * * type: float
 * * order: horizontal
 */
void cores2f_cdf97_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
);

/**
 * @brief Single-loop core.
 *
 * * levels: 1
 * * dims: 2-D
 * * dir: forward
 * * core: vert2x2
 * * type: float
 * * order: horizontal
 */
void cores2f_cdf53_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
);

/**
 * @brief Single-loop core.
 *
 * * levels: 1
 * * dims: 2-D
 * * dir: inverse
 * * core: vert2x2
 * * type: float
 * * order: horizontal
 */
void cores2i_cdf97_v2x2_f32(
	struct image_t *src,
	struct image_t *dst
);

/**
 * @brief Single-loop core.
 *
 * * levels: 1
 * * dims: 2-D
 * * dir: forward
 * * core: vert2x2
 * * type: int32_t Q1.15.16
 * * order: horizontal
 */
void cores2f_cdf97_v2x2_x32(
	struct image_t *src,
	struct image_t *dst
);

/**
 * @brief Single-loop core.
 *
 * * levels: 1
 * * dims: 2-D
 * * dir: forward
 * * core: vert2x2
 * * type: int16_t Q1.7.8
 * * order: horizontal
 */
void cores2f_cdf97_v2x2_x16(
	struct image_t *src,
	struct image_t *dst
);

#endif
