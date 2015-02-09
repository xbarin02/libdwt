#ifndef VOLUME_DWT_H
#define VOLUME_DWT_H

#include "volume.h" // struct volume_t

/**
 * @brief Forward 3-D transform.
 *
 * Plain 1-D horizontal vectorization separated for each direction.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: horizontal
 * * approach: separated for each dimension
 * * strategy: inplace
 * * layout: interleaved subbands
 */
void cdf97_3f_ip_sep_horizontal_s(struct volume_t *volume);

/**
 * @brief Forward 3-D transform.
 *
 * Plain 1-D horizontal vectorization separated for each direction.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: horizontal
 * * approach: separated for each dimension
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_sep_horizontal_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * Plain 1-D vertical vectorization separated for each direction.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: vertical
 * * approach: separated for each dimension
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_sep_vertical_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * 2-D single-loop core 4x4 vertical vectorization per each z-slice, then separated vertical vectorization for z-axe.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: combined (both vertical)
 * * approach: combined, 2-D core vert4x4 per slices, then separated vertical over z-axe
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_slices_vert4x4_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * Non-optimized variant of 3-D single-loop core approach with 2x2x2 vertically vectorized core.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: vertical
 * * approach: 3-D core vert2x2x2
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_baseline_vert2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * 3-D single-loop core approach with 2x2x2 vertically vectorized core, horizontal order.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: vertical
 * * approach: 3-D core vert2x2x2
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_HORIZ_vert2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * 3-D single-loop core approach with 4x4x2 vertically vectorized core.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: vertical
 * * approach: 3-D core vert4x4x2
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_cube_vert4x4x2_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * 3-D single-loop core approach with 4x4x2 vertically vectorized core, horizontal order.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: vertical
 * * approach: 3-D core vert4x4x2
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_HORIZ_vert4x4x2_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * Non-optimized variant of 3-D single-loop core approach with 2x2x2 diagonally vectorized core.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: diagonal
 * * approach: 3-D core diag2x2x2
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_baseline_diag2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * Non-optimized variant of 3-D single-loop core approach with 2x2x2 diagonally vectorized core. horizontal order.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: diagonal
 * * approach: 3-D core diag2x2x2
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_HORIZ_diag2x2x2_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Forward 3-D transform.
 *
 * 3-D single-loop core approach with 4x4x4 vertically vectorized core, horizontal order.
 *
 * * dimensions: 3-D
 * * direction: forward
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: vertical
 * * approach: 3-D core vert4x4x4
 * * strategy: out-of-place
 * * layout: interleaved subbands
 */
void cdf97_3f_op_HORIZ_vert4x4x4_s(struct volume_t *volume_src, struct volume_t *volume_dst);

/**
 * @brief Inverse 3-D transform.
 *
 * Plain 1-D horizontal vectorization separated for each direction
 *
 * * dimensions: 3-D
 * * direction: inverse
 * * scales: 1
 * * wavelet: CDF 9/7
 * * data type: float
 * * vectorization: horizontal
 * * approach: separated for each dimension
 * * strategy: inplace
 * * layout: interleaved subbands
 */
void cdf97_3i_ip_sep_horizontal_s(struct volume_t *volume);

enum volume_approach {
	VOL_SEP_HORIZONTAL = 0,
	VOL_SEP_VERTICAL = 1,
	VOL_SLICES_VERT4X4 = 2,
	VOL_BASELINE_VERT2X2X2 = 3,
	VOL_HORIZ_VERT2X2X2 = 4,
	VOL_BASELINE_VERT4X4X2 = 5,
	VOL_HORIZ_VERT4X4X2 = 6,
	VOL_HORIZ_VERT4X4X4 = 7,
	VOL_BASELINE_DIAG2X2X2 = 8,
	VOL_HORIZ_DIAG2X2X2 = 9,
	VOL_SEP_HORIZONTAL_X = 10,
	VOL_SEP_HORIZONTAL_Y = 11,
	VOL_SEP_HORIZONTAL_Z = 12,
	VOL_LAST
};

void cdf97_3f_op_wrapper_s(struct volume_t *volume_src, struct volume_t *volume_dst, enum volume_approach approach);

/**
 * @brief Perform a single measurment.
 *
 * @return zero if the test pass; non-zero if fails
 */
int volume_perftest_fwd97op_s(
	int size, // size_x, size_y_ size_z
	int opt_stride,
	enum volume_approach approach,
	int N, // tests, select minimum
	double *secs, // seconds per pixel
	long unsigned *faults // page faults
);

/**
 * @brief Perform series of measurments.
 */
int volume_measure_fwd97op_s(
	int size_min,
	int size_max,
	int size_step,
	int N,
	int opt_stride,
	enum volume_approach approach
);

#endif
