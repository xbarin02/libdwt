#ifndef DWT_SIMPLE_H
#define DWT_SIMPLE_H

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 9/7
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: horizontal
 */
void fdwt_cdf97_horizontal_s(
	void *ptr,
	int size,
	int stride
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 9/7
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: vertical
 */
void fdwt_cdf97_vertical_s(
	void *ptr,
	int size,
	int stride
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 9/7
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: diagonal
 */
void fdwt_cdf97_diagonal_s(
	void *ptr,
	int size,
	int stride
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 9/7
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: horizontal
 * @li parallelization: OpenMP
 */
void fdwt2_cdf97_horizontal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

void fdwt1_cdf97_horizontal_s(
	void *ptr,
	int size,
	int stride,
	int *j_max_ptr
);

void fdwt1_single_cdf97_horizontal_s(
	void *ptr,
	int size,
	int stride
);

void fdwt1_single_cdf97_horizontal_min5_s(
	void *ptr,
	int size,
	int stride
);

void fdwt1_single_cdf97_vertical_min5_s(
	void *ptr,
	int size,
	int stride
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 9/7
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: vertical
 * @li parallelization: OpenMP
 */
void fdwt2_cdf97_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

// TODO: horizontal transforms only
void fdwt2h1_cdf97_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

// TODO: vertical transforms only
void fdwt2v1_cdf97_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 9/7
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: diagonal
 * @li parallelization: OpenMP
 */
void fdwt2_cdf97_diagonal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: horizontal
 */
void fdwt_cdf53_horizontal_s(
	void *ptr,
	int size,
	int stride
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: WCDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: horizontal
 */
void fdwt_eaw53_horizontal_s(
	void *ptr,
	int size,
	int stride,
	const float *eaw_w
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: WCDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: vertical
 */
void fdwt_eaw53_vertical_s(
	void *ptr,
	int size,
	int stride,
	const float *eaw_w
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: WCDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: diagonal
 */
void fdwt_eaw53_diagonal_s(
	void *ptr,
	int size,
	int stride,
	const float *eaw_w
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: vertical
 */
void fdwt_cdf53_vertical_s(
	void *ptr,
	int size,
	int stride
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 1-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li vectorisation: diagonal
 */
void fdwt_cdf53_diagonal_s(
	void *ptr,
	int size,
	int stride
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: horizontal
 * @li parallelization: OpenMP
 */
void fdwt2_cdf53_horizontal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: WCDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: horizontal
 * @li parallelization: OpenMP
 */
void fdwt2_eaw53_horizontal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one,
	float *wH[],
	float *wV[],
	float alpha
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: WCDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: vertical
 * @li parallelization: OpenMP
 */
void fdwt2_eaw53_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one,
	float *wH[],
	float *wV[],
	float alpha
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: WCDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: diagonal
 * @li parallelization: OpenMP
 */
void fdwt2_eaw53_diagonal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one,
	float *wH[],
	float *wV[],
	float alpha
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: vertical
 * @li parallelization: OpenMP
 */
void fdwt2_cdf53_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

/**
 * @brief Wavelet transform.
 *
 * @li data: 2-D
 * @li type: single (float)
 * @li transform: DWT
 * @li direction: forward
 * @li wavelet: CDF 5/3
 * @li layout: interleaved subbands (in-place lifting)
 * @li approach: separable filtering
 * @li vectorisation: diagonal
 * @li parallelization: OpenMP
 */
void fdwt2_cdf53_diagonal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

#endif
