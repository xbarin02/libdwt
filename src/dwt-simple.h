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
