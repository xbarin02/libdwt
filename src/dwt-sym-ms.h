#ifndef DWT_SYM_MS_H
#define DWT_SYM_MS_H

/**
 * @brief Multi-scale single-loop forward implementation using SSE vectorized core.
 *
 * @warning experimental
 */
void ms_cdf97_2f_dl_4x4_s(
	int size_x,
	int size_y,
	void *ptr,
	int stride_x,
	int stride_y,
	int J
);

#endif
