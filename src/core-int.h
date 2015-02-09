#ifndef CORE_INT_H
#define CORE_INT_H

/**
 * @brief 2-D forward DWT.
 *
 * approach: single-loop core, vertical vectorization, 2x2 core
 * layout: interleaved subbands
 * type: int32_t
 * levels: 1
 * wavelet: 9/7-F in M. D. Adams. Reversible integer-to-integer wavelet transforms for image coding. 2002. http://www.ece.uvic.ca/~frodo/publications/phdthesis.pdf
 */
void dwt_cdf97_2f_vert2x2_i(
	void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
);

#endif
