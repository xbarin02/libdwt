#ifndef DWT_SIMPLE_H
#define DWT_SIMPLE_H

void fdwt2_cdf97_vertical_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

void fdwt2_cdf97_horizontal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

void fdwt2_cdf97_diagonal_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *j_max_ptr,
	int decompose_one
);

#endif
