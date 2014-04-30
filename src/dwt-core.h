#ifndef DWT_CORE_H
#define DWT_CORE_H

/**
 * @brief Forward DWT with CDF 9/7 wavelet using single-loop @f$ 2 \times 2 @f$ core.
 *
 * As input, the image extended with 4 zeros around each edge is expected.
 * As output, the transform with 4 decay coefficients around each edge is returned.
 * This function operates with a @p float data type.
 * This function produces interleaved L and H subbands.
 *
 * @warning experimental
 * @todo optimize this function
 */
void fdwt_diag_2x2(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void fdwt_diag_2x2_full(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void fdwt_diag_2x2_HORIZ(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void fdwt_diag_2x2_VERT(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void fdwt_diag_2x2_HORIZ_STRIPS(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void fdwt_diag_2x2_VERT_STRIPS(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void fdwt_diag_2x2_HORIZ_BLOCK(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

void fdwt_diag_2x2_VERT_BLOCK(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

enum order {
	ORDER_HORIZ = 0,
	ORDER_VERT = 1,
	ORDER_HORIZ_STRIPS = 2,
	ORDER_VERT_STRIPS = 3,
	ORDER_HORIZ_BLOCKS = 4,
	ORDER_VERT_BLOCKS = 5,
	// vertical core
	ORDER_HORIZ_4X4 = 6,  // NOTE: [vertical core] need size_x and size_y be multiles of 4
	ORDER_HORIZ_8X2 = 7,  // NOTE: [vertical core] 8 horizontally, 2 vertically
	ORDER_HORIZ_2X8 = 8,  // NOTE: [vertical core] 2 horizontally, 8 vertically
	ORDER_HORIZ_8X8 = 9,  // TODO: [vertical core] 8 horizontally, 8 vertically (buggy, slow, incomplete)
	// diagonal core
	ORDER_HORIZ_6X2 = 10, // NOTE: [diagonal core] 6 horizontally, 2 vertically
	ORDER_HORIZ_2X6 = 11, // NOTE: [diagonal core] 2 horizontally, 6 vertically
	ORDER_HORIZ_6X6 = 12, // TODO: [diagonal core] 6 horizontally, 6 vertically (buggy, slow, not optimized)
	ORDER_LAST
};

typedef void (*fdwt_diag_2x2_func_t)(void *, int, int, int, int);

typedef void (*fdwt_vert_2x2_func_t)(void *, int, int, int, int);

/**
 * Select a proper function according to the order.
 */
fdwt_diag_2x2_func_t get_fdwt_diag_2x2_func(enum order order);

fdwt_vert_2x2_func_t get_fdwt_vert_2x2_func(enum order order);

#endif
