/**
 * @brief image_t infrastructure
 */

#ifndef IMAGE_H
#define IMAGE_H

#include "libdwt.h" // enum dwt_subbands

struct image_t;

#if 1
struct image_t {
	void *ptr;
	int size_x;
	int size_y;
	int stride_x;
	int stride_y;
};
#endif

typedef struct image_t image_t;

void image_init(
	image_t *image,
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y
);

int image_alloc(
	image_t *image
);

image_t *image_create_s(
	int size_x,
	int size_y
);

int image_load_from_mat_s(
	image_t *image,
	const char *path
);

int image_save_to_mat_s(
	image_t *image,
	const char *path
);

void image_save_to_pgm_s(
	image_t *image,
	const char *path
);

void image_zero(
	image_t *image
);

void image_subband(
	image_t *src,
	image_t *dst,
	int j,
	enum dwt_subbands band
);

float *image_coeff_s(
	image_t *image,
	int y,
	int x
);

void image_copy(
	image_t *src,
	image_t *dst
);

void image_diff(
	image_t *dst,
	image_t *src0,
	image_t *src1
);

void image_idwt_interp53_s(
	image_t *image,
	int levels
);

void image_idwt_cdf53_s(
	image_t *image,
	int levels
);

void image_idwt_cdf97_s(
	image_t *image,
	int levels
);

int image_fdwt_interp53_s(
	image_t *image,
	int levels
);

enum wavelet_t {
	WAVELET_CDF97 = 0,
	WAVELET_CDF53,
	WAVELET_INTERP53,
	WAVELET_LAST
};

void image_idwt_s(
	image_t *image,
	int levels,
	enum wavelet_t wavelet
);

#endif