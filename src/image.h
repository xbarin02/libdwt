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
	int size;
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

/** Allocate image data */
int image_alloc(
	image_t *image
);

/** Free image data */
void image_free(
	image_t *image
);

/** Free image data as well as image structure itself */
void image_destroy(
	image_t *image
);

/** Allocate image structure and image data */
image_t *image_create_s(
	int size_x,
	int size_y
);

/** Allocate image structure and image data, use an optimal stride. */
image_t *image_create_opt_s(
	int size_x,
	int size_y
);

int image_load_from_mat_s(
	image_t *image,
	const char *path
);

image_t *image_create_from_mat_s(
	const char *path
);

int image_save_to_mat_s(
	image_t *image,
	const char *path
);

void image_save_to_mat_format_s(
	image_t *image,
	const char *format,
	...
);

void image_save_to_pgm_s(
	image_t *image,
	const char *path
);

void image_save_to_pgm2_s(
	image_t *image,
	const char *path
);

void image_save_to_pgm_format_s(
	image_t *image,
	const char *format,
	...
);

void image_save_log_to_pgm_s(
	image_t *image,
	const char *path
);

int image_save_to_svm_s(
	image_t *classes,
	image_t *features,
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

int *image_coeff_i(
	image_t *image,
	int y,
	int x
);

float *image_row_s(
	image_t *image,
	int y
);

int *image_row_i(
	image_t *image,
	int y
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

int image_fdwt_cdf53_s(
	image_t *image,
	int levels
);

int image_fdwt_cdf97_s(
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

int image_fdwt_s(
	image_t *image,
	int levels,
	enum wavelet_t wavelet
);

image_t *image_extend_s(
	const image_t *src,
	int pixels
);

#endif
