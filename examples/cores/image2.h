#ifndef IMAGE2_H
#define IMAGE2_H

#include "image.h" // struct image_t
#include <stddef.h> // size_t

enum dwt_types {
	TYPE_INT32,
	TYPE_FLOAT32,
	TYPE_FIX32,
	TYPE_FIX16,
	TYPE_LAST
};

size_t sizeof_type(enum dwt_types data_type);

struct image_t *image2_create_ex(size_t size_pel, int size_x, int size_y, int opt_stride);

void image2_conv_float32_to_fix32(struct image_t *source, struct image_t *target);

void image2_conv_float32_to_fix16(struct image_t *source, struct image_t *target);

void image2_conv_fix32_to_float32(struct image_t *source, struct image_t *target);

void image2_conv_fix16_to_float32(struct image_t *source, struct image_t *target);

void image2_fill(struct image_t *image, enum dwt_types data_type);

void image2_save_to_pgm_s(struct image_t *image, const char *path);

void image2_save_to_pgm(struct image_t *image, const char *path, enum dwt_types data_type);

void image2_save_log_to_pgm_s(struct image_t *image, const char *path);

int image2_save_log_to_pgm_i(struct image_t *image, const char *path);

void image2_save_log_to_pgm(struct image_t *image, const char *path, enum dwt_types data_type);

void image2_idwt_cdf97_ip(struct image_t *image, enum dwt_types data_type);

void image2_idwt_cdf53_ip(struct image_t *image, enum dwt_types data_type);

void image2_idwt_cdf97_op(struct image_t *source, struct image_t *target, enum dwt_types data_type);

void image2_fdwt_cdf97_op(struct image_t *source, struct image_t *target, enum dwt_types data_type);

void image2_fdwt_cdf53_op(struct image_t *source, struct image_t *target, enum dwt_types data_type);

int image2_compare(struct image_t *source, struct image_t *target, enum dwt_types data_type);

int image2_compare_map(struct image_t *source, struct image_t *target, struct image_t *map, enum dwt_types data_type);

void image2_flush_cache(struct image_t *image);

float image2_mse(struct image_t *source, struct image_t *target, enum dwt_types data_type);

#endif
