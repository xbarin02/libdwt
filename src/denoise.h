#ifndef DENOISE_H
#define DENOISE_H

float denoise_estimate_threshold(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

#endif
