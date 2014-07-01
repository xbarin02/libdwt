#include "denoise.h"
#include "libdwt.h"
#include <assert.h>
#include <math.h>
#include "inline.h"

void dwt_util_abs2_s(
	const void *src,
	void *dst,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( src );
	assert( dst );

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float *in  = dwt_util_addr_coeff_const_s(src, y, x, stride_x, stride_y);
			      float *out = dwt_util_addr_coeff_s      (dst, y, x, stride_x, stride_y);

			*out = fabsf(*in);
		}
	}
}

float denoise_estimate_threshold(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( ptr );

	// FIXME: where is the HH(1) subband located?
	const void *subband_ptr = dwt_util_addr_coeff_const_s(ptr, 1, 1, stride_x, stride_y);

	const int subband_stride_x = mul_pow2(stride_x, 1);
	const int subband_stride_y = mul_pow2(stride_x, 1);

	const int subband_size_x = floor_div2(size_x);
	const int subband_size_y = floor_div2(size_y);

	void *magnitudes = dwt_util_alloc_image2(subband_stride_x, subband_stride_y, subband_size_x, subband_size_y);

	// magnitudes[] = abs(ptr->HH)
	dwt_util_abs2_s(
		subband_ptr, // src
		magnitudes, // dst
		subband_stride_x,
		subband_stride_y,
		subband_size_x,
		subband_size_y
	);

	// median = med(magnitudes[])
	float median = dwt_util_band_med_s(
		magnitudes,
		subband_stride_x,
		subband_stride_y,
		subband_size_x,
		subband_size_y
	);

	float sigma = median / 0.6745f;

	float lambda = sigma*sqrtf(2.f * logf(size_x*size_y));

	free(magnitudes);

	return lambda;
}
