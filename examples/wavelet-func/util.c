/**
 * @brief Utility functions.
 */

#include "util.h"
#include "signal.h"
#include "libdwt.h"

void dwt_util_convolve1_s(
	// output response
	void *y_ptr,
	int y_stride,
	int y_size,
	int y_center,
	// input signal
	const void *x_ptr,
	int x_stride,
	int x_size,
	int x_center,
	// kernel
	const void *g_ptr,
	int g_stride,
	int g_size,
	int g_center,
	// parameters
	int y_downsample_factor,
	int g_upsample_factor
)
{
	signal_t *y_signal = signal_create(y_ptr, y_stride, y_size, y_center);
	signal_const_t *x_signal = signal_const_create(x_ptr, x_stride, x_size, x_center);
	signal_const_t *g_signal = signal_const_create(g_ptr, g_stride, g_size, g_center);

	for(int y_idx = signal_left(y_signal); y_idx <= signal_right(y_signal); y_idx++)
	{
		//dwt_util_log(LOG_DBG, "y_idx=%i (y_idx0=%i)\n", y_idx, signal_get_s(y_signal, y_idx)-(float *)y_ptr);

		float *y_coeff = signal_get_s(y_signal, y_idx);

		*y_coeff = 0.f;

		for(int g_idx = signal_const_left(g_signal); g_idx <= signal_const_right(g_signal); g_idx++)
		{
			//float x_coeff  = *signal_const_get_s(x_signal, g_upsample_factor*((y_downsample_factor*y_idx-g_idx)));
			float x_coeff  = *signal_const_get_s(x_signal, (y_downsample_factor*y_idx-g_upsample_factor*g_idx));
			float g_coeff  = *signal_const_get_s(g_signal, g_idx);

			*y_coeff += x_coeff * g_coeff;
		}
	}

	signal_destroy(y_signal);
	signal_const_destroy(x_signal);
	signal_const_destroy(g_signal);
}
