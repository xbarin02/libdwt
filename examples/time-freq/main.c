/**
 * @brief Time-frequency analysis by a naive cross-correlation.
 */
#include "libdwt.h"
#include "gabor.h"
#include <math.h> // M_PI
#include <stdlib.h>
#include <complex.h>

void demo(
	float *test_ptr,
	int test_stride,
	int test_size,
	float sigma,
	float freq,
	float a
)
{
	int size = gaussian_size(sigma, a);
	int center = gaussian_center(sigma, a);

	dwt_util_log(LOG_INFO, "kernel: sigma=%f freq=%f size=%i\n", sigma, freq, size);

	// complex kernel
	float complex *ckern = 0;
	int ckern_stride = sizeof(float complex);

	gabor_gen_kernel(&ckern, sizeof(float complex), sigma, freq, a);

	// save the real component of the kernel
	float kern[size];

	dwt_util_vec_creal_cs(kern, sizeof(float), ckern, ckern_stride, size);

	dwt_util_save_to_mat_s(
		"kern.mat",
		kern,
		size,
		1,
		0,
		sizeof(float)
	);

	// a response on the test signal
	float y[test_size];

	timefreq_line(y, sizeof(float), test_ptr, test_stride, test_size, ckern, ckern_stride, size, center);

	dwt_util_save_to_mat_s(
		"out.mat",
		y,
		test_size,
		1,
		0,
		sizeof(float)
	);

	free(ckern);
}

int main()
{
	// generate a test signal
	int test_size = 512;
	float *test_ptr = NULL;
	int test_stride = sizeof(float);
	test_signal(&test_ptr, test_stride, test_size, 6);

	// save the test signal
	dwt_util_save_to_mat_s(
		"sig.mat",
		test_ptr,
		test_size,
		1,
		0,
		test_stride
	);

	demo(
		test_ptr,
		test_stride,
		test_size,
		/* sigma = */ 10.0f,
		/* freq = */  (float)M_PI,
		/* a = */     1.0f
	);

	// TF plane

	int bins = 256;

	const int size_x = test_size;
	const int size_y = bins;
	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(stride_y * size_x);
	void *data = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);

	// STFT
	dwt_util_log(LOG_INFO, "FT...\n");

	gabor_ft_s(
		// input
		test_ptr,
		test_stride,
		test_size,
		// output
		data,
		stride_x,
		stride_y,
		bins,
		// params
		/* sigma = */ 20.0f
	);

	dwt_util_save_log_to_pgm_s(
		"plane-ft.pgm",
		data,
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	// CWT
	dwt_util_log(LOG_INFO, "WT...\n");

	gabor_wt_s(
		// input
		test_ptr,
		test_stride,
		test_size,
		// output
		data,
		stride_x,
		stride_y,
		bins,
		// params
		/* sigma = */        10.0f, // 10; 2
		/* freq = */         0.75f*(float)M_PI // pi; pi; pi/2
	);

	dwt_util_save_log_to_pgm_s(
		"plane-wt.pgm",
		data,
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	// S transform
	dwt_util_log(LOG_INFO, "ST...\n");

	gabor_st_s(
		// input
		test_ptr,
		test_stride,
		test_size,
		// output
		data,
		stride_x,
		stride_y,
		bins
		// params
	);

	dwt_util_save_log_to_pgm_s(
		"plane-st.pgm",
		data,
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	dwt_util_free_image(&data);
	free(test_ptr);
}
