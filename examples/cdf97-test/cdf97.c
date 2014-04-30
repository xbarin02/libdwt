#include "libdwt.h"
#include "dwt-simple.h"

int main()
{
	const int template_type = 1;

	const int size_x = 4096;
	const int size_y = 4096;

	const int stride_y = sizeof(float);
	const int stride_x = dwt_util_get_opt_stride(stride_y * size_x);

	int j = 1;

	dwt_util_init();

	dwt_util_log(LOG_INFO, "generating test images %ix%i...\n", size_x, size_y);

	// generate a template
	void *ptr;
	dwt_util_alloc_image(&ptr, stride_x, stride_y, size_x, size_y);
	dwt_util_test_image_fill2_s(ptr, stride_x, stride_y, size_x, size_y, 0, template_type);

	// save the original image
	dwt_util_save_to_mat_s(
		"input-image.mat",
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y
	);

	// generate a template
	void *ptrH;
	dwt_util_alloc_image(&ptrH, stride_x, stride_y, size_x, size_y);
	dwt_util_test_image_fill2_s(ptrH, stride_x, stride_y, size_x, size_y, 0, template_type);

	// generate a template
	void *ptrV;
	dwt_util_alloc_image(&ptrV, stride_x, stride_y, size_x, size_y);
	dwt_util_test_image_fill2_s(ptrV, stride_x, stride_y, size_x, size_y, 0, template_type);

	// full transform (interleaved subbands)
	fdwt2_cdf97_vertical_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		&j,
		0
	);

	// save the full transform
	dwt_util_save_to_mat_s(
		"transform.mat",
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y
	);

	// horizontal transform
	fdwt2h1_cdf97_vertical_s(
		ptrH,
		size_x,
		size_y,
		stride_x,
		stride_y,
		&j,
		0
	);

	// save the horizontal transform
	dwt_util_save_to_mat_s(
		"transform-horizontal.mat",
		ptrH,
		size_x,
		size_y,
		stride_x,
		stride_y
	);

	// vertical transform
	fdwt2v1_cdf97_vertical_s(
		ptrV,
		size_x,
		size_y,
		stride_x,
		stride_y,
		&j,
		0
	);

	// save the vertical transform
	dwt_util_save_to_mat_s(
		"transform-vertical.mat",
		ptrV,
		size_x,
		size_y,
		stride_x,
		stride_y
	);

	// free memory
	dwt_util_free_image(&ptr);
	dwt_util_free_image(&ptrH);
	dwt_util_free_image(&ptrV);

	dwt_util_finish();

	return 0;
}
