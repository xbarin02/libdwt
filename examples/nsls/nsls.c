// [Iwahashi2010] IWAHASHI, KIYA: A NEW LIFTING STRUCTURE OF NON SEPARABLE 2D DWT WITH COMPATIBILITY TO JPEG 2000. ICASSP 2010.

#include "libdwt.h"
#include "image.h"
#include "inline.h"
#include "dwt-simple.h"

// P-PP-U-S
void core_pppu(
	float *green,
	float *blue_top,
	float *blue_left,
	float *red
)
{
	// P
	{
		// square += +1/4*circle -1/2*cross -1/2*triangle
		red[0] +=
			+0.25f * ( green[3] + blue_top[3] + blue_left[3] + red[3] )
			-0.5f  * ( blue_top[2] + red[2] )
			-0.5f  * ( blue_left[1] + red[1] );
	}
	// PP
	{
		// cross += -1/2*circle +1/4*square
		blue_top[2] +=
			-0.5f  * ( green[3] + blue_top[3] )
			+0.25f * ( blue_top[0] + red[0] );

		// triangle += -1/2*circle +1/4*square
		blue_left[1] +=
			-0.5f  * ( green[3] + blue_left[3] )
			+0.25f * ( blue_left[0] + red[0] );
	}
	// U
	{
		// circle += +1/4*cross +1/4*triangle -1/16*square
		green[3] +=
			+0.25f   * ( green[2] + blue_top[2] )
			+0.25f   * ( green[1] + blue_left[1] )
			-0.0625f * ( green[0] + blue_top[0] + blue_left[0] + red[0] );
	}
	// S
	{
		green[0] *= 0.5f;
		green[1] *= 1.0f;
		green[2] *= 1.0f;
		green[3] *= 2.0f;
	}
}

void fdwt_cdf53_nsls(image_t *image)
{
#if 0
	// separable
	int off = 0;

	// H P (for each row)
	for(int x = 1+off; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y; y++)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) -= 0.50f*(
			*addr2_s(image->ptr, y, x-1, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y, x+1, image->stride_x, image->stride_y)
		);
	}

	// H U (for each row)
	for(int x = 2+off; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y; y++)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += 0.25f*(
			*addr2_s(image->ptr, y, x-1, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y, x+1, image->stride_x, image->stride_y)
		);
	}

	// V P (for each column)
	for(int x = 0; x < image->size_x; x++)
	for(int y = 1+off; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) -= 0.50f*(
			*addr2_s(image->ptr, y-1, x, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y+1, x, image->stride_x, image->stride_y)
		);
	}

	// V U (for each column)
	for(int x = 0; x < image->size_x; x++)
	for(int y = 2+off; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += 0.25f*(
			*addr2_s(image->ptr, y-1, x, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y+1, x, image->stride_x, image->stride_y)
		);
	}

	// S
	for(int x = 0; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y) *= 2;
		*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y) *= 0.5;
	}
#endif
#if 0
	// non-separable
	int off = 0;

	// P
	for(int x = 1+off; x < image->size_x-1; x += 2)
	for(int y = 1+off; y < image->size_y-1; y += 2)
	{
		// HH = (y=0,x=0)
		*addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y) += (
		// 0.5 * 0.5 * LL(diag)
			+0.25 * (
				*addr2_s(image->ptr, y-1, x-1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y-1, x+1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1, x-1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y)
			)
		// 0.5 * LH(vert)
			-0.5 * (
				*addr2_s(image->ptr, y-1, x, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1, x, image->stride_x, image->stride_y)
			)
		// 0.5 * HL(horiz)
			-0.5 * (
				*addr2_s(image->ptr, y, x-1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y, x+1, image->stride_x, image->stride_y)
			)
		);
	}

	// P+P
	for(int x = 1+off; x < image->size_x-1; x += 2)
	for(int y = 1+off; y < image->size_y-1; y += 2)
	{
		// LH = (y=1,x=0)
		*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y) += (
			// 0.5 * LL(horiz)
			-0.5 * (
				*addr2_s(image->ptr, y+1, x-1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y)
			)
			// 0.25 * HH(vert)
			+0.25 * (
				*addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+2, x+0, image->stride_x, image->stride_y)
			)
		);
		// HL = (y=0,x=1)
		*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y) += (
			// 0.5 * LL(vert)
			-0.5 * (
				*addr2_s(image->ptr, y-1, x+1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y)
			)
			// 0.25 * HH(horiz)
			+0.25 * (
				*addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+0, x+2, image->stride_x, image->stride_y)
			)
		);
	}

	// U
	for(int x = 1+off; x < image->size_x-1; x += 2)
	for(int y = 1+off; y < image->size_y-1; y += 2)
	{
		// LL = (y=1,x=1)
		*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y) += (
			// 0.25 * LH(horiz)
			+0.25 * (
				*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1, x+2, image->stride_x, image->stride_y)
			)
			// 0.25 * HL(vert)
			+0.25 * (
				*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+2, x+1, image->stride_x, image->stride_y)
			)
			// -0.0625 * HH(diag)
			-0.0625 * (
				*addr2_s(image->ptr, y+1-1, x+1-1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1-1, x+1+1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1+1, x+1-1, image->stride_x, image->stride_y) +
				*addr2_s(image->ptr, y+1+1, x+1+1, image->stride_x, image->stride_y)
			)
		);
	}

	// S
	for(int x = 0; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y) *= 2;
		*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y) *= 0.5;
	}
#endif
#if 1
	const int off = 1;

	// online/pipelined
	float short_buffer[4]; // green
	float input_buffer[4]; // red
	float long_buffer[2*image->size_x]; // blue

	dwt_util_zero_vec_s(input_buffer, 4);
	dwt_util_zero_vec_s(long_buffer, 2*image->size_x);

	// for each 2col
	for(int y = off+2; y < image->size_y-1; y += 2)
	{
		dwt_util_zero_vec_s(short_buffer, 4);

		// for each 2row
		for(int x = off+2; x < image->size_x-1; x += 2)
		{
			// pointer into blue buffer
			float *long_buffer_ptr_left = long_buffer+2*(x-2); // buffer+{0..3}
			float *long_buffer_ptr_top  = long_buffer+2*(x+0);

			// call core
			{
				dwt_util_log(LOG_DBG, "core @ (%i,%i)\n", y, x);

				// read input into red buffer from (y,x)
				{
					input_buffer[0] = *addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y);
					input_buffer[1] = *addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y);
					input_buffer[2] = *addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y);
					input_buffer[3] = *addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y);
				}
				// perform P-2P-U-S
				core_pppu(
					short_buffer,
					long_buffer_ptr_top,
					long_buffer_ptr_left,
					input_buffer
				);
				// write green buffer at (y-2,x-2)
				{
					const int shift = 2;
					*addr2_s(image->ptr, y+0-shift, x+0-shift, image->stride_x, image->stride_y) = short_buffer[0];
					*addr2_s(image->ptr, y+0-shift, x+1-shift, image->stride_x, image->stride_y) = short_buffer[1];
					*addr2_s(image->ptr, y+1-shift, x+0-shift, image->stride_x, image->stride_y) = short_buffer[2];
					*addr2_s(image->ptr, y+1-shift, x+1-shift, image->stride_x, image->stride_y) = short_buffer[3];
				}
				// move green<-blue<-red (red is now empty)
				{
					// green<-blue_top
					short_buffer[0] = long_buffer_ptr_top[0];
					short_buffer[1] = long_buffer_ptr_top[1];
					short_buffer[2] = long_buffer_ptr_top[2];
					short_buffer[3] = long_buffer_ptr_top[3];
					// blue_top<-red
					long_buffer_ptr_top[0] = input_buffer[0];
					long_buffer_ptr_top[1] = input_buffer[1];
					long_buffer_ptr_top[2] = input_buffer[2];
					long_buffer_ptr_top[3] = input_buffer[3];
				}
			}
		}
	}
#endif
}

int main()
{
	dwt_util_init();

	const int size_x = 512;
	const int size_y = 512;
	const int J = 1;

	// alloc test image
	image_t *image = image_create_opt_s(size_x, size_y);
	dwt_util_test_image_fill2_s(image->ptr, image->stride_x, image->stride_y, image->size_x, image->size_y, 0, 1);

	// clone image
	image_t *ref = image_create_opt_s(size_x, size_y);
	image_copy(image, ref);

	image_save_to_pgm2_s(image, "image.pgm");

	// fwd cdf 5/3
#if 0
	{
		int j = J;
		fdwt2_cdf53_horizontal_s(image->ptr, image->size_x, image->size_y, image->stride_x, image->stride_y, &j, 0);
	}
#else
	fdwt_cdf53_nsls(image);
#endif

	image_save_log_to_pgm_s(image, "transform.pgm");

#if 0
	// HACK: compare transforms
	{
		int j = J;
		fdwt2_cdf53_horizontal_s(ref->ptr, ref->size_x, ref->size_y, ref->stride_x, ref->stride_y, &j, 0);
	}
	dwt_util_compare2_destructive_s(
		image->ptr, // destroy this
		ref->ptr, // const
		image->stride_x,
		image->stride_y,
		ref->stride_x,
		ref->stride_y,
		size_x,
		size_y
	);
	image_save_to_pgm2_s(image, "transform-diff.pgm");
#endif

	// inv cdf 5/3
#if 1
	dwt_cdf53_2i_inplace_s(
		image->ptr,
		image->stride_x,
		image->stride_y,
		image->size_x,
		image->size_y,
		image->size_x,
		image->size_y,
		J,
		0,
		0
	);
#endif

	image_save_to_pgm2_s(image, "result.pgm");

	// compare images
	int res = dwt_util_compare2_s(
		image->ptr,
		ref->ptr,
		image->stride_x,
		image->stride_y,
		ref->stride_x,
		ref->stride_y,
		size_x,
		size_y
	);

	if( res )
		dwt_util_log(LOG_ERR, "fail\n");
	else
		dwt_util_log(LOG_INFO, "pass\n");

	dwt_util_finish();

	return 0;
}
