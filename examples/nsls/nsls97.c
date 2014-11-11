// [Iwahashi2010] IWAHASHI, KIYA: A NEW LIFTING STRUCTURE OF NON SEPARABLE 2D DWT WITH COMPATIBILITY TO JPEG 2000. ICASSP 2010.

#include "libdwt.h"
#include "image.h"
#include "inline.h"
#include "dwt-simple.h"

// H1 = -1.58613434342059f
#define H1 (-1.58613434342059f)
// H2 = -0.0529801185729f
#define H2 (-0.0529801185729f)
// H3 = +0.8829110755309f
#define H3 (+0.8829110755309f)
// H4 = +0.4435068520439f
#define H4 (+0.4435068520439f)
// H11 = +H1*H1 = +(-1.58613434342059f * -1.58613434342059f) = +2.51582215538f
#define H11 (+2.51582215538f)
// H22 = -H2*H2 = -(-0.0529801185729f  * -0.0529801185729f ) = -0.00280689296f
#define H22 (-0.00280689296f)
// H33 = +H3*H3 = +(+0.8829110755309f  * +0.8829110755309f ) = +0.77953196729f
#define H33 (+0.77953196729f)
// H44 = -H4*H4 = -(+0.4435068520439f  * +0.4435068520439f ) = -0.19669832781f
#define H44 (-0.19669832781f)

void op_star(image_t *image, int y, int x, float d, float h, float v)
{
	*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += (
		+h * (
			*addr2_s(image->ptr, y+0, x-1, image->stride_x, image->stride_y) +
			*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y)
		)
		+v * (
			*addr2_s(image->ptr, y-1, x+0, image->stride_x, image->stride_y) +
			*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y)
		)
		+d * (
			*addr2_s(image->ptr, y-1, x-1, image->stride_x, image->stride_y) +
			*addr2_s(image->ptr, y-1, x+1, image->stride_x, image->stride_y) +
			*addr2_s(image->ptr, y+1, x-1, image->stride_x, image->stride_y) +
			*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y)
		)
	);
}

void op_cross(image_t *image, int y, int x, float h, float v)
{
	*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += (
		+h * (
			*addr2_s(image->ptr, y+0, x-1, image->stride_x, image->stride_y) +
			*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y)
		)
		+v * (
			*addr2_s(image->ptr, y-1, x+0, image->stride_x, image->stride_y) +
			*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y)
		)
	);
}

void copy4(float *dst, const float *src)
{
	for(int i = 0; i < 4; i++)
		dst[i] = src[i];
}

void core(
	float *red, // 4 = input
	float *green, // 4 = output
	float *blue0, // 8
	float *blue1, // 8
	float *blue2, // 8
	float *yellow // 4
)
{
	// 1 P
	{
		// HH = star(y+0,x+0) (d/LL*H11, v/LH*H1, h/HL*H1)
		red[0] +=
			+ H1  * blue2[6] // N  = v
			+ H11 * blue2[7] // NE = d
			+ H1  * red[1]   // E  = h
			+ H11 * red[3]   // SE = d
			+ H1  * red[2]   // S  = v
			+ H11 * blue1[7] // SW = d
			+ H1  * blue1[5] // W  = h
			+ H11 * blue1[3] // NW = d
			;
	}
	// 1 PP
	{
		// LH = cross(y+1,x+0) (h/LL*H1, v/HH*H2)
		blue2[6] +=
			+ H2 * blue2[4] // N = v
			+ H1 * blue2[7] // E = h
			+ H2 * red[0]   // S = v
			+ H1 * blue1[3] // W = h
			;
		// HL = cross(y+0,x+1) (v/LL*H1, h/HH*H2)
		blue1[5] +=
			+ H1 * blue1[3] // N = v
			+ H2 * red[0]   // E = h
			+ H1 * blue1[7] // S = v
			+ H2 * blue1[4] // W = h
			;
	}
	// 1 U
	{
		// LL = op_star(y=1,x=1) (h/LH*H2, v/HL*H2, d/HH*H22)
		blue1[3] +=
			+ H2  * blue1[1] // N  = v
			+ H22 * blue2[4] // NE = d
			+ H2  * blue2[6] // E  = h
			+ H22 * red[0]   // SE = d
			+ H2  * blue1[5] // S  = v
			+ H22 * blue1[4] // SW = d
			+ H2  * blue1[2] // W  = h
			+ H22 * blue1[0] // NW = d
			;
	}
	// 2 P
	{
		// HH = star(y+0,x+0) (d/LL*H33, v/LH*H3, h/HL*H3)
		blue1[0] +=
			+ H3  * yellow[2] // N  = v
			+ H33 * yellow[3] // NE = d
			+ H3  * blue1[1]  // E  = h
			+ H33 * blue1[3]  // SE = d
			+ H3  * blue1[2]  // S  = v
			+ H33 * blue0[3]  // SW = d
			+ H3  * blue0[1]  // W  = h
			+ H33 * green[3]  // NW = d
			;
	}
	// 2 PP
	{
		// LH = cross(y+1,x+0) (h/LL*H3, v/HH*H4)
		yellow[2] +=
			+ H4 * yellow[0] // N  = v
			+ H3 * yellow[3] // E  = h
			+ H4 * blue1 [0] // S  = v
			+ H3 * green [3] // W  = h
			;
		// HL = cross(y+0,x+1) (v/LL*H3, h/HH*H4)
		blue0[1] +=
			+ H3 * green[3] // N  = v
			+ H4 * blue1[0] // E  = h
			+ H3 * blue0[3] // S  = v
			+ H4 * blue0[0] // W  = h
			;
	}
	// 2 U
	{
		// LL = op_star(y=1,x=1) (h/LH*H4, v/HL*H4, d/HH*H44)
		green[3] +=
			+ H4  * green[1] // N  = v
			+ H44 * yellow[0] // NE = d
			+ H4  * yellow[2] // E  = h
			+ H44 * blue1[0] // SE = d
			+ H4  * blue0[1] // S  = v
			+ H44 * blue0[0] // SW = d
			+ H4  * green[2] // W  = h
			+ H44 * green[0] // NW = d
			;
	}
	// S
	{
		green[0] *= 1.f/(1.1496043988602f*1.1496043988602f);
		green[1] *= 1.0f;
		green[2] *= 1.0f;
		green[3] *= 1.f*(1.1496043988602f*1.1496043988602f);
	}
}

void fdwt_cdf97_nsls(image_t *image)
{
#if 0
	// separable
	int off = 0;

	// NOTE

	// H P (for each row)
	for(int x = 1+off; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y; y++)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) -= 1.58613434342059f*(
			*addr2_s(image->ptr, y, x-1, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y, x+1, image->stride_x, image->stride_y)
		);
	}

	// H U (for each row)
	for(int x = 2+off; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y; y++)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += -0.0529801185729f*(
			*addr2_s(image->ptr, y, x-1, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y, x+1, image->stride_x, image->stride_y)
		);
	}

	// H P (for each row)
	for(int x = 1+off; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y; y++)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) -= -0.8829110755309f*(
			*addr2_s(image->ptr, y, x-1, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y, x+1, image->stride_x, image->stride_y)
		);
	}

	// H U (for each row)
	for(int x = 2+off; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y; y++)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += 0.4435068520439f*(
			*addr2_s(image->ptr, y, x-1, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y, x+1, image->stride_x, image->stride_y)
		);
	}

	// NOTE

	// V P (for each column)
	for(int x = 0; x < image->size_x; x++)
	for(int y = 1+off; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) -= 1.58613434342059f*(
			*addr2_s(image->ptr, y-1, x, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y+1, x, image->stride_x, image->stride_y)
		);
	}

	// V U (for each column)
	for(int x = 0; x < image->size_x; x++)
	for(int y = 2+off; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += -0.0529801185729f*(
			*addr2_s(image->ptr, y-1, x, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y+1, x, image->stride_x, image->stride_y)
		);
	}

	// V P (for each column)
	for(int x = 0; x < image->size_x; x++)
	for(int y = 1+off; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) -= -0.8829110755309f*(
			*addr2_s(image->ptr, y-1, x, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y+1, x, image->stride_x, image->stride_y)
		);
	}

	// V U (for each column)
	for(int x = 0; x < image->size_x; x++)
	for(int y = 2+off; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y, x, image->stride_x, image->stride_y) += 0.4435068520439f*(
			*addr2_s(image->ptr, y-1, x, image->stride_x, image->stride_y)
			+
			*addr2_s(image->ptr, y+1, x, image->stride_x, image->stride_y)
		);
	}

	// NOTE

	// S
	for(int x = 0; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y) *= 1.f*(1.1496043988602f*1.1496043988602f);
		*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y) *= 1.f/(1.1496043988602f*1.1496043988602f);
	}
#endif
#if 0
	// non-separable

	// 1 P
	for(int x = 1; x < image->size_x-1; x += 2)
	for(int y = 1; y < image->size_y-1; y += 2)
	{
		// HH = star(y+0,x+0) (d/LL*H11, v/LH*H1, h/HL*H1)
		op_star(image, y+0, x+0, H11, H1, H1);
	}

	// 1 PP
	for(int x = 1; x < image->size_x-1; x += 2)
	for(int y = 1; y < image->size_y-1; y += 2)
	{
		// LH = cross(y+1,x+0) (h/LL*H1, v/HH*H2)
		op_cross(image, y+1, x+0, H1, H2);
		// HL = cross(y+0,x+1) (v/LL*H1, h/HH*H2)
		op_cross(image, y+0, x+1, H2, H1);
	}

	// 1 U
	for(int x = 1; x < image->size_x-1; x += 2)
	for(int y = 1; y < image->size_y-1; y += 2)
	{
		// LL = op_star(y=1,x=1) (h/LH*H2, v/HL*H2, d/HH*H22)
		op_star(image, y+1, x+1, H22, H2, H2);
	}

	// 2 P
	for(int x = 1; x < image->size_x-1; x += 2)
	for(int y = 1; y < image->size_y-1; y += 2)
	{
		// HH = star(y+0,x+0) (LL*H33, LH*H3, HL*H3)
		op_star(image, y+0, x+0, H33, H3, H3);
	}

	// 2 PP
	for(int x = 1; x < image->size_x-1; x += 2)
	for(int y = 1; y < image->size_y-1; y += 2)
	{
		// LH = cross(y+1,x+0) (LL*H3, HH*H4)
		op_cross(image, y+1, x+0, H3, H4);
		// HL = cross(y+0,x+1) (LL*H3, HH*H4)
		op_cross(image, y+0, x+1, H4, H3);
	}

	// 2 U
	for(int x = 1; x < image->size_x-1; x += 2)
	for(int y = 1; y < image->size_y-1; y += 2)
	{
		// LL = op_star(y=1,x=1) (h/LH*H4, v/HL*H4, d/HH*H44)
		op_star(image, y+1, x+1, H44, H4, H4);
	}

	// S
	for(int x = 0; x < image->size_x-1; x += 2)
	for(int y = 0; y < image->size_y-1; y += 2)
	{
		*addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y) *= 1.f*(1.1496043988602f*1.1496043988602f);
		*addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y) *= 1;
		*addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y) *= 1.f/(1.1496043988602f*1.1496043988602f);
	}
#endif
#if 1
	// non-separable core

	const int off = 1;
	const int shift = 4;

	float blue[4*image->size_x]; // x+=2 && ptr=blue[4*x] => ptr[0..7]
	float red[4];
	float yellow[4];
	float green[4];

	dwt_util_zero_vec_s(blue, 4*image->size_x);

	// for each 2col
	for(int y = off+shift; y < image->size_y-1; y += 2)
	{
		dwt_util_zero_vec_s(green, 4);
		dwt_util_zero_vec_s(yellow, 4);

		// for each 2row
		for(int x = off+shift; x < image->size_x-1; x += 2)
		{
			float *blue0 = blue+4*(x-4);
			float *blue1 = blue+4*(x-2);
			float *blue2 = blue+4*(x+0);

// 			dwt_util_log(LOG_DBG, "core @ (%i,%i)\n", y, x);

			// read input into red buffer from (y,x)
			{
				red[0] = *addr2_s(image->ptr, y+0, x+0, image->stride_x, image->stride_y);
				red[1] = *addr2_s(image->ptr, y+0, x+1, image->stride_x, image->stride_y);
				red[2] = *addr2_s(image->ptr, y+1, x+0, image->stride_x, image->stride_y);
				red[3] = *addr2_s(image->ptr, y+1, x+1, image->stride_x, image->stride_y);
			}
			// perform lifting
			{
				core(
					red, // 4 = input
					green, // 4 = output
					blue0, // 8
					blue1, // 8
					blue2, // 8
					yellow // 4
				);
			}
			// write green buffer at (y-shift,x-shift)
			{
				*addr2_s(image->ptr, y+0-shift, x+0-shift, image->stride_x, image->stride_y) = green[0];
				*addr2_s(image->ptr, y+0-shift, x+1-shift, image->stride_x, image->stride_y) = green[1];
				*addr2_s(image->ptr, y+1-shift, x+0-shift, image->stride_x, image->stride_y) = green[2];
				*addr2_s(image->ptr, y+1-shift, x+1-shift, image->stride_x, image->stride_y) = green[3];
			}
			// shuffle buffers
			{
				// NOTE: next <- curr
				// NOTE: copy4(dst, src);

				// green <- yellow
				copy4(green, yellow);
				// yellow <- blue2.top
				copy4(yellow, blue2+0);
				// blue
				copy4(blue2+0, blue2+4);
				// blue <- red
				copy4(blue2+4, red);
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

	// fwd cdf 9/7
#if 0
	{
		int j = J;
		fdwt2_cdf97_horizontal_s(image->ptr, image->size_x, image->size_y, image->stride_x, image->stride_y, &j, 0);
	}
#else
	fdwt_cdf97_nsls(image);
#endif

	image_save_log_to_pgm_s(image, "transform.pgm");

#if 0
	// HACK: compare transforms
	{
		int j = J;
		fdwt2_cdf97_horizontal_s(ref->ptr, ref->size_x, ref->size_y, ref->stride_x, ref->stride_y, &j, 0);
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

	// inv cdf 9/7
#if 1
	dwt_cdf97_2i_inplace_s(
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
