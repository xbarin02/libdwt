#include "core-int.h"
#include <stdint.h>
#include "image.h"

static
int virt2real(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		real *= -1;
	if( real > size-1 )
		real = 2*(size-1) - real;

	return real;
}

static
int virt2real_error(int pos, int offset, int overlap, int size)
{
	int real = pos + offset - overlap;

	if( real < 0 )
		return -1;
	if( real > size-1 )
		return -1;

	return real;
}

static
int32_t *get_pixel(struct image_t *image, unsigned x, unsigned y)
{
	return (int32_t *)( (intptr_t)image->ptr + x*image->stride_x + y*image->stride_y );
}

static
int32_t op(int32_t l, int32_t r, int32_t w, int s)
{
	const int k = 1<<(s-1);

	return ( w*(l+r) + k ) >> s;
}

static
void vert_2x1(
        int32_t *data0, // left [1]
        int32_t *data1, // right [1]
        int32_t *buff // [4]
)
{
	// weights
	const int32_t w[4] = { +1817, +113, -217, -203 }; // [ u2 p2 u1 p1 ]
	// shifts
	const int s[4] = { 12, 7, 12, 7 }; // [ u2 p2 u1 p1 ]

	// variables
	int32_t c[4];
	int32_t r[4];
	int32_t x0, x1;
	int32_t y0, y1;

	// load
	int32_t *l = buff;

	// inputs
	x0 = *data0;
	x1 = *data1;

	// shuffles
	y0   = l[0];
	c[0] = l[1];
	c[1] = l[2];
	c[2] = l[3];
	c[3] = x0;

	// operation
	r[3] = x1;
	r[2] = c[3] + op(l[3], r[3], w[3], s[3]);
	r[1] = c[2] + op(l[2], r[2], w[2], s[2]);
	r[0] = c[1] + op(l[1], r[1], w[1], s[1]);
	y1   = c[0] + op(l[0], r[0], w[0], s[0]);

	// update
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	// outputs
	*data0 = y0;
	*data1 = y1;
}

static
void core_vert2x2(
	struct image_t *src,
	struct image_t *dst,
	int x,
	int y,
	int32_t *buffer_x_ptr,
	int32_t *buffer_y_ptr
)
{
	int overlap_x_L = 5;
	int overlap_y_L = 5;

	const int step_y = 2;
	const int step_x = 2;

	const int shift = 4;

	// 2x2
	int32_t t[4];

	// load
	for(int xx = 0; xx < step_x; xx++)
        {
                for(int yy = 0; yy < step_y; yy++)
                {
                        // virtual to real coordinates
                        const int pos_x = virt2real(x, xx, overlap_x_L, src->size_x);
                        const int pos_y = virt2real(y, yy, overlap_y_L, src->size_y);

                        t[yy*step_x+xx] = *get_pixel(src, pos_x, pos_y);
                }
        }

	// calc
	vert_2x1(t+0, t+1, buffer_y_ptr+0);
	vert_2x1(t+2, t+3, buffer_y_ptr+4);
	vert_2x1(t+0, t+2, buffer_x_ptr+0);
	vert_2x1(t+1, t+3, buffer_x_ptr+4);

	// store
        for(int yy = 0; yy < step_y; yy++)
        {
                for(int xx = 0; xx < step_x; xx++)
                {
                        // virtual to real coordinates
                        const int pos_x = virt2real_error(x-shift, xx, overlap_x_L, src->size_x);
                        const int pos_y = virt2real_error(y-shift, yy, overlap_y_L, src->size_y);
                        if( pos_x < 0 || pos_y < 0 )
                                continue;

                        *get_pixel(dst, pos_x, pos_y) = t[yy*step_x+xx];
                }
        }
}

void dwt_cdf97_2f_vert2x2_i(
	void *src_ptr,
	int src_stride_x,
	int src_stride_y,
	void *dst_ptr,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	image_t src = (image_t){ .ptr = src_ptr, .size_x = size_x, .size_y = size_y, .stride_x = src_stride_x, .stride_y = src_stride_y };
	image_t dst = (image_t){ .ptr = dst_ptr, .size_x = size_x, .size_y = size_y, .stride_x = dst_stride_x, .stride_y = dst_stride_y };

	int overlap_x_L = 5;
	int overlap_y_L = 5;
	int overlap_x_R = 5;
	int overlap_y_R = 5;

	int super_x = overlap_x_L + size_x + overlap_x_R;
	int super_y = overlap_y_L + size_y + overlap_y_R;

	const int buff_elem_size = 4;

	int32_t buffer_x[buff_elem_size*super_x];
	int32_t buffer_y[buff_elem_size*super_y];

	int step_x = 2;
	int step_y = 2;

	for(int y = 0; y < super_y; y += step_y)
		for(int x = 0; x < super_x; x += step_x)
			core_vert2x2(&src, &dst, x, y, buffer_x+x*buff_elem_size, buffer_y+y*buff_elem_size);
}
