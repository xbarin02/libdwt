#include "eaw-experimental.h"
#include "libdwt.h"
#include "inline.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#ifdef _OPENMP
	#include <omp.h>
#endif

/**
 * @brief Copy memory area.
 *
 * This function copies @p n floats from memory area @p src to memory area
 * @p dst. Memory areas can be sparse. The strides (in bytes) are determined by
 * @p stride_dst and @p stride_src arguments.
 *
 * @returns The function returns a pointer to @p dst.
 */
static
void *dwt_util_memcpy_stride_s(
	void *restrict dst,
	ssize_t stride_dst,
	const void *restrict src,
	ssize_t stride_src,
	size_t n		///< Number of floats to be copied, not number of bytes.
)
{
	assert( NULL != dst && NULL != src );

	const size_t size = sizeof(float);

	if( (ssize_t)size == stride_src && (ssize_t)size == stride_dst )
	{
		memcpy(dst, src, n*size);
	}
	else
	{
		char *restrict ptr_dst = (char *restrict)dst;
		const char *restrict ptr_src = (const char *restrict)src;
		for(size_t i = 0; i < n; i++)
		{
			*(float *restrict)ptr_dst = *(const float *restrict)ptr_src;
	
			ptr_dst += stride_dst;
			ptr_src += stride_src;
		}
	}

	return dst;
}

static
float dwt_eaw_w(float n, float m, float alpha)
{
	const float eps = 1.0e-5f;

	return 1.f / (powf(fabsf(n-m), alpha) + eps);
}

static
void dwt_calc_eaw_w(float *w, float *arr, int N, float alpha)
{
	for(int i = 0; i < N-1; i++)
	{
		w[i] = dwt_eaw_w(arr[i], arr[i+1], alpha);
	}
	w[N-1] = 0.f; // not necessary
}

void dwt_eaw97_f_ex_stride_s(
	const float *src,
	float *dst_l,
	float *dst_h,
	float *tmp,
	int N,
	int stride,
	float *w,
	float alpha
)
{
	assert( N >= 0 && NULL != src && NULL != dst_l && NULL != dst_h && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst_l[0] = src[0] * dwt_cdf97_s1_s;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_s(tmp, sizeof(float), src, stride, N);

	dwt_calc_eaw_w(w, tmp, N, alpha);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] -= (wL * tmp[i-1] + wR * tmp[i+1]) / (wL+wR) * (2.f*dwt_cdf97_p1_s);
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] += (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_u1_s);
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] -= (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_p1_s);
	}

	{
		float wL = w[0];
		float wR = w[0];

		tmp[0] += (wL * tmp[1] + wR * tmp[1]) / (wL+wR) * (2.f*dwt_cdf97_u1_s);
	}

	for(int i=2; i<N-(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] += (wL * tmp[i-1] + wR * tmp[i+1]) / (wL+wR) * (2.f*dwt_cdf97_u1_s);
	}

	// predict 2 + update 2
	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] -= (wL * tmp[i-1] + wR * tmp[i+1]) / (wL+wR) * (2.f*dwt_cdf97_p2_s);
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] += (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_u2_s);
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] -= (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_p2_s);
	}

	{
		float wL = w[0];
		float wR = w[0];

		tmp[0] += (wL * tmp[1] + wR * tmp[1]) / (wL+wR) * (2.f*dwt_cdf97_u2_s);
	}

	for(int i=2; i<N-(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] += (wL * tmp[i-1] + wR * tmp[i+1]) / (wL+wR) * (2.f*dwt_cdf97_u2_s);
	}

	// scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s1_s;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s2_s;

	// copy tmp into dst
	dwt_util_memcpy_stride_s(dst_l, stride, tmp+0, 2*sizeof(float),  ceil_div2(N));
	dwt_util_memcpy_stride_s(dst_h, stride, tmp+1, 2*sizeof(float), floor_div2(N));
}

void dwt_eaw97_i_ex_stride_s(
	const float *src_l,
	const float *src_h,
	float *dst,
	float *tmp,
	int N,
	int stride,
	float *w
)
{
	assert( N >= 0 && NULL != src_l && NULL != src_h && NULL != dst && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst[0] = src_l[0] * dwt_cdf97_s2_s;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_s(tmp+0, 2*sizeof(float), src_l, stride,  ceil_div2(N));
	dwt_util_memcpy_stride_s(tmp+1, 2*sizeof(float), src_h, stride, floor_div2(N));

	// inverse scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s2_s;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s1_s;

	// backward update 2 + backward predict 2
	for(int i=2; i<N-(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] -= ( wL*tmp[i-1] + wR*tmp[i+1] ) / (wL+wR) * (2.f*dwt_cdf97_u2_s);
	}

	{
		float wL = w[0];
		float wR = w[0];

		tmp[0] -= (wL * tmp[1] + wR * tmp[1]) / (wL+wR) * (2.f*dwt_cdf97_u2_s);
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];
		
		tmp[N-1] -= (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_u2_s);
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];
		
		tmp[N-1] += (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_p2_s);
	}

	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] += ( wL*tmp[i-1] + wR*tmp[i+1] ) / (wL+wR) * (2.f*dwt_cdf97_p2_s);
	}

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] -= ( wL*tmp[i-1] + wR*tmp[i+1] ) / (wL+wR) * (2.f*dwt_cdf97_u1_s);
	}

	{
		float wL = w[0];
		float wR = w[0];

		tmp[0] -= (wL * tmp[1] + wR * tmp[1]) / (wL+wR) * (2.f*dwt_cdf97_u1_s);
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];
		
		tmp[N-1] -= (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_u1_s);
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];
		
		tmp[N-1] += (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR) * (2.f*dwt_cdf97_p1_s);
	}

	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] += ( wL*tmp[i-1] + wR*tmp[i+1] ) / (wL+wR) * (2.f*dwt_cdf97_p1_s);
	}

	// copy tmp into dst
	dwt_util_memcpy_stride_s(dst, stride, tmp, sizeof(float), N);
}

void dwt_eaw97_2f_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding,
	float *wH[],
	float *wV[],
	float alpha
)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	float temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

// 		dwt_util_log(LOG_DBG, "FWD-EAW-5/3: j = %i with wH[%i] wV[%i]\n", j, j, j);

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		wH[j] = dwt_util_alloc(size_o_src_y * size_i_src_x, sizeof(float));
		wV[j] = dwt_util_alloc(size_o_src_x * size_i_src_y, sizeof(float));

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_src_y; y++)
			dwt_eaw97_f_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x, // N
				stride_y,
				&wH[j][y*size_i_src_x],
				alpha
			);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_eaw97_f_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp,
				size_i_src_y, // N
				stride_x,
				&wV[j][x*size_i_src_y],
				alpha
			);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_s(
					addr2_s(ptr,y,0,stride_x,stride_y),
					addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_s(
					addr2_s(ptr,0,x,stride_x,stride_y),
					addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}
}

void dwt_eaw97_2i_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding,
	float *wH[],
	float *wV[]
)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	float temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

// 		dwt_util_log(LOG_DBG, "INV-EAW-5/3: j = %i with wH[%i] wV[%i]\n", j, j-1, j-1);

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_eaw97_i_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y, // N
				stride_x,
				&wV[j-1][x*size_i_dst_y]
			);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_dst_y; y++)
			dwt_eaw97_i_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x, // N
				stride_y,
				&wH[j-1][y*size_i_dst_x]
			);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_dst_y; y++)
				dwt_zero_padding_i_stride_s(
					addr2_s(ptr,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_s(
					addr2_s(ptr,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}
}
