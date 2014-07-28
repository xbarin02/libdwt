#include "gabor.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "inline.h"
#include "libdwt.h"

float complex gabor_atom(
	float t,	///< time around 0
	float alpha,	///< Gaussian envelope parameter
	float omega	///< frequency in radians
)
{
	assert( alpha >= 0.f );

#if 0
	return sqrtf(alpha/(float)M_PI) * expf(-alpha*t*t) * cexpf(-I*omega*t);
#else
	return sqrtf(alpha/(float)M_PI) * expf(-alpha*t*t) * cexpf(+I*omega*t);
#endif
}

float complex gabor_function(
	float t,	///< time, centered at 0
	float sigma,	///< standard deviation
	float f		///< frequency in radians
)
{
	assert( sigma > 0.f );

	float alpha = 1.f/2.f/sigma/sigma;

	return gabor_atom(t, alpha, f);
}

float complex gabor_wavelet(
	float t,	///< time, centered at 0
	float sigma,	///< standard deviation
	float f,	///< frequency in radians
	float a		///< scale
)
{
	assert( sigma > 0.f );
	assert( a > 0.f );

	float alpha = 1.f/2.f/sigma/sigma;

	t /= a;

	return 1.f/fabsf(a) * gabor_atom(t, alpha, f);
}

float gabor_freq(float f, float a)
{
	assert( a != 0.f );

	return f / a;
}

// f_c .. frequency of the mother wavelet
// f   .. (0; +pi]
// returns the scale "a"
float gabor_scale(float f_c, float f)
{
	assert( f != 0.f );

	return f_c / f;
}

// 3*sigma rule
float gaussian_limit(
	float sigma,
	float a
)
{
	return (4.f*sigma)*a;
}

int gaussian_size(
	float sigma,
	float a
)
{
	return (int)ceilf( 1.f + 2.f*gaussian_limit(sigma, a) );
}

int gaussian_center(
	float sigma,
	float a
)
{
	return gaussian_size(sigma, a) / 2;
}

static
int saturate_i(int val, int lo, int hi)
{
	if( val < lo )
		return lo;
	if( val > hi )
		return hi;
	return val;
}

static
float complex cdot1_s(
	const float *func,
	int func_size,
	int func_stride,
	int func_center,
	const float complex *kern,
	int kern_size,
	int kern_stride,
	int kern_center
)
{
	const int left  = -( min(
		func_center,
		kern_center) );
	const int right = +( min(
		func_size-func_center-1,
		kern_size-kern_center-1) );

	float complex sum = 0.f;

	for(int i = left; i <= right; i++)
	{
		const int func_idx = saturate_i(func_center+i, 0, func_size-1);
		const int kern_idx = saturate_i(kern_center+i, 0, kern_size-1);

		const float         func_sample = *addr1_const_s (func, func_idx, func_stride);
		const float complex kern_sample = *addr1_const_cs(kern, kern_idx, kern_stride);

		sum += func_sample * conjf(kern_sample);
	}

	return sum;
}

float complex dwt_util_cdot1_s(
	const float *func,
	int func_size,
	int func_stride,
	int func_center,
	const float complex *kern,
	int kern_size,
	int kern_stride,
	int kern_center
)
{
	return cdot1_s(func, func_size, func_stride, func_center, kern, kern_size, kern_stride, kern_center);
}

void timefreq_line(
	float *dst,
	int dst_stride,
	const float *src,
	int src_stride,
	int size,
	const float complex *kern,
	int kern_stride,
	int kern_size,
	int kern_center
)
{
	for(int i = 0; i < size; i++)
	{
		*addr1_s(dst, i, dst_stride) = cabsf(
			cdot1_s(
				src, size, src_stride, i,
				kern, kern_size, kern_stride, kern_center));
	}
}

void timefreq_arg_line(
	float *dst,
	int dst_stride,
	const float *src,
	int src_stride,
	int size,
	const float complex *kern,
	int kern_stride,
	int kern_size,
	int kern_center
)
{
	for(int i = 0; i < size; i++)
	{
		*addr1_s(dst, i, dst_stride) = cargf(
			cdot1_s(
				src, size, src_stride, i,
				kern, kern_size, kern_stride, kern_center));
	}
}

void dwt_util_vec_creal_cs(
	float *dst,
	int dst_stride,
	const float complex *src,
	int src_stride,
	int size
)
{
	for(int i = 0; i < size; i++)
	{
		*addr1_s(dst, i, dst_stride) = creal(*addr1_const_cs(src, i, src_stride));
	}
}

void dwt_util_vec_cabs_cs(
	float *dst,
	int dst_stride,
	const float complex *src,
	int src_stride,
	int size
)
{
	for(int i = 0; i < size; i++)
	{
		*addr1_s(dst, i, dst_stride) = cabs(*addr1_const_cs(src, i, src_stride));
	}
}

float dwt_util_band_lpnorm_cs(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	float p
)
{
	float sum = 0.0f;

	if( +INFINITY == p )
		dwt_util_error("unimplemented\n");

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float complex c = *addr2_const_cs(ptr, y, x, stride_x, stride_y);

			sum += powf(cabsf(c), p);
		}
	}

	return powf(sum, 1.f/p);
}

float complex dwt_util_band_mean_cs(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	float complex sum = 0.0f;

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float complex coeff = *addr2_const_cs(ptr, y, x, stride_x, stride_y);

			sum += coeff;
		}
	}

	sum /= size_x * size_y;

	return sum;
}

// Gabor transform
void gabor_gen_kernel(
	float complex **ckern,
	int stride,
	float sigma,
	float freq,
	float a
)
{
	int size = gaussian_size(sigma, a);
	int center = gaussian_center(sigma, a);

	*ckern = realloc(*ckern, stride * size);

	for(int i = 0; i < size; i++)
	{
		*addr1_cs(*ckern, i, stride) = gabor_wavelet(i-center, sigma, freq, a);
	}

#if 0
	float complex mean = dwt_util_band_mean_cs(*ckern, 0, sizeof(const float complex), size, 1);

	dwt_util_log(LOG_DBG, "gabor_gen_kernel: 1-norm=%f 2-norm=%f mean/abs=%f size=%i\n",
		dwt_util_band_lpnorm_cs(
			*ckern,
			0,
			sizeof(const float complex),
			size,
			1,
			1.f // p
		),
		dwt_util_band_lpnorm_cs(
			*ckern,
			0,
			sizeof(const float complex),
			size,
			1,
			2.f // p
		),
		cabsf(mean),
		size
	);
#endif
}

// sigma of Gaussian for S transform
float s_sigma(float f)
{
	assert( f != 0.f );
	const float alpha = f * f;
	const float sigma = sqrtf(1.f/2.f/alpha);
	return sigma;
}

// kernel of S transform
void s_gen_kernel(
	float complex **ckern,
	int stride,
	float f		///< 0..1
)
{
	// FIXME: zero frequency implies an infinitely long window
	assert( f != 0.f );

	// Gaussian function: g(t) = C1 * e^(-alpha * t^2)
	const float alpha = f * f;

	// Gaussian function: g(t) = C2 * e^(-1/2/sigma/sigma * t^2)
	const float sigma = sqrtf(1.f/2.f/alpha);

	const float omega = 2.f * (float)M_PI * f;

	// scale
	const float a = 1.f;

	const int size = gaussian_size(sigma, a);
	const int center = gaussian_center(sigma, a);

	*ckern = realloc(*ckern, stride * size);

	for(int i = 0; i < size; i++)
	{
		const int t = i-center;

		*addr1_cs(*ckern, i, stride) =
			gabor_atom(t, alpha, omega);
	}

#if 0
	float complex mean = dwt_util_band_mean_cs(*ckern, 0, sizeof(const float complex), size, 1);

	dwt_util_log(LOG_DBG, "s_gen_kernel: 1-norm=%f 2-norm=%f mean/abs=%f size=%i\n",
		dwt_util_band_lpnorm_cs(
			*ckern,
			0,
			sizeof(const float complex),
			size,
			1,
			1.f // p
		),
		dwt_util_band_lpnorm_cs(
			*ckern,
			0,
			sizeof(const float complex),
			size,
			1,
			2.f // p
		),
		cabsf(mean),
		size
	);
#endif
}

void test_signal(
	float **dest,
	int stride,
	int size,
	int type
)
{
	*dest = (float *)realloc(*dest, stride * size);

	for(int i = 0; i < size; i++)
	{
		switch(type)
		{
			// constant chirp, two parts
			case 0:
			{
				float f0 = 1.f/3.f * (size-1)/2.f;
				float f1 = 2.f/3.f * (size-1)/2.f;
				const float omega0 = 2.f * (float)M_PI * f0;
				const float omega1 = 2.f * (float)M_PI * f1;
				const float t = (float)i/size;

				*addr1_s(*dest, i, stride) = 0.f;

				if( i < 2*size/3 )
					*addr1_s(*dest, i, stride) +=
						+cos( t * omega0 );
				if( i > 1*size/3 )
					*addr1_s(*dest, i, stride) +=
						+cos( t * omega1 );
			}
				break;
			// constant chirp, two parts
			case 1:
			{
				float f0 = 1.f/3.f * (size-1)/2.f;
				float f1 = 2.f/3.f * (size-1)/2.f;
				const float omega0 = 2.f * (float)M_PI * f0;
				const float omega1 = 2.f * (float)M_PI * f1;
				const float t = (float)i/size;

				if( i < size/2 )
					*addr1_s(*dest, i, stride) =
						+cos( t * omega0 );
				else
					*addr1_s(*dest, i, stride) =
						+cos( t * omega1 );
			}
				break;
			// linear chirp
			case 2:
			{
				const float f = (size-1)/2.f;
				const float omega = 2.f * (float)M_PI * f;
				const float t = (float)i/size;

				*addr1_s(*dest, i, stride) =
					+cos( 0.5f * t * t * omega );
			}
				break;
			// two linear chirps
			case 3:
			{
				const float f = (size-1)/4.f;
				const float omega = 2.f * (float)M_PI * f;
				const float t = (float)i/size;

				*addr1_s(*dest, i, stride) =
					+cos( 0.5f * t * t * omega )
					+cos( 0.5f * t * t * omega + 0.4f*t*omega );
			}
				break;
			// two linear chirps
			case 4:
			{
				const float f = (size-1)/2.f;
				const float omega = 2.f * (float)M_PI * f;
				const float t0 = (float)i/size;
				const float t1 = 1.0f - t0;

				*addr1_s(*dest, i, stride) =
					+cos( 0.5f * t0 * t0 * omega )
					+cos( 0.5f * t1 * t1 * omega );
			}
				break;
			// hyperbolic chirp
			case 5:
			{
				const float f = (size-1)/2.f;
				const float a = 2.0f * 0.5f * 2.f * (float)M_PI * f;
				const float b = 2.0f;
				const float t = (float)i/size;

				*addr1_s(*dest, i, stride) =
					+cos( a / (b - t) );
			}
				break;
			// two hyperbolic chirps
			case 6:
			{
				const float f = (size-1)/2.f;
				const float a0 = 1.0f * 0.5f * 2.f * (float)M_PI * f;
				const float a1 = 2.0f * 0.5f * 2.f * (float)M_PI * f;
				const float b = 2.0f;
				const float t = (float)i/size;

				*addr1_s(*dest, i, stride) =
					+cos( a0 / (b - t) )
					+cos( a1 / (b - t) );
			}
				break;
			// Gabor function
			case 7:
			{
				const float center = size/2;
				const float sigma  = size/8;
				const float freq   = 0.5f;

				*addr1_s(*dest, i, stride) =
					creal(gabor_function(-center+i, sigma, freq));
			}
				break;
			// two Gabor functions
			case 8:
			{
				const float center0 = 1*size/4;
				const float center1 = 3*size/4;
				const float sigma  = size/16;
				const float freq   = 1.0f;

				*addr1_s(*dest, i, stride) =
					+creal(gabor_function(-center0+i, sigma,     freq))
					+creal(gabor_function(-center1+i, sigma, 2.f*freq));
			}
				break;
			default:
				dwt_util_abort();
		}
	}
}

void gabor_ft_s(
	// input
	const float *sig,	///< the analysed signal
	int sig_stride,		///< the stride of the signal
	int sig_size,		///< the length of the signal, the width of the plane
	// output
	void *plane,		///< put the plane here
	int stride_x,		///< stride of rows of the plane
	int stride_y,		///< stride of columns of the plane
	int bins,		///< the height of the plane
	// params
	float sigma		///< std. deviation of the baseline kernel (implies the window size)
)
{
	assert( plane );

	float complex *ckern = 0;
	int ckern_stride = sizeof(float complex);

	int size_y = bins;

	for(int y = 0; y < size_y; y++)
	{
		float norm_y = y/(float)size_y;

		// TF-plane
		float *row = dwt_util_addr_coeff_s(
			plane,
			size_y-y-1,
			0,
			stride_x,
			stride_y
		);

		// gen. kernel
		float freq = norm_y * 1.0f * (float)M_PI;
		float a = 1.0f;

		gabor_gen_kernel(&ckern, ckern_stride, sigma, freq, a);

		int kern_size = gaussian_size(sigma, a);
		int kern_center = gaussian_center(sigma, a);

		// response
		timefreq_line(row, stride_y, sig, sig_stride, sig_size, ckern, ckern_stride, kern_size, kern_center);
	}

	free(ckern);
}

void gabor_ft_arg_s(
	// input
	const float *sig,	///< the analysed signal
	int sig_stride,		///< the stride of the signal
	int sig_size,		///< the length of the signal, the width of the plane
	// output
	void *plane,		///< put the plane here
	int stride_x,		///< stride of rows of the plane
	int stride_y,		///< stride of columns of the plane
	int bins,		///< the height of the plane
	// params
	float sigma		///< std. deviation of the baseline kernel (implies the window size)
)
{
	assert( plane );

	float complex *ckern = 0;
	int ckern_stride = sizeof(float complex);

	int size_y = bins;

	for(int y = 0; y < size_y; y++)
	{
		float norm_y = y/(float)size_y;

		// TF-plane
		float *row = dwt_util_addr_coeff_s(
			plane,
			size_y-y-1,
			0,
			stride_x,
			stride_y
		);

		// gen. kernel
		float freq = norm_y * 1.0f * (float)M_PI;
		float a = 1.0f;

		gabor_gen_kernel(&ckern, ckern_stride, sigma, freq, a);

		int kern_size = gaussian_size(sigma, a);
		int kern_center = gaussian_center(sigma, a);

		// response
		timefreq_arg_line(row, stride_y, sig, sig_stride, sig_size, ckern, ckern_stride, kern_size, kern_center);
	}

	free(ckern);
}

// wavelet transform
void gabor_wt_s(
	// input
	const float *sig,	///< the analysed signal
	int sig_stride,		///< the stride of the signal
	int sig_size,		///< the length of the signal, the width of the plane
	// output
	void *plane,		///< put the plane here
	int stride_x,		///< stride of rows of the plane
	int stride_y,		///< stride of columns of the plane
	int bins,		///< the height of the plane
	// params
	float sigma,		///< std. deviation of the baseline kernel (implies the window size)
	float freq		///< frequency of the baseline kernel, in radians
)
{
	assert( plane );

// 	dwt_util_log(LOG_DBG, "analytic wavelet: (sigma^2)*(eta^2) >> 1 = %f\n", sigma*sigma*freq*freq);

	{
		const float f = 1.f/(float)bins * 0.5f * 2.f * (float)M_PI;
		const float a = gabor_scale(freq, f);
		dwt_util_log(LOG_DBG, "min: kernel_size=%i scale=%f freq=%f\n",
			gaussian_size(sigma, a), a, f);
	}
	{
		const float f = 1.f * 0.5f * 2.f * (float)M_PI;
		const float a = gabor_scale(freq, f);
		dwt_util_log(LOG_DBG, "max: kernel_size=%i scale=%f freq=%f\n",
			gaussian_size(sigma, a), a, f);
	}

	float complex *ckern = 0;
	int ckern_stride = sizeof(float complex);

	int size_y = bins;

	for(int y = 0; y < size_y; y++)
	{
		//float norm0_y = (y+0.f)/(float)size_y; // [0; 1)
		float norm1_y = (y+1.f)/(float)size_y; // (0; 1]

		// TF-plane
		float *row = dwt_util_addr_coeff_s(
			plane,
			size_y-y-1,
			0,
			stride_x,
			stride_y
		);

		// gen. kernel
		float f = norm1_y * 0.5f * 2.f*(float)M_PI; // (0; +pi]
		float a = gabor_scale(freq, f);

// 		dwt_util_log(LOG_DBG, "y=%i, freq(a=%f) = %f, freq*a = %f\n", size_y-y-1, a, gabor_freq(freq, a), gabor_freq(freq, a)*a);

		gabor_gen_kernel(&ckern, ckern_stride, sigma, freq, a);

		int kern_size = gaussian_size(sigma, a);
		int kern_center = gaussian_center(sigma, a);

		// response
		timefreq_line(row, stride_y, sig, sig_stride, sig_size, ckern, ckern_stride, kern_size, kern_center);
	}

	free(ckern);
}

// S transform
void gabor_st_s(
	// input
	const float *sig,	///< the analysed signal
	int sig_stride,		///< the stride of the signal
	int sig_size,		///< the length of the signal, the width of the plane
	// output
	void *plane,		///< put the plane here
	int stride_x,		///< stride of rows of the plane
	int stride_y,		///< stride of columns of the plane
	int bins		///< the height of the plane
	// no params
)
{
	assert( plane );

	float complex *ckern = 0;
	int ckern_stride = sizeof(float complex);

	int size_y = bins;

	for(int y = 0; y < size_y; y++)
	{
		//float norm0_y = (y+0.f)/(float)size_y;
		float norm1_y = (y+1.f)/(float)size_y;;

		// TF-plane
		float *row = dwt_util_addr_coeff_s(
			plane,
#if 1
			size_y-y-1,
#else
			y,
#endif
			0,
			stride_x,
			stride_y
		);

		// frequencies (0; 0.5] => (0; pi]
#if 1
		float f = norm1_y * 0.5f;
#else
		float f = norm1_y;
#endif
		float a = 1.f;
		float sigma = s_sigma(f);

// 		dwt_util_log(LOG_DBG, "S transform: y=%i, freq = %f\n", y, f);

		s_gen_kernel(&ckern, ckern_stride, f);

		int kern_size = gaussian_size(sigma, a);
		int kern_center = gaussian_center(sigma, a);

		// response
		timefreq_line(row, stride_y, sig, sig_stride, sig_size, ckern, ckern_stride, kern_size, kern_center);
	}

	free(ckern);
}

void gabor_st_arg_s(
	// input
	const float *sig,	///< the analysed signal
	int sig_stride,		///< the stride of the signal
	int sig_size,		///< the length of the signal, the width of the plane
	// output
	void *plane,		///< put the plane here
	int stride_x,		///< stride of rows of the plane
	int stride_y,		///< stride of columns of the plane
	int bins		///< the height of the plane
	// no params
)
{
	assert( plane );

	float complex *ckern = 0;
	int ckern_stride = sizeof(float complex);

	int size_y = bins;

	for(int y = 0; y < size_y; y++)
	{
		//float norm0_y = (y+0.f)/(float)size_y;
		float norm1_y = (y+1.f)/(float)size_y;;

		// TF-plane
		float *row = dwt_util_addr_coeff_s(
			plane,
#if 1
			size_y-y-1,
#else
			y,
#endif
			0,
			stride_x,
			stride_y
		);

		// frequencies (0; 0.5] => (0; pi]
#if 1
		float f = norm1_y * 0.5f;
#else
		float f = norm1_y;
#endif
		float a = 1.f;
		float sigma = s_sigma(f);

		s_gen_kernel(&ckern, ckern_stride, f);

		int kern_size = gaussian_size(sigma, a);
		int kern_center = gaussian_center(sigma, a);

		// response
		timefreq_arg_line(row, stride_y, sig, sig_stride, sig_size, ckern, ckern_stride, kern_size, kern_center);
	}

	free(ckern);
}

void gabor_wt_arg_s(
	// input
	const float *sig,	///< the analysed signal
	int sig_stride,		///< the stride of the signal
	int sig_size,		///< the length of the signal, the width of the plane
	// output
	void *plane,		///< put the plane here
	int stride_x,		///< stride of rows of the plane
	int stride_y,		///< stride of columns of the plane
	int bins,		///< the height of the plane
	// params
	float sigma,		///< std. deviation of the baseline kernel (implies the window size)
	float freq		///< frequency of the baseline kernel, in radians
)
{
	assert( plane );

	float complex *ckern = 0;
	int ckern_stride = sizeof(float complex);

	int size_y = bins;

	for(int y = 0; y < size_y; y++)
	{
		//float norm0_y = (y+0.f)/(float)size_y; // [0; 1)
		float norm1_y = (y+1.f)/(float)size_y; // (0; 1]

		// TF-plane
		float *row = dwt_util_addr_coeff_s(
			plane,
			size_y-y-1,
			0,
			stride_x,
			stride_y
		);

		// gen. kernel
		float f = norm1_y * 0.5f * 2.f*(float)M_PI; // (0; +pi]
		float a = gabor_scale(freq, f);

		gabor_gen_kernel(&ckern, ckern_stride, sigma, freq, a);

		int kern_size = gaussian_size(sigma, a);
		int kern_center = gaussian_center(sigma, a);

		// response
		timefreq_arg_line(row, stride_y, sig, sig_stride, sig_size, ckern, ckern_stride, kern_size, kern_center);
	}

	free(ckern);
}

void phase_derivative_s(
	const void *angle,
	void *derivative,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	float limit
)
{
	assert( limit > 0.f );

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float *out = addr2_s(derivative, y, x, stride_x, stride_y);

			if( 0 == x )
			{
				*out = 0.f;
			}
			else
			{
				*out =
					- *addr2_const_s(angle, y, x-1, stride_x, stride_y)
					+ *addr2_const_s(angle, y, x,   stride_x, stride_y);

				while( *out > +limit )
					*out -= 2.f*(float)M_PI;
				while( *out < -limit )
					*out += 2.f*(float)M_PI;
			}
		}
	}
}

void detect_ridges1_s(
	const void *magnitude,
	void *ridges,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	float threshold
)
{
	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float *out = addr2_s(ridges, y, x, stride_x, stride_y);

			*out = 0.f;

			if( x > 0 && x < size_x-1 )
			{
				float mag_val = *addr2_const_s(magnitude, y, x, stride_x, stride_y);

				float mag_line[3] = {
					*addr2_const_s(magnitude, y, x-1, stride_x, stride_y),
					*addr2_const_s(magnitude, y, x,   stride_x, stride_y),
					*addr2_const_s(magnitude, y, x+1, stride_x, stride_y),
				};

				float mag_factor = -1.f * (mag_line[0] - mag_line[1]) * (mag_line[1] - mag_line[2]);

				if( mag_factor > 0.f && mag_val > threshold )
				{
// 					dwt_util_log(LOG_DBG, "ridge point: y=%i x=%i mag=%f\n", y, x, mag_val);
#if 0
					*out = 1.f;
#else
					*out = mag_val/2.f/(float)M_PI;
#endif
				}
			}
		}
	}
}

void detect_ridges2_s(
	const void *inst_freq,
	void *ridges,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	float threshold
)
{
	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float *out = addr2_s(ridges, y, x, stride_x, stride_y);

			*out = 0.f;

			if( x > 0 && x < size_x-1 )
			{
				float arg_val = *addr2_const_s(inst_freq, y, x, stride_x, stride_y);

				if( arg_val < 0.f && fabsf(arg_val) > threshold )
				{
// 					dwt_util_log(LOG_DBG, "ridge point: y=%i x=%i arg=%f\n", y, x, arg_val);
#if 0
					*out = 1.f;
#else
					*out = fabsf(arg_val)/2.f/(float)M_PI;
#endif
				}
			}
		}
	}
}

// difference of two samples
float coeff_diff_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int pos0_x,
	int pos0_y,
	int pos1_x,
	int pos1_y
)
{
	float a = *dwt_util_addr_coeff_const_s(
		ptr,
		pos1_y,
		pos1_x,
		stride_x,
		stride_y
	);
	float b = *dwt_util_addr_coeff_const_s(
		ptr,
		pos0_y,
		pos0_x,
		stride_x,
		stride_y
	);

	return a - b;
}

// angle (-pi; +pi) of gradient at position (x,y)
float grad_angle_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int pos_x,
	int pos_y
)
{
	float dx = coeff_diff_s(ptr, stride_x, stride_y, pos_x-1, pos_y, pos_x+1, pos_y) / 2.f;
	float dy = coeff_diff_s(ptr, stride_x, stride_y, pos_x, pos_y-1, pos_x, pos_y+1) / 2.f;

// 	dwt_util_log(LOG_DBG, "angle: d=(%f,%f)\n", dx, dy);

	return atan2f(dy, dx);
}

// magnitude in the direction of gradient
const float *addr_coeff_in_grad_dir_const_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int pos_x,
	int pos_y
)
{
	float angle = grad_angle_s(
		ptr,
		stride_x,
		stride_y,
		pos_x,
		pos_y
	);

	const float ampl = 1.f;
	const float half = ampl / 2.f;

	float dir_x = ampl * cosf(angle);
	float dir_y = ampl * sinf(angle);

	// right-to-left associativity of ?:
	int next_x =
		( dir_x < -half ) ? -1 :
		( dir_x > +half ) ? +1 :
		0;
	int next_y =
		( dir_y < -half ) ? -1 :
		( dir_y > +half ) ? +1 :
		0;

// 	dwt_util_log(LOG_DBG, "pos=(%i,%i) angle=%f grad=(%i,%i)\n", pos_x, pos_y, angle, next_x, next_y);

// 	if( 0 == next_x && 0 == next_y )
// 		dwt_util_log(LOG_DBG, "pos=(%i,%i) angle=%f grad=(%i,%i) => plane\n", pos_x, pos_y, angle, next_x, next_y);

	return dwt_util_addr_coeff_const_s(
		ptr,
		pos_y+next_y,
		pos_x+next_x,
		stride_x,
		stride_y
	);
}

// is maximum in direction of gradient
int grad_max_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int pos_x,
	int pos_y
)
{
	const float this = *dwt_util_addr_coeff_const_s(
		ptr,
		pos_y,
		pos_x,
		stride_x,
		stride_y
	);
	const float next = *addr_coeff_in_grad_dir_const_s(
		ptr,
		stride_x,
		stride_y,
		pos_x,
		pos_y
	);

	return this >= next;
}

void detect_ridges3_s(
	const void *magnitude,
	void *ridges,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	float threshold
)
{
	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float *out = addr2_s(ridges, y, x, stride_x, stride_y);

			*out = 0.f;

			if( x > 0 && x < size_x-1 && y > 0 && y < size_y-1 )
			{
				float mag_val = *addr2_const_s(magnitude, y, x, stride_x, stride_y);

				const int is_max = grad_max_s(
					magnitude,
					stride_x,
					stride_y,
					x,
					y
				);

				if( is_max && mag_val > threshold )
				{
// 					dwt_util_log(LOG_DBG, "ridge point: y=%i x=%i arg=%f\n", y, x, arg_val);
#if 0
					*out = 1.f;
#else
					*out = mag_val/2.f/(float)M_PI;
#endif
				}
			}
		}
	}
}
