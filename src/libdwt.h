/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Fast wavelet transform implemented via lifting scheme.
 */
#ifndef LIBDWT_H
#define LIBDWT_H

#include <stdint.h> // int64_t
#include <stdio.h> // FILE
#include <stdarg.h> // va_list
#include <stddef.h> // size_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup c_dwt C interface
 * @{
 **/

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 *
 * @deprecated Use @ref dwt_cdf97_f_ex_d instead.
 */
void dwt_cdf97_f_d(
	const double *src,	///< input signal of the length @e N
	double *dst,		///< output signal of the length @e N, i.e. one level of discrete wavelet transform, the L and H channels are concatenated
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 *
 * @deprecated Use @ref dwt_cdf53_f_ex_d instead.
 */
void dwt_cdf53_f_d(
	const double *src,	///< input signal of the length @e N
	double *dst,		///< output signal of the length @e N, i.e. one level of discrete wavelet transform, the L and H channels are concatenated
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @deprecated Use @ref dwt_cdf97_f_ex_s instead.
 */
void dwt_cdf97_f_s(
	const float *src,	///< input signal of the length @e N
	float *dst,		///< output signal of the length @e N, i.e. one level of discrete wavelet transform, the L and H channels are concatenated
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @deprecated Use @ref dwt_cdf53_f_ex_s instead.
 */
void dwt_cdf53_f_s(
	const float *src,	///< input signal of the length @e N
	float *dst,		///< output signal of the length @e N, i.e. one level of discrete wavelet transform, the L and H channels are concatenated
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 *
 * @deprecated Use @ref dwt_cdf97_i_ex_d instead.
 */
void dwt_cdf97_i_d(
	const double *src,	///< input signal of the length @e N, i.e. the discrete wavelet transform
	double *dst,		///< output signal of the length @e N, i.e. the reconstructed signal
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 *
 * @deprecated Use @ref dwt_cdf53_i_ex_d instead.
 */
void dwt_cdf53_i_d(
	const double *src,	///< input signal of the length @e N, i.e. the discrete wavelet transform
	double *dst,		///< output signal of the length @e N, i.e. the reconstructed signal
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @deprecated Use @ref dwt_cdf97_i_ex_s instead.
 */
void dwt_cdf97_i_s(
	const float *src,	///< input signal of the length @e N, i.e. the discrete wavelet transform
	float *dst,		///< output signal of the length @e N, i.e. the reconstructed signal
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @deprecated Use @ref dwt_cdf53_i_ex_s instead.
 */
void dwt_cdf53_i_s(
	const float *src,	///< input signal of the length @e N, i.e. the discrete wavelet transform
	float *dst,		///< output signal of the length @e N, i.e. the reconstructed signal
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf97_f_ex_d(
	const double *src,	///< input signal of the length @e N
	double *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	double *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the input signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf53_f_ex_d(
	const double *src,	///< input signal of the length @e N
	double *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	double *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the input signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_f_ex_s(
	const float *src,	///< input signal of the length @e N
	float *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	float *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the input signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf53_f_ex_s(
	const float *src,	///< input signal of the length @e N
	float *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	float *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the input signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 9/7 wavelet performed on column.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf97_f_ex_stride_d(
	const double *src,	///< input signal of the length @e N
	double *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	double *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the input signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes between two neighboring pixels, no matter if in the row or column, common for @e src, @e dst_l and @e dst_h
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 5/3 wavelet performed on column.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf53_f_ex_stride_d(
	const double *src,	///< input signal of the length @e N
	double *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	double *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the input signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes between two neighboring pixels, no matter if in the row or column, common for @e src, @e dst_l and @e dst_h
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 9/7 wavelet performed on column.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_f_ex_stride_s(
	const float *src,	///< input signal of the length @e N
	float *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	float *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the input signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes between two neighboring pixels, no matter if in the row or column, common for @e src, @e dst_l and @e dst_h
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 5/3 wavelet performed on column.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf53_f_ex_stride_i(
	const int *src,		///< input signal of the length @e N
	int *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	int *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	int *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the input signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes between two neighboring pixels, no matter if in the row or column, common for @e src, @e dst_l and @e dst_h
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 9/7 wavelet performed on column.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf97_f_ex_stride_i(
	const int *src,		///< input signal of the length @e N
	int *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	int *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	int *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the input signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes between two neighboring pixels, no matter if in the row or column, common for @e src, @e dst_l and @e dst_h
);

/**
 * @brief Lifting implementation of one level of fast wavelet transform using CDF 5/3 wavelet performed on column.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf53_f_ex_stride_s(
	const float *src,	///< input signal of the length @e N
	float *dst_l,		///< output L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	float *dst_h,		///< output H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the input signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes between two neighboring pixels, no matter if in the row or column, common for @e src, @e dst_l and @e dst_h
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf97_i_ex_d(
	const double *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const double *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *dst,		///< reconstructed (output) signal
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the reconstructed (output) signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf53_i_ex_d(
	const double *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const double *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *dst,		///< reconstructed (output) signal
	double *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the reconstructed (output) signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_i_ex_s(
	const float *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const float *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *dst,		///< reconstructed (output) signal
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the reconstructed (output) signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf53_i_ex_s(
	const float *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const float *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *dst,		///< reconstructed (output) signal
	float *tmp,		///< temporary memory space of the length @e N
	int N			///< length of the reconstructed (output) signal, odd or even length
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf97_i_ex_stride_d(
	const double *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const double *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *dst,		///< reconstructed (output) signal
	double *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the reconstructed (output) signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst, @e src_l and @e src_h
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf53_i_ex_stride_d(
	const double *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const double *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	double *dst,		///< reconstructed (output) signal
	double *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the reconstructed (output) signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst, @e src_l and @e src_h
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_i_ex_stride_s(
	const float *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const float *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *dst,		///< reconstructed (output) signal
	float *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the reconstructed (output) signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst, @e src_l and @e src_h
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf53_i_ex_stride_i(
	const int *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const int *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	int *dst,		///< reconstructed (output) signal
	int *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the reconstructed (output) signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst, @e src_l and @e src_h
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 9/7 wavelet.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf97_i_ex_stride_i(
	const int *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const int *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	int *dst,		///< reconstructed (output) signal
	int *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the reconstructed (output) signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst, @e src_l and @e src_h
);

/**
 * @brief Lifting implementation of one level of fast inverse wavelet transform using CDF 5/3 wavelet.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf53_i_ex_stride_s(
	const float *src_l,	///< input L (low pass) channel of length @f$ \lfloor N/2 \rfloor @f$ for even @e N or @f$ \lfloor N/2 \rfloor + 1 @f$ for odd @e N
	const float *src_h,	///< input H (high pass) channel of length @f$ \lfloor N/2 \rfloor @f$
	float *dst,		///< reconstructed (output) signal
	float *tmp,		///< temporary memory space of the length @e N
	int N,			///< length of the reconstructed (output) signal, odd or even length
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst, @e src_l and @e src_h
);

/**
 * @brief Fill padding in L and H signals with zeros. Useful for nice looking output. Suitable for decomposition.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_zero_padding_f_d(
	double *dst_l,		///< L (low pass) channel which will be padded with zeros
	double *dst_h,		///< H (high pass) channel which will be padded with zeros
	int N,			///< length of a signal used for decomposition into the L and H channels, odd or even length
	int N_dst_L,		///< length of the L channel
	int N_dst_H		///< length of the H channel
);

/**
 * @brief Fill padding in L and H signals with zeros. Useful for nice looking output. Suitable for decomposition.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_zero_padding_f_s(
	float *dst_l,		///< L (low pass) channel which will be padded with zeros
	float *dst_h,		///< H (high pass) channel which will be padded with zeros
	int N,			///< length of a signal used for decomposition into the L and H channels, odd or even length
	int N_dst_L,		///< length of the L channel
	int N_dst_H		///< length of the H channel
);

/**
 * @brief Fill padding in L and H signals with zeros. Useful for nice looking output. Suitable for decomposition.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_zero_padding_f_stride_d(
	double *dst_l,		///< L (low pass) channel which will be padded with zeros
	double *dst_h,		///< H (high pass) channel which will be padded with zeros
	int N,			///< length of a signal used for decomposition into the L and H channels, odd or even length
	int N_dst_L,		///< length of the L channel
	int N_dst_H,		///< length of the H channel
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst_l and @e dst_h
);

/**
 * @brief Fill padding in L and H signals with zeros. Useful for nice looking output. Suitable for decomposition.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_zero_padding_f_stride_s(
	float *dst_l,		///< L (low pass) channel which will be padded with zeros
	float *dst_h,		///< H (high pass) channel which will be padded with zeros
	int N,			///< length of a signal used for decomposition into the L and H channels, odd or even length
	int N_dst_L,		///< length of the L channel
	int N_dst_H,		///< length of the H channel
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst_l and @e dst_h
);

/**
 * @brief Fill padding in L and H signals with zeros. Useful for nice looking output. Suitable for decomposition.
 *
 * This function works with integers (i.e. int  data type).
 */
void dwt_zero_padding_f_stride_i(
	int *dst_l,		///< L (low pass) channel which will be padded with zeros
	int *dst_h,		///< H (high pass) channel which will be padded with zeros
	int N,			///< length of a signal used for decomposition into the L and H channels, odd or even length
	int N_dst_L,		///< length of the L channel
	int N_dst_H,		///< length of the H channel
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels, common for @e dst_l and @e dst_h
);

/**
 * @brief Fill padding in L signal with zeros. Useful for nice looking output. Suitable for reconstruction.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_zero_padding_i_d(
	double *dst_l,		///< L (low pass) channel which will be padded with zeros
	int N,			///< length of composed L (actual usage)
	int N_dst		///< total length of the L channel
);

/**
 * @brief Fill padding in L signal with zeros. Useful for nice looking output. Suitable for reconstruction.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_zero_padding_i_s(
	float *dst_l,		///< L (low pass) channel which will be padded with zeros
	int N,			///< length of composed L (actual usage)
	int N_dst		///< total length of the L channel
);

/**
 * @brief Fill padding in L signal with zeros. Useful for nice looking output. Suitable for reconstruction.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_zero_padding_i_stride_d(
	double *dst_l,		///< L (low pass) channel which will be padded with zeros
	int N,			///< length of composed L (actual usage)
	int N_dst,		///< total length of the L channel
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels
);

/**
 * @brief Fill padding in L signal with zeros. Useful for nice looking output. Suitable for reconstruction.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_zero_padding_i_stride_s(
	float *dst_l,		///< L (low pass) channel which will be padded with zeros
	int N,			///< length of composed L (actual usage)
	int N_dst,		///< total length of the L channel
	int stride		///< image stride, i.e. the number of bytes from one row of pixels to the next row of pixels
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf97_2f_d(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf53_2f_d(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_2f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works on image data itself, i.e. no data is copied. That has the consequence the DWT subbands are interleaved in place of the original image.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * This function implements single-loop (SL) approach using double-loop (DL) horizontally and double-loop (DL) vertically.
 *
 * @warning experimental
 */
void dwt_cdf97_2f_inplace_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

void dwt_cdf53_2f_inplace_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

void dwt_cdf97_2f_inplace_sep_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

void dwt_cdf97_2f_inplace_sep_sdl_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works on image data itself, i.e. no data is copied. That has the consequence the DWT subbands are interleaved in place of the original image.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * This function implements single-loop (SL) approach using shifted double-loop (SDL) horizontally and shifted double-loop (SDL) vertically.
 *
 * @warning experimental
 */
void dwt_cdf97_2f_inplace_sdl_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, out-of-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_2f_s2(
	const void *src,	///< source image
	void *dst,		///< destination image, already allocated by caller
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf53_2f_i(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf97_2f_i(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf53_2f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using WCDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * See <a href="http://www.cs.huji.ac.il/~raananf/projects/eaw/">Edge-Avoiding Wavelets page</a>.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_eaw53_2f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	float *wH[],		///< array (according to @e j_max_ptr) of pointers to prediction weights for horizontal passes (data will be allocated by this function)
	float *wV[],		///< array (according to @e j_max_ptr) of pointers to prediction weights for vertical passes (data will be allocated by this function)
	float alpha
);

void dwt_eaw53_2f_inplace_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	float *wH[],		///< array (according to @e j_max_ptr) of pointers to prediction weights for horizontal passes (data will be allocated by this function)
	float *wV[],		///< array (according to @e j_max_ptr) of pointers to prediction weights for vertical passes (data will be allocated by this function)
	float alpha
);

/**
 * @brief Dummy implementation of forward image fast wavelet transform.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf53_2f_dummy_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one	///< should be row or column of size one pixel decomposed? zero value if not
);

/**
 * @brief Dummy implementation of forward image fast wavelet transform.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_eaw53_2f_dummy_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one	///< should be row or column of size one pixel decomposed? zero value if not
);

/**
 * @brief Forward image fast wavelet transform using CDF 5/3 wavelet and lifting scheme without update step, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_interp53_2f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf97_2i_d(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_cdf53_2i_d(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_2i_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works on image data itself, i.e. no data is copied. That has the consequence the DWT subbands are interleaved in place of the original image.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf97_2i_inplace_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

void dwt_cdf97_1i_inplace_s(
	void *ptr,
	int stride,
	int size,
	int j_max
);

void dwt_cdf97_2i_inplace_hole_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

void dwt_cdf97_2i_inplace_zero_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works on image data itself, i.e. no data is copied. That has the consequence the DWT subbands are interleaved in place of the original image.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf53_2i_inplace_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, out-of-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf97_2i_s2(
	const void *src,	///< pointer to beginning of the source image data (keeps unaffected)
	void *dst,		///< pointer to beginning of the destination image data (have to be allocated already)
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf53_2i_i(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf97_2i_i(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version, interleaved subbands.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf97_2i_inplace_i(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward image fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version, interleaved subbands.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_cdf97_2f_inplace_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding
);

/**
 * @brief Inverse image fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_cdf53_2i_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse image fast wavelet transform using WCDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * See <a href="http://www.cs.huji.ac.il/~raananf/projects/eaw/">Edge-Avoiding Wavelets page</a>.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_eaw53_2i_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	float *wH[],		///< same as for @ref dwt_eaw53_2f_s
	float *wV[]		///< same as for @ref dwt_eaw53_2f_s
);

void dwt_eaw53_2i_inplace_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	float *wH[],		///< same as for @ref dwt_eaw53_2f_s
	float *wV[]		///< same as for @ref dwt_eaw53_2f_s
);

/**
 * @brief Inverse image fast wavelet transform using CDF 5/3 wavelet and lifting scheme without update step, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_interp53_2i_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< pointer to the number of achieved decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward 1-D fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf97_1f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward 1-D fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf53_1f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Forward 1-D fast wavelet transform using CDF 5/3 wavelet without update step and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_interp53_1f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

void dwt_interp2_1f_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse 1-D fast wavelet transform using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf97_1i_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int j_max,		///< the number of decomposition levels (scales)
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse 1-D fast wavelet transform using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf53_1i_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int j_max,		///< the number of decomposition levels (scales)
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Inverse 1-D fast wavelet transform using CDF 5/3 wavelet without update step and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_interp53_1i_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int j_max,		///< the number of decomposition levels (scales)
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Series of forward 1-D fast wavelet transforms using CDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf97_2f1_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Series of forward 1-D fast wavelet transforms using CDF 5/3 wavelet and lifting scheme, in-place version.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @warning experimental
 */
void dwt_cdf53_2f1_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int *j_max_ptr,		///< pointer to the number of intended decomposition levels (scales), the number of achieved decomposition levels will be stored also here
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @}
 */

/**
 * @defgroup c_dwt_util C utility interface
 * @{
 **/

/**
 * @brief Smallest integer not less than the base 2 logarithm of x, i.e. @f$ \lceil log_2(x) \rceil @f$.
 */
int dwt_util_ceil_log2(
	int x			///< input value
);

/**
 * @brief Closest power of two greater or equal to x, i.e. @f$ 2^{\lceil log_2(x) \rceil} @f$.
 */
int dwt_util_pow2_ceil_log2(
	int x			///< input value
);

/**
 * @brief Fill image with test pattern.
 *
 * This function works with double precision floating point numbers (i.e. double data type).
 */
void dwt_util_test_image_fill_d(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int rand		///< random seed
);

/**
 * @brief Fill image with test pattern.
 *
 * This function works with integers (i.e. int data type).
 */
void dwt_util_test_image_fill_i(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int rand		///< random seed
);

void dwt_util_test_image_fill2_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y,
	int rand,
	int type
);

/**
 * @brief Fill image with test pattern.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_util_test_image_fill_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int rand		///< random seed
);

/**
 * @brief Fill image with test pattern.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_util_test_image_fill2_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int rand,		///< random seed
	int type		///< type of pattern
);

/**
 * @brief Fill image with zeros.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_util_test_image_zero_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Allocate image.
 *
 * Allocates memory for image of given sizes.
 */
void dwt_util_alloc_image(
	void **pptr,		///< place pointer to newly allocated data here
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y	///< height of outer image frame (in elements)
);

/**
 * @brief Allocate image.
 *
 * Allocates memory for image of given sizes.
 */
void *dwt_util_alloc_image2(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Free image.
 *
 * Frees memory allocated by @ref dwt_util_alloc_image.
 */
void dwt_util_free_image(
	void **pptr		///< pointer to data that will be released
);

/**
 * @brief Compare two images.
 *
 * This function compares two images and returns zero value if they equal.
 * This function works with double precision floating point numbers (i.e. double data type).
 */
int dwt_util_compare_d(
	void *ptr1,		///< pointer to data of first image
	void *ptr2,		///< pointer to data of second image
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Compare two images.
 *
 * This function compares two images and returns zero value if they equal.
 * This function works with integers (i.e. int data type).
 */
int dwt_util_compare_i(
	void *ptr1,		///< pointer to data of first image
	void *ptr2,		///< pointer to data of second image
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Compare two images.
 *
 * This function compares two images and returns zero value if they equal.
 * This function works with integers (i.e. int data type).
 */
int dwt_util_compare2_i(
	void *ptr1,
	void *ptr2,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int size_x,
	int size_y
);

/**
 * @brief Compare two images.
 *
 * This function compares two images and returns zero value if they equal.
 * This function works with single precision floating point numbers (i.e. float data type).
 */
int dwt_util_compare_s(
	void *ptr1,		///< pointer to data of first image
	void *ptr2,		///< pointer to data of second image
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

int dwt_util_compare2_s(
	void *ptr1,
	void *ptr2,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int size_x,
	int size_y
);

int dwt_util_compare2_destructive_s(
	void *ptr1,
	const void *ptr2,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int size_x,
	int size_y
);

int dwt_util_compare2_destructive2_s(
	void *ptr1,
	const void *ptr2,
	void *map,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int map_stride_x,
	int map_stride_y,
	int size_x,
	int size_y
);

/**
 * @brief Difference of two images.
 */
void dwt_util_diff_i(
	const void *src0,	///< pointer to data of first image
	const void *src1,	///< pointer to data of second image
	void *dst,		///< pointer to data of destination
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Copy one image into another.
 */
void dwt_util_copy_s(
	const void *src,	///< pointer to data of source image
	void *dst,		///< pointer to (already allocated) data of destination
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Copy one image into another.
 */
void dwt_util_copy3_s(
	const void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
);

/**
 * @brief Copy one image into another.
 */
void dwt_util_copy_i(
	const void *src,	///< pointer to data of source image
	void *dst,		///< pointer to (already allocated) data of destination
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Returns library version string.
 */
const char *dwt_util_version();

/**
 * @brief Architecture we are running on.
 */
const char *dwt_util_arch();

/**
 * @brief Timer types.
 *
 * Timer sources used in @ref dwt_util_clock_available, @ref dwt_util_get_frequency and @ref dwt_util_get_clock functions.
 */
enum dwt_timer_types
{
	DWT_TIME_CLOCK_GETTIME,				///< use clock_gettime() function from <time.h>; defined by POSIX; with appropriate argument
	DWT_TIME_CLOCK_GETTIME_REALTIME,		///< use clock_gettime() function from <time.h>; defined by POSIX; with argument CLOCK_REALTIME
	DWT_TIME_CLOCK_GETTIME_MONOTONIC,		///< use clock_gettime() function from <time.h>; defined by POSIX; with argument CLOCK_MONOTONIC
	DWT_TIME_CLOCK_GETTIME_MONOTONIC_RAW,		///< use clock_gettime() function from <time.h>; defined by POSIX; with argument CLOCK_MONOTONIC_RAW
	DWT_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID,	///< use clock_gettime() function from <time.h>; defined by POSIX; with argument CLOCK_PROCESS_CPUTIME_ID
	DWT_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID,	///< use clock_gettime() function from <time.h>; defined by POSIX; with argument CLOCK_THREAD_CPUTIME_ID
	DWT_TIME_CLOCK,					///< use clock() function from <time.h>; defined since C89 and by POSIX
	DWT_TIME_TIMES,					///< use times() function from <sys/times.h>; defined by POSIX; only user time is considered
	DWT_TIME_GETRUSAGE,				///< use getrusage() function from <sys/resource.h>; defined by POSIX; with appropriate argument
	DWT_TIME_GETRUSAGE_SELF,			///< use getrusage() function from <sys/resource.h>; defined by POSIX; with argument RUSAGE_SELF
	DWT_TIME_GETRUSAGE_CHILDREN,			///< use getrusage() function from <sys/resource.h>; defined by POSIX; with argument RUSAGE_CHILDREN
	DWT_TIME_GETRUSAGE_THREAD,			///< use getrusage() function from <sys/resource.h>; defined by POSIX; with argument RUSAGE_THREAD
	DWT_TIME_GETTIMEOFDAY,				///< use gettimeofday() function from <sys/time.h>; defined by POSIX
	DWT_TIME_IOCTL_RTC,				///< use ioctl() function from <sys/ioctl.h>; with argument RTC_RD_TIME; available on Linux except EdkDSP platform
	DWT_TIME_AUTOSELECT				///< autoselect appropriate timer
};

/**
 * @brief Indicate if the given clock type is available.
 *
 * @returns Return 0 if the clock @p type is available, or -1 if is not.
 */
int dwt_util_clock_available(
	int type		///< timer type, see @ref dwt_timer_types
);

/**
 * @brief Integer type for storing time or number of clock ticks.
 *
 * This type is used by @ref dwt_util_get_frequency and @ref dwt_util_get_clock functions.
 */
typedef int64_t dwt_clock_t;

/**
 * @brief Autoselect appropriate timer.
 *
 * Try to select appropriate timer according to system dispositions.
 */
int dwt_util_clock_autoselect();

/**
 * @brief Number of ticks per second.
 *
 * This function returns a number of clock ticks per second according to indicated clock source type.
 */
dwt_clock_t dwt_util_get_frequency(
	int type		///< timer type, see @ref dwt_timer_types
);

/**
 * @brief Number of ticks from certain event.
 *
 * This function returns a number of clock ticks from some event (e.g. system boot). Used clock source is indicated by @p type argument.
 *
 * Example usage:
 * @code
 * int type = dwt_util_clock_autoselect();
 *
 * dwt_clock_t start = dwt_util_get_clock(type);
 * // some computation...
 * dwt_clock_t stop = dwt_util_get_clock(type);
 *
 * double elapsed_time_in_seconds = (stop - start)/(double)dwt_util_get_frequency(type);
 * @endcode
 */
dwt_clock_t dwt_util_get_clock(
	int type		///< timer type, see @ref dwt_timer_types
);

/**
 * @brief Wrapper to @p omp_get_max_threads function.
 *
 * Returns the maximum number of threads what can be used in parallel region.
 *
 * @warning experimental
 */
int dwt_util_get_max_threads();

/**
 * @brief Get the maximum number of workers available.
 */
int dwt_util_get_max_workers();

/**
 * @brief Wrapper to @p omp_set_num_threads function.
 *
 * Sets the number of threads that will be used in parallel region.
 *
 * @warning experimental
 */
void dwt_util_set_num_threads(
	int num_threads		///< the number of threads
);

/**
 * @brief Set the number of active workers.
 *
 * On ASVP platform, the worker represents a BCE unit.
 * On PC platform, workers correspond to elements of SIMD registers processed in parallel.
 */
void dwt_util_set_num_workers(
	int num_workers		///< the number of workers
);

/**
 * @brief Set algorithm for acceleration of lifting scheme.
 *
 * On all platforms, select from one several loop algorithms.
 * On UTIA ASVP/EdkDSP platform, enable block-acceleration using workers.
 * On x86 architecture, enable SIMD acceleration using SSE instruction set.
 * In the list bellow, SSE implementation means manually coded function using compiler intrinsics.
 *
 * @param[in] accel_type Means
 *   @li  0 for CPU multi-loop algorithm,
 *   @li  1 for BCE multi-loop algorithm (ASVP platform),
 *   @li  2 for empty implementation (for performance measurement),
 *   @li  3 for BCE multi-loop algorithm on whole data vector (testing purposes, incorrect results, ASVP platform),
 *   @li  4 for CPU double-loop algorithm,
 *   @li  5 for CPU shifted double-loop algorithm (reference implementation),
 *   @li  6 for CPU shifted double-loop algorithm (2 iterations merged),
 *   @li  7 for CPU shifted double-loop algorithm (6 iterations merged),
 *   @li  8 for CPU shifted double-loop algorithm (2 iterations merged, SSE implementation, x86 platform),
 *   @li  9 for CPU shifted double-loop algorithm (6 iterations merged, SSE implementation, x86 platform),
 *   @li 10 for CPU double-loop algorithm (4 workers),
 *   @li 11 for CPU double-loop algorithm (4 workers, SSE implementation, x86 platform),
 *   @li 12 for CPU multi-loop algorithm (4 workers, SSE implementation, x86 platform),
 *   @li 13 for CPU multi-loop algorithm (SSE disabled),
 *   @li 14 for CPU double-loop algorithm (SSE disabled),
 *   @li 15 for CPU double-loop algorithm in groups of 4,
 *   @li 16 for CPU double-loop algorithm in groups of 4 (SSE implementation).
 *
 * @note This function currently affects only single precision floating point CDF 9/7 transform.
 * @warning experimental
 */
void dwt_util_set_accel(
	int accel_type);

/**
 * @brief Get acceleration algorithm identifier.
 *
 * @returns Values set by @ref dwt_util_set_accel function.
 *
 * @warning experimental
 */
int dwt_util_get_accel();

/**
 * @brief Initialize workers in UTIA ASVP platform.
 */
void dwt_util_init();

/**
 * @brief Release all resources allocated in @ref dwt_util_init function.
 */
void dwt_util_finish();

/**
 * @brief Save grayscale image into ASCII-type PGM file.
 *
 * See <a href="http://netpbm.sourceforge.net/">the home page for Netpbm</a>.
 * This function works with integers (i.e. int data type).
 *
 * @note Use @ref dwt_util_conv_show_i function before this function call to save a transform.
 */
int dwt_util_save_to_pgm_i(
	const char *filename,	///< target file name, e.g. "output.pgm"
	int max_value, 		///< maximum value of pixel, e.g. 255 if image values lie inside an interval [0; 255]
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Save grayscale image into ASCII-type PGM file.
 *
 * See <a href="http://netpbm.sourceforge.net/">the home page for Netpbm</a>.
 * This function works with single precision floating point numbers (i.e. float data type).
 *
 * @note Use @ref dwt_util_conv_show_s function before this function call to save a transform.
 */
int dwt_util_save_to_pgm_s(
	const char *filename,	///< target file name, e.g. "output.pgm"
	float max_value, 	///< maximum value of pixel, e.g. 1.0f if image values lie inside an interval [0.0; 1.0]
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Save a logarithm of spectrum magnutudes into an ASCII-type PGM file.
 *
 * See <a href="http://netpbm.sourceforge.net/">the home page for Netpbm</a>.
 * This function works with single precision floating point numbers (i.e. float data type).
 */
int dwt_util_save_log_to_pgm_s(
	const char *path,	///< target file name, e.g. "output.pgm"
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of nested image (in elements)
	int size_y		///< height of nested image (in elements)
);

/**
 * @brief Save a symmetric (positive and negative valued) coefficients into an ASCII-type PGM file.
 *
 * See <a href="http://netpbm.sourceforge.net/">the home page for Netpbm</a>.
 * This function works with single precision floating point numbers (i.e. float data type).
 */
int dwt_util_save_sym_to_pgm_s(
	const char *path,	///< target file name, e.g. "output.pgm"
	float max_value, 	///< maximum value of pixel, e.g. 1.0f if image values lie inside an interval [-1.0; +1.0]
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of nested image (in elements)
	int size_y		///< height of nested image (in elements)
);

/**
 * @brief Save grayscale image into ASCII-type MAT file.
 *
 * @warning experimental
 */
int dwt_util_save_to_mat_s(
	const char *path,	///< target file name, e.g. "output.dat"
	const void *ptr,	///< pointer to beginning of image data
	int size_x,		///< width of nested image (in elements)
	int size_y,		///< height of nested image (in elements)
	int stride_x,		///< difference between rows (in bytes)
	int stride_y		///< difference between columns (in bytes)
);

/**
 * @brief Save matrix of vectors (float) and single-column matrix of labels (int) into LIBSVM format.
 *
 * See <a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">LIBSVM homepage</a> for more details.
 *
 * @warning experimental
 */
int dwt_util_save_to_svm_s(
	const char *path,	///< target file name, e.g. "output.dat"
	const void *ptr,	///< pointer to beginning of matrix of vectors (float type)
	int size_x,		///< width of nested matrix (in elements)
	int size_y,		///< height of nested matrix (in elements)
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	const void *cls_ptr,	///< pointer to beginning of matrix of labels (int type)
	int cls_size_x,		///< width of nested matrix (in elements), must be 1
	int cls_size_y,		///< height of nested matrix (in elements), must be same as @e size_y
	int cls_stride_x,	///< difference between rows (in bytes)
	int cls_stride_y	///< difference between columns (in bytes)
);

/**
 * @brief Save grayscale image into ASCII-type PGM file.
 *
 * See <a href="http://netpbm.sourceforge.net/">the home page for Netpbm</a>.
 * This function works with double precision floating point numbers (i.e. double data type).
 *
 * @note Use @ref dwt_util_conv_show_d function before this function call to save a transform.
 */
int dwt_util_save_to_pgm_d(
	const char *filename,	///< target file name, e.g. "output.pgm"
	double max_value, 	///< maximum value of pixel, e.g. 1.0 if image values lie inside an interval [0.0; 1.0]
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Load grayscale image from ASCII-type PGM file.
 *
 * See <a href="http://netpbm.sourceforge.net/">the home page for Netpbm</a>.
 * This function works with single precision floating point numbers (i.e. double data type).
 *
 * @return Returns zero value if success.
 */
int dwt_util_load_from_pgm_s(
	const char *filename,	///< input file name, e.g. "input.pgm"
	float max_value,	///< maximum desired value of pixel, e.g. 1.0f if image values lie inside an interval [0.0f; 1.0f]
	void **pptr,		///< place the pointer to beginning of image data at this address
	int *pstride_x,		///< place the difference between rows (in bytes) at this address
	int *pstride_y,		///< place the difference between columns (in bytes) at this address
	int *psize_big_x,	///< place the width of the image (in elements) at this address
	int *psize_big_y	///< place the height of the image (in elements) at this address
);

/**
 * @brief Load grayscale image from ASCII-type MAT file.
 *
 * @warning experimental
 */
int dwt_util_load_from_mat_s(
	const char *path,	///< input file name, e.g. "input.dat"
	void **ptr,		///< place the pointer to beginning of image data at this address
	int *size_x,		///< place the width of the image (in elements) at this address
	int *size_y,		///< place the height of the image (in elements) at this address
	int *stride_x,		///< place the difference between rows (in bytes) at this address
	int *stride_y		///< place the difference between columns (in bytes) at this address
);

/**
 * @brief Load grayscale image from ASCII-type PGM file.
 * 
 * See <a href="http://netpbm.sourceforge.net/">the home page for Netpbm</a>.
 * This function works with integers (i.e. int data type).
 * 
 * @return Returns zero value if success.
 */
int dwt_util_load_from_pgm_i(
	const char *filename,	///< input file name, e.g. "input.pgm"
	int max_value,		///< maximum desired value of pixel, e.g. 255 if image values lie inside an interval [0; 255]
	void **pptr,		///< place the pointer to beginning of image data at this address
	int *pstride_x,		///< place the difference between rows (in bytes) at this address
	int *pstride_y,		///< place the difference between columns (in bytes) at this address
	int *psize_big_x,	///< place the width of the image (in elements) at this address
	int *psize_big_y	///< place the height of the image (in elements) at this address
);

/**
 * @brief Load grayscale image from ASCII-type MAT file.
 *
 * @warning experimental
 */
int dwt_util_load_from_mat_i(
	const char *path,	///< input file name, e.g. "input.dat"
	void **ptr,		///< place the pointer to beginning of image data at this address
	int *size_x,		///< place the width of the image (in elements) at this address
	int *size_y,		///< place the height of the image (in elements) at this address
	int *stride_x,		///< place the difference between rows (in bytes) at this address
	int *stride_y		///< place the difference between columns (in bytes) at this address
);

/**
 * @brief Wrapper to @p omp_get_thread_num function.
 *
 * @warning experimental
 */
int dwt_util_get_thread_num();

/**
 * @brief Wrapper to @p omp_get_num_threads function.
 *
 * @warning experimental
 */
int dwt_util_get_num_threads();

/**
 * @brief Get number of active workers.
 */
int dwt_util_get_num_workers();

/**
 * @brief Identifier of PicoBlaze operation.
 *
 * @warning experimental
 */
enum dwt_op
{
	DWT_OP_NONE,		///< undefined operation
	DWT_OP_LIFT4SA,		///< CDF 9/7 wavelet, forward transform
	DWT_OP_LIFT4SB		///< CDF 9/7 wavelet, inverse transform
};

/**
 * @brief Set PicoBlaze operation.
 *
 * Function changes active PicoBlaze firmware. This makes sense only on UTIA
 * EdkDSP platform.
 *
 * @warning experimental
 */
void dwt_util_switch_op(
	enum dwt_op op		///< identifier of PicoBlaze operation
);

/**
 * @brief Check the correct function of ASVP (EdkDSP) platform.
 */
void dwt_util_test();

/**
 * @brief Terminate the program.
 */
void dwt_util_abort();

/**
 * @brief Actively waits the specified number of milliseconds.
 */
void dwt_util_wait(
	int ms		///< the number of milliseconds
);

/**
 * @brief Allocate vector of @e size floats.
 *
 * Allocate vector of given size that have to be even. Allocated memory has alignment on 64-bits boundary.
 *
 * @return Pointer to allocated memory that is 64-bits aligned.
 */
float *dwt_util_allocate_vec_s(
	int size	///< the number of elements (floats) to allocate, must be even
);

/**
 * @brief Fill vector of floats with simple sequence. Useful for testing.
 *
 * @return Return non-zero value if an error occurred.
 */
int dwt_util_generate_vec_s(
	float *addr,	///< pointer to the vector
	int size	///< number of vector elements
);

/**
 * @brief Fill vector with zeros.
 *
 * @return Return non-zero value if an error occurred.
 */
int dwt_util_zero_vec_s(
	float *addr,	///< pointer to the vector
	int size	///< number of vector elements
);

/**
 * @brief Copy vector of given size and check if values was transferred correctly.
 *
 * @return Return non-zero value if an error occurred.
 */
int dwt_util_copy_vec_s(
	const float *src,	///< source vector
	float *dst,		///< destination vector
	int size		///< number of vector elements
);

/**
 * @brief Compare two vectors.
 *
 * @return Return non-zero value in case of the two vectors are different.
 */
int dwt_util_cmp_vec_s(
	const float *a,		///< first vector
	const float *b,		///< second vector
	int size		///< number of vectors' elements
);

/**
 * @brief Print vector.
 *
 * @warning experimental
 */
void dwt_util_print_vec_s(
	const float *addr,
	int size
);

/**
 * @brief Return formatted vector as string.
 *
 * Do not call free on returned pointer.
 *
 * @warning experimental
 */
const char *dwt_util_str_vec_s(
	const float *vec,
	int size
);

/**
 * @brief Unit impulse.
 *
 * Zero the vector and put "1" into its center with the offset.
 *
 * @warning experimental
 */
void dwt_util_unit_vec_s(
	float *addr,
	int size,
	int offset
);

/**
 * @brief Replacement for @p vfprintf.
 */
int dwt_util_vfprintf(
	FILE *stream,		///< output stream
	const char *format,	///< format with same meaning like in @p printf function
	va_list ap		///< @p va_list encapsulating variable number of arguments
);

/**
 * @brief Replacement for @p vprintf.
 */
int dwt_util_vprintf(
	const char *format,	///< format with same meaning like in @p printf function
	va_list ap		///< @p va_list encapsulating variable number of arguments
);

/**
 * @brief Replacement for @p fprintf.
 */
int dwt_util_fprintf(
	FILE *stream,		///< output stream
	const char *format,	///< format with same meaning like in @p printf function
	...			///< variable number of arguments
);

/**
 * @brief Replacement for @p printf.
 */
int dwt_util_printf(
	const char *format,	///< format with same meaning like in @p printf function
	...			///< variable number of arguments
);

/**
 * @brief Log levels for @ref dwt_util_log function.
 */
enum dwt_util_loglevel {
	LOG_NONE = 0,	///< messages without prefix
	LOG_DBG,	///< debug messages
	LOG_INFO,	///< informational messages
	LOG_WARN,	///< warnings
	LOG_ERR,	///< errors
	LOG_TEST	///< tests
};

/**
 * @brief Formatted output. Same syntax like @p printf function.
 */
int dwt_util_log(
	enum dwt_util_loglevel level,	///< log level
	const char *format,		///< format string that specifies how subsequent arguments re converted for output
	...				///< the subsequent arguments
);

/**
 * @brief Print formatted output and abort the program.
 *
 * Print an error message and exit the program.
 * Format has same syntax like @p printf function.
 *
 * @warning experimental
 */
void dwt_util_error(
	const char *format,
	...
);

/**
 * @brief Check if memory is aligned to 128 bits.
 * 
 * @returns Returns 0 when not aligned or 1 when aligned.
 */
int dwt_util_is_aligned_16(
	const void *ptr		///< pointer to the memory
);

/**
 * @brief Check if memory is aligned to 64 bits.
 * 
 * @returns Returns 0 when not aligned or 1 when aligned.
 */
int dwt_util_is_aligned_8(
	const void *ptr		///< pointer to the memory
);

/**
 * @brief Check if memory is aligned to 32 bits.
 * 
 * @returns Returns 0 when not aligned or 1 when aligned.
 */
int dwt_util_is_aligned_4(
	const void *ptr		///< pointer to the memory
);

/**
 * @brief Get node (machine) name which we are running on.
 * 
 * @returns Returns pointer to string. Do not pass this pointer to @p free. This string can be changed by next @ref dwt_util_node function call.
 */
const char *dwt_util_node();

/**
 * @brief Get program name.
 * 
 * @returns Returns pointer to null-terminated string. Do not pass this pointer to @p free. This string can be changed by next @ref dwt_util_appname function call.
 */
const char *dwt_util_appname();

/**
 * Gets optimal data stride according to cache usage.
 * 
 * @return Returns optimal stride in bytes.
 */
int dwt_util_get_opt_stride(
	int min_stride		///< minimum required stride (in bytes)
);

/**
 * @brief Gets data stride according to selected method and cache usage.
 *
 * @warning experimental
 *
 * @return Returns stride in bytes.
 */
int dwt_util_get_stride(
	int min_stride,		///< minimum required stride (in bytes)
	int opt			///< use optimal stride which should have better performance (set non-zero value for this case here)
);

/**
 * @brief Determine if a number is probable prime.
 * 
 * Currently uses variant of Fermat primality test for base-2.
 * 
 * @return Non-zero value for probable prime, zero otherwise.
 */
int dwt_util_is_prime(
	int N	///< the number to test
);

/**
 * @brief Find smallest probable prime number not less than @e N.
 *
 * @return Return the probable prime found.
 */
int dwt_util_next_prime(
	int N	///< the number
);

/**
 * @brief Subbands.
 *
 * Discrete wavelet transform consists of four subbands at each level of decomposition.
 * The LL subband is filtered by low-pass filter in both directions.
 * This subband is futher decomposed.
 * The HL, LH and HH subbands are filtered by combination of high-pass and low-pass filters.
 * The HH subband contains mostly noise.
 */
enum dwt_subbands {
	DWT_LL,		///< subband filtered by LP filter horizontally and vertically
	DWT_HL,		///< subband filtered by HP horizontally and LP vertically
	DWT_LH,		///< subband filtered by HP vertically and LP horizontally
	DWT_HH		///< subband filtered by HP filter horizontally and vertically
};

/**
 * @brief Gets pointer to and sizes of the selected subband (LL, HL, LH or HH).
 */
void dwt_util_subband_i(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	enum dwt_subbands band,	///< subband of interest (LL, HL, LH, HH)
	void **dst_ptr,		///< here will be stored pointer to beginning of subband data
	int *dst_size_x,	///< here will be stored width of subband
	int *dst_size_y		///< here will be stored height of subband
);

/**
 * @brief Gets pointer to and sizes of the selected subband (LL, HL, LH or HH).
 */
void dwt_util_subband_s(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	enum dwt_subbands band,	///< subband of interest (LL, HL, LH, HH)
	void **dst_ptr,		///< here will be stored pointer to beginning of subband data
	int *dst_size_x,	///< here will be stored width of subband
	int *dst_size_y		///< here will be stored height of subband
);

void dwt_util_subband_const_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	enum dwt_subbands band,
	const void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y
);

/**
 * @brief Gets pointer to and sizes of the selected subband (LL, HL, LH or HH).
 */
void dwt_util_subband_d(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	enum dwt_subbands band,	///< subband of interest (LL, HL, LH, HH)
	void **dst_ptr,		///< here will be stored pointer to beginning of subband data
	int *dst_size_x,	///< here will be stored width of subband
	int *dst_size_y		///< here will be stored height of subband
);

/**
 * @brief Gets pointer to and sizes of the selected subband (LL, HL, LH or HH).
 */
void dwt_util_subband(
	void *ptr,		///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	enum dwt_subbands band,	///< subband of interest (LL, HL, LH, HH)
	void **dst_ptr,		///< here will be stored pointer to beginning of subband data
	int *dst_size_x,	///< here will be stored width of subband
	int *dst_size_y		///< here will be stored height of subband
);

/**
 * @brief Compute address of given transform coefficient or image pixel.
 * 
 * @warning This function is slow; faster way is calculate the address directly.
 */
int *dwt_util_addr_coeff_i(
	void *ptr,		///< pointer to beginning of image data
	int y,			///< y-coordinate
	int x,			///< x-coordinate
	int stride_x,		///< difference between rows (in bytes)
	int stride_y		///< difference between columns (in bytes)
);

/**
 * @brief Compute address of given transform coefficient or image pixel.
 * 
 * @warning This function is slow; faster way is calculate the address directly.
 */
const int *dwt_util_addr_coeff_const_i(
	const void *ptr,	///< pointer to beginning of image data
	int y,			///< y-coordinate
	int x,			///< x-coordinate
	int stride_x,		///< difference between rows (in bytes)
	int stride_y		///< difference between columns (in bytes)
);

/**
 * @brief Compute address of given transform coefficient or image pixel.
 * 
 * @warning This function is slow; faster way is calculate the address directly.
 */
float *dwt_util_addr_coeff_s(
	void *ptr,		///< pointer to beginning of image data
	int y,			///< y-coordinate
	int x,			///< x-coordinate
	int stride_x,		///< difference between rows (in bytes)
	int stride_y		///< difference between columns (in bytes)
);

/**
 * @brief Compute address of given transform coefficient or image pixel.
 * 
 * @warning This function is slow; faster way is calculate the address directly.
 */
const float *dwt_util_addr_coeff_const_s(
	const void *ptr,	///< pointer to beginning of image data
	int y,			///< y-coordinate
	int x,			///< x-coordinate
	int stride_x,		///< difference between rows (in bytes)
	int stride_y		///< difference between columns (in bytes)
);

/**
 * @brief Compute address of given transform coefficient or image pixel.
 * 
 * @warning This function is slow; faster way is calculate the address directly.
 */
double *dwt_util_addr_coeff_d(
	void *ptr,		///< pointer to beginning of image data
	int y,			///< y-coordinate
	int x,			///< x-coordinate
	int stride_x,		///< difference between rows (in bytes)
	int stride_y		///< difference between columns (in bytes)
);

/**
 * @brief Compute address of given transform coefficient or image pixel.
 * 
 * @warning This function is slow; faster way is calculate the address directly.
 */
void *dwt_util_addr_coeff(
	void *ptr,		///< pointer to beginning of image data
	int y,			///< x-coordinate
	int x,			///< y-coordinate
	int stride_x,		///< difference between rows (in bytes)
	int stride_y		///< difference between columns (in bytes)
);

/**
 * @brief Convert transform to viewable format.
 */
void dwt_util_conv_show_i(
	const void *src,	///< transform
	void *dst,		///< viewable image of transform
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Convert transform to viewable format.
 */
void dwt_util_conv_show_s(
	const void *src,	///< transform
	void *dst,		///< viewable image of transform
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Convert transform to viewable format.
 */
void dwt_util_conv_show_d(
	const void *src,	///< transform
	void *dst,		///< viewable image of transform
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y	///< height of nested image (in elements)
);

/**
 * @brief Performance test of 2-D DWT with CDF 9/7 wavelet.
 *
 * @warning experimental
 */
void dwt_util_perf_cdf97_2_s(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	int M,			///< one test loop consists of transform of M images
	int N,			///< number of test loops performed
	int clock_type,		///< timer type
	float *fwd_secs,	///< store resulting time for forward transform here
	float *inv_secs		///< store resulting time for inverse transform here
);

/**
 * @brief Performance test of 2-D DWT with CDF 9/7 wavelet.
 *
 * @warning experimental
 */
void dwt_util_perf_cdf97_2_inplace_s(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	int M,			///< one test loop consists of transform of M images
	int N,			///< number of test loops performed
	int clock_type,		///< timer type
	float *fwd_secs,	///< store resulting time for forward transform here
	float *inv_secs		///< store resulting time for inverse transform here
);

void dwt_util_perf_cdf97_2_inplace_sep_s(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	int M,			///< one test loop consists of transform of M images
	int N,			///< number of test loops performed
	int clock_type,		///< timer type
	float *fwd_secs,	///< store resulting time for forward transform here
	float *inv_secs		///< store resulting time for inverse transform here
);

void dwt_util_perf_cdf97_2_inplace_sep_sdl_s(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	int M,			///< one test loop consists of transform of M images
	int N,			///< number of test loops performed
	int clock_type,		///< timer type
	float *fwd_secs,	///< store resulting time for forward transform here
	float *inv_secs		///< store resulting time for inverse transform here
);

/**
 * @brief Performance test of 2-D DWT with CDF 9/7 wavelet.
 *
 * @warning experimental
 */
void dwt_util_perf_cdf97_2_inplace_sdl_s(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	int M,			///< one test loop consists of transform of M images
	int N,			///< number of test loops performed
	int clock_type,		///< timer type
	float *fwd_secs,	///< store resulting time for forward transform here
	float *inv_secs		///< store resulting time for inverse transform here
);

/**
 * @brief Performance test of 2-D DWT with CDF 9/7 wavelet.
 *
 * @warning experimental
 */
void dwt_util_perf_cdf97_2_d(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding,	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
	int M,			///< one test loop consists of transform of M images
	int N,			///< number of test loops performed
	int clock_type,		///< timer type
	double *fwd_secs,	///< store resulting time for forward transform here
	double *inv_secs	///< store resulting time for inverse transform here
);

/**
 * @brief Type of storage of the original missized image for fast transform purpose.
 */
enum dwt_array
{
	DWT_ARR_SIMPLE,		///< enlarge the image to the smallest power of two value not less than original image size
	DWT_ARR_SPARSE,		///< place the original image into bigger (outer) image of size of power of two value
	DWT_ARR_PACKED		///< do not enlarge anything
};

/**
 * @brief Get sizes (width, height, strides) of appropriately resized image.
 *
 * @note This function considers 1-channel images.
 */
void dwt_util_get_sizes_i(
	enum dwt_array array_type,	///< how to extend the original image
	int size_x,			///< width of original image
	int size_y,			///< height of original image
	int opt_stride,			///< use optimal stride
	int *stride_x,			///< store difference between rows here (in bytes)
	int *stride_y,			///< store difference between columns here (in bytes)
	int *size_o_big_x,		///< store width of outer (frame) image frame here (in elements)
	int *size_o_big_y,		///< store height of outer (frame) image frame here (in elements)
	int *size_i_big_x,		///< store width of nested (inner) image here (in elements)
	int *size_i_big_y		///< store height of nested (inner) image here (in elements)
);

/**
 * @brief Get sizes (width, height, strides) of appropriately resized image.
 *
 * @note This function considers 1-channel images.
 */
void dwt_util_get_sizes_s(
	enum dwt_array array_type,	///< how to extend the original image
	int size_x,			///< width of original image
	int size_y,			///< height of original image
	int opt_stride,			///< use optimal stride
	int *stride_x,			///< store difference between rows here (in bytes)
	int *stride_y,			///< store difference between columns here (in bytes)
	int *size_o_big_x,		///< store width of outer (frame) image frame here (in elements)
	int *size_o_big_y,		///< store height of outer (frame) image frame here (in elements)
	int *size_i_big_x,		///< store width of nested (inner) image here (in elements)
	int *size_i_big_y		///< store height of nested (inner) image here (in elements)
);

/**
 * @brief Get sizes (width, height, strides) of appropriately resized image.
 *
 * @note This function considers 1-channel images.
 */
void dwt_util_get_sizes_d(
	enum dwt_array array_type,	///< how to extend the original image
	int size_x,			///< width of original image
	int size_y,			///< height of original image
	int opt_stride,			///< use optimal stride
	int *stride_x,			///< store difference between rows here (in bytes)
	int *stride_y,			///< store difference between columns here (in bytes)
	int *size_o_big_x,		///< store width of outer (frame) image frame here (in elements)
	int *size_o_big_y,		///< store height of outer (frame) image frame here (in elements)
	int *size_i_big_x,		///< store width of nested (inner) image here (in elements)
	int *size_i_big_y		///< store height of nested (inner) image here (in elements)
);

/**
 * @brief Measure performance of 1-D transform.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_1_s(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< should be 1
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

/**
 * @brief Measure performance of 1-D transform.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_1_d(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< should be 1
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

/**
 * @brief Measure performance of 2-D transform.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_2_s(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< decompose up to single coefficient?
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

/**
 * @brief Measure performance of 2-D transform.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_2_inplace_s(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< decompose up to single coefficient?
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

/**
 * @brief Measure performance of 2-D transform.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_2_inplace_sep_s(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< decompose up to single coefficient?
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

void dwt_util_measure_perf_cdf97_2_inplace_sep_sdl_s(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< decompose up to single coefficient?
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

/**
 * @brief Measure performance of 2-D transform.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_2_inplace_sdl_s(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< decompose up to single coefficient?
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

/**
 * @brief Measure performance of 2-D transform.
 *
 * @warning experimental
 */
void dwt_util_measure_perf_cdf97_2_d(
	enum dwt_array array_type,	///< how to extend the original image
	int min_x,			///< starting vector size
	int max_x,			///< maximal vector size
	int opt_stride,			///< use optimal stride
	int j_max,			///< level of decomposition
	int decompose_one,		///< decompose up to single coefficient?
	int zero_padding,		///< fill residual transform areas with zeros
	int M,				///< number of transform in one test loop
	int N,				///< test loops to perform
	int clock_type,			///< timer type
	FILE *fwd_plot_data,		///< store resulting plot data for forward transform here (gnuplot compatible format)
	FILE *inv_plot_data		///< store resulting plot data for inverse transform here (gnuplot compatible format)
);

/**
 * @brief Checks if the number is a power of two.
 * @return Non-zero value if the number is a power of two.
 */
int dwt_util_is_pow2(
	int x				///< the number to check, should be greater than zero
);

/**
 * @brief Calculates a size (in bytes) that the image occupies in a memory.
 *
 * @return Returns image size in bytes.
 */
size_t dwt_util_image_size(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y	///< height of outer image frame (in elements)
);

/**
 * @brief Prints some technical information. For debugging purposes.
 *
 * @warning experimental
 */
void dwt_util_print_info();

/**
 * @brief Get the number of processors. Not all CPUs need to be online.
 *
 * @warning experimental
 */
long dwt_util_get_ncpus();

/**
 * @brief Rectified wavelet power spectrum for a specific subband.
 *
 * @returns The value of the wavelet power spectrum.
 *
 * @warning experimental
 */
float dwt_util_band_wps_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y,		///< height of outer image frame (in elements)
	int j			///< decomposition level of given subband
);

/**
 * @brief Index of coefficient with maximal magnitude in a specific subband.
 *
 * @returns The integer index as float.
 *
 * @warning experimental
 */
float dwt_util_band_maxidx_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief The arithmetic mean for a specific subband.
 *
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns The mean.
 *
 * @warning experimental
 */
float dwt_util_band_mean_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Median of a specific subband.
 *
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns The median.
 *
 * @warning experimental
 */
float dwt_util_band_med_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Moment.
 *
 * The @e n -th moment about the @e c value.
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns Moment.
 *
 * @warning experimental
 */
float dwt_util_band_moment_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y,		///< height of outer image frame (in elements)
	int n,			///< the n-th moment
	float c			///< moment about this value
);

/**
 * @brief Central moment.
 *
 * The @e n -th moment about the mean.
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns Moment.
 *
 * @warning experimental
 */
float dwt_util_band_cmoment_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y,		///< height of outer image frame (in elements)
	int n			///< the n-th moment
);

/**
 * @brief Variance.
 *
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns The variance.
 *
 * @warning experimental
 */
float dwt_util_band_var_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Standard deviation.
 *
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns The standard deviation.
 *
 * @warning experimental
 */
float dwt_util_band_stdev_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Standardized moment.
 *
 * The normalized/standardized n-th central moment.
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns Moment.
 *
 * @warning experimental
 */
float dwt_util_band_smoment_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y,		///< height of outer image frame (in elements)
	int n			///< the n-th moment
);

/**
 * @brief Skewness.
 *
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns The skewness.
 *
 * @warning experimental
 */
float dwt_util_band_skew_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Kurtosis.
 *
 * This can be computed on magnitudes, see @ref dwt_util_abs_s function.
 *
 * @returns The kurtosis.
 *
 * @warning experimental
 */
float dwt_util_band_kurt_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Maximum norm.
 *
 * @returns The norm of the subband.
 *
 * @warning experimental
 */
float dwt_util_band_maxnorm_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief The @f$ \ell^p @f$ norm.
 *
 * @returns The norm of the subband.
 *
 * @warning experimental
 */
float dwt_util_band_lpnorm_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y,		///< height of outer image frame (in elements)
	float p			///< the parameter of the norm (p=1 for the taxicab norm, p=2 for the Euclidean norm)
);

/**
 * @brief The Euclidean norm.
 *
 * @returns The norm of the subband.
 *
 * @warning experimental
 */
float dwt_util_band_norm_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of outer image frame (in elements)
	int size_y		///< height of outer image frame (in elements)
);

/**
 * @brief Count the subbands up to given decomposition level.
 *
 * @returns The number of subbands.
 *
 * @warning experimental
 */
int dwt_util_count_subbands_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max		///< the decomposition level of interest
);

/**
 * @brief Calculate the wavelet power spectra for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_wps_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate the indices of coefficients with maximal magnitudes for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_maxidx_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate the arithmetic mean for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_mean_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate medians for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_med_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate variances for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_var_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate standard deviations for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_stdev_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate skewnesses for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_skew_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate kurtosises for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_kurt_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate the maximum norm for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_maxnorm_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Calculate the p-norm for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_lpnorm_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv,		///< store feature vector here
	float p			///< the parameter of the norm (p=1 for the taxicab norm, p=2 for the Euclidean norm)
);

/**
 * @brief Calculate the Euclidean norm for all the subbands.
 *
 * @warning experimental
 */
void dwt_util_norm_s(
	const void *ptr,	///< pointer to beginning of image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the decomposition level of interest
	float *fv		///< store feature vector here
);

/**
 * @brief Gets necessary data alignment for the current platform.
 *
 * The memory alignment depends on the data type (size of this data type).
 * For instance, some SSE instructions (MOVAPS) on x86_64 platform require correctly aligned data.
 *
 * @return Minimum required alignment (in bytes).
 *
 * @warning experimental
 */
size_t dwt_util_alignment(
	size_t type_size	///< sizeof requested data type, e.g. sizeof(float)
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7.
 *
 * @warning experimental
 */
int dwt_util_test_cdf97_2_s(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7 (out-of-place transform).
 *
 * @warning experimental
 */
int dwt_util_test_cdf97_2_s2(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7.
 *
 * @warning experimental
 */
int dwt_util_test_cdf97_2_d(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7.
 *
 * @warning experimental
 */
int dwt_util_test_cdf97_2_i(
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_o_big_x,	///< width of outer image frame (in elements)
	int size_o_big_y,	///< height of outer image frame (in elements)
	int size_i_big_x,	///< width of nested image (in elements)
	int size_i_big_y,	///< height of nested image (in elements)
	int j_max,		///< the number of intended decomposition levels (scales)
	int decompose_one,	///< should be row or column of size one pixel decomposed? zero value if not
	int zero_padding	///< fill padding in channels with zeros? zero value if not, should be non zero only for sparse decomposition
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7.
 *
 * @warning experimental
 */
int dwt_util_test2_cdf97_2_s(
	enum dwt_array array_type,	///< how to extend the original image
	int size_x,			///< width of original image
	int size_y,			///< height of original image
	int opt_stride,			///< use optimal stride
	int j_max,			///< the number of intended decomposition levels (scales)
	int decompose_one		///< should be row or column of size one pixel decomposed? zero value if not
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7 (out-of-place transform).
 *
 * @warning experimental
 */
int dwt_util_test2_cdf97_2_s2(
	enum dwt_array array_type,	///< how to extend the original image
	int size_x,			///< width of original image
	int size_y,			///< height of original image
	int opt_stride,			///< use optimal stride
	int j_max,			///< the number of intended decomposition levels (scales)
	int decompose_one		///< should be row or column of size one pixel decomposed? zero value if not
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7.
 *
 * @warning experimental
 */
int dwt_util_test2_cdf97_2_d(
	enum dwt_array array_type,	///< how to extend the original image
	int size_x,			///< width of original image
	int size_y,			///< height of original image
	int opt_stride,			///< use optimal stride
	int j_max,			///< the number of intended decomposition levels (scales)
	int decompose_one		///< should be row or column of size one pixel decomposed? zero value if not
);

/**
 * @brief Test correct function of 2D DWT with CDF 9/7.
 *
 * @warning experimental
 */
int dwt_util_test2_cdf97_2_i(
	enum dwt_array array_type,	///< how to extend the original image
	int size_x,			///< width of original image
	int size_y,			///< height of original image
	int opt_stride,			///< use optimal stride
	int j_max,			///< the number of intended decomposition levels (scales)
	int decompose_one		///< should be row or column of size one pixel decomposed? zero value if not
);

/**
 * @brief Absolute values of image.
 *
 * Calculate absolute values (magnitudes) of given subband or whole image/transform, in-place version.
 *
 * @warning experimental
 */
void dwt_util_abs_s(
	void *ptr,		///< pointer to image data
	int stride_x,		///< difference between rows (in bytes)
	int stride_y,		///< difference between columns (in bytes)
	int size_x,		///< width of the image
	int size_y		///< height of the image
);

/**
 * @brief Dot product.
 *
 * The dot/inner product of two signals with single precission floating point format.
 * The second signal is displaced with respect to the first one.
 *
 * @param ptr1 The reference signal.
 * @param ptr2 The displaced signal.
 * @param displ_x The x-coordinate of the displacement.
 * @param displ_y The y-coordinate of the displacement.
 */
float dwt_util_dot_s(
	const void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
);

/**
 * @brief Normalize signal.
 *
 * The samples of the input signal are divided by the signal norm.
 *
 * @param p The p-norm used for the normalization.
 */
void dwt_util_normalize_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float p
);

/**
 * @brief Addition of two signals.
 *
 * @param[in,out] ptr1 The destination signal.
 * @param[in]     ptr2 The source signal.
 */
void dwt_util_add_s(
	void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
);

/**
 * @brief Multiplication of two signals.
 *
 * @param[in,out] ptr1 The destination signal.
 * @param[in]     ptr2 The source signal.
 */
void dwt_util_mul_s(
	void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
);

/**
 * @brief Find minimum and maximum of matrix.
 *
 * @warning experimental
 */
int dwt_util_find_min_max_s(
	const void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float *min,
	float *max
);

int dwt_util_find_min_max_i(
	const void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *min,
	int *max
);

/**
 * @brief Add a constant to all elements of matrix.
 *
 * @warning experimental
 */
int dwt_util_shift_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float a
);

/**
 * @brief Shift each vector (row) in matrix by itw median.
 *
 * @warning experimental
 */
void dwt_util_shift21_med_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y
);

/**
 * @brief Multiply all elements of matrix by a constant.
 *
 * @warning experimental
 */
int dwt_util_scale_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float a
);

/**
 * @brief Scale 1-D vectors (rows) in matrix to interval @e lo .. @e hi (independently).
 *
 * @warning experimental
 */
int dwt_util_scale21_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float lo,
	float hi
);

/**
 * @brief Displace given vector (1-D) in-place.
 *
 * @warning experimental
 */
int dwt_util_displace1_s(
	void *ptr,
	int size_x,
	int stride_y,
	int displ_x
);

/**
 * @brief Displace given vector (1-D) in-place, zero padding.
 *
 * @warning experimental
 */
int dwt_util_displace1_zero_s(
	void *ptr,
	int size_x,
	int stride_y,
	int displ_x
);

/**
 * @brief Get a position of the center of given vector (1-D).
 *
 * @warning experimental
 */
int dwt_util_get_center1_s(
	const void *ptr,
	int size_x,
	int stride_y
);

/**
 * @brief Center given vector (1-D).
 *
 * @warning experimental
 */
int dwt_util_center1_s(
	void *ptr,
	int size_x,
	int stride_y,
	int max_iters
);

/**
 * @brief Center vectors (1-D) in matrix (vectors as rows).
 *
 * @warning experimental
 */
int dwt_util_center21_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int max_iters
);

/**
 * @brief Get viewport to matrix/image.
 *
 * Example usage:
 * @code
 * // the original image
 * int size_x = 640, size_y = 480, stride_y = sizeof(float), stride_x = stride_y * size_x;
 * void *ptr;
 * dwt_util_alloc_image(&ptr, stride_x, stride_y, size_x, size_y);
 *
 * // the viewport
 * int view_x = 320, view_y = 240, offset_x = 160, offset_y = 120;
 * void *view = dwt_util_viewport(ptr, size_x, size_y, stride_x, stride_y, offset_x, offset_y);
 * @endcode
 *
 * @return pointer to the viewport
 *
 * @warning experimental
 */
void *dwt_util_viewport(
	void *ptr,		///< original matrix pointer
	int size_x,		///< original matrix width
	int size_y,		///< original matrix height
	int stride_x,		///< original matrix stride
	int stride_y,		///< original matrix stride
	int offset_x,		///< requested offset in x
	int offset_y		///< requested offset in y
);

/**
 * @brief Crop @p len_x samples around the center of each vector.
 *
 * @warning experimental
 */
void *dwt_util_crop21(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int len_x
);

/**
 * @brief Size of array in elements.
 */
#define sizeof_arr(a) (sizeof(a)/sizeof(*(a)))

/**
 * @brief Allocate memory.
 *
 * @warning experimental
 */
void *dwt_util_alloc(
	int elems,
	size_t elem_size
);

void dwt_util_flush_cache(
	void *addr,	///< base address
	size_t size	///< length of memory in bytes
);

/**
 * @brief Round up (ceil) to an even integer.
 */
int dwt_util_up_to_even(
	int x
);

/**
 * @brief Round up to a multiply of 4.
 */
int dwt_util_up_to_mul4(
	int x
);

/**
 * @brief Round down (floor) to an even integer.
 */
int dwt_util_to_even(
	int x
);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
