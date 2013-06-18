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
	int j			///< decomposition levels
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
	int size_y,		///< height of outer image frame (in elements)
	int j			///< decomposition levels
);

/**
 * @brief The arithmetic mean for a specific subband.
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
	int size_y,		///< height of outer image frame (in elements)
	int j			///< decomposition levels
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
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
