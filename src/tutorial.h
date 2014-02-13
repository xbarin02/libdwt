/**
 * @page tutorial Brief tutorial to libdwt usage
 *
 * @section compiling Compiling
 *
 * First, download and unpack library source codes. Then, compile the libdwt static library like follows.
 *
 * @code{.sh}
 * make ARCH=x86_64 BUILD=release -C src libdwt.a
 * @endcode
 *
 * Currently, following architectures are supported.
 *
 * architecture     | description
 * ---------------- | -----------
 * @c x86_64        | AMD64, EM64T
 * @c microblaze    | MicroBlaze core in Xilinx FPGA
 * @c asvp          | UTIA's ASVP platform
 * @c armv6l        | ARM11 family, e.g. Raspberry Pi
 * @c armv7l        | Cortex-A8 family, e.g. Nokia N900
 *
 * Now, you can use the library in your program (include @c libdwt.h and link against @c libdwt.a).
 * Finally, do not forget to link your application also with @c -lm, @c -lrt, and enable OpenMP with @c -fopenmp.
 * Further, look at @c example directory for understanding the use of this library.
 *
 * @section simple-program Simple program
 *
 * Include @c libdwt.h header file and call @ref dwt_util_init and @ref dwt_util_finish functions that allocate and release resources of the library.
 *
 * @code{.c}
 * #include "libdwt.h"
 * 
 * int main()
 * {
 *     dwt_util_init();
 * 
 *     // your code here
 * 
 *     dwt_util_finish();
 * 
 *     return 0;
 * }
 * @endcode
 * 
 * @section image-allocation Image allocation
 *
 * Use @ref dwt_util_alloc_image function to allocate a memory for your images.
 * The transform is computed in-place.
 * Thus, you do not need any extra memory allocated for handling of transform.
 * Here, we will use @c float data type.
 * Every function handling this data type has @c _s suffix.
 * Furthermore, we consider a single-channel image.
 * So, the distance between subsequent columns equals to @c sizeof(float).
 * Conversely distance between subsequent lines should be computed using @ref dwt_util_get_opt_stride function.
 * When you finish with your work, the allocated memory should be freed with @ref dwt_util_free_image function call.
 *
 * @code{.c}
 * // image sizes
 * const int x = 1920, y = 1080;
 *
 * // image strides
 * const int stride_y = sizeof(float);
 * const int stride_x = dwt_util_get_opt_stride(x * sizeof(float));
 *
 * // image data
 * void *data;
 *
 * dwt_util_alloc_image(&data, stride_x, stride_y, x, y);
 * @endcode
 *
 * @section performing-transform Performing transform
 *
 * In this section, we focus on CDF 9/7 wavelet used in the <a href="http://www.jpeg.org/jpeg2000/">JPEG 2000</a> image compression standard.
 * Moreover, we want one level of decomposition (LL1, HL1, LH1 and HH1 sub-bands).
 * The 2-D forward transform can be performed by @ref dwt_cdf97_2f_s function.
 * Analogously, the inverse transform can be performed using @ref dwt_cdf97_2i_s with same parameters. Both functions work in-place.
 *
 * @code{.c}
 * // level of decomposition
 * int j = 1;
 *
 * // forward transform
 * dwt_cdf97_2f_s(data, stride_x, stride_y, x, y, x, y, &j, 0, 0);
 *
 * // process the transform here
 *
 * // inverse transform
 * dwt_cdf97_2i_s(data, stride_x, stride_y, x, y, x, y, j, 0, 0);
 * @endcode
 *
 * @section subband-access Sub-band access
 *
 * On each level of decomposition (on each scale), we obtain four sub-bands which correspond to the image approximation and its horizontal, vertical and diagonal edges.
 * Sub-band with image approximation (LL) is preserved only on the highest level of image decomposition (the coarsest scale).
 * Thus, for each scale we can obtain three sub-bands (HL, LH and HH).
 * Finally, at the highest decomposition level we get all four sub-bands (LL, HL, LH and HH).
 * To access to the coefficients in a particular sub-band at a certain scale, you can use the @ref dwt_util_subband_s function.
 * Through the parameters of this function, we pass a level of decomposition and a sub-band identifier (either @ref DWT_LL, @ref DWT_HL, @ref DWT_LH or @ref DWT_HH).
 * The function will return a sub-band dimensions and a pointer to its data.
 *
 * @code{.c}
 * void *subband_ptr;
 * int subband_size_x;
 * int subband_size_y;
 *
 * dwt_util_subband_s(data, stride_x, stride_y, x, y, x, y, j, DWT_LH, &subband_ptr, &subband_size_x, &subband_size_y);
 * @endcode
 *
 * @section pixel-access Pixel access
 *
 * The address of any coefficient (or pixel) can be computed using @ref dwt_util_addr_coeff_s function.
 * Feel free to change a value at the obtained address.
 *
 * @code{.c}
 * float *coeff = dwt_util_addr_coeff_s(data, y, x, stride_x, stride_y);
 * @endcode
 *
 * @section data-types Data types
 *
 * The libdwt library currently supports single precision floating point type (@c single), double precision floating point type (@c double) and integer type (@c int).
 * The corresponding functions have suffixes @c _s (single precision FP), @c _d (double precision FP) and @c _i (integer).
 * The situation is illustrated in the following table.
 *
 * transform | float | double | int
 * --------- | ----- | ------ | ---
 * WCDF 5/3  | yes   | no     | no
 * WCDF 9/7  | yes   | no     | no
 * CDF 5/3   | yes   | yes    | yes
 * CDF 9/7   | yes<sup>*</sup>| yes    | yes
 *
 * Notes:
 * <br>
 * <sup>*</sup> accelerated with SSE (PC platform) or BCE (ASVP platform)
 *
 * @section feature-description Feature description
 *
 * Once the image is transformed into discrete wavelet transform, the feature vector can be extracted using one of the following aggregate functions.
 * Vector elements correspond to the subbands.
 *
 * * @ref dwt_util_wps_s – the rectified wavelet power spectra
 * * @ref dwt_util_maxidx_s – the indices of coefficients with maximal magnitudes
 * * @ref dwt_util_mean_s – the arithmetic means
 * * @ref dwt_util_med_s – the medians
 * * @ref dwt_util_var_s – the variances
 * * @ref dwt_util_stdev_s – the standard deviations
 * * @ref dwt_util_skew_s – the skewnesses
 * * @ref dwt_util_kurt_s – the kurtosises
 * * @ref dwt_util_maxnorm_s – the maximum norms
 * * @ref dwt_util_lpnorm_s – the p-norms
 * * @ref dwt_util_norm_s – the Euclidean norms
 */
