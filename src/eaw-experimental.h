#ifndef EAW_EXPERIMENTAL_H
#define EAW_EXPERIMENTAL_H

/**
 * @brief Inverse image fast wavelet transform using WCDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * See <a href="http://www.cs.huji.ac.il/~raananf/projects/eaw/">Edge-Avoiding Wavelets page</a>.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_eaw97_2i_s(
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
	float *wH[],		///< same as for @ref dwt_eaw97_2f_s
	float *wV[]		///< same as for @ref dwt_eaw97_2f_s
);

/**
 * @brief Forward image fast wavelet transform using WCDF 9/7 wavelet and lifting scheme, in-place version.
 *
 * See <a href="http://www.cs.huji.ac.il/~raananf/projects/eaw/">Edge-Avoiding Wavelets page</a>.
 *
 * This function works with single precision floating point numbers (i.e. float data type).
 */
void dwt_eaw97_2f_s(
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

#endif
