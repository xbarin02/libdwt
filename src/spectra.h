#ifndef SPECTRA_H
#define SPECTRA_H

void *spectra_load(
	const char *path,
	int *stride_x,
	int *stride_y,
	int *size_x,
	int *size_y
);

void spectra_unload(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
);

/**
 * @brief Compute address of a matrix row.
 */
float *dwt_util_addr_row_s(
	void *ptr,		///< pointer to beginning of matrix data
	int y,			///< y-coordinate
	int stride_x		///< difference between rows (in bytes)
);

/**
 * @brief Compute address of a matrix row.
 */
int *dwt_util_addr_row_i(
	void *ptr,
	int y,
	int stride_x
);

/**
 * @brief Save grayscale image into ASCII-type MAT file.
 */
int dwt_util_save1_to_mat_s(
	const char *path,	///< target file name, e.g. "output.dat"
	const void *ptr,	///< pointer to beginning of image data
	int size_x,		///< width of nested image (in elements)
	int stride_y		///< difference between columns (in bytes)
);

#endif
