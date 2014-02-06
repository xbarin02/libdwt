#ifndef EXR_H
#define EXR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load one channel of OpenEXR image file.
 *
 * This function works with single precision floating point numbers (i.e. double data type).
 *
 * @return Returns zero value if success.
 */
int dwt_util_load_from_exr_s(
	const char *filename,	///< input file name, e.g. "input.exr"
	const char *channel,	///< "R", "G", or "B"
	void **pptr,		///< place the pointer to beginning of image data at this address
	int *pstride_x,		///< place the difference between rows (in bytes) at this address
	int *pstride_y,		///< place the difference between columns (in bytes) at this address
	int *psize_x,		///< place the width of the image (in elements) at this address
	int *psize_y		///< place the height of the image (in elements) at this address
);

#ifdef __cplusplus
}
#endif

#endif
