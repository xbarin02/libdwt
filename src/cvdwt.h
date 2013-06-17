/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief OpenCV interface to libdwt.
 */
#ifndef CVDWT_H
#define CVDWT_H

#include <cv.h> // cv::
#include <string> // std::string

/**
 * @defgroup cpp_cvdwt C++ OpenCV interface
 * @{
 **/
#ifdef __cplusplus

/**
 * @namespace dwt
 * @brief Binding to OpenCV
 */
namespace dwt
{

/**
 * @brief Flags for @ref dwt::transform function.
 */
enum
{
	/**
	 * Perform forward transform.
	 */
	DWT_FORWARD = 1,

	/**
	 * Perform inverse transform.
	 */
	DWT_INVERSE = 2,

	/**
	 * Decompose one row or column.
	 */
	DWT_EXTREME = 4,

	/**
	 * Perform zero padding.
	 */
	DWT_PADDING = 8,

	/**
 	 * Simple image DWT.
	 * Size of image (outer frame) become a power of two. Size of nested
	 * image should be smaller or equal to outer frame. Outer frame is
	 * padded with zeros. Coefficients with large aplitude appear on inner
	 * image edges. Slowest variant.
	 */
	DWT_SIMPLE = 16,	

	/**
	 * Sparse image DWT.
	 * Size of image (outer frame) become a power of two. Size of nested
	 * image should be smaller or equal to outer frame. Coefficients with
	 * large aplitude do not appear on inner image edges. Undefined values
	 * remain in unused image area when you do not set @ref DWT_PADDING.
	 * Otherwise, unused image area is filled with zeros (slower than
	 * variant without zero padding).
	 */
	DWT_SPARSE = 32,

	/**
	 * Packed image DWT.
	 * Outer frame size is equal to inner image size. Inner image size can
	 * be of any size. Coefficients with large aplitude do not appear on
	 * inner image edges. Fastest variant.
	 */
	DWT_PACKED = 64,

	/**
	 * Use CDF 5/3 (2,2) wavelet.
	 */
	DWT_CDF53 = 128,

	/**
	 * Use CDF 9/7 (4,4) wavelet.
	 */
	DWT_CDF97 = 256
};

/**
 * @brief Perform forward or inverse discrete wavelet transform of given image.
 * 
 * @note This function operates IN-PLACE.
 */
void transform(
	cv::Mat &img,				///< image/transform (outer size)
	cv::Size size,				///< size of inner image (inside of input image)
	int &j,					///< target scale (get and set when performing forward transform, get when performing inverse transform)
	int flags = DWT_FORWARD|DWT_SIMPLE	///< flags
);

/**
 * @brief Perform forward or inverse discrete wavelet transform of given image.
 * 
 * @note This function can operate OUT-OF-PLACE (src != dst) or IN-PLACE (src == dst).
  */
void transform(
	const cv::Mat &src,			///< input image/transform (outer size)
	cv::Mat &dst,				///< place output image/transform here
	cv::Size size,				///< size of inner image (inside of input image)
	int &j,					///< target scale (get and set when performing forward transform, get when performing inverse transform)
	int flags = DWT_FORWARD|DWT_SIMPLE	///< type of transform
);

/**
 * @brief Resize image to power of two sizes.
 *
 * @note This function operates IN-PLACE.
 */
void resizePOT(
	cv::Mat &img,				///< image that can be resized if needed
	int borderType = cv::BORDER_CONSTANT	///< the border type
);

/**
 * @brief Resize image to power of two sizes.
 *
 * @note This function operates OUT-OF-PLACE.
 */
void resizePOT(
	const cv::Mat &src,			///< input image
	cv::Mat &dst,				///< place resized image here
	int borderType = cv::BORDER_CONSTANT	///< the border type
);

/**
 * @brief Check if image size equals to power of two size.
 */
int isPOT(
	const cv::Mat &img			///< tested image
);

/**
 * @brief Displays the discrete wavelet transform in the specified window.
 *
 * This function is similar to cv::imshow from OpenCV.
 */
void wtshow(
	const std::string &winname,		///< name of the window
	const cv::Mat &image			///< transform to be shown
);

/**
 * @brief Create test image.
 */
void createTestImage(
	cv::Mat &img,				///< place test image here
	const cv::Size &size,			///< matrix size specification
	int type				///< matrix element type, e.g. CV_64FC3
);

/**
 * @brief List of subbands for @ref dwt:subband function.
 */
enum {
	/**
	 * Subband filtered by LP filter horizontally and vertically.
	 */
	DWT_LL = 1,

	/**
	 * Subband filtered by HP horizontally and LP vertically.
	 */
	DWT_HL,

	/**
	 * Subband filtered by HP vertically and LP horizontally.
	 */
	DWT_LH,

	/**
	 * Subband filtered by HP filter horizontally and vertically.
	 */
	DWT_HH
};

/**
 * @brief Get access to a subband of the transform.
 */
cv::Mat subband(
	const cv::Mat &src,		///< a transform at least with @e j levels of decomposition
	const cv::Size size,		///< size of inner image (inside of input image) which was transformed
	int j = 0,			///< level of decomposition for which you want to obtain a subband
	int band = dwt::DWT_LL		///< which subband (LL, HL, LH or HH)
);

}

#endif
/**
 * @}
 */

#endif
