/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief OpenCV interface to libdwt.
 */
#include "cvdwt.h"

#include "libdwt.h"

#include <highgui.h> // imshow

using namespace cv;
using namespace dwt;

void dwt::resizePOT(
	Mat &img,
	int borderType)
{
	const int rows = dwt_util_pow2_ceil_log2(img.rows);
	const int cols = dwt_util_pow2_ceil_log2(img.cols);

	copyMakeBorder(img.clone(), img, 0, rows-img.rows, 0, cols-img.cols, borderType);
}

void dwt::resizePOT(
	const Mat &src,
	Mat &dst,
	int borderType)
{
	const int rows = dwt_util_pow2_ceil_log2(src.rows);
	const int cols = dwt_util_pow2_ceil_log2(src.cols);

	copyMakeBorder(src, dst, 0, rows-src.rows, 0, cols-src.cols, borderType);
}

int dwt::isPOT(
	const Mat &img)
{
	const int rows = dwt_util_pow2_ceil_log2(img.rows);
	const int cols = dwt_util_pow2_ceil_log2(img.cols);

	return (img.rows == rows) && (img.cols == cols);
}

// FIXME: dwt_util_conv_show_*
void dwt::wtshow(
	const string &winname,
	const Mat &image)
{
	CV_Assert( image.depth() == CV_64F || image.depth() == CV_32F || image.depth() == CV_32S );

	switch(image.depth())
	{
		case CV_64F:
		case CV_32F:
		{
			double a = 100;
			double b = 10;

			Mat g;

			log(1+abs(image)*a, g);

			imshow(winname, g/b);
		}
		break;
		case CV_32S:
		{
			Mat g;

			g = abs(image);

			imshow(winname, g);
		}
		break;
	}
}

static
int is_set(
	int word,
	int flag)
{
	return word&flag ? 1 : 0;
}

static
void cv_dwt_cdf97_2f(
	Mat &img,
	const int &channel,
	const Size &size,
	int &j,
	const int &flags)
{
	switch(img.depth())
	{
		case CV_64F:
			dwt_cdf97_2f_d(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				&j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32F:
			dwt_cdf97_2f_s(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				&j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32S:
			dwt_cdf97_2f_i(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				&j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		default:
			CV_Error(CV_StsOutOfRange, "The matrix element depth value is out of range.");
	}
}

static
void cv_dwt_cdf53_2f(
	Mat &img,
	const int &channel,
	const Size &size,
	int &j,
	const int &flags)
{
	switch(img.depth())
	{
		case CV_64F:
			dwt_cdf53_2f_d(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				&j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32F:
			dwt_cdf53_2f_s(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				&j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32S:
			dwt_cdf53_2f_i(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				&j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		default:
			CV_Error(CV_StsOutOfRange, "The matrix element depth value is out of range.");
	}
}

static
void cv_dwt_cdf97_2i(
	Mat &img,
	const int &channel,
	const Size &size,
	const int &j,
	const int &flags)
{
	switch(img.depth())
	{
		case CV_64F:
			dwt_cdf97_2i_d(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32F:
			dwt_cdf97_2i_s(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32S:
			dwt_cdf97_2i_i(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		default:
			CV_Error(CV_StsOutOfRange, "The matrix element depth value is out of range.");
	}
}

static
void cv_dwt_cdf53_2i(
	Mat &img,
	const int &channel,
	const Size &size,
	const int &j,
	const int &flags)
{
	switch(img.depth())
	{
		case CV_64F:
			dwt_cdf53_2i_d(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32F:
			dwt_cdf53_2i_s(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		case CV_32S:
			dwt_cdf53_2i_i(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				img.size().width,
				img.size().height,
				size.width,
				size.height,
				j,
				is_set(flags, DWT_EXTREME),
				is_set(flags, DWT_PADDING));
			break;
		default:
			CV_Error(CV_StsOutOfRange, "The matrix element depth value is out of range.");
	}
}

void dwt::transform(
	Mat &img,
	Size size,
	int &j,
	int flags)
{
	CV_Assert( img.depth() == CV_64F || img.depth() == CV_32F || img.depth() == CV_32S );
	CV_Assert( 1 == is_set(flags, DWT_FORWARD) + is_set(flags, DWT_INVERSE) );
	CV_Assert( 1 == is_set(flags, DWT_SIMPLE) + is_set(flags, DWT_SPARSE) + is_set(flags, DWT_PACKED) );
	CV_Assert( 1 >= is_set(flags, DWT_CDF53) + is_set(flags, DWT_CDF97) );

	if( is_set(flags, DWT_FORWARD) )
	{
		if( is_set(flags, DWT_SIMPLE) || is_set(flags, DWT_SPARSE) )
			resizePOT(img);
	
		if( is_set(flags, DWT_SIMPLE) || is_set(flags, DWT_PACKED) )
			size = img.size();

		for(int c = 0; c < img.channels(); c++)
		{
			if( is_set(flags, DWT_CDF97) )
				cv_dwt_cdf97_2f(img, c, size, j, flags);
			else
				cv_dwt_cdf53_2f(img, c, size, j, flags);
		}
	}
	else
	{
		if( is_set(flags, DWT_SIMPLE) || is_set(flags, DWT_SPARSE) )
			CV_Assert( isPOT(img) );

		if( is_set(flags, DWT_SIMPLE) || is_set(flags, DWT_PACKED) )
			size = img.size();

		for(int c = 0; c < img.channels(); c++)
		{
			if( is_set(flags, DWT_CDF97) )
				cv_dwt_cdf97_2i(img, c, size, j, flags);
			else
				cv_dwt_cdf53_2i(img, c, size, j, flags);
		}
	}
}

void dwt::transform(
	const Mat &src,
	Mat &dst,
	Size size,
	int &j,
	int flags)
{
	if(src.data == dst.data)
		dst = src; // no data is copied
	else
		dst = src.clone(); // full copy

	transform(
		dst,
		size,
		j,
		flags);
}

static
void dwt_util_test_image_fill(
	Mat &img,
	const int &channel,
	const Size &size)
{
	switch(img.depth())
	{
		case CV_64F:
			dwt_util_test_image_fill_d(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				size.width,
				size.height,
				channel);
			break;
		case CV_32F:
			dwt_util_test_image_fill_s(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				size.width,
				size.height,
				channel);
			break;
		case CV_32S:
			dwt_util_test_image_fill_i(
				img.data+img.elemSize1()*channel,
				img.step,
				img.elemSize(),
				size.width,
				size.height,
				channel);
			break;
		default:
			CV_Error(CV_StsOutOfRange, "The matrix element depth value is out of range.");
	}
}

void dwt::createTestImage(
	Mat &img,
	const Size &size,
	int type)
{
	img.create(size, type);

	for(int c = 0; c < img.channels(); c++)
		dwt_util_test_image_fill(
			img,
			c,
			img.size());
}

enum dwt_subbands conv_band(int band)
{
	enum dwt_subbands conv[dwt::DWT_HH+1];

	conv[dwt::DWT_LL] = ::DWT_LL;
	conv[dwt::DWT_HL] = ::DWT_HL;
	conv[dwt::DWT_LH] = ::DWT_LH;
	conv[dwt::DWT_HH] = ::DWT_HH;

	return conv[band];
}

Mat dwt::subband(
	const Mat &src,
	const Size size,
	int j,
	int band)
{
	CV_Assert( band >= dwt::DWT_LL && band <= dwt::DWT_HH );

	void *ptr = src.data;
	const int stride_x = src.step;
	const int stride_y = src.elemSize();
	const int inner_x = size.width;
	const int inner_y = size.height;
	const int outer_x = src.cols;
	const int outer_y = src.rows;
	
	void *subband_ptr;
	int subband_size_x;
	int subband_size_y;

	dwt_util_subband(
		ptr,
		stride_x,
		stride_y,
		outer_x,
		outer_y,
		inner_x,
		inner_y,
		j,
		conv_band(band),
		&subband_ptr,
		&subband_size_x,
		&subband_size_y);
	
	const int sizes[] = {subband_size_y, subband_size_x};
	const size_t steps[] = {stride_x};
	void *data = subband_ptr;

	return Mat(/*dims*/2, /*sizes*/sizes, /*type*/src.type(), /*data*/data, /*steps*/steps);
}
