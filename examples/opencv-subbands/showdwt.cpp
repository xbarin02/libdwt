/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Example application demonstrating subband access.
 */
#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "cvdwt.h"

/**
 * This example can work with grayscale or color image.
 */
#define WITH_COLOR

/**
 * This example can work with single or double precision floating point numbers.
 */
#define USE_DOUBLE

using namespace std;
using namespace cv;
using namespace dwt;

/**
 * Subband access demo.
 */
void demo(const Mat &src, int flags = DWT_SIMPLE, int j = -1)
{
	// make a copy of input image
	Mat big = src.clone();

	// show input image
	imshow("source", big);

	// forward transform
	transform(big, src.size(), j, DWT_FORWARD|flags);

	// show transform
	wtshow("transform", big);

	// access to subbands at level "j"
	Mat LL = subband(big, src.size(), j, DWT_LL);
	Mat HL = subband(big, src.size(), j, DWT_HL);
	Mat LH = subband(big, src.size(), j, DWT_LH);
	Mat HH = subband(big, src.size(), j, DWT_HH);

	// show these subbands
	imshow("LL", LL);
	if( j != 0 )
	{
		imshow("LH", LH);
		imshow("HL", HL);
		imshow("HH", HH);
	}

	// erase HL subband at all the levels
	for(int jj = 1; jj <= j; jj++)
	{
		cout << "erasing HL subband at level " << jj << "..." << endl;
		subband(big, src.size(), jj, DWT_HL) = Scalar(0);
	}

	// show transform with out these HL subbands
	wtshow("transform w/o HL", big);

	// inverse transform such a distorted transform
	transform(big, src.size(), j, DWT_INVERSE|flags);

	// show reconstructed image with distortion
	imshow("reconstructed", big);

	// compare input and distorted images
	cout << "error: " << norm(src(Rect(Point(0, 0), src.size())) - big(Rect(Point(0, 0), src.size()))) << endl;

	waitKey();
}

int main(int argc, char **argv)
{
	Mat f;

	const char *imagename = argc > 1 ? argv[1] : "Lenna.png";
	cout << "Loading " << imagename << endl;

	imread(imagename,
#ifdef WITH_COLOR
		CV_LOAD_IMAGE_COLOR
#else
		CV_LOAD_IMAGE_GRAYSCALE
#endif
	).convertTo(f,
#ifdef USE_DOUBLE
		CV_64F
#else
		CV_32F
#endif
	);
	f /= 256;

	if(!f.data)
	{
		cerr << "Unable to load input image, using test image" << endl;

		createTestImage(
			f,
			Size(512, 512),
#ifdef USE_DOUBLE
	#ifdef WITH_COLOR
			CV_64FC3
	#else
			CV_64FC1
	#endif
#else
	#ifdef WITH_COLOR
			CV_32FC3
	#else
			CV_32FC1
	#endif
#endif
		);
	}

	demo(f, DWT_SPARSE, 2);
	demo(f, DWT_SPARSE);

	demo(f, DWT_PACKED, 2);
	demo(f, DWT_PACKED);

	demo(f, DWT_SIMPLE, 2);
	demo(f, DWT_SIMPLE);

	destroyAllWindows();

	return 0;
}
