/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Example application showing variants of image fast wavelet transform.
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

#define TIMER_INIT double T
#define TIMER_START T = (double)getTickCount()
#define TIMER_PRINT cout << "time: " << ((double)getTickCount() - T)/getTickFrequency() << " secs" << endl
#define COMPARE(a,b,s) cout << "error: " << norm(a(Rect(Point(0, 0), s)) - b(Rect(Point(0, 0), s))) << endl

/**
 * Discrete wavelet transform demo.
 */
void demo(const Mat &src, int flags = DWT_SIMPLE, int j = -1)
{
	Mat big = src.clone();

	imshow("source", big);

	TIMER_INIT;

	TIMER_START;
	transform(big, src.size(), j, DWT_FORWARD|flags);
	TIMER_PRINT;

	wtshow("transform", big);

	TIMER_START;
	transform(big, src.size(), j, DWT_INVERSE|flags);
	TIMER_PRINT;

	imshow("reconstructed", big);

	COMPARE(src, big, src.size());

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

	demo(f, DWT_SIMPLE, 2);

	demo(f, DWT_SIMPLE);

	demo(f, DWT_SPARSE);

	demo(f, DWT_SPARSE|DWT_PADDING);

	demo(f, DWT_PACKED);

	return 0;
}
