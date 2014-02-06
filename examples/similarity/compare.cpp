#include <vector>
#include <algorithm>
#include <numeric>
#include <cv.h>
#include <highgui.h>
#include <cmath>

using namespace std;
using namespace cv;

float mse(Mat a, Mat b)
{
	CV_Assert( a.depth() == CV_32F && b.depth() == CV_32F );

	double area = a.cols*a.rows;

	Mat diff = abs(a-b);
	pow(diff, 2.0, diff);

	double r = sum(diff)[0]/area;

	return r;
}

float psnr(Mat a, Mat b)
{
	float r_mse = 1/mse(a, b);

	return 10 * log10f(r_mse);
}

// http://docs.opencv.org/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html
Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d     = CV_32F;

	Mat I1, I2;
	i1.convertTo(I1, d);           // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	Mat I2_2   = I2.mul(I2);        // I2^2
	Mat I1_2   = I1.mul(I1);        // I1^2
	Mat I1_I2  = I1.mul(I2);        // I1 * I2

	/***********************PRELIMINARY COMPUTING ******************************/

	Mat mu1, mu2;   //
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2   =   mu1.mul(mu1);
	Mat mu2_2   =   mu2.mul(mu2);
	Mat mu1_mu2 =   mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
	return mssim;
}

float ssim(const Mat& a, const Mat& b)
{
	return getMSSIM(a, b).val[0];
}

void findBestOffset(const Mat& a, const Mat& b, int &dx, int &dy, int range = 10)
{
	const int size_x = a.size().width -2*range;
	const int size_y = a.size().height-2*range;

	Mat va(a, Rect(range,range,size_x, size_y));

	float max = 0.f;
	dx = 0;
	dy = 0;

	for(int yy=-range; yy<=+range; yy++)
		for(int xx=-range; xx<=+range; xx++)
		{
			Mat vb(b, Rect(range+xx,range+yy,size_x, size_y));

			Scalar s = sum(va.mul(vb));

			if( s[0] > max )
			{
				max = s[0];
				dx = xx;
				dy = yy;
			}
		}
}

float calcDot(const Mat& A, const Mat& B)
{
//	const int ksize = a.size().width;
// 	Mat k = Mat::zeros(ksize, ksize, CV_32F);
// 	k.at<float>(ksize/2, ksize/2) = 1.f;
// 	GaussianBlur(k, k, Size(ksize*2+1, ksize*2+1), 0);
// 	normalize(k, k, ksize*ksize, 0., NORM_L1);
// 	imshow("k", k);

	Mat a = A.clone();
	Mat b = B.clone();

	Scalar amean, astddev;
	Scalar bmean, bstddev;

	meanStdDev(a, amean, astddev);
	meanStdDev(b, bmean, bstddev);

	a -= amean;
	b -= bmean;

// 	Mat m = a.mul(b).mul(k);
	Mat m = a.mul(b);
	Scalar s = sum(m);

	return s[0];
}

float patches(const Mat& a, const Mat& b)
{
	std::vector<cv::Point2f> c;
	const int maxCorners = 128;
	const double minDistance = 50;
	goodFeaturesToTrack(a, c, maxCorners, 0.20, minDistance);

	const int radius = 10;
	const int window = 60;

	cerr << "DEBUG: corners=" << c.size() << endl;

	int total = 0;
	int hit = 0;
	std::vector<float> dists;

	for(std::vector<cv::Point2f>::size_type i = 0; i != c.size(); i++)
	{
		if( c[i].x < window+radius || c[i].y < window+radius )
			continue;
		if( c[i].x > a.size().width-window-radius || c[i].y > a.size().height-window-radius )
			continue;

		// reference patch
		Mat r(a, Rect(c[i].x-window/2, c[i].y-window/2, window, window));
		
// 		imshow("r", r);

		float max_f = 0.f;
		int max_xx = 0, max_yy = 0;
		for(int xx=-radius; xx<=+radius; xx++)
			for(int yy=-radius; yy<=+radius; yy++)
			{
				// test patch
				Mat t(b, Rect(xx+c[i].x-window/2, yy+c[i].y-window/2, window, window));
				float f = fabsf(calcDot(r, t));

				if( f > max_f )
				{
					max_f = f;
					max_xx = xx;
					max_yy = yy;
				}
			}

		total++;
		if( abs(max_xx) >= radius || abs(max_yy) >= radius )
			continue;
		hit++;
		float d = sqrt(max_xx*max_xx+max_yy*max_yy);
// 		cerr << "d=" << d << endl;
		dists.push_back(d);

// 		Mat t(b, Rect(max_xx+c[i].x-window/2, max_yy+c[i].y-window/2, window, window));
// 		imshow("t", t);
// 		waitKey();
	}

	sort(dists.begin(), dists.end());

	float med = dists.at(hit/2);
	float avg = std::accumulate(dists.begin(), dists.end(), 0.f) / (float)dists.size();

	cerr << "DEBUG: hits=" << hit << "/" << total << " med=" << med << " avg=" << avg << endl;

	return avg;
}

int main(int argc, char **argv)
{
	bool crop = true;

	if( 3 != argc )
	{
		cerr << "Usage: " << argv[0] << " <reference.png> <compared.png>" << endl;
	}

	const char *path0 = (argc>1) ? argv[1] : "data/reference.png";
	const char *path1 = (argc>2) ? argv[2] : "data/compared.png";

	Mat img0;
	Mat img1;

	imread(path0, CV_LOAD_IMAGE_GRAYSCALE).convertTo(img0, CV_32F);
	imread(path1, CV_LOAD_IMAGE_GRAYSCALE).convertTo(img1, CV_32F);

	img0 /= 256;
	img1 /= 256;

	if( !img0.data )
	{
		cerr << "Unable to load input image: " << path0 << endl;
		return 1;
	}
	if( !img1.data )
	{
		cerr << "Unable to load input image: " << path1 << endl;
		return 1;
	}

	if( img0.size() != img1.size() )
	{
		cerr << "Sizes: " << img0.size().width << "x" << img0.size().height << endl;
		cerr << "Sizes: " << img1.size().width << "x" << img1.size().height << endl;

		if( crop )
		{
			Mat &big = img0.size().width > img1.size().width
				? img0
				: img1;
			Mat &sml = img0.size().width > img1.size().width
				? img1
				: img0;

			Rect rect(
				(big.size().width  - sml.size().width )/2,
				(big.size().height - sml.size().height)/2,
				sml.size().width,
				sml.size().height
			     );

			big = big(rect);
		}
		else
		{
			Size size = img0.size().width > img1.size().width
				? img0.size()
				: img1.size();

			if( size != img0.size() )
				resize(img0, img0, size, 0, 0, INTER_LANCZOS4);
			if( size != img1.size() )
				resize(img1, img1, size, 0, 0, INTER_LANCZOS4);;
		}

		cerr << "Sizes: " << img0.size().width << "x" << img0.size().height << endl;
		cerr << "Sizes: " << img1.size().width << "x" << img1.size().height << endl;
#if 0
		imshow("img0", img0);
		imshow("img1", img1);

		waitKey();
#endif
	}

	int dx,dy;
	const int range = 10;
	findBestOffset(img0, img1, dx, dy, range);

	cout << std::fixed;
	cout << "global=" << sqrt(dx*dx+dy*dy) << endl;

	img0 = Mat(img0, Rect(range+0,  range+0,  img0.size().width-range*2, img0.size().height-range*2));
	img1 = Mat(img1, Rect(range+dx, range+dy, img1.size().width-range*2, img1.size().height-range*2));

	cout << "mse="     << mse(img0, img1)     << endl;
	cout << "psnr="    << psnr(img0, img1)    << endl;
	cout << "ssim="    << ssim(img0, img1)    << endl;
	cout << "patches=" << patches(img0, img1) << endl;

	return 0;
}
