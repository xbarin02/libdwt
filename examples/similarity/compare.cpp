#include <cv.h>
#include <highgui.h>
#include <math.h>

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

	cout << "mse=" << mse(img0, img1) << endl;
	cout << "psnr=" << psnr(img0, img1) << endl;
	cout << "ssim=" << ssim(img0, img1) << endl;

	return 0;
}
