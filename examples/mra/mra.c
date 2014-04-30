#include "libdwt.h"
#include <math.h>

// gnuplot 1-D
int dwt_util_save1_to_gp_s(
	const char *path,	///< target file name, e.g. "output.dat"
	const void *data,	///< pointer to beginning of signal data
	int size_x,		///< width of nested/inner signal (in elements)
	int stride_y		///< difference between columns (in bytes)
)
{
	// FIXME: assert

	FILE *file = fopen(path, "w");

	if( NULL == file )
		return 1;

	int symb_delim[] = { ',', ';', '\t', ' ' };
	int symb_newline[] = { '\n', '\r' };

	for(int x = 0; x < size_x; x++)
	{
		const float val_x = (float)x;
		const float val_y = *dwt_util_addr_coeff_const_s(
			data,
			0,
			x,
			0,
			stride_y
		);

		fprintf(file, "%f%c%f%c", val_x, symb_delim[2], val_y, symb_newline[0]);
	}

	fclose(file);

	return 0;
}

int main()
{
	const float f0 = 16.0;
	const float a0 = 0.2;
	const float f1 = 4.0;
	const float a1 = 1.0;

	// signal length
	const int N = 1024;

	// signal on the stack
	float x[N];

	const int stride_y = sizeof(*x);

	for(int n = 0; n < N; n++)
		x[n] =
			+ a0*sinf(2*M_PI*f0 * n/N)
			+ a1*sinf(2*M_PI*f1 * n/N);

	dwt_util_save_to_mat_s(
		"signal.mat",
		x,
		N,
		1,
		0,
		stride_y
	);

	dwt_util_save1_to_gp_s(
		"signal.dat",
		x,
		N,
		stride_y
	);

	int J = -1;

	// DWT
	 dwt_cdf97_1f_s(
		x,
		stride_y,
		N,
		N,
		&J,
		0
	);

	dwt_util_log(LOG_INFO, "J = %i\n", J);

	dwt_util_save_to_mat_s(
		"transform.mat",
		x,
		N,
		1,
		0,
		stride_y
	);

	const int preserve_j = 7; // 7, 5

	// reset H bands
	for(int j = 1; j <= J; j++)
	{
		void *band;
		int band_N;
		int band_size_y;

		dwt_util_subband_s(x,
			0, stride_y,
			N, 1,
			N, 1,
			j, DWT_HL,
			&band, &band_N, &band_size_y);

		dwt_util_log(LOG_DBG, "H subband %2i of size N=%3i\n", j, band_N);

		if( preserve_j != j )
		{
			for(int n = 0; n < band_N; n++)
			{
				*dwt_util_addr_coeff_s(band, 1, n, 0, stride_y) = 0.0f;
			}
		}
	}

	// reset L band
	{
		void *band;
		int band_N;
		int band_size_y;

		dwt_util_subband_s(x,
			0, stride_y,
			N, 1,
			N, 1,
			J, DWT_LL,
			&band, &band_N, &band_size_y);

		dwt_util_log(LOG_DBG, "L subband %2i of size N=%3i\n", J, band_N);

		if( preserve_j <= J )
		{
			for(int n = 0; n < band_N; n++)
			{
				*dwt_util_addr_coeff_s(band, 1, n, 0, stride_y) = 0.0f;
			}
		}
	}

	dwt_util_save_to_mat_s(
		"sparse.mat",
		x,
		N, 1,
		0, stride_y
	);

	// inverse DWT
	dwt_cdf97_1i_s(
		x,
		stride_y,
		N,
		N,
		J,
		0
	);

	dwt_util_save_to_mat_s(
		"level.mat",
		x,
		N, 1,
		0, stride_y
	);

	dwt_util_save1_to_gp_s(
		"level.dat",
		x,
		N,
		stride_y
	);

	return 0;
}
