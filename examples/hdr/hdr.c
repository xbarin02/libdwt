#include "libdwt.h"
#include "exr.h"

int main()
{
	const char *path = "./data/KernerEnvLatLong.exr";

	void *ptr;
	int size_x, size_y, stride_x, stride_y;

	dwt_util_log(LOG_INFO, "Loading '%s'...\n", path);

	dwt_util_load_from_exr_s(
		path,
		"G",
		&ptr,
		&stride_x,
		&stride_y,
		&size_x,
		&size_y
	);

	dwt_util_log(LOG_INFO, "Loaded %ix%i\n", size_x, size_y);

	dwt_util_save_to_pgm_s(
		"output_g.pgm",
		1.f,
		ptr,
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	return 0;
}
