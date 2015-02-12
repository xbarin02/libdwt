#include <libdwt.h>
#include "cores.h"
#include "image.h"
#include <stddef.h>
#include "system.h"
#include <assert.h>
#include "fix.h"
#include "image2.h"

int main()
{
	dwt_util_init();

	dwt_util_set_cpufreq();
	dwt_util_set_realtime_scheduler();
	dwt_util_env_single_threading();

	// NOTE: TYPE_FLOAT32 TYPE_INT32 TYPE_FIX32 TYPE_FIX16
	enum dwt_types data_type = TYPE_FIX16;

	const int opt_stride = 2;
	const int size_pel = sizeof_type(data_type);

	const int size = 512;
	const int pixels = size*size;
	const int bytes = pixels*sizeof(float);

	dwt_util_log(LOG_DBG, "allocate roughly %u MiB\n", (bytes*2)>>20);

	// alloc
	struct image_t *source = image2_create_ex(size_pel, size, size, opt_stride);
	struct image_t *target = image2_create_ex(size_pel, size, size, opt_stride);

	// fill
	image2_fill(source, data_type);

#if 1
	// (optionally) dump
	image2_save_to_pgm(source, "source.pgm", data_type);
#endif

#if 1
	// (optionally) invalidate cache
	image2_flush_cache(source);
	image2_flush_cache(target);
#endif

	// forward core
	image2_fdwt_cdf97_op(source, target, data_type);

#if 1
	// (optionally) dump log
	image2_save_log_to_pgm(target, "subbands.pgm", data_type);
#endif

	// inverse separable
	image2_idwt_cdf97_ip(target, data_type);

#if 1
	// (optionally) dump
	image2_save_to_pgm(target, "target.pgm", data_type);
#endif

	// compare
	if( image2_compare(source, target, data_type) )
		dwt_util_log(LOG_ERR, "fail\n");
	else
		dwt_util_log(LOG_INFO, "pass\n");

	dwt_util_log(LOG_INFO, "mse = %.14f\n",
		image2_mse(source, target, data_type)
	);

	// free
	image_destroy(source);
	image_destroy(target);

	dwt_util_finish();

	return 0;
}
