#include "clock.h"
#include "libdwt.h"

dwt_clock_t g_clock_start;
int g_clock_type;

void dwt_quick_timer_start()
{
	g_clock_type = dwt_util_clock_autoselect();
	g_clock_start = dwt_util_get_clock(g_clock_type);
}

void dwt_quick_timer_dump()
{
	dwt_clock_t clock_stop = dwt_util_get_clock(g_clock_type);
	dwt_util_log(LOG_INFO, "elapsed time: %f secs\n", (double)(clock_stop - g_clock_start) / dwt_util_get_frequency(g_clock_type));
}
