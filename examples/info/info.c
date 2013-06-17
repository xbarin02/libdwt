/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Simple application showing various system informations.
 */

#include "libdwt.h"
#include <stdio.h> // fopen, fscanf, fgets, fclose
#include <string.h> // strcmp, strchr, strstr
#include <ctype.h> // isspace

int main()
{
	dwt_util_print_info();

	dwt_util_log(LOG_INFO, "CPU flags:\n");
	#define CPUINFO_MAX 8192
	char buff[CPUINFO_MAX] = {0};
	FILE *fcpuinfo = fopen("/proc/cpuinfo", "r");
	if(!fcpuinfo)
	{
		dwt_util_log(LOG_DBG, "cpuinfo opening failure!\n");
		goto close_cpuinfo;
	}

	// opened
	while( strcmp("flags", buff) )
	{
		int res = fscanf(fcpuinfo, "%s", buff);
		if( EOF == res )
		{
			dwt_util_log(LOG_DBG, "cpuinfo parsing failure!\n");
			goto close_cpuinfo;
		}
	}

	// got 'flags' line
	char *ptr = fgets(buff, CPUINFO_MAX, fcpuinfo);
	if(!ptr)
	{
		dwt_util_log(LOG_DBG, "fgets failure!\n");
		goto close_cpuinfo;
	}
	ptr = strchr(ptr, ':');
	if(!ptr)
	{
		dwt_util_log(LOG_DBG, "strchr failure!\n");
		goto close_cpuinfo;
	}
	ptr++;

	// got content of 'flags'
	struct {char *key, *desc;} flags[] = {
		{ .key="lm", .desc="long mode (64-bit mode)" },
		{ .key="mmx", .desc="MMX" },
		{ .key="3dnow", .desc="3DNow!" },
		{ .key="sse", .desc="SSE (Streaming SIMD Extensions)" },
		{ .key="sse2", .desc="SSE2 (Streaming SIMD Extensions 2)" },
		{ .key="pni", .desc="SSE3 (Streaming SIMD Extensions 3)" },
		{ .key="ssse3", .desc="SSSE3 (Supplemental Streaming SIMD Extensions 3)" },
		{ .key="sse4_1", .desc="SSE4.1 (Streaming SIMD Extensions 4.1)" },
		{ .key="sse4_2", .desc="SSE4.2 (Streaming SIMD Extensions 4.2)" },
		{ .key="sse4a", .desc="SSE4A (Streaming SIMD Extensions 4a)" },
		{ .key="avx", .desc="AVX (Advanced Vector Extensions)" },
		{ .key="avx2", .desc="AVX2 (Advanced Vector Extensions 2)" },
		{ .key="fma4", .desc="FMA4 (4-operand Fused Multiply/Add)" },
		{ .key="fma", .desc="FMA (Fused Multiply/Add)" },
	};
	const int nflags = sizeof(flags)/sizeof(*flags);
	for(int f = 0; f < nflags; f++)
	{
		char *flag = strstr(ptr, flags[f].key);
		if( flag )
		{
			int cl = *(flag-1);
			int cr = *(flag+strlen(flags[f].key));
			if( isspace(cl) && isspace(cr) )
			{
				dwt_util_log(LOG_INFO, "%s\n", flags[f].desc);
			}
		}
	}

close_cpuinfo:
	fclose(fcpuinfo);

	return 0;
}
