#include "system.h"

// dwt_util_error
#include "libdwt.h"

// malloc, free
#include <stdlib.h> 

// size_t
#include <stddef.h>

// assert
#include <assert.h>

// memcpy, str*
#include <string.h>

// PATH_MAX
#include <limits.h>
#ifndef microblaze
#include <linux/limits.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

// memalign
#include <malloc.h>

// mmap
#include <sys/mman.h>

// sched_setscheduler
#include <sched.h>

// errno
#include <errno.h>

// va_*
#include <stdarg.h>

// sysconf
#include <unistd.h>

static
size_t get_path_max()
{
	return PATH_MAX;
}

size_t dwt_util_get_path_max()
{
	return get_path_max();
}

void *dwt_util_alloc1(
	size_t size
)
{
	void *ptr = malloc(size);

	return ptr;
}

void *dwt_util_reliably_alloc1(
	size_t size
)
{
	void *ptr = malloc(size);

	if( !ptr )
	{
		dwt_util_error("Unable to allocate memory.\n");
	}

	return ptr;
}

void *dwt_util_alloc2(
	unsigned long elems,
	size_t size
)
{
	return dwt_util_alloc1(elems*size);
}

void *dwt_util_reliably_alloc2(
	unsigned long elems,
	size_t size
)
{
	return dwt_util_reliably_alloc1(elems*size);
}

void dwt_util_free(
	void *ptr
)
{
	free(ptr);
}

void *dwt_util_memcpy_stride_s(
	void *restrict dst,
	size_t stride_dst,
	const void *restrict src,
	size_t stride_src,
	int elements		///< Number of floats to be copied, not number of bytes.
)
{
	assert( dst && src );

	const size_t size = sizeof(float);

	if( size == stride_src && size == stride_dst )
	{
		memcpy(dst, src, elements*size);
	}
	else
	{
		char *restrict ptr_dst = (char *restrict)dst;
		const char *restrict ptr_src = (const char *restrict)src;
		for(int i = 0; i < elements; i++)
		{
			*(float *restrict)ptr_dst = *(const float *restrict)ptr_src;

			ptr_dst += stride_dst;
			ptr_src += stride_src;
		}
	}

	return dst;
}

void *dwt_util_memcpy_stride_i(
	void *restrict dst,
	size_t stride_dst,
	const void *restrict src,
	size_t stride_src,
	int elements		///< Number of ints to be copied, not number of bytes.
)
{
	assert( dst && src );

	const size_t size = sizeof(int);

	if( size == stride_src && size == stride_dst )
	{
		memcpy(dst, src, elements*size);
	}
	else
	{
		char *restrict ptr_dst = (char *restrict)dst;
		const char *restrict ptr_src = (const char *restrict)src;
		for(int i = 0; i < elements; i++)
		{
			*(int *restrict)ptr_dst = *(const int *restrict)ptr_src;

			ptr_dst += stride_dst;
			ptr_src += stride_src;
		}
	}

	return dst;
}

void *dwt_util_memcpy_stride_d(
	void *restrict dst,
	size_t stride_dst,
	const void *restrict src,
	size_t stride_src,
	int elements		///< Number of doubles to be copied, not number of bytes.
)
{
	assert( dst && src );

	const size_t size = sizeof(double);

	if( size == stride_src && size == stride_dst )
	{
		memcpy(dst, src, elements*size);
	}
	else
	{
		char *restrict ptr_dst = (char *restrict)dst;
		const char *restrict ptr_src = (const char *restrict)src;
		for(int i = 0; i < elements; i++)
		{
			*(double *restrict)ptr_dst = *(const double *restrict)ptr_src;

			ptr_dst += stride_dst;
			ptr_src += stride_src;
		}
	}

	return dst;
}

static
void *alloc_aligned_ex(
	unsigned elements,
	size_t elem_size,
	size_t align
)
{
	assert( is_pow2(elem_size) );

	const size_t size = elements * elem_size;

	void *addr = (void *)0;

	addr = (void *)memalign(align, size);

	assert( is_aligned(addr, align) );

	return addr;
}

void *dwt_util_alloc_aligned_ex(
	unsigned elements,
	size_t elem_size,
	size_t align
)
{
	return alloc_aligned_ex(elements, elem_size, align);
}

static
void *alloc_aligned_ex_reliably(
	unsigned elements,
	size_t elem_size,
	size_t align
)
{
	void *ptr = alloc_aligned_ex(elements, elem_size, align);

	if( !ptr )
		dwt_util_error("out of memory\n");

	return ptr;
}

void *dwt_util_alloc_aligned_ex_reliably(
	unsigned elements,
	size_t elem_size,
	size_t align
)
{
	return alloc_aligned_ex_reliably(elements, elem_size, align);
}

void *dwt_util_alloc_locked(size_t size)
{
	const size_t page_mask = 4096-1;

	// align
	size += page_mask;
	size &= ~page_mask;

	dwt_util_log(LOG_DBG, "%s(%zi) ~= %zi MiB\n", __FUNCTION__, size, size>>20);

	void *address = mmap(
		NULL,
		size,
		PROT_READ|PROT_WRITE,
		MAP_PRIVATE|MAP_ANONYMOUS|MAP_LOCKED|MAP_POPULATE,
		-1,
		0
	);

	if( MAP_FAILED == address )
	{
		perror(0);
		dwt_util_error("mmap failed :( check your 'ulimit -l'\n");
	}

	return address;
}

void *dwt_util_reliably_alloc_locked(
	size_t size
)
{
	static int g_reliably_alloc_locked_error = 0;

	void *ptr = malloc(size);

	if( !ptr )
	{
		dwt_util_error("Unable to allocate memory.\n");
	}

	if( -1 == mlock(ptr, size) )
	{
		if( !g_reliably_alloc_locked_error )
		{
			perror(0);
			dwt_util_log(LOG_WARN, "unable to lock memory :( this error will be reported only once\n");
		}
		g_reliably_alloc_locked_error++;
	}

	return ptr;
}

void dwt_util_set_realtime_scheduler()
{
	struct sched_param param = { .sched_priority = 50 };

	if( -1 == sched_setscheduler(0, SCHED_RR, &param) )
	{
		perror(0);
		dwt_util_log(LOG_WARN, "unable to set scheduling policy :(\n");
	}
}

void dwt_util_set_affinity()
{
	cpu_set_t mask;

	if( -1 == sched_getaffinity(0, sizeof(cpu_set_t), &mask) )
	{
		perror(0);
		dwt_util_log(LOG_WARN, "unable to get CPU affinity :(\n");
		return;
	}

	unsigned cpu_count = 0;
	unsigned last_set = 0;
	for(unsigned c = 0; c < sizeof(cpu_set_t); c++)
	{
		int is_set = CPU_ISSET(c, &mask);
		cpu_count += is_set;
		if( is_set )
			last_set = c;

		//printf("CPU(%i): %i\n", c, is_set);
	}

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "#CPUs=%i last=%i\n", cpu_count, last_set);
#endif

	CPU_ZERO(&mask);
	CPU_SET(last_set, &mask);

	if( -1 == sched_setaffinity(0, sizeof(cpu_set_t), &mask) )
	{
		perror(0);
		dwt_util_log(LOG_WARN, "unable to set CPU affinity :(\n");
	}
}

long unsigned dwt_util_get_page_fault()
{
	FILE *file_stat = fopen("/proc/self/stat", "r");

	if( !file_stat )
	{
		dwt_util_error("unable to open stat file\n");
	}

	int pid;
	char comm[4096];
	char state;
	int ppid;
	int pgrp;
	int session;
	int tty_nr;
	int tpgid;
	unsigned flags;
	long unsigned minflt;
	long unsigned cminflt;
	long unsigned majflt;
	long unsigned cmajflt;
	long unsigned utime;
	long unsigned stime;
	long int cutime;
	long int cstime;

	// NOTE: see proc(5)
	int ret_code = fscanf(
		file_stat,
	       "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu %ld %ld",
		/* 1*/ &pid,
		/* 2*/ comm,
		/* 3*/ &state,
		/* 4*/ &ppid,
		/* 5*/ &pgrp,
		/* 6*/ &session,
		/* 7*/ &tty_nr,
		/* 8*/ &tpgid,
		/* 9*/ &flags,
		/*10*/ &minflt,
		/*11*/ &cminflt,
		/*12*/ &majflt,
		/*13*/ &cmajflt,
		/*14*/ &utime,
		/*15*/ &stime,
		/*16*/ &cutime,
		/*17*/ &cstime
	);

	if( 17 != ret_code )
	{
		dwt_util_error("stat file read error\n");
	}

	fclose(file_stat);

	return minflt+majflt;
}

int dwt_util_get_cpu_count()
{
	// _SC_NPROCESSORS_CONF _SC_NPROCESSORS_ONLN
	long count = sysconf(_SC_NPROCESSORS_ONLN);

	return (int)count;
}

const char *fopen_gets(const char *format, ...)
{
	static char path[4096];

	va_list args;
	va_start(args, format);

	vsprintf(path, format, args);

	va_end(args);

	FILE *file = fopen(path, "r");

	if( !file )
		return NULL;

	if( !fgets(path, 4096, file) )
		return NULL;

	fclose(file);

	for(size_t c = 0; c < strlen(path); c++)
	{
		if( '\n' == path[c]  )
			path[c] = 0;
	}

	return path;
}

int fopen_puts(const char *string, const char *format, ...)
{
	static char path[4096];

	va_list args;
	va_start(args, format);

	vsprintf(path, format, args);

	va_end(args);

	FILE *file = fopen(path, "w");

	if( !file )
		return -1;

	if( EOF == fputs(string, file) )
		return -2;

	fclose(file);

	return 0;
}

void dwt_util_set_cpufreq()
{
	// for each cpu
	for(int c = 0; c < dwt_util_get_cpu_count(); c++)
	{
		// echo /sys/devices/system/cpu/cpu%i/cpufreq/scaling_governor
#ifdef DEBUG
		dwt_util_log(LOG_DBG, "cpu %i governor: %s\n", c, fopen_gets("/sys/devices/system/cpu/cpu%i/cpufreq/scaling_governor", c));
#endif
		char frequencies[4096];
		const char *temp = fopen_gets("/sys/devices/system/cpu/cpu%i/cpufreq/scaling_available_frequencies", c);
		if( !temp )
			continue;
		strcpy(frequencies, temp);
#ifdef DEBUG
		dwt_util_log(LOG_DBG, "cpu %i frequencies: %s\n", c, frequencies);
#endif

		// for each frequency in /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
		for(char *begin = frequencies, *end; begin && (end = strstr(begin, " ")); begin = end+1)
		{
			*end = 0;

			// set frequency
			fopen_puts(begin, "/sys/devices/system/cpu/cpu%i/cpufreq/scaling_min_freq", c);
			fopen_puts(begin, "/sys/devices/system/cpu/cpu%i/cpufreq/scaling_cur_freq", c);

			// get frequency
			if( strcmp(begin, fopen_gets("/sys/devices/system/cpu/cpu%i/cpufreq/scaling_cur_freq", c)) )
			{
				dwt_util_log(LOG_WARN, "cannot set frequency %s for cpu %i\n", begin, c);
			}
			else
			{
				dwt_util_log(LOG_INFO, "frequency %s for cpu %i was set\n", begin, c);
				break;
			}
		}
	}
}

void dwt_util_env_single_threading()
{
	dwt_util_set_affinity();

	dwt_util_set_num_threads(1);
	dwt_util_set_num_workers(1);
}
