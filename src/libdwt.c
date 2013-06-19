/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Fast wavelet transform implemented via lifting scheme.
 */
#include "libdwt.h"

#define MEASURE_PER_PIXEL
//#define TEMP_ON_HEAP
//#define DEBUG_VERBOSE
//#define DISABLE_MEMCPY
//#define ENABLE_LAZY_MEMCPY
//#define DISABLE_Y
//#define DISABLE_X
//#define MEASURE_FACTOR 1000
#define MEASURE_FACTOR 1

#define STRING(x) #x

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if (GCC_VERSION >= 30200) && (GCC_VERSION < 30300)
	#warning "GCC 3.2"
#endif
#if (GCC_VERSION >= 30300) && (GCC_VERSION < 30400)
	#warning "GCC 3.3"
#endif
#if (GCC_VERSION >= 30400) && (GCC_VERSION < 40000)
	#warning "GCC 3.4"
#endif
#if (GCC_VERSION >= 40000) && (GCC_VERSION < 40100)
	#warning "GCC 4.0"
#endif
#if (GCC_VERSION >= 40100) && (GCC_VERSION < 40200)
	#warning "GCC 4.1"
#endif
#if (GCC_VERSION >= 40200) && (GCC_VERSION < 40300)
	#warning "GCC 4.2"
#endif
#if (GCC_VERSION >= 40300) && (GCC_VERSION < 40400)
	#warning "GCC 4.3"
#endif
#if (GCC_VERSION >= 40400) && (GCC_VERSION < 40500)
	#warning "GCC 4.4"
#endif
#if (GCC_VERSION >= 40500) && (GCC_VERSION < 40600)
	#warning "GCC 4.5"
#endif
#if (GCC_VERSION >= 40600) && (GCC_VERSION < 40700)
	#warning "GCC 4.6"
#endif
#if (GCC_VERSION >= 40700) && (GCC_VERSION < 40800)
	#warning "GCC 4.7"
#endif
#if (GCC_VERSION >= 40800)
	#warning "GCC 4.8+"
#endif

#if (GCC_VERSION < 40300)
	#warning Missing GCC 4.3+
	#warning Missing __builtin___clear_cache function
	#define __builtin___clear_cache(begin,end)
#endif

#ifdef NDEBUG
	/* Release build */
	#undef DEBUG

	#define FUNC_BEGIN
	#define FUNC_END

	#define dbg(msg, ...)
#else
	/* Debug build */
	#ifndef DEBUG
		#define DEBUG
	#endif

	#ifdef DEBUG_VERBOSE
		#define FUNC_BEGIN dwt_util_log(LOG_DBG, "%s ENTRY\n", __FUNCTION__)
		#define FUNC_END   dwt_util_log(LOG_DBG, "%s EXIT\n",  __FUNCTION__)
	#else
		#define FUNC_BEGIN
		#define FUNC_END
	#endif

	#define dbg(msg, ...) dwt_util_log(LOG_DBG, ("%s: " msg), __FUNCTION__, ##__VA_ARGS__)
#endif

/** UTIA ASVP/EdkDSP specific code */
#ifdef microblaze
	#define WAL_NATIVE_DMA
	#include <wal.h>
	#include <wal_bce_dma.h>
	#include <bce_dma_config.h>

#ifndef BCE_DMA_CFGTABLE_NUM_ITEMS
	#warning BCE_DMA_CFGTABLE_NUM_ITEMS was not defined, using default value of 2
	#define BCE_DMA_CFGTABLE_NUM_ITEMS 2
#endif

	WAL_REGISTER_WORKER(worker0, BCE_DMA_GENERIC_4D, bce_dma_cfgtable, 0, 1, 0);
	WAL_REGISTER_WORKER(worker1, BCE_DMA_GENERIC_4D, bce_dma_cfgtable, 1, 1, 0);
	WAL_REGISTER_WORKER(worker2, BCE_DMA_GENERIC_4D, bce_dma_cfgtable, 2, 1, 0);
	WAL_REGISTER_WORKER(worker3, BCE_DMA_GENERIC_4D, bce_dma_cfgtable, 3, 1, 0);

	wal_worker_t *worker[BCE_DMA_CFGTABLE_NUM_ITEMS] = {
		&worker0_data_structure,
#if BCE_DMA_CFGTABLE_NUM_ITEMS > 1
		&worker1_data_structure,
#endif
#if BCE_DMA_CFGTABLE_NUM_ITEMS > 2
		&worker2_data_structure,
#endif
#if BCE_DMA_CFGTABLE_NUM_ITEMS > 3
		&worker3_data_structure,
#endif
	};

	#include "firmware/fw_fp01_lift4sa.h"
	#include "firmware/fw_fp01_lift4sb.h"

	#define BANK_SIZE 1024

	#define WAL_BANK_POS(off) ( off )

	#define WAL_DMA_MASK(ch) ( 1<<(ch) )

	#ifdef NDEBUG
		/* Release build */
		#define WAL_CHECK(expr) (expr)
	#else
		/* Debug build */
		#define WAL_CHECK(expr) ( wal_abort(STRING(expr), expr) )
	#endif
#endif

/** UNUSED macro */
#define UNUSED(expr) do { (void)(expr); } while (0)

#ifndef BANK_SIZE
	#define BANK_SIZE 4096
#endif

/** disable timers when using Par4All tool */
#if !defined(P4A)
	#define USE_TIME_CLOCK
	#define USE_TIME_CLOCK_GETTIME
	#define USE_TIME_CLOCK_GETTIME_REALTIME
	#define USE_TIME_CLOCK_GETTIME_MONOTONIC
	#define USE_TIME_CLOCK_GETTIME_MONOTONIC_RAW
	#define USE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
	#define USE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
	#define USE_TIME_TIMES
	#define USE_TIME_IOCTL_RTC
	#define USE_TIME_GETTIMEOFDAY
	#define USE_TIME_GETRUSAGE
	#define USE_TIME_GETRUSAGE_SELF
	#define USE_TIME_GETRUSAGE_CHILDREN
	#define USE_TIME_GETRUSAGE_THREAD
#endif

// FIXME: glibc only
#include <features.h>

/** include LINUX_VERSION_CODE and KERNEL_VERSION macros */
#if defined(__linux) && !defined(microblaze)
	#include <linux/version.h>
#endif

/** define HAVE_TIME_* macros when corresponding timers are available */
#if defined(_GNU_SOURCE) || defined(_ISOC99_SOURCE) || defined(_POSIX_C_SOURCE)
	#define HAVE_TIME_CLOCK
#endif

#if _POSIX_C_SOURCE >= 199309L || _XOPEN_SOURCE >= 500
	#define HAVE_TIME_CLOCK_GETTIME

	#ifdef _POSIX_C_SOURCE
		#include <unistd.h> // _POSIX_TIMERS, _POSIX_MONOTONIC_CLOCK, _POSIX_CPUTIME, _POSIX_THREAD_CPUTIME

		#ifdef _POSIX_TIMERS
			#define HAVE_TIME_CLOCK_GETTIME_REALTIME

			#ifdef _POSIX_MONOTONIC_CLOCK
				#define HAVE_TIME_CLOCK_GETTIME_MONOTONIC
			#endif

			#if defined(__linux) && !defined(microblaze)
				#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,28)
					// FIXME: glibc 2.12.1+
					#if defined(__GLIBC__)
						#if __GLIBC_PREREQ(2,12)
							#pragma message "INFO: Have glibc 2.12+"
							#define HAVE_TIME_CLOCK_GETTIME_MONOTONIC_RAW
						#endif
					#endif
				#endif
			#endif

			#ifdef _POSIX_CPUTIME
				#define HAVE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
			#endif

			#ifdef _POSIX_THREAD_CPUTIME
				#define HAVE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
			#endif
		#endif
	#else
		#define HAVE_TIME_CLOCK_GETTIME_REALTIME
		#define HAVE_TIME_CLOCK_GETTIME_MONOTONIC
		#define HAVE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
		#define HAVE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
	#endif
#endif

#if defined(_GNU_SOURCE) || defined(_SVID_SOURCE) || defined(_BSD_SOURCE) || defined(_POSIX_C_SOURCE)
	#define HAVE_TIME_TIMES
#endif

#if defined(__linux) && !defined(microblaze)
	#define HAVE_TIME_IOCTL_RTC
#endif

#if defined(_GNU_SOURCE) || defined(_SVID_SOURCE) || defined(_BSD_SOURCE) || defined(_POSIX_C_SOURCE)
	#define HAVE_TIME_GETTIMEOFDAY
#endif

#if defined(_GNU_SOURCE) || defined(_SVID_SOURCE) || defined(_BSD_SOURCE) || defined(_POSIX_C_SOURCE)
	#define HAVE_TIME_GETRUSAGE
	#define HAVE_TIME_GETRUSAGE_SELF
	#define HAVE_TIME_GETRUSAGE_CHILDREN

	#if defined(__linux) && !defined(microblaze)
		#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,26)
			// FIXME: glibc 2.15+ needed
			#if defined(__GLIBC__)
				#if __GLIBC_PREREQ(2,15)
					#pragma message "INFO: Have glibc 2.15+"
					#define HAVE_TIME_GETRUSAGE_THREAD
				#endif
			#endif
		#endif
	#endif
#endif

/** define ENABLE_TIME_* macros when they are available and intended for use */
#if defined(USE_TIME_CLOCK) && defined(HAVE_TIME_CLOCK)
	#define ENABLE_TIME_CLOCK
#endif

#if defined(USE_TIME_CLOCK_GETTIME) && defined(HAVE_TIME_CLOCK_GETTIME)
	#define ENABLE_TIME_CLOCK_GETTIME
#endif

#if defined(USE_TIME_CLOCK_GETTIME_REALTIME) && defined(HAVE_TIME_CLOCK_GETTIME_REALTIME)
	#define ENABLE_TIME_CLOCK_GETTIME_REALTIME
#endif

#if defined(USE_TIME_CLOCK_GETTIME_MONOTONIC) && defined(HAVE_TIME_CLOCK_GETTIME_MONOTONIC)
	#define ENABLE_TIME_CLOCK_GETTIME_MONOTONIC
#endif

#if defined(USE_TIME_CLOCK_GETTIME_MONOTONIC_RAW) && defined(HAVE_TIME_CLOCK_GETTIME_MONOTONIC_RAW)
	#define ENABLE_TIME_CLOCK_GETTIME_MONOTONIC_RAW
#endif

#if defined(USE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID) && defined(HAVE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID)
	#define ENABLE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
#endif

#if defined(USE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID) && defined(HAVE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID)
	#define ENABLE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
#endif

#if defined(USE_TIME_TIMES) && defined(HAVE_TIME_TIMES)
	#define ENABLE_TIME_TIMES
#endif

#if defined(USE_TIME_GETRUSAGE) && defined(HAVE_TIME_GETRUSAGE)
	#define ENABLE_TIME_GETRUSAGE
#endif

#if defined(USE_TIME_IOCTL_RTC) && defined(HAVE_TIME_IOCTL_RTC)
	#define ENABLE_TIME_IOCTL_RTC
#endif

#if defined(USE_TIME_GETTIMEOFDAY) && defined(HAVE_TIME_GETTIMEOFDAY)
	#define ENABLE_TIME_GETTIMEOFDAY
#endif

#if defined(USE_TIME_GETRUSAGE_SELF) && defined(HAVE_TIME_GETRUSAGE_SELF)
	#define ENABLE_TIME_GETRUSAGE_SELF
#endif

#if defined(USE_TIME_GETRUSAGE_CHILDREN) && defined(HAVE_TIME_GETRUSAGE_CHILDREN)
	#define ENABLE_TIME_GETRUSAGE_CHILDREN
#endif

#if defined(USE_TIME_GETRUSAGE_THREAD) && defined(HAVE_TIME_GETRUSAGE_THREAD)
	#define ENABLE_TIME_GETRUSAGE_THREAD
#endif

#pragma message "Enabled timers:"
#ifdef ENABLE_TIME_CLOCK
	#pragma message "TIME_CLOCK: enabled"
#else
	#pragma message "TIME_CLOCK: disabled"
#endif
#ifdef ENABLE_TIME_CLOCK_GETTIME
	#pragma message "TIME_CLOCK_GETTIME: enabled"
#else
	#pragma message "TIME_CLOCK_GETTIME: disabled"
#endif
#ifdef ENABLE_TIME_CLOCK_GETTIME_REALTIME
	#pragma message "TIME_CLOCK_GETTIME_REALTIME: enabled"
#else
	#pragma message "TIME_CLOCK_GETTIME_REALTIME: disabled"
#endif
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC
	#pragma message "TIME_CLOCK_GETTIME_MONOTONIC: enabled"
#else
	#pragma message "TIME_CLOCK_GETTIME_MONOTONIC: disabled"
#endif
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC_RAW
	#pragma message "TIME_CLOCK_GETTIME_MONOTONIC_RAW: enabled"
#else
	#pragma message "TIME_CLOCK_GETTIME_MONOTONIC_RAW: disabled"
#endif
#ifdef ENABLE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
	#pragma message "TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID: enabled"
#else
	#pragma message "TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID: disabled"
#endif
#ifdef ENABLE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
	#pragma message "TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID: enabled"
#else
	#pragma message "TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID: disabled"
#endif
#ifdef ENABLE_TIME_TIMES
	#pragma message "TIME_TIMES: enabled"
#else
	#pragma message "TIME_TIMES: disabled"
#endif
#ifdef ENABLE_TIME_GETRUSAGE
	#pragma message "TIME_GETRUSAGE: enabled"
#else
	#pragma message "TIME_GETRUSAGE: disabled"
#endif
#ifdef ENABLE_TIME_IOCTL_RTC
	#pragma message "TIME_IOCTL_RTC: enabled"
#else
	#pragma message "TIME_IOCTL_RTC: disabled"
#endif
#ifdef ENABLE_TIME_GETTIMEOFDAY
	#pragma message "TIME_GETTIMEOFDAY: enabled"
#else
	#pragma message "TIME_GETTIMEOFDAY: disabled"
#endif
#ifdef ENABLE_TIME_GETRUSAGE_SELF
	#pragma message "TIME_GETRUSAGE_SELF: enabled"
#else
	#pragma message "TIME_GETRUSAGE_SELF: disabled"
#endif
#ifdef ENABLE_TIME_GETRUSAGE_CHILDREN
	#pragma message "TIME_GETRUSAGE_CHILDREN: enabled"
#else
	#pragma message "TIME_GETRUSAGE_CHILDREN: disabled"
#endif
#ifdef ENABLE_TIME_GETRUSAGE_THREAD
	#pragma message "TIME_GETRUSAGE_THREAD: enabled"
#else
	#pragma message "TIME_GETRUSAGE_THREAD: disabled"
#endif

/** include necessary headers for selected timers */
#if defined(ENABLE_TIME_CLOCK_GETTIME) \
	|| defined(ENABLE_TIME_CLOCK_GETTIME_REALTIME) \
	|| defined(ENABLE_TIME_CLOCK_GETTIME_MONOTONIC) \
	|| defined(ENABLE_TIME_CLOCK_GETTIME_MONOTONIC_RAW) \
	|| defined(ENABLE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID) \
	|| defined(ENABLE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID)

	// NOTE: -lrt
	#include <time.h>  // struct timespec, clock_gettime, CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID
#endif

#ifdef ENABLE_TIME_CLOCK
	#include <time.h> // clock, CLOCKS_PER_SEC
#endif

#ifdef ENABLE_TIME_TIMES
	#include <sys/times.h> // struct tms, times
	#include <unistd.h> // sysconf, _SC_CLK_TCK
#endif

#ifdef ENABLE_TIME_IOCTL_RTC
	#include <sys/ioctl.h> // ioctl
	#include <linux/rtc.h> // struct rtc_time, RTC_RD_TIME
	#include <fcntl.h> // open, O_NONBLOCK, O_RDONLY
	#include <unistd.h> // close
	#include <time.h> // struct tm, mktime
#endif

#ifdef ENABLE_TIME_GETTIMEOFDAY
	#include <sys/time.h> // struct timeval, gettimeofday
#endif

#if defined(ENABLE_TIME_GETRUSAGE) \
	|| defined(ENABLE_TIME_GETRUSAGE_SELF) \
	|| defined(ENABLE_TIME_GETRUSAGE_CHILDREN) \
	|| defined(ENABLE_TIME_GETRUSAGE_THREAD)

	#include <time.h> // struct timespec
	#include <unistd.h>
	#include <sys/resource.h> // getrusage, RUSAGE_SELF, struct rusage
	#include <sys/time.h> // struct timeval, TIMEVAL_TO_TIMESPEC
#endif

/** other headers */
#include <assert.h> // assert
#include <stddef.h> // NULL, size_t

#ifdef __x86_64__
#ifndef __SSE__
	 // HACK(x86): stdlib.h: error: SSE register return with SSE disabled
	#warning INFO: Using ugly non-SSE workaround
	#undef __USE_EXTERN_INLINES
#endif /* __SSE__ */
#endif /* __x86_64__ */

#include <stdlib.h> // abort, malloc, free, qsort
#include <limits.h> // CHAR_BIT
// NOTE: -lm
#include <math.h> // fabs, fabsf, isnan, isinf
#include <stdio.h> // FILE, fopen, fprintf, fclose
#include <string.h> // memcpy
#include <stdarg.h> // va_start, va_end
#include <malloc.h> // memalign
#include <unistd.h> // sysconf, _SC_HOST_NAME_MAX, _SC_PAGESIZE, _SC_LEVEL1_DCACHE_SIZE, _SC_LEVEL1_DCACHE_ASSOC, _SC_LEVEL1_DCACHE_LINESIZE
#include <float.h> // FLT_EPSILON, DBL_EPSILON
#include <stddef.h> // ptrdiff_t
#include <ctype.h> // isspace

/** SSE intrinsics */
#ifdef __SSE__
	#pragma message "INFO: Using SSE"
	#include <xmmintrin.h>
#endif

/** OpenMP header when used */
#ifdef _OPENMP
	#pragma message "INFO: Using OpenMP"
	#include <omp.h>
#endif

#define ASM_MARKER __asm volatile ("# MARKER: " QUOTE(__LINE__))

#ifdef microblaze
/** total number of workers available */
const int dwt_util_global_total_workers = BCE_DMA_CFGTABLE_NUM_ITEMS;

static int get_total_workers()
{
	return dwt_util_global_total_workers;
}
#endif

/** how many workers use for computation (can be less than total number of workers) */
#ifdef microblaze
int dwt_util_global_active_workers = BCE_DMA_CFGTABLE_NUM_ITEMS;
#else
int dwt_util_global_active_workers = 1;
#endif

static
int get_active_workers()
{
	return dwt_util_global_active_workers;
}

static
void set_active_workers(int active_workers)
{
	dwt_util_global_active_workers = active_workers;
}

/** in bytes; offset in src[] and dst[] is given by worker_id * dwt_util_global_data_step */
ptrdiff_t dwt_util_global_data_step = 0;

static
ptrdiff_t get_data_step_s()
{
	return dwt_util_global_data_step;
}

static void set_data_step_s(ptrdiff_t data_step)
{
	dwt_util_global_data_step = data_step;
}

/** in elements; offset in temp[] is given by worker_id * dwt_util_global_temp_step */
int dwt_util_global_temp_step = 0;

static
int get_temp_step()
{
	return dwt_util_global_temp_step;
}

static
void set_temp_step(int temp_step)
{
	dwt_util_global_temp_step = temp_step;
}

/** active firmware in all ASVP acceleration units */
enum dwt_op dwt_util_global_active_op = DWT_OP_NONE;

// non-SSE workaround
#define DWT_CEIL_D(x) ((double)((int)((x)+1.0-DBL_EPSILON)))
#define DWT_CEIL_S(x) ((float)((int)((x)+1.0f-FLT_EPSILON)))

/** this PACKAGE_STRING macro must be defined via compiler's command line */
#ifndef PACKAGE_STRING
	#error PACKAGE_STRING is not defined
#endif

/** quoting macros */
#define QUOTE(x) STRING(x)

/** Calc offset in src[] or dst[] array for current worker. */
static
float *calc_data_offset_s(
	float *addr,	///< pointer to array assigned to worker 0
	int worker_id	///< identifier of current worker
       )
{
	return (float *)( (intptr_t)addr + (get_data_step_s() * worker_id) );
}

/** Calc offset in src[] or dst[] array for current worker. */
static
const float *calc_data_offset_const_s(
	const float *addr,	///< pointer to array assigned to worker 0
	int worker_id		///< identifier of current worker
       )
{
	return (const float *)( (intptr_t)addr + (get_data_step_s() * worker_id) );
}

#ifdef microblaze
static inline
void flush_cache(
	void *addr,	///< base address
	size_t size	///< length of memory in bytes
	)
{
	FUNC_BEGIN;

	// FIXME(ASVP): 4 or 8, should be detected
	const size_t dcache_line_len = 4;

	intptr_t tmp = size + (dcache_line_len * 4);
	do {
		__asm volatile (
			"wdc %0, %1;"
			:
			: "r" ((intptr_t)addr+tmp), "r" (0)
			: "memory"
		);
		tmp -= dcache_line_len * 4;
	} while( tmp >= 0 );

	FUNC_END;
}
#endif

#ifdef __x86_64__
static inline
void flush_cache(
	void *addr,	///< base address
	size_t size	///< length of memory in bytes
	)
{
	FUNC_BEGIN;

	const intptr_t begin = (intptr_t)addr;
	const intptr_t end = (intptr_t)addr + size;

	for(intptr_t p = begin; p < end; p++)
	{
		__asm volatile ("clflush (%0)" : : "r"((void *)p) : "memory");
	}

	__asm volatile ("mfence");

	FUNC_END;
}
#endif

#ifdef __arm__
static inline
void flush_cache(
	void *addr,	///< base address
	size_t size	///< length of memory in bytes
	)
{
	__builtin___clear_cache(addr, (char *)addr+size);
}
#endif

static inline
void flush_cache_s(
	float *addr,
	size_t size
	)
{
	flush_cache( (void *)addr, size * sizeof(float) );
}

#ifdef microblaze
void wal_abort(const char *str, int res)
{
#ifdef DEBUG_VERBOSE
	dwt_util_log(LOG_DBG, "%s = ", str);
#else
	UNUSED(str);
#endif
	switch(res)
	{
		case WAL_RES_OK:
		#ifdef DEBUG_VERBOSE
			printf("WAL_RES_OK (all is OK)\n");
		#endif
			return;
			break;
		case WAL_RES_WNULL:
			printf("WAL_RES_WNULL (argument is a NULL)\n");
			return;
			break;
		case WAL_RES_ERR:
			printf("WAL_RES_ERR (generic error)\n");
			break;
		case WAL_RES_ENOINIT:
			printf("WAL_RES_ENOINIT (not initiated)\n");
			break;
		case WAL_RES_ENULL:
			printf("WAL_RES_ENULL (null pointer)\n");
			break;
		case WAL_RES_ERUNNING:
			printf("WAL_RES_ERUNNING (worker is running)\n");
			break;
		case WAL_RES_ERANGE:
			printf("WAL_RES_ERANGE (index/value is out of range)\n");
			break;
		default:
			printf("(unknown error)\n");
	}

	dwt_util_abort();
}
#endif

const char *dwt_util_version()
{
	return QUOTE(PACKAGE_STRING);
}

const char *dwt_util_arch()
{
#ifdef microblaze
	// HACK: ugly buggy gcc workaround
	return "microblaze";
#endif

	return QUOTE(ARCH);
}

int dwt_util_global_accel_type = 0;

static
void set_accel_type(int accel_type)
{
	dwt_util_global_accel_type = accel_type;
}

static
int get_accel_type()
{
	return dwt_util_global_accel_type;
}

/**
 * @{
 * @brief CDF 9/7 lifting scheme constants
 * These constants are found in S. Mallat. A Wavelet Tour of Signal Processing: The Sparse Way (Third Edition). 3rd edition, 2009 on page 370.
 */
static const int    dwt_cdf97_k_s  =    2;
static const float  dwt_cdf97_p1_s =    1.58613434342059;
static const float  dwt_cdf97_u1_s =   -0.0529801185729;
static const float  dwt_cdf97_p2_s =   -0.8829110755309;
static const float  dwt_cdf97_u2_s =    0.4435068520439;
static const float  dwt_cdf97_s1_s =    1.1496043988602;
static const float  dwt_cdf97_s2_s =  1/1.1496043988602; // FIXME: unnecessary

static const int    dwt_cdf97_k_d  =    2;
static const double dwt_cdf97_p1_d =    1.58613434342059;
static const double dwt_cdf97_u1_d =   -0.0529801185729;
static const double dwt_cdf97_p2_d =   -0.8829110755309;
static const double dwt_cdf97_u2_d =    0.4435068520439;
static const double dwt_cdf97_s1_d =    1.1496043988602;
static const double dwt_cdf97_s2_d =  1/1.1496043988602; // FIXME: unnecessary
/**@}*/

/**
 * @{
 * @brief CDF 5/3 lifting scheme constants
 * These constants are found in S. Mallat. A Wavelet Tour of Signal Processing: The Sparse Way (Third Edition). 3rd edition, 2009 on page 369.
 */
static const int    dwt_cdf53_k_s  =    1;
static const float  dwt_cdf53_p1_s =    0.5;
static const float  dwt_cdf53_u1_s =    0.25;
static const float  dwt_cdf53_s1_s =    1.41421356237309504880;
static const float  dwt_cdf53_s2_s =    0.70710678118654752440; // FIXME: unnecessary

static const int    dwt_cdf53_k_d  =    1;
static const double dwt_cdf53_p1_d =    0.5;
static const double dwt_cdf53_u1_d =    0.25;
static const double dwt_cdf53_s1_d =    1.41421356237309504880;
static const double dwt_cdf53_s2_d =    0.70710678118654752440; // FIXME: unnecessary
/**@}*/

/**
 * @brief Power of two using greater or equal to x, i.e. 2^(ceil(log_2(x)).
 */
static
int pow2_ceil_log2(
	int x)
{
	assert( x > 0 );

	x--;

	unsigned shift = 1;

	while(shift < sizeof(int) * CHAR_BIT)
	{
		x |= x >> shift;
		shift <<= 1;
	}

	x++;

	return x;
}

/**
 * @brief Number of 1-bits in x, in parallel.
 */
static
int bits(
	unsigned x)
{
	x -= x >> 1 & (unsigned)~(unsigned)0/3;
	x = (x & (unsigned)~(unsigned)0/15*3) + (x >> 2 & (unsigned)~(unsigned)0/15*3);
	x = (x + (x >> 4)) & (unsigned)~(unsigned)0/255*15;
	return (x * ((unsigned)~(unsigned)0/255)) >> (sizeof(unsigned) - 1) * CHAR_BIT;
}

/**
 * @brief Smallest integer not less than the base 2 logarithm of x, i.e. ceil(log_2(x)).
 * @returns (int)ceil(log2(x))
 */
static
int ceil_log2(
	int x)
{
	return bits(pow2_ceil_log2(x) - 1);
}

int dwt_util_ceil_log2(
	int x)
{
	return ceil_log2(x);
}

int dwt_util_pow2_ceil_log2(
	int x)
{
	return pow2_ceil_log2(x);
}

/**
 * @returns (int)ceil(x/(double)y)
 */
static
int ceil_div(
	int x,
	int y)
{
	return (x + y - 1) / y;
}

int dwt_util_ceil_div(
	int x,
	int y)
{
	return ceil_div(x, y);
}

/**
 * @returns (int)floor(x/(double)y)
 */
static
int floor_div(
	int x,
	int y)
{
	return x / y;
}

int dwt_util_floor_div(
	int x,
	int y)
{
	return floor_div(x, y);
}

/**
 * @returns (int)ceil(i/(double)(1<<j))
 */
static
int ceil_div_pow2(
	int i,
	int j)
{
	return (i + (1 << j) - 1) >> j;
}

/** @returns (int)floor(x/(double)2) */
static
int floor_div2(
	int x)
{
	return x >> 1;
}

/** @returns (int)ceil(x/(double)2) */
static
int ceil_div2(
	int x)
{
	return (x + 1) >> 1;
}

/**
 * @brief Minimum of two integers.
 */
static
int min(
	int a,
	int b)
{
	return a > b ? b : a;
}

/**
 * @brief Maximum of two integers.
 */
static
int max(
	int a,
	int b)
{
	return a > b ? a : b;
}

/**
 * @brief returns 1 if x is odd, 0 otherwise; works also for negative numbers
 */
static
int is_odd(
	int x)
{
	return x & 1;
}

/**
 * @brief returns 1 if x is even, 0 otherwise; works also for negative numbers
 */
static
int is_even(
	int x)
{
	return 1 & ~x;
}

/**
 * @brief returns closest even integer not larger than x; works also for negative numbers
 */
static
int to_even(
	int x)
{
	return x & ~1;
}

int dwt_util_to_even(
	int x)
{
	return to_even(x);
}

/**
 * @brief returns closest odd integer not larger than x; works also for negative numbers
 */
static
int to_odd(
	int x)
{
	return x - (1 & ~x);
}

int dwt_util_to_odd(
	int x)
{
	return to_odd(x);
}

static
int is_aligned_4(
	const void *ptr)
{
	return ( (intptr_t)ptr&(intptr_t)(4-1) ) ? 0 : 1;
}

static
int is_aligned_8(
	const void *ptr)
{
	return ( (intptr_t)ptr&(intptr_t)(8-1) ) ? 0 : 1;
}

static
int is_aligned_16(
	const void *ptr)
{
	return ( (intptr_t)ptr&(intptr_t)(16-1) ) ? 0 : 1;
}

static
intptr_t align_4(
	intptr_t p)
{
	return (p+(4-1))&(~(4-1));
}

static
intptr_t align_8(
	intptr_t p)
{
	return (p+(8-1))&(~(8-1));
}

static
intptr_t align_16(
	intptr_t p)
{
	return (p+(16-1))&(~(16-1));
}

static
intptr_t align_64(
	intptr_t p)
{
	return (p+(64-1))&(~(64-1));
}

static
intptr_t align_4096(
	intptr_t p)
{
	return (p+(4096-1))&(~(4096-1));
}

intptr_t dwt_util_align_4(
	intptr_t p)
{
	return align_4(p);
}

intptr_t dwt_util_align_8(
	intptr_t p)
{
	return align_8(p);
}

intptr_t dwt_util_align_16(
	intptr_t p)
{
	return align_16(p);
}

/** Calc offset in temp[] array for current worker. Preserves alignment on 8 bytes. */
static
float *calc_temp_offset_s(
	float *addr,	///< pointer to array assigned to worker 0
	int worker_id	///< identifier of current worker
	)
{
#ifdef microblaze
	return addr + to_even( 2 + (get_temp_step()+2) * worker_id );
#else
	// FIXME: could be alignment to 16 bytes threated here instead of +1 in cdf97_2{f|i}_s function call?
	// FIXME: is this calculation correct regarding to alignment?
	// FIXME: to align_16 has no sence here due to units are sizeof(float) = 4
	return addr + align_16( get_temp_step() * worker_id );
#endif
}

static
int calc_and_set_temp_size(
	int temp_step)
{
#ifdef microblaze
	set_temp_step(temp_step);

	return 2 + to_even( 2 + (temp_step+2) * get_active_workers() );
#else
	set_temp_step(temp_step);
	// HACK: + 3*float => 16-align
	// FIXME: is this calculation correct regarding to alignment?
	// HACK: + 2 due to segfault
	return 3 + align_16(temp_step * get_active_workers()) + 2;
#endif
}

#if 0
/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (const double *).
 */
static
const double *addr1_const_d(
	const void *ptr,
	int i,
	int stride)
{
	return (const double *)((const char *)ptr+i*stride);
}
#endif

#if 0
/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (const double *).
 */
static
const float *addr1_const_s(
	const void *ptr,
	int i,
	int stride)
{
	return (const float *)((const char *)ptr+i*stride);
}
#endif

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (double *).
 */
static
double *addr1_d(
	void *ptr,
	int i,
	int stride)
{
	return (double *)((char *)ptr+i*stride);
}

static
int *addr1_i(
	void *ptr,
	int i,
	int stride)
{
	return (int *)((char *)ptr+i*stride);
}

/**
 * @brief Helper function returning address of given element.
 *
 * Evaluate address of (i) image element, returns (double *).
 */
static
float *addr1_s(
	void *ptr,
	int i,
	int stride)
{
	return (float *)((char *)ptr+i*stride);
}

static
void *addr2(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (void *)((char *)ptr+y*stride_x+x*stride_y);
}

static
const void *addr2_const(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (const void *)((const char *)ptr+y*stride_x+x*stride_y);
}

/**
 * @brief Helper function returning address of given pixel.
 *
 * Evaluate address of (x,y) image element, returns (double *).
 */
static
double *addr2_d(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (double *)((char *)ptr+y*stride_x+x*stride_y);
}

/**
 * @brief Helper function returning address of given pixel.
 *
 * Evaluate address of (x,y) image element, returns (int *).
 */
static
int *addr2_i(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (int *)((char *)ptr+y*stride_x+x*stride_y);
}

/**
 * @brief Helper function returning address of given pixel.
 *
 * Evaluate address of (x,y) image element, returns (float *).
 */
static
float *addr2_s(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (float *)((char *)ptr+y*stride_x+x*stride_y);
}

static
const int *addr2_const_i(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (const int *)((const char *)ptr+y*stride_x+x*stride_y);
}

static
const float *addr2_const_s(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (const float *)((const char *)ptr+y*stride_x+x*stride_y);
}

static
const double *addr2_const_d(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return (const double *)((const char *)ptr+y*stride_x+x*stride_y);
}

int *dwt_util_addr_coeff_i(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return addr2_i(ptr, y, x, stride_x, stride_y);
}

float *dwt_util_addr_coeff_s(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return addr2_s(ptr, y, x, stride_x, stride_y);
}

const float *dwt_util_addr_coeff_const_s(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return addr2_const_s(ptr, y, x, stride_x, stride_y);
}

double *dwt_util_addr_coeff_d(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return addr2_d(ptr, y, x, stride_x, stride_y);
}

void *dwt_util_addr_coeff(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y)
{
	return addr2(ptr, y, x, stride_x, stride_y);
}

/**
 * @brief Copy memory area.
 *
 * This function copies @p n ints from memory area @p src to memory area
 * @p dst. Memory areas can be sparse. The strides (in bytes) are determined by
 * @p stride_dst and @p stride_src arguments.
 *
 * @returns The function returns a pointer to @p dst.
 */
static
void *dwt_util_memcpy_stride_i(
	void *restrict dst,
	ssize_t stride_dst,
	const void *restrict src,
	ssize_t stride_src,
	size_t n		///< Number of ints to be copied, not number of bytes.
	)
{
	assert( NULL != dst && NULL != src );

	const size_t size = sizeof(int);

	if( (ssize_t)size == stride_src && (ssize_t)size == stride_dst )
	{
		memcpy(dst, src, n*size);
	}
	else
	{
		char *restrict ptr_dst = (char *restrict)dst;
		const char *restrict ptr_src = (const char *restrict)src;
		for(size_t i = 0; i < n; i++)
		{
			*(int *restrict)ptr_dst = *(const int *restrict)ptr_src;
	
			ptr_dst += stride_dst;
			ptr_src += stride_src;
		}
	}

	return dst;
}

/**
 * @brief Copy memory area.
 *
 * This function copies @p n floats from memory area @p src to memory area
 * @p dst. Memory areas can be sparse. The strides (in bytes) are determined by
 * @p stride_dst and @p stride_src arguments.
 *
 * @returns The function returns a pointer to @p dst.
 */
static
void *dwt_util_memcpy_stride_s(
	void *restrict dst,
	ssize_t stride_dst,
	const void *restrict src,
	ssize_t stride_src,
	size_t n		///< Number of floats to be copied, not number of bytes.
	)
{
	assert( NULL != dst && NULL != src );

	const size_t size = sizeof(float);

	if( (ssize_t)size == stride_src && (ssize_t)size == stride_dst )
	{
		memcpy(dst, src, n*size);
	}
	else
	{
		char *restrict ptr_dst = (char *restrict)dst;
		const char *restrict ptr_src = (const char *restrict)src;
		for(size_t i = 0; i < n; i++)
		{
			*(float *restrict)ptr_dst = *(const float *restrict)ptr_src;
	
			ptr_dst += stride_dst;
			ptr_src += stride_src;
		}
	}

	return dst;
}

/**
 * @brief Copy memory area.
 *
 * This function copies @p n doubles from memory area @p src to memory area
 * @p dst. Memory areas can be sparse. The strides (in bytes) are determined by
 * @p stride_dst and @p stride_src arguments.
 *
 * @returns The function returns a pointer to @p dst.
 */
static
void *dwt_util_memcpy_stride_d(
	void *restrict dst,
	ssize_t stride_dst,
	const void *restrict src,
	ssize_t stride_src,
	size_t n		///< Number of doubles to be copied, not number of bytes.
	)
{
	assert( NULL != dst && NULL != src );

	const size_t size = sizeof(double);

	if( (ssize_t)size == stride_src && (ssize_t)size == stride_dst )
	{
		memcpy(dst, src, n*size);
	}
	else
	{
		char *restrict ptr_dst = (char *restrict)dst;
		const char *restrict ptr_src = (const char *restrict)src;
		for(size_t i = 0; i < n; i++)
		{
			*(double *restrict)ptr_dst = *(const double *restrict)ptr_src;
	
			ptr_dst += stride_dst;
			ptr_src += stride_src;
		}
	}

	return dst;
}

/**
 * @brief Pixel value of test image.
 */
static
void dwt_util_test_image_value_i_d(
	double *dest,
	int x,
	int y,
	int rand,
	int type)
{
	switch(type)
	{
		case 0:
			x >>= rand;
			*dest = 2*x*y / (double)(x*x + y*y + 1);
			break;
#ifdef __SSE__
		case 1:
			x >>= rand;
			*dest = 2*x*y / (double)(x*x + y*y + 1) * fabs(sin(x/10.)) * fabs(cos(y*x/5.));
			break;
#endif
		default:
		{
			dwt_util_log(LOG_ERR, "Unknown test image type.\n");

			dwt_util_abort();
		}
	}
}

static
void dwt_util_test_image_value_i_i(
	int *dest,
	int x,
	int y,
	int rand,
	int type)
{
	switch(type)
	{
		case 0:
			x >>= rand;
			*dest = 255 * (2*x*y) / (x*x + y*y + 1);
			break;
		case 2:
			*dest = x^y;
			*dest &= 0xff;
			break;
		default:
		{
			dwt_util_log(LOG_ERR, "Unknown test image type.\n");

			dwt_util_abort();
		}
	}
}

/**
 * @brief Pixel value of test image.
 */
static
void dwt_util_test_image_value_i_s(
	float *dest,
	int x,
	int y,
	int rand,
	int type)
{
	switch(type)
	{
		case 0:
			x >>= rand;
			*dest = 2*x*y / (float)(x*x + y*y + 1);
			break;
#ifdef __SSE__
		case 1:
			x >>= rand;
			*dest = 2*x*y / (float)(x*x + y*y + 1) * fabsf(sinf(x/10.f)) * fabsf(cosf(y*x/5.f));
			break;
#endif
		default:
		{
			dwt_util_log(LOG_ERR, "Unknown test image type.\n");

			dwt_util_abort();
		}
	}
}

// TODO: propagate type of test image
void dwt_util_test_image_fill_d(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y,
	int rand)
{
	assert( NULL != ptr );

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
			dwt_util_test_image_value_i_d(
				addr2_d(ptr, y, x, stride_x, stride_y),
				x,
				y,
				rand,
				0
			);
}

// TODO: propagate type of test image
void dwt_util_test_image_fill_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y,
	int rand)
{
	assert( NULL != ptr );

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
			dwt_util_test_image_value_i_i(
				addr2_i(ptr, y, x, stride_x, stride_y),
				x,
				y,
				rand,
				0
			);
}

// TODO: propagate type of test image
void dwt_util_test_image_fill_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y,
	int rand)
{
	FUNC_BEGIN;

	assert( NULL != ptr );

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
			dwt_util_test_image_value_i_s(
				addr2_s(ptr, y, x, stride_x, stride_y),
				x,
				y,
				rand,
				0
			);

	FUNC_END;
}

void dwt_util_test_image_zero_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( NULL != ptr );

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
			*addr2_s(ptr, y, x, stride_x, stride_y) = 0.0f;
}

static
size_t image_size(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y)
{
	UNUSED(stride_y);
	UNUSED(size_o_big_x);

	return stride_x * size_o_big_y;
}

size_t dwt_util_image_size(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y)
{
	return image_size(
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);
}

void dwt_util_alloc_image(
	void **pptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y)
{
	FUNC_BEGIN;

	assert( NULL != pptr );

	UNUSED(stride_y);
	UNUSED(size_o_big_x);

	*pptr = (void *)memalign(16, stride_x*size_o_big_y);
	if(NULL == *pptr)
	{
		dwt_util_log(LOG_ERR, "Unable to allocate memory.\n");
		dwt_util_abort();
	}

	FUNC_END;
}

void dwt_util_free_image(
	void **pptr)
{
	assert( pptr != NULL );

	free(*pptr);
	*pptr = NULL;
}

int dwt_util_compare_d(
	void *ptr1,
	void *ptr2,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( ptr1 != NULL && ptr2 != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	const double eps = 1e-6;

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
		{
			const double a = *addr2_d(ptr1, y, x, stride_x, stride_y);
			const double b = *addr2_d(ptr2, y, x, stride_x, stride_y);

			if( isnan(a) || isinf(a) || isnan(b) || isinf(b) )
				return 1;

			if( fabs(a - b) > eps )
				return 1;
		}

	return 0;
}

int dwt_util_compare_i(
	void *ptr1,
	void *ptr2,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( ptr1 != NULL && ptr2 != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
		{
			const int a = *addr2_i(ptr1, y, x, stride_x, stride_y);
			const int b = *addr2_i(ptr2, y, x, stride_x, stride_y);

			if( abs(a - b) > 0 )
			{
#ifdef DEBUG
				dwt_util_log(LOG_DBG, "%s: %i != %i at (x=%i, y=%i)\n", __FUNCTION__, a, b, x, y);
#endif
				return 1;
			}
		}

	return 0;
}

int dwt_util_compare_s(
	void *ptr1,
	void *ptr2,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( ptr1 != NULL && ptr2 != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	const float eps = 1e-3;

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
		{
			const float a = *addr2_s(ptr1, y, x, stride_x, stride_y);
			const float b = *addr2_s(ptr2, y, x, stride_x, stride_y);

			if( isnan(a) || isinf(a) || isnan(b) || isinf(b) )
				return 1;

			if( fabsf(a - b) > eps )
				return 1;
		}

	return 0;
}

void dwt_cdf97_f_d(
	const double *src,
	double *dst,
	double *tmp,
	int N)
{
	dwt_cdf97_f_ex_d(
		src,
		dst,
		dst + ceil_div2(N),
		tmp,
		N
	);
}

void dwt_cdf53_f_d(
	const double *src,
	double *dst,
	double *tmp,
	int N)
{
	dwt_cdf53_f_ex_d(
		src,
		dst,
		dst + ceil_div2(N),
		tmp,
		N
	);
}

void dwt_cdf97_f_s(
	const float *src,
	float *dst,
	float *tmp,
	int N)
{
	dwt_cdf97_f_ex_s(
		src,
		dst,
		dst + ceil_div2(N),
		tmp,
		N
	);
}

void dwt_cdf53_f_s(
	const float *src,
	float *dst,
	float *tmp,
	int N)
{
	dwt_cdf53_f_ex_s(
		src,
		dst,
		dst + ceil_div2(N),
		tmp,
		N
	);
}

void dwt_cdf97_i_d(
	const double *src,
	double *dst,
	double *tmp,
	int N)
{
	dwt_cdf97_i_ex_d(
		src,
		src + ceil_div2(N),
		dst,
		tmp,
		N
	);
}

void dwt_cdf53_i_d(
	const double *src,
	double *dst,
	double *tmp,
	int N)
{
	dwt_cdf53_i_ex_d(
		src,
		src + ceil_div2(N),
		dst,
		tmp,
		N
	);
}

void dwt_cdf97_i_s(
	const float *src,
	float *dst,
	float *tmp,
	int N)
{
	dwt_cdf97_i_ex_s(
		src,
		src + ceil_div2(N),
		dst,
		tmp,
		N
	);
}

void dwt_cdf53_i_s(
	const float *src,
	float *dst,
	float *tmp,
	int N)
{
	dwt_cdf53_i_ex_s(
		src,
		src + ceil_div2(N),
		dst,
		tmp,
		N
	);
}

void dwt_cdf97_f_ex_d(
	const double *src,
	double *dst_l,
	double *dst_h,
	double *tmp,
	int N)
{
	dwt_cdf97_f_ex_stride_d(
		src,
		dst_l,
		dst_h,
		tmp,
		N,
		sizeof(double)
	);
}

void dwt_cdf53_f_ex_d(
	const double *src,
	double *dst_l,
	double *dst_h,
	double *tmp,
	int N)
{
	dwt_cdf53_f_ex_stride_d(
		src,
		dst_l,
		dst_h,
		tmp,
		N,
		sizeof(double)
	);
}

void dwt_cdf97_f_ex_s(
	const float *src,
	float *dst_l,
	float *dst_h,
	float *tmp,
	int N)
{
	dwt_cdf97_f_ex_stride_s(
		src,
		dst_l,
		dst_h,
		tmp,
		N,
		sizeof(float)
	);
}

void dwt_cdf53_f_ex_i(
	const int *src,
	int *dst_l,
	int *dst_h,
	int *tmp,
	int N)
{
	dwt_cdf53_f_ex_stride_i(
		src,
		dst_l,
		dst_h,
		tmp,
		N,
		sizeof(int)
	);
}

void dwt_cdf53_f_ex_s(
	const float *src,
	float *dst_l,
	float *dst_h,
	float *tmp,
	int N)
{
	dwt_cdf53_f_ex_stride_s(
		src,
		dst_l,
		dst_h,
		tmp,
		N,
		sizeof(float)
	);
}

void dwt_cdf97_f_ex_stride_d(
	const double *src,
	double *dst_l,
	double *dst_h,
	double *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src && NULL != dst_l && NULL != dst_h && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst_l[0] = src[0] * dwt_cdf97_s1_d;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_d(tmp, sizeof(double), src, stride, N);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] -= dwt_cdf97_p1_d * (tmp[i-1] + tmp[i+1]);

	if(is_odd(N))
		tmp[N-1] += 2 * dwt_cdf97_u1_d * tmp[N-2];
	else
		tmp[N-1] -= 2 * dwt_cdf97_p1_d * tmp[N-2];

	tmp[0] += 2 * dwt_cdf97_u1_d * tmp[1];

	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] += dwt_cdf97_u1_d * (tmp[i-1] + tmp[i+1]);

	// predict 2 + update 2
	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] -= dwt_cdf97_p2_d * (tmp[i-1] + tmp[i+1]);

	if(is_odd(N))
		tmp[N-1] += 2 * dwt_cdf97_u2_d * tmp[N-2];
	else
		tmp[N-1] -= 2 * dwt_cdf97_p2_d * tmp[N-2];

	tmp[0] += 2 * dwt_cdf97_u2_d * tmp[1];

	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] += dwt_cdf97_u2_d * (tmp[i-1] + tmp[i+1]);

	// scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s1_d;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s2_d;

	// copy tmp into dst
	dwt_util_memcpy_stride_d(dst_l, stride, tmp+0, 2*sizeof(double),  ceil_div2(N));
	dwt_util_memcpy_stride_d(dst_h, stride, tmp+1, 2*sizeof(double), floor_div2(N));
}

void dwt_cdf53_f_ex_stride_d(
	const double *src,
	double *dst_l,
	double *dst_h,
	double *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src && NULL != dst_l && NULL != dst_h && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst_l[0] = src[0] * dwt_cdf53_s1_d;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_d(tmp, sizeof(double), src, stride, N);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] -= dwt_cdf53_p1_d * (tmp[i-1] + tmp[i+1]);

	if(is_odd(N))
		tmp[N-1] += 2 * dwt_cdf53_u1_d * tmp[N-2];
	else
		tmp[N-1] -= 2 * dwt_cdf53_p1_d * tmp[N-2];

	tmp[0] += 2 * dwt_cdf53_u1_d * tmp[1];

	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] += dwt_cdf53_u1_d * (tmp[i-1] + tmp[i+1]);

	// scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s1_d;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s2_d;

	// copy tmp into dst
	dwt_util_memcpy_stride_d(dst_l, stride, tmp+0, 2*sizeof(double),  ceil_div2(N));
	dwt_util_memcpy_stride_d(dst_h, stride, tmp+1, 2*sizeof(double), floor_div2(N));
}

#ifdef __x86_64__
static
void accel_lift_op4s_main_nosse_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
// this long string disables SSE support (only no-sse is not enough)
__attribute__ ((__target__ ("no-mmx,no-sse,no-sse2,no-sse3,no-sse4,no-sse4.1")));
#endif
static
void accel_lift_op4s_main_nosse_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	if( scaling < 0 )
	{
		// inv
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// descale
			float *out = arr_local+4;

			for(int s = 0; s < steps; s++)
			{
				out[0] *= v[0];
				out[1] *= v[1];

				out += 2;
			}

			// operations
			for(int off = 4; off >= 1; off--)
			{
				float *out = arr_local+off;
				const float c = w[off-1];

				for(int s = 0; s < steps; s++)
				{
					out[0] += c * (out[-1] + out[+1]);

					out += 2;
				}
			}
		}
	}
	else if( scaling > 0 )
	{
		// fwd
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// operations
			for(int off = 4; off >= 1; off--)
			{
				float *out = arr_local+off;
				const float c = w[off-1];

				for(int s = 0; s < steps; s++)
				{
					out[0] += c * (out[-1] + out[+1]);

					out += 2;
				}
			}

			// scale
			float *out = arr_local+0;

			for(int s = 0; s < steps; s++)
			{
				out[0] *= v[0];
				out[1] *= v[1];

				out += 2;
			}
		}
	}
	else
	{
		// uni
		dwt_util_abort();
	}
}

#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 7
	#pragma message "INFO: Running on GCC 4.7+"
	#define ASSUME_ALIGNED(lvalueptr, align) __builtin_assume_aligned((lvalueptr), (align))
#else
	#pragma message "INFO: Missing GCC 4.7+"
	#define ASSUME_ALIGNED(lvalueptr, align) (lvalueptr)
#endif

/**
 * @brief Non-accelerated PicoBlaze operation.
 *
 * Two pairs (predict and update) of lifting steps and coefficients scaling
 * merged together.
 *
 * @param[in] scaling Perform scaling of coefficients. Possible values are:
 *   @li s = 0 : without scaling,
 *   @li s > 0 : scaling after lifting,
 *   @li s < 0 : scaling before lifting.
 */
static
void accel_lift_op4s_main_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	FUNC_BEGIN;

	assert( steps >= 0 );

	// FIXME: ASSUME_ALIGNED(..., 16) implies is_aligned_16(...)
	assert( is_aligned_16(calc_temp_offset_s(arr, 0)) );

	if( scaling < 0 )
	{
		// inv
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = ASSUME_ALIGNED(calc_temp_offset_s(arr, w), 16);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// descale
			float *out = arr_local+4;

			for(int s = 0; s < steps; s++)
			{
				out[0] *= v[0];
				out[1] *= v[1];

				out += 2;
			}

			// operations
			for(int off = 4; off >= 1; off--)
			{
				float *out = arr_local+off;
				const float c = w[off-1];

				for(int s = 0; s < steps; s++)
				{
					out[0] += c * (out[-1] + out[+1]);

					out += 2;
				}
			}
		}
	}
	else if( scaling > 0 )
	{
		// fwd
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = ASSUME_ALIGNED(calc_temp_offset_s(arr, w), 16);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// operations
			for(int off = 4; off >= 1; off--)
			{
				float *out = arr_local+off;
				const float c = w[off-1];

				for(int s = 0; s < steps; s++)
				{
					out[0] += c * (out[-1] + out[+1]);

					out += 2;
				}
			}

			// scale
			float *out = arr_local+0;

			for(int s = 0; s < steps; s++)
			{
				out[0] *= v[0];
				out[1] *= v[1];

				out += 2;
			}
		}
	}
	else
	{
		// uni
		dwt_util_abort();
	}

	FUNC_END;
}

#ifdef __SSE__
/**
 * multi-loop algorithm with 4 workers
 */
static
void accel_lift_op4s_main_ml4_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	if( scaling < 0 )
	{
		// inv
		assert( 4 == dwt_util_get_num_workers() );

		// constants
		__m128 w[4] = {
			{ delta, delta, delta, delta },
			{ gamma, gamma, gamma, gamma },
			{ beta,  beta,  beta,  beta  },
			{ alpha, alpha, alpha, alpha }
		};
		__m128 v[2] = {
			{ 1/zeta, 1/zeta, 1/zeta, 1/zeta },
			{   zeta,   zeta,   zeta,   zeta }
		};

		float *arr_local[4];

		// pointers
		for(int worker = 0; worker < 4; worker++)
		{
			arr_local[worker] = calc_temp_offset_s(arr, worker);
		}

		// buffer
		const int buff_size = 4 + 2*steps;
		__m128 buff[buff_size]; // FIXME(x86): huge array on the stack
		__m128 *out;

		// load buffer
		assert( is_aligned_16(arr_local[0]) );

		const int t4 = buff_size >> 2;
		const int t3 = buff_size & ~3;

		for(int t = 0; t < t4; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			__m128 s0 = _mm_load_ps(&arr_local[0][4*t]);
			__m128 s1 = _mm_load_ps(&arr_local[1][4*t]);
			__m128 s2 = _mm_load_ps(&arr_local[2][4*t]);
			__m128 s3 = _mm_load_ps(&arr_local[3][4*t]);

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			buff[4*t+0] = s0;
			buff[4*t+1] = s1;
			buff[4*t+2] = s2;
			buff[4*t+3] = s3;
		}

		for(int t = t3; t < buff_size; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			buff[t][0] = arr_local[0][t];
			buff[t][1] = arr_local[1][t];
			buff[t][2] = arr_local[2][t];
			buff[t][3] = arr_local[3][t];
		}

		out = buff + 4;

		// descale
		for(int s = 0; s < steps; s++)
		{
			out[0] *= v[0];
			out[1] *= v[1];

			out += 2;
		}

		for(int off = 4; off >= 1; off--)
		{
			const __m128 coeff = w[off-1];
			out = buff + off;

			// operation
			for(int s = 0; s < steps; s++)
			{
				out[0] += coeff * (out[-1] + out[+1]);

				out += 2;
			}
		}

		// store buffer
		for(int t = 0; t < t4; t++)
		{
			__m128 s0 = buff[4*t+0];
			__m128 s1 = buff[4*t+1];
			__m128 s2 = buff[4*t+2];
			__m128 s3 = buff[4*t+3];

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			_mm_store_ps(&arr_local[0][4*t], s0);
			_mm_store_ps(&arr_local[1][4*t], s1);
			_mm_store_ps(&arr_local[2][4*t], s2);
			_mm_store_ps(&arr_local[3][4*t], s3);
		}

		for(int t = t3; t < buff_size; t++)
		{
			arr_local[0][t] = buff[t][0];
			arr_local[1][t] = buff[t][1];
			arr_local[2][t] = buff[t][2];
			arr_local[3][t] = buff[t][3];
		}
	}
	else if( scaling > 0 )
	{
		// fwd
		assert( 4 == dwt_util_get_num_workers() );

		// constants
		__m128 w[4] = {
			{ delta, delta, delta, delta },
			{ gamma, gamma, gamma, gamma },
			{ beta,  beta,  beta,  beta  },
			{ alpha, alpha, alpha, alpha }
		};
		__m128 v[2] = {
			{ 1/zeta, 1/zeta, 1/zeta, 1/zeta },
			{   zeta,   zeta,   zeta,   zeta }
		};

		float *arr_local[4];

		// pointers
		for(int worker = 0; worker < 4; worker++)
		{
			arr_local[worker] = calc_temp_offset_s(arr, worker);
		}

		// buffer
		const int buff_size = 4 + 2*steps;
		__m128 buff[buff_size]; // FIXME(x86): huge array on the stack
		__m128 *out;

		// load buffer
		assert( is_aligned_16(arr_local[0]) );

		const int t4 = buff_size >> 2;
		const int t3 = buff_size & ~3;

		for(int t = 0; t < t4; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			__m128 s0 = _mm_load_ps(&arr_local[0][4*t]);
			__m128 s1 = _mm_load_ps(&arr_local[1][4*t]);
			__m128 s2 = _mm_load_ps(&arr_local[2][4*t]);
			__m128 s3 = _mm_load_ps(&arr_local[3][4*t]);

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			buff[4*t+0] = s0;
			buff[4*t+1] = s1;
			buff[4*t+2] = s2;
			buff[4*t+3] = s3;
		}

		for(int t = t3; t < buff_size; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			buff[t][0] = arr_local[0][t];
			buff[t][1] = arr_local[1][t];
			buff[t][2] = arr_local[2][t];
			buff[t][3] = arr_local[3][t];
		}

		for(int off = 4; off >= 1; off--)
		{
			const __m128 coeff = w[off-1];
			out = buff + off;

			// operation
			for(int s = 0; s < steps; s++)
			{
				out[0] += coeff * (out[-1] + out[+1]);

				out += 2;
			}
		}

		out = buff + 0;

		// descale
		for(int s = 0; s < steps; s++)
		{
			out[0] *= v[0];
			out[1] *= v[1];

			out += 2;
		}

		// store buffer
		for(int t = 0; t < t4; t++)
		{
			__m128 s0 = buff[4*t+0];
			__m128 s1 = buff[4*t+1];
			__m128 s2 = buff[4*t+2];
			__m128 s3 = buff[4*t+3];

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			_mm_store_ps(&arr_local[0][4*t], s0);
			_mm_store_ps(&arr_local[1][4*t], s1);
			_mm_store_ps(&arr_local[2][4*t], s2);
			_mm_store_ps(&arr_local[3][4*t], s3);
		}

		for(int t = t3; t < buff_size; t++)
		{
			arr_local[0][t] = buff[t][0];
			arr_local[1][t] = buff[t][1];
			arr_local[2][t] = buff[t][2];
			arr_local[3][t] = buff[t][3];
		}
	}
	else
	{
		// uni
		dwt_util_abort();
	}
}
#endif

static
void op4s_sdl2_import_preload_s_ref(float *out, const float *restrict addr)
{
	out[0] = addr[0];
	out[1] = addr[1];
	out[2] = addr[2];
	out[3] = addr[3];
}

#ifdef __SSE__
#define op4s_sdl2_import_preload_s_sse(out, addr) \
do { \
	(out) = _mm_load_ps(addr); \
} while(0)
#endif

static
void op4s_sdl2_import_s_ref(float *l, int idx, const float *out)
{
	l[idx] = out[idx];
}

#ifdef __SSE__
#define op4s_sdl2_import_s_sse(l, idx, out) \
do { \
	(out) = _mm_shuffle_ps((out), (out), _MM_SHUFFLE(2,1,0,3)); \
	(l) = _mm_move_ss((l), (out)); \
	(l) = _mm_shuffle_ps((l), (l), _MM_SHUFFLE((3==idx)?0:3,(2==idx)?0:2,(1==idx)?0:1,(0==idx)?0:0)); \
} while(0)
#endif

static
void op4s_sdl6_import_s_ref(float *l, int idx, const float *out)
{
	l[idx] = out[idx];
}

#ifdef __SSE__
#define op4s_sdl6_import_s_sse(l, idx, out) \
do { \
	(out) = _mm_shuffle_ps((out), (out), _MM_SHUFFLE(2,1,0,3)); \
	(l) = _mm_move_ss((l), (out)); \
	(l) = _mm_shuffle_ps((l), (l), _MM_SHUFFLE((3==idx)?0:3,(2==idx)?0:2,(1==idx)?0:1,(0==idx)?0:0)); \
} while(0)
#endif

static
void op4s_sdl2_load_s_ref(float *in, const float *addr)
{
	in[0] = addr[0];
	in[1] = addr[1];
	in[2] = addr[2];
	in[3] = addr[3];
}

#ifdef __SSE__
#define op4s_sdl2_load_s_sse(in, addr) \
do { \
	(in) = _mm_load_ps((const float *)(addr)); \
} while(0)
#endif

static
void op4s_sdl2_shuffle_s_ref(float *c, float *r)
{
	c[0]=c[1]; c[1]=c[2]; c[2]=c[3];
	r[0]=r[1]; r[1]=r[2]; r[2]=r[3];
}

#ifdef __SSE__
#define op4s_sdl2_shuffle_s_sse(c, r) \
do { \
	(c) = _mm_shuffle_ps((c), (c), _MM_SHUFFLE(0,3,2,1)); \
	(r) = _mm_shuffle_ps((r), (r), _MM_SHUFFLE(0,3,2,1)); \
} while(0)
#endif

static
void op4s_sdl2_input_low_s_ref(const float *in, float *c, float *r)
{
	c[3] = in[0];
	r[3] = in[1];
}

#ifdef __SSE__
#define op4s_sdl2_input_low_s_sse(in, c, r) \
do { \
	__m128 t; \
	(t) = (c); \
	(t) = _mm_shuffle_ps((t), (in), _MM_SHUFFLE(1,0,3,2)); \
	(c) = _mm_shuffle_ps((c), (t),  _MM_SHUFFLE(2,0,1,0)); \
	(t) = _mm_shuffle_ps((t), (r),  _MM_SHUFFLE(3,2,3,2)); \
	(r) = _mm_shuffle_ps((r), (t),  _MM_SHUFFLE(1,2,1,0)); \
} while(0)
#endif

static
void op4s_sdl2_input_high_s_ref(const float *in, float *c, float *r)
{
	c[3] = in[2];
	r[3] = in[3];
}

static
void op4s_sdl2_shuffle_input_low_s_ref(const float *in, float *c, float *r)
{
	op4s_sdl2_shuffle_s_ref(c, r);
	op4s_sdl2_input_low_s_ref(in, c, r);
}

#ifdef __SSE__
#define op4s_sdl2_shuffle_input_low_s_sse(in, c, r) \
do { \
	__m128 t; \
	(t) = (in); \
	(t) = _mm_shuffle_ps((t), (c), _MM_SHUFFLE(3,2,1,0)); \
	(c) = _mm_shuffle_ps((c), (t), _MM_SHUFFLE(0,3,2,1)); \
	(t) = _mm_shuffle_ps((t), (r), _MM_SHUFFLE(3,2,1,0)); \
	(r) = _mm_shuffle_ps((r), (t), _MM_SHUFFLE(1,3,2,1)); \
} while(0)
#endif

static
void op4s_sdl2_shuffle_input_high_s_ref(const float *in, float *c, float *r)
{
	op4s_sdl2_shuffle_s_ref(c, r);
	op4s_sdl2_input_high_s_ref(in, c, r);
}

#ifdef __SSE__
#define op4s_sdl2_shuffle_input_high_s_sse(in, c, r) \
do { \
	(in) = _mm_shuffle_ps( (in), (c), _MM_SHUFFLE(3,2,3,2) ); \
	(c)  = _mm_shuffle_ps( (c), (in), _MM_SHUFFLE(0,3,2,1) ); \
	(in) = _mm_shuffle_ps( (in), (r), _MM_SHUFFLE(3,2,1,0) ); \
	(r)  = _mm_shuffle_ps( (r), (in), _MM_SHUFFLE(1,3,2,1) ); \
} while(0)
#endif

static
void op4s_sdl2_op_s_ref(float *z, const float *c, const float *w, const float *l, const float *r)
{
	z[3] = c[3] + w[3] * ( l[3] + r[3] );
	z[2] = c[2] + w[2] * ( l[2] + r[2] );
	z[1] = c[1] + w[1] * ( l[1] + r[1] );
	z[0] = c[0] + w[0] * ( l[0] + r[0] );
}

#ifdef __SSE__
#define op4s_sdl2_op_s_sse(z, c, w, l, r) \
do { \
	(z) = (l); \
	(z) = _mm_add_ps((z), (r)); \
	(z) = _mm_mul_ps((z), (w)); \
	(z) = _mm_add_ps((z), (c)); \
} while(0)
#endif

static
void op4s_sdl6_op_s_ref(float *z, const float *w, const float *l, const float *r)
{
	z[3] = z[3] + w[3] * ( l[3] + r[3] );
	z[2] = z[2] + w[2] * ( l[2] + r[2] );
	z[1] = z[1] + w[1] * ( l[1] + r[1] );
	z[0] = z[0] + w[0] * ( l[0] + r[0] );
}

#ifdef __SSE__
#define op4s_sdl6_op_s_sse(z, w, l, r) \
do { \
	__m128 t; \
	(t) = (l); \
	(t) = _mm_add_ps((t), (r)); \
	(t) = _mm_mul_ps((t), (w)); \
	(z) = _mm_add_ps((z), (t)); \
} while(0)
#endif

static
void op4s_sdl2_update_s_ref(float *c, float *l, float *r, const float *z)
{
	c[0] = l[0];
	c[1] = l[1];
	c[2] = l[2];
	c[3] = l[3];

	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	r[0] = z[0];
	r[1] = z[1];
	r[2] = z[2];
	r[3] = z[3];
}

#ifdef __SSE__
#define op4s_sdl2_update_s_sse(c, l, r, z) \
do { \
	(c) = (l); \
	(l) = (r); \
	(r) = (z); \
} while(0)
#endif

static
void op4s_sdl6_update_s_ref(float *z, float *l, float *r)
{
	float t[4];

	t[0] = z[0];
	t[1] = z[1];
	t[2] = z[2];
	t[3] = z[3];

	z[0] = l[0];
	z[1] = l[1];
	z[2] = l[2];
	z[3] = l[3];

	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	r[0] = t[0];
	r[1] = t[1];
	r[2] = t[2];
	r[3] = t[3];
}

#ifdef __SSE__
#define op4s_sdl6_update_s_sse(z, l, r) \
do { \
	__m128 t; \
	(t) = (z); \
	(z) = (l); \
	(l) = (r); \
	(r) = (t); \
} while(0)
#endif

static
void op4s_sdl2_output_low_s_ref(float *out, const float *l, const float *z)
{
	out[0] = l[0];
	out[1] = z[0];
}

#ifdef __SSE__
#define op4s_sdl2_output_low_s_sse(out, l, z) \
do { \
	(out) = (l); \
	(out) = _mm_unpacklo_ps((out), (z)); \
} while(0)
#endif

static
void op4s_sdl2_output_high_s_ref(float *out, const float *l, const float *z)
{
	out[2] = l[0];
	out[3] = z[0];
}

#ifdef __SSE__
#define op4s_sdl2_output_high_s_sse(out, l, z) \
do { \
	__m128 t; \
	(t) = (l); \
	(t) = _mm_unpacklo_ps((t), (z)); \
	(out) = _mm_shuffle_ps((out), t, _MM_SHUFFLE(1,0,1,0)); \
} while(0)
#endif

static
void op4s_sdl2_scale_s_ref(float *out, const float *v)
{
	out[0] *= v[0];
	out[1] *= v[1];
	out[2] *= v[2];
	out[3] *= v[3];
}

#ifdef __SSE__
#define op4s_sdl2_scale_s_sse(out, v) \
do { \
	(out) = _mm_mul_ps((out), (v)); \
} while(0)
#endif

static
void op4s_sdl2_descale_s_ref(float *in, const float *v)
{
	in[0] *= v[0];
	in[1] *= v[1];
	in[2] *= v[2];
	in[3] *= v[3];
}

#ifdef __SSE__
#define op4s_sdl2_descale_s_sse(in, v) \
do { \
	(in) = _mm_mul_ps((in), (v)); \
} while(0)
#endif

static
void op4s_sdl2_save_s_ref(float *out, float *addr)
{
	addr[0] = out[0];
	addr[1] = out[1];
}

#ifdef __SSE__
#define op4s_sdl2_save_s_sse(out, addr) \
do { \
	_mm_storel_pi((__m64 *)(addr), (out)); \
} while(0)
#endif

static
void op4s_sdl2_save_shift_s_ref(float *out, float *addr)
{
	addr[0] = out[0];
	addr[1] = out[1];
	addr[2] = out[2];
	addr[3] = out[3];
}

#ifdef __SSE__
#define op4s_sdl2_save_shift_s_sse(out, addr) \
do { \
	_mm_store_ps((float *)(addr), (out)); \
} while(0)
#endif

static
void op4s_sdl2_export_s_ref(const float *l, float *addr, int idx)
{
	addr[idx] = l[idx];
}

#ifdef __SSE__
#define op4s_sdl2_export_s_sse(l, addr, idx) \
do { \
	(addr)[(idx)] = (l)[(idx)]; \
} while(0)
#endif

static
void op4s_sdl6_export_s_ref(const float *l, float *addr, int idx)
{
	addr[idx] = l[idx];
}

#ifdef __SSE__
#define op4s_sdl6_export_s_sse(l, addr, idx) \
do { \
	(addr)[(idx)] = (l)[(idx)]; \
} while(0)
#endif

static
void op4s_sdl_import_s_ref(float *l, const float *restrict addr, int idx)
{
	l[idx] = addr[idx];
}

static
void op4s_sdl_shuffle_s_ref(float *c, float *r)
{
	c[0]=c[1]; c[1]=c[2]; c[2]=c[3];
	r[0]=r[1]; r[1]=r[2]; r[2]=r[3];
}

static
void op4s_sdl_load_s_ref(float *in, const float *restrict addr)
{
	in[0] = addr[0];
	in[1] = addr[1];
}

static
void op4s_sdl_input_s_ref(const float *in, float *c, float *r)
{
	c[3] = in[0];
	r[3] = in[1];
}

static
void op4s_sdl_op_s_ref(float *z, const float *c, const float *w, const float *l, const float *r)
{
	z[3] = c[3] + w[3] * ( l[3] + r[3] );
	z[2] = c[2] + w[2] * ( l[2] + r[2] );
	z[1] = c[1] + w[1] * ( l[1] + r[1] );
	z[0] = c[0] + w[0] * ( l[0] + r[0] );
}

static
void op4s_sdl_update_s_ref(float *c, float *l, float *r, const float *z)
{
	c[0] = l[0];
	c[1] = l[1];
	c[2] = l[2];
	c[3] = l[3];

	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];

	r[0] = z[0];
	r[1] = z[1];
	r[2] = z[2];
	r[3] = z[3];
}

static
void op4s_sdl_output_s_ref(float *out, const float *l, const float *z)
{
	out[0] = l[0];
	out[1] = z[0];
}

static
void op4s_sdl_scale_s_ref(float *out, const float *v)
{
	out[0] *= v[0];
	out[1] *= v[1];
}

static
void op4s_sdl_descale_s_ref(float *in, const float *v)
{
	in[0] *= v[0];
	in[1] *= v[1];
}

static
void op4s_sdl_save_s_ref(float *out, float *restrict addr)
{
	addr[0] = out[0];
	addr[1] = out[1];
}

static
void op4s_sdl_export_s_ref(const float *l, float *restrict addr, int idx)
{
	addr[idx] = l[idx];
}

static
void op4s_sdl2_preload_prolog_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(w);
	UNUSED(v);
	UNUSED(l);
	UNUSED(c);
	UNUSED(r);
	UNUSED(z);
	UNUSED(in);

	op4s_sdl2_import_preload_s_ref(out, (*addr));

 	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl2_preload_prolog_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_import_preload_s_sse((out), (*(addr))); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_preload_prolog_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(w);
	UNUSED(v);
	UNUSED(l);
	UNUSED(r);
	UNUSED(z);
	UNUSED(in);

	op4s_sdl2_import_preload_s_ref(out, (*addr));

 	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_preload_prolog_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_import_preload_s_sse((out), (*(addr))); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl2_pass_fwd_prolog_full_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);

	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl2_pass_fwd_prolog_full_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl2_pass_inv_prolog_full_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);

	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// descale
	op4s_sdl2_descale_s_ref(in, v);

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl2_pass_inv_prolog_full_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_descale_s_sse((in), (v)); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_prolog_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);

	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// descale
	op4s_sdl2_descale_s_ref(in, v);

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// update
	op4s_sdl6_update_s_ref(z, l, r);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_prolog_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_descale_s_sse((in), (v)); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_prolog_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);

	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// (descale)

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// update
	op4s_sdl6_update_s_ref(z, l, r);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_prolog_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl2_pass_fwd_prolog_light_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_fwd_prolog_light_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl2_pass_inv_prolog_light_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_inv_prolog_light_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_prolog_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_prolog_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_prolog_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_prolog_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl2_pass_fwd_core_light_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_fwd_core_light_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl2_pass_inv_core_light_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_inv_core_light_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_core_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// (update)
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_core_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_core_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// (update)
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_core_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_postcore_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_postcore_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_postcore_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(addr);

	// shuffle + input-high
	op4s_sdl2_shuffle_input_high_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_postcore_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_input_high_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl2_pass_fwd_core_full_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// scale
	op4s_sdl2_scale_s_ref(out, v);

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl2_pass_fwd_core_full_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_scale_s_sse((out), (v)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl2_pass_inv_core_full_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// descale
	op4s_sdl2_descale_s_ref(in, v);

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl2_pass_inv_core_full_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_descale_s_sse((in), (v)); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_core_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// descale
	op4s_sdl2_descale_s_ref(in, v);

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// (update)

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_core_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_descale_s_sse((in), (v)); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_core_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// (descale)

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)
	op4s_sdl2_scale_s_ref(out, v);

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// (update)

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_core_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_scale_s_sse((out), (v)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_postcore_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// descale
	op4s_sdl2_descale_s_ref(in, v);

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl6_update_s_ref(z, l, r);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_postcore_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_descale_s_sse((in), (v)); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_postcore_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// load
	op4s_sdl2_load_s_ref(in, (*addr));

	// (descale)

	// shuffle + input-low
	op4s_sdl2_shuffle_input_low_s_ref(in, z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)
	op4s_sdl2_scale_s_ref(out, v);

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl6_update_s_ref(z, l, r);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_postcore_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_load_s_sse((in), (*(addr))); \
	op4s_sdl2_shuffle_input_low_s_sse((in), (z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_scale_s_sse((out), (v)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl2_pass_fwd_epilog_full_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// scale
	op4s_sdl2_scale_s_ref(out, v);

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl2_pass_fwd_epilog_full_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_scale_s_sse((out), (v)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl2_pass_inv_epilog_full_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl2_pass_inv_epilog_full_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_epilog_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl6_update_s_ref(z, l, r);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_epilog_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_epilog_full_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-high
	op4s_sdl2_output_high_s_ref(out, l, z);

	// (scale)
	op4s_sdl2_scale_s_ref(out, v);

	// save-shift
	op4s_sdl2_save_shift_s_ref(out, (*addr)-12);

	// update
	op4s_sdl6_update_s_ref(z, l, r);

	// pointers
	(*addr) += 4;
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_epilog_full_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_high_s_sse((out), (l), (z)); \
	op4s_sdl2_scale_s_sse((out), (v)); \
	op4s_sdl2_save_shift_s_sse((out), (*(addr))-12); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
	(*(addr)) += 4; \
} while(0)
#endif

static
void op4s_sdl2_pass_fwd_epilog_light_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);
	UNUSED(addr);

	// shuffle
	op4s_sdl2_shuffle_s_ref(c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_fwd_epilog_light_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl2_pass_inv_epilog_light_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);
	UNUSED(addr);

	// shuffle
	op4s_sdl2_shuffle_s_ref(c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_inv_epilog_light_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_epilog_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);
	UNUSED(addr);

	// shuffle
	op4s_sdl2_shuffle_s_ref(z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_epilog_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_epilog_light_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);
	UNUSED(addr);

	// shuffle
	op4s_sdl2_shuffle_s_ref(z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_epilog_light_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl2_pass_fwd_epilog_flush_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// scale
	op4s_sdl2_scale_s_ref(out, v);

	// save
	op4s_sdl2_save_s_ref(out, (*addr)-12);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_fwd_epilog_flush_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_scale_s_sse((out), (v)); \
	op4s_sdl2_save_s_sse((out), (*(addr))-12); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl2_pass_inv_epilog_flush_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(c, r);

	// operation
	op4s_sdl2_op_s_ref(z, c, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// (scale)

	// save
	op4s_sdl2_save_s_ref(out, (*addr)-12);

	// update
	op4s_sdl2_update_s_ref(c, l, r, z);
}

#ifdef __SSE__
#define op4s_sdl2_pass_inv_epilog_flush_s_sse(w, v, l, c, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((c), (r)); \
	op4s_sdl2_op_s_sse((z), (c), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_save_s_sse((out), (*(addr))-12); \
	op4s_sdl2_update_s_sse((c), (l), (r), (z)); \
} while(0)
#endif

static
void op4s_sdl6_pass_inv_epilog_flush_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// (scale)

	// save
	op4s_sdl2_save_s_ref(out, (*addr)-12);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_inv_epilog_flush_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_save_s_sse((out), (*(addr))-12); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl6_pass_fwd_epilog_flush_s_ref(const float *w, const float *v, float *l, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);

	// shuffle
	op4s_sdl2_shuffle_s_ref(z, r);

	// operation
	op4s_sdl6_op_s_ref(z, w, l, r);

	// output-low
	op4s_sdl2_output_low_s_ref(out, l, z);

	// (scale)
	op4s_sdl2_scale_s_ref(out, v);

	// save
	op4s_sdl2_save_s_ref(out, (*addr)-12);

	// update
	op4s_sdl6_update_s_ref(z, l, r);
}

#ifdef __SSE__
#define op4s_sdl6_pass_fwd_epilog_flush_s_sse(w, v, l, r, z, in, out, addr) \
do { \
	op4s_sdl2_shuffle_s_sse((z), (r)); \
	op4s_sdl6_op_s_sse((z), (w), (l), (r)); \
	op4s_sdl2_output_low_s_sse((out), (l), (z)); \
	op4s_sdl2_scale_s_sse((out), (v)); \
	op4s_sdl2_save_s_sse((out), (*(addr))-12); \
	op4s_sdl6_update_s_sse((z), (l), (r)); \
} while(0)
#endif

static
void op4s_sdl_pass_fwd_prolog_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(out);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_s_ref(in, *addr+4);

	// (descale)

	// input
	op4s_sdl_input_s_ref(in, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// (output)

	// (scale)

	// (save)

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 2;
}

static
void op4s_sdl_pass_inv_prolog_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(out);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_s_ref(in, *addr+4);

	// descale
	op4s_sdl_descale_s_ref(in, v);

	// input
	op4s_sdl_input_s_ref(in, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// (output)

	// (scale)

	// (save)

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 2;
}

static
void op4s_sdl_pass_fwd_core_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_s_ref(in, *addr+4);

	// (descale)

	// input
	op4s_sdl_input_s_ref(in, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op4s_sdl_output_s_ref(out, l, z);

	// scale
	op4s_sdl_scale_s_ref(out, v);

	// save
	op4s_sdl_save_s_ref(out, *addr-6);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 2;
}

static
void op4s_sdl_pass_inv_core_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_s_ref(in, *addr+4);

	// descale
	op4s_sdl_descale_s_ref(in, v);

	// input
	op4s_sdl_input_s_ref(in, c, r);

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op4s_sdl_output_s_ref(out, l, z);

	// (scale)

	// save
	op4s_sdl_save_s_ref(out, *addr-6);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 2;
}

static
void op4s_sdl_pass_fwd_epilog_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(in);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// (load)

	// (descale)

	// (input)

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// output
	op4s_sdl_output_s_ref(out, l, z);

	// scale
	op4s_sdl_scale_s_ref(out, v);

	// save
	op4s_sdl_save_s_ref(out, *addr-6);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 2;
}

static
void op4s_sdl_pass_inv_epilog_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr)
{
	UNUSED(v);
	UNUSED(in);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// (load)

	// (descale)

	// (input)

	// operation
	op4s_sdl_op_s_ref(z, c, w, l, r);

	// (output)
	op4s_sdl_output_s_ref(out, l, z);

	// (scale)

	// save
	op4s_sdl_save_s_ref(out, *addr-6);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) += 2;
}


#ifdef __SSE__
/**
 * @brief Shifted Double-Loop implementation of lifting scheme with 6
 * iterations merged.
 *
 * i.e. 12 = (6)*(2) = (2*3)*(2) coefficients per one iteration.
 */
static
void accel_lift_op4s_main_sdl6_sse_s(
	float *restrict arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	// 6+ coeffs implies 3+ steps
	assert( steps >= 3 );

	assert( is_aligned_16(arr) );

	const __m128 w = { delta, gamma, beta, alpha };
	const __m128 v = { 1/zeta, zeta, 1/zeta, zeta };
	__m128 l;
	__m128 r;
	__m128 z;
	__m128 in;
	__m128 out;

	const int S = steps-3;
	const int U = S / 6;
	const int M = S % 6;
	const int T = M >> 1;

	if( scaling < 0 )
	{
		// ****** inverse transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *addr = ASSUME_ALIGNED(arr, 16);

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl6_preload_prolog_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl6_import_s_sse(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_inv_prolog_full_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl6_import_s_sse(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl6_pass_inv_prolog_light_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl6_import_s_sse(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_inv_prolog_full_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl6_import_s_sse(l, 0, out);

		// *** core ***

		// core: for u = 0 to U
		for(int u = 0; u < U; u++)
		{
			// NOTE: l, r, z

			// core: pass1-core-light
			op4s_sdl6_pass_inv_core_light_s_sse(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass1-core-full
			op4s_sdl6_pass_inv_core_full_s_sse(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass2-core-light
			op4s_sdl6_pass_inv_core_light_s_sse(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z

			// core: pass2-core-full
			op4s_sdl6_pass_inv_core_full_s_sse(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass3-core-light
			op4s_sdl6_pass_inv_core_light_s_sse(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass3-core-full
			op4s_sdl6_pass_inv_core_full_s_sse(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z
		}

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl6_pass_inv_postcore_light_s_sse(w, v, l, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl6_pass_inv_postcore_full_s_sse(w, v, l, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl6_pass_inv_postcore_light_s_sse(w, v, l, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_inv_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_inv_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_inv_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_inv_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_inv_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl6_pass_inv_epilog_flush_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 0);
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *addr = ASSUME_ALIGNED(arr, 16);

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl6_preload_prolog_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl6_import_s_sse(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_fwd_prolog_full_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl6_import_s_sse(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl6_pass_fwd_prolog_light_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl6_import_s_sse(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_fwd_prolog_full_s_sse(w, v, l, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl6_import_s_sse(l, 0, out);

		// *** core ***

		// core: for u = 0 to U
		for(int u = 0; u < U; u++)
		{
			// NOTE: l, r, z

			// core: pass1-core-light
			op4s_sdl6_pass_fwd_core_light_s_sse(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass1-core-full
			op4s_sdl6_pass_fwd_core_full_s_sse(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass2-core-light
			op4s_sdl6_pass_fwd_core_light_s_sse(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z

			// core: pass2-core-full
			op4s_sdl6_pass_fwd_core_full_s_sse(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass3-core-light
			op4s_sdl6_pass_fwd_core_light_s_sse(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass3-core-full
			op4s_sdl6_pass_fwd_core_full_s_sse(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z
		}

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl6_pass_fwd_postcore_light_s_sse(w, v, l, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl6_pass_fwd_postcore_full_s_sse(w, v, l, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl6_pass_fwd_postcore_light_s_sse(w, v, l, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_fwd_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_fwd_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_fwd_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_fwd_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_fwd_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl6_pass_fwd_epilog_flush_s_sse(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_sse(l, &arr[2*steps], 0);
		}
	}
	else
	{
		// ****** transform w/o scaling ******

		// not implemented yet
		dwt_util_abort();
	}
}
#endif /* __SSE__ */

/**
 * @brief Shifted Double-Loop implementation of lifting scheme with 6
 * iterations merger.
 *
 * i.e. 12 = (6)*(2) = (2*3)*(2) coefficients per one iteration.
 */
static
void accel_lift_op4s_main_sdl6_ref_s(
	float *restrict arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	// 6+ coeffs implies 3+ steps
	assert( steps >= 3 );

	assert( is_aligned_16(arr) );

	const float w[4] = { delta, gamma, beta, alpha };
	const float v[4] = { 1/zeta, zeta, 1/zeta, zeta };
	float l[4];
	float r[4];
	float z[4];
	float in[4];
	float out[4];

	const int S = steps-3;
	const int U = S / 6;
	const int M = S % 6;
	const int T = M >> 1;

	if( scaling < 0 )
	{
		// ****** inverse transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *addr = ASSUME_ALIGNED(arr, 16);

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl6_preload_prolog_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl6_import_s_ref(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_inv_prolog_full_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl6_import_s_ref(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl6_pass_inv_prolog_light_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl6_import_s_ref(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_inv_prolog_full_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl6_import_s_ref(l, 0, out);

		// *** core ***

		// core: for u = 0 to U
		for(int u = 0; u < U; u++)
		{
			// NOTE: l, r, z

			// core: pass1-core-light
			op4s_sdl6_pass_inv_core_light_s_ref(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass1-core-full
			op4s_sdl6_pass_inv_core_full_s_ref(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass2-core-light
			op4s_sdl6_pass_inv_core_light_s_ref(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z

			// core: pass2-core-full
			op4s_sdl6_pass_inv_core_full_s_ref(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass3-core-light
			op4s_sdl6_pass_inv_core_light_s_ref(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass3-core-full
			op4s_sdl6_pass_inv_core_full_s_ref(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z
		}

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl6_pass_inv_postcore_light_s_ref(w, v, l, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl6_pass_inv_postcore_full_s_ref(w, v, l, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl6_pass_inv_postcore_light_s_ref(w, v, l, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_inv_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_inv_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_inv_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_inv_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_inv_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl6_pass_inv_epilog_flush_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 0);
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *addr = ASSUME_ALIGNED(arr, 16);

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl6_preload_prolog_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl6_import_s_ref(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_fwd_prolog_full_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl6_import_s_ref(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl6_pass_fwd_prolog_light_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl6_import_s_ref(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl6_pass_fwd_prolog_full_s_ref(w, v, l, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl6_import_s_ref(l, 0, out);

		// *** core ***

		// core: for u = 0 to U
		for(int u = 0; u < U; u++)
		{
			// NOTE: l, r, z

			// core: pass1-core-light
			op4s_sdl6_pass_fwd_core_light_s_ref(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass1-core-full
			op4s_sdl6_pass_fwd_core_full_s_ref(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass2-core-light
			op4s_sdl6_pass_fwd_core_light_s_ref(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z

			// core: pass2-core-full
			op4s_sdl6_pass_fwd_core_full_s_ref(w, v, /*l*/l, /*r*/r, /*z*/z, in, out, &addr);

			// NOTE: z => l, l => r, r => z

			// core: pass3-core-light
			op4s_sdl6_pass_fwd_core_light_s_ref(w, v, /*l*/r, /*r*/z, /*z*/l, in, out, &addr);

			// NOTE: (r => z) => l, (z => l) => r, (l => r) => z

			// core: pass3-core-full
			op4s_sdl6_pass_fwd_core_full_s_ref(w, v, /*l*/z, /*r*/l, /*z*/r, in, out, &addr);

			// NOTE: ((l => r) => z) => l, ((r => z) => l) => r, ((z => l) => r) => z
		}

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl6_pass_fwd_postcore_light_s_ref(w, v, l, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl6_pass_fwd_postcore_full_s_ref(w, v, l, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl6_pass_fwd_postcore_light_s_ref(w, v, l, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_fwd_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_fwd_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_fwd_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl6_pass_fwd_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl6_pass_fwd_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl6_pass_fwd_epilog_flush_s_ref(w, v, l, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl6_export_s_ref(l, &arr[2*steps], 0);
		}
	}
	else
	{
		// ****** transform w/o scaling ******

		// not implemented yet
		dwt_util_abort();
	}
}

/**
 * @brief SDL with 2 iterations merged.
 *
 * i.e. loads and stores 4 = (2)*(2) coefficients in every iteration.
 */
static
void accel_lift_op4s_main_sdl2_ref_s(
	float *restrict arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	// 6+ coeffs implies 3+ steps
	assert( steps >= 3 );

	assert( is_aligned_16(arr) );

	const float w[4] = { delta, gamma, beta, alpha };
	const float v[4] = { 1/zeta, zeta, 1/zeta, zeta };
	float l[4];
	float c[4];
	float r[4];
	float z[4];
	float in[4];
	float out[4];

	const int S = steps-3;
	const int T = S >> 1;

	if( scaling < 0 )
	{
		// ****** inverse transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *restrict addr = arr;

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl2_preload_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl2_import_s_ref(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_inv_prolog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl2_import_s_ref(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl2_pass_inv_prolog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl2_import_s_ref(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_inv_prolog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl2_import_s_ref(l, 0, out);

		// *** core ***

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl2_pass_inv_core_light_s_ref(w, v, l, c, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl2_pass_inv_core_full_s_ref(w, v, l, c, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl2_pass_inv_core_light_s_ref(w, v, l, c, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_inv_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_inv_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_inv_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_inv_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_inv_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl2_pass_inv_epilog_flush_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 0);
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *restrict addr = arr;

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl2_preload_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl2_import_s_ref(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_fwd_prolog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl2_import_s_ref(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl2_pass_fwd_prolog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl2_import_s_ref(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_fwd_prolog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl2_import_s_ref(l, 0, out);

		// *** core ***

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl2_pass_fwd_core_light_s_ref(w, v, l, c, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl2_pass_fwd_core_full_s_ref(w, v, l, c, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl2_pass_fwd_core_light_s_ref(w, v, l, c, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_fwd_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_fwd_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_fwd_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_fwd_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_fwd_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl2_pass_fwd_epilog_flush_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_ref(l, &arr[2*steps], 0);
		}
	}
	else
	{
		// ****** transform w/o scaling ******

		// not implemented yet
		dwt_util_abort();
	}
}

#ifdef __SSE__
static
void accel_lift_op4s_main_sdl2_sse_s(
	float *restrict arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	// 6+ coeffs implies 3+ steps
	assert( steps >= 3 );

	assert( is_aligned_16(arr) );

	// FIXME: make global variables?
	const __m128 w = { delta, gamma, beta, alpha };
	const __m128 v = { 1/zeta, zeta, 1/zeta, zeta };
	__m128 l;
	__m128 c;
	__m128 r;
	__m128 z;
	__m128 in;
	__m128 out;

	const int S = steps-3;
	const int T = S >> 1;

	if( scaling < 0 )
	{
		// ****** inverse transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *restrict addr = arr;

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl2_preload_prolog_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl2_import_s_sse(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_inv_prolog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl2_import_s_sse(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl2_pass_inv_prolog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl2_import_s_sse(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_inv_prolog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl2_import_s_sse(l, 0, out);

		// *** core ***

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl2_pass_inv_core_light_s_sse(w, v, l, c, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl2_pass_inv_core_full_s_sse(w, v, l, c, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl2_pass_inv_core_light_s_sse(w, v, l, c, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_inv_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_inv_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_inv_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_inv_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_inv_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl2_pass_inv_epilog_flush_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 0);
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		float *restrict addr = arr;

		// *** prolog2 ***

		// prolog2: import-preload
		op4s_sdl2_preload_prolog_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(3)
		op4s_sdl2_import_s_sse(l, 3, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_fwd_prolog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(2)
		op4s_sdl2_import_s_sse(l, 2, out);

		// prolog2: pass-prolog-light
		op4s_sdl2_pass_fwd_prolog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl2_import_s_sse(l, 1, out);

		// prolog2: pass-prolog-full
		op4s_sdl2_pass_fwd_prolog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl2_import_s_sse(l, 0, out);

		// *** core ***

		// core: for t = 0 to T do
		for(int t = 0; t < T; t++)
		{
			// core: pass-core-light
			op4s_sdl2_pass_fwd_core_light_s_sse(w, v, l, c, r, z, in, out, &addr);

			// core: pass-core-full
			op4s_sdl2_pass_fwd_core_full_s_sse(w, v, l, c, r, z, in, out, &addr);
		}

		// core: if odd then
		if( is_odd(S) )
		{
			// core: pass-core-light
			op4s_sdl2_pass_fwd_core_light_s_sse(w, v, l, c, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		if( is_odd(S) )
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_fwd_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_fwd_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_fwd_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 0);
		}
		else
		{
			// epilog2: export(3)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 3);

			// epilog2: pass-epilog-light
			op4s_sdl2_pass_fwd_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 2);

			// epilog2: pass-epilog-full
			op4s_sdl2_pass_fwd_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 1);

			// epilog2: pass-epilog-flush
			op4s_sdl2_pass_fwd_epilog_flush_s_sse(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl2_export_s_sse(l, &arr[2*steps], 0);
		}
	}
	else
	{
		// ****** transform w/o scaling ******

		// not implemented yet
		dwt_util_abort();
	}
}
#endif /* __SSE__ */

/**
 * @brief Shifted double-loop algorithm.
 *
 * This function processes 2 coefficients (even + odd) per one iteration.
 */
static
void accel_lift_op4s_main_sdl_ref_s(
	float *restrict arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	// 6+ coeffs implies 3+ steps
	assert( steps >= 3 );

	assert( is_aligned_16(arr) );

	const float w[4] = { delta, gamma, beta, alpha };
	const float v[4] = { 1/zeta, zeta, 1/zeta, zeta };
	float l[4];
	float c[4];
	float r[4];
	float z[4];
	float in[4];
	float out[4];

	const int S = steps-3;

	if( scaling < 0 )
	{
		// ****** inverse transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		// this pointer is needed because of use of arr[] in import and export functions
		float *addr = ASSUME_ALIGNED(arr, 16);

		// *** prolog2 ***

		// prolog2: import(3)
		op4s_sdl_import_s_ref(l, arr, 3);

		// prolog2: pass-prolog
		op4s_sdl_pass_inv_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);
	
		// prolog2: import(2)
		op4s_sdl_import_s_ref(l, arr, 2);

		// prolog2: pass-prolog
		op4s_sdl_pass_inv_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl_import_s_ref(l, arr, 1);

		// prolog2: pass-prolog
		op4s_sdl_pass_inv_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl_import_s_ref(l, arr, 0);

		// *** core ***

		// core: for s = 0 to S do
		for(int s = 0; s < S; s++)
		{
			// core: pass-core
			op4s_sdl_pass_inv_core_s_ref(w, v, l, c, r, z, in, out, &addr);

		}

		// *** epilog2 ***

		// epilog2: export(3)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 3);

		// epilog2: pass-epilog
		op4s_sdl_pass_inv_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// epilog2: export(2)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 2);

		// epilog2: pass-epilog
		op4s_sdl_pass_inv_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// epilog2: export(1)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 1);

		// epilog2: pass-epilog
		op4s_sdl_pass_inv_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// epilog2: export(0)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 0);
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		// FIXME: support for several workers
		assert( 1 == dwt_util_get_num_workers() );

		// *** init ***

		// this pointer is needed because of use of arr[] in import and export functions
		float *addr = ASSUME_ALIGNED(arr, 16);

		// *** prolog2 ***

		// prolog2: import(3)
		op4s_sdl_import_s_ref(l, arr, 3);

		// prolog2: pass-prolog
		op4s_sdl_pass_fwd_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);
	
		// prolog2: import(2)
		op4s_sdl_import_s_ref(l, arr, 2);

		// prolog2: pass-prolog
		op4s_sdl_pass_fwd_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(1)
		op4s_sdl_import_s_ref(l, arr, 1);

		// prolog2: pass-prolog
		op4s_sdl_pass_fwd_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// prolog2: import(0)
		op4s_sdl_import_s_ref(l, arr, 0);

		// *** core ***

		// core: for s = 0 to S do
		for(int s = 0; s < S; s++)
		{
			// core: pass-core
			op4s_sdl_pass_fwd_core_s_ref(w, v, l, c, r, z, in, out, &addr);
		}

		// *** epilog2 ***

		// epilog2: export(3)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 3);

		// epilog2: pass-epilog
		op4s_sdl_pass_fwd_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// epilog2: export(2)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 2);

		// epilog2: pass-epilog
		op4s_sdl_pass_fwd_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// epilog2: export(1)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 1);

		// epilog2: pass-epilog
		op4s_sdl_pass_fwd_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

		// epilog2: export(0)
		op4s_sdl_export_s_ref(l, &arr[2*steps], 0);
	}
	else
	{
		// ****** transform w/o scaling ******

		// not implemented yet
		dwt_util_abort();
	}
}

#ifdef __x86_64__
static
void accel_lift_op4s_main_dl_nosse_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
// this long string disables SSE support (only no-sse is not enough)
__attribute__ ((__target__ ("no-mmx,no-sse,no-sse2,no-sse3,no-sse4,no-sse4.1")));
#endif
static
void accel_lift_op4s_main_dl_nosse_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	if( scaling < 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// aux. variables
			float in[2];
			float out[2];
			float r[4];
			float c[4];
			float l[4];

			// values that have to be passed from iteration to iteration
			// slide in left border
			l[0] = arr_local[0];
			l[1] = arr_local[1];
			l[2] = arr_local[2];
			l[3] = arr_local[3];

			// init
			float *addr = arr_local + 4;

			// loop by pairs from left to right
			for(int s = 0; s < steps; s++)
			{
				// inputs
				in[0] = addr[0];
				in[1] = addr[1];

				// scales
				in[0] = in[0] * v[0];
				in[1] = in[1] * v[1];

				// shuffles
				out[0] = l[0];
				c[0]   = l[1];
				c[1]   = l[2];
				c[2]   = l[3];
				c[3]   = in[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3]   = in[1];
				r[2]   = c[3]+w[3]*(l[3]+r[3]);
				r[1]   = c[2]+w[2]*(l[2]+r[2]);
				r[0]   = c[1]+w[1]*(l[1]+r[1]);
				out[1] = c[0]+w[0]*(l[0]+r[0]);

				// outputs
				addr[0-4] = out[0];
				addr[1-4] = out[1];

				// update l[]
				l[0] = r[0];
				l[1] = r[1];
				l[2] = r[2];
				l[3] = r[3];

				// pointers
				addr += 2;
			}

			// slide out right border
			addr[0-4] = l[0];
			addr[1-4] = l[1];
			addr[2-4] = l[2];
			addr[3-4] = l[3];
		}
	}
	else if ( scaling > 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// aux. variables
			float in[2];
			float out[2];
			float r[4];
			float c[4];
			float l[4];

			// values that have to be passed from iteration to iteration
			// slide in left border
			l[0] = arr_local[0];
			l[1] = arr_local[1];
			l[2] = arr_local[2];
			l[3] = arr_local[3];

			// init
			float *addr = arr_local + 4;

			// loop by pairs from left to right
			for(int s = 0; s < steps; s++)
			{
				// inputs
				in[0] = addr[0];
				in[1] = addr[1];

				// shuffles
				out[0] = l[0];
				c[0]   = l[1];
				c[1]   = l[2];
				c[2]   = l[3];
				c[3]   = in[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3]   = in[1];
				r[2]   = c[3]+w[3]*(l[3]+r[3]);
				r[1]   = c[2]+w[2]*(l[2]+r[2]);
				r[0]   = c[1]+w[1]*(l[1]+r[1]);
				out[1] = c[0]+w[0]*(l[0]+r[0]);

				// scales
				out[0] = out[0] * v[0];
				out[1] = out[1] * v[1];

				// outputs
				addr[0-4] = out[0];
				addr[1-4] = out[1];

				// update l[]
				l[0] = r[0];
				l[1] = r[1];
				l[2] = r[2];
				l[3] = r[3];

				// pointers
				addr += 2;
			}

			// slide out right border
			addr[0-4] = l[0];
			addr[1-4] = l[1];
			addr[2-4] = l[2];
			addr[3-4] = l[3];
		}
	}
	else
	{
		// fallback, not implemented
		accel_lift_op4s_main_s(arr, steps, alpha, beta, gamma, delta, zeta, scaling);
	}
}

/**
 * @brief Double-loop algorithm from Rade Kutil: A Single-Loop Approach to
 * SIMD Parallelization of 2-D Wavelet Lifting.
 */
static
void accel_lift_op4s_main_dl_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	if( scaling < 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// aux. variables
			float in[2];
			float out[2];
			float r[4];
			float c[4];
			float l[4];

			// values that have to be passed from iteration to iteration
			// slide in left border
			l[0] = arr_local[0];
			l[1] = arr_local[1];
			l[2] = arr_local[2];
			l[3] = arr_local[3];

			// init
			float *addr = ASSUME_ALIGNED(arr_local + 4, 16);

			// loop by pairs from left to right
			for(int s = 0; s < steps; s++)
			{
				// inputs
				in[0] = addr[0];
				in[1] = addr[1];

				// scales
				in[0] = in[0] * v[0];
				in[1] = in[1] * v[1];

				// shuffles
				out[0] = l[0];
				c[0]   = l[1];
				c[1]   = l[2];
				c[2]   = l[3];
				c[3]   = in[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3]   = in[1];
				r[2]   = c[3]+w[3]*(l[3]+r[3]);
				r[1]   = c[2]+w[2]*(l[2]+r[2]);
				r[0]   = c[1]+w[1]*(l[1]+r[1]);
				out[1] = c[0]+w[0]*(l[0]+r[0]);

				// outputs
				addr[0-4] = out[0];
				addr[1-4] = out[1];

				// update l[]
				l[0] = r[0];
				l[1] = r[1];
				l[2] = r[2];
				l[3] = r[3];

				// pointers
				addr += 2;
			}

			// slide out right border
			addr[0-4] = l[0];
			addr[1-4] = l[1];
			addr[2-4] = l[2];
			addr[3-4] = l[3];
		}
	}
	else if ( scaling > 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// aux. variables
			float in[2];
			float out[2];
			float r[4];
			float c[4];
			float l[4];

			// values that have to be passed from iteration to iteration
			// slide in left border
			l[0] = arr_local[0];
			l[1] = arr_local[1];
			l[2] = arr_local[2];
			l[3] = arr_local[3];

			// init
			float *addr = ASSUME_ALIGNED(arr_local + 4, 16);

			// loop by pairs from left to right
			for(int s = 0; s < steps; s++)
			{
				// inputs
				in[0] = addr[0];
				in[1] = addr[1];

				// shuffles
				out[0] = l[0];
				c[0]   = l[1];
				c[1]   = l[2];
				c[2]   = l[3];
				c[3]   = in[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3]   = in[1];
				r[2]   = c[3]+w[3]*(l[3]+r[3]);
				r[1]   = c[2]+w[2]*(l[2]+r[2]);
				r[0]   = c[1]+w[1]*(l[1]+r[1]);
				out[1] = c[0]+w[0]*(l[0]+r[0]);

				// scales
				out[0] = out[0] * v[0];
				out[1] = out[1] * v[1];

				// outputs
				addr[0-4] = out[0];
				addr[1-4] = out[1];

				// update l[]
				l[0] = r[0];
				l[1] = r[1];
				l[2] = r[2];
				l[3] = r[3];

				// pointers
				addr += 2;
			}

			// slide out right border
			addr[0-4] = l[0];
			addr[1-4] = l[1];
			addr[2-4] = l[2];
			addr[3-4] = l[3];
		}
	}
	else
	{
		// fallback, not implemented
		accel_lift_op4s_main_s(arr, steps, alpha, beta, gamma, delta, zeta, scaling);
	}
}

static
void accel_lift_op4s_main_dl4line_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	const int PAIR = 2;
	const int TAPS = 4; // 4-fold SIMD
	//const int LIFT_STEPS = 4; // 4 lifting steps for CDF 9/7

	if( scaling < 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4/*LIFT_STEPS*/] = { delta, gamma, beta, alpha };
			const float v[2/*PAIR*/] = { 1/zeta, zeta };

			// aux. variables
			float x[2];
			float y[2];
			float r[4];
			float c[4];
			float l[4];

			// values that have to be passed from iteration to iteration
			// slide in left border
			l[0] = arr_local[0];
			l[1] = arr_local[1];
			l[2] = arr_local[2];
			l[3] = arr_local[3];

			// init
			float *addr = ASSUME_ALIGNED(arr_local + 4, 16);

			float c4[TAPS], l4[TAPS], r4[TAPS];

			const float w4[4/*LIFT_STEPS*/][4/*TAPS*/] = {
				{ w[0], w[0], w[0], w[0] },
				{ w[1], w[1], w[1], w[1] },
				{ w[2], w[2], w[2], w[2] },
				{ w[3], w[3], w[3], w[3] }
			};

			const int S = steps;
			const int S4 = S / TAPS;
			const int R4 = S - S4*TAPS;

			// loop by group of four pairs from left to right
			for(int s = 0; s < S4; s++)
			{
				// inputs
				for(int i = 0; i < TAPS; i++)
				{
					l4[i] = addr[2*i+0];
					r4[i] = addr[2*i+1];
				}

				// scales
				for(int i = 0; i < TAPS; i++)
				{
					l4[i] *= v[0];
					r4[i] *= v[1];
				}

#if 0
				for(int t = LIFT_STEPS-1; t >= 0; t--)
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
#else
				int t;
				t = 3;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
				t = 2;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
				t = 1;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
				t = 0;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
#endif

				// outputs
				for(int i = 0; i < TAPS; i++)
				{
					addr[2*i+0-4] = l4[i];
					addr[2*i+1-4] = r4[i];
				}

				// pointers
				addr += PAIR * TAPS;
			}

			// loop by pairs from left to right
			for(int s = 0; s < R4; s++)
			{
				// inputs
				x[0] = addr[0];
				x[1] = addr[1];
					
				// scales
				x[0] *= v[0];
				x[1] *= v[1];

				// shuffles
				y[0] = l[0];
				c[0] = l[1];
				c[1] = l[2];
				c[2] = l[3];
				c[3] = x[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3] = x[1];
				r[2] = c[3]+w[3]*(l[3]+r[3]);
				r[1] = c[2]+w[2]*(l[2]+r[2]);
				r[0] = c[1]+w[1]*(l[1]+r[1]);
				y[1] = c[0]+w[0]*(l[0]+r[0]);

				// outputs
				addr[0-4] = y[0];
				addr[1-4] = y[1];

				// update l[]
				l[0] = r[0];
				l[1] = r[1];
				l[2] = r[2];
				l[3] = r[3];

				// pointers
				addr += PAIR;
			}
			
			// slide out right border
			addr[0-4] = l[0];
			addr[1-4] = l[1];
			addr[2-4] = l[2];
			addr[3-4] = l[3];
		}
	}
	else if ( scaling > 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = calc_temp_offset_s(arr, w);

			// constants
			const float w[4/*LIFT_STEPS*/] = { delta, gamma, beta, alpha };
			const float v[2/*PAIR*/] = { 1/zeta, zeta };

			// aux. variables
			float x[2];
			float y[2];
			float r[4];
			float c[4];
			float l[4];

			// values that have to be passed from iteration to iteration
			// slide in left border
			l[0] = arr_local[0];
			l[1] = arr_local[1];
			l[2] = arr_local[2];
			l[3] = arr_local[3];

			// init
			float *addr = ASSUME_ALIGNED(arr_local + 4, 16);

			float c4[TAPS], l4[TAPS], r4[TAPS];

			const float w4[4/*LIFT_STEPS*/][4/*TAPS*/] = {
				{ w[0], w[0], w[0], w[0] },
				{ w[1], w[1], w[1], w[1] },
				{ w[2], w[2], w[2], w[2] },
				{ w[3], w[3], w[3], w[3] }
			};

			const int S = steps;
			const int S4 = S / TAPS;
			const int R4 = S - S4*TAPS;

			// loop by group of four pairs from left to right
			for(int s = 0; s < S4; s++)
			{
				// inputs
				for(int i = 0; i < TAPS; i++)
				{
					l4[i] = addr[2*i+0];
					r4[i] = addr[2*i+1];
				}

#if 0
				for(int t = LIFT_STEPS-1; t >= 0; t--)
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
#else
				int t;
				t = 3;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
				t = 2;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
				t = 1;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
				t = 0;
				{
					// c[]
					c4[0] = l4[0];
					c4[1] = l4[1];
					c4[2] = l4[2];
					c4[3] = l4[3];

					// l[]
					l4[0] = l[t];
					l4[1] = r4[0];
					l4[2] = r4[1];
					l4[3] = r4[2];
					 l[t] = r4[3];

					// r[]
					r4[0] = c4[0]+w4[t][0]*(l4[0]+r4[0]);
					r4[1] = c4[1]+w4[t][1]*(l4[1]+r4[1]);
					r4[2] = c4[2]+w4[t][2]*(l4[2]+r4[2]);
					r4[3] = c4[3]+w4[t][3]*(l4[3]+r4[3]);
				}
#endif

				// scales
				for(int i = 0; i < TAPS; i++)
				{
					l4[i] *= v[0];
					r4[i] *= v[1];
				}

				// outputs
				for(int i = 0; i < TAPS; i++)
				{
					addr[2*i+0-4] = l4[i];
					addr[2*i+1-4] = r4[i];
				}

				// pointers
				addr += PAIR * TAPS;
			}

			// loop by pairs from left to right
			for(int s = 0; s < R4; s++)
			{
				// inputs
				x[0] = addr[0];
				x[1] = addr[1];

				// shuffles
				y[0] = l[0];
				c[0] = l[1];
				c[1] = l[2];
				c[2] = l[3];
				c[3] = x[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3] = x[1];
				r[2] = c[3]+w[3]*(l[3]+r[3]);
				r[1] = c[2]+w[2]*(l[2]+r[2]);
				r[0] = c[1]+w[1]*(l[1]+r[1]);
				y[1] = c[0]+w[0]*(l[0]+r[0]);

				// scales
				y[0] *= v[0];
				y[1] *= v[1];

				// outputs
				addr[0-4] = y[0];
				addr[1-4] = y[1];

				// update l[]
				l[0] = r[0];
				l[1] = r[1];
				l[2] = r[2];
				l[3] = r[3];

				// pointers
				addr += PAIR;
			}
			
			// slide out right border
			addr[0-4] = l[0];
			addr[1-4] = l[1];
			addr[2-4] = l[2];
			addr[3-4] = l[3];
		}
	}
	else
	{
		// fallback, not implemented
		accel_lift_op4s_main_s(arr, steps, alpha, beta, gamma, delta, zeta, scaling);
	}
}

#ifdef __SSE__
static
void accel_lift_op4s_main_dl4line_sse_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	const int PAIR = 2; // even and odd coefficient form a pair
	const int TAPS = 4; // 4-fold SIMD
	//const int LIFT_STEPS = 4; // 4 lifting steps for CDF 9/7

	const int S = steps;
	const int S4 = S / TAPS;
	const int R4 = S - S4*TAPS;

	if( scaling < 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = ASSUME_ALIGNED(calc_temp_offset_s(arr, w), 16);

			// intermediate results; force into registers
			__m128 l0, l1, l2, l3;

			// slide in left border
			{
				__m128 tmp0 = _mm_load_ps(arr_local);
				l0 = (__m128){ tmp0[0], tmp0[0], tmp0[0], tmp0[0] }; // movss+shufps
				l1 = (__m128){ tmp0[1], tmp0[1], tmp0[1], tmp0[1] }; // movss+shufps
				l2 = (__m128){ tmp0[2], tmp0[2], tmp0[2], tmp0[2] }; // movss+shufps
				l3 = (__m128){ tmp0[3], tmp0[3], tmp0[3], tmp0[3] }; // movss+shufps
			}

			// start from addr.
			float *addr = ASSUME_ALIGNED(arr_local + 4, 16);

			// aux. variables
			__m128 c4, l4, r4;

			__m128 w0 = { delta, delta, delta, delta }; // movss+shufps
			__m128 w1 = { gamma, gamma, gamma, gamma }; // movss+shufps
			__m128 w2 = { beta, beta, beta, beta }; // movss+shufps
			__m128 w3 = { alpha, alpha, alpha, alpha }; // movss+shufps
			__m128 v0 = { 1/zeta, 1/zeta, 1/zeta, 1/zeta }; // movss+shufps
			__m128 v1 = { zeta, zeta, zeta, zeta }; // movss+shufps

			// loop by group of four pairs from left to right
			for(int s = 0; s < S4; s++)
			{
				// inputs
				{
					r4 = _mm_load_ps(addr+0); // movaps
					__m128 tmp0 = _mm_load_ps(addr+4); // movaps
					l4 = r4; // movaps
					r4 = _mm_shuffle_ps(r4, tmp0, 0xdd/*221*/); // shufps
					l4 = _mm_shuffle_ps(l4, tmp0, 0x88/*136*/); // shufps
				}

				// scales
				{
					l4 *= v0; // mulps
					r4 *= v1; // mulps
				}

				// loop 3
#if 0
				{
					// c
					c4 = l4; // movaps, kill l4

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93/*147*/); // shufps
					l4[0] = l3[0]; // movss
					l3 = r4; // movaps
					l3 = _mm_shuffle_ps(l3, r4, 0xff/*255*/); // shufps

					// r
					r4 += l4; // addps
					r4 *= w3; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l3], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l3] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l3] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w3], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l3]"+x"(l3), [c4]"=x"(c4)
					: [w3]"x"(w3)
					:
				);
#endif
				// loop 2
#if 0
				{
					// c
					c4 = l4; // movaps

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93); // shufps
					l4[0] = l2[0]; // movss
					l2 = r4; // movaps
					l2 = _mm_shuffle_ps(l2, r4, 0xff); // shufps

					// r
					r4 += l4; // addps
					r4 *= w2; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l2], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l2] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l2] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w2], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l2]"+x"(l2), [c4]"=x"(c4)
					: [w2]"x"(w2)
					:
				);
#endif
				// loop 1
#if 0
				{
					// c
					c4 = l4; // movaps

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93); // shufps
					l4[0] = l1[0]; // movss
					l1 = r4; // movaps
					l1 = _mm_shuffle_ps(l1, r4, 0xff); // shufps

					// r
					r4 += l4; // addps
					r4 *= w1; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l1], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l1] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l1] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w1], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l1]"+x"(l1), [c4]"=x"(c4)
					: [w1]"x"(w1)
					:
				);
#endif

				// loop 0
#if 0
				{
					// c
					c4 = l4; // movaps

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93); // shufps
					l4[0] = l0[0]; // movss
					l0 = r4; // movaps
					l0 = _mm_shuffle_ps(l0, r4, 0xff); // shufps

					// r
					r4 += l4; // addps
					r4 *= w0; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l0], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l0] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l0] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w0], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l0]"+x"(l0), [c4]"=x"(c4)
					: [w0]"x"(w0)
					:
				);
#endif

				// outputs
				{
					// [ A B C D ] [ E F G H ] => [ A E B F ] [ C G D H ]
					__m128 tmp0 = l4; // movaps
					tmp0 = _mm_unpacklo_ps(tmp0, r4); // unpcklps
					_mm_store_ps(addr-4, tmp0); // movaps
					l4 = _mm_unpackhi_ps(l4, r4); // unpckhps
					_mm_store_ps(addr-0, l4); // movaps
				}

				// pointers
				addr += PAIR * TAPS;
			}

			// constants
			const __m128 w = { delta, gamma, beta, alpha };
			const __m128 v = { 1/zeta, zeta, 1/zeta, zeta };

			// variables
			__m128 x, y, r, c, l;

			// intermediate results
			l = (__m128){ l0[0], l1[0], l2[0], l3[0] };

			// loop by pairs from left to right
			for(int s = 0; s < R4; s++)
			{
				// inputs
				x[0] = addr[0];
				x[1] = addr[1];

				// scales
				x[0] *= v[0];
				x[1] *= v[1];

				// shuffles
				c = l;
				c = _mm_shuffle_ps(c,c,0x39);
				y[0] = l[0];
				c[3] = x[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3] = x[1];
				r[2] = c[3]+w[3]*(l[3]+r[3]);
				r[1] = c[2]+w[2]*(l[2]+r[2]);
				r[0] = c[1]+w[1]*(l[1]+r[1]);
				y[1] = c[0]+w[0]*(l[0]+r[0]);

				// outputs
				addr[0-4] = y[0];
				addr[1-4] = y[1];

				// update l[]
				l = r;

				// pointers
				addr += PAIR;
			}
			
			// slide out right border
			_mm_storeu_ps(addr-4, l);
		}
	}
	else if ( scaling > 0 )
	{
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = ASSUME_ALIGNED(calc_temp_offset_s(arr, w), 16);

			// intermediate results, force into registers
			__m128 l0, l1, l2, l3;

			// slide in left border
			{
				__m128 tmp0 = _mm_load_ps(arr_local);
				l0 = (__m128){ tmp0[0], tmp0[0], tmp0[0], tmp0[0] }; // movss+shufps
				l1 = (__m128){ tmp0[1], tmp0[1], tmp0[1], tmp0[1] }; // movss+shufps
				l2 = (__m128){ tmp0[2], tmp0[2], tmp0[2], tmp0[2] }; // movss+shufps
				l3 = (__m128){ tmp0[3], tmp0[3], tmp0[3], tmp0[3] }; // movss+shufps
			}

			// start from addr.
			float *addr = ASSUME_ALIGNED(arr_local + 4, 16);

			// aux. variables
			__m128 c4, l4, r4;

			__m128 w0 = { delta, delta, delta, delta }; // movss+shufps
			__m128 w1 = { gamma, gamma, gamma, gamma }; // movss+shufps
			__m128 w2 = { beta, beta, beta, beta }; // movss+shufps
			__m128 w3 = { alpha, alpha, alpha, alpha }; // movss+shufps
			__m128 v0 = { 1/zeta, 1/zeta, 1/zeta, 1/zeta }; // movss+shufps
			__m128 v1 = { zeta, zeta, zeta, zeta }; // movss+shufps

			// loop by group of four pairs from left to right
			for(int s = 0; s < S4; s++)
			{
				// inputs
				{
					r4 = _mm_load_ps(addr+0); // movaps
					__m128 tmp0 = _mm_load_ps(addr+4); // movaps
					l4 = r4; // movaps
					r4 = _mm_shuffle_ps(r4, tmp0, 0xdd/*221*/); // shufps
					l4 = _mm_shuffle_ps(l4, tmp0, 0x88/*136*/); // shufps
				}

				// loop 3
#if 0
				{
					// c
					c4 = l4; // movaps, kill l4

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93/*147*/); // shufps
					l4[0] = l3[0]; // movss
					l3 = r4; // movaps
					l3 = _mm_shuffle_ps(l3, r4, 0xff/*255*/); // shufps

					// r
					r4 += l4; // addps
					r4 *= w3; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l3], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l3] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l3] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w3], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l3]"+x"(l3), [c4]"=x"(c4)
					: [w3]"x"(w3)
					:
				);
#endif
				// loop 2
#if 0
				{
					// c
					c4 = l4; // movaps

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93); // shufps
					l4[0] = l2[0]; // movss
					l2 = r4; // movaps
					l2 = _mm_shuffle_ps(l2, r4, 0xff); // shufps

					// r
					r4 += l4; // addps
					r4 *= w2; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l2], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l2] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l2] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w2], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l2]"+x"(l2), [c4]"=x"(c4)
					: [w2]"x"(w2)
					:
				);
#endif
				// loop 1
#if 0
				{
					// c
					c4 = l4; // movaps

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93); // shufps
					l4[0] = l1[0]; // movss
					l1 = r4; // movaps
					l1 = _mm_shuffle_ps(l1, r4, 0xff); // shufps

					// r
					r4 += l4; // addps
					r4 *= w1; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l1], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l1] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l1] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w1], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l1]"+x"(l1), [c4]"=x"(c4)
					: [w1]"x"(w1)
					:
				);
#endif

				// loop 0
#if 0
				{
					// c
					c4 = l4; // movaps

					// l
					l4 = r4; // movaps
					l4 = _mm_shuffle_ps(l4, l4, 0x93); // shufps
					l4[0] = l0[0]; // movss
					l0 = r4; // movaps
					l0 = _mm_shuffle_ps(l0, r4, 0xff); // shufps

					// r
					r4 += l4; // addps
					r4 *= w0; // mulps
					r4 += c4; // addps
				}
#else
				__asm__ __volatile__(
					"movaps %[l4], %[c4] \n\t" // l4 => c4
					"movaps %[r4], %[l4] \n\t" // r4 => l4
					"shufps $147, %[l4], %[l4] \n\t" // l4 => l4
					"movss %[l0], %[l4] \n\t" // l3 => l4
					"movaps %[r4], %[l0] \n\t" // l4 => l3
					"shufps $255, %[r4], %[l0] \n\t" // r4 => l4
					"addps %[l4], %[r4] \n\t" // l4 => r4
					"mulps %[w0], %[r4] \n\t" // w3 => r4
					"addps %[c4], %[r4] \n\t" // c4 => r4
					: [l4]"+x"(l4), [r4]"+x"(r4), [l0]"+x"(l0), [c4]"=x"(c4)
					: [w0]"x"(w0)
					:
				);
#endif

				// scales
				{
					l4 *= v0; // mulps
					r4 *= v1; // mulps
				}
				
				// outputs
				{
					// [ A B C D ] [ E F G H ] => [ A E B F ] [ C G D H ]
					__m128 tmp0 = l4; // movaps
					tmp0 = _mm_unpacklo_ps(tmp0, r4); // unpcklps
					_mm_store_ps(addr-4, tmp0); // movaps
					l4 = _mm_unpackhi_ps(l4, r4); // unpckhps
					_mm_store_ps(addr-0, l4); // movaps
				}

				// pointers
				addr += PAIR * TAPS;
			}

			// constants
			const __m128 w = { delta, gamma, beta, alpha };
			const __m128 v = { 1/zeta, zeta, 1/zeta, zeta };

			// variables
			__m128 x, y, r, c, l;

			// intermediate results
			l = (__m128){ l0[0], l1[0], l2[0], l3[0] };

			// loop by pairs from left to right
			for(int s = 0; s < R4; s++)
			{
				// inputs
				x[0] = addr[0];
				x[1] = addr[1];

				// shuffles
				c = l;
				c = _mm_shuffle_ps(c,c,0x39);
				y[0] = l[0];
				c[3] = x[0];

				// operation z[] = c[] + w[] * ( l[] + r[] )
				// by sequential computation from top/right to bottom/left
				r[3] = x[1];
				r[2] = c[3]+w[3]*(l[3]+r[3]);
				r[1] = c[2]+w[2]*(l[2]+r[2]);
				r[0] = c[1]+w[1]*(l[1]+r[1]);
				y[1] = c[0]+w[0]*(l[0]+r[0]);

				// scales
				y[0] *= v[0];
				y[1] *= v[1];

				// outputs
				addr[0-4] = y[0];
				addr[1-4] = y[1];

				// update l[]
				l = r;

				// pointers
				addr += PAIR;
			}
			
			// slide out right border
			_mm_storeu_ps(addr-4, l);
		}
	}
	else
	{
		// fallback, not implemented
		accel_lift_op4s_main_s(arr, steps, alpha, beta, gamma, delta, zeta, scaling);
	}
}
#endif

/**
 * double-loop algorithm for 4 rows in parallel
 */
static
void accel_lift_op4s_main_dl4_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	if( scaling < 0 )
	{
		assert( 4 == dwt_util_get_num_workers() );

		// constants
		const float w[4] = { delta, gamma, beta, alpha };
		const float v[2] = { 1/zeta, zeta };

		// aux. variables
		float in[4][2];
		float out[4][2];
		float r[4][4];
		float c[4][4];
		float l[4][4];

		// pointers
		float *addr[4];

		for(int worker = 0; worker < dwt_util_get_num_workers(); worker++)
		{
			// init
			addr[worker] = calc_temp_offset_s(arr, worker);

			// import
			l[worker][0] = addr[worker][0];
			l[worker][1] = addr[worker][1];
			l[worker][2] = addr[worker][2];
			l[worker][3] = addr[worker][3];

			addr[worker] += 4;
		}

		// loop
		for(int s = 0; s < steps; s++)
		{
			for(int worker = 0; worker < dwt_util_get_num_workers(); worker++)
			{
				// inputs
				in[worker][0] = addr[worker][0];
				in[worker][1] = addr[worker][1];

				// scales
				in[worker][0] = in[worker][0] * v[0];
				in[worker][1] = in[worker][1] * v[1];

				// shuffles
				out[worker][0] = l[worker][0];
				c[worker][0]   = l[worker][1];
				c[worker][1]   = l[worker][2];
				c[worker][2]   = l[worker][3];
				c[worker][3]   = in[worker][0];

				// operation
				r[worker][3]   = in[worker][1];
				r[worker][2]   = c[worker][3] + w[3]*(l[worker][3] + r[worker][3]);
				r[worker][1]   = c[worker][2] + w[2]*(l[worker][2] + r[worker][2]);
				r[worker][0]   = c[worker][1] + w[1]*(l[worker][1] + r[worker][1]);
				out[worker][1] = c[worker][0] + w[0]*(l[worker][0] + r[worker][0]);

				// outputs
				addr[worker][0-4] = out[worker][0];
				addr[worker][1-4] = out[worker][1];

				// update
				l[worker][0] = r[worker][0];
				l[worker][1] = r[worker][1];
				l[worker][2] = r[worker][2];
				l[worker][3] = r[worker][3];

				// pointers
				addr[worker] += 2;
			}
		}


		for(int worker = 0; worker < dwt_util_get_num_workers(); worker++)
		{
			// export
			addr[worker][0-4] = l[worker][0];
			addr[worker][1-4] = l[worker][1];
			addr[worker][2-4] = l[worker][2];
			addr[worker][3-4] = l[worker][3];
		}
	}
	else if ( scaling > 0 )
	{
		assert( 4 == dwt_util_get_num_workers() );

		// constants
		const float w[4] = { delta, gamma, beta, alpha };
		const float v[2] = { 1/zeta, zeta };

		// aux. variables
		float in[4][2];
		float out[4][2];
		float r[4][4];
		float c[4][4];
		float l[4][4];

		// pointers
		float *addr[4];

		for(int worker = 0; worker < dwt_util_get_num_workers(); worker++)
		{
			// init
			addr[worker] = calc_temp_offset_s(arr, worker);

			// import
			l[worker][0] = addr[worker][0];
			l[worker][1] = addr[worker][1];
			l[worker][2] = addr[worker][2];
			l[worker][3] = addr[worker][3];

			addr[worker] += 4;
		}

		// loop
		for(int s = 0; s < steps; s++)
		{
			for(int worker = 0; worker < dwt_util_get_num_workers(); worker++)
			{
				// inputs
				in[worker][0] = addr[worker][0];
				in[worker][1] = addr[worker][1];

				// shuffles
				out[worker][0] = l[worker][0];
				c[worker][0]   = l[worker][1];
				c[worker][1]   = l[worker][2];
				c[worker][2]   = l[worker][3];
				c[worker][3]   = in[worker][0];

				// operation
				r[worker][3]   = in[worker][1];
				r[worker][2]   = c[worker][3] + w[3]*(l[worker][3] + r[worker][3]);
				r[worker][1]   = c[worker][2] + w[2]*(l[worker][2] + r[worker][2]);
				r[worker][0]   = c[worker][1] + w[1]*(l[worker][1] + r[worker][1]);
				out[worker][1] = c[worker][0] + w[0]*(l[worker][0] + r[worker][0]);

				// scales
				out[worker][0] = out[worker][0] * v[0];
				out[worker][1] = out[worker][1] * v[1];

				// outputs
				addr[worker][0-4] = out[worker][0];
				addr[worker][1-4] = out[worker][1];

				// update
				l[worker][0] = r[worker][0];
				l[worker][1] = r[worker][1];
				l[worker][2] = r[worker][2];
				l[worker][3] = r[worker][3];

				// pointers
				addr[worker] += 2;
			}
		}

		for(int worker = 0; worker < dwt_util_get_num_workers(); worker++)
		{
			// export
			addr[worker][0-4] = l[worker][0];
			addr[worker][1-4] = l[worker][1];
			addr[worker][2-4] = l[worker][2];
			addr[worker][3-4] = l[worker][3];
		}
	}
	else
	{
		// fallback, not implemented
		accel_lift_op4s_main_s(arr, steps, alpha, beta, gamma, delta, zeta, scaling);
	}
}

#ifdef __SSE__
#define op4s_dl4_preload_s_sse(addr, idx, temp, s) \
do { \
	(*(addr))[(idx)] = (temp)[(idx)][(s)]; \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_preload_s_sse4(addr, temp, s) \
do { \
	op4s_dl4_preload_s_sse((addr), 0, (temp), (s)); \
	op4s_dl4_preload_s_sse((addr), 1, (temp), (s)); \
	op4s_dl4_preload_s_sse((addr), 2, (temp), (s)); \
	op4s_dl4_preload_s_sse((addr), 3, (temp), (s)); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_import_s_sse(l, addr) \
do { \
	(l) = *(addr); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_import_s_sse4(l, addr) \
do { \
	op4s_dl4_import_s_sse((l)[0], (addr) + 0); \
	op4s_dl4_import_s_sse((l)[1], (addr) + 1); \
	op4s_dl4_import_s_sse((l)[2], (addr) + 2); \
	op4s_dl4_import_s_sse((l)[3], (addr) + 3); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_input_s_sse(in, addr) \
do { \
	(in) = *(addr); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_input_s_sse4(in, addr) \
do { \
	op4s_dl4_input_s_sse((in)[0], (addr) + 0); \
	op4s_dl4_input_s_sse((in)[1], (addr) + 1); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_scale_s_sse(in, v) \
do { \
	(in) *= (v); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_descale_s_sse(out, v) \
do { \
	(out) *= (v); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_scale_s_sse4(in, v) \
do { \
	op4s_dl4_scale_s_sse((in)[0], (v)[0]); \
	op4s_dl4_scale_s_sse((in)[1], (v)[1]); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_descale_s_sse4(out, v) \
do { \
	op4s_dl4_descale_s_sse((out)[0], (v)[0]); \
	op4s_dl4_descale_s_sse((out)[1], (v)[1]); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_mov_s_sse(z, l) \
do { \
	(z) = (l); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_op3_s_sse(z, c, w, r) \
do { \
	(z) += (r); \
	(z) *= (w); \
	(z) += (c); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_op_s_sse4(in, out, w, c, l, r) \
do { \
	op4s_dl4_mov_s_sse((out)[0], (l)[0]); \
	op4s_dl4_mov_s_sse((c)[0],   (l)[1]); \
	op4s_dl4_mov_s_sse((c)[1],   (l)[2]); \
	op4s_dl4_mov_s_sse((c)[2],   (l)[3]); \
	op4s_dl4_mov_s_sse((c)[3],   (in)[0]); \
	op4s_dl4_mov_s_sse((r)[3],   (in)[1]); \
	op4s_dl4_mov_s_sse((r)[2],   (l)[3]); op4s_dl4_op3_s_sse((r)[2],   (c)[3], (w)[3], (r)[3]); \
	op4s_dl4_mov_s_sse((r)[1],   (l)[2]); op4s_dl4_op3_s_sse((r)[1],   (c)[2], (w)[2], (r)[2]); \
	op4s_dl4_mov_s_sse((r)[0],   (l)[1]); op4s_dl4_op3_s_sse((r)[0],   (c)[1], (w)[1], (r)[1]); \
	op4s_dl4_mov_s_sse((out)[1], (l)[0]); op4s_dl4_op3_s_sse((out)[1], (c)[0], (w)[0], (r)[0]); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_output_s_sse(addr, out) \
do { \
	*(addr) = (out); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_output_s_sse4(addr, out) \
do { \
	op4s_dl4_output_s_sse((addr) + 0, (out)[0]); \
	op4s_dl4_output_s_sse((addr) + 1, (out)[1]); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_update_s_sse(l, r) \
do { \
	(l) = (r); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_update_s_sse4(l, r) \
do { \
	op4s_dl4_update_s_sse((l)[0], (r)[0]); \
	op4s_dl4_update_s_sse((l)[1], (r)[1]); \
	op4s_dl4_update_s_sse((l)[2], (r)[2]); \
	op4s_dl4_update_s_sse((l)[3], (r)[3]); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_export_s_sse(addr, l) \
do { \
	*(addr) = (l); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_export_s_sse4(addr, l) \
do { \
	op4s_dl4_export_s_sse((addr) + 0, (l)[0]); \
	op4s_dl4_export_s_sse((addr) + 1, (l)[1]); \
	op4s_dl4_export_s_sse((addr) + 2, (l)[2]); \
	op4s_dl4_export_s_sse((addr) + 3, (l)[3]); \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_postsave_s_sse(temp, s, addr, idx) \
do { \
	(temp)[(idx)][(s)] = (*(addr))[(idx)]; \
} while(0)
#endif

#ifdef __SSE__
#define op4s_dl4_postsave_s_sse4(temp, s, addr) \
do { \
	op4s_dl4_postsave_s_sse((temp), (s), (addr), 0); \
	op4s_dl4_postsave_s_sse((temp), (s), (addr), 1); \
	op4s_dl4_postsave_s_sse((temp), (s), (addr), 2); \
	op4s_dl4_postsave_s_sse((temp), (s), (addr), 3); \
} while(0)
#endif

#ifdef __SSE__
/**
 * double-loop algorithm for 4 rows in parallel using SSE
 */
static
void accel_lift_op4s_main_dl4_sse_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( steps >= 0 );

	if( scaling < 0 )
	{
		assert( 4 == dwt_util_get_num_workers() );

		// constants
		const __m128 w[4] = {
			{ delta, delta, delta, delta },
			{ gamma, gamma, gamma, gamma },
			{ beta,  beta,  beta,  beta },
			{ alpha, alpha, alpha, alpha }
		};
		const __m128 v[2] = {
			{ 1/zeta, 1/zeta, 1/zeta, 1/zeta },
			{   zeta,   zeta,   zeta,   zeta }
		};

		// aux. variables
		__m128 in[2];
		__m128 out[2];
		__m128 r[4];
		__m128 c[4];
		__m128 l[4];

		// pointers
		float *arr_local[4];

		// init
		for(int worker = 0; worker < 4; worker++)
		{
			arr_local[worker] = calc_temp_offset_s(arr, worker);
		}

		// buffer
		const int buff_size = 4 + 2*steps;
		__m128 buff[buff_size]; // FIXME(x86): huge array on the stack
		__m128 *buff_ptr = buff;

		// load buffer
		assert( is_aligned_16(arr_local[0]) && is_aligned_16(arr_local[1]) && is_aligned_16(arr_local[2]) && is_aligned_16(arr_local[3]) );

		const int t4 = buff_size >> 2;
		const int t3 = buff_size & ~3;

		for(int t = 0; t < t4; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			__m128 s0 = _mm_load_ps(&arr_local[0][4*t]);
			__m128 s1 = _mm_load_ps(&arr_local[1][4*t]);
			__m128 s2 = _mm_load_ps(&arr_local[2][4*t]);
			__m128 s3 = _mm_load_ps(&arr_local[3][4*t]);

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			buff[4*t+0] = s0;
			buff[4*t+1] = s1;
			buff[4*t+2] = s2;
			buff[4*t+3] = s3;
		}

		for(int t = t3; t < buff_size; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			buff[t][0] = arr_local[0][t];
			buff[t][1] = arr_local[1][t];
			buff[t][2] = arr_local[2][t];
			buff[t][3] = arr_local[3][t];
		}

		// import
		op4s_dl4_import_s_sse4(l, buff_ptr);

		// pointers
		buff_ptr += 4;

		// loop
		for(int s = 0; s < steps; s++)
		{
			// inputs
			op4s_dl4_input_s_sse4(in, buff_ptr);

			// scales
			op4s_dl4_scale_s_sse4(in, v);

			// shuffles + operation
			op4s_dl4_op_s_sse4(in, out, w, c, l, r);

			// outputs
			op4s_dl4_output_s_sse4(buff_ptr-4, out);

			// update
			op4s_dl4_update_s_sse4(l, r);

			// pointers
			buff_ptr += 2;
		}

		// export
		op4s_dl4_export_s_sse4(buff_ptr-4, l);

		// store buffer
		for(int t = 0; t < t4; t++)
		{
			__m128 s0 = buff[4*t+0];
			__m128 s1 = buff[4*t+1];
			__m128 s2 = buff[4*t+2];
			__m128 s3 = buff[4*t+3];

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			_mm_store_ps(&arr_local[0][4*t], s0);
			_mm_store_ps(&arr_local[1][4*t], s1);
			_mm_store_ps(&arr_local[2][4*t], s2);
			_mm_store_ps(&arr_local[3][4*t], s3);
		}

		for(int t = t3; t < buff_size; t++)
		{
			arr_local[0][t] = buff[t][0];
			arr_local[1][t] = buff[t][1];
			arr_local[2][t] = buff[t][2];
			arr_local[3][t] = buff[t][3];
		}
	}
	else if ( scaling > 0 )
	{
		assert( 4 == dwt_util_get_num_workers() );

		// constants
		const __m128 w[4] = {
			{ delta, delta, delta, delta },
			{ gamma, gamma, gamma, gamma },
			{ beta,  beta,  beta,  beta },
			{ alpha, alpha, alpha, alpha }
		};
		const __m128 v[2] = {
			{ 1/zeta, 1/zeta, 1/zeta, 1/zeta },
			{   zeta,   zeta,   zeta,   zeta }
		};

		// aux. variables
		__m128 in[2];
		__m128 out[2];
		__m128 r[4];
		__m128 c[4];
		__m128 l[4];

		// pointers
		float *arr_local[4];

		// init
		for(int worker = 0; worker < 4; worker++)
		{
			arr_local[worker] = calc_temp_offset_s(arr, worker);
		}

		// buffer
		const int buff_size = 4 + 2*steps;
		__m128 buff[buff_size]; // FIXME(x86): huge array on the stack
		__m128 *buff_ptr = buff;

		// load buffer
		assert( is_aligned_16(arr_local[0]) && is_aligned_16(arr_local[1]) && is_aligned_16(arr_local[2]) && is_aligned_16(arr_local[3]) );

		const int t4 = buff_size >> 2;
		const int t3 = buff_size & ~3;

		for(int t = 0; t < t4; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			__m128 s0 = _mm_load_ps(&arr_local[0][4*t]);
			__m128 s1 = _mm_load_ps(&arr_local[1][4*t]);
			__m128 s2 = _mm_load_ps(&arr_local[2][4*t]);
			__m128 s3 = _mm_load_ps(&arr_local[3][4*t]);

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			buff[4*t+0] = s0;
			buff[4*t+1] = s1;
			buff[4*t+2] = s2;
			buff[4*t+3] = s3;
		}

		for(int t = t3; t < buff_size; t++)
		{
			// FIXME(x86): access to image data directly instead of storing it into temp[]
			buff[t][0] = arr_local[0][t];
			buff[t][1] = arr_local[1][t];
			buff[t][2] = arr_local[2][t];
			buff[t][3] = arr_local[3][t];
		}

		// import
		op4s_dl4_import_s_sse4(l, buff_ptr);

		// pointers
		buff_ptr += 4;

		// loop
		for(int s = 0; s < steps; s++)
		{
			// inputs
			op4s_dl4_input_s_sse4(in, buff_ptr);

			// shuffles + operation
			op4s_dl4_op_s_sse4(in, out, w, c, l, r);

			// descales
			op4s_dl4_descale_s_sse4(out, v);

			// outputs
			op4s_dl4_output_s_sse4(buff_ptr-4, out);

			// update
			op4s_dl4_update_s_sse4(l, r);

			// pointers
			buff_ptr += 2;
		}

		// export
		op4s_dl4_export_s_sse4(buff_ptr-4, l);

		// store buffer
		for(int t = 0; t < t4; t++)
		{
			__m128 s0 = buff[4*t+0];
			__m128 s1 = buff[4*t+1];
			__m128 s2 = buff[4*t+2];
			__m128 s3 = buff[4*t+3];

			_MM_TRANSPOSE4_PS(s0, s1, s2, s3);

			_mm_store_ps(&arr_local[0][4*t], s0);
			_mm_store_ps(&arr_local[1][4*t], s1);
			_mm_store_ps(&arr_local[2][4*t], s2);
			_mm_store_ps(&arr_local[3][4*t], s3);
		}

		for(int t = t3; t < buff_size; t++)
		{
			arr_local[0][t] = buff[t][0];
			arr_local[1][t] = buff[t][1];
			arr_local[2][t] = buff[t][2];
			arr_local[3][t] = buff[t][3];
		}
	}
	else
	{
		// fallback, not implemented
		accel_lift_op4s_main_s(arr, steps, alpha, beta, gamma, delta, zeta, scaling);
	}
}
#endif

int dwt_util_is_aligned_16(
	const void *ptr)
{
	return is_aligned_16(ptr);
}

int dwt_util_is_aligned_8(
	const void *ptr)
{
	return is_aligned_8(ptr);
}

int dwt_util_is_aligned_4(
	const void *ptr)
{
	return is_aligned_4(ptr);
}

/**
 * @brief Accelerated PicoBlaze operation.
 *
 * Two pairs (predict and update) of lifting steps and coefficients scaling
 * merged together. This function is accelerated on ASVP/EdkDSP.
 */
static
void accel_lift_op4s_main_pb_s(
	float *addr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	FUNC_BEGIN;

	assert( steps >= 0 );

#ifdef microblaze
	UNUSED(scaling);
	UNUSED(alpha);
	UNUSED(beta);
	UNUSED(gamma);
	UNUSED(delta);
	UNUSED(zeta);

	assert( steps <= (BANK_SIZE - 4) / 2 );

	const int size = 2*steps + 4;

	assert( is_aligned_8(addr) );
	assert( is_even(size) );

	for(int w = 0; w < get_active_workers(); w++)
	{
		// FIXME(ASVP): channel w according to worker ID; but each worker has independent DMA channels, thus this is not necessary
		const uint8_t ch = w;
		float *addr_local = calc_temp_offset_s(addr, w);
		
		assert( is_aligned_8(addr_local) );

		WAL_CHECK( wal_dma_configure(worker[w], ch, addr_local, 0, WAL_BCE_JSY_DMEM_A, WAL_BANK_POS(0), size) );
		WAL_CHECK( wal_dma_start(worker[w], ch, WAL_DMA_REQ_RD) );
	}

	for(int w = 0; w < get_active_workers(); w++)
	{
		// HACK(ASVP): wait for completing memory transfers on all 8 channels; but each worker has independent DMA channels
		while( wal_dma_isbusy(worker[w], /*WAL_DMA_MASK(ch)*/ 0x0f) )
			;
	}

	const uint32_t steps_32 = (uint32_t)steps;

	// start BCE computations
	for(int w = 0; w < get_active_workers(); w++)
	{
		WAL_CHECK( wal_mb2cmem(worker[w], WAL_CMEM_MB2PB, 0x01, &steps_32, 1) );

		WAL_CHECK( wal_mb2pb(worker[w], 1) );
	}

	// wait for finishing every BCE computation
	for(int w = 0; w < get_active_workers(); w++)
	{
		WAL_CHECK( wal_pb2mb(worker[w], NULL) );
	}

	assert( is_aligned_8(addr) );
	assert( is_even(size) );

	for(int w = 0; w < get_active_workers(); w++)
	{
		const uint8_t ch = w;
		float *addr_local = calc_temp_offset_s(addr, w);

		assert( is_aligned_8(addr_local) );

		WAL_CHECK( wal_dma_configure(worker[w], ch, addr_local, 0, WAL_BCE_JSY_DMEM_C, WAL_BANK_POS(0), size) );
		WAL_CHECK( wal_dma_start(worker[w], ch, WAL_DMA_REQ_WR) );
	}

	for(int w = 0; w < get_active_workers(); w++)
	{
		while( wal_dma_isbusy(worker[w], /*WAL_DMA_MASK(ch)*/ 0x0f) )
			;
	}

	for(int w = 0; w < get_active_workers(); w++)
	{
		float *addr_local = calc_temp_offset_s(addr, w);

		flush_cache_s(addr_local-1, size); // HACK(ASVP): why -1?
	}
#else /* microblaze */
	// fallback
	accel_lift_op4s_main_s(addr, steps, alpha, beta, gamma, delta, zeta, scaling);
#endif /* microblaze */

	FUNC_END;
}

static
void accel_lift_op4s_prolog_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( N-off >= 4 );

#ifdef NDEBUG
	UNUSED(N);
#endif

	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *arr_local = calc_temp_offset_s(arr, w);
		
		if(off)
		{
			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			arr_local[1] += alpha*(arr_local[0]+arr_local[2]);
			arr_local[3] += alpha*(arr_local[2]+arr_local[4]);

			// beta
			arr_local[0] += 2*beta*(arr_local[1]);
			arr_local[2] += beta*(arr_local[1]+arr_local[3]);
		
			// gamma
			arr_local[1] += gamma*(arr_local[0]+arr_local[2]);
		
			// delta
			arr_local[0] += 2*delta*(arr_local[1]);

			// scaling
			if( scaling > 0)
			{
				arr_local[0] *= zeta;
			}
		}
		else
		{
			// inv-scaling
			if( scaling < 0 )
			{
				arr_local[0] *= 1/zeta;
				arr_local[1] *= zeta;
				arr_local[2] *= 1/zeta;
				arr_local[3] *= zeta;
			}

			// alpha
			arr_local[0] += 2*alpha*(arr_local[1]);
			arr_local[2] += alpha*(arr_local[1]+arr_local[3]);
			
			// beta
			arr_local[1] += beta*(arr_local[0]+arr_local[2]);
		
			// gamma
			arr_local[0] += 2*gamma*(arr_local[1]);
		
			// delta
			// none

			// scaling
			// none
		}
	}
}

static
void accel_lift_op4s_epilog_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( N-off >= 4 );

	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *arr_local = calc_temp_offset_s(arr, w);

		if( is_even(N-off) )
		{
			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			// none

			// beta
			arr_local[N-1] += 2*beta*(arr_local[N-2]);

			// gamma
			arr_local[N-2] += gamma*(arr_local[N-1]+arr_local[N-3]);

			// delta
			arr_local[N-1] += 2*delta*(arr_local[N-2]);
			arr_local[N-3] += delta*(arr_local[N-4]+arr_local[N-2]);

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				arr_local[N-4] *= 1/zeta;
				arr_local[N-3] *= zeta;
				arr_local[N-2] *= 1/zeta;
				arr_local[N-1] *= zeta;
			}
		}
		else /* is_odd(N-off) */
		{
			// inv-scaling
			if( scaling < 0 )
			{
				arr_local[N-1] *= 1/zeta;
			}

			// alpha
			arr_local[N-1] += 2*alpha*(arr_local[N-2]);

			// beta
			arr_local[N-2] += beta*(arr_local[N-1]+arr_local[N-3]);

			// gamma
			arr_local[N-1] += 2*gamma*(arr_local[N-2]);
			arr_local[N-3] += gamma*(arr_local[N-2]+arr_local[N-4]);

			// delta
			arr_local[N-2] += delta*(arr_local[N-1]+arr_local[N-3]);
			arr_local[N-4] += delta*(arr_local[N-5]+arr_local[N-3]);

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				arr_local[N-5] *= 1/zeta;
				arr_local[N-4] *= zeta;
				arr_local[N-3] *= 1/zeta;
				arr_local[N-2] *= zeta;
				arr_local[N-1] *= 1/zeta;
			}
		}
	}
}

/**
 * @brief Prolog and epilog for N-off < 4.
 */
static
void accel_lift_op4s_short_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	assert( N-off < 4 );

	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *arr_local = calc_temp_offset_s(arr, w);

		if(off)
		{
			if( N == 2 )
			{
				// inv-scaling
				if( scaling < 0 )
				{
					// TODO
				}

				// alpha
				arr_local[1] += 2*alpha*(arr_local[0]);

				// beta
				arr_local[0] += 2*beta*(arr_local[1]);

				// gamma
				arr_local[1] += 2*gamma*(arr_local[0]);

				// delta
				arr_local[0] += 2*delta*(arr_local[1]);

				// scaling
				if( scaling > 0 )
				{
					arr_local[0] *= zeta;
					arr_local[1] *= 1/zeta;
				}
			}
			else
			if( N == 3 )
			{
				// inv-scaling
				if( scaling < 0 )
				{
					// TODO
				}

				// alpha
				arr_local[1] += alpha*(arr_local[0]+arr_local[2]);

				// beta
				arr_local[0] += 2*beta*(arr_local[1]);
				arr_local[2] += 2*beta*(arr_local[1]);

				// gamma
				arr_local[1] += gamma*(arr_local[0]+arr_local[2]);

				// delta
				arr_local[0] += 2*delta*(arr_local[1]);
				arr_local[2] += 2*delta*(arr_local[1]);

				// scaling
				if( scaling > 0 )
				{
					arr_local[0] *= zeta;
					arr_local[1] *= 1/zeta;
					arr_local[2] *= zeta;
				}
			}
			else /* N == 4 */
			{
				// inv-scaling
				if( scaling < 0 )
				{
					// TODO
				}

				// alpha
				arr_local[1] += alpha*(arr_local[0]+arr_local[2]);
				arr_local[3] += 2*alpha*(arr_local[2]);

				// beta
				arr_local[0] += 2*beta*(arr_local[1]);
				arr_local[2] += beta*(arr_local[1]+arr_local[3]);

				// gamma
				arr_local[1] += gamma*(arr_local[0]+arr_local[2]);
				arr_local[3] += 2*gamma*(arr_local[2]);

				// delta
				arr_local[0] += 2*delta*(arr_local[1]);
				arr_local[2] += delta*(arr_local[1]+arr_local[3]);

				// scaling
				if( scaling > 0 )
				{
					arr_local[0] *= zeta;
					arr_local[1] *= 1/zeta;
					arr_local[2] *= zeta;
					arr_local[3] *= 1/zeta;
				}
			}
		}
		else /* !off */
		{
			if( N == 2 )
			{
				// inv-scaling
				if( scaling < 0 )
				{
					arr_local[0] *= 1/zeta;
					arr_local[1] *= zeta;
				}

				// alpha
				arr_local[0] += 2*alpha*(arr_local[1]);

				// beta
				arr_local[1] += 2*beta*(arr_local[0]);

				// gamma
				arr_local[0] += 2*gamma*(arr_local[1]);

				// delta
				arr_local[1] += 2*delta*(arr_local[0]);

				// scaling
				if( scaling > 0 )
				{
					// TODO
				}
			}
			else /* N == 3 */
			{
				// inv-scaling
				if( scaling < 0 )
				{
					arr_local[0] *= 1/zeta;
					arr_local[1] *= zeta;
					arr_local[2] *= 1/zeta;
				}

				// alpha
				arr_local[0] += 2*alpha*(arr_local[1]);
				arr_local[2] += 2*alpha*(arr_local[1]);

				// beta
				arr_local[1] += beta*(arr_local[0]+arr_local[2]);

				// gamma
				arr_local[0] += 2*gamma*(arr_local[1]);
				arr_local[2] += 2*gamma*(arr_local[1]);

				// delta
				arr_local[1] += delta*(arr_local[0]+arr_local[2]);

				// scaling
				if( scaling > 0 )
				{
					// TODO
				}
			}
		}
	}
}

static
void accel_lift_op4s_s(
	float *restrict arr,
	int off,
	int len,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling)
{
	FUNC_BEGIN;

	assert( len >= 2 );
	assert( 0 == off || 1 == off );

	if( len-off < 4 )
	{
		accel_lift_op4s_short_s(arr, off, len, alpha, beta, gamma, delta, zeta, scaling);
	}
	else
	{
		accel_lift_op4s_prolog_s(arr, off, len, alpha, beta, gamma, delta, zeta, scaling);

		// FIXME: with GCC use (un)likely, i.e. __builtin_expect
		if(1 == get_accel_type())
		{
			const int max_inner_len = to_even(BANK_SIZE) - 4;
			const int inner_len = to_even(len-off) - 4;
			const int blocks = inner_len / max_inner_len;

			// full length blocks
			for(int b = 0; b < blocks; b++)
			{
				const int left = off + b * max_inner_len;
				const int steps = max_inner_len/2;

				accel_lift_op4s_main_pb_s(&arr[left], steps, alpha, beta, gamma, delta, zeta, scaling);
			}

			// last block
			if( blocks*max_inner_len < inner_len )
			{
				const int left = off + blocks * max_inner_len;
				const int steps = (off + inner_len - left)/2;

				// TODO(ASVP): here should be a test if last block should be accelerated on PicoBlaze or rather computed on MicroBlaze
				if( steps > 25 )
					accel_lift_op4s_main_pb_s(&arr[left], steps, alpha, beta, gamma, delta, zeta, scaling);
				else
					accel_lift_op4s_main_s(&arr[left], steps, alpha, beta, gamma, delta, zeta, scaling);
			}
		}
		else if(0 == get_accel_type())
		{
			accel_lift_op4s_main_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(2 == get_accel_type())
		{
			// empty
		}
		else if(3 == get_accel_type())
		{
			off = 0;
			accel_lift_op4s_main_pb_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(4 == get_accel_type())
		{
			accel_lift_op4s_main_dl_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(5 == get_accel_type())
		{
			const int steps = (to_even(len-off)-4)/2;

			if( steps < 3 )
				accel_lift_op4s_main_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
			else
				accel_lift_op4s_main_sdl_ref_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(6 == get_accel_type())
		{
			const int steps = (to_even(len-off)-4)/2;

			if( steps < 3 )
				accel_lift_op4s_main_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
			else
				accel_lift_op4s_main_sdl2_ref_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(7 == get_accel_type())
		{
			const int steps = (to_even(len-off)-4)/2;

			if( steps < 3 )
				accel_lift_op4s_main_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
			else
				accel_lift_op4s_main_sdl6_ref_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(8 == get_accel_type())
		{
			const int steps = (to_even(len-off)-4)/2;

			if( steps < 3 )
				accel_lift_op4s_main_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
			else
#ifdef __SSE__
				accel_lift_op4s_main_sdl2_sse_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
#else
				accel_lift_op4s_main_sdl2_ref_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
#endif
		}
		else if(9 == get_accel_type())
		{
			const int steps = (to_even(len-off)-4)/2;

			if( steps < 3 )
				accel_lift_op4s_main_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
			else
			{
#ifdef __SSE__
				accel_lift_op4s_main_sdl6_sse_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
#else
				accel_lift_op4s_main_sdl6_ref_s(arr+off, steps, alpha, beta, gamma, delta, zeta, scaling);
#endif
			}
		}
		else if(10 == get_accel_type())
		{
			// FIXME: this needs to be threated inside of caller
			if( 4 != dwt_util_get_num_workers() )
				accel_lift_op4s_main_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
			else
				accel_lift_op4s_main_dl4_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(11 == get_accel_type())
		{
			// FIXME: this needs to be threated inside of caller
			if( 4 != dwt_util_get_num_workers() )
				accel_lift_op4s_main_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
			else
#ifdef __SSE__
				accel_lift_op4s_main_dl4_sse_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
#else
				accel_lift_op4s_main_dl4_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
#endif
		}
		else if(12 == get_accel_type())
		{
			// FIXME: this needs to be threated inside of caller
			if( 4 != dwt_util_get_num_workers() )
			{
				accel_lift_op4s_main_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
			}
			else
			{
#ifdef __SSE__
				accel_lift_op4s_main_ml4_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
#else
				accel_lift_op4s_main_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
#endif
			}
		}
		else if(13 == get_accel_type())
		{
			accel_lift_op4s_main_nosse_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(14 == get_accel_type())
		{
			accel_lift_op4s_main_dl_nosse_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(15 == get_accel_type())
		{
			accel_lift_op4s_main_dl4line_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
		}
		else if(16 == get_accel_type())
		{
#ifdef __SSE__
			accel_lift_op4s_main_dl4line_sse_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
#else
			accel_lift_op4s_main_dl4line_s(arr+off, (to_even(len-off)-4)/2, alpha, beta, gamma, delta, zeta, scaling);
#endif
		}
		else
		{
			dwt_util_log(LOG_ERR, "Unsupported value of acceleration.\n");
			dwt_util_abort();
		}

		accel_lift_op4s_epilog_s(arr, off, len, alpha, beta, gamma, delta, zeta, scaling);
	}

	FUNC_END;
}

void dwt_cdf97_f_ex_stride_s(
	const float *src,
	float *dst_l,
	float *dst_h,
	float *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src && NULL != dst_l && NULL != dst_h && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst_l[0] = src[0] * dwt_cdf97_s1_s;
		return;
	}

	// copy src into tmp
	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *tmp_local = calc_temp_offset_s(tmp, w);
		const float *src_local = calc_data_offset_const_s(src, w);

#ifndef DISABLE_MEMCPY
		dwt_util_memcpy_stride_s(tmp_local, sizeof(float), src_local, stride, N);
#endif
#ifdef ENABLE_LAZY_MEMCPY
		// FIXME: copy only if column is processed
		if( sizeof(float) != stride )
			dwt_util_memcpy_stride_s(tmp_local, sizeof(float), src_local, stride, N);
#endif
	}

	accel_lift_op4s_s(tmp, 1, N, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1);

	// copy tmp into dst
	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *tmp_local = calc_temp_offset_s(tmp, w);
		float *dst_l_local = calc_data_offset_s(dst_l, w);
		float *dst_h_local = calc_data_offset_s(dst_h, w);

#ifndef DISABLE_MEMCPY
		dwt_util_memcpy_stride_s(dst_l_local, stride, tmp_local+0, 2*sizeof(float),  ceil_div2(N));
		dwt_util_memcpy_stride_s(dst_h_local, stride, tmp_local+1, 2*sizeof(float), floor_div2(N));
#endif
#ifdef ENABLE_LAZY_MEMCPY
		// FIXME: copy only if column is processed; keep L and H subbands interleaved
		const float *src_local = calc_data_offset_const_s(src, w);
		if( sizeof(float) != stride )
			dwt_util_memcpy_stride_s(src_local, stride, tmp_local, sizeof(float), N);
#endif
	}
}

/*
 * CDF 9/7 (4,4)
 * http://www.ece.uvic.ca/~frodo/publications/phdthesis.pdf
 * * 9/7-F, p. 88, 89
 * * [24] M. Antonini, M. Barlaud, P. Mathieu, and I. Daubechies. Image coding using wavelet transform. IEEE Trans. on Image Processing, 1(2):205–220, April 1992.
 * * [40] A. R. Calderbank, I. Daubechies,W. Sweldens, and B.-L. Yeo. Wavelet transforms that map integers to integers. Applied and Computational Harmonic Analysis, 5(3):332–369, July 1998.
 */
void dwt_cdf97_f_ex_stride_i(
	const int *src,
	int *dst_l,
	int *dst_h,
	int *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src && NULL != dst_l && NULL != dst_h && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
		return;

	// copy src into tmp
	dwt_util_memcpy_stride_i(tmp, sizeof(int), src, stride, N);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] -= ( +203*(tmp[i-1]+tmp[i+1]) - (1<<6) ) >> 7;

	if(is_odd(N))
		tmp[N-1] += ( -217*(tmp[N-2]+tmp[N-2]) + (1<<11) ) >> 12;
	else
		tmp[N-1] -= ( +203*(tmp[N-2]+tmp[N-2]) - (1<<6) ) >> 7;
	tmp[0] += ( -217*(tmp[1]+tmp[1]) + (1<<11) ) >> 12;

	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] += ( -217*(tmp[i-1]+tmp[i+1]) + (1<<11) ) >> 12;

	// predict 2 + update 2
	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] -= ( -113*(tmp[i-1]+tmp[i+1]) - (1<<6) ) >> 7;

	if(is_odd(N))
		tmp[N-1] += ( 1817*(tmp[N-2]+tmp[N-2]) + (1<<11) ) >> 12;
	else
		tmp[N-1] -= ( -113*(tmp[N-2]+tmp[N-2]) - (1<<6) ) >> 7;
	tmp[0] += ( 1817*(tmp[1]+tmp[1]) + (1<<11) ) >> 12;

	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] += ( 1817*(tmp[i-1]+tmp[i+1]) + (1<<11) ) >> 12;

	// copy tmp into dst
	dwt_util_memcpy_stride_i(dst_l, stride, tmp+0, 2*sizeof(int),  ceil_div2(N));
	dwt_util_memcpy_stride_i(dst_h, stride, tmp+1, 2*sizeof(int), floor_div2(N));
}

// http://www.ece.uvic.ca/~frodo/publications/phdthesis.pdf
void dwt_cdf53_f_ex_stride_i(
	const int *src,
	int *dst_l,
	int *dst_h,
	int *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src && NULL != dst_l && NULL != dst_h && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
		return;

	// copy src into tmp
	dwt_util_memcpy_stride_i(tmp, sizeof(int), src, stride, N);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] -= (tmp[i-1] + tmp[i+1]) >> 1;

	if(is_odd(N))
		tmp[N-1] += (tmp[N-2] + 1) >> 1;
	else
		tmp[N-1] -= tmp[N-2];

	tmp[0] += (tmp[1] + 1) >> 1;

	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] += ( (tmp[i-1] + tmp[i+1]) + 2 ) >> 2;

	// copy tmp into dst
	dwt_util_memcpy_stride_i(dst_l, stride, tmp+0, 2*sizeof(int),  ceil_div2(N));
	dwt_util_memcpy_stride_i(dst_h, stride, tmp+1, 2*sizeof(int), floor_div2(N));
}

void dwt_cdf53_f_ex_stride_s(
	const float *src,
	float *dst_l,
	float *dst_h,
	float *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src && NULL != dst_l && NULL != dst_h && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst_l[0] = src[0] * dwt_cdf53_s1_s;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_s(tmp, sizeof(float), src, stride, N);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] -= dwt_cdf53_p1_s * (tmp[i-1] + tmp[i+1]);

	if(is_odd(N))
		tmp[N-1] += 2 * dwt_cdf53_u1_s * tmp[N-2];
	else
		tmp[N-1] -= 2 * dwt_cdf53_p1_s * tmp[N-2];

	tmp[0] += 2 * dwt_cdf53_u1_s * tmp[1];

	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] += dwt_cdf53_u1_s * (tmp[i-1] + tmp[i+1]);

	// scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s1_s;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s2_s;

	// copy tmp into dst
	dwt_util_memcpy_stride_s(dst_l, stride, tmp+0, 2*sizeof(float),  ceil_div2(N));
	dwt_util_memcpy_stride_s(dst_h, stride, tmp+1, 2*sizeof(float), floor_div2(N));
}

void dwt_cdf97_i_ex_d(
	const double *src_l,
	const double *src_h,
	double *dst,
	double *tmp,
	int N)
{
	dwt_cdf97_i_ex_stride_d(
		src_l,
		src_h,
		dst,
		tmp,
		N,
		sizeof(double)
	);
}

void dwt_cdf53_i_ex_d(
	const double *src_l,
	const double *src_h,
	double *dst,
	double *tmp,
	int N)
{
	dwt_cdf53_i_ex_stride_d(
		src_l,
		src_h,
		dst,
		tmp,
		N,
		sizeof(double)
	);
}

void dwt_cdf97_i_ex_s(
	const float *src_l,
	const float *src_h,
	float *dst,
	float *tmp,
	int N)
{
	dwt_cdf97_i_ex_stride_s(
		src_l,
		src_h,
		dst,
		tmp,
		N,
		sizeof(float)
	);
}

void dwt_cdf53_i_ex_i(
	const int *src_l,
	const int *src_h,
	int *dst,
	int *tmp,
	int N)
{
	dwt_cdf53_i_ex_stride_i(
		src_l,
		src_h,
		dst,
		tmp,
		N,
		sizeof(int)
	);
}

void dwt_cdf53_i_ex_s(
	const float *src_l,
	const float *src_h,
	float *dst,
	float *tmp,
	int N)
{
	dwt_cdf53_i_ex_stride_s(
		src_l,
		src_h,
		dst,
		tmp,
		N,
		sizeof(float)
	);
}

void dwt_cdf97_i_ex_stride_d(
	const double *src_l,
	const double *src_h,
	double *dst,
	double *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src_l && NULL != src_h && NULL != dst && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst[0] = src_l[0] * dwt_cdf97_s2_d;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_d(tmp+0, 2*sizeof(double), src_l, stride,  ceil_div2(N));
	dwt_util_memcpy_stride_d(tmp+1, 2*sizeof(double), src_h, stride, floor_div2(N));

	// inverse scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s2_d;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf97_s1_d;

	// backward update 2 + backward predict 2
	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] -= dwt_cdf97_u2_d * (tmp[i-1] + tmp[i+1]);

	tmp[0] -= 2 * dwt_cdf97_u2_d * tmp[1];

	if(is_odd(N))
		tmp[N-1] -= 2 * dwt_cdf97_u2_d * tmp[N-2];
	else
		tmp[N-1] += 2 * dwt_cdf97_p2_d * tmp[N-2];

	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] += dwt_cdf97_p2_d * (tmp[i-1] + tmp[i+1]);

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] -= dwt_cdf97_u1_d * (tmp[i-1] + tmp[i+1]);

	tmp[0] -= 2 * dwt_cdf97_u1_d * tmp[1];

	if(is_odd(N))
		tmp[N-1] -= 2 * dwt_cdf97_u1_d * tmp[N-2];
	else
		tmp[N-1] += 2 * dwt_cdf97_p1_d * tmp[N-2];

	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] += dwt_cdf97_p1_d * (tmp[i-1] + tmp[i+1]);

	// copy tmp into dst
	dwt_util_memcpy_stride_d(dst, stride, tmp, sizeof(double), N);
}

void dwt_cdf53_i_ex_stride_d(
	const double *src_l,
	const double *src_h,
	double *dst,
	double *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src_l && NULL != src_h && NULL != dst && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst[0] = src_l[0] * dwt_cdf53_s2_d;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_d(tmp+0, 2*sizeof(double), src_l, stride,  ceil_div2(N));
	dwt_util_memcpy_stride_d(tmp+1, 2*sizeof(double), src_h, stride, floor_div2(N));

	// inverse scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s2_d;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s1_d;

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] -= dwt_cdf53_u1_d * (tmp[i-1] + tmp[i+1]);

	tmp[0] -= 2 * dwt_cdf53_u1_d * tmp[1];

	if(is_odd(N))
		tmp[N-1] -= 2 * dwt_cdf53_u1_d * tmp[N-2];
	else
		tmp[N-1] += 2 * dwt_cdf53_p1_d * tmp[N-2];

	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] += dwt_cdf53_p1_d * (tmp[i-1] + tmp[i+1]);

	// copy tmp into dst
	dwt_util_memcpy_stride_d(dst, stride, tmp, sizeof(double), N);
}

void dwt_cdf97_i_ex_stride_s(
	const float *src_l,
	const float *src_h,
	float *dst,
	float *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src_l && NULL != src_h && NULL != dst && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst[0] = src_l[0] * dwt_cdf97_s2_s; // FIXME: 1/zeta
		return;
	}

	// copy src into tmp
	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *tmp_local = calc_temp_offset_s(tmp, w);
		const float *src_l_local = calc_data_offset_const_s(src_l, w);
		const float *src_h_local = calc_data_offset_const_s(src_h, w);

		dwt_util_memcpy_stride_s(tmp_local+0, 2*sizeof(float), src_l_local, stride,  ceil_div2(N));
		dwt_util_memcpy_stride_s(tmp_local+1, 2*sizeof(float), src_h_local, stride, floor_div2(N));
	}

	accel_lift_op4s_s(tmp, 0, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1);

	// copy tmp into dst
	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *tmp_local = calc_temp_offset_s(tmp, w);
		float *dst_local = calc_data_offset_s(dst, w);

		dwt_util_memcpy_stride_s(dst_local, stride, tmp_local, sizeof(float), N);
	}
}

void dwt_cdf97_i_ex_stride_i(
	const int *src_l,
	const int *src_h,
	int *dst,
	int *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src_l && NULL != src_h && NULL != dst && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
		return;

	// copy src into tmp
	dwt_util_memcpy_stride_i(tmp+0, 2*sizeof(int), src_l, stride,  ceil_div2(N));
	dwt_util_memcpy_stride_i(tmp+1, 2*sizeof(int), src_h, stride, floor_div2(N));

	// backward update 2 + backward predict 2
	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] -= ( 1817*(tmp[i-1]+tmp[i+1]) + (1<<11) ) >> 12;

	tmp[0] -= ( 1817*(tmp[1]+tmp[1]) + (1<<11) ) >> 12;

	if(is_odd(N))
		tmp[N-1] -= ( 1817*(tmp[N-2]+tmp[N-2]) + (1<<11) ) >> 12;
	else
		tmp[N-1] += ( -113*(tmp[N-2]+tmp[N-2]) - (1<<6) ) >> 7;

	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] += ( -113*(tmp[i-1]+tmp[i+1]) - (1<<6) ) >> 7;

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] -= ( -217*(tmp[i-1]+tmp[i+1]) + (1<<11) ) >> 12;

	tmp[0] -= ( -217*(tmp[1]+tmp[1]) + (1<<11) ) >> 12;

	if(is_odd(N))
		tmp[N-1] -= ( -217*(tmp[N-2]+tmp[N-2]) + (1<<11) ) >> 12;
	else
		tmp[N-1] += ( +203*(tmp[N-2]+tmp[N-2]) - (1<<6) ) >> 7;

	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] += ( +203*(tmp[i-1]+tmp[i+1]) - (1<<6) ) >> 7;

	// copy tmp into dst
	dwt_util_memcpy_stride_i(dst, stride, tmp, sizeof(int), N);
}

void dwt_cdf53_i_ex_stride_i(
	const int *src_l,
	const int *src_h,
	int *dst,
	int *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src_l && NULL != src_h && NULL != dst && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
		return;

	// copy src into tmp
	dwt_util_memcpy_stride_i(tmp+0, 2*sizeof(int), src_l, stride,  ceil_div2(N));
	dwt_util_memcpy_stride_i(tmp+1, 2*sizeof(int), src_h, stride, floor_div2(N));

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] -= ( (tmp[i-1] + tmp[i+1]) + 2 ) >> 2;

	tmp[0] -= (tmp[1] + 1) >> 1;

	if(is_odd(N))
		tmp[N-1] -= (tmp[N-2] + 1) >> 1;
	else
		tmp[N-1] += tmp[N-2];

	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] += ( tmp[i-1] + tmp[i+1] ) >> 1;

	// copy tmp into dst
	dwt_util_memcpy_stride_i(dst, stride, tmp, sizeof(int), N);
}

void dwt_cdf53_i_ex_stride_s(
	const float *src_l,
	const float *src_h,
	float *dst,
	float *tmp,
	int N,
	int stride)
{
	assert( N >= 0 && NULL != src_l && NULL != src_h && NULL != dst && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			dst[0] = src_l[0] * dwt_cdf53_s2_s;
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_s(tmp+0, 2*sizeof(float), src_l, stride,  ceil_div2(N));
	dwt_util_memcpy_stride_s(tmp+1, 2*sizeof(float), src_h, stride, floor_div2(N));

	// inverse scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s2_s;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s1_s;

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		tmp[i] -= dwt_cdf53_u1_s * (tmp[i-1] + tmp[i+1]);

	tmp[0] -= 2 * dwt_cdf53_u1_s * tmp[1];

	if(is_odd(N))
		tmp[N-1] -= 2 * dwt_cdf53_u1_s * tmp[N-2];
	else
		tmp[N-1] += 2 * dwt_cdf53_p1_s * tmp[N-2];

	for(int i=1; i<N-2+(N&1); i+=2)
		tmp[i] += dwt_cdf53_p1_s * (tmp[i-1] + tmp[i+1]);

	// copy tmp into dst
	dwt_util_memcpy_stride_s(dst, stride, tmp, sizeof(float), N);
}

void dwt_zero_padding_f_d(
	double *dst_l,
	double *dst_h,
	int N,
	int N_dst_L,
	int N_dst_H)
{
	dwt_zero_padding_f_stride_d(
		dst_l,
		dst_h,
		N,
		N_dst_L,
		N_dst_H,
		sizeof(double)
	);
}

void dwt_zero_padding_f_s(
	float *dst_l,
	float *dst_h,
	int N,
	int N_dst_L,
	int N_dst_H)
{
	dwt_zero_padding_f_stride_s(
		dst_l,
		dst_h,
		N,
		N_dst_L,
		N_dst_H,
		sizeof(float)
	);
}

void dwt_zero_padding_f_stride_d(
	double *dst_l,
	double *dst_h,
	int N,
	int N_dst_L,
	int N_dst_H,
	int stride)
{
	assert( N >= 0 && N_dst_L >= 0 && N_dst_H >= 0 && 0 == ((N_dst_L-N_dst_H)&~1) && NULL != dst_l && NULL != dst_h && 0 != stride ); // FIXME: 0 == ((N_dst_L-N_dst_H)&~1)

	if(N_dst_L || N_dst_H)
	{
		const double zero = 0;

		dwt_util_memcpy_stride_d(addr1_d(dst_l,  ceil_div2(N), stride), stride, &zero, 0, N_dst_L -  ceil_div2(N));
		dwt_util_memcpy_stride_d(addr1_d(dst_h, floor_div2(N), stride), stride, &zero, 0, N_dst_H - floor_div2(N));
	}
}

void dwt_zero_padding_f_stride_i(
	int *dst_l,
	int *dst_h,
	int N,
	int N_dst_L,
	int N_dst_H,
	int stride)
{
	assert( N >= 0 && N_dst_L >= 0 && N_dst_H >= 0 && 0 == ((N_dst_L-N_dst_H)&~1) && NULL != dst_l && NULL != dst_h && 0 != stride ); // FIXME: 0 == ((N_dst_L-N_dst_H)&~1)

	if(N_dst_L || N_dst_H)
	{
		const float zero = 0;

		dwt_util_memcpy_stride_i(addr1_i(dst_l,  ceil_div2(N), stride), stride, &zero, 0, N_dst_L -  ceil_div2(N));
		dwt_util_memcpy_stride_i(addr1_i(dst_h, floor_div2(N), stride), stride, &zero, 0, N_dst_H - floor_div2(N));
	}
}

void dwt_zero_padding_f_stride_s(
	float *dst_l,
	float *dst_h,
	int N,
	int N_dst_L,
	int N_dst_H,
	int stride)
{
	assert( N >= 0 && N_dst_L >= 0 && N_dst_H >= 0 && 0 == ((N_dst_L-N_dst_H)&~1) && NULL != dst_l && NULL != dst_h && 0 != stride ); // FIXME: 0 == ((N_dst_L-N_dst_H)&~1)

	if(N_dst_L || N_dst_H)
	{
		const float zero = 0;

		dwt_util_memcpy_stride_s(addr1_s(dst_l,  ceil_div2(N), stride), stride, &zero, 0, N_dst_L -  ceil_div2(N));
		dwt_util_memcpy_stride_s(addr1_s(dst_h, floor_div2(N), stride), stride, &zero, 0, N_dst_H - floor_div2(N));
	}
}

void dwt_zero_padding_i_d(
	double *dst_l,
	int N,
	int N_dst)
{
	dwt_zero_padding_i_stride_d(
		dst_l,
		N,
		N_dst,
		sizeof(double)
	);
}

void dwt_zero_padding_i_s(
	float *dst_l,
	int N,
	int N_dst)
{
	dwt_zero_padding_i_stride_s(
		dst_l,
		N,
		N_dst,
		sizeof(float)
	);
}

void dwt_zero_padding_i_stride_d(
	double *dst_l,
	int N,
	int N_dst,
	int stride)
{
	assert( N >= 0 && N_dst >= 0 && NULL != dst_l && 0 != stride );

	const double zero = 0;

	dwt_util_memcpy_stride_d(
		addr1_d(dst_l, N, stride),
		stride,
		&zero,
		0,
		N_dst - N);
}

void dwt_zero_padding_i_stride_i(
	int *dst_l,
	int N,
	int N_dst,
	int stride)
{
	assert( N >= 0 && N_dst >= 0 && NULL != dst_l && 0 != stride );

	const int zero = 0;

	dwt_util_memcpy_stride_i(
		addr1_i(dst_l, N, stride),
		stride,
		&zero,
		0,
		N_dst - N);
}

void dwt_zero_padding_i_stride_s(
	float *dst_l,
	int N,
	int N_dst,
	int stride)
{
	assert( N >= 0 && N_dst >= 0 && NULL != dst_l && 0 != stride );

	const float zero = 0;

	dwt_util_memcpy_stride_s(
		addr1_s(dst_l, N, stride),
		stride,
		&zero,
		0,
		N_dst - N);
}

void dwt_util_switch_op(
	enum dwt_op op)
{
	FUNC_BEGIN;

#ifdef microblaze
	if( op == dwt_util_global_active_op )
		return;

	//WAL_CHECK( wal_mb2pb(worker, 0) );

	//WAL_CHECK( wal_bce_jk_sync_operation(worker) );

	for(int w = 0; w < get_total_workers(); w++)
	{
		WAL_CHECK( wal_reset_worker(worker[w]) );
	}

	switch(op)
	{
		case DWT_OP_LIFT4SA:
		{
			for(int w = 0; w < get_total_workers(); w++)
			{
				WAL_CHECK( wal_start_operation(worker[w], WAL_PBID_P0) );
			}

			float alpha = -dwt_cdf97_p1_s,
				beta = dwt_cdf97_u1_s,
				gamma = -dwt_cdf97_p2_s,
				delta = dwt_cdf97_u2_s,
				zeta = dwt_cdf97_s1_s;

			const int size = 12;
			// FIXME(ASVP): for these coeeficients, use memory bank "D"
			const float coeffs[12] = { delta, 0.0f, gamma, 0.0f, beta, 0.0f, alpha, 0.0f, zeta, 0.0f, 1/zeta, 0.0f };
			float *addr = dwt_util_allocate_vec_s(size);
			if(!addr)
			{
				dwt_util_log(LOG_ERR, "Failed to allocate vector of %i floats.\n", size);
				dwt_util_abort();
			}
			if( dwt_util_copy_vec_s(coeffs, addr, size) )
				dwt_util_abort();

			assert( is_even(size) );
			assert( is_aligned_8(addr) );

			for(int w = 0; w < get_total_workers(); w++)
			{
				WAL_CHECK( wal_dma_configure(worker[w], 0, addr, 0, WAL_BCE_JSY_DMEM_B, 0, size) );
				WAL_CHECK( wal_dma_start(worker[w], 0, WAL_DMA_REQ_RD) );
				while( wal_dma_isbusy(worker[w], 0x01) )
					;
			}

			free(addr);
		}
		break;
		case DWT_OP_LIFT4SB:
		{
			for(int w = 0; w < get_total_workers(); w++)
			{
				WAL_CHECK( wal_start_operation(worker[w], WAL_PBID_P1) );
			}

			float alpha = -dwt_cdf97_u2_s,
				beta = dwt_cdf97_p2_s,
				gamma = -dwt_cdf97_u1_s,
				delta = dwt_cdf97_p1_s,
				zeta = dwt_cdf97_s1_s;

			const int size = 12;
			// FIXME(ASVP): for these coeeficients, use memory bank "D"
			const float coeffs[12] = { delta, 0.0f, gamma, 0.0f, beta, 0.0f, alpha, 0.0f, zeta, 0.0f, 1/zeta, 0.0f };
			float *addr = dwt_util_allocate_vec_s(size);
			if(!addr)
			{
				dwt_util_log(LOG_ERR, "Failed to allocate vector of %i floats.\n", size);
				dwt_util_abort();
			}
			if( dwt_util_copy_vec_s(coeffs, addr, size) )
				dwt_util_abort();

			assert( is_even(size) );
			assert( is_aligned_8(addr) );

			for(int w = 0; w < get_total_workers(); w++)
			{
				WAL_CHECK( wal_dma_configure(worker[w], 0, addr, 0, WAL_BCE_JSY_DMEM_B, 0, size) );
				WAL_CHECK( wal_dma_start(worker[w], 0, WAL_DMA_REQ_RD) );
				while( wal_dma_isbusy(worker[w], 0x01) )
					;
			}

			free(addr);
		}
		break;
		default:
		{
			dwt_util_log(LOG_ERR, "Unknown operation.\n");
	
			dwt_util_abort();
		}
	}

	dwt_util_global_active_op = op;
#else
	UNUSED(op);
#endif

	FUNC_END;
}

void dwt_cdf97_2f_d(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	double temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_src_y; y++)
			dwt_cdf97_f_ex_stride_d(
				addr2_d(ptr,y,0,stride_x,stride_y),
				addr2_d(ptr,y,0,stride_x,stride_y),
				addr2_d(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_cdf97_f_ex_stride_d(
				addr2_d(ptr,0,x,stride_x,stride_y),
				addr2_d(ptr,0,x,stride_x,stride_y),
				addr2_d(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp,
				size_i_src_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_d(
					addr2_d(ptr,y,0,stride_x,stride_y),
					addr2_d(ptr,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_d(
					addr2_d(ptr,0,x,stride_x,stride_y),
					addr2_d(ptr,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}
}

void dwt_cdf53_2f_d(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	double temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_src_y; y++)
			dwt_cdf53_f_ex_stride_d(
				addr2_d(ptr,y,0,stride_x,stride_y),
				addr2_d(ptr,y,0,stride_x,stride_y),
				addr2_d(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_cdf53_f_ex_stride_d(
				addr2_d(ptr,0,x,stride_x,stride_y),
				addr2_d(ptr,0,x,stride_x,stride_y),
				addr2_d(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp,
				size_i_src_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_d(
					addr2_d(ptr,y,0,stride_x,stride_y),
					addr2_d(ptr,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_d(
					addr2_d(ptr,0,x,stride_x,stride_y),
					addr2_d(ptr,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}
}

void dwt_cdf97_2f_s2(
	const void *src,
	void *dst,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding)
{
	FUNC_BEGIN;

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SA);
#endif
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	// FIXME: allocate temp[max_threads][temp_size] and remove private() in omp parallel
	// FIXME: OpenMP cannot use this in private()
#ifdef microblaze
	#define TEMP_OFFSET 1
	float *temp = dwt_util_allocate_vec_s(calc_and_set_temp_size(size_o_big_max));
#else /* microblaze */
	//#define TEMP_OFFSET 0
	// FIXME(x86) HACK: __attribute__ ((aligned (16)))
	#define TEMP_OFFSET 3

#ifdef TEMP_ON_HEAP
	float *temp = dwt_util_allocate_vec_s(calc_and_set_temp_size(size_o_big_max));
#else /* TEMP_ON_HEAP */
	// FIXME(x86) BUG: temp[] is allocated on stack! so stack overflow is caused by big sizes of temp[size]
	float temp[calc_and_set_temp_size(size_o_big_max)] __attribute__ ((aligned (16)));
#endif /* TEMP_ON_HEAP */

	if( !is_aligned_16(temp) )
		dwt_util_abort();
#endif /* microblaze */

	if(NULL == temp)
		dwt_util_abort();

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );
#ifdef _OPENMP
		const int threads = dwt_util_get_num_threads();
#endif
		const int workers = dwt_util_get_num_workers();
		
		const int lines_x = size_o_src_x;
		const int lines_y = size_o_src_y;

		const int workers_segment_y = floor_div(lines_y, workers);
		const int workers_segment_x = floor_div(lines_x, workers);
#ifdef _OPENMP
		const int threads_segment_y = ceil_div(workers_segment_y, threads);
		const int threads_segment_x = ceil_div(workers_segment_x, threads);
#endif
		const int workers_lines_y = workers_segment_y * workers;
		const int workers_lines_x = workers_segment_x * workers;

		set_data_step_s( stride_x );

		#pragma omp parallel for private(temp) schedule(static, threads_segment_y)
		for(int y = 0; y < workers_lines_y; y += workers)
		{
			dwt_cdf97_f_ex_stride_s(
				addr2_const_s(src,y,0,stride_x,stride_y),
				addr2_s(dst,y,0,stride_x,stride_y),
				addr2_s(dst,y,size_o_dst_x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_x,
				stride_y);
		}
		dwt_util_set_num_workers(1);
		for(int y = workers_lines_y; y < lines_y; y++)
		{
			dwt_cdf97_f_ex_stride_s(
				addr2_const_s(src,y,0,stride_x,stride_y),
				addr2_s(dst,y,0,stride_x,stride_y),
				addr2_s(dst,y,size_o_dst_x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_x,
				stride_y);
		}
		dwt_util_set_num_workers(workers);

		set_data_step_s( stride_y );

		#pragma omp parallel for private(temp) schedule(static, threads_segment_x)
		for(int x = 0; x < workers_lines_x; x += workers)
		{
			dwt_cdf97_f_ex_stride_s(
				addr2_s(dst,0,x,stride_x,stride_y),
				addr2_s(dst,0,x,stride_x,stride_y),
				addr2_s(dst,size_o_dst_y,x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_y,
				stride_x);
		}
		dwt_util_set_num_workers(1);
		for(int x = workers_lines_x; x < lines_x; x++)
		{
			dwt_cdf97_f_ex_stride_s(
				addr2_s(dst,0,x,stride_x,stride_y),
				addr2_s(dst,0,x,stride_x,stride_y),
				addr2_s(dst,size_o_dst_y,x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_y,
				stride_x);
		}
		dwt_util_set_num_workers(workers);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_s(
					addr2_s(dst,y,0,stride_x,stride_y),
					addr2_s(dst,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_s(
					addr2_s(dst,0,x,stride_x,stride_y),
					addr2_s(dst,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}

#undef TEMP_OFFSET

#ifdef microblaze
	free(temp);
#endif

#ifdef TEMP_ON_HEAP
	free(temp);
#endif

	FUNC_END;
}

void dwt_cdf97_2f_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding)
{
	FUNC_BEGIN;

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SA);
#endif
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	// FIXME: allocate temp[max_threads][temp_size] and remove private() in omp parallel
	// FIXME: OpenMP cannot use this in private()
#ifdef microblaze
	#define TEMP_OFFSET 1
	float *temp = dwt_util_allocate_vec_s(calc_and_set_temp_size(size_o_big_max));
#else /* microblaze */
	//#define TEMP_OFFSET 0
	// FIXME(x86) HACK: __attribute__ ((aligned (16)))
	#define TEMP_OFFSET 3

#ifdef TEMP_ON_HEAP
	float *temp = dwt_util_allocate_vec_s(calc_and_set_temp_size(size_o_big_max));
#else /* TEMP_ON_HEAP */
	// FIXME(x86) BUG: temp[] is allocated on stack! so stack overflow is caused by big sizes of temp[size]
	float temp[calc_and_set_temp_size(size_o_big_max)] __attribute__ ((aligned (16)));
#endif /* TEMP_ON_HEAP */

	if( !is_aligned_16(temp) )
	{
		dwt_util_log(LOG_ERR, "Unaligned temp[] buffer\n");
		dwt_util_abort();
	}
#endif /* microblaze */
	if(NULL == temp)
		dwt_util_abort();

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );
#ifdef _OPENMP
		const int threads = dwt_util_get_num_threads();
#endif
		const int workers = dwt_util_get_num_workers();
		
		const int lines_x = size_o_src_x;
		const int lines_y = size_o_src_y;

		const int workers_segment_y = floor_div(lines_y, workers);
		const int workers_segment_x = floor_div(lines_x, workers);
#ifdef _OPENMP
		const int threads_segment_y = ceil_div(workers_segment_y, threads);
		const int threads_segment_x = ceil_div(workers_segment_x, threads);
#endif
		const int workers_lines_y = workers_segment_y * workers;
		const int workers_lines_x = workers_segment_x * workers;

#ifndef DISABLE_Y
	if( lines_x > 1 )
	{
		set_data_step_s( stride_x );
#ifdef TEMP_ON_HEAP
		for(int y = 0; y < workers_lines_y; y += workers)
#else /* TEMP_ON_HEAP */
		// FIXME: OpenMP ignores __attribute__ ((aligned (16)))
		#pragma omp parallel for private(temp) schedule(static, threads_segment_y)
		for(int y = 0; y < workers_lines_y; y += workers)
#endif /* TEMP_ON_HEAP */
		{
			assert( is_aligned_16(temp) );

			dwt_cdf97_f_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_x,
				stride_y);
		}
		dwt_util_set_num_workers(1);
		for(int y = workers_lines_y; y < lines_y; y++)
		{
			dwt_cdf97_f_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_x,
				stride_y);
		}
		dwt_util_set_num_workers(workers);
	}
#endif

#ifndef DISABLE_X
	if( lines_y > 1 )
	{
		set_data_step_s( stride_y );
#ifdef TEMP_ON_HEAP
		for(int x = 0; x < workers_lines_x; x += workers)
#else /* TEMP_ON_HEAP */
		#pragma omp parallel for private(temp) schedule(static, threads_segment_x)
		for(int x = 0; x < workers_lines_x; x += workers)
#endif /* TEMP_ON_HEAP */
		{
			dwt_cdf97_f_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_y,
				stride_x);
		}
		dwt_util_set_num_workers(1);
		for(int x = workers_lines_x; x < lines_x; x++)
		{
			dwt_cdf97_f_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp + TEMP_OFFSET, // HACK: +1
				size_i_src_y,
				stride_x);
		}
		dwt_util_set_num_workers(workers);
	}
#endif

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_s(
					addr2_s(ptr,y,0,stride_x,stride_y),
					addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_s(
					addr2_s(ptr,0,x,stride_x,stride_y),
					addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}

#undef TEMP_OFFSET

#ifdef microblaze
	free(temp);
#endif

#ifdef TEMP_ON_HEAP
	free(temp);
#endif

	FUNC_END;
}

void dwt_cdf53_2f_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_src_y; y++)
			dwt_cdf53_f_ex_stride_i(
				addr2_i(ptr,y,0,stride_x,stride_y),
				addr2_i(ptr,y,0,stride_x,stride_y),
				addr2_i(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_cdf53_f_ex_stride_i(
				addr2_i(ptr,0,x,stride_x,stride_y),
				addr2_i(ptr,0,x,stride_x,stride_y),
				addr2_i(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp,
				size_i_src_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_i(
					addr2_i(ptr,y,0,stride_x,stride_y),
					addr2_i(ptr,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_i(
					addr2_i(ptr,0,x,stride_x,stride_y),
					addr2_i(ptr,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}
}

void dwt_cdf97_2f_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_src_y; y++)
			dwt_cdf97_f_ex_stride_i(
				addr2_i(ptr,y,0,stride_x,stride_y),
				addr2_i(ptr,y,0,stride_x,stride_y),
				addr2_i(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_cdf97_f_ex_stride_i(
				addr2_i(ptr,0,x,stride_x,stride_y),
				addr2_i(ptr,0,x,stride_x,stride_y),
				addr2_i(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp,
				size_i_src_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_i(
					addr2_i(ptr,y,0,stride_x,stride_y),
					addr2_i(ptr,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_i(
					addr2_i(ptr,0,x,stride_x,stride_y),
					addr2_i(ptr,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}
}

void dwt_cdf53_2f_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	float temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_src_y; y++)
			dwt_cdf53_f_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_cdf53_f_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp,
				size_i_src_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_src_y; y++)
				dwt_zero_padding_f_stride_s(
					addr2_s(ptr,y,0,stride_x,stride_y),
					addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
					size_i_src_x,
					size_o_dst_x,
					size_o_src_x-size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_src_x; x++)
				dwt_zero_padding_f_stride_s(
					addr2_s(ptr,0,x,stride_x,stride_y),
					addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
					size_i_src_y,
					size_o_dst_y,
					size_o_src_y-size_o_dst_y,
					stride_x);
		}

		j++;
	}
}

void dwt_cdf97_2i_d(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	double temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_dst_y; y++)
			dwt_cdf97_i_ex_stride_d(
				addr2_d(ptr,y,0,stride_x,stride_y),
				addr2_d(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_d(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_cdf97_i_ex_stride_d(
				addr2_d(ptr,0,x,stride_x,stride_y),
				addr2_d(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_d(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_dst_y; y++)
				dwt_zero_padding_i_stride_d(
					addr2_d(ptr,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_d(
					addr2_d(ptr,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}
}

void dwt_cdf53_2i_d(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	double temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_dst_y; y++)
			dwt_cdf53_i_ex_stride_d(
				addr2_d(ptr,y,0,stride_x,stride_y),
				addr2_d(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_d(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_cdf53_i_ex_stride_d(
				addr2_d(ptr,0,x,stride_x,stride_y),
				addr2_d(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_d(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_dst_y; y++)
				dwt_zero_padding_i_stride_d(
					addr2_d(ptr,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_d(
					addr2_d(ptr,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}
}

void dwt_cdf97_2i_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding)
{
	FUNC_BEGIN;

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SB);
#endif
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	// FIXME: allocate temp[max_threads][temp_size] and remove private() in omp parallel
	// FIXME: OpenMP cannot use this in private()
#ifdef microblaze
	float *temp = dwt_util_allocate_vec_s(calc_and_set_temp_size(size_o_big_max));
#else /* microblaze */

#ifdef TEMP_ON_HEAP
	float *temp = dwt_util_allocate_vec_s(calc_and_set_temp_size(size_o_big_max));
#else /* TEMP_ON_HEAP */
	// FIXME(x86) BUG: temp[] is allocated on stack! so stack overflow is caused by big sizes of temp[size]
	// FIXME: is aligned(16) needed?
	float temp[calc_and_set_temp_size(size_o_big_max)] __attribute__ ((aligned (16)));
#endif /* TEMP_ON_HEAP */

	if( !is_aligned_16(temp) )
	{
		dwt_util_log(LOG_ERR, "Unaligned temp[] buffer\n");
		dwt_util_abort();
	}
#endif /* microblaze */

	if(NULL == temp)
		dwt_util_abort();

	int j = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);
#ifdef _OPENMP
		const int threads = dwt_util_get_num_threads();
#endif
		const int workers = dwt_util_get_num_workers();

		const int lines_y = size_o_dst_y;
		const int lines_x = size_o_dst_x;

		const int workers_segment_y = floor_div(lines_y, workers);
		const int workers_segment_x = floor_div(lines_x, workers);
#ifdef _OPENMP
		const int threads_segment_y = ceil_div(workers_segment_y, threads);
		const int threads_segment_x = ceil_div(workers_segment_x, threads);
#endif
		const int workers_lines_y = workers_segment_y * workers;
		const int workers_lines_x = workers_segment_x * workers;

	if( lines_x > 1 )
	{
		set_data_step_s( stride_x );

#ifdef TEMP_ON_HEAP
		for(int y = 0; y < workers_lines_y; y += workers)
#else /* TEMP_ON_HEAP */
		#pragma omp parallel for private(temp) schedule(static, threads_segment_y)
		for(int y = 0; y < workers_lines_y; y += workers)
#endif /* TEMP_ON_HEAP */
		{
			assert( is_aligned_16(temp) );

			dwt_cdf97_i_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);
		}
		dwt_util_set_num_workers(1);
		for(int y = workers_lines_y; y < lines_y; y++)
		{
			dwt_cdf97_i_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);
		}
		dwt_util_set_num_workers(workers);
	}

	if( lines_y > 1 )
	{
		set_data_step_s( stride_y );

#ifdef TEMP_ON_HEAP
		for(int x = 0; x < workers_lines_x; x += workers)
#else /* TEMP_ON_HEAP */
		#pragma omp parallel for private(temp) schedule(static, threads_segment_x)
		for(int x = 0; x < workers_lines_x; x += workers)
#endif /* TEMP_ON_HEAP */
		{
			dwt_cdf97_i_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y,
				stride_x);
		}
		dwt_util_set_num_workers(1);
		for(int x = workers_lines_x; x < lines_x; x++)
		{
			dwt_cdf97_i_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y,
				stride_x);
		}
		dwt_util_set_num_workers(workers);
	}

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < size_o_dst_y; y++)
				dwt_zero_padding_i_stride_s(
					addr2_s(ptr,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_s(
					addr2_s(ptr,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}

#ifdef microblaze
	free(temp);
#endif

#ifdef TEMP_ON_HEAP
	free(temp);
#endif

	FUNC_END;
}

/* http://www.ece.uvic.ca/~frodo/publications/phdthesis.pdf
Unlike in the case of conventional (linear) versions of transforms, however, the order in which
rows and columns are transformed is important. That is, the inverse transform must operate on rows and columns in
the reverse order from that used in the forward transform; otherwise, invertibility cannot be guaranteed.
 */
void dwt_cdf53_2i_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_cdf53_i_ex_stride_i(
				addr2_i(ptr,0,x,stride_x,stride_y),
				addr2_i(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_i(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y,
				stride_x);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_dst_y; y++)
			dwt_cdf53_i_ex_stride_i(
				addr2_i(ptr,y,0,stride_x,stride_y),
				addr2_i(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_i(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_dst_y; y++)
				dwt_zero_padding_i_stride_i(
					addr2_i(ptr,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_i(
					addr2_i(ptr,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}
}

void dwt_cdf97_2i_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_cdf97_i_ex_stride_i(
				addr2_i(ptr,0,x,stride_x,stride_y),
				addr2_i(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_i(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y,
				stride_x);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_dst_y; y++)
			dwt_cdf97_i_ex_stride_i(
				addr2_i(ptr,y,0,stride_x,stride_y),
				addr2_i(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_i(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_dst_y; y++)
				dwt_zero_padding_i_stride_i(
					addr2_i(ptr,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_i(
					addr2_i(ptr,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}
}

void dwt_cdf53_2i_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	float temp[size_o_big_max];
	if(NULL == temp)
		abort();

	int j = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_dst_y; y++)
			dwt_cdf53_i_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_cdf53_i_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y,
				stride_x);

		if(zero_padding)
		{
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
			for(int y = 0; y < size_o_dst_y; y++)
				dwt_zero_padding_i_stride_s(
					addr2_s(ptr,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_s(
					addr2_s(ptr,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}
}

int dwt_util_clock_autoselect()
{
#ifdef ENABLE_TIME_CLOCK_GETTIME
	return DWT_TIME_CLOCK_GETTIME;
#endif
#ifdef ENABLE_TIME_TIMES
	return DWT_TIME_TIMES;
#endif
#ifdef ENABLE_TIME_CLOCK
	return DWT_TIME_CLOCK;
#endif
#ifdef ENABLE_TIME_GETRUSAGE
	return DWT_TIME_GETRUSAGE;
#endif
#ifdef ENABLE_TIME_GETTIMEOFDAY
	return DWT_TIME_GETTIMEOFDAY;
#endif
#ifdef ENABLE_TIME_IOCTL_RTC
	return DWT_TIME_IOCTL_RTC;
#endif
	// fallback
	return DWT_TIME_AUTOSELECT;
}

dwt_clock_t dwt_util_get_frequency(
	int type)
{
	if(DWT_TIME_AUTOSELECT == type)
		type = dwt_util_clock_autoselect();

	dwt_clock_t return_freq;

	switch(type)
	{
		case DWT_TIME_CLOCK_GETTIME:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_REALTIME:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_REALTIME
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_MONOTONIC:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_MONOTONIC_RAW:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC_RAW
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK:
		{
#ifdef ENABLE_TIME_CLOCK
			return_freq = (dwt_clock_t)CLOCKS_PER_SEC;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_TIMES:
		{
#ifdef ENABLE_TIME_TIMES
			return_freq = (dwt_clock_t)sysconf(_SC_CLK_TCK);
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE:
		{
#ifdef ENABLE_TIME_GETRUSAGE
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_IOCTL_RTC:
		{
#ifdef ENABLE_TIME_IOCTL_RTC
			return_freq = (dwt_clock_t)1;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETTIMEOFDAY:
		{
#ifdef ENABLE_TIME_GETTIMEOFDAY
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_SELF:
		{
#ifdef ENABLE_TIME_GETRUSAGE_SELF
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_CHILDREN:
		{
#ifdef ENABLE_TIME_GETRUSAGE_CHILDREN
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_THREAD:
		{
#ifdef ENABLE_TIME_GETRUSAGE_THREAD
			return_freq = (dwt_clock_t)1000000000;
#else
			abort();
#endif
		}
		break;
		default:
			abort();
	}

	return return_freq;
}

dwt_clock_t dwt_util_get_clock(
	int type)
{
	if(DWT_TIME_AUTOSELECT == type)
		type = dwt_util_clock_autoselect();

	dwt_clock_t return_time;

	switch(type)
	{
		case DWT_TIME_CLOCK_GETTIME:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME
			clockid_t clk_id = CLOCK_REALTIME;

			struct timespec ts;

			if(clock_gettime(clk_id, &ts))
				abort();

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_REALTIME:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_REALTIME
			clockid_t clk_id = CLOCK_REALTIME;

			struct timespec ts;

			if(clock_gettime(clk_id, &ts))
				abort();

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_MONOTONIC:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC
			clockid_t clk_id = CLOCK_MONOTONIC;

			struct timespec ts;

			if(clock_gettime(clk_id, &ts))
				abort();

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_MONOTONIC_RAW:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC_RAW
			clockid_t clk_id = CLOCK_MONOTONIC_RAW;

			struct timespec ts;

			if(clock_gettime(clk_id, &ts))
				abort();

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
			clockid_t clk_id = CLOCK_PROCESS_CPUTIME_ID;

			struct timespec ts;

			if(clock_gettime(clk_id, &ts))
				abort();

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
			clockid_t clk_id = CLOCK_THREAD_CPUTIME_ID;

			struct timespec ts;

			if(clock_gettime(clk_id, &ts))
				abort();

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_CLOCK:
		{
#ifdef ENABLE_TIME_CLOCK
			clock_t time;

			time = clock();

			return_time = (dwt_clock_t)time;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_TIMES:
		{
#ifdef ENABLE_TIME_TIMES
			struct tms tms_i;

			if( (clock_t)-1 == times(&tms_i) )
				abort();

			return_time = (dwt_clock_t)tms_i.tms_utime;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE:
		{
#ifdef ENABLE_TIME_GETRUSAGE
			int who = RUSAGE_SELF;

			struct rusage rusage_i;
			struct timespec ts;

			if( -1 == getrusage(who, &rusage_i) )
				abort();

			TIMEVAL_TO_TIMESPEC(&rusage_i.ru_utime, &ts);

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_IOCTL_RTC:
		{
#ifdef ENABLE_TIME_IOCTL_RTC
			int fd = open("/dev/rtc", O_RDONLY|O_NONBLOCK);
			if( -1 == fd )
				abort();

			struct rtc_time rtc_time_i;

			if( -1 == ioctl(fd, RTC_RD_TIME, &rtc_time_i) )
				abort();

			if( -1 == close(fd) )
				abort();

			time_t time = mktime( (struct tm *)&rtc_time_i );
			if( (time_t)-1 == time )
				abort();

			return_time = (dwt_clock_t)time;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETTIMEOFDAY:
		{
#ifdef ENABLE_TIME_GETTIMEOFDAY
			struct timeval tv;
			struct timespec ts;

			if( -1 == gettimeofday(&tv, NULL) )
				abort();

			TIMEVAL_TO_TIMESPEC(&tv, &ts);

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_SELF:
		{
#ifdef ENABLE_TIME_GETRUSAGE_SELF
			int who = RUSAGE_SELF;

			struct rusage rusage_i;
			struct timespec ts;

			if( -1 == getrusage(who, &rusage_i) )
				abort();

			TIMEVAL_TO_TIMESPEC(&rusage_i.ru_utime, &ts);

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_CHILDREN:
		{
#ifdef ENABLE_TIME_GETRUSAGE_CHILDREN
			int who = RUSAGE_CHILDREN;

			struct rusage rusage_i;
			struct timespec ts;

			if( -1 == getrusage(who, &rusage_i) )
				abort();

			TIMEVAL_TO_TIMESPEC(&rusage_i.ru_utime, &ts);

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_THREAD:
		{
#ifdef ENABLE_TIME_GETRUSAGE_THREAD
			int who = RUSAGE_THREAD;

			struct rusage rusage_i;
			struct timespec ts;

			if( -1 == getrusage(who, &rusage_i) )
				abort();

			TIMEVAL_TO_TIMESPEC(&rusage_i.ru_utime, &ts);

			return_time = (dwt_clock_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#else
			abort();
#endif
		}
		break;
		default:
			abort();
	}

	return return_time;
}

int dwt_util_clock_available(
	int type)
{
	switch(type)
	{
		case DWT_TIME_CLOCK_GETTIME:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME
			return 0;
#endif
		}
		break;

		case DWT_TIME_CLOCK_GETTIME_REALTIME:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_REALTIME
			return 0;
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_MONOTONIC:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC
			return 0;
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_MONOTONIC_RAW:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_MONOTONIC_RAW
			return 0;
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_PROCESS_CPUTIME_ID
			return 0;
#endif
		}
		break;
		case DWT_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID:
		{
#ifdef ENABLE_TIME_CLOCK_GETTIME_THREAD_CPUTIME_ID
			return 0;
#endif
		}
		break;
		case DWT_TIME_CLOCK:
		{
#ifdef ENABLE_TIME_CLOCK
			return 0;
#endif
		}
		break;
		case DWT_TIME_TIMES:
		{
#ifdef ENABLE_TIME_TIMES
			return 0;
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE:
		{
#ifdef ENABLE_TIME_GETRUSAGE
			return 0;
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_SELF:
		{
#ifdef ENABLE_TIME_GETRUSAGE_SELF
			return 0;
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_CHILDREN:
		{
#ifdef ENABLE_TIME_GETRUSAGE_CHILDREN
			return 0;
#endif
		}
		break;
		case DWT_TIME_GETRUSAGE_THREAD:
		{
#ifdef ENABLE_TIME_GETRUSAGE_THREAD
			return 0;
#endif
		}
		break;
		case DWT_TIME_GETTIMEOFDAY:
		{
#ifdef ENABLE_TIME_GETTIMEOFDAY
			return 0;
#endif
		}
		break;
		case DWT_TIME_IOCTL_RTC:
		{
#ifdef ENABLE_TIME_IOCTL_RTC
			return 0;
#endif
		}
		break;
		case DWT_TIME_AUTOSELECT:
		{
#ifdef ENABLE_TIME_AUTOSELECT
			return 0;
#endif
		}
		break;
	}

	return -1;
}

void dwt_util_wait(int ms)
{
	assert( ms > 0 );

	const int type = dwt_util_clock_autoselect();

	const dwt_clock_t freq = dwt_util_get_frequency(type);

	const dwt_clock_t start = dwt_util_get_clock(type);

	while( 1000.0f * (dwt_util_get_clock(type) - start) / freq < (float)ms )
		;
}

int dwt_util_get_thread_num()
{
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}

int dwt_util_get_max_threads()
{
#ifdef _OPENMP
	return omp_get_max_threads();
#else /* _OPENMP */
	return 1;
#endif /* _OPENMP */
}

int dwt_util_get_max_workers()
{
#ifdef microblaze
	return get_total_workers();
#else /* microblaze */
	return 1;
#endif /* microblaze */
}

void dwt_util_set_num_threads(
	int num_threads)
{
	assert( num_threads > 0 );

#ifdef _OPENMP
	omp_set_num_threads(num_threads);
#else
	UNUSED(num_threads);
#endif
}

void dwt_util_set_num_workers(
	int num_workers)
{
	assert( num_workers > 0 );

	set_active_workers(num_workers);
}

int dwt_util_get_num_threads()
{
#ifdef _OPENMP
	int num_threads;
	#pragma omp parallel
	{
		#pragma omp master
		{
			num_threads = omp_get_num_threads();
		}
	}
	return num_threads;
#else
	return 1;
#endif
}

int dwt_util_get_num_workers()
{
	return get_active_workers();
}

void dwt_util_init()
{
	FUNC_BEGIN;

#ifdef microblaze
	for(int w = 0; w < get_total_workers(); w++)
	{
		WAL_CHECK( wal_init_worker(worker[w]) );

		// FIXME(ASVP): translate DWT_OP_LIFT4SA into WAL_PBID_P0 by function
	
		WAL_CHECK( wal_set_firmware(worker[w], WAL_PBID_P0 /*DWT_OP_LIFT4SA*/, fw_fp01_lift4sa, -1) );

		WAL_CHECK( wal_set_firmware(worker[w], WAL_PBID_P1 /*DWT_OP_LIFT4SB*/, fw_fp01_lift4sb, -1) );

		// TODO(ASVP): call switch_op()

		WAL_CHECK( wal_reset_worker(worker[w]) );

		WAL_CHECK( wal_start_operation(worker[w], WAL_PBID_P0) );
	}

	dwt_util_set_accel(1);
#endif /* microblaze */

	FUNC_END;
}

void dwt_util_finish()
{
	FUNC_BEGIN;

#ifdef microblaze
	for(int w = 0; w < get_total_workers(); w++)
	{
		//WAL_CHECK( wal_mb2pb(worker, 0) );

		//WAL_CHECK( wal_bce_jk_sync_operation(worker) );

		WAL_CHECK( wal_done_worker(worker[w]) );
	}
#endif

	FUNC_END;
}

void dwt_util_abort()
{
	FUNC_BEGIN;

#ifdef microblaze
	for(int w = 0; w < get_total_workers(); w++)
	{
		// FIXME(ASVP): is this legal? although the operation was not running?
		wal_end_operation(worker[w]);

		// deinitialize worker
		wal_done_worker(worker[w]);
	}
#endif /* microblaze */

	abort();

	FUNC_END;
}

int dwt_util_dump_i(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( size_i_big_x >= 0 && size_i_big_y >= 0 );

	FILE *file = stdout;

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const int px = *addr2_const_i(ptr, y, x, stride_x, stride_y);

			fprintf(file, "%i ", px);
		}

		fprintf(file, "\n");
	}

	return 0;
}

int dwt_util_save_to_pgm_i(
	const char *filename,
	int max_value,
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( max_value != 0 && size_i_big_x >= 0 && size_i_big_y >= 0 );

	const int target_max_value = 255;

	FILE *file = fopen(filename, "w");
	if(NULL == file)
		return 1;

	fprintf(file, "P2\n%i %i\n%i\n", size_i_big_x, size_i_big_y, target_max_value);

	int err = 0;

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const int px = *addr2_const_i(ptr, y, x, stride_x, stride_y);

			int val = (target_max_value*px/max_value);

			if( px > max_value )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Maximum pixel intensity exceeded (%i > %i) at (x=%i, y=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, max_value, x, y);
			}

			if( px > max_value )
			{
				val = target_max_value;
			}

			if( px < 0 )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Minimum pixel intensity exceeded (%i < %i) at (x=%i, y=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, 0, x, y);
			}

			if( px < 0 )
			{
				val = 0;
			}

			if( fprintf(file, "%i\n", val) < 0 )
			{
				dwt_util_log(LOG_WARN, "%s: error writing into file.\n", __FUNCTION__);
				fclose(file);
				return 1;
			}
		}
	}

	fclose(file);

	if( err )
		dwt_util_log(LOG_WARN, "%s: %i errors ocurred while saving a file.\n", __FUNCTION__, err);

	return 0;
}

static
int skip_mess(FILE *file)
{
	assert( file );

	int c;

	while(1)
	{
		c = fgetc(file);

		if( EOF == c )
			return EOF;

		if( isspace(c) )
			continue;

		if( '#' == c )
		{
			// comment to EOL
			while(1)
			{
				c = fgetc(file);

				if( EOF == c )
					return EOF;

				if( '\n' == c )
					break;
			}

			continue;
		}

		break;
	}

	if( EOF == ungetc(c, file) )
		return EOF;

	return 0;
}

int dwt_util_load_from_pgm_s(
	const char *filename,
	float max_value,
	void **pptr,
	int *pstride_x,
	int *pstride_y,
	int *psize_x,
	int *psize_y)
{
	assert( filename && pptr && pstride_x && pstride_y && psize_x && psize_y );

	FILE *file = fopen(filename, "r");
	if(NULL == file)
	{
		dwt_util_log(LOG_ERR, "Cannot open file '%s'.\n", filename);
		return 1;
	}

	int target_max_value;

	if( 'P' != fgetc(file) || '2' != fgetc(file) )
	{
		dwt_util_log(LOG_ERR, "Invalid file header.\n");
		return 2;
	}

	skip_mess(file);

	if( 1 != fscanf(file, "%i", psize_x) )
	{
		dwt_util_log(LOG_ERR, "Invalid file metadata.\n");
		return 2;
	}

	skip_mess(file);

	if( 1 != fscanf(file, "%i", psize_y) )
	{
		dwt_util_log(LOG_ERR, "Invalid file metadata.\n");
		return 2;
	}

	skip_mess(file);

	if( 1 != fscanf(file, "%i", &target_max_value) )
	{
		dwt_util_log(LOG_ERR, "Invalid file metadata.\n");
		return 2;
	}

	if( target_max_value >= 65536 || target_max_value <= 0 )
	{
		dwt_util_log(LOG_ERR, "Invalid depth.\n");
		return 3;
	}

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "%s: going to read (%ix%i) image of depth %i...\n", __FUNCTION__, *psize_x, *psize_y, target_max_value);
#endif

	*pstride_y = sizeof(float);
	*pstride_x = dwt_util_get_opt_stride(*pstride_y * *psize_x);

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "%s: with strides (%i,%i)...\n", __FUNCTION__, *pstride_x, *pstride_y);
#endif

	dwt_util_alloc_image(pptr, *pstride_x, *pstride_y, *psize_x, *psize_y);

	for(int y = 0; y < *psize_y; y++)
	{
		for(int x = 0; x < *psize_x; x++)
		{
			float *ppx = addr2_s(*pptr, y, x, *pstride_x, *pstride_y);

			int val;

			skip_mess(file);

			if( 1 != fscanf(file, "%i", &val) )
			{
				dwt_util_log(LOG_ERR, "Invalid data.\n");
				return 4;
			}

			if( val < 0 || val > target_max_value )
			{
				dwt_util_log(LOG_ERR, "Invalid data depth.\n");
				return 5;
			}

			*ppx = max_value*val/target_max_value;
		}
	}

	fclose(file);

	return 0;
}

int dwt_util_load_from_pgm_i(
	const char *filename,
	int max_value,
	void **pptr,
	int *pstride_x,
	int *pstride_y,
	int *psize_x,
	int *psize_y)
{
	assert( filename && pptr && pstride_x && pstride_y && psize_x && psize_y );

	FILE *file = fopen(filename, "r");
	if(NULL == file)
	{
		dwt_util_log(LOG_ERR, "Cannot open file '%s'.\n", filename);
		return 1;
	}

	int target_max_value;

	if( 'P' != fgetc(file) || '2' != fgetc(file) )
	{
		dwt_util_log(LOG_ERR, "Invalid file header.\n");
		return 2;
	}

	skip_mess(file);

	if( 1 != fscanf(file, "%i", psize_x) )
	{
		dwt_util_log(LOG_ERR, "Invalid file metadata.\n");
		return 2;
	}

	skip_mess(file);

	if( 1 != fscanf(file, "%i", psize_y) )
	{
		dwt_util_log(LOG_ERR, "Invalid file metadata.\n");
		return 2;
	}

	skip_mess(file);

	if( 1 != fscanf(file, "%i", &target_max_value) )
	{
		dwt_util_log(LOG_ERR, "Invalid file metadata.\n");
		return 2;
	}

	if( target_max_value >= 65536 || target_max_value <= 0 )
	{
		dwt_util_log(LOG_ERR, "Invalid depth.\n");
		return 3;
	}

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "%s: going to read (%ix%i) image of depth %i...\n", __FUNCTION__, *psize_x, *psize_y, target_max_value);
#endif

	*pstride_y = sizeof(int);
	*pstride_x = dwt_util_get_opt_stride(*pstride_y * *psize_x);

#ifdef DEBUG
	dwt_util_log(LOG_DBG, "%s: with strides (%i,%i)...\n", __FUNCTION__, *pstride_x, *pstride_y);
#endif

	dwt_util_alloc_image(pptr, *pstride_x, *pstride_y, *psize_x, *psize_y);

	for(int y = 0; y < *psize_y; y++)
	{
		for(int x = 0; x < *psize_x; x++)
		{
			int *ppx = addr2_i(*pptr, y, x, *pstride_x, *pstride_y);

			int val;

			skip_mess(file);

			if( 1 != fscanf(file, "%i", &val) )
			{
				dwt_util_log(LOG_ERR, "Invalid data.\n");
				return 4;
			}

			if( val < 0 || val > target_max_value )
			{
				dwt_util_log(LOG_ERR, "Invalid data depth.\n");
				return 5;
			}

			*ppx = max_value*val/target_max_value;
		}
	}

	fclose(file);

	return 0;
}

int dwt_util_save_to_pgm_s(
	const char *filename,
	float max_value,
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( max_value != 0.0f && size_i_big_x >= 0 && size_i_big_y >= 0 );

	const int target_max_value = 255;

	FILE *file = fopen(filename, "w");
	if(NULL == file)
		return 1;

	fprintf(file, "P2\n%i %i\n%i\n", size_i_big_x, size_i_big_y, target_max_value);

	int err = 0;

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const float px = *addr2_const_s(ptr, y, x, stride_x, stride_y);

			int val = (target_max_value*px/max_value);

			if( px - 1e-3f > max_value )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Maximum pixel intensity exceeded (%f > %f). Such an incident will be reported only once.\n", __FUNCTION__, px, max_value);
			}

			if( px > max_value )
			{
				val = target_max_value;
			}

			if( px + 1e-3f < 0.0f )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Minimum pixel intensity exceeded (%f < %f). Such an incident will be reported only once.\n", __FUNCTION__, px, 0.0f);
			}

			if( px < 0.0f )
			{
				val = 0;
			}

			if( fprintf(file, "%i\n", val) < 0)
			{
				dwt_util_log(LOG_WARN, "%s: error writing into file.\n", __FUNCTION__);
				fclose(file);
				return 1;
			}
		}
	}

	fclose(file);

	if( err )
		dwt_util_log(LOG_WARN, "%s: %i errors ocurred while saving a file.\n", __FUNCTION__, err);

	return 0;
}

int dwt_util_save_to_pgm_d(
	const char *filename,
	double max_value,
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	assert( max_value != 0.0f && size_i_big_x >= 0 && size_i_big_y >= 0 );

	const int target_max_value = 255;

	FILE *file = fopen(filename, "w");
	if(NULL == file)
		return 1;

	fprintf(file, "P2\n%i %i\n%i\n", size_i_big_x, size_i_big_y, target_max_value);

	int err = 0;

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const double px = *addr2_const_d(ptr, y, x, stride_x, stride_y);

			int val = (target_max_value*px/max_value);

			if( px - 1e-6 > max_value )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Maximum pixel intensity exceeded (%f > %f). Such an incident will be reported only once.\n", __FUNCTION__, px, max_value);
			}

			if( px > max_value )
			{
				val = target_max_value;
			}

			if( px + 1e-6 < 0.0 )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Minimum pixel intensity exceeded (%f < %f). Such an incident will be reported only once.\n", __FUNCTION__, px, 0.0f);
			}

			if( px < 0.0 )
			{
				val = 0;
			}

			if( fprintf(file, "%i\n", val) < 0)
			{
				dwt_util_log(LOG_WARN, "%s: error writing into file.\n", __FUNCTION__);
				fclose(file);
				return 1;
			}
		}
	}

	fclose(file);

	if( err )
		dwt_util_log(LOG_WARN, "%s: %i errors ocurred while saving a file.\n", __FUNCTION__, err);

	return 0;
}

void dwt_util_set_accel(
	int accel_type)
{
	set_accel_type(accel_type);
}

#define iszero(x) (fpclassify(x) == FP_ZERO)

int dwt_util_is_normal_or_zero_i(const float *a)
{
	if( isnormal(*a) || iszero(*a) )
		return 1;

	return 0;
}

int dwt_util_is_normal_or_zero(float a)
{
	return dwt_util_is_normal_or_zero_i(&a);
}

int dwt_util_cmp_s_i(const float *a, const float *b)
{
	assert( a );
	assert( b );
	
	const float eps = 1e-4; // FIXME: magic constant

	if( !dwt_util_is_normal_or_zero_i(a) || !dwt_util_is_normal_or_zero_i(b) )
	{
		dwt_util_log(LOG_ERR, "%f or %f is not normal nor zero!\n", *a, *b);
		return 1;
	}

	if( fabsf( (*a) - (*b) ) > eps )
	{
		dwt_util_log(LOG_ERR, "%f should be %f!\n", *a, *b);
		return 1;
	}

	return 0;
}

int dwt_util_cmp_s(float a, float b)
{
	return dwt_util_cmp_s_i(&a, &b);
}

int dwt_util_generate_vec_s(float *addr, int size)
{
	for(int i = 0; i < size; i++)
		addr[i] = (float)i;

	for(int i = 0; i < size; i++)
	{
		if( dwt_util_cmp_s(addr[i], (float)i) )
			return 1;
	}

	return 0;
}

// 4-bytes alignment
float *dwt_util_allocate_4_vec_s(int size)
{
	assert( is_even(size) );

	float *addr = (float *)0;

	// http://git.uclibc.org/uClibc/tree/include - memalign, posix_memalign
	addr = (float *)memalign(4, sizeof(float) * size);

	assert( is_aligned_4(addr) );

	return addr;
}

// 8-bytes alignment
float *dwt_util_allocate_8_vec_s(int size)
{
	assert( is_even(size) );

	float *addr = (float *)0;

	// http://git.uclibc.org/uClibc/tree/include - memalign, posix_memalign
	addr = (float *)memalign(8, sizeof(float) * size);

	assert( is_aligned_8(addr) );

	return addr;
}

// 16-bytes alignment
float *dwt_util_allocate_16_vec_s(int size)
{
	assert( is_even(size) );

	float *addr = (float *)0;

	// http://git.uclibc.org/uClibc/tree/include - memalign, posix_memalign
	addr = (float *)memalign(16, sizeof(float) * size);

	assert( is_aligned_16(addr) );

	return addr;
}

static
int is_pow2(int x)
{
	return 0 == (x & (x - 1));
}

float *dwt_util_allocate_vec_s(int size)
{
	// FIXME: why must be even???
	size = to_even(size);

	float *addr = (float *)0;

	// http://git.uclibc.org/uClibc/tree/include - memalign, posix_memalign
	addr = (float *)memalign(16, sizeof(float) * size);

	assert( is_aligned_16(addr) );

	return addr;
}

int dwt_util_zero_vec_s(float *addr, int size)
{
	for(int i = 0; i < size; i++)
		addr[i] = (float)0;

	for(int i = 0; i < size; i++)
	{
		if( dwt_util_cmp_s(addr[i], (float)0) )
			return 1;
	}

	return 0;
}

int dwt_util_copy_vec_s(const float *src, float *dst, int size)
{
	dwt_util_memcpy_stride_s(dst, sizeof(dst[0]), src, sizeof(src[0]), size);

	for(int i = 0; i < size; i++)
	{
		if( dwt_util_cmp_s(dst[i], src[i]) )
			return 1;
	}

	return 0;
}

int dwt_util_cmp_vec_s(const float *a, const float *b, int size)
{
	for(int i = size-1; i >= 0; i--)
	{
		if( dwt_util_cmp_s(a[i], b[i]) )
			return 1;
	}

	return 0;
}

void dwt_util_print_vec_s(const float *addr, int size)
{
	dwt_util_log(LOG_NONE, "[ ");
	for(int i = 0; i < size; i++)
		dwt_util_log(LOG_NONE, "%f ", addr[i]);
	dwt_util_log(LOG_NONE, "]\n");
}

void dwt_util_test()
{
	for(int i = 2; i <= BANK_SIZE; i *= 2)
	{
		dwt_util_log(LOG_TEST, "allocate vector of %i floats...\n", i);
		float *addr = dwt_util_allocate_vec_s(i);
		if( !addr )
		{
			dwt_util_log(LOG_ERR, "Failed to allocate vector of %i floats.\n", i);
			dwt_util_abort();
		}
		free(addr);
		dwt_util_log(LOG_TEST, "ok\n");
	}

#ifdef microblaze
	for(int w = 0; w < get_total_workers(); w++)
	{
		dwt_util_log(LOG_TEST, "worker %i: init worker...\n", w);
		if( wal_init_worker(worker[w]) )
			abort();
		if( wal_reset_worker(worker[w]) )
			abort();

		const int size = BANK_SIZE;

		dwt_util_log(LOG_TEST, "allocating vector of %i floats...\n", size);
		float *addr = dwt_util_allocate_vec_s(size);
		if( !addr )
			dwt_util_abort();

		if( dwt_util_generate_vec_s(addr, size) )
			dwt_util_abort();

		dwt_util_log(LOG_TEST, "making copy of vector...\n");

		float *copy = dwt_util_allocate_vec_s(size);
		if( !copy )
			dwt_util_abort();

		if( dwt_util_copy_vec_s(addr, copy, size) )
			dwt_util_abort();
		if( dwt_util_cmp_vec_s(addr, copy, size) )
			dwt_util_abort();
		
		dwt_util_log(LOG_TEST, "worker %i: memory transfer to BCE memory using new-style function...\n", w);

		if( wal_dma_configure(worker[w], 0, addr, 0, WAL_BCE_JSY_DMEM_A, 0, size) )
			abort();

		if( wal_dma_start(worker[w], 0, WAL_DMA_REQ_RD) )
			abort();

		while( wal_dma_isbusy(worker[w], 0x1) )
			;

		dwt_util_log(LOG_TEST, "zeroing memory...\n");

		if( dwt_util_zero_vec_s(addr, size) )
			dwt_util_abort();

		dwt_util_log(LOG_TEST, "worker %i: memory transfer from BCE memory using new-style function...\n", w);

		if( wal_dma_start(worker[w], 0, WAL_DMA_REQ_WR) )
			abort();
		while( wal_dma_isbusy(worker[w], 0x1) )
			;

		dwt_util_log(LOG_TEST, "flushing cache...\n");

		flush_cache_s(addr, size);

		dwt_util_log(LOG_TEST, "comparing with original sequence...\n");

		if( dwt_util_cmp_vec_s(addr, copy, size) )
			dwt_util_abort();

		dwt_util_log(LOG_TEST, "worker %i: calling done worker...\n", w);
		
		wal_done_worker(worker[w]);

		dwt_util_log(LOG_TEST, "all tests done\n");
	}
#endif
}

int dwt_util_vfprintf(FILE *stream, const char *format, va_list ap)
{
	return vfprintf(stream, format, ap);
}

int dwt_util_vprintf(const char *format, va_list ap)
{
	return dwt_util_vfprintf(stdout, format, ap);
}

int dwt_util_fprintf(FILE *stream, const char *format, ...)
{
	va_list ap;

	va_start(ap, format);
	int ret = dwt_util_vfprintf(stream, format, ap);
	va_end(ap);

	return ret;
}

int dwt_util_printf(const char *format, ...)
{
	va_list ap;

	va_start(ap, format);
	int ret = dwt_util_vprintf(format, ap);
	va_end(ap);

	return ret;
}

int dwt_util_log(
	enum dwt_util_loglevel level,
	const char *format,
	...)
{
	int ret = 0;
	FILE *stream = stderr;

	const char *prefix[] = {
		[LOG_NONE] = "",
		[LOG_DBG]  = "DEBUG: ",
		[LOG_INFO] = "INFO: ",
		[LOG_WARN] = "WARNING: ",
		[LOG_ERR]  = "ERROR: ",
		[LOG_TEST] = "TEST: ",
	};

	flockfile(stream);

	ret += dwt_util_fprintf(stream, prefix[level]);

	va_list ap;

	va_start(ap, format);
	ret += dwt_util_vfprintf(stream, format, ap);
	va_end(ap);

	fflush(stream);

	funlockfile(stream);

	return ret;
}

void dwt_util_error(
	const char *format,
	...)
{
	va_list ap;

	va_start(ap, format);
	dwt_util_log(LOG_ERR, format, ap);
	va_end(ap);

	dwt_util_abort();
}

static
void *alloc(size_t size)
{
	void *ptr = malloc(size);

	if(!ptr)
	{
		dwt_util_log(LOG_ERR, "Unable to allocate memory.\n");
		dwt_util_abort();
	}

	return ptr;
}

static
const char *node()
{
	long host_name_max = sysconf(_SC_HOST_NAME_MAX);
	if( -1 == host_name_max )
		host_name_max = 255;
	host_name_max++; // the terminating null byte
	
	// NOTE: should gethostname be called instead of reading from procfs?
	FILE *f = fopen("/proc/sys/kernel/hostname", "r");
	if(f)
	{
		static char *buff = NULL; // NOTE: global variable
		if(!buff)
			buff = (char *)alloc(host_name_max);

		const char *ret = fgets(buff, host_name_max, f);
		fclose(f);

		if(ret)
		{
			char *nl = strchr(buff, '\n');
			if(nl)
				*nl = 0;

			return buff;
		}
		else
			return "unknown";
	}
	else
		return "unknown";
}

const char *dwt_util_node()
{
	return node();
}

static
const char *appname()
{
	long page_size = sysconf(_SC_PAGESIZE);
	if( -1 == page_size )
		page_size = 4096;

	FILE *f = fopen("/proc/self/cmdline", "r");
	if(f)
	{
		static char *buff = NULL; // NOTE: global variable
		if(!buff)
			buff = (char *)alloc(page_size);

		const char *ret = fgets(buff, page_size, f);
		fclose(f);
		
		if(ret)
			return basename(buff);
		else
			return "unknown";
	}
	else
		return "unknown";
}

const char *dwt_util_appname()
{
	return appname();
}

static
int find_dfa_seq(int N)
{
	int state = 1;

	int count = 0;

	do
	{
		const int addr = 2 * state;

		state = addr - N * (addr >= N);

		count++;

		if( 1 == state )
			return count;
	}
	while( count < 2*N );

	return 0;
}

/**
 * @brief Variant of Fermat primality test for base-2.
 */
static
int is_prime(int N)
{
	// 2 is prime
	if( 2 == N )
		return 1;

	// even numbers are not primes, i.e. 0, 2, 4, 6, 8, ...
	if( !(N & 1) )
		return 0;

	// negative numbers and unity are not prime numbers, i.e. ..., -2, -1, 0, 1
	if( N < 2 )
		return 0;

	// number of zeros after leading one-bit in left side of Fermat's little theorem with base 2
	const int d = N - 1;

	// length of zero-bit sequence after leading one-bit accepted by DFA which accepts numbers congruent to 1 modulo N
	const int c = find_dfa_seq(N);

	// can DFA accept a big number in the left side of Fermat's little theorem?
	const int r = d % c;

	// if can then we got probably prime
	if( 0 == r )
		return 1;

	return 0;
}

int dwt_util_is_prime(int N)
{
	return is_prime(N);
}

/**
 * @brief Returns smallest prime not less than N.
 */
static
int next_prime(int N)
{
	if( N <= 2 )
		return 2;

	N |= 1;

	while( !is_prime(N) )
		N += 2;

	return N;
}

int dwt_util_next_prime(int N)
{
	return next_prime(N);
}

int dwt_util_is_pow2(int x)
{
	return is_pow2(x);
}

long dwt_util_get_ncpus()
{
	long nprocessors_conf = sysconf(_SC_NPROCESSORS_CONF);
	if( -1 == nprocessors_conf )
		nprocessors_conf = 1;
	return nprocessors_conf;
}

void dwt_util_print_info()
{
	dwt_util_log(LOG_INFO, "architecture: \"%s\"\n", dwt_util_arch());

	dwt_util_log(LOG_INFO, "address:\n");

#if microblaze
	size_t ptr_size = sizeof(void*);
	long ptr_size_bits = ptr_size<<3;
	dwt_util_log(LOG_INFO, "[addr:%u]\n", ptr_size_bits);
#endif

#ifdef __x86_64__
	size_t ptr_size = sizeof(void*);
	long ptr_size_bits = ptr_size<<3;
	long LEVEL1_DCACHE_SIZE = sysconf(_SC_LEVEL1_DCACHE_SIZE);
	long LEVEL1_DCACHE_ASSOC = sysconf(_SC_LEVEL1_DCACHE_ASSOC);
	long LEVEL1_DCACHE_LINESIZE = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
	unsigned dcache_offset_bits = (unsigned)ceil_log2(LEVEL1_DCACHE_LINESIZE);
	// FIXME: div. by zero
	long dcache_sets = LEVEL1_DCACHE_SIZE / LEVEL1_DCACHE_ASSOC / LEVEL1_DCACHE_LINESIZE;
	unsigned dcache_set_bits = (unsigned)ceil_log2(dcache_sets);
	unsigned tag_bits = (unsigned)(ptr_size_bits - dcache_set_bits - dcache_offset_bits);

	dwt_util_log(LOG_INFO, "[addr:%u] => [tag:%u][cache_set:%u][offset:%u]\n", ptr_size_bits, tag_bits, dcache_set_bits, dcache_offset_bits);
#endif

#if __arm__
	size_t ptr_size = sizeof(void*);
	long ptr_size_bits = ptr_size<<3;
	dwt_util_log(LOG_INFO, "[addr:%u]\n", ptr_size_bits);
#endif

	dwt_util_log(LOG_INFO, "number of CPUs = %lu\n", dwt_util_get_ncpus());
}

static
int get_opt_stride(int min_stride)
{
	assert( min_stride > 0 );

#ifdef microblaze
	// align to 32 bits due to MicroBlaze constraints

	// higher stride has better performance (observed)
	const int stride = align_8(next_prime(min_stride));

	// powers of two have worse performance (observed)
	return is_pow2(stride) ? align_8(stride+1) : stride;
#endif

#ifdef __x86_64__
	// find prime number not lesser than min_stride
	return next_prime(min_stride);
#endif

#ifdef __arm__
	// FIXME: what align is really needed?
	return align_8(min_stride);
#endif
}

int dwt_util_get_opt_stride(int min_stride)
{
	return get_opt_stride(min_stride);
}

#define up_to_odd(x) ((x)|1)

/*
	getconf -a | grep -i cache
	/sys/devices/system/cpu/cpu0/cache/

	LEVEL1_DCACHE_LINESIZE = 64-byte cache line => log2(64) = 6-bit [offset]
	LEVEL1_DCACHE_ASSOC = associativity = 8-way
	LEVEL1_DCACHE_SIZE = cache size = 64 sets x 8 ways x 64-byte line = 32K
	number of sets = 64 sets = 32K / 8-ways / 64-byte line => log2(64) = 6-bit [set]
	32-bit address: [tag:20][set:6][offset:6]

	[                val]
	[              prime]
	[      prime][000000]
	[tag][000001][000000]
 */
int dwt_util_get_stride(int min_stride, int opt)
{
#ifdef microblaze
	return opt ? get_opt_stride(min_stride) : min_stride;
#endif
#ifdef __arm__
	// FIXME: ARM needs special alignment for floats, e.g. "case 2" bellow
	return opt ? get_opt_stride(min_stride) : min_stride;
#endif
	switch(opt)
	{
		case 0:
			// [                val]
			return min_stride;
		case 1:
			// [              prime]
			return next_prime(min_stride);
		case 2:
			// [      prime][000000]
			return next_prime(align_64(min_stride)>>6)<<6;
		case 3:
			// [tag][000001][000000]
			return ((align_4096(min_stride)>>6)+1)<<6;
		case 4:
			// [tag][   val][000000]
			return align_64(min_stride);
		case 5:
			// [tag][xx0000][000000]
			return align_4096(min_stride) + (1<<ceil_log2(min_stride));
		case 6:
			// [                odd]
			return up_to_odd(min_stride);
		case 7:
			// [        odd][000000]
			return up_to_odd(align_64(min_stride)>>6)<<6;
		default:
		{
			dwt_util_log(LOG_DBG, "%s: invalid stride choice (%i)\n", __FUNCTION__, opt);
			return min_stride;
		}
	}
}

void dwt_util_subband(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	enum dwt_subbands band,
	void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y)
{
	assert( ptr != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 && size_o_big_x >= 0 && size_o_big_y >= 0 );

	int inner_H_x = 0;
	int inner_H_y = 0;
	int inner_L_x = size_i_big_x;
	int inner_L_y = size_i_big_y;
	int outer_x = size_o_big_x;
	int outer_y = size_o_big_y;

	for(int j = 1; j <= j_max; j++)
	{
		inner_H_x = floor_div2(inner_L_x);
		inner_H_y = floor_div2(inner_L_y);
		inner_L_x = ceil_div2 (inner_L_x);
		inner_L_y = ceil_div2 (inner_L_y);
		outer_x   = ceil_div2 (outer_x);
		outer_y   = ceil_div2 (outer_y);
	}

	switch(band)
	{
		case DWT_LL:
			*dst_ptr = addr2(ptr,
				0, 0,
				stride_x, stride_y);
			*dst_size_x = inner_L_x;
			*dst_size_y = inner_L_y;
			break;
		case DWT_HL:
			*dst_ptr = addr2(ptr,
				0, outer_x,
				stride_x, stride_y);
			*dst_size_x = inner_H_x;
			*dst_size_y = inner_L_y;
			break;
		case DWT_LH:
			*dst_ptr = addr2(ptr,
				outer_y, 0,
				stride_x, stride_y);
			*dst_size_x = inner_L_x;
			*dst_size_y = inner_H_y;
			break;
		case DWT_HH:
			*dst_ptr = addr2(ptr,
				outer_y, outer_x,
				stride_x, stride_y);
			*dst_size_x = inner_H_x;
			*dst_size_y = inner_H_y;
			break;
	}
}

void dwt_util_subband_const(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	enum dwt_subbands band,
	const void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y)
{
	assert( ptr != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 && size_o_big_x >= 0 && size_o_big_y >= 0 );

	int inner_H_x = 0;
	int inner_H_y = 0;
	int inner_L_x = size_i_big_x;
	int inner_L_y = size_i_big_y;
	int outer_x = size_o_big_x;
	int outer_y = size_o_big_y;

	for(int j = 1; j <= j_max; j++)
	{
		inner_H_x = floor_div2(inner_L_x);
		inner_H_y = floor_div2(inner_L_y);
		inner_L_x = ceil_div2 (inner_L_x);
		inner_L_y = ceil_div2 (inner_L_y);
		outer_x   = ceil_div2 (outer_x);
		outer_y   = ceil_div2 (outer_y);
	}

	switch(band)
	{
		case DWT_LL:
			*dst_ptr = addr2_const(ptr,
				0, 0,
				stride_x, stride_y);
			*dst_size_x = inner_L_x;
			*dst_size_y = inner_L_y;
			break;
		case DWT_HL:
			*dst_ptr = addr2_const(ptr,
				0, outer_x,
				stride_x, stride_y);
			*dst_size_x = inner_H_x;
			*dst_size_y = inner_L_y;
			break;
		case DWT_LH:
			*dst_ptr = addr2_const(ptr,
				outer_y, 0,
				stride_x, stride_y);
			*dst_size_x = inner_L_x;
			*dst_size_y = inner_H_y;
			break;
		case DWT_HH:
			*dst_ptr = addr2_const(ptr,
				outer_y, outer_x,
				stride_x, stride_y);
			*dst_size_x = inner_H_x;
			*dst_size_y = inner_H_y;
			break;
	}
}

void dwt_util_subband_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	enum dwt_subbands band,
	void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y)
{
	dwt_util_subband(
		ptr,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		band,
		dst_ptr,
		dst_size_x,
		dst_size_y);
}

void dwt_util_subband_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	enum dwt_subbands band,
	void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y)
{
	dwt_util_subband(
		ptr,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		band,
		dst_ptr,
		dst_size_x,
		dst_size_y);
}

void dwt_util_subband_const_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	enum dwt_subbands band,
	const void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y)
{
	dwt_util_subband_const(
		ptr,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		band,
		dst_ptr,
		dst_size_x,
		dst_size_y);
}

void dwt_util_subband_d(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	enum dwt_subbands band,
	void **dst_ptr,
	int *dst_size_x,
	int *dst_size_y)
{
	dwt_util_subband(
		ptr,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		band,
		dst_ptr,
		dst_size_x,
		dst_size_y);
}

void dwt_util_diff_i(
	const void *src0,
	const void *src1,
	void *dst,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	FUNC_BEGIN;

	assert( src0 && src1 && dst && size_i_big_x >= 0 && size_i_big_y >= 0 );

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const int c0 = *addr2_const_i(src0, y, x, stride_x, stride_y);
			const int c1 = *addr2_const_i(src1, y, x, stride_x, stride_y);
			int *c = addr2_i(dst, y, x, stride_x, stride_y);

			*c = c1 - c0;
		}
	}

	FUNC_END;
}

/**
 * @brief Natural logarithm of @e x, i.e. ln(x) or log_{e}(x).
 */
void log_i_s(float *result, float x)
{
	*result = log(x);
}

void log_i_d(double *result, double x)
{
	*result = log(x);
}

void dwt_util_conv_show_i(
	const void *src,
	void *dst,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	FUNC_BEGIN;

	assert( src != NULL && dst != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const int coeff = *addr2_const_i(src, y, x, stride_x, stride_y);
			int *log_coeff = addr2_i(dst, y, x, stride_x, stride_y);

			// FIXME: *log_coeff = ceil_log2(1+abs(coeff)<<a)<<b;
			*log_coeff = abs(coeff);
		}
	}

	FUNC_END;
}

void dwt_util_conv_show_s(
	const void *src,
	void *dst,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	FUNC_BEGIN;

	assert( src != NULL && dst != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	// magic constants
	const float a = 100.f;
	const float b = 10.f;

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const float coeff = *addr2_const_s(src, y, x, stride_x, stride_y);
			float *log_coeff = addr2_s(dst, y, x, stride_x, stride_y);

			float temp;
			log_i_s(&temp, 1.f+fabsf(coeff)*a);
			temp /= b;

			*log_coeff = temp;
		}
	}

	FUNC_END;
}

void dwt_util_conv_show_d(
	const void *src,
	void *dst,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	FUNC_BEGIN;

	assert( src != NULL && dst != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	// magic constants
	const double a = 100.;
	const double b = 10.;

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const double coeff = *addr2_const_d(src, y, x, stride_x, stride_y);
			double *log_coeff = addr2_d(dst, y, x, stride_x, stride_y);

			double temp;
			log_i_d(&temp, 1.+fabs(coeff)*a);
			temp /= b;

			*log_coeff = temp;
		}
	}

	FUNC_END;
}

void dwt_util_copy_s(
	const void *src,
	void *dst,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	FUNC_BEGIN;

	assert( src != NULL && dst != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const float src_coeff = *addr2_const_s(src, y, x, stride_x, stride_y);
			float *dst_coeff = addr2_s(dst, y, x, stride_x, stride_y);

			*dst_coeff = src_coeff;
		}
	}

	FUNC_END;
}

void dwt_util_copy_i(
	const void *src,
	void *dst,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y)
{
	FUNC_BEGIN;

	assert( src != NULL && dst != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const int src_coeff = *addr2_const_i(src, y, x, stride_x, stride_y);
			int *dst_coeff = addr2_i(dst, y, x, stride_x, stride_y);

			*dst_coeff = src_coeff;
		}
	}

	FUNC_END;
}

// TODO: propagate "flush"
void dwt_util_perf_cdf53_2_i(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs)
{
	FUNC_BEGIN;

	assert( M > 0 && N > 0 && fwd_secs && inv_secs );

	assert( size_o_big_x > 0 && size_o_big_y > 0 && size_i_big_x > 0 && size_i_big_y > 0 );

	// pointer to M pointers to image data
	void *ptr[M];
	int j[M];
	
	// allocate M images
	for(int m = 0; m < M; m++)
	{
		// copy j_max to j[]
		j[m] = j_max;

		// allocate
		dwt_util_alloc_image(
			&ptr[m],
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y);
		
		// fill with test pattern
		dwt_util_test_image_fill_i(
			ptr[m],
			stride_x,
			stride_y,
			size_i_big_x,
			size_i_big_y,
			0);
		
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
			flush_cache(ptr[m], image_size(stride_x, stride_y, size_o_big_x, size_o_big_y) );
#endif

		// start timer
		const dwt_clock_t time_fwd_start = dwt_util_get_clock(clock_type);
		// perform M fwd transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf53_2f_i(
				ptr[m],
				stride_x,
				stride_y,
				size_o_big_x,
				size_o_big_y,
				size_i_big_x,
				size_i_big_y,
				&j[m],
				decompose_one,
				zero_padding);
		}
		// stop timer
		const dwt_clock_t time_fwd_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_fwd_secs = (float)(time_fwd_stop - time_fwd_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_fwd_secs < *fwd_secs )
			*fwd_secs = time_fwd_secs;

#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
			flush_cache(ptr[m], image_size(stride_x, stride_y, size_o_big_x, size_o_big_y) );
#endif

		// start timer
		const dwt_clock_t time_inv_start = dwt_util_get_clock(clock_type);
		// perform M inv transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf53_2i_i(
				ptr[m],
				stride_x,
				stride_y,
				size_o_big_x,
				size_o_big_y,
				size_i_big_x,
				size_i_big_y,
				j[m],
				decompose_one,
				zero_padding);
		}
		// stop timer
		const dwt_clock_t time_inv_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_inv_secs = (float)(time_inv_stop - time_inv_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_inv_secs < *inv_secs )
			*inv_secs = time_inv_secs;
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}

	FUNC_END;
}

// TODO: propagate "flush"
void dwt_util_perf_cdf97_2_s(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding,
	int M,
	int N,
	int clock_type,
	float *fwd_secs,
	float *inv_secs)
{
	FUNC_BEGIN;

	assert( M > 0 && N > 0 && fwd_secs && inv_secs );

	assert( size_o_big_x > 0 && size_o_big_y > 0 && size_i_big_x > 0 && size_i_big_y > 0 );

	// pointer to M pointers to image data
	void *ptr[M];
	int j[M];
	
	// allocate M images
	for(int m = 0; m < M; m++)
	{
		// copy j_max to j[]
		j[m] = j_max;

		// allocate
		dwt_util_alloc_image(
			&ptr[m],
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y);
		
		// fill with test pattern
		dwt_util_test_image_fill_s(
			ptr[m],
			stride_x,
			stride_y,
			size_i_big_x,
			size_i_big_y,
			0);
		
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
			flush_cache(ptr[m], image_size(stride_x, stride_y, size_o_big_x, size_o_big_y) );
#endif

		// start timer
		const dwt_clock_t time_fwd_start = dwt_util_get_clock(clock_type);
		// perform M fwd transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2f_s(
				ptr[m],
				stride_x,
				stride_y,
				size_o_big_x,
				size_o_big_y,
				size_i_big_x,
				size_i_big_y,
				&j[m],
				decompose_one,
				zero_padding);
		}
		// stop timer
		const dwt_clock_t time_fwd_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_fwd_secs = (float)(time_fwd_stop - time_fwd_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_fwd_secs < *fwd_secs )
			*fwd_secs = time_fwd_secs;

#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
			flush_cache(ptr[m], image_size(stride_x, stride_y, size_o_big_x, size_o_big_y) );
#endif

		// start timer
		const dwt_clock_t time_inv_start = dwt_util_get_clock(clock_type);
		// perform M inv transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2i_s(
				ptr[m],
				stride_x,
				stride_y,
				size_o_big_x,
				size_o_big_y,
				size_i_big_x,
				size_i_big_y,
				j[m],
				decompose_one,
				zero_padding);
		}
		// stop timer
		const dwt_clock_t time_inv_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const float time_inv_secs = (float)(time_inv_stop - time_inv_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_inv_secs < *inv_secs )
			*inv_secs = time_inv_secs;
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}

	FUNC_END;
}

// TODO: propagate "flush"
void dwt_util_perf_cdf97_2_d(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding,
	int M,
	int N,
	int clock_type,
	double *fwd_secs,
	double *inv_secs)
{
	FUNC_BEGIN;

	assert( M > 0 && N > 0 && fwd_secs && inv_secs );

	assert( size_o_big_x > 0 && size_o_big_y > 0 && size_i_big_x > 0 && size_i_big_y > 0 );

	// pointer to M pointers to image data
	void *ptr[M];
	int j[M];
	
	// allocate M images
	for(int m = 0; m < M; m++)
	{
		// copy j_max to j[]
		j[m] = j_max;

		// allocate
		dwt_util_alloc_image(
			&ptr[m],
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y);
		
		// fill with test pattern
		dwt_util_test_image_fill_d(
			ptr[m],
			stride_x,
			stride_y,
			size_i_big_x,
			size_i_big_y,
			0);
		
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
			flush_cache(ptr[m], image_size(stride_x, stride_y, size_o_big_x, size_o_big_y) );
#endif

		// start timer
		const dwt_clock_t time_fwd_start = dwt_util_get_clock(clock_type);
		// perform M fwd transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2f_d(
				ptr[m],
				stride_x,
				stride_y,
				size_o_big_x,
				size_o_big_y,
				size_i_big_x,
				size_i_big_y,
				&j[m],
				decompose_one,
				zero_padding);
		}
		// stop timer
		const dwt_clock_t time_fwd_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const double time_fwd_secs = (double)(time_fwd_stop - time_fwd_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_fwd_secs < *fwd_secs )
			*fwd_secs = time_fwd_secs;

#if 1
		// FIXME: flush memory
		for(int m = 0; m < M; m++)
			flush_cache(ptr[m], image_size(stride_x, stride_y, size_o_big_x, size_o_big_y) );
#endif

		// start timer
		const dwt_clock_t time_inv_start = dwt_util_get_clock(clock_type);
		// perform M inv transforms
		for(int m = 0; m < M; m++)
		{
			dwt_cdf97_2i_d(
				ptr[m],
				stride_x,
				stride_y,
				size_o_big_x,
				size_o_big_y,
				size_i_big_x,
				size_i_big_y,
				j[m],
				decompose_one,
				zero_padding);
		}
		// stop timer
		const dwt_clock_t time_inv_stop = dwt_util_get_clock(clock_type);
		// calc avg
		const double time_inv_secs = (double)(time_inv_stop - time_inv_start) / M * MEASURE_FACTOR / dwt_util_get_frequency(clock_type);
		// select min
		if( time_inv_secs < *inv_secs )
			*inv_secs = time_inv_secs;
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}

	FUNC_END;
}

void dwt_util_get_sizes_i(
	enum dwt_array array_type,
	int size_x,
	int size_y,
	int opt_stride,
	int *stride_x,
	int *stride_y,
	int *size_o_big_x,
	int *size_o_big_y,
	int *size_i_big_x,
	int *size_i_big_y)
{
	FUNC_BEGIN;

	assert( size_x > 0 && size_y > 0 );

	assert( stride_x && stride_y && size_o_big_x && size_o_big_y && size_i_big_x && size_i_big_y );

	*stride_y = sizeof(int);
	*stride_x = dwt_util_get_stride(
		(*stride_y) * dwt_util_pow2_ceil_log2(size_x), opt_stride);

	*size_o_big_x = size_x;
	*size_o_big_y = size_y;
	*size_i_big_x = size_x;
	*size_i_big_y = size_y;

	if( DWT_ARR_SPARSE == array_type || DWT_ARR_SIMPLE == array_type )
	{
		*size_o_big_x = dwt_util_pow2_ceil_log2(*size_o_big_x);
		*size_o_big_y = dwt_util_pow2_ceil_log2(*size_o_big_y);
	}

	if( DWT_ARR_SIMPLE == array_type )
	{
		*size_i_big_x = *size_i_big_x;
		*size_i_big_y = *size_i_big_y;
	}

	FUNC_END;
}

void dwt_util_get_sizes_s(
	enum dwt_array array_type,
	int size_x,
	int size_y,
	int opt_stride,
	int *stride_x,
	int *stride_y,
	int *size_o_big_x,
	int *size_o_big_y,
	int *size_i_big_x,
	int *size_i_big_y)
{
	FUNC_BEGIN;

	assert( size_x > 0 && size_y > 0 );

	assert( stride_x && stride_y && size_o_big_x && size_o_big_y && size_i_big_x && size_i_big_y );

	*stride_y = sizeof(float);
	*stride_x = dwt_util_get_stride(
		(*stride_y) * dwt_util_pow2_ceil_log2(size_x), opt_stride);

	*size_o_big_x = size_x;
	*size_o_big_y = size_y;
	*size_i_big_x = size_x;
	*size_i_big_y = size_y;

	if( DWT_ARR_SPARSE == array_type || DWT_ARR_SIMPLE == array_type )
	{
		*size_o_big_x = dwt_util_pow2_ceil_log2(*size_o_big_x);
		*size_o_big_y = dwt_util_pow2_ceil_log2(*size_o_big_y);
	}

	if( DWT_ARR_SIMPLE == array_type )
	{
		*size_i_big_x = *size_i_big_x;
		*size_i_big_y = *size_i_big_y;
	}

	FUNC_END;
}

void dwt_util_get_sizes_d(
	enum dwt_array array_type,
	int size_x,
	int size_y,
	int opt_stride,
	int *stride_x,
	int *stride_y,
	int *size_o_big_x,
	int *size_o_big_y,
	int *size_i_big_x,
	int *size_i_big_y)
{
	FUNC_BEGIN;

	assert( size_x > 0 && size_y > 0 );

	assert( stride_x && stride_y && size_o_big_x && size_o_big_y && size_i_big_x && size_i_big_y );

	*stride_y = sizeof(double);
	*stride_x = dwt_util_get_stride(
		(*stride_y) * dwt_util_pow2_ceil_log2(size_x), opt_stride);

	*size_o_big_x = size_x;
	*size_o_big_y = size_y;
	*size_i_big_x = size_x;
	*size_i_big_y = size_y;

	if( DWT_ARR_SPARSE == array_type || DWT_ARR_SIMPLE == array_type )
	{
		*size_o_big_x = dwt_util_pow2_ceil_log2(*size_o_big_x);
		*size_o_big_y = dwt_util_pow2_ceil_log2(*size_o_big_y);
	}

	if( DWT_ARR_SIMPLE == array_type )
	{
		*size_i_big_x = *size_i_big_x;
		*size_i_big_y = *size_i_big_y;
	}

	FUNC_END;
}

// 1.618, 1.333, 1.28, 1.13, 1.06, 1.02
const float g_growth_factor_s = 1.28f;
const float g_growth_factor_d = 1.28;

void dwt_util_measure_perf_cdf97_1_s(
	enum dwt_array array_type,
	int min_x,
	int max_x,
	int opt_stride,
	int j_max,
	int decompose_one,
	int zero_padding,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data
)
{
	FUNC_BEGIN;

	assert( min_x > 0 && min_x < max_x );

	assert( M > 0 && N > 0 );

	assert( fwd_plot_data && inv_plot_data );

	const float growth_factor = g_growth_factor_s;

	// for x = min_x to max_x
	for(int x = min_x; x <= max_x; x = /*ceilf*/DWT_CEIL_S(x * growth_factor))
	{
		// fixed y
		const int y = 1;

		int stride_x;
		int stride_y;
		int size_o_big_x;
		int size_o_big_y;
		int size_i_big_x;
		int size_i_big_y;

		// get sizes
		dwt_util_get_sizes_s(
			array_type,
			x, y,
			opt_stride,
		        &stride_x,
			&stride_y,
			&size_o_big_x,
			&size_o_big_y,
			&size_i_big_x,
			&size_i_big_y
		);

		dwt_util_log(LOG_DBG, "performance test for [%ix%i] in [%ix%i] with strides (%i, %i)...\n", size_i_big_x, size_i_big_y, size_o_big_x, size_o_big_y, stride_x, stride_y);

		float fwd_secs;
		float inv_secs;

		// call perf()
		dwt_util_perf_cdf97_2_s(
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y,
			size_i_big_x,
			size_i_big_y,
			j_max,
			decompose_one,
			zero_padding,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs
		);

#ifdef MEASURE_PER_PIXEL
		const int denominator = x*y;
#else
		const int denominator = 1;
#endif

		// printf into file
		fprintf(fwd_plot_data, "%i\t%.10f\n", x*y, fwd_secs/denominator);
		fprintf(inv_plot_data, "%i\t%.10f\n", x*y, inv_secs/denominator);
	}

	FUNC_END;
}

void dwt_util_measure_perf_cdf97_1_d(
	enum dwt_array array_type,
	int min_x,
	int max_x,
	int opt_stride,
	int j_max,
	int decompose_one,
	int zero_padding,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data
)
{
	FUNC_BEGIN;

	assert( min_x > 0 && min_x < max_x );

	assert( M > 0 && N > 0 );

	assert( fwd_plot_data && inv_plot_data );

	const double growth_factor = g_growth_factor_d;

	// for x = min_x to max_x
	for(int x = min_x; x <= max_x; x = /*ceil*/DWT_CEIL_D(x * growth_factor))
	{
		// fixed y
		const int y = 1;

		int stride_x;
		int stride_y;
		int size_o_big_x;
		int size_o_big_y;
		int size_i_big_x;
		int size_i_big_y;

		// get sizes
		dwt_util_get_sizes_d(
			array_type,
			x, y,
			opt_stride,
		        &stride_x,
			&stride_y,
			&size_o_big_x,
			&size_o_big_y,
			&size_i_big_x,
			&size_i_big_y
		);

		dwt_util_log(LOG_DBG, "performance test for [%ix%i] in [%ix%i] with strides (%i, %i)...\n", size_i_big_x, size_i_big_y, size_o_big_x, size_o_big_y, stride_x, stride_y);

		double fwd_secs;
		double inv_secs;

		// call perf()
		dwt_util_perf_cdf97_2_d(
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y,
			size_i_big_x,
			size_i_big_y,
			j_max,
			decompose_one,
			zero_padding,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs
		);

		// printf into file
		fprintf(fwd_plot_data, "%i\t%.10f\n", x*y, fwd_secs);
		fprintf(inv_plot_data, "%i\t%.10f\n", x*y, inv_secs);

	}

	FUNC_END;
}

void dwt_util_measure_perf_cdf97_2_s(
	enum dwt_array array_type,
	int min_x,
	int max_x,
	int opt_stride,
	int j_max,
	int decompose_one,
	int zero_padding,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data
)
{
	FUNC_BEGIN;

	assert( min_x > 0 && min_x < max_x );

	assert( M > 0 && N > 0 );

	assert( fwd_plot_data && inv_plot_data );

	const float growth_factor = g_growth_factor_s;

	// for x = min_x to max_x
	for(int x = min_x; x <= max_x; x = /*ceilf*/DWT_CEIL_S(x * growth_factor))
	{
		// y is equal to x
		const int y = x;

		int stride_x;
		int stride_y;
		int size_o_big_x;
		int size_o_big_y;
		int size_i_big_x;
		int size_i_big_y;

		// get sizes
		dwt_util_get_sizes_s(
			array_type,
			x, y,
			opt_stride,
		        &stride_x,
			&stride_y,
			&size_o_big_x,
			&size_o_big_y,
			&size_i_big_x,
			&size_i_big_y
		);

		dwt_util_log(LOG_DBG, "performance test for [%ix%i] in [%ix%i] with strides (%i, %i)...\n", size_i_big_x, size_i_big_y, size_o_big_x, size_o_big_y, stride_x, stride_y);

		float fwd_secs;
		float inv_secs;

		// call perf()
		dwt_util_perf_cdf97_2_s(
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y,
			size_i_big_x,
			size_i_big_y,
			j_max,
			decompose_one,
			zero_padding,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs
		);

#ifdef MEASURE_PER_PIXEL
		const int denominator = x*y;
#else
		const int denominator = 1;
#endif

		// printf into file
		fprintf(fwd_plot_data, "%i\t%.10f\n", x*y, fwd_secs/denominator);
		fprintf(inv_plot_data, "%i\t%.10f\n", x*y, inv_secs/denominator);

	}

	FUNC_END;
}

void dwt_util_measure_perf_cdf97_2_d(
	enum dwt_array array_type,
	int min_x,
	int max_x,
	int opt_stride,
	int j_max,
	int decompose_one,
	int zero_padding,
	int M,
	int N,
	int clock_type,
	FILE *fwd_plot_data,
	FILE *inv_plot_data
)
{
	FUNC_BEGIN;

	assert( min_x > 0 && min_x < max_x );

	assert( M > 0 && N > 0 );

	assert( fwd_plot_data && inv_plot_data );

	const double growth_factor = g_growth_factor_d;

	// for x = min_x to max_x
	for(int x = min_x; x <= max_x; x = /*ceil*/DWT_CEIL_D(x * growth_factor))
	{
		// y is equal to x
		const int y = x;

		int stride_x;
		int stride_y;
		int size_o_big_x;
		int size_o_big_y;
		int size_i_big_x;
		int size_i_big_y;

		// get sizes
		dwt_util_get_sizes_d(
			array_type,
			x, y,
			opt_stride,
		        &stride_x,
			&stride_y,
			&size_o_big_x,
			&size_o_big_y,
			&size_i_big_x,
			&size_i_big_y
		);

		dwt_util_log(LOG_DBG, "performance test for [%ix%i] in [%ix%i] with strides (%i, %i)...\n", size_i_big_x, size_i_big_y, size_o_big_x, size_o_big_y, stride_x, stride_y);

		double fwd_secs;
		double inv_secs;

		// call perf()
		dwt_util_perf_cdf97_2_d(
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y,
			size_i_big_x,
			size_i_big_y,
			j_max,
			decompose_one,
			zero_padding,
			M,
			N,
			clock_type,
			&fwd_secs,
			&inv_secs
		);

		// printf into file
		fprintf(fwd_plot_data, "%i\t%.10f\n", x*y, fwd_secs);
		fprintf(inv_plot_data, "%i\t%.10f\n", x*y, inv_secs);

	}

	FUNC_END;
}

float dwt_util_band_wps_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	int j
)
{
	float sum = 0.0f;

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			const float *coeff = dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y);

			// ^2
			sum += *coeff * *coeff;
		}

	// rectification
	// Liu, Y., X.S. Liang, and R.H. Weisberg, 2007: Rectification of the bias in the wavelet power spectrum. Journal of Atmospheric and Oceanic Technology, 24(12), 2093-2102.
	// http://ocgweb.marine.usf.edu/~liu/wavelet.html
	// http://ocgweb.marine.usf.edu/~liu/Papers/Liu_etal_2007_JAOT_wavelet.pdf
	sum /= 1<<j;

	return sum;
}

static
int cmp_s(const void *p1, const void *p2)
{
	return (const float *)p1 > (const float *)p2;
}

float dwt_util_band_med_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	int j
)
{
	UNUSED(j);

	const int size = size_x * size_y;

	// FIXME: alloc
	float arr[size];

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			// FIXME: memcpy
			arr[y * size_x + x] = *dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y);
		}


	qsort(arr, size, sizeof(float), cmp_s);

	return arr[size/2];
}

int dwt_util_count_subbands_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max
)
{
	int count = 0;

	for(int j = 1; j < j_max; j++)
	{
		const void *band_ptr;
		int band_x;
		int band_y;
		
		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HL, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			count++;

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			count++;

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			count++;
	}

	return count;
}

void dwt_util_wps_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	float *fv
)
{
	int count = 0;

	for(int j = 1; j < j_max; j++)
	{
		const void *band_ptr;
		int band_x;
		int band_y;
		
		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HL, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_wps_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_wps_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_wps_s(band_ptr, stride_x, stride_y, band_x, band_y, j);
	}
}

void dwt_util_med_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	float *fv
)
{
	int count = 0;

	for(int j = 1; j < j_max; j++)
	{
		const void *band_ptr;
		int band_x;
		int band_y;
		
		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HL, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_med_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_med_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_med_s(band_ptr, stride_x, stride_y, band_x, band_y, j);
	}
}

float dwt_util_band_maxidx_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	int j
)
{
	UNUSED(j);

	int idx = -1;
	float val;

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			float coeff = fabsf(*dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y));

			if( -1 == idx || coeff > val )
			{
				val = coeff;
				idx = y * size_x + x;
			}
		}

	return (float)idx;
}

float dwt_util_band_mean_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	int j
)
{
	UNUSED(j);

	float sum = 0.0f;

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			float coeff = *dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y);

			sum += coeff;
		}

	sum /= size_x * size_y;

	return sum;
}

void dwt_util_maxidx_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	float *fv
)
{
	int count = 0;

	for(int j = 1; j < j_max; j++)
	{
		const void *band_ptr;
		int band_x;
		int band_y;
		
		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HL, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_maxidx_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_maxidx_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_maxidx_s(band_ptr, stride_x, stride_y, band_x, band_y, j);
	}
}

void dwt_util_mean_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	float *fv
)
{
	int count = 0;

	for(int j = 1; j < j_max; j++)
	{
		const void *band_ptr;
		int band_x;
		int band_y;
		
		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HL, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_mean_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_mean_s(band_ptr, stride_x, stride_y, band_x, band_y, j);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_mean_s(band_ptr, stride_x, stride_y, band_x, band_y, j);
	}
}