/**
 * @file
 * @author David Barina <ibarina@fit.vutbr.cz>
 * @brief Fast wavelet transform implemented via lifting scheme.
 */
#include "libdwt.h"

#define MEASURE_PER_PIXEL
//#define DEBUG_VERBOSE
//#define DISABLE_MEMCPY
//#define ENABLE_LAZY_MEMCPY
//#define DISABLE_Y
//#define DISABLE_X
//#define MEASURE_FACTOR 1000
#define MEASURE_FACTOR 1
//#define FV_ON_MAGNITUDES

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
#if (GCC_VERSION >= 40800) && (GCC_VERSION < 40900)
	#warning "GCC 4.8"
#endif
#if (GCC_VERSION >= 40900)
	#warning "GCC 4.9+"
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
#ifdef __asvp__
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
#include "inline.h"

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

#include <stdlib.h> // abort, malloc, free, qsort
#include <limits.h> // CHAR_BIT
// NOTE: -lm
#include <math.h> // fabs, fabsf, isnan, isinf, powf
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

#ifdef microblaze
inline
float powf(
	float x,
	float y
)
{
	return __builtin_powf(x, y);
}
#endif

#define ASM_MARKER __asm volatile ("# MARKER: " QUOTE(__LINE__))

#ifdef __asvp__
/** total number of workers available */
const int dwt_util_global_total_workers = BCE_DMA_CFGTABLE_NUM_ITEMS;

static int get_total_workers()
{
	return dwt_util_global_total_workers;
}
#endif

/** how many workers use for computation (can be less than total number of workers) */
#ifdef __asvp__
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
void set_active_workers(
	int active_workers
)
{
	dwt_util_global_active_workers = active_workers;
}

static
size_t alignment(
	size_t type_size
)
{
	assert( type_size );

#ifdef microblaze
	// DMA memory transfers seems to need alignment of 2*sizeof(float) = 8
	// http://www.xilinx.com/support/documentation/sw_manuals/mb_ref_guide.pdf => Memory Architecture
	UNUSED(type_size);
	return 8;
#endif

#ifdef __x86_64__
	// due to SSE memory access sizeof(__m128) = 16
	// FIXME: this should return proper value according to accel_type (not for each implementation the SSE alignment is needed)
	UNUSED(type_size);
	return 16;
#endif

#ifdef __arm__
	// http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0301h/Cdfifaec.html
	// FIXME: according to manual, should be "type_size" only. however, this fails on RasPi atleast (for floats, doubles and int are OK)
	return 2*type_size;
#endif

	// fallback: unaligned data
	return 1;
}

size_t dwt_util_alignment(
	size_t type_size
)
{
	return alignment(type_size);
}

static
int is_aligned_s(
	const void *ptr
)
{
	const size_t alignment = dwt_util_alignment(sizeof(float));

	return ( (intptr_t)ptr & (intptr_t)(alignment-1) ) ? 0 : 1;
}

static
void *align(
	void *ptr,
	size_t alignment
)
{
	return (void *)( ((intptr_t)ptr+(alignment-1)) & (~(alignment-1)) );
}

/** in bytes; offset in src[] and dst[] is given by worker_id * dwt_util_global_data_step */
ptrdiff_t dwt_util_global_data_step = 0;

static
ptrdiff_t get_data_step_s()
{
	return dwt_util_global_data_step;
}

static
void set_data_step_s(
	ptrdiff_t data_step
)
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
void set_temp_step(
	int temp_step
)
{
	dwt_util_global_temp_step = temp_step;
}

/** active firmware in all ASVP acceleration units */
enum dwt_op dwt_util_global_active_op = DWT_OP_NONE;

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

void dwt_util_flush_cache(
	void *addr,
	size_t size
)
{
	flush_cache(addr, size);
}

static inline
void flush_cache_s(
	float *addr,
	size_t size
)
{
	flush_cache( (void *)addr, size * sizeof(float) );
}

#ifdef __asvp__
void wal_abort(
	const char *str,
	int res
)
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
void set_accel_type(
	int accel_type
)
{
	dwt_util_global_accel_type = accel_type;
}

static
int get_accel_type()
{
	return dwt_util_global_accel_type;
}

int dwt_util_get_accel()
{
	return get_accel_type();
}

#include "inline.h"

int dwt_util_ceil_log2(
	int x
)
{
	return ceil_log2(x);
}

int dwt_util_pow2_ceil_log2(
	int x
)
{
	return pow2_ceil_log2(x);
}

#include "inline.h"

int dwt_util_ceil_div(
	int x,
	int y
)
{
	return ceil_div(x, y);
}

int dwt_util_floor_div(
	int x,
	int y
)
{
	return floor_div(x, y);
}

#include "inline.h"

int dwt_util_to_even(
	int x
)
{
	return to_even(x);
}

int dwt_util_up_to_even(
	int x
)
{
	return up_to_even(x);
}

int dwt_util_up_to_mul4(
	int x
)
{
	return up_to_mul4(x);
}

int dwt_util_to_odd(
	int x
)
{
	return to_odd(x);
}

static
int is_aligned_4(
	const void *ptr
)
{
	return ( (intptr_t)ptr&(intptr_t)(4-1) ) ? 0 : 1;
}

static
int is_aligned_8(
	const void *ptr
)
{
	return ( (intptr_t)ptr&(intptr_t)(8-1) ) ? 0 : 1;
}

static
int is_aligned_16(
	const void *ptr
)
{
	return ( (intptr_t)ptr&(intptr_t)(16-1) ) ? 0 : 1;
}

static
intptr_t align_4(
	intptr_t p
)
{
	return (p+(4-1))&(~(4-1));
}

static
intptr_t align_8(
	intptr_t p
)
{
	return (p+(8-1))&(~(8-1));
}

static
intptr_t align_16(
	intptr_t p
)
{
	return (p+(16-1))&(~(16-1));
}

static
intptr_t align_64(
	intptr_t p
)
{
	return (p+(64-1))&(~(64-1));
}

static
intptr_t align_4096(
	intptr_t p
)
{
	return (p+(4096-1))&(~(4096-1));
}

intptr_t dwt_util_align_4(
	intptr_t p
)
{
	return align_4(p);
}

intptr_t dwt_util_align_8(
	intptr_t p
)
{
	return align_8(p);
}

intptr_t dwt_util_align_16(
	intptr_t p
)
{
	return align_16(p);
}

static
intptr_t align_int(
	intptr_t addr,
	size_t alignment
)
{
	return (intptr_t)align((void *)addr, alignment);
}

static
int temp_calc_internal(
	size_t alignment,	///< alignment (bytes)
	size_t elem_size,	///< element size (bytes)
	int offset,		///< offset (elements)
	int elements,		///< number of elements (elements)
	int worker		///< worker_id or total_workers (workers)
)
{
	const int padding = 1; // elements
	const int offset1 = offset*elem_size; // bytes
	const int offset2 = (alignment-offset1) + align_int(padding*elem_size, alignment); // bytes
	const int size = elements*elem_size; // bytes
	const int block_size = align_int(offset2 + size + padding*elem_size, alignment); // bytes

	const int total_bytes = block_size * worker + offset2; // bytes
	const int total_elems = total_bytes/elem_size; // elements

	return total_elems;
}

/**
 * @note should return in bytes, only for compatibility in elements
 * @returns in elements
 */
static
int calc_and_set_temp_size_s(
	int elements,		///< number of elements (floats)
	int offset		///< in elements, e.g. +1 float
)
{
	const size_t elem_size = sizeof(float); // bytes
	const int workers = get_active_workers(); // workers
	const size_t alignment = dwt_util_alignment(sizeof(float)); // bytes

	set_temp_step(elements); // elements

	return temp_calc_internal(alignment, elem_size, offset, elements, workers);
}

#include "system.h" // is_aligned

static
void *ptralign_down(
	void *ptr,
	size_t alignment
)
{
	return (void *)( (intptr_t)ptr & ~(alignment-1) );
}

static
float *calc_temp_offset2_s(
	float *addr,	///< pointer to temp[] or temp[]+offset
	int worker_id,	///< identifier of current worker
	int offset	///< offset
)
{
	const size_t elem_size = sizeof(float); // bytes
	const size_t alignment = dwt_util_alignment(sizeof(float)); // bytes

	int return_offset = 0; // elements

	// correct the alignment
	if( !is_aligned(addr, alignment) )
	{
		float *old = addr;
		addr = (float *)ptralign_down(addr, alignment);
		ptrdiff_t ptrdiff = (intptr_t)old - (intptr_t)addr;
		return_offset = ptrdiff / elem_size;
	}

	// if requested with offset then offset cannot be zero
	if( 0 == offset && 0 != return_offset )
	{
		offset = return_offset;
	}

	const int elements = get_temp_step(); // elements

	return addr + temp_calc_internal(alignment, elem_size, offset, elements, worker_id) + return_offset;
}

#include "inline.h"

int *dwt_util_addr_coeff_i(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return addr2_i(ptr, y, x, stride_x, stride_y);
}

const int *dwt_util_addr_coeff_const_i(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return addr2_const_i(ptr, y, x, stride_x, stride_y);
}

float *dwt_util_addr_coeff_s(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return addr2_s(ptr, y, x, stride_x, stride_y);
}

const float *dwt_util_addr_coeff_const_s(
	const void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return addr2_const_s(ptr, y, x, stride_x, stride_y);
}

double *dwt_util_addr_coeff_d(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return addr2_d(ptr, y, x, stride_x, stride_y);
}

void *dwt_util_addr_coeff(
	void *ptr,
	int y,
	int x,
	int stride_x,
	int stride_y
)
{
	return addr2(ptr, y, x, stride_x, stride_y);
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
	int type
)
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
	int type
)
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
	int type
)
{
	x++;
	y++;

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
		case 2:
		{
			int i = x^y;
			i &= 0xff;
			*dest = (float)i/32;
			break;
		}
		case 3:
		{
			int v = (((x&1)<<1)|(y&1))+1;
			*dest = v/4.f;
			break;
		}
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
	int rand
)
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
	int rand
)
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

void dwt_util_test_image_fill2_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y,
	int rand,
	int type
)
{
	assert( NULL != ptr );

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
			dwt_util_test_image_value_i_i(
				addr2_i(ptr, y, x, stride_x, stride_y),
				x,
				y,
				rand,
				type
			);
}

void dwt_util_test_image_fill_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y,
	int rand
)
{
	FUNC_BEGIN;

	assert( NULL != ptr );

	const int type = 0;

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
			dwt_util_test_image_value_i_s(
				addr2_s(ptr, y, x, stride_x, stride_y),
				x,
				y,
				rand,
				type
			);

	FUNC_END;
}

void dwt_util_test_image_fill2_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y,
	int rand,
	int type
)
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
				type
			);

	FUNC_END;
}

void dwt_util_test_image_zero_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y
)
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
	int size_o_big_y
)
{
	UNUSED(stride_y);
	UNUSED(size_o_big_x);

	return stride_x * size_o_big_y;
}

size_t dwt_util_image_size(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y
)
{
	return image_size(
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);
}

// TODO: this function should return a pointer
void dwt_util_alloc_image(
	void **pptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y
)
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

void *dwt_util_alloc_image2(
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	void *ptr;

	dwt_util_alloc_image(
		&ptr,
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	return ptr;
}

void dwt_util_free_image(
	void **pptr
)
{
	assert( pptr != NULL );

	free(*pptr);
	*pptr = NULL;
}

static
int is_nan_or_inf_d(double x)
{
#ifdef microblaze
	return ( ((*(uint32_t *)(void *)&x)>>20) & 0x7ff ) == 0x7ff;
#else
	return isnan(x) || isinf(x);
#endif
}

int dwt_util_compare_d(
	void *ptr1,
	void *ptr2,
	int stride_x,
	int stride_y,
	int size_i_big_x,
	int size_i_big_y
)
{
	assert( ptr1 != NULL && ptr2 != NULL && size_i_big_x >= 0 && size_i_big_y >= 0 );

	const double eps = 1e-6;

	for(int y = 0; y < size_i_big_y; y++)
		for(int x = 0; x < size_i_big_x; x++)
		{
			const double a = *addr2_d(ptr1, y, x, stride_x, stride_y);
			const double b = *addr2_d(ptr2, y, x, stride_x, stride_y);

			if( is_nan_or_inf_d(a) || is_nan_or_inf_d(b) )
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
	int size_i_big_y
)
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

int dwt_util_compare2_i(
	void *ptr1,
	void *ptr2,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int size_x,
	int size_y
)
{
	assert( ptr1 != NULL && ptr2 != NULL && size_x >= 0 && size_y >= 0 );

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const int a = *addr2_i(ptr1, y, x, stride1_x, stride1_y);
			const int b = *addr2_i(ptr2, y, x, stride2_x, stride2_y);

			if( abs(a - b) > 0 )
			{
#ifdef DEBUG
				dwt_util_log(LOG_DBG, "%s: %i != %i at (x=%i, y=%i)\n", __FUNCTION__, a, b, x, y);
#endif
				return 1;
			}
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
	int size_i_big_y
)
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

int dwt_util_compare2_s(
	void *ptr1,
	void *ptr2,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int size_x,
	int size_y
)
{
	assert( ptr1 != NULL && ptr2 != NULL && size_x >= 0 && size_y >= 0 );

	int ret = 0;

	const float eps = 1.e-3f;

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float a = *addr2_s(ptr1, y, x, stride1_x, stride1_y);
			const float b = *addr2_s(ptr2, y, x, stride2_x, stride2_y);

			if( isnan(a) || isinf(a) || isnan(b) || isinf(b) )
			{
#ifdef COMPARE_DESTROY
				*addr2_s(ptr1, y, x, stride1_x, stride1_y) = 1.f;
#endif
				ret = 1;
			}

			if( fabsf(a - b) > eps )
			{
#ifdef COMPARE_DESTROY
				*addr2_s(ptr1, y, x, stride1_x, stride1_y) = 1.f;
#endif
				ret = 1;
			}
			else
			{
#ifdef COMPARE_DESTROY
				*addr2_s(ptr1, y, x, stride1_x, stride1_y) = 0.f;
#endif
			}
		}
	}

	return ret;
}

int dwt_util_compare2_destructive_s(
	void *ptr1,
	const void *ptr2,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int size_x,
	int size_y
)
{
	assert( ptr1 != NULL && ptr2 != NULL && size_x >= 0 && size_y >= 0 );

	int ret = 0;

	const float eps = 1.e-3f;

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float a = *addr2_s      (ptr1, y, x, stride1_x, stride1_y);
			const float b = *addr2_const_s(ptr2, y, x, stride2_x, stride2_y);

			float *dest = addr2_s(ptr1, y, x, stride1_x, stride1_y);

			*dest = 0.f;

			if( isnan(a) || isinf(a) || isnan(b) || isinf(b) )
			{
				*dest = 1.f;
				ret = 1;
			}

			if( fabsf(a - b) > eps )
			{
				*dest = 1.f;
				ret = 1;
			}
		}
	}

	return ret;
}

int dwt_util_compare2_destructive2_s(
	void *ptr1,
	const void *ptr2,
	void *map,
	int stride1_x,
	int stride1_y,
	int stride2_x,
	int stride2_y,
	int map_stride_x,
	int map_stride_y,
	int size_x,
	int size_y
)
{
	assert( ptr1 && ptr2 && map && size_x >= 0 && size_y >= 0 );

	int ret = 0;

	const float eps = 1.e-3f;

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float a = *addr2_s      (ptr1, y, x, stride1_x, stride1_y);
			const float b = *addr2_const_s(ptr2, y, x, stride2_x, stride2_y);

			float *dest = addr2_s(map, y, x, map_stride_x, map_stride_y);

			*dest = 0.f;

			if( isnan(a) || isinf(a) || isnan(b) || isinf(b) )
			{
				*dest = 1.f;
				ret = 1;
			}

			if( fabsf(a - b) > eps )
			{
				*dest = 1.f;
				ret = 1;
			}
		}
	}

	return ret;
}

void dwt_cdf97_f_d(
	const double *src,
	double *dst,
	double *tmp,
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int N
)
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
	int stride
)
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
	int stride
)
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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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

#define ASSUME_ALIGNED_S(lvalueptr) ASSUME_ALIGNED((lvalueptr), alignment(sizeof(float)))
#define ASSUME_ALIGNED_D(lvalueptr) ASSUME_ALIGNED((lvalueptr), alignment(sizeof(double)))
#define ASSUME_ALIGNED_I(lvalueptr) ASSUME_ALIGNED((lvalueptr), alignment(sizeof(int)))

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

	if( scaling < 0 )
	{
		// inv
		for(int w = 0; w < dwt_util_get_num_workers(); w++)
		{
			float *arr_local = ASSUME_ALIGNED_S(calc_temp_offset2_s(arr, w, 0));

			assert( is_aligned_s(arr_local) );

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
			float *arr_local = ASSUME_ALIGNED_S(calc_temp_offset2_s(arr, w, 0));

			assert( is_aligned_s(arr_local) );

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

static
void accel_lift_op4s_main_stride_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	FUNC_BEGIN;

	assert( steps >= 0 );

	if( scaling < 0 )
	{
		// inv
		assert( 1 == dwt_util_get_num_workers() );
		{
			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// descale
			float *out = addr1_s(arr, 4, stride);

			for(int s = 0; s < steps; s++)
			{
				*addr1_s(out, 0, stride) *= v[0];
				*addr1_s(out, 1, stride) *= v[1];

				out = addr1_s(out, 2, stride);
			}

			// operations
			for(int off = 4; off >= 1; off--)
			{
				float *out = addr1_s(arr, off, stride);
				const float c = w[off-1];

				for(int s = 0; s < steps; s++)
				{
					*addr1_s(out, 0, stride) += c * (*addr1_s(out, -1, stride) + *addr1_s(out, +1, stride));

					out = addr1_s(out, 2, stride);
				}
			}
		}
	}
	else if( scaling > 0 )
	{
		// fwd
		assert( 1 == dwt_util_get_num_workers() );
		{
			// constants
			const float w[4] = { delta, gamma, beta, alpha };
			const float v[2] = { 1/zeta, zeta };

			// operations
			for(int off = 4; off >= 1; off--)
			{
				float *out = addr1_s(arr, off, stride);
				const float c = w[off-1];

				for(int s = 0; s < steps; s++)
				{
					*addr1_s(out, 0, stride) += c * (*addr1_s(out, -1, stride) + *addr1_s(out, +1, stride));

					out = addr1_s(out, 2, stride);
				}
			}

			// scale
			float *out = addr1_s(arr, 0, stride);

			for(int s = 0; s < steps; s++)
			{
				*addr1_s(out, 0, stride) *= v[0];
				*addr1_s(out, 1, stride) *= v[1];

				out = addr1_s(out, 2, stride);
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

/**
 * horizontal vectorisation (multi-loop approach), forward transform
 */
static
void accel_lift_op4s_fwd_main_stride_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	FUNC_BEGIN;

	assert( steps >= 0 );
	assert( scaling > 0 );

	// constants
	const float w[4] = { delta, gamma, beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// operations
	for(int off = 4; off >= 1; off--)
	{
		float *out = addr1_s(arr, off, stride);
		const float c = w[off-1];

		for(int s = 0; s < steps; s++)
		{
			*addr1_s(out, 0, stride) += c * (*addr1_s(out, -1, stride) + *addr1_s(out, +1, stride));

			out = addr1_s(out, 2, stride);
		}
	}

	// scale
	float *out = addr1_s(arr, 0, stride);

	for(int s = 0; s < steps; s++)
	{
		*addr1_s(out, 0, stride) *= v[0];
		*addr1_s(out, 1, stride) *= v[1];

		out = addr1_s(out, 2, stride);
	}

	FUNC_END;
}

/**
 * horizontal vectorisation (multi-loop approach), inverse transform
 */
static
void accel_lift_op4s_inv_main_stride_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	FUNC_BEGIN;

	assert( steps >= 0 );
	assert( scaling < 0 );

	// constants
	const float w[4] = { delta, gamma, beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// descale
	float *out = addr1_s(arr, 4, stride);

	for(int s = 0; s < steps; s++)
	{
		*addr1_s(out, 0, stride) *= v[0];
		*addr1_s(out, 1, stride) *= v[1];

		out = addr1_s(out, 2, stride);
	}

	// operations
	for(int off = 4; off >= 1; off--)
	{
		float *out = addr1_s(arr, off, stride);
		const float c = w[off-1];

		for(int s = 0; s < steps; s++)
		{
			*addr1_s(out, 0, stride) += c * (*addr1_s(out, -1, stride) + *addr1_s(out, +1, stride));

			out = addr1_s(out, 2, stride);
		}
	}

	FUNC_END;
}

static
void accel_lift_op2s_inv_main_stride_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float zeta,
	int scaling,
	int stride
)
{
	FUNC_BEGIN;

	assert( steps >= 0 );
	assert( scaling < 0 );

	// constants
	const float w[2] = { beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// descale
	float *out = addr1_s(arr, 2, stride);

	for(int s = 0; s < steps; s++)
	{
		*addr1_s(out, 0, stride) *= v[0];
		*addr1_s(out, 1, stride) *= v[1];

		out = addr1_s(out, 2, stride);
	}

	// operations
	for(int off = 2; off >= 1; off--)
	{
		float *out = addr1_s(arr, off, stride);
		const float c = w[off-1];

		for(int s = 0; s < steps; s++)
		{
			*addr1_s(out, 0, stride) += c * (*addr1_s(out, -1, stride) + *addr1_s(out, +1, stride));

			out = addr1_s(out, 2, stride);
		}
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
			arr_local[worker] = calc_temp_offset2_s(arr, worker, 0);
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
			arr_local[worker] = calc_temp_offset2_s(arr, worker, 0);
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
void op4s_sdl_import_stride_s_ref(float *l, const float *restrict addr, int idx, int stride)
{
	l[idx] = *addr1_const_s(addr, idx, stride);
}

#ifdef __SSE__
#define op4s_sdl_import_stride_s_sse(l, addr, idx, stride) \
do { \
	l[idx] = *addr1_const_s(addr, idx, stride); \
} while(0)
#endif

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
void op4s_sdl_load_stride_s_ref(float *in, const float *restrict addr, int stride)
{
	in[0] = *addr1_const_s(addr,0,stride);
	in[1] = *addr1_const_s(addr,1,stride);
}

#ifdef __SSE__
#define op4s_sdl_load_stride_s_sse(in, addr, stride) \
do { \
	in[0] = *addr1_const_s(addr,0,stride); \
	in[1] = *addr1_const_s(addr,1,stride); \
} while(0)
#endif

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
void op4s_sdl_save_stride_s_ref(float *out, float *restrict addr, int stride)
{
	*addr1_s(addr,0,stride) = out[0];
	*addr1_s(addr,1,stride) = out[1];
}

#ifdef __SSE__
#define op4s_sdl_save_stride_s_sse(out, addr, stride) \
do { \
	*addr1_s(addr,0,stride) = out[0]; \
	*addr1_s(addr,1,stride) = out[1]; \
} while(0)
#endif

static
void op4s_sdl_export_s_ref(const float *l, float *restrict addr, int idx)
{
	addr[idx] = l[idx];
}

static
void op4s_sdl_export_stride_s_ref(const float *l, float *restrict addr, int idx, int stride)
{
	*addr1_s(addr,idx,stride) = l[idx];
}

#ifdef __SSE__
#define op4s_sdl_export_stride_s_sse(l,addr,idx,stride) \
do { \
	*addr1_s(addr,idx,stride) = l[idx]; \
} while(0)
#endif

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
void op4s_sdl_pass_fwd_prolog_stride_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	UNUSED(v);
	UNUSED(out);

	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_stride_s_ref(in, addr1_s(*addr,4,stride), stride);

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
	*addr = addr1_s(*addr,2,stride);
}

#ifdef __SSE__
#define op4s_sdl_pass_fwd_prolog_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride) \
do { \
	op4s_sdl2_shuffle_s_sse(c, r); \
	op4s_sdl_load_stride_s_sse(in, addr1_s(*addr,4,stride), stride); \
	op4s_sdl2_input_low_s_sse(in, c, r); \
	op4s_sdl2_op_s_sse(z, c, w, l, r); \
	op4s_sdl2_update_s_sse(c, l, r, z); \
	*addr = addr1_s(*addr,2,stride); \
} while(0)
#endif

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
void op4s_sdl_pass_fwd_core_stride_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
{
	// shuffle
	op4s_sdl_shuffle_s_ref(c, r);

	// load
	op4s_sdl_load_stride_s_ref(in, addr1_s(*addr,4,stride), stride);

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
	op4s_sdl_save_stride_s_ref(out, addr1_s(*addr,-6,stride), stride);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	*addr = addr1_s(*addr,2,stride);
}

#ifdef __SSE__
#define op4s_sdl_pass_fwd_core_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride) \
do { \
	op4s_sdl2_shuffle_s_sse(c, r); \
	op4s_sdl_load_stride_s_sse(in, addr1_s(*addr,4,stride), stride); \
	op4s_sdl2_input_low_s_sse(in, c, r); \
	op4s_sdl2_op_s_sse(z, c, w, l, r); \
	op4s_sdl2_output_low_s_sse(out, l, z); \
	op4s_sdl2_scale_s_sse(out, v); \
	op4s_sdl_save_stride_s_sse(out, addr1_s(*addr,-6,stride), stride); \
	op4s_sdl2_update_s_sse(c, l, r, z); \
	*addr = addr1_s(*addr,2,stride); \
} while(0)
#endif

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
void op4s_sdl_pass_fwd_epilog_stride_s_ref(const float *w, const float *v, float *l, float *c, float *r, float *z, float *in, float *out, float *restrict *addr, int stride)
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
	op4s_sdl_save_stride_s_ref(out, addr1_s(*addr,-6,stride), stride);

	// update
	op4s_sdl_update_s_ref(c, l, r, z);

	// pointers
	(*addr) = addr1_s(*addr,2,stride);
}

#ifdef __SSE__
#define op4s_sdl_pass_fwd_epilog_stride_s_sse(w,v,l,c,r,z,in,out,addr,stride) \
do { \
	op4s_sdl2_shuffle_s_sse(c, r); \
	op4s_sdl2_op_s_sse(z, c, w, l, r); \
	op4s_sdl2_output_low_s_sse(out, l, z); \
	op4s_sdl2_scale_s_sse(out, v); \
	op4s_sdl_save_stride_s_sse(out, addr1_s(*addr,-6,stride), stride); \
	op4s_sdl2_update_s_sse(c, l, r, z); \
	(*addr) = addr1_s(*addr,2,stride); \
} while(0)
#endif

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

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_16(addr) );
			float *base = addr;

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
				op4s_sdl6_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_inv_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_inv_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_inv_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_inv_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_inv_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl6_pass_inv_epilog_flush_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 0);
			}
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_16(addr) );
			float *base = addr;

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
				op4s_sdl6_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_fwd_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_fwd_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_fwd_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_fwd_epilog_light_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_fwd_epilog_full_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl6_pass_fwd_epilog_flush_s_sse(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_sse(l, &base[2*steps], 0);
			}
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

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_s(addr) );
			float *base = addr;

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
				op4s_sdl6_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_inv_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_inv_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_inv_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_inv_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_inv_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl6_pass_inv_epilog_flush_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 0);
			}
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_s(addr) );
			float *base = addr;

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
				op4s_sdl6_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_fwd_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_fwd_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_fwd_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl6_pass_fwd_epilog_light_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl6_pass_fwd_epilog_full_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl6_pass_fwd_epilog_flush_s_ref(w, v, l, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl6_export_s_ref(l, &base[2*steps], 0);
			}
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

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_s(addr) );
			float *base = addr;

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
				op4s_sdl2_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_inv_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_inv_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_inv_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_inv_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_inv_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl2_pass_inv_epilog_flush_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 0);
			}
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_s(addr) );
			float *base = addr;

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
				op4s_sdl2_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_fwd_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_fwd_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_fwd_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_fwd_epilog_light_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_fwd_epilog_full_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl2_pass_fwd_epilog_flush_s_ref(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_ref(l, &base[2*steps], 0);
			}
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

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_16(addr) );
			float *base = addr;

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
				op4s_sdl2_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_inv_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_inv_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_inv_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_inv_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_inv_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl2_pass_inv_epilog_flush_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 0);
			}
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			assert( is_aligned_16(addr) );
			float *base = addr;

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
				op4s_sdl2_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_fwd_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_fwd_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_fwd_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 0);
			}
			else
			{
				// epilog2: export(3)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 3);

				// epilog2: pass-epilog-light
				op4s_sdl2_pass_fwd_epilog_light_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(2)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 2);

				// epilog2: pass-epilog-full
				op4s_sdl2_pass_fwd_epilog_full_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(1)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 1);

				// epilog2: pass-epilog-flush
				op4s_sdl2_pass_fwd_epilog_flush_s_sse(w, v, l, c, r, z, in, out, &addr);

				// epilog2: export(0)
				op4s_sdl2_export_s_sse(l, &base[2*steps], 0);
			}
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

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			float *base = addr+0;
			assert( is_aligned_s(addr) );

			// *** prolog2 ***

			// prolog2: import(3)
			op4s_sdl_import_s_ref(l, base, 3);

			// prolog2: pass-prolog
			op4s_sdl_pass_inv_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// prolog2: import(2)
			op4s_sdl_import_s_ref(l, base, 2);

			// prolog2: pass-prolog
			op4s_sdl_pass_inv_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// prolog2: import(1)
			op4s_sdl_import_s_ref(l, base, 1);

			// prolog2: pass-prolog
			op4s_sdl_pass_inv_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// prolog2: import(0)
			op4s_sdl_import_s_ref(l, base, 0);

			// *** core ***

			// core: for s = 0 to S do
			for(int s = 0; s < S; s++)
			{
				// core: pass-core
				op4s_sdl_pass_inv_core_s_ref(w, v, l, c, r, z, in, out, &addr);

			}

			// *** epilog2 ***

			// epilog2: export(3)
			op4s_sdl_export_s_ref(l, &base[2*steps], 3);

			// epilog2: pass-epilog
			op4s_sdl_pass_inv_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl_export_s_ref(l, &base[2*steps], 2);

			// epilog2: pass-epilog
			op4s_sdl_pass_inv_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl_export_s_ref(l, &base[2*steps], 1);

			// epilog2: pass-epilog
			op4s_sdl_pass_inv_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl_export_s_ref(l, &base[2*steps], 0);
		}
	}
	else if ( scaling > 0 )
	{
		// ****** forward transform ******

		for(int wrk = 0; wrk < dwt_util_get_num_workers(); wrk++)
		{
			// *** init ***

			float *addr = ASSUME_ALIGNED(calc_temp_offset2_s(arr, wrk, 0), 16);
			float *base = addr+0;
			assert( is_aligned_s(addr) );

			// *** prolog2 ***

			// prolog2: import(3)
			op4s_sdl_import_s_ref(l, base, 3);

			// prolog2: pass-prolog
			op4s_sdl_pass_fwd_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// prolog2: import(2)
			op4s_sdl_import_s_ref(l, base, 2);

			// prolog2: pass-prolog
			op4s_sdl_pass_fwd_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// prolog2: import(1)
			op4s_sdl_import_s_ref(l, base, 1);

			// prolog2: pass-prolog
			op4s_sdl_pass_fwd_prolog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// prolog2: import(0)
			op4s_sdl_import_s_ref(l, base, 0);

			// *** core ***

			// core: for s = 0 to S do
			for(int s = 0; s < S; s++)
			{
				// core: pass-core
				op4s_sdl_pass_fwd_core_s_ref(w, v, l, c, r, z, in, out, &addr);
			}

			// *** epilog2 ***

			// epilog2: export(3)
			op4s_sdl_export_s_ref(l, &base[2*steps], 3);

			// epilog2: pass-epilog
			op4s_sdl_pass_fwd_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(2)
			op4s_sdl_export_s_ref(l, &base[2*steps], 2);

			// epilog2: pass-epilog
			op4s_sdl_pass_fwd_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(1)
			op4s_sdl_export_s_ref(l, &base[2*steps], 1);

			// epilog2: pass-epilog
			op4s_sdl_pass_fwd_epilog_s_ref(w, v, l, c, r, z, in, out, &addr);

			// epilog2: export(0)
			op4s_sdl_export_s_ref(l, &base[2*steps], 0);
		}
	}
	else
	{
		// ****** transform w/o scaling ******

		// not implemented yet
		dwt_util_abort();
	}
}

static
void accel_lift_op4s_fwd_main_sdl_stride_ref_part_prolog2_s(
	float *base,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *in,
	float *out,
	float **addr,
	int stride
)
{
	// prolog2: import(3)
	op4s_sdl_import_stride_s_ref(l, base, 3, stride);

	// prolog2: pass-prolog
	op4s_sdl_pass_fwd_prolog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// prolog2: import(2)
	op4s_sdl_import_stride_s_ref(l, base, 2, stride);

	// prolog2: pass-prolog
	op4s_sdl_pass_fwd_prolog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// prolog2: import(1)
	op4s_sdl_import_stride_s_ref(l, base, 1, stride);

	// prolog2: pass-prolog
	op4s_sdl_pass_fwd_prolog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// prolog2: import(0)
	op4s_sdl_import_stride_s_ref(l, base, 0, stride);
}

#ifdef __SSE__
#define accel_lift_op4s_fwd_main_sdl_stride_sse_part_prolog2_s(base,w,v,l,c,r,z,in,out,addr,stride) \
do { \
	op4s_sdl_import_stride_s_sse(l, base, 3, stride); \
	op4s_sdl_pass_fwd_prolog_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride); \
	op4s_sdl_import_stride_s_sse(l, base, 2, stride); \
	op4s_sdl_pass_fwd_prolog_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride); \
	op4s_sdl_import_stride_s_sse(l, base, 1, stride); \
	op4s_sdl_pass_fwd_prolog_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride); \
	op4s_sdl_import_stride_s_sse(l, base, 0, stride); \
} while(0)
#endif

static
void op4_fwd_sdl_prolog2_s(
	float *ptr0,
	float *ptr1,
	float *ptr2,
	float *ptr3,
	float *ptr4,
	float *ptr5,
	float *ptr6,
	float *ptr7,
	float *ptr8,
	float *ptr9,
	const float *w,
	const float *v, // unused
	float *l,
	float *c,
	float *r
)
{
	UNUSED(v);

	float buff[2];
	float t[4];
// part0
	// prolog2: import(3)
	l[3] = *ptr3; // base+3
// part1
	// prolog2: pass-prolog
	op4s_sdl_shuffle_s_ref(c, r);
	buff[0] = *ptr4; // base+0+4+0
	buff[1] = *ptr5; // base+0+4+1
	op4s_sdl_input_s_ref(buff, c, r);
	op4s_sdl_op_s_ref(t, c, w, l, r);
	op4s_sdl_update_s_ref(c, l, r, t);
// part2
	// prolog2: import(2)
	l[2] = *ptr2; // base+2
// part3
	// prolog2: pass-prolog
	op4s_sdl_shuffle_s_ref(c, r);
	buff[0] = *ptr6; // base+2+4+0
	buff[1] = *ptr7; // base+2+4+1
	op4s_sdl_input_s_ref(buff, c, r);
	op4s_sdl_op_s_ref(t, c, w, l, r);
	op4s_sdl_update_s_ref(c, l, r, t);
// part4
	// prolog2: import(1)
	l[1] = *ptr1; // base+1
// part5
	// prolog2: pass-prolog
	op4s_sdl_shuffle_s_ref(c, r);
	buff[0] = *ptr8; // base+4+4+0
	buff[1] = *ptr9; // base+4+4+1
	op4s_sdl_input_s_ref(buff, c, r);
	op4s_sdl_op_s_ref(t, c, w, l, r);
	op4s_sdl_update_s_ref(c, l, r, t);
// part6
	// prolog2: import(0)
	l[0] = *ptr0; // base+0
}

// part0
static
void op4_fwd_sdl_prolog2_part0_s(
	float *ptr3,
	const float *w,
	const float *v, // unused
	float *l,
	float *c,
	float *r
)
{
	UNUSED(w);
	UNUSED(v);
	UNUSED(c);
	UNUSED(r);

	// prolog2: import(3)
	l[3] = *ptr3; // base+3
}

static
void op4_fwd_sdl_prolog2_import_s(
	float *ptr,
	float *lcr,
	int idx
)
{
	// prolog2: import(i)
	(lcr+0)[idx] = *ptr; // base+i
}

// part2
static
void op4_fwd_sdl_prolog2_part2_s(
	float *ptr2,
	const float *w,
	const float *v, // unused
	float *l,
	float *c,
	float *r
)
{
	UNUSED(w);
	UNUSED(v);
	UNUSED(c);
	UNUSED(r);

	// prolog2: import(2)
	l[2] = *ptr2; // base+2
}

// part4
static
void op4_fwd_sdl_prolog2_part4_s(
	float *ptr1,
	const float *w,
	const float *v, // unused
	float *l,
	float *c,
	float *r
)
{
	UNUSED(w);
	UNUSED(v);
	UNUSED(c);
	UNUSED(r);

	// prolog2: import(1)
	l[1] = *ptr1; // base+1
}

// part6
static
void op4_fwd_sdl_prolog2_part6_s(
	float *ptr0,
	const float *w,
	const float *v, // unused
	float *l,
	float *c,
	float *r
)
{
	UNUSED(w);
	UNUSED(v);
	UNUSED(c);
	UNUSED(r);

	// prolog2: import(0)
	l[0] = *ptr0; // base+0
}


// part_odd
static
void op4_fwd_sdl_prolog2_part_s(
	float *ptr0,
	float *ptr1,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r
)
{
	UNUSED(v);
#ifdef __SSE__
	__m128 buff, z;
	buff[0] = *ptr0;
	buff[1] = *ptr1;
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)c, *(__m128 *)r);
	op4s_sdl2_op_s_sse(z, *(__m128 *)c, *(__m128 *)w, *(__m128 *)l, *(__m128 *)r);
	op4s_sdl2_update_s_sse(*(__m128 *)c, *(__m128 *)l, *(__m128 *)r, z);
#else
	// TODO: test this
	float buff[4], z[4];
	buff[0] = *ptr0;
	buff[1] = *ptr1;
	op4s_sdl2_shuffle_input_low_s_ref(buff, c, r);
	op4s_sdl2_op_s_ref(z, c, w, l, r);
	op4s_sdl2_update_s_ref(c, l, r, z);
#endif
}


static
void accel_lift_op4s_fwd_main_sdl_stride_ref_part_epilog2_s(
	float *base,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r,
	float *z,
	float *in,
	float *out,
	float **addr,
	int stride
)
{
	// epilog2: export(3)
	op4s_sdl_export_stride_s_ref(l, base, 3, stride);

	// epilog2: pass-epilog
	op4s_sdl_pass_fwd_epilog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// epilog2: export(2)
	op4s_sdl_export_stride_s_ref(l, base, 2, stride);

	// epilog2: pass-epilog
	op4s_sdl_pass_fwd_epilog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// epilog2: export(1)
	op4s_sdl_export_stride_s_ref(l, base, 1, stride);

	// epilog2: pass-epilog
	op4s_sdl_pass_fwd_epilog_stride_s_ref(w, v, l, c, r, z, in, out, addr, stride);

	// epilog2: export(0)
	op4s_sdl_export_stride_s_ref(l, base, 0, stride);
}

#ifdef __SSE__
#define accel_lift_op4s_fwd_main_sdl_stride_sse_part_epilog2_s(base,w,v,l,c,r,z,in,out,addr,stride) \
do { \
	op4s_sdl_export_stride_s_sse(l, base, 3, stride); \
	op4s_sdl_pass_fwd_epilog_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride); \
	op4s_sdl_export_stride_s_sse(l, base, 2, stride); \
	op4s_sdl_pass_fwd_epilog_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride); \
	op4s_sdl_export_stride_s_sse(l, base, 1, stride); \
	op4s_sdl_pass_fwd_epilog_stride_s_sse(w, v, l, c, r, z, in, out, addr, stride); \
	op4s_sdl_export_stride_s_sse(l, base, 0, stride); \
} while(0)
#endif

#if 1
static
void op4_fwd_sdl_epilog2_part_s(
	float *ptr0,
	float *ptr1,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r
)
{
#ifndef __SSE__
	float buff[2];
	float t[4];

	op4s_sdl_shuffle_s_ref(c, r);
	op4s_sdl_op_s_ref(t, c, w, l, r);
	op4s_sdl_output_s_ref(buff, l, t);
	op4s_sdl_scale_s_ref(buff, v);
	op4s_sdl_update_s_ref(c, l, r, t);
#else
	__m128 buff, z;

	op4s_sdl2_shuffle_s_sse(*(__m128 *)c, *(__m128 *)r);
	op4s_sdl2_op_s_sse(z, *(__m128 *)c, *(__m128 *)w, *(__m128 *)l, *(__m128 *)r);
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)l, z);
	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);
	op4s_sdl2_update_s_sse(*(__m128 *)c, *(__m128 *)l, *(__m128 *)r, z);
#endif
	*ptr0 = buff[0];
	*ptr1 = buff[1];
}
#endif

#ifdef __SSE__
static
void op4_fwd_sdl_epilog2_s(
	float *ptr_6,
	float *ptr_5,
	float *ptr_4,
	float *ptr_3,
	float *ptr_2,
	float *ptr_1,
	float *ptr0,
	float *ptr1,
	float *ptr2,
	float *ptr3,
	const float *w,
	const float *v,
	float *l,
	float *c,
	float *r
)
{
	// epilog2: export(3)
	*ptr3 = l[3]; // base+3

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		ptr_6,
		ptr_5,
		w, v,
		l, c, r
	);

	// epilog2: export(2)
	*ptr2 = l[2]; // base+2

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		ptr_4,
		ptr_3,
		w, v,
		l, c, r
	);

	// epilog2: export(1)
	*ptr1 = l[1]; // base+1

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		ptr_2,
		ptr_1,
		w, v,
		l, c, r
	);

	// epilog2: export(0)
	*ptr0 = l[0]; // base+0
}
#endif

#if 1
static
void op4_fwd_sdl_epilog2_fast_s(
	float *ptr0,
	float *ptr1,
	float *ptr2,
	float *ptr3,
	float *ptr4,
	float *ptr5,
	float *ptr6,
	float *ptr7,
	float *ptr8,
	float *ptr9,
	const float *w,
	const float *v,
	float *lcr
)
{
	// epilog2: export(3)
	*ptr9 = (lcr+0)[3]; // base+3

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		ptr0,
		ptr1,
		w, v,
		(lcr+0), (lcr+4), (lcr+8)
	);

	// epilog2: export(2)
	*ptr8 = (lcr+0)[2]; // base+2

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		ptr2,
		ptr3,
		w, v,
		(lcr+0), (lcr+4), (lcr+8)
	);

	// epilog2: export(1)
	*ptr7 = (lcr+0)[1]; // base+1

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		ptr4,
		ptr5,
		w, v,
		(lcr+0), (lcr+4), (lcr+8)
	);

	// epilog2: export(0)
	*ptr6 = (lcr+0)[0]; // base+0
}
#endif

static
void accel_lift_op4s_fwd_main_dl_stride_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
);

static
void accel_lift_op4s_fwd_main_sdl_stride_ref_part_exception_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	if( steps < 3 )
	{
		accel_lift_op4s_fwd_main_dl_stride_s(
			arr,
			steps,
			alpha,
			beta,
			gamma,
			delta,
			zeta,
			scaling,
			stride
		);
	}
}

static
void accel_lift_op4s_fwd_main_sdl_stride_ref_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( scaling > 0 );
	assert( 1 == dwt_util_get_num_workers() );

	accel_lift_op4s_fwd_main_sdl_stride_ref_part_exception_s(
		arr,
		steps,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		scaling,
		stride
	);

	if( steps < 3 )
		return;

	const float w[4] = { delta, gamma, beta, alpha };
	const float v[4] = { 1/zeta, zeta, 1/zeta, zeta };

	float l[4];
	float c[4];
	float r[4];
	float z[4];
	float in[4];
	float out[4];

	const int S = steps-3;

	// *** init ***
	float *addr = arr;

	// *** prolog2 ***
	accel_lift_op4s_fwd_main_sdl_stride_ref_part_prolog2_s(arr, w, v, l, c, r, z, in, out, &addr, stride);

	// *** core ***
	for(int s = 0; s < S; s++)
	{
		// core: pass-core
		op4s_sdl_pass_fwd_core_stride_s_ref(w, v, l, c, r, z, in, out, &addr, stride);
	}

	// *** epilog2 ***
	accel_lift_op4s_fwd_main_sdl_stride_ref_part_epilog2_s(addr1_s(arr,2*steps,stride), w, v, l, c, r, z, in, out, &addr, stride);
}

#ifdef __SSE__
static
void accel_lift_op4s_fwd_main_sdl_stride_sse_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( scaling > 0 );
	assert( 1 == dwt_util_get_num_workers() );

	accel_lift_op4s_fwd_main_sdl_stride_ref_part_exception_s(
		arr,
		steps,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		scaling,
		stride
	);

	if( steps < 3 )
		return;

	const __m128 w = { delta, gamma, beta, alpha };
	const __m128 v = { 1/zeta, zeta, 1/zeta, zeta };

	__m128 l;
	__m128 c;
	__m128 r;
	__m128 z;
	__m128 in;
	__m128 out;

	const int S = steps-3;

	// *** init ***
	float *addr = arr;

	// *** prolog2 ***
	accel_lift_op4s_fwd_main_sdl_stride_sse_part_prolog2_s(arr, w, v, l, c, r, z, in, out, &addr, stride);

	// *** core ***
	for(int s = 0; s < S; s++)
	{
		// core: pass-core
		op4s_sdl_pass_fwd_core_stride_s_sse(w, v, l, c, r, z, in, out, &addr, stride);
	}

	// *** epilog2 ***
	accel_lift_op4s_fwd_main_sdl_stride_sse_part_epilog2_s(addr1_s(arr,2*steps,stride), w, v, l, c, r, z, in, out, &addr, stride);
}
#endif

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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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
void accel_lift_op4s_fwd_main_dl_stride_pair_prolog0_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	float *l // [4]
)
{
	UNUSED(out0);
	UNUSED(out1);
	UNUSED(alpha);
	UNUSED(beta);
	UNUSED(gamma);
	UNUSED(delta);
	UNUSED(zeta);

	l[0] = *ptr0;
	l[1] = *ptr1;
}

static
void accel_lift_op4s_fwd_main_dl_stride_pair_prolog1_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	float *l // [4]
)
{
	UNUSED(out0);
	UNUSED(out1);
	UNUSED(alpha);
	UNUSED(beta);
	UNUSED(gamma);
	UNUSED(delta);
	UNUSED(zeta);

	l[2] = *ptr0;
	l[3] = *ptr1;
}

static
void accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	float *l // [4]
)
{
	// constants
	const float w[4] = { delta, gamma, beta, alpha };
	const float v[2] = { 1/zeta, zeta };

	// aux. variables
	float in[2];
	float out[2];
	float r[4];
	float c[4];

	// inputs
	in[0] = *ptr0;
	in[1] = *ptr1;

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
	*out0 = out[0];
	*out1 = out[1];

	// update l[]
	l[0] = r[0];
	l[1] = r[1];
	l[2] = r[2];
	l[3] = r[3];
}

#ifdef __SSE__
static
void cdf97_fwd_core_dl_sc_sse_2x2_s(
	float *ptr_y0_x0, // in
	float *ptr_y0_x1, // in
	float *ptr_y1_x0, // in
	float *ptr_y1_x1, // in
	float *out_y0_x0, // out
	float *out_y0_x1, // out
	float *out_y1_x0, // out
	float *out_y1_x1, // out
	float *buff_h0, // [4]
	float *buff_h1, // [4]
	float *buff_v0, // [4]
	float *buff_v1  // [4]
)
{
	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };

	const __m128 v_vertL = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f,
		0.f, 0.f };
	const __m128 v_vertR = { 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s),
		0.f, 0.f };

	// temp
	__m128 t;

	// aux. variables
	__m128 x, y, r, c;

	// horiz 1
	{
		float *l = buff_h0;

		// inputs
		x[0] = *ptr_y0_x0;
		x[1] = *ptr_y0_x1;

		// shuffles
		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		// outputs
		t[0] = y[0];
		t[1] = y[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	// horiz 2
	{
		float *l = buff_h1;

		// inputs
		x[0] = *ptr_y1_x0;
		x[1] = *ptr_y1_x1;

		// shuffles
		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		// outputs
		t[2] = y[0];
		t[3] = y[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	// vert 1
	{
		float *l = buff_v0;

		// inputs
		x[0] = t[0];
		x[1] = t[2];

		// shuffles
		y[0] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[1] = c[0]+w[0]*(l[0]+r[0]);

		// scaling
		y[0] *= v_vertL[0];
		y[1] *= v_vertL[1];

		// outputs
		*out_y0_x0 = y[0];
		*out_y1_x0 = y[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}

	// vert 2
	{
		float *l = buff_v1;

		// inputs
		x[0] = t[1];
		x[1] = t[3];

		// shuffles
		y[2] = l[0];
		c[0] = l[1];
		c[1] = l[2];
		c[2] = l[3];
		c[3] = x[0];

		// operation
		r[3] = x[1];
		r[2] = c[3]+w[3]*(l[3]+r[3]);
		r[1] = c[2]+w[2]*(l[2]+r[2]);
		r[0] = c[1]+w[1]*(l[1]+r[1]);
		y[3] = c[0]+w[0]*(l[0]+r[0]);

		// scaling
		y[2] *= v_vertR[0];
		y[3] *= v_vertR[1];

		// outputs
		*out_y0_x1 = y[2];
		*out_y1_x1 = y[3];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];
	}
}
#endif

static
void accel_lift_op4s_fwd_main_dl_stride_pair_core_2x2_s(
	float *ptr_y0_x0, // in
	float *ptr_y0_x1, // in
	float *ptr_y1_x0, // in
	float *ptr_y1_x1, // in
	float *out_y0_x0, // out
	float *out_y0_x1, // out
	float *out_y1_x0, // out
	float *out_y1_x1, // out
	float alpha, // w
	float beta, // w
	float gamma, // w
	float delta, // w
	float zeta, // v
	float *buff_h0, // [4]
	float *buff_h1, // [4]
	float *buff_v0, // [4]
	float *buff_v1  // [4]
)
{
	float tmp[4];
	float *tmp_y0_x0 = tmp+0;
	float *tmp_y0_x1 = tmp+1;
	float *tmp_y1_x0 = tmp+2;
	float *tmp_y1_x1 = tmp+3;

	// horizontal 0
	// [y+0, x+0], [y+0, x+1] => [y+0, x+0-4], [y+0, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		ptr_y0_x0,
		ptr_y0_x1,
		tmp_y0_x0,
		tmp_y0_x1,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		buff_h0
	);

	// horizontal 1
	// [y+1, x+0], [y+1, x+1] => [y+1, x+0-4], [y+1, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		ptr_y1_x0,
		ptr_y1_x1,
		tmp_y1_x0,
		tmp_y1_x1,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		buff_h1
	);

	// vertical 0
	// [y+0, x+0-4] [y+1, x+0-4] => [y+0-4, x+0-4] [y+1-4, x+0-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x0,
		tmp_y1_x0,
		out_y0_x0,
		out_y1_x0,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		buff_v0
	);

	// vertical 1
	// [y+0, x+1-4] [y+1, x+1-4] => [y+0-4, x+1-4] [y+1-4, x+1-4]
	accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
		tmp_y0_x1,
		tmp_y1_x1,
		out_y0_x1,
		out_y1_x1,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		buff_v1
	);
}

static
void accel_lift_op4s_fwd_main_dl_stride_pair_epilog0_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	float *l // [4]
)
{
	UNUSED(ptr0);
	UNUSED(ptr1);
	UNUSED(alpha);
	UNUSED(beta);
	UNUSED(gamma);
	UNUSED(delta);
	UNUSED(zeta);

	*out0 = l[0];
	*out1 = l[1];
}

static
void accel_lift_op4s_fwd_main_dl_stride_pair_epilog1_s(
	float *ptr0,
	float *ptr1,
	float *out0,
	float *out1,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	float *l // [4]
)
{
	UNUSED(ptr0);
	UNUSED(ptr1);
	UNUSED(alpha);
	UNUSED(beta);
	UNUSED(gamma);
	UNUSED(delta);
	UNUSED(zeta);

	*out0 = l[2];
	*out1 = l[3];
}

/**
 * vertical vectorisation (double-loop approach), forward transform
 */
static
void accel_lift_op4s_fwd_main_dl_stride_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( steps >= 0 );
	assert( scaling > 0 );

	float l[4];

	accel_lift_op4s_fwd_main_dl_stride_pair_prolog0_s(
		addr1_s(arr, 0, stride),
		addr1_s(arr, 1, stride),
		NULL,
		NULL,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		l
	);
	accel_lift_op4s_fwd_main_dl_stride_pair_prolog1_s(
		addr1_s(arr, 2, stride),
		addr1_s(arr, 3, stride),
		NULL,
		NULL,
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		l
	);

	// init
	float *addr = addr1_s(arr, 4, stride);

	// loop by pairs from left to right
	for(int s = 0; s < steps; s++)
	{
		accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
			addr1_s(addr, 0, stride),
			addr1_s(addr, 1, stride),
			addr1_s(addr, 0-4, stride),
			addr1_s(addr, 1-4, stride),
			alpha,
			beta,
			gamma,
			delta,
			zeta,
			l
		);

		// pointers
		addr = addr1_s(addr, 2, stride);
	}

	accel_lift_op4s_fwd_main_dl_stride_pair_epilog0_s(
		NULL,
		NULL,
		addr1_s(addr, 0-4, stride),
		addr1_s(addr, 1-4, stride),
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		l
	);
	accel_lift_op4s_fwd_main_dl_stride_pair_epilog1_s(
		NULL,
		NULL,
		addr1_s(addr, 2-4, stride),
		addr1_s(addr, 3-4, stride),
		alpha,
		beta,
		gamma,
		delta,
		zeta,
		l
	);
}

/**
 * vertical vectorisation (double-loop approach), inverse transform
 */
static
void accel_lift_op4s_inv_main_dl_stride_s(
	float *arr,
	int steps,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( steps >= 0 );
	assert( scaling < 0 );

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
	l[0] = *addr1_s(arr, 0, stride);
	l[1] = *addr1_s(arr, 1, stride);
	l[2] = *addr1_s(arr, 2, stride);
	l[3] = *addr1_s(arr, 3, stride);

	// init
	float *addr = addr1_s(arr, 4, stride);

	// loop by pairs from left to right
	for(int s = 0; s < steps; s++)
	{
		// inputs
		in[0] = *addr1_s(addr, 0, stride);
		in[1] = *addr1_s(addr, 1, stride);

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
		*addr1_s(addr, 0-4, stride) = out[0];
		*addr1_s(addr, 1-4, stride) = out[1];

		// update l[]
		l[0] = r[0];
		l[1] = r[1];
		l[2] = r[2];
		l[3] = r[3];

		// pointers
		addr = addr1_s(addr, 2, stride);
	}

	// slide out right border
	*addr1_s(addr, 0-4, stride) = l[0];
	*addr1_s(addr, 1-4, stride) = l[1];
	*addr1_s(addr, 2-4, stride) = l[2];
	*addr1_s(addr, 3-4, stride) = l[3];
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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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
			float *arr_local = calc_temp_offset2_s(arr, w, 0);

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
			float *arr_local = ASSUME_ALIGNED(calc_temp_offset2_s(arr, w, 0), 16);

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
			float *arr_local = ASSUME_ALIGNED(calc_temp_offset2_s(arr, w, 0), 16);

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
			addr[worker] = calc_temp_offset2_s(arr, worker, 0);

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
			addr[worker] = calc_temp_offset2_s(arr, worker, 0);

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
			arr_local[worker] = calc_temp_offset2_s(arr, worker, 0);
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
			arr_local[worker] = calc_temp_offset2_s(arr, worker, 0);
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

#ifdef __asvp__
	UNUSED(scaling);
	UNUSED(alpha);
	UNUSED(beta);
	UNUSED(gamma);
	UNUSED(delta);
	UNUSED(zeta);

	assert( steps <= (BANK_SIZE - 4) / 2 );

	const int size = 2*steps + 4;

	assert( is_even(size) );

	for(int w = 0; w < get_active_workers(); w++)
	{
		// FIXME(ASVP): channel w according to worker ID; but each worker has independent DMA channels, thus this is not necessary
		const uint8_t ch = w;
		float *addr_local = calc_temp_offset2_s(addr, w, 0);

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

	assert( is_even(size) );

	for(int w = 0; w < get_active_workers(); w++)
	{
		const uint8_t ch = w;
		float *addr_local = calc_temp_offset2_s(addr, w, 0);

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
		float *addr_local = calc_temp_offset2_s(addr, w, 0);

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
	int scaling
)
{
	assert( N-off >= 4 );

#ifdef NDEBUG
	UNUSED(N);
#endif

	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *arr_local = calc_temp_offset2_s(arr, w, off);

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
void accel_lift_op4s_prolog_stride_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( N-off >= 4 );

#ifdef NDEBUG
	UNUSED(N);
#endif

	assert( 1 == dwt_util_get_num_workers() );
	{
		if( off )
		{
			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
			*addr1_s(arr, 3, stride) += alpha*(*addr1_s(arr, 2, stride) + *addr1_s(arr, 4, stride));

			// beta
			*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));
			*addr1_s(arr, 2, stride) += beta*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

			// gamma
			*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

			// delta
			*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));

			// scaling
			if( scaling > 0)
			{
				*addr1_s(arr, 0, stride) *= zeta;
			}
		}
		else
		{
			// inv-scaling
			if( scaling < 0 )
			{
				*addr1_s(arr, 0, stride) *= 1/zeta;
				*addr1_s(arr, 1, stride) *= zeta;
				*addr1_s(arr, 2, stride) *= 1/zeta;
				*addr1_s(arr, 3, stride) *= zeta;
			}

			// alpha
			*addr1_s(arr, 0, stride) += 2*alpha*(*addr1_s(arr, 1, stride));
			*addr1_s(arr, 2, stride) += alpha*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

			// beta
			*addr1_s(arr, 1, stride) += beta*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

			// gamma
			*addr1_s(arr, 0, stride) += 2*gamma*(*addr1_s(arr, 1, stride));

			// delta
			// none

			// scaling
			// none
		}
	}
}

// hole
static
void accel_lift_op4s_prolog_stride_hole_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( N-off >= 4 );

#ifdef NDEBUG
	UNUSED(N);
#endif

	assert( 1 == dwt_util_get_num_workers() );
	{
		if( off )
		{
			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
			*addr1_s(arr, 3, stride) += alpha*(*addr1_s(arr, 2, stride) + *addr1_s(arr, 4, stride));

			// beta
			*addr1_s(arr, 0, stride) += beta *(*addr1_s(arr, 1, stride) + 0.f);
			*addr1_s(arr, 2, stride) += beta *(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

			// gamma
			*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

			// delta
			*addr1_s(arr, 0, stride) += delta*(*addr1_s(arr, 1, stride) + 0.f);

			// scaling
			if( scaling > 0)
			{
				*addr1_s(arr, 0, stride) *= zeta;
			}
		}
		else
		{
			// inv-scaling
			if( scaling < 0 )
			{
				*addr1_s(arr, 0, stride) *= 1/zeta;
				*addr1_s(arr, 1, stride) *= zeta;
				*addr1_s(arr, 2, stride) *= 1/zeta;
				*addr1_s(arr, 3, stride) *= zeta;
			}

			// alpha
			*addr1_s(arr, 0, stride) += alpha*(*addr1_s(arr, 1, stride) + 0.f);
			*addr1_s(arr, 2, stride) += alpha*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

			// beta
			*addr1_s(arr, 1, stride) += beta *(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

			// gamma
			*addr1_s(arr, 0, stride) += gamma*(*addr1_s(arr, 1, stride) + 0.f);

			// delta
			// none

			// scaling
			// none
		}
	}
}

// zero
static
void accel_lift_op4s_prolog_stride_zero_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( N-off >= 4 );

#ifdef NDEBUG
	UNUSED(N);
#endif

	assert( 1 == dwt_util_get_num_workers() );
	{
		if( off )
		{
			// initially zeros
			float neg1 = 0.f;
			float neg2 = 0.f;

			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			neg1                     += alpha*(0.f/*[-2]*/              + *addr1_s(arr, 0, stride));
			*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
			*addr1_s(arr, 3, stride) += alpha*(*addr1_s(arr, 2, stride) + *addr1_s(arr, 4, stride));

			// beta
			neg2                     += beta *(0.f/*[-3]*/              + neg1/*[-1]*/);
			*addr1_s(arr, 0, stride) += beta *(neg1/*[-1]*/             + *addr1_s(arr, 1, stride));
			*addr1_s(arr, 2, stride) += beta *(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

			// gamma
			neg1                     += gamma*(neg2/*[-2]*/             + *addr1_s(arr, 0, stride));
			*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

			// delta
			*addr1_s(arr, 0, stride) += delta*(neg1/*[-1]*/             + *addr1_s(arr, 1, stride));

			// scaling
			if( scaling > 0)
			{
				*addr1_s(arr, 0, stride) *= zeta;
			}
		}
		else
		{
			// initially zeros
			float neg1 = 0.f;

			// inv-scaling
			if( scaling < 0 )
			{
				*addr1_s(arr, 0, stride) *= 1/zeta;
				*addr1_s(arr, 1, stride) *= zeta;
				*addr1_s(arr, 2, stride) *= 1/zeta;
				*addr1_s(arr, 3, stride) *= zeta;
			}

			// alpha
			*addr1_s(arr, 0, stride) += alpha*(0.f/*[-1]*/              + *addr1_s(arr, 1, stride));
			*addr1_s(arr, 2, stride) += alpha*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

			// beta
			neg1                     += beta *(0.f/*[-2]*/              + *addr1_s(arr, 0, stride));
			*addr1_s(arr, 1, stride) += beta *(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

			// gamma
			*addr1_s(arr, 0, stride) += gamma*(*addr1_s(arr, 1, stride) + neg1/*[-1]*/);

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
		float *arr_local = calc_temp_offset2_s(arr, w, off);

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

static
void accel_lift_op4s_epilog_stride_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( N-off >= 4 );

	assert( 1 == dwt_util_get_num_workers() );
	{
		if( is_even(N - off) )
		{
			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			// none

			// beta
			*addr1_s(arr, N-1, stride) += 2*beta*(*addr1_s(arr, N-2, stride));

			// gamma
			*addr1_s(arr, N-2, stride) += gamma*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

			// delta
			*addr1_s(arr, N-1, stride) += 2*delta*(*addr1_s(arr, N-2, stride));
			*addr1_s(arr, N-3, stride) += delta*(*addr1_s(arr, N-4, stride) + *addr1_s(arr, N-2, stride));

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				*addr1_s(arr, N-4, stride) *= 1/zeta;
				*addr1_s(arr, N-3, stride) *= zeta;
				*addr1_s(arr, N-2, stride) *= 1/zeta;
				*addr1_s(arr, N-1, stride) *= zeta;
			}
		}
		else /* is_odd(N-off) */
		{
			// inv-scaling
			if( scaling < 0 )
			{
				*addr1_s(arr, N-1, stride) *= 1/zeta;
			}

			// alpha
			*addr1_s(arr, N-1, stride) += 2*alpha*(*addr1_s(arr, N-2, stride));

			// beta
			*addr1_s(arr, N-2, stride) += beta*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

			// gamma
			*addr1_s(arr, N-1, stride) += 2*gamma*(*addr1_s(arr, N-2, stride));
			*addr1_s(arr, N-3, stride) += gamma*(*addr1_s(arr, N-2, stride) + *addr1_s(arr, N-4, stride));

			// delta
			*addr1_s(arr, N-2, stride) += delta*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));
			*addr1_s(arr, N-4, stride) += delta*(*addr1_s(arr, N-5, stride) + *addr1_s(arr, N-3, stride));

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				*addr1_s(arr, N-5, stride) *= 1/zeta;
				*addr1_s(arr, N-4, stride) *= zeta;
				*addr1_s(arr, N-3, stride) *= 1/zeta;
				*addr1_s(arr, N-2, stride) *= zeta;
				*addr1_s(arr, N-1, stride) *= 1/zeta;
			}
		}
	}
}

// hole
static
void accel_lift_op4s_epilog_stride_hole_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( N-off >= 4 );

	assert( 1 == dwt_util_get_num_workers() );
	{
		if( is_even(N - off) )
		{
			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			// none

			// beta
			*addr1_s(arr, N-1, stride) += beta *(*addr1_s(arr, N-2, stride) + 0.f);

			// gamma
			*addr1_s(arr, N-2, stride) += gamma*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

			// delta
			*addr1_s(arr, N-1, stride) += delta*(*addr1_s(arr, N-2, stride) + 0.f);
			*addr1_s(arr, N-3, stride) += delta*(*addr1_s(arr, N-4, stride) + *addr1_s(arr, N-2, stride));

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				*addr1_s(arr, N-4, stride) *= 1/zeta;
				*addr1_s(arr, N-3, stride) *= zeta;
				*addr1_s(arr, N-2, stride) *= 1/zeta;
				*addr1_s(arr, N-1, stride) *= zeta;
			}
		}
		else /* is_odd(N-off) */
		{
			// inv-scaling
			if( scaling < 0 )
			{
				*addr1_s(arr, N-1, stride) *= 1/zeta;
			}

			// alpha
			*addr1_s(arr, N-1, stride) += alpha*(*addr1_s(arr, N-2, stride) + 0.f);

			// beta
			*addr1_s(arr, N-2, stride) += beta *(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

			// gamma
			*addr1_s(arr, N-1, stride) += gamma*(*addr1_s(arr, N-2, stride) + 0.f);
			*addr1_s(arr, N-3, stride) += gamma*(*addr1_s(arr, N-2, stride) + *addr1_s(arr, N-4, stride));

			// delta
			*addr1_s(arr, N-2, stride) += delta*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));
			*addr1_s(arr, N-4, stride) += delta*(*addr1_s(arr, N-5, stride) + *addr1_s(arr, N-3, stride));

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				*addr1_s(arr, N-5, stride) *= 1/zeta;
				*addr1_s(arr, N-4, stride) *= zeta;
				*addr1_s(arr, N-3, stride) *= 1/zeta;
				*addr1_s(arr, N-2, stride) *= zeta;
				*addr1_s(arr, N-1, stride) *= 1/zeta;
			}
		}
	}
}

// zero
static
void accel_lift_op4s_epilog_stride_zero_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( N-off >= 4 );

	assert( 1 == dwt_util_get_num_workers() );
	{
		if( is_even(N - off) )
		{
			// inititally zeros
			float n0 = 0.f;
			float n1 = 0.f;

			// inv-scaling
			if( scaling < 0 )
			{
				// TODO
			}

			// alpha
			n0                         += alpha*(*addr1_s(arr, N-1, stride) + 0.f/*[N+1]*/);;

			// beta
			n1                         += beta *(n0/*[N-0]*/                + 0.f/*[N+2]*/);
			*addr1_s(arr, N-1, stride) += beta *(*addr1_s(arr, N-2, stride) + n0/*[N-0]*/);

			// gamma
			n0                         += gamma*(*addr1_s(arr, N-1, stride) + n1/*[N+1]*/);
			*addr1_s(arr, N-2, stride) += gamma*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

			// delta
			*addr1_s(arr, N-1, stride) += delta*(*addr1_s(arr, N-2, stride) + n0/*[N-0]*/);
			*addr1_s(arr, N-3, stride) += delta*(*addr1_s(arr, N-4, stride) + *addr1_s(arr, N-2, stride));

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				*addr1_s(arr, N-4, stride) *= 1/zeta;
				*addr1_s(arr, N-3, stride) *= zeta;
				*addr1_s(arr, N-2, stride) *= 1/zeta;
				*addr1_s(arr, N-1, stride) *= zeta;
			}
		}
		else /* is_odd(N-off) */
		{
			// inititally zero
			float n0 = 0.f;

			// inv-scaling
			if( scaling < 0 )
			{
				*addr1_s(arr, N-1, stride) *= 1/zeta;
			}

			// alpha
			*addr1_s(arr, N-1, stride) += alpha*(*addr1_s(arr, N-2, stride) + 0.f/*[N+0]*/);

			// beta
			n0                         += beta *(*addr1_s(arr, N-1, stride) + 0.f/*[N+1]*/);
			*addr1_s(arr, N-2, stride) += beta *(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));

			// gamma
			*addr1_s(arr, N-1, stride) += gamma*(*addr1_s(arr, N-2, stride) + n0/*[N+0]*/);
			*addr1_s(arr, N-3, stride) += gamma*(*addr1_s(arr, N-2, stride) + *addr1_s(arr, N-4, stride));

			// delta
			*addr1_s(arr, N-2, stride) += delta*(*addr1_s(arr, N-1, stride) + *addr1_s(arr, N-3, stride));
			*addr1_s(arr, N-4, stride) += delta*(*addr1_s(arr, N-5, stride) + *addr1_s(arr, N-3, stride));

			// scaling
			if( scaling > 0 )
			{
				// FIXME: this is dependend on "off"
				*addr1_s(arr, N-5, stride) *= 1/zeta;
				*addr1_s(arr, N-4, stride) *= zeta;
				*addr1_s(arr, N-3, stride) *= 1/zeta;
				*addr1_s(arr, N-2, stride) *= zeta;
				*addr1_s(arr, N-1, stride) *= 1/zeta;
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
	int scaling
)
{
	assert( N-off < 4 );

	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *arr_local = calc_temp_offset2_s(arr, w, off);

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
void accel_lift_op4s_short_stride_s(
	float *arr,
	int off,
	int N,
	float alpha,
	float beta,
	float gamma,
	float delta,
	float zeta,
	int scaling,
	int stride
)
{
	assert( N-off < 4 );

	assert( 1 == dwt_util_get_num_workers() );

	{
		if( off )
		{
			if( N == 2 )
			{
				// inv-scaling
				if( scaling < 0 )
				{
					// TODO
				}

				// alpha
				*addr1_s(arr, 1, stride) += 2*alpha*(*addr1_s(arr, 0, stride));

				// beta
				*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));

				// gamma
				*addr1_s(arr, 1, stride) += 2*gamma*(*addr1_s(arr, 0, stride));

				// delta
				*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));

				// scaling
				if( scaling > 0 )
				{
					*addr1_s(arr, 0, stride) *= zeta;
					*addr1_s(arr, 1, stride) *= 1/zeta;
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
				*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

				// beta
				*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));
				*addr1_s(arr, 2, stride) += 2*beta*(*addr1_s(arr, 1, stride));

				// gamma
				*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

				// delta
				*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));
				*addr1_s(arr, 2, stride) += 2*delta*(*addr1_s(arr, 1, stride));

				// scaling
				if( scaling > 0 )
				{
					*addr1_s(arr, 0, stride) *= zeta;
					*addr1_s(arr, 1, stride) *= 1/zeta;
					*addr1_s(arr, 2, stride) *= zeta;
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
				*addr1_s(arr, 1, stride) += alpha*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
				*addr1_s(arr, 3, stride) += 2*alpha*(*addr1_s(arr, 2, stride));

				// beta
				*addr1_s(arr, 0, stride) += 2*beta*(*addr1_s(arr, 1, stride));
				*addr1_s(arr, 2, stride) += beta*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

				// gamma
				*addr1_s(arr, 1, stride) += gamma*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));
				*addr1_s(arr, 3, stride) += 2*gamma*(*addr1_s(arr, 2, stride));

				// delta
				*addr1_s(arr, 0, stride) += 2*delta*(*addr1_s(arr, 1, stride));
				*addr1_s(arr, 2, stride) += delta*(*addr1_s(arr, 1, stride) + *addr1_s(arr, 3, stride));

				// scaling
				if( scaling > 0 )
				{
					*addr1_s(arr, 0, stride) *= zeta;
					*addr1_s(arr, 1, stride) *= 1/zeta;
					*addr1_s(arr, 2, stride) *= zeta;
					*addr1_s(arr, 3, stride) *= 1/zeta;
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
					*addr1_s(arr, 0, stride) *= 1/zeta;
					*addr1_s(arr, 1, stride) *= zeta;
				}

				// alpha
				*addr1_s(arr, 0, stride) += 2*alpha*(*addr1_s(arr, 1, stride));

				// beta
				*addr1_s(arr, 1, stride) += 2*beta*(*addr1_s(arr, 0, stride));

				// gamma
				*addr1_s(arr, 0, stride) += 2*gamma*(*addr1_s(arr, 1, stride));

				// delta
				*addr1_s(arr, 1, stride) += 2*delta*(*addr1_s(arr, 0, stride));

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
					*addr1_s(arr, 0, stride) *= 1/zeta;
					*addr1_s(arr, 1, stride) *= zeta;
					*addr1_s(arr, 2, stride) *= 1/zeta;
				}

				// alpha
				*addr1_s(arr, 0, stride) += 2*alpha*(*addr1_s(arr, 1, stride));
				*addr1_s(arr, 2, stride) += 2*alpha*(*addr1_s(arr, 1, stride));

				// beta
				*addr1_s(arr, 1, stride) += beta*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

				// gamma
				*addr1_s(arr, 0, stride) += 2*gamma*(*addr1_s(arr, 1, stride));
				*addr1_s(arr, 2, stride) += 2*gamma*(*addr1_s(arr, 1, stride));

				// delta
				*addr1_s(arr, 1, stride) += delta*(*addr1_s(arr, 0, stride) + *addr1_s(arr, 2, stride));

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
	int scaling
)
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

	const int offset = 1;

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
		float *tmp_local = calc_temp_offset2_s(tmp, w, offset);
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

	accel_lift_op4s_s(tmp, offset, N, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1);

	// copy tmp into dst
	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *tmp_local = calc_temp_offset2_s(tmp, w, offset);
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

static
void dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 1;

	if( N < 2 )
	{
		// respect stride
		if( 1 == N )
			ptr[0] *= dwt_cdf97_s1_s;
		return;
	}
	else
	{
		accel_lift_op4s_short_stride_s(ptr, offset, N, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
	}
}

static
void dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 1;

	accel_lift_op4s_prolog_stride_s(ptr, offset, N, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
}

static
void dwt_cdf97_f_ex_stride_inplace_part_core_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 1;

#if 0
	accel_lift_op4s_fwd_main_stride_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-4)/2, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
#endif
#if 1
	accel_lift_op4s_fwd_main_dl_stride_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-4)/2, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
#endif
#if 0
	accel_lift_op4s_fwd_main_sdl_stride_ref_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-4)/2, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
#endif
}

static
void dwt_cdf97_f_ex_stride_inplace_part_core_sdl_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 1;

	accel_lift_op4s_fwd_main_sdl_stride_ref_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-4)/2, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
}

#ifdef __SSE__
static
void dwt_cdf97_f_ex_stride_inplace_part_core_sdl_sse_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 1;

	accel_lift_op4s_fwd_main_sdl_stride_sse_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-4)/2, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
}
#endif

static
void dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 1;

	accel_lift_op4s_epilog_stride_s(ptr, offset, N, -dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride);
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

void dwt_cdf53_f_ex_stride_inplace_s(
	float *tmp,
	int N,
	int stride
)
{
	assert( N >= 0 && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			*addr1_s(tmp,0,stride) *= dwt_cdf53_s1_s;
		return;
	}

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_s(tmp,i,stride) -= dwt_cdf53_p1_s * (*addr1_s(tmp,i-1,stride) + *addr1_s(tmp,i+1,stride));

	if(is_odd(N))
		*addr1_s(tmp,N-1,stride) += 2 * dwt_cdf53_u1_s * *addr1_s(tmp,N-2,stride);
	else
		*addr1_s(tmp,N-1,stride) -= 2 * dwt_cdf53_p1_s * *addr1_s(tmp,N-2,stride);

	*addr1_s(tmp,0,stride) += 2 * dwt_cdf53_u1_s * *addr1_s(tmp,1,stride);

	for(int i=2; i<N-(N&1); i+=2)
		*addr1_s(tmp,i,stride) += dwt_cdf53_u1_s * (*addr1_s(tmp,i-1,stride) + *addr1_s(tmp,i+1,stride));

	// scale
	for(int i=0; i<N; i+=2)
		*addr1_s(tmp,i,stride) *= dwt_cdf53_s1_s;
	for(int i=1; i<N; i+=2)
		*addr1_s(tmp,i,stride) *= dwt_cdf53_s2_s;
}

static
float dwt_eaw_w(float n, float m, float alpha)
{
	const float eps = 1.0e-5f;

	return 1.f / (powf(fabsf(n-m), alpha) + eps);
}

static
void dwt_calc_eaw_w(float *w, float *arr, int N, float alpha)
{
	for(int i = 0; i < N-1; i++)
	{
		w[i] = dwt_eaw_w(arr[i], arr[i+1], alpha);
	}
	w[N-1] = 0.f; // not necessary
}

static
void dwt_calc_eaw_w_stride_s(
	float *w,
	float *arr, int N, int stride,
	float alpha
)
{
	for(int i = 0; i < N-1; i++)
	{
		w[i] = dwt_eaw_w(
			*addr1_s(arr,i,stride),
			*addr1_s(arr,i+1,stride),
			alpha);
	}
	w[N-1] = 0.f; // not necessary
}

// http://www.cs.huji.ac.il/~raananf/projects/eaw/
// TODO: move calculation of weights outside of this function
void dwt_eaw53_f_ex_stride_s(
	const float *src,
	float *dst_l,
	float *dst_h,
	float *tmp,
	int N,
	int stride,
	float *w,	// float w[N]
	float alpha
)
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

	// FIXME: move outside
	dwt_calc_eaw_w(w, tmp, N, alpha);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] -= (wL * tmp[i-1] + wR * tmp[i+1]) / (wL+wR);
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] += (wL * tmp[N-2] + wR * tmp[N-2]) / ( 2.f * (wL+wR) );
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] -= (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR);
	}

	{
		float wL = w[0];
		float wR = w[0];

		tmp[0] += (wL * tmp[1] + wR * tmp[1]) / ( 2.f * (wL+wR) );
	}

	for(int i=2; i<N-(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] += (wL * tmp[i-1] + wR * tmp[i+1]) / ( 2.f * (wL+wR) );
	}

	// scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s1_s;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s2_s;

	// copy tmp into dst
	dwt_util_memcpy_stride_s(dst_l, stride, tmp+0, 2*sizeof(float),  ceil_div2(N));
	dwt_util_memcpy_stride_s(dst_h, stride, tmp+1, 2*sizeof(float), floor_div2(N));
}

void dwt_eaw53_f_ex_stride_inplace_s(
	float *tmp,
	int N,
	int stride,
	float *w,	// float w[N]
	float alpha
)
{
	assert( N >= 0 && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			*addr1_s(tmp, 0, stride) *= dwt_cdf53_s1_s;
		return;
	}

	// calc weights
	dwt_calc_eaw_w_stride_s(w, tmp, N, stride, alpha);

	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		*addr1_s(tmp, i, stride) -= (wL * *addr1_s(tmp, i-1, stride) + wR * *addr1_s(tmp, i+1, stride)) / (wL+wR);
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];

		*addr1_s(tmp, N-1, stride) += (wL * *addr1_s(tmp, N-2, stride) + wR * *addr1_s(tmp, N-2, stride)) / ( 2.f * (wL+wR) );
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];

		*addr1_s(tmp, N-1, stride) -= (wL * *addr1_s(tmp, N-2, stride) + wR * *addr1_s(tmp, N-2, stride)) / (wL+wR);
	}

	{
		float wL = w[0];
		float wR = w[0];

		*addr1_s(tmp, 0, stride) += (wL * *addr1_s(tmp, 1, stride) + wR * *addr1_s(tmp, 1, stride)) / ( 2.f * (wL+wR) );
	}

	for(int i=2; i<N-(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		*addr1_s(tmp, i, stride) += (wL * *addr1_s(tmp, i-1, stride) + wR * *addr1_s(tmp, i+1, stride)) / ( 2.f * (wL+wR) );
	}

	// scale
	for(int i=0; i<N; i+=2)
		*addr1_s(tmp, i, stride) *= dwt_cdf53_s1_s;
	for(int i=1; i<N; i+=2)
		*addr1_s(tmp, i, stride) *= dwt_cdf53_s2_s;
}

// TODO: interpolating version of CDF 5/3
// FIXME: fix scaling
void dwt_interp53_f_ex_stride_s(
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
		;
	else
		tmp[N-1] -= 2 * dwt_cdf53_p1_s * tmp[N-2];

	// scale
	for(int i=0; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s1_s;
	for(int i=1; i<N; i+=2)
		tmp[i] = tmp[i] * dwt_cdf53_s2_s;

	// copy tmp into dst
	dwt_util_memcpy_stride_s(dst_l, stride, tmp+0, 2*sizeof(float),  ceil_div2(N));
	dwt_util_memcpy_stride_s(dst_h, stride, tmp+1, 2*sizeof(float), floor_div2(N));
}

// TODO: implement for N < 4
void dwt_interp2_f_ex_stride_s(
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

	if(N < 4)
	{
		dwt_util_log(LOG_WARN, "not implemented\n"); // FIXME
		return;
	}

	// copy src into tmp
	dwt_util_memcpy_stride_s(tmp, sizeof(float), src, stride, N);

	// predict 1 + update 1
	for(int i=1+2; i<N-2+(N&1)-2; i+=2)
		tmp[i] -= 0.1f*tmp[i-3] + 0.4f*tmp[i-1] + 0.4f*tmp[i+1] + 0.1f*tmp[i+3];

	if( is_even(N) )
		tmp[N-1] -= 2*0.1f*tmp[N-5] + 2*0.4f*tmp[N-2];

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

	const int offset = 0;

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
		float *tmp_local = calc_temp_offset2_s(tmp, w, 0);
		const float *src_l_local = calc_data_offset_const_s(src_l, w);
		const float *src_h_local = calc_data_offset_const_s(src_h, w);

		dwt_util_memcpy_stride_s(tmp_local+0, 2*sizeof(float), src_l_local, stride,  ceil_div2(N));
		dwt_util_memcpy_stride_s(tmp_local+1, 2*sizeof(float), src_h_local, stride, floor_div2(N));
	}

	accel_lift_op4s_s(tmp, offset, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1);

	// copy tmp into dst
	for(int w = 0; w < dwt_util_get_num_workers(); w++)
	{
		float *tmp_local = calc_temp_offset2_s(tmp, w, 0);
		float *dst_local = calc_data_offset_s(dst, w);

		dwt_util_memcpy_stride_s(dst_local, stride, tmp_local, sizeof(float), N);
	}
}

static
void dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	if( N < 2 )
	{
		// respect stride
		if( 1 == N )
			ptr[0] *= dwt_cdf97_s2_s;
		return;
	}
	else
	{
		accel_lift_op4s_short_stride_s(ptr, offset, N,-dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
	}
}

static
void dwt_cdf97_i_ex_stride_inplace_part_prolog_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	accel_lift_op4s_prolog_stride_s(ptr, offset, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
}

// hole
static
void dwt_cdf97_i_ex_stride_inplace_part_prolog_hole_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	accel_lift_op4s_prolog_stride_hole_s(ptr, offset, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
}

// zero
static
void dwt_cdf97_i_ex_stride_inplace_part_prolog_zero_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	accel_lift_op4s_prolog_stride_zero_s(ptr, offset, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
}

static
void dwt_cdf97_i_ex_stride_inplace_part_core_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

#if 0
	accel_lift_op4s_inv_main_stride_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-4)/2, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
#else
	accel_lift_op4s_inv_main_dl_stride_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-4)/2, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
#endif
}

static
void dwt_cdf53_i_ex_stride_inplace_part_core_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	accel_lift_op2s_inv_main_stride_s(addr1_s(ptr, offset, stride), (to_even(N-offset)-2)/2, -dwt_cdf53_u1_s, dwt_cdf53_p1_s, dwt_cdf53_s1_s, -1, stride);
}

static
void dwt_cdf97_i_ex_stride_inplace_part_epilog_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	accel_lift_op4s_epilog_stride_s(ptr, offset, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
}

// hole
static
void dwt_cdf97_i_ex_stride_inplace_part_epilog_hole_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	accel_lift_op4s_epilog_stride_hole_s(ptr, offset, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
}

// zero
static
void dwt_cdf97_i_ex_stride_inplace_part_epilog_zero_s(
	float *ptr,
	int N,
	int stride
)
{
	const int offset = 0;

	accel_lift_op4s_epilog_stride_zero_s(ptr, offset, N, -dwt_cdf97_u2_s, dwt_cdf97_p2_s, -dwt_cdf97_u1_s, dwt_cdf97_p1_s, dwt_cdf97_s1_s, -1, stride);
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

void dwt_cdf53_i_ex_stride_inplace_s(
	float *tmp,
	int N,
	int stride
)
{
	assert( N >= 0 && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			*addr1_s(tmp, 0, stride) *= dwt_cdf53_s2_s;
		return;
	}

	// inverse scale
	for(int i=0; i<N; i+=2)
		*addr1_s(tmp, i, stride) *= dwt_cdf53_s2_s;
	for(int i=1; i<N; i+=2)
		*addr1_s(tmp, i, stride) *= dwt_cdf53_s1_s;

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		*addr1_s(tmp, i, stride) -= dwt_cdf53_u1_s * (*addr1_s(tmp, i-1, stride) + *addr1_s(tmp, i+1, stride));

	*addr1_s(tmp, 0, stride) -= 2 * dwt_cdf53_u1_s * *addr1_s(tmp, 1, stride);

	if( is_odd(N) )
		*addr1_s(tmp, N-1, stride) -= 2 * dwt_cdf53_u1_s * *addr1_s(tmp, N-2, stride);
	else
		*addr1_s(tmp, N-1, stride) += 2 * dwt_cdf53_p1_s * *addr1_s(tmp, N-2, stride);

	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_s(tmp, i, stride) += dwt_cdf53_p1_s * (*addr1_s(tmp, i-1, stride) + *addr1_s(tmp, i+1, stride));
}

void dwt_eaw53_i_ex_stride_s(
	const float *src_l,
	const float *src_h,
	float *dst,
	float *tmp,
	int N,
	int stride,
	float *w	// float w[N]
)
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
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] -= ( wL*tmp[i-1] + wR*tmp[i+1] ) / ( 2.f*(wL+wR) );
	}

	{
		float wL = w[0];
		float wR = w[0];

		tmp[0] -= (wL * tmp[1] + wR * tmp[1]) / ( 2.f * (wL+wR) );
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] -= (wL * tmp[N-2] + wR * tmp[N-2]) / ( 2.f * (wL+wR) );
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];

		tmp[N-1] += (wL * tmp[N-2] + wR * tmp[N-2]) / (wL+wR);
	}

	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		tmp[i] += ( wL*tmp[i-1] + wR*tmp[i+1] ) / (wL+wR);
	}

	// copy tmp into dst
	dwt_util_memcpy_stride_s(dst, stride, tmp, sizeof(float), N);
}

void dwt_eaw53_i_ex_stride_inplace_s(
	float *tmp,
	int N,
	int stride,
	float *w	// float w[N]
)
{
	assert( N >= 0 && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
	{
		if(1 == N)
			*addr1_s(tmp, 0, stride) *= dwt_cdf53_s2_s;
		return;
	}

	// inverse scale
	for(int i=0; i<N; i+=2)
		*addr1_s(tmp, i, stride) *= dwt_cdf53_s2_s;
	for(int i=1; i<N; i+=2)
		*addr1_s(tmp, i, stride) *= dwt_cdf53_s1_s;

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		*addr1_s(tmp, i, stride) -= ( wL * *addr1_s(tmp, i-1, stride) + wR * *addr1_s(tmp, i+1, stride) ) / ( 2.f*(wL+wR) );
	}

	{
		float wL = w[0];
		float wR = w[0];

		*addr1_s(tmp, 0, stride) -= (wL * *addr1_s(tmp, 1, stride) + wR * *addr1_s(tmp, 1, stride)) / ( 2.f * (wL+wR) );
	}

	if( is_odd(N) )
	{
		float wL = w[N-2];
		float wR = w[N-2];

		*addr1_s(tmp, N-1, stride) -= (wL * *addr1_s(tmp, N-2, stride) + wR * *addr1_s(tmp, N-2, stride)) / ( 2.f * (wL+wR) );
	}
	else
	{
		float wL = w[N-2];
		float wR = w[N-2];

		*addr1_s(tmp, N-1, stride) += (wL * *addr1_s(tmp, N-2, stride) + wR * *addr1_s(tmp, N-2, stride)) / (wL+wR);
	}

	for(int i=1; i<N-2+(N&1); i+=2)
	{
		float wL = w[i-1];
		float wR = w[i+0];

		*addr1_s(tmp, i, stride) += ( wL * *addr1_s(tmp, i-1, stride) + wR * *addr1_s(tmp, i+1, stride) ) / (wL+wR);
	}
}

void dwt_interp53_i_ex_stride_s(
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

	if(is_odd(N))
		;
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

#ifdef __asvp__
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

/** allocated memory aligned on current platform for type of size of elem_size bytes */
static
void *alloc_aligned(
	int elements,
	size_t elem_size
)
{
	assert( is_pow2(elem_size) );

	// alignment for type of given size
	const size_t align = alignment(elem_size);

	const size_t size = elements * elem_size;

	void *addr = (void *)0;

	addr = (void *)memalign(align, size);

	assert( is_aligned(addr, align) );

	return addr;
}

void *dwt_util_alloc(
	int elems,
	size_t elem_size
)
{
	return malloc(elems*elem_size);
}

static
void **alloc_temp(
	int threads,
	int elements,
	size_t elem_size
)
{
	void **temp;

	temp = (void **)malloc( sizeof(void*) * threads );
	if( !temp )
		dwt_util_error("malloc fails!\n");

	for(int t = 0; t < threads; t++)
	{
		temp[t] = alloc_aligned(elements, elem_size);

		if( !temp[t] )
			dwt_util_error("Unable to allocate temp[] buffer!\n");
	}

	return temp;
}

static
float **alloc_temp_s(
	int threads,
	int elements
)
{
	return (float **)alloc_temp(threads, elements, sizeof(float));
}

static
double **alloc_temp_d(
	int threads,
	int elements
)
{
	return (double **)alloc_temp(threads, elements, sizeof(double));
}

static
int **alloc_temp_i(
	int threads,
	int elements
)
{
	return (int **)alloc_temp(threads, elements, sizeof(int));
}

static
void free_temp(
	int threads,
	void **temp
)
{
	for(int t = 0; t < threads; t++)
		free(temp[t]);
	free(temp);
}

static
void free_temp_s(
	int threads,
	float **temp
)
{
	free_temp(threads, (void **)temp);
}

static
void free_temp_d(
	int threads,
	double **temp
)
{
	free_temp(threads, (void **)temp);
}

static
void free_temp_i(
	int threads,
	int **temp
)
{
	free_temp(threads, (void **)temp);
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

	// FIXME(microblaze): align on 8 bytes boundary (GCC's __attribure__ is ignored)
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

	// FIXME(microblaze): align on 8 bytes boundary (GCC's __attribure__ is ignored)
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

	const int threads = dwt_util_get_num_threads();
	const int workers = dwt_util_get_num_workers();

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SA);
#endif
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	const int offset = 1;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_max, offset)
	);

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
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < workers_lines_y; y += workers)
			{
				dwt_cdf97_f_ex_stride_s(
					addr2_const_s(src,y,0,stride_x,stride_y),
					addr2_s(dst,y,0,stride_x,stride_y),
					addr2_s(dst,y,size_o_dst_x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
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
					temp[dwt_util_get_thread_num()],
					size_i_src_x,
					stride_y);
			}
			dwt_util_set_num_workers(workers);
			// in the next iteration, the dst takes the role of src
			// otherwise, the src will be unaffected in the second iteration
			src = dst;
		}
#endif

#ifndef DISABLE_X
		if( lines_y > 1 )
		{
			set_data_step_s( stride_y );
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < workers_lines_x; x += workers)
			{
				dwt_cdf97_f_ex_stride_s(
					addr2_const_s(src,0,x,stride_x,stride_y),
					addr2_s(dst,0,x,stride_x,stride_y),
					addr2_s(dst,size_o_dst_y,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_src_y,
					stride_x);
			}
			dwt_util_set_num_workers(1);
			for(int x = workers_lines_x; x < lines_x; x++)
			{
				dwt_cdf97_f_ex_stride_s(
					addr2_const_s(src,0,x,stride_x,stride_y),
					addr2_s(dst,0,x,stride_x,stride_y),
					addr2_s(dst,size_o_dst_y,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_src_y,
					stride_x);
			}
			dwt_util_set_num_workers(workers);
			// in the next iteration, the dst takes the role of src
			// otherwise, the src will be unaffected in the second iteration
			src = dst;
		}
#endif

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

	free_temp_s(threads, temp);

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

	const int threads = dwt_util_get_num_threads();
	const int workers = dwt_util_get_num_workers();

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SA);
#endif
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	const int offset = 1;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_max, offset)
	);

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
			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < workers_lines_y; y += workers)
			{
				dwt_cdf97_f_ex_stride_s(
					addr2_const_s(ptr,y,0,stride_x,stride_y),
					addr2_s(ptr,y,0,stride_x,stride_y),
					addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_src_x,
					stride_y);
			}
			dwt_util_set_num_workers(1);
			for(int y = workers_lines_y; y < lines_y; y++)
			{
				dwt_cdf97_f_ex_stride_s(
					addr2_const_s(ptr,y,0,stride_x,stride_y),
					addr2_s(ptr,y,0,stride_x,stride_y),
					addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
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
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < workers_lines_x; x += workers)
			{
				dwt_cdf97_f_ex_stride_s(
					addr2_const_s(ptr,0,x,stride_x,stride_y),
					addr2_s(ptr,0,x,stride_x,stride_y),
					addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_src_y,
					stride_x);
			}
			dwt_util_set_num_workers(1);
			for(int x = workers_lines_x; x < lines_x; x++)
			{
				dwt_cdf97_f_ex_stride_s(
					addr2_const_s(ptr,0,x,stride_x,stride_y),
					addr2_s(ptr,0,x,stride_x,stride_y),
					addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
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

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_cdf97_2f_inplace_s(
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

	assert( 1 == dwt_util_get_num_workers() );

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	const int offset = 1;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

// 		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
// 		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
// 		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
// 		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		const int stride_y_j = stride_y * (1 << (j));
		const int stride_x_j = stride_x * (1 << (j));

		const int size_x = size_i_src_x;
		const int size_y = size_i_src_y;

		const int pairs_x = (to_even(size_x-offset)-4)/2;
// 		const int pairs_y = (to_even(size_y-offset)-4)/2;

		const int max_y = to_even(size_y-offset)+offset;

		if( size_x > 1 && size_x < 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y < 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		if( size_x > 1 && size_x >= 5 && size_y > 1 && size_y >= 5 )
		{
			// this should be stored in CPU cache
			float l_buff[4 * size_x];

			// for y=0 to offset step 1: horizontal only, no vertical
			for(int y = 0; y < offset; y += 1)
			{
				float *ptr1_x = addr2_s(ptr, y+0, offset, stride_x_j, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_s(
					ptr1_x,
					pairs_x,
					-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride_y_j);
			}
			// for y=offset to offset+4 step 2: horizontal, vertical_prolog0
			for(int y = offset; y < offset+2; y += 2)
			{
				float *ptr1_x = addr2_s(ptr, y+0, offset, stride_x_j, stride_y_j);
				float *ptr2_x = addr2_s(ptr, y+1, offset, stride_x_j, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_s(
					ptr1_x,
					pairs_x,
					-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_s(
					ptr2_x,
					pairs_x,
					-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride_y_j);

				for(int x = 0; x < size_x; x++)
				{
					float *l4 = &l_buff[4*x];

					// input addr for 1st coeff in the pair
					float *ptr0_y = addr2_s(ptr, y+0, x, stride_x_j, stride_y_j);
					// input addr for 2nd coeff in the pair
					float *ptr1_y = addr2_s(ptr, y+1, x, stride_x_j, stride_y_j);
					// output addr for 1st coeff in the pair
					float *out0_y = addr2_s(ptr, y+0-4, x, stride_x_j, stride_y_j);
					// output addr for 2nd coeff in the pair
					float *out1_y = addr2_s(ptr, y+1-4, x, stride_x_j, stride_y_j);

					accel_lift_op4s_fwd_main_dl_stride_pair_prolog0_s(
						ptr0_y, // in
						ptr1_y, // in
						out0_y, // out
						out1_y, // out
						-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s,
						l4
					);
				}
			}
			// for y=offset to offset+4 step 2: horizontal, vertical_prolog1
			for(int y = offset+2; y < offset+4; y += 2)
			{
				float *ptr1_x = addr2_s(ptr, y+0, offset, stride_x_j, stride_y_j);
				float *ptr2_x = addr2_s(ptr, y+1, offset, stride_x_j, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_s(
					ptr1_x,
					pairs_x,
					-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_s(
					ptr2_x,
					pairs_x,
					-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride_y_j);

				for(int x = 0; x < size_x; x++)
				{
					float *l4 = &l_buff[4*x];

					// input addr for 1st coeff in the pair
					float *ptr0_y = addr2_s(ptr, y+0, x, stride_x_j, stride_y_j);
					// input addr for 2nd coeff in the pair
					float *ptr1_y = addr2_s(ptr, y+1, x, stride_x_j, stride_y_j);
					// output addr for 1st coeff in the pair
					float *out0_y = addr2_s(ptr, y+0-4, x, stride_x_j, stride_y_j);
					// output addr for 2nd coeff in the pair
					float *out1_y = addr2_s(ptr, y+1-4, x, stride_x_j, stride_y_j);

					accel_lift_op4s_fwd_main_dl_stride_pair_prolog1_s(
						ptr0_y, // in
						ptr1_y, // in
						out0_y, // out
						out1_y, // out
						-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s,
						l4
					);
				}
			}
			// for y=offset+4 to max_y step 2: horizontal, vertical_core
			for(int y = offset+4; y < max_y; y += 2)
			{
				float *ptr1_x = addr2_s(ptr, y+0, offset, stride_x_j, stride_y_j);
				float *ptr2_x = addr2_s(ptr, y+1, offset, stride_x_j, stride_y_j);
				float *out1_x = addr2_s(ptr, y+0-4, offset, stride_x_j, stride_y_j);
				float *out2_x = addr2_s(ptr, y+1-4, offset, stride_x_j, stride_y_j);

				float *l4 = l_buff;

				// offset column
				for(int x = 0; x < offset; x++)
				{
					// input addr for 1st coeff in the pair
					float *ptr0_y = addr2_s(ptr, y+0, x, stride_x_j, stride_y_j);
					// input addr for 2nd coeff in the pair
					float *ptr1_y = addr2_s(ptr, y+1, x, stride_x_j, stride_y_j);
					// output addr for 1st coeff in the pair
					float *out0_y = addr2_s(ptr, y+0-4, x, stride_x_j, stride_y_j);
					// output addr for 2nd coeff in the pair
					float *out1_y = addr2_s(ptr, y+1-4, x, stride_x_j, stride_y_j);

					accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
						ptr0_y, // in
						ptr1_y, // in
						out0_y, // out
						out1_y, // out
						-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s,
						l4
					);
				}
				l4 += 4;

				float alpha = -dwt_cdf97_p1_s;
				float beta = dwt_cdf97_u1_s;
				float gamma = -dwt_cdf97_p2_s;
				float delta = dwt_cdf97_u2_s;
				float zeta = dwt_cdf97_s1_s;

				float l1[4];
				float l2[4];

				accel_lift_op4s_fwd_main_dl_stride_pair_prolog0_s(
					addr1_s(ptr1_x, 0, stride_y_j),
					addr1_s(ptr1_x, 1, stride_y_j),
					NULL,
					NULL,
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l1
				);
				accel_lift_op4s_fwd_main_dl_stride_pair_prolog0_s(
					addr1_s(ptr2_x, 0, stride_y_j),
					addr1_s(ptr2_x, 1, stride_y_j),
					NULL,
					NULL,
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l2
				);
				ptr1_x = addr1_s(ptr1_x, 2, stride_y_j);
				ptr2_x = addr1_s(ptr2_x, 2, stride_y_j);
				out1_x = addr1_s(out1_x, 2, stride_y_j);
				out2_x = addr1_s(out2_x, 2, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_pair_prolog1_s(
					addr1_s(ptr1_x, 0, stride_y_j),
					addr1_s(ptr1_x, 1, stride_y_j),
					NULL,
					NULL,
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l1
				);
				accel_lift_op4s_fwd_main_dl_stride_pair_prolog1_s(
					addr1_s(ptr2_x, 0, stride_y_j),
					addr1_s(ptr2_x, 1, stride_y_j),
					NULL,
					NULL,
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l2
				);
				ptr1_x = addr1_s(ptr1_x, 2, stride_y_j);
				ptr2_x = addr1_s(ptr2_x, 2, stride_y_j);
				out1_x = addr1_s(out1_x, 2, stride_y_j);
				out2_x = addr1_s(out2_x, 2, stride_y_j);

				// loop by pairs from left to right
				for(int s = 0; s < pairs_x; s++)
				{
					float *ptr_y0_x0 = addr1_s(ptr1_x, 0, stride_y_j);
					float *ptr_y0_x1 = addr1_s(ptr1_x, 1, stride_y_j);
					float *ptr_y1_x0 = addr1_s(ptr2_x, 0, stride_y_j);
					float *ptr_y1_x1 = addr1_s(ptr2_x, 1, stride_y_j);

					float *out_y0_x0 = addr1_s(out1_x, 0-4, stride_y_j);
					float *out_y0_x1 = addr1_s(out1_x, 1-4, stride_y_j);
					float *out_y1_x0 = addr1_s(out2_x, 0-4, stride_y_j);
					float *out_y1_x1 = addr1_s(out2_x, 1-4, stride_y_j);

#if 1
					accel_lift_op4s_fwd_main_dl_stride_pair_core_2x2_s(
						ptr_y0_x0, // in
						ptr_y0_x1, // in
						ptr_y1_x0, // in
						ptr_y1_x1, // in
						out_y0_x0, // out
						out_y0_x1, // out
						out_y1_x0, // out
						out_y1_x1, // out
						alpha, // w
						beta, // w
						gamma, // w
						delta, // w
						zeta, // v
						l1, // [4]
						l2, // [4]
						l4+0, // [4]
						l4+4  // [4]
					);
#else
					// BUG: this cannot work with contemporary prologs/epilogs
					cdf97_fwd_core_dl_sc_sse_2x2_s(
						ptr_y0_x0, // in
						ptr_y0_x1, // in
						ptr_y1_x0, // in
						ptr_y1_x1, // in
						out_y0_x0, // out
						out_y0_x1, // out
						out_y1_x0, // out
						out_y1_x1, // out
						l1, // [4]
						l2, // [4]
						l4+0, // [4]
						l4+4  // [4]
					);
#endif
					l4 += 8;

					// update pointers
					ptr1_x = addr1_s(ptr1_x, 2, stride_y_j);
					ptr2_x = addr1_s(ptr2_x, 2, stride_y_j);
					out1_x = addr1_s(out1_x, 2, stride_y_j);
					out2_x = addr1_s(out2_x, 2, stride_y_j);
				}

				accel_lift_op4s_fwd_main_dl_stride_pair_epilog0_s(
					NULL,
					NULL,
					addr1_s(ptr1_x, 0-4, stride_y_j),
					addr1_s(ptr1_x, 1-4, stride_y_j),
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l1
				);
				accel_lift_op4s_fwd_main_dl_stride_pair_epilog0_s(
					NULL,
					NULL,
					addr1_s(ptr2_x, 0-4, stride_y_j),
					addr1_s(ptr2_x, 1-4, stride_y_j),
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l2
				);
				ptr1_x = addr1_s(ptr1_x, 2, stride_y_j);
				ptr2_x = addr1_s(ptr2_x, 2, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_pair_epilog1_s(
					NULL,
					NULL,
					addr1_s(ptr1_x, 0-4, stride_y_j),
					addr1_s(ptr1_x, 1-4, stride_y_j),
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l1
				);
				accel_lift_op4s_fwd_main_dl_stride_pair_epilog1_s(
					NULL,
					NULL,
					addr1_s(ptr2_x, 0-4, stride_y_j),
					addr1_s(ptr2_x, 1-4, stride_y_j),
					alpha,
					beta,
					gamma,
					delta,
					zeta,
					l2
				);

				// perhaps, this loop can be interleaved with epilog
				for(int x = 2*pairs_x+offset; x < size_x; x++)
				{
					float *l4 = &l_buff[4*x];

					// input addr for 1st coeff in the pair
					float *ptr0_y = addr2_s(ptr, y+0, x, stride_x_j, stride_y_j);
					// input addr for 2nd coeff in the pair
					float *ptr1_y = addr2_s(ptr, y+1, x, stride_x_j, stride_y_j);
					// output addr for 1st coeff in the pair
					float *out0_y = addr2_s(ptr, y+0-4, x, stride_x_j, stride_y_j);
					// output addr for 2nd coeff in the pair
					float *out1_y = addr2_s(ptr, y+1-4, x, stride_x_j, stride_y_j);

					accel_lift_op4s_fwd_main_dl_stride_pair_core_s(
						ptr0_y, // in
						ptr1_y, // in
						out0_y, // out
						out1_y, // out
						-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s,
						l4
					);
				}
			}
			for(int y = max_y; y < size_y; y++)
			{
				float *ptr1_x = addr2_s(ptr, y+0, offset, stride_x_j, stride_y_j);

				accel_lift_op4s_fwd_main_dl_stride_s(
					ptr1_x,
					pairs_x,
					-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s, +1, stride_y_j);
			}
			// for y=max_y to max_y+4 step 2: no horizontal, vertical_epilog0 only
			for(int y = max_y; y < max_y+2; y += 2)
			{
				for(int x = 0; x < size_x; x++)
				{
					float *l4 = &l_buff[4*x];

					// input addr for 1st coeff in the pair
					float *ptr0_y = addr2_s(ptr, y+0, x, stride_x_j, stride_y_j);
					// input addr for 2nd coeff in the pair
					float *ptr1_y = addr2_s(ptr, y+1, x, stride_x_j, stride_y_j);
					// output addr for 1st coeff in the pair
					float *out0_y = addr2_s(ptr, y+0-4, x, stride_x_j, stride_y_j);
					// output addr for 2nd coeff in the pair
					float *out1_y = addr2_s(ptr, y+1-4, x, stride_x_j, stride_y_j);

					accel_lift_op4s_fwd_main_dl_stride_pair_epilog0_s(
						ptr0_y, // in
						ptr1_y, // in
						out0_y, // out
						out1_y, // out
						-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s,
						l4
					);
				}
			}
			// for y=max_y to max_y+4 step 2: no horizontal, vertical_epilog1 only
			for(int y = max_y+2; y < max_y+4; y += 2)
			{
				for(int x = 0; x < size_x; x++)
				{
					float *l4 = &l_buff[4*x];

					// input addr for 1st coeff in the pair
					float *ptr0_y = addr2_s(ptr, y+0, x, stride_x_j, stride_y_j);
					// input addr for 2nd coeff in the pair
					float *ptr1_y = addr2_s(ptr, y+1, x, stride_x_j, stride_y_j);
					// output addr for 1st coeff in the pair
					float *out0_y = addr2_s(ptr, y+0-4, x, stride_x_j, stride_y_j);
					// output addr for 2nd coeff in the pair
					float *out1_y = addr2_s(ptr, y+1-4, x, stride_x_j, stride_y_j);

					accel_lift_op4s_fwd_main_dl_stride_pair_epilog1_s(
						ptr0_y, // in
						ptr1_y, // in
						out0_y, // out
						out1_y, // out
						-dwt_cdf97_p1_s, dwt_cdf97_u1_s, -dwt_cdf97_p2_s, dwt_cdf97_u2_s, dwt_cdf97_s1_s,
						l4
					);
				}
			}
		}
		else
		{
			if( size_x > 1 && size_x >= 5 )
			{
				for(int y = 0; y < size_i_src_y; y++)
				{
					dwt_cdf97_f_ex_stride_inplace_part_core_s(
						addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
						size_x, // N
						stride_y_j);
				}
			}
			if( size_y > 1 && size_y >= 5 )
			{
				for(int x = 0; x < size_x; x++)
				{
					dwt_cdf97_f_ex_stride_inplace_part_core_s(
						addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
						size_y, // N
						stride_x_j);
				}
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		// TODO
// 		if(zero_padding)
// 		{
// 			#pragma omp parallel for schedule(static, threads_segment_y)
// 			for(int y = 0; y < size_o_src_y; y++)
// 				dwt_zero_padding_f_stride_s(
// 					addr2_s(ptr,y,0,stride_x,stride_y),
// 					addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
// 					size_i_src_x,
// 					size_o_dst_x,
// 					size_o_src_x-size_o_dst_x,
// 					stride_y);
// 			#pragma omp parallel for schedule(static, threads_segment_x)
// 			for(int x = 0; x < size_o_src_x; x++)
// 				dwt_zero_padding_f_stride_s(
// 					addr2_s(ptr,0,x,stride_x,stride_y),
// 					addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
// 					size_i_src_y,
// 					size_o_dst_y,
// 					size_o_src_y-size_o_dst_y,
// 					stride_x);
// 		}

		j++;
	}

	FUNC_END;
}

// TODO: test it
// two-loops, not a single loop
void dwt_cdf97_2f_inplace_sep_s(
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

	assert( 1 == dwt_util_get_num_workers() );

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

// 	const int offset = 1;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

// 		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
// 		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
// 		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
// 		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		const int stride_y_j = stride_y * (1 << (j));
		const int stride_x_j = stride_x * (1 << (j));

		const int size_x = size_i_src_x;
		const int size_y = size_i_src_y;

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y, threads); // FIXME: should be size_o_src_y?
		const int threads_segment_x = ceil_div(size_x, threads);
#endif

// 		const int pairs_x = (to_even(size_x-offset)-4)/2;
// 		const int pairs_y = (to_even(size_y-offset)-4)/2;

// 		const int max_y = to_even(size_y-offset)+offset;

		if( size_x > 1 && size_x < 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y < 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		if( 1 )
		{
			if( size_x > 1 && size_x >= 5 )
			{
				#pragma omp parallel for schedule(static, threads_segment_y)
				for(int y = 0; y < size_i_src_y; y++)
				{
					dwt_cdf97_f_ex_stride_inplace_part_core_s(
						addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
						size_x, // N
						stride_y_j);
				}
			}
			if( size_y > 1 && size_y >= 5 )
			{
				#pragma omp parallel for schedule(static, threads_segment_x)
				for(int x = 0; x < size_x; x++)
				{
					dwt_cdf97_f_ex_stride_inplace_part_core_s(
						addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
						size_y, // N
						stride_x_j);
				}
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		j++;
	}

	FUNC_END;
}

// two-loops, not a single loop
void dwt_cdf97_2f_inplace_sep_sdl_s(
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

	assert( 1 == dwt_util_get_num_workers() );

#ifdef _OPENMP
	const int threads = dwt_util_get_num_threads();
#endif

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

// 	const int offset = 1;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

// 		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
// 		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
// 		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
// 		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		const int stride_y_j = stride_y * (1 << (j));
		const int stride_x_j = stride_x * (1 << (j));

		const int size_x = size_i_src_x;
		const int size_y = size_i_src_y;

#ifdef _OPENMP
		const int threads_segment_y = ceil_div(size_y, threads); // FIXME: should be size_o_src_y?
		const int threads_segment_x = ceil_div(size_x, threads);
#endif

// 		const int pairs_x = (to_even(size_x-offset)-4)/2;
// 		const int pairs_y = (to_even(size_y-offset)-4)/2;

// 		const int max_y = to_even(size_y-offset)+offset;

		if( size_x > 1 && size_x < 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y < 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		if( 1 )
		{
			if( size_x > 1 && size_x >= 5 )
			{
				#pragma omp parallel for schedule(static, threads_segment_y)
				for(int y = 0; y < size_i_src_y; y++)
				{
#ifdef __SSE__
					dwt_cdf97_f_ex_stride_inplace_part_core_sdl_sse_s(
#else
					dwt_cdf97_f_ex_stride_inplace_part_core_sdl_s(
#endif
						addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
						size_x, // N
						stride_y_j);
				}
			}
			if( size_y > 1 && size_y >= 5 )
			{
				#pragma omp parallel for schedule(static, threads_segment_x)
				for(int x = 0; x < size_x; x++)
				{
#ifdef __SSE__
					dwt_cdf97_f_ex_stride_inplace_part_core_sdl_sse_s(
#else
					dwt_cdf97_f_ex_stride_inplace_part_core_sdl_s(
#endif
						addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
						size_y, // N
						stride_x_j);
				}
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		j++;
	}

	FUNC_END;
}

#ifdef __SSE__
static
void op4_fwd_sdl_2x1A_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *out, // output as __m128 in format [ L0 L1 R0 R1 ]
	const float *w, const float *v,
	float *lL, float *cL, float *rL,
	float *lR, float *cR, float *rR
)
{
	__m128 buff;
	__m128 zL, zR;

	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cL, *(__m128 *)rL);
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cR, *(__m128 *)rR);

	op4s_sdl2_op_s_sse(zL, *(__m128 *)cL, *(__m128 *)w, *(__m128 *)lL, *(__m128 *)rL);
	op4s_sdl2_op_s_sse(zR, *(__m128 *)cR, *(__m128 *)w, *(__m128 *)lR, *(__m128 *)rR);

	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)lL, zL);
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)lR, zR);

	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	*(__m128 *)out = buff;

	op4s_sdl2_update_s_sse(*(__m128 *)cL, *(__m128 *)lL, *(__m128 *)rL, zL);
	op4s_sdl2_update_s_sse(*(__m128 *)cR, *(__m128 *)lR, *(__m128 *)rR, zR);
}
#endif

#ifdef __SSE__
static
void op4_fwd_sdl_2x1B_s(
	float *ptr, // input as __m128 in format [ L0 L1 R0 R1 ]
	float *outL0, float *outL1,
	float *outR0, float *outR1,
	const float *w, const float *v,
	float *lL, float *cL, float *rL,
	float *lR, float *cR, float *rR
)
{
	__m128 buff;
	__m128 zL, zR;

	buff = *(__m128 *)ptr;

	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cL, *(__m128 *)rL);
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cR, *(__m128 *)rR);

	op4s_sdl2_op_s_sse(zL, *(__m128 *)cL, *(__m128 *)w, *(__m128 *)lL, *(__m128 *)rL);
	op4s_sdl2_op_s_sse(zR, *(__m128 *)cR, *(__m128 *)w, *(__m128 *)lR, *(__m128 *)rR);

	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)lL, zL);
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)lR, zR);

	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];

	op4s_sdl2_update_s_sse(*(__m128 *)cL, *(__m128 *)lL, *(__m128 *)rL, zL);
	op4s_sdl2_update_s_sse(*(__m128 *)cR, *(__m128 *)lR, *(__m128 *)rR, zR);
}
#endif

#ifdef __SSE__
static
void cdf97_fwd_core2_sdl_2x2_sc_sse_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outL1,
	float *outR0, float *outR1,
	float *lAL, float *cAL, float *rAL,
	float *lAR, float *cAR, float *rAR,
	float *lBL, float *cBL, float *rBL,
	float *lBR, float *cBR, float *rBR
)
{
	UNUSED(cAL);
	UNUSED(rAL);
	UNUSED(cAR);
	UNUSED(rAR);
	UNUSED(cBL);
	UNUSED(rBL);
	UNUSED(cBR);
	UNUSED(rBR);

	const __m128 w = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
	const __m128 v_vert = { 1/(dwt_cdf97_s1_s*dwt_cdf97_s1_s), 1.f, 1.f, (dwt_cdf97_s1_s*dwt_cdf97_s1_s) };

	__m128 buff;
	__m128 z;

	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	// A/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lAL+4), *(__m128 *)(lAL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lAR+4), *(__m128 *)(lAR+8));

	// A/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAL+4), w, *(__m128 *)(lAL+0), *(__m128 *)(lAL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lAL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAL+4), *(__m128 *)(lAL+0), *(__m128 *)(lAL+8), z);

	// A/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAR+4), w, *(__m128 *)(lAR+0), *(__m128 *)(lAR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lAR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAR+4), *(__m128 *)(lAR+0), *(__m128 *)(lAR+8), z);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lBL+4), *(__m128 *)(lBL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lBR+4), *(__m128 *)(lBR+8));

	// B/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBL+4), w, *(__m128 *)(lBL+0), *(__m128 *)(lBL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lBL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lBL+4), *(__m128 *)(lBL+0), *(__m128 *)(lBL+8), z);

	// B/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBR+4), w, *(__m128 *)(lBR+0), *(__m128 *)(lBR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lBR+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(lBR+4), *(__m128 *)(lBR+0), *(__m128 *)(lBR+8), z);

	// B/L+R
	op4s_sdl2_scale_s_sse(buff, v_vert);

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
}
#endif

#ifdef __SSE__
static
void op4_fwd_sdl_2x2_fast_s(
	float *ptrL0, float *ptrL1,
	float *ptrR0, float *ptrR1,
	float *outL0, float *outL1,
	float *outR0, float *outR1,
	const float *w, const float *v,
	float *lAL, float *cAL, float *rAL,
	float *lAR, float *cAR, float *rAR,
	float *lBL, float *cBL, float *rBL,
	float *lBR, float *cBR, float *rBR
)
{
#if 0
	__m128 tmp;

	// A
	op4_fwd_sdl_2x1A_s(
		ptrL0, ptrL1,
		ptrR0, ptrR1,
		(float *)&tmp,
		w, v,
		lAL, cAL, rAL,
		lAR, cAR, rAR
	);

	// swap
	tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(3,1,2,0));

	// B
	op4_fwd_sdl_2x1B_s(
		(float *)&tmp,
		outL0, outL1,
		outR0, outR1,
		w, v,
		lBL, cBL, rBL,
		lBR, cBR, rBR
	);
#endif
#if 0
	__m128 buff;
	__m128 zL, zR;

	// FIXME: if there is a pressure for number of SSE registers, it is possible to merge zL and zR into z (get rid interleaving)

	// A
	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cAL, *(__m128 *)rAL);
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cAR, *(__m128 *)rAR);

	op4s_sdl2_op_s_sse(zL, *(__m128 *)cAL, *(__m128 *)w, *(__m128 *)lAL, *(__m128 *)rAL);
	op4s_sdl2_op_s_sse(zR, *(__m128 *)cAR, *(__m128 *)w, *(__m128 *)lAR, *(__m128 *)rAR);

	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)lAL, zL);
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)lAR, zR);

	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	op4s_sdl2_update_s_sse(*(__m128 *)cAL, *(__m128 *)lAL, *(__m128 *)rAL, zL);
	op4s_sdl2_update_s_sse(*(__m128 *)cAR, *(__m128 *)lAR, *(__m128 *)rAR, zR);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cBL, *(__m128 *)rBL);
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cBR, *(__m128 *)rBR);

	op4s_sdl2_op_s_sse(zL, *(__m128 *)cBL, *(__m128 *)w, *(__m128 *)lBL, *(__m128 *)rBL);
	op4s_sdl2_op_s_sse(zR, *(__m128 *)cBR, *(__m128 *)w, *(__m128 *)lBR, *(__m128 *)rBR);

	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)lBL, zL);
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)lBR, zR);

	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	op4s_sdl2_update_s_sse(*(__m128 *)cBL, *(__m128 *)lBL, *(__m128 *)rBL, zL);
	op4s_sdl2_update_s_sse(*(__m128 *)cBR, *(__m128 *)lBR, *(__m128 *)rBR, zR);

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
#endif
#if 0
	__m128 buff;

	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	__m128 z;
	__m128 t;

	__asm__ __volatile__(
		// op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cAL, *(__m128 *)rAL);
		"movaps %[buff], %[t] \n\t"// (t) = (buff);
		"shufps %[shuffle3210], %[cAL], %[t] \n\t" // (t) = _mm_shuffle_ps((t), (*(__m128 *)cAL), _MM_SHUFFLE(3,2,1,0));
		"shufps %[shuffle0321], %[t], %[cAL] \n\t" // (*(__m128 *)cAL) = _mm_shuffle_ps((*(__m128 *)cAL), (t), _MM_SHUFFLE(0,3,2,1));
		"shufps %[shuffle3210], %[rAL], %[t] \n\t" // (t) = _mm_shuffle_ps((t), (*(__m128 *)rAL), _MM_SHUFFLE(3,2,1,0));
		"shufps %[shuffle1321], %[t], %[rAL] \n\t" // (*(__m128 *)rAL) = _mm_shuffle_ps((*(__m128 *)rAL), (t), _MM_SHUFFLE(1,3,2,1));
		// op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cAR, *(__m128 *)rAR);
		"shufps %[shuffle3232], %[cAR], %[buff] \n\t" // (buff) = _mm_shuffle_ps( (buff), (*(__m128 *)cAR), _MM_SHUFFLE(3,2,3,2) );
		"shufps %[shuffle0321], %[buff], %[cAR] \n\t" // (*(__m128 *)cAR)  = _mm_shuffle_ps( (*(__m128 *)cAR), (buff), _MM_SHUFFLE(0,3,2,1) );
		"shufps %[shuffle3210], %[rAR], %[buff] \n\t" // (buff) = _mm_shuffle_ps( (buff), (*(__m128 *)rAR), _MM_SHUFFLE(3,2,1,0) );
		"shufps %[shuffle1321], %[buff], %[rAR] \n\t" // (*(__m128 *)rAR)  = _mm_shuffle_ps( (*(__m128 *)rAR), (buff), _MM_SHUFFLE(1,3,2,1) );
		// op4s_sdl2_op_s_sse(z, *(__m128 *)cAL, *(__m128 *)w, *(__m128 *)lAL, *(__m128 *)rAL);
		"movaps %[lAL], %[z] \n\t" // (z) = (*(__m128 *)lAL);
		"addps %[rAL], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)rAL));
		"mulps %[w], %[z] \n\t" // (z) = _mm_mul_ps((z), (*(__m128 *)w));
		"addps %[cAL], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)cAL));
		// op4s_sdl2_output_low_s_sse(buff, *(__m128 *)lAL, z);
		"movaps %[lAL], %[buff] \n\t" // (buff) = (*(__m128 *)lAL);
		"unpcklps %[z], %[buff] \n\t" // (buff) = _mm_unpacklo_ps((buff), (z));
		// op4s_sdl2_update_s_sse(*(__m128 *)cAL, *(__m128 *)lAL, *(__m128 *)rAL, z);
		"movaps %[lAL], %[cAL] \n\t" // (*(__m128 *)cAL) = ( *(__m128 *)lAL);
		"movaps %[rAL], %[lAL] \n\t" // ( *(__m128 *)lAL) = (*(__m128 *)rAL);
		"movaps %[z], %[rAL] \n\t" // (*(__m128 *)rAL) = (z);
		// op4s_sdl2_op_s_sse(z, *(__m128 *)cAR, *(__m128 *)w, *(__m128 *)lAR, *(__m128 *)rAR);
		"movaps %[lAR], %[z] \n\t" // (z) = (*(__m128 *)lAR);
		"addps %[rAR], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)rAR));
		"mulps %[w], %[z] \n\t" // (z) = _mm_mul_ps((z), (*(__m128 *)w));
		"addps %[cAR], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)cAR));
		// op4s_sdl2_output_high_s_sse(buff, *(__m128 *)lAR, z);
		"movaps %[lAR], %[t] \n\t" // (t) = (*(__m128 *)lAR);
		"unpcklps %[z], %[t] \n\t" // (t) = _mm_unpacklo_ps((t), (z));
		"shufps %[shuffle1010], %[t], %[buff] \n\t" // (buff) = _mm_shuffle_ps((buff), t, _MM_SHUFFLE(1,0,1,0));
		// op4s_sdl2_update_s_sse(*(__m128 *)cAR, *(__m128 *)lAR, *(__m128 *)rAR, z);
		"movaps %[lAR], %[cAR] \n\t" // (*(__m128 *)cAR) = (*(__m128 *)lAR);
		"movaps %[rAR], %[lAR] \n\t" // (*(__m128 *)lAR) = (*(__m128 *)rAR);
		"movaps %[z], %[rAR] \n\t" // (*(__m128 *)rAR) = (z);
		// op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);
		"mulps %[v], %[buff] \n\t" // (buff) = _mm_mul_ps((buff), (*(__m128 *)v));
		// swap
		"shufps %[shuffle3120], %[buff], %[buff] \n\t" // buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));
		: /* input/output */
			// FIXME: some "&" are maybe unnecessary
			[t]"=&x"(t),
			[z]"=&x"(z),
			[buff]"+&x"(buff),
			[cAL]"+&x"(*(__m128 *)cAL),
			[rAL]"+&x"(*(__m128 *)rAL),
			[cAR]"+&x"(*(__m128 *)cAR),
			[rAR]"+&x"(*(__m128 *)rAR),
			[lAL]"+&x"(*(__m128 *)lAL),
			[lAR]"+&x"(*(__m128 *)lAR)
		: /* input only  */
			[w]"x"(*(__m128 *)w),
			[v]"x"(*(__m128 *)v),
			[shuffle3210]"i"(_MM_SHUFFLE(3,2,1,0)),
			[shuffle0321]"i"(_MM_SHUFFLE(0,3,2,1)),
			[shuffle1321]"i"(_MM_SHUFFLE(1,3,2,1)),
			[shuffle3232]"i"(_MM_SHUFFLE(3,2,3,2)),
			[shuffle1010]"i"(_MM_SHUFFLE(1,0,1,0)),
			[shuffle3120]"i"(_MM_SHUFFLE(3,1,2,0))
		: /* clobbered */
	);

	__asm__ __volatile__(
		// op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cBL, *(__m128 *)rBL);
		"movaps %[buff], %[t] \n\t" // (t) = (buff);
		"shufps %[shuffle3210], %[cBL], %[t] \n\t" // (t) = _mm_shuffle_ps((t), (*(__m128 *)cBL), _MM_SHUFFLE(3,2,1,0));
		"shufps %[shuffle0321], %[t], %[cBL] \n\t" // (*(__m128 *)cBL) = _mm_shuffle_ps((*(__m128 *)cBL), (t), _MM_SHUFFLE(0,3,2,1));
		"shufps %[shuffle3210], %[rBL], %[t] \n\t" // (t) = _mm_shuffle_ps((t), (*(__m128 *)rBL), _MM_SHUFFLE(3,2,1,0));
		"shufps %[shuffle1321], %[t], %[rBL] \n\t" // (*(__m128 *)rBL) = _mm_shuffle_ps((*(__m128 *)rBL), (t), _MM_SHUFFLE(1,3,2,1));
		// op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cBR, *(__m128 *)rBR);
		"shufps %[shuffle3232], %[cBR], %[buff] \n\t" // (buff) = _mm_shuffle_ps( (buff), (*(__m128 *)cBR), _MM_SHUFFLE(3,2,3,2) );
		"shufps %[shuffle0321], %[buff], %[cBR] \n\t" // (*(__m128 *)cBR)  = _mm_shuffle_ps( (*(__m128 *)cBR), (buff), _MM_SHUFFLE(0,3,2,1) );
		"shufps %[shuffle3210], %[rBR], %[buff] \n\t" // (buff) = _mm_shuffle_ps( (buff), (*(__m128 *)rBR), _MM_SHUFFLE(3,2,1,0) );
		"shufps %[shuffle1321], %[buff], %[rBR] \n\t" // (*(__m128 *)rBR)  = _mm_shuffle_ps( (*(__m128 *)rBR), (buff), _MM_SHUFFLE(1,3,2,1) );
		// op4s_sdl2_op_s_sse(z, *(__m128 *)cBL, *(__m128 *)w, *(__m128 *)lBL, *(__m128 *)rBL);
		"movaps %[lBL], %[z] \n\t" // (z) = (*(__m128 *)lBL);
		"addps %[rBL], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)rBL));
		"mulps %[w], %[z] \n\t" // (z) = _mm_mul_ps((z), (*(__m128 *)w));
		"addps %[cBL], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)cBL));
		// op4s_sdl2_output_low_s_sse(buff, *(__m128 *)lBL, z);
		"movaps %[lBL], %[buff] \n\t" // (buff) = (*(__m128 *)lBL);
		"unpcklps %[z], %[buff] \n\t" // (buff) = _mm_unpacklo_ps((buff), (z));
		// op4s_sdl2_update_s_sse(*(__m128 *)cBL, *(__m128 *)lBL, *(__m128 *)rBL, z);
		"movaps %[lBL], %[cBL] \n\t" // (*(__m128 *)cBL) = (*(__m128 *)lBL);
		"movaps %[rBL], %[lBL] \n\t" // (*(__m128 *)lBL) = (*(__m128 *)rBL);
		"movaps %[z], %[rBL] \n\t" // (*(__m128 *)rBL) = (z);
		// op4s_sdl2_op_s_sse(z, *(__m128 *)cBR, *(__m128 *)w, *(__m128 *)lBR, *(__m128 *)rBR);
		"movaps %[lBR], %[z] \n\t" // (z) = (*(__m128 *)lBR);
		"addps %[rBR], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)rBR));
		"mulps %[w], %[z] \n\t" // (z) = _mm_mul_ps((z), (*(__m128 *)w));
		"addps %[cBR], %[z] \n\t" // (z) = _mm_add_ps((z), (*(__m128 *)cBR));
		// op4s_sdl2_output_high_s_sse(buff, *(__m128 *)lBR, z);
		"movaps %[lBR], %[t] \n\t" // (t) = (*(__m128 *)lBR);
		"unpcklps %[z], %[t] \n\t" // (t) = _mm_unpacklo_ps((t), (z));
		"shufps %[shuffle1010], %[t], %[buff] \n\t" // (buff) = _mm_shuffle_ps((buff), t, _MM_SHUFFLE(1,0,1,0));
		// op4s_sdl2_update_s_sse(*(__m128 *)cBR, *(__m128 *)lBR, *(__m128 *)rBR, z);
		"movaps %[lBR], %[cBR] \n\t" // (*(__m128 *)cBR) = (*(__m128 *)lBR);
		"movaps %[rBR], %[lBR] \n\t" // (*(__m128 *)lBR) = (*(__m128 *)rBR);
		"movaps %[z], %[rBR] \n\t" // (*(__m128 *)rBR) = (z);
		// op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);
		"mulps %[v], %[buff] \n\t" // (buff) = _mm_mul_ps((buff), (*(__m128 *)v));
		: /* input/output */
			[t]"=&x"(t),
			[z]"=&x"(z),
			[buff]"+&x"(buff),
			[cBL]"+&x"(*(__m128 *)cBL),
			[rBL]"+&x"(*(__m128 *)rBL),
			[cBR]"+&x"(*(__m128 *)cBR),
			[rBR]"+&x"(*(__m128 *)rBR),
			[lBL]"+&x"(*(__m128 *)lBL),
			[lBR]"+&x"(*(__m128 *)lBR)
		: /* input only  */
			[w]"x"(*(__m128 *)w),
			[v]"x"(*(__m128 *)v),
			[shuffle3210]"i"(_MM_SHUFFLE(3,2,1,0)),
			[shuffle0321]"i"(_MM_SHUFFLE(0,3,2,1)),
			[shuffle1321]"i"(_MM_SHUFFLE(1,3,2,1)),
			[shuffle3232]"i"(_MM_SHUFFLE(3,2,3,2)),
			[shuffle1010]"i"(_MM_SHUFFLE(1,0,1,0))
		: /* clobbered */
	);

	// NOTE: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=39847 (error: more than 30 operands in ‘asm’)

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
#endif
#if 1
	UNUSED(cAL);
	UNUSED(rAL);
	UNUSED(cAR);
	UNUSED(rAR);
	UNUSED(cBL);
	UNUSED(rBL);
	UNUSED(cBR);
	UNUSED(rBR);

	__m128 buff;
	__m128 z;

	buff[0] = *ptrL0;
	buff[1] = *ptrL1;
	buff[2] = *ptrR0;
	buff[3] = *ptrR1;

	// A/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lAL+4), *(__m128 *)(lAL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lAR+4), *(__m128 *)(lAR+8));

	// A/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAL+4), *(__m128 *)w, *(__m128 *)(lAL+0), *(__m128 *)(lAL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lAL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAL+4), *(__m128 *)(lAL+0), *(__m128 *)(lAL+8), z);

	// A/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lAR+4), *(__m128 *)w, *(__m128 *)(lAR+0), *(__m128 *)(lAR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lAR+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lAR+4), *(__m128 *)(lAR+0), *(__m128 *)(lAR+8), z);

	// A/L+R
	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B/L+R
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)(lBL+4), *(__m128 *)(lBL+8));
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)(lBR+4), *(__m128 *)(lBR+8));

	// B/L
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBL+4), *(__m128 *)w, *(__m128 *)(lBL+0), *(__m128 *)(lBL+8));
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)(lBL+0), z);
	op4s_sdl2_update_s_sse(*(__m128 *)(lBL+4), *(__m128 *)(lBL+0), *(__m128 *)(lBL+8), z);

	// B/R
	op4s_sdl2_op_s_sse(z, *(__m128 *)(lBR+4), *(__m128 *)w, *(__m128 *)(lBR+0), *(__m128 *)(lBR+8));
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)(lBR+0), z); 
	op4s_sdl2_update_s_sse(*(__m128 *)(lBR+4), *(__m128 *)(lBR+0), *(__m128 *)(lBR+8), z);

	// B/L+R
	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	*outL0 = buff[0];
	*outL1 = buff[1];
	*outR0 = buff[2];
	*outR1 = buff[3];
#endif
}
#endif

static
void op4_fwd_sdl_core_s(
	const float *ptr0,
	const float *ptr1,
	float *out0,
	float *out1,
	const float *w,
	const float *v,
	float *l, float *c, float *r
)
{
#ifndef __SSE__
	float buff[2];
	float t[4];

	op4s_sdl_shuffle_s_ref(c, r);
	buff[0] = *ptr0;
	buff[1] = *ptr1;
	op4s_sdl_input_s_ref(buff, c, r);
	op4s_sdl_op_s_ref(t, c, w, l, r);
	op4s_sdl_output_s_ref(buff, l, t);
	op4s_sdl_scale_s_ref(buff, v);
	*out0 = buff[0];
	*out1 = buff[1];
	op4s_sdl_update_s_ref(c, l, r, t);
#else
	// NOTE: very stupid SSE implementation

	__m128 buff;
	__m128 z;

	buff[0] = *ptr0;
	buff[1] = *ptr1;

	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)c, *(__m128 *)r);
	op4s_sdl2_op_s_sse(z, *(__m128 *)c, *(__m128 *)w, *(__m128 *)l, *(__m128 *)r);
	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)l, z);
	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);
	op4s_sdl2_update_s_sse(*(__m128 *)c, *(__m128 *)l, *(__m128 *)r, z);

	*out0 = buff[0];
	*out1 = buff[1];
#endif
}

static
void op4_fwd_sdl_2x2_s(
	float *ptr_y0_x0,
	float *ptr_y0_x1,
	float *ptr_y1_x0,
	float *ptr_y1_x1,
	float *out_y0_x0,
	float *out_y0_x1,
	float *out_y1_x0,
	float *out_y1_x1,
	const float *w,
	const float *v,
	float *buff_y0,
	float *buff_y1,
	float *buff_x0,
	float *buff_x1
)
{
#ifndef __SSE__
	float a, b, c, d;

	op4_fwd_sdl_core_s(
		ptr_y0_x0,
		ptr_y0_x1,
		&a,
		&b,
		w, v,
		buff_y0+0, buff_y0+4, buff_y0+8
	);
	op4_fwd_sdl_core_s(
		ptr_y1_x0,
		ptr_y1_x1,
		&c,
		&d,
		w, v,
		buff_y1+0, buff_y1+4, buff_y1+8
	);
	op4_fwd_sdl_core_s(
		&a,
		&c,
		out_y0_x0,
		out_y1_x0,
		w, v,
		buff_x0+0, buff_x0+4, buff_x0+8
	);
	op4_fwd_sdl_core_s(
		&b,
		&d,
		out_y0_x1,
		out_y1_x1,
		w, v,
		buff_x1+0, buff_x1+4, buff_x1+8
	);
#else
	op4_fwd_sdl_2x2_fast_s(
		ptr_y0_x0, ptr_y0_x1,
		ptr_y1_x0, ptr_y1_x1,
		out_y0_x0, out_y1_x0,
		out_y0_x1, out_y1_x1,
		w, v,
		buff_y0+0, buff_y0+4, buff_y0+8,
		buff_y1+0, buff_y1+4, buff_y1+8,
		buff_x0+0, buff_x0+4, buff_x0+8,
		buff_x1+0, buff_x1+4, buff_x1+8
	);
#endif
#if 0
	cdf97_fwd_core2_sdl_2x2_sc_sse_s(
		ptr_y0_x0, ptr_y0_x1,
		ptr_y1_x0, ptr_y1_x1,
		out_y0_x0, out_y1_x0,
		out_y0_x1, out_y1_x1,
		buff_y0+0, buff_y0+4, buff_y0+8,
		buff_y1+0, buff_y1+4, buff_y1+8,
		buff_x0+0, buff_x0+4, buff_x0+8,
		buff_x1+0, buff_x1+4, buff_x1+8
	);
#endif
}

static
void op4_fwd_sdl_core_prolog2_2x2_s(
	/*const*/ float *y0x0,
	/*const*/ float *y0x1,
	/*const*/ float *y1x0,
	/*const*/ float *y1x1,
	const float *w,
	const float *v,
	float *lcr_y0,
	float *lcr_y1,
	float *lcr_x0,
	float *lcr_x1
)
{
#ifndef __SSE__
	float a, b, c, d;

	op4_fwd_sdl_core_s(
		y0x0,
		y0x1,
		&a,
		&b,
		w, v,
		lcr_y0+0, lcr_y0+4, lcr_y0+8
	);
	op4_fwd_sdl_core_s(
		y1x0,
		y1x1,
		&c,
		&d,
		w, v,
		lcr_y1+0, lcr_y1+4, lcr_y1+8
	);
	op4_fwd_sdl_prolog2_part_s(
		&a,
		&c,
		w, v,
		lcr_x0+0, lcr_x0+4, lcr_x0+8
	);
	op4_fwd_sdl_prolog2_part_s(
		&b,
		&d,
		w, v,
		lcr_x1+0, lcr_x1+4, lcr_x1+8
	);
#else
	// A y
	float *lAL = lcr_y0+0;
	float *cAL = lcr_y0+4;
	float *rAL = lcr_y0+8;
	float *lAR = lcr_y1+0;
	float *cAR = lcr_y1+4;
	float *rAR = lcr_y1+8;
	// B x
	float *lBL = lcr_x0+0;
	float *cBL = lcr_x0+4;
	float *rBL = lcr_x0+8;
	float *lBR = lcr_x1+0;
	float *cBR = lcr_x1+4;
	float *rBR = lcr_x1+8;

	__m128 buff;
	__m128 zL, zR;

	// A
	buff[0] = *y0x0;
	buff[1] = *y0x1;
	buff[2] = *y1x0;
	buff[3] = *y1x1;

	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cAL, *(__m128 *)rAL);
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cAR, *(__m128 *)rAR);

	op4s_sdl2_op_s_sse(zL, *(__m128 *)cAL, *(__m128 *)w, *(__m128 *)lAL, *(__m128 *)rAL);
	op4s_sdl2_op_s_sse(zR, *(__m128 *)cAR, *(__m128 *)w, *(__m128 *)lAR, *(__m128 *)rAR);

	op4s_sdl2_output_low_s_sse(buff, *(__m128 *)lAL, zL);
	op4s_sdl2_output_high_s_sse(buff, *(__m128 *)lAR, zR);

	op4s_sdl2_scale_s_sse(buff, *(__m128 *)v);

	op4s_sdl2_update_s_sse(*(__m128 *)cAL, *(__m128 *)lAL, *(__m128 *)rAL, zL);
	op4s_sdl2_update_s_sse(*(__m128 *)cAR, *(__m128 *)lAR, *(__m128 *)rAR, zR);

	// swap, this should by done by single shuffle instruction
	buff = _mm_shuffle_ps(buff, buff, _MM_SHUFFLE(3,1,2,0));

	// B
	op4s_sdl2_shuffle_input_low_s_sse(buff, *(__m128 *)cBL, *(__m128 *)rBL);
	op4s_sdl2_shuffle_input_high_s_sse(buff, *(__m128 *)cBR, *(__m128 *)rBR);

	op4s_sdl2_op_s_sse(zL, *(__m128 *)cBL, *(__m128 *)w, *(__m128 *)lBL, *(__m128 *)rBL);
	op4s_sdl2_op_s_sse(zR, *(__m128 *)cBR, *(__m128 *)w, *(__m128 *)lBR, *(__m128 *)rBR);

	op4s_sdl2_update_s_sse(*(__m128 *)cBL, *(__m128 *)lBL, *(__m128 *)rBL, zL);
	op4s_sdl2_update_s_sse(*(__m128 *)cBR, *(__m128 *)lBR, *(__m128 *)rBR, zR);
#endif
}

static
void op4_fwd_sdl_core_prolog2_2x1_s(
	const float *y0x0,
	const float *y0x1,
	float *w,
	float *v,
	float *lcr_y0,
	float *lcr_x0,
	float *lcr_x1,
	int idx
)
{
	op4_fwd_sdl_core_s(
		y0x0,
		y0x1,
		(lcr_x0+0+idx),
		(lcr_x1+0+idx),
		w, v,
		lcr_y0+0, lcr_y0+4, lcr_y0+8
	);
}

static
void op4_fwd_sdl_prolog2_fast_s(
	float *ptr0,
	float *ptr1,
	float *ptr2,
	float *ptr3,
	float *ptr4,
	float *ptr5,
	float *ptr6,
	float *ptr7,
	float *ptr8,
	float *ptr9,
	const float *w,
	const float *v,
	float *lcr
)
{
// part0
	// prolog2: import(3)
	(lcr+0)[3] = *ptr3; // base+3
// part1
	// prolog2: pass-prolog
	op4_fwd_sdl_prolog2_part_s(
		ptr4,
		ptr5,
		w, v,
		lcr+0, lcr+4, lcr+8
	);
// part2
	// prolog2: import(2)
	(lcr+0)[2] = *ptr2; // base+2
// part3
	// prolog2: pass-prolog
	op4_fwd_sdl_prolog2_part_s(
		ptr6,
		ptr7,
		w, v,
		lcr+0, lcr+4, lcr+8
	);
// part4
	// prolog2: import(1)
	(lcr+0)[1] = *ptr1; // base+1
// part5
	// prolog2: pass-prolog
	op4_fwd_sdl_prolog2_part_s(
		ptr8,
		ptr9,
		w, v,
		lcr+0, lcr+4, lcr+8
	);
// part6
	// prolog2: import(0)
	(lcr+0)[0] = *ptr0; // base+0
}

static
void op4_fwd_sdl_s(float *ptr, int stride, int steps, const float *w, const float *v)
{
	float lcr[12];
	float *l = lcr+0;
	float *c = lcr+4;
	float *r = lcr+8;

	op4_fwd_sdl_prolog2_fast_s(
		addr1_s(ptr, 0, stride),
		addr1_s(ptr, 1, stride),
		addr1_s(ptr, 2, stride),
		addr1_s(ptr, 3, stride),
		addr1_s(ptr, 4, stride),
		addr1_s(ptr, 5, stride),
		addr1_s(ptr, 6, stride),
		addr1_s(ptr, 7, stride),
		addr1_s(ptr, 8, stride),
		addr1_s(ptr, 9, stride),
		w, v,
		lcr
	);

	for(int s = 0; s < steps-3; s++)
	{
		op4_fwd_sdl_core_s(
			addr1_s(ptr, 10, stride),
			addr1_s(ptr, 11, stride),
			addr1_s(ptr,  0, stride),
			addr1_s(ptr,  1, stride),
			w, v,
			l, c, r
		);

		ptr = addr1_s(ptr, +2, stride);
	}

	op4_fwd_sdl_epilog2_fast_s(
		addr1_s(ptr, 0, stride),
		addr1_s(ptr, 1, stride),
		addr1_s(ptr, 2, stride),
		addr1_s(ptr, 3, stride),
		addr1_s(ptr, 4, stride),
		addr1_s(ptr, 5, stride),
		addr1_s(ptr, 6, stride),
		addr1_s(ptr, 7, stride),
		addr1_s(ptr, 8, stride),
		addr1_s(ptr, 9, stride),
		w, v,
		lcr
	);
}

static
void op4_fwd_sdl_epilog2_prolog2_10x1_s(
	const float *w,
	const float *v,
	float *lcr_y,
	float *lcr_x,
	int idx
)
{
#if 0
	float a[10];
	op4_fwd_sdl_epilog2_fast_s(
		a+0,
		a+1,
		a+2,
		a+3,
		a+4,
		a+5,
		a+6,
		a+7,
		a+8,
		a+9,
		w, v,
		lcr_y
	);
	for(int i = 0; i < 10; i++)
	{
		op4_fwd_sdl_prolog2_import_s(
			a+i,
			lcr_x + i*12,
			idx
		);
	}
#else
	// epilog2: export(3)
	*(lcr_x+9*12+0+idx) = *(lcr_y+0+3);

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		lcr_x+0*12+0+idx,
		lcr_x+1*12+0+idx,
		w, v,
		(lcr_y+0), (lcr_y+4), (lcr_y+8)
	);

	// epilog2: export(2)
	*(lcr_x+8*12+0+idx) = *(lcr_y+0+2);

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		lcr_x+2*12+0+idx,
		lcr_x+3*12+0+idx,
		w, v,
		(lcr_y+0), (lcr_y+4), (lcr_y+8)
	);

	// epilog2: export(1)
	*(lcr_x+7*12+0+idx) = *(lcr_y+0+1);

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_part_s(
		lcr_x+4*12+0+idx,
		lcr_x+5*12+0+idx,
		w, v,
		(lcr_y+0), (lcr_y+4), (lcr_y+8)
	);

	// epilog2: export(0)
	*(lcr_x+6*12+0+idx) = *(lcr_y+0+0);
#endif
}

static
void op4_fwd_sdl_epilog2_prolog2_2x2_s(
	const float *w,
	const float *v,
	float *y0_lcr,
	float *y1_lcr,
	float *x0_lcr,
	float *x1_lcr
)
{
#ifndef __SSE__
	float tmp[4];

	op4_fwd_sdl_epilog2_part_s(
		tmp+0,
		tmp+1,
		w, v,
		(y0_lcr+0), (y0_lcr+4), (y0_lcr+8)
	);
	op4_fwd_sdl_epilog2_part_s(
		tmp+2,
		tmp+3,
		w, v,
		(y1_lcr+0), (y1_lcr+4), (y1_lcr+8)
	);
	op4_fwd_sdl_prolog2_part_s(
		tmp+0,
		tmp+2,
		w, v,
		x0_lcr+0, x0_lcr+4, x0_lcr+8
	);
	op4_fwd_sdl_prolog2_part_s(
		tmp+1,
		tmp+3,
		w, v,
		x1_lcr+0, x1_lcr+4, x1_lcr+8
	);
#else
	__m128 tmp;
	__m128 z0, z1;

	// can be interleaved with single "z"
	op4s_sdl2_shuffle_s_sse(*(__m128 *)(y0_lcr+4), *(__m128 *)(y0_lcr+8));
	op4s_sdl2_shuffle_s_sse(*(__m128 *)(y1_lcr+4), *(__m128 *)(y1_lcr+8));
	op4s_sdl2_op_s_sse(z0, *(__m128 *)(y0_lcr+4), *(__m128 *)w, *(__m128 *)(y0_lcr+0), *(__m128 *)(y0_lcr+8));
	op4s_sdl2_op_s_sse(z1, *(__m128 *)(y1_lcr+4), *(__m128 *)w, *(__m128 *)(y1_lcr+0), *(__m128 *)(y1_lcr+8));
	op4s_sdl2_output_low_s_sse(tmp, *(__m128 *)(y0_lcr+0), z0);
	op4s_sdl2_output_high_s_sse(tmp, *(__m128 *)(y1_lcr+0), z1);
	op4s_sdl2_scale_s_sse(tmp, *(__m128 *)v);
	op4s_sdl2_update_s_sse(*(__m128 *)(y0_lcr+4), *(__m128 *)(y0_lcr+0), *(__m128 *)(y0_lcr+8), z0);
	op4s_sdl2_update_s_sse(*(__m128 *)(y1_lcr+4), *(__m128 *)(y1_lcr+0), *(__m128 *)(y1_lcr+8), z1);

	// swap
	tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(3,1,2,0));

	op4s_sdl2_shuffle_input_low_s_sse(tmp, *(__m128 *)(x0_lcr+4), *(__m128 *)(x0_lcr+8));
	op4s_sdl2_shuffle_input_high_s_sse(tmp, *(__m128 *)(x1_lcr+4), *(__m128 *)(x1_lcr+8));
	op4s_sdl2_op_s_sse(z0, *(__m128 *)(x0_lcr+4), *(__m128 *)w, *(__m128 *)(x0_lcr+0), *(__m128 *)(x0_lcr+8));
	op4s_sdl2_op_s_sse(z1, *(__m128 *)(x1_lcr+4), *(__m128 *)w, *(__m128 *)(x1_lcr+0), *(__m128 *)(x1_lcr+8));
	op4s_sdl2_update_s_sse(*(__m128 *)(x0_lcr+4), *(__m128 *)(x0_lcr+0), *(__m128 *)(x0_lcr+8), z0);
	op4s_sdl2_update_s_sse(*(__m128 *)(x1_lcr+4), *(__m128 *)(x1_lcr+0), *(__m128 *)(x1_lcr+8), z1);
#endif
}

static
void op4_fwd_sdl_epilog2_prolog2_10x2_s(
	float *w,
	float *v,
	float *y0_lcr,
	float *y1_lcr,
	float *x0_lcr
)
{
#if 0
	float a0[10], a1[10];
	op4_fwd_sdl_epilog2_fast_s(
		a0+0,
		a0+1,
		a0+2,
		a0+3,
		a0+4,
		a0+5,
		a0+6,
		a0+7,
		a0+8,
		a0+9,
		w, v,
		y0_lcr
	);
	op4_fwd_sdl_epilog2_fast_s(
		a1+0,
		a1+1,
		a1+2,
		a1+3,
		a1+4,
		a1+5,
		a1+6,
		a1+7,
		a1+8,
		a1+9,
		w, v,
		y1_lcr
	);
	for(int i = 0; i < 10; i++)
	{
		op4_fwd_sdl_prolog2_part_s(
			a0+i,
			a1+i,
			w, v,
			x0_lcr+i*12+0, x0_lcr+i*12+4, x0_lcr+i*12+8
		);
	}
#else
	// epilog2: export(3)
	op4_fwd_sdl_prolog2_part_s(
		(y0_lcr+0+3),
		(y1_lcr+0+3),
		w, v,
		x0_lcr+9*12+0, x0_lcr+9*12+4, x0_lcr+9*12+8
	);

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_prolog2_2x2_s(
		w, v,
		y0_lcr,
		y1_lcr,
		x0_lcr+0*12,
		x0_lcr+1*12
	);

	// epilog2: export(2)
	op4_fwd_sdl_prolog2_part_s(
		(y0_lcr+0+2),
		(y1_lcr+0+2),
		w, v,
		x0_lcr+8*12+0, x0_lcr+8*12+4, x0_lcr+8*12+8
	);

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_prolog2_2x2_s(
		w, v,
		y0_lcr,
		y1_lcr,
		x0_lcr+2*12,
		x0_lcr+3*12
	);

	// epilog2: export(1)
	op4_fwd_sdl_prolog2_part_s(
		(y0_lcr+0+1),
		(y1_lcr+0+1),
		w, v,
		x0_lcr+7*12+0, x0_lcr+7*12+4, x0_lcr+7*12+8
	);

	// epilog2: pass-epilog
	op4_fwd_sdl_epilog2_prolog2_2x2_s(
		w, v,
		y0_lcr,
		y1_lcr,
		x0_lcr+4*12,
		x0_lcr+5*12
	);

	// epilog2: export(0)
	op4_fwd_sdl_prolog2_part_s(
		(y0_lcr+0+0),
		(y1_lcr+0+0),
		w, v,
		x0_lcr+6*12+0, x0_lcr+6*12+4, x0_lcr+6*12+8
	);
#endif
}

// TODO: improve single-loop approach
void dwt_cdf97_2f_inplace_sdl_s(
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

	assert( 1 == dwt_util_get_num_workers() );

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	const int offset = 1;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

// 		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
// 		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
// 		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
// 		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		const int stride_y_j = stride_y * (1 << (j));
		const int stride_x_j = stride_x * (1 << (j));

		const int size_x = size_i_src_x;
		const int size_y = size_i_src_y;

		const int pairs_x = (to_even(size_x-offset)-4)/2;
		const int pairs_y = (to_even(size_y-offset)-4)/2;

// 		const int max_y = to_even(size_y-offset)+offset;

		if( size_x > 1 && size_x < 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y < 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		const int offset = 1;
//#define DISABLE_SLOW
//#define DISABLE_CORE
		if(
			size_x > 1 && size_x >= 5 && size_y > 1 && size_y >= 5 // single-loop block
			&& pairs_x > 3 && pairs_y > 3 // exception (steps<3)
		)
		{
			const float w[4] __attribute__ ((aligned (16))) = { dwt_cdf97_u2_s, -dwt_cdf97_p2_s, dwt_cdf97_u1_s, -dwt_cdf97_p1_s };
			const float v[4] __attribute__ ((aligned (16))) = { 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s, 1/dwt_cdf97_s1_s, dwt_cdf97_s1_s };

			// single
			const int prolog2_coeffs = 10;
// 			const int epilog2_coeffs = 10;

			int max_y = to_even(size_y - offset) + offset;

			float y0_lcr[12] __attribute__ ((aligned (16)));
// 			float *y0_l __attribute__ ((aligned (16))) = &y0_lcr[0];
// 			float *y0_c __attribute__ ((aligned (16))) = &y0_lcr[4];
// 			float *y0_r __attribute__ ((aligned (16))) = &y0_lcr[8];

			float y1_lcr[12] __attribute__ ((aligned (16)));
// 			float *y1_l __attribute__ ((aligned (16))) = &y1_lcr[0];
// 			float *y1_c __attribute__ ((aligned (16))) = &y1_lcr[4];
// 			float *y1_r __attribute__ ((aligned (16))) = &y1_lcr[8];

			float x0_lcr[4*3*size_x] __attribute__ ((aligned (16))); // FIXME: long array on the stack

			// horizontal full + vertical none (offset)
			for(int y = 0; y < 0+offset; y++)
			{
				float *ptr0_x = addr2_s(ptr, y, offset, stride_x_j, stride_y_j);

				op4_fwd_sdl_s(ptr0_x, stride_y_j, pairs_x, w, v); // FIXME: slow
			}
			// horizontal full + vertical prolog2 (10)
#if 1
			for(int y = 0+offset; y < 0+offset+prolog2_coeffs; y++)
			{
				float *ptr0_x = addr2_s(ptr, y+0, offset, stride_x_j, stride_y_j);
				op4_fwd_sdl_s(ptr0_x, stride_y_j, pairs_x, w, v);
			}
			for(int x = 0; x < size_x; x++)
			{
				float *ptr0_y = addr2_s(ptr, offset, x, stride_x_j, stride_y_j);
				op4_fwd_sdl_prolog2_s(
					addr1_s(ptr0_y, 0, stride_x_j),
					addr1_s(ptr0_y, 1, stride_x_j),
					addr1_s(ptr0_y, 2, stride_x_j),
					addr1_s(ptr0_y, 3, stride_x_j),
					addr1_s(ptr0_y, 4, stride_x_j),
					addr1_s(ptr0_y, 5, stride_x_j),
					addr1_s(ptr0_y, 6, stride_x_j),
					addr1_s(ptr0_y, 7, stride_x_j),
					addr1_s(ptr0_y, 8, stride_x_j),
					addr1_s(ptr0_y, 9, stride_x_j),
					w, v,
					&x0_lcr[12*x+0], &x0_lcr[12*x+4], &x0_lcr[12*x+8]
				);
			}
#else
			// part0
			{
				int row = 3;
				int idx = 3;

				int y0 = 0+offset+row;
				float *buff0_x0_lcr = x0_lcr;
				float *ptr0_x = addr2_s(ptr, y0+0, 0, stride_x_j, stride_y_j);

				for(int x = 0; x < offset; x++)
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
					buff0_x0_lcr += 1*12;
					ptr0_x = addr1_s(ptr0_x, +1, stride_y_j);
				}

				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);

				for(int s = 0; s < pairs_x-3; s++)
				{
					op4_fwd_sdl_core_prolog2_2x1_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						w, v,
						y0_lcr,
						buff0_x0_lcr,
						buff0_x0_lcr+12,
						idx
					);
					buff0_x0_lcr += 2*12;
					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_prolog2_10x1_s(
					w, v,
					y0_lcr,
					buff0_x0_lcr,
					idx
				);
				buff0_x0_lcr += 10*12;
				ptr0_x = addr1_s(ptr0_x, +10, stride_y_j);

				if( is_even(size_x) )
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
				}
			}
			// part1
			{
				int row0 = 4;
				int row1 = 5;

				int y0 = 0+offset+row0;
				int y1 = 0+offset+row1;
				float *buff0_x0_lcr = x0_lcr;
				float *ptr0_x = addr2_s(ptr, y0+0, 0, stride_x_j, stride_y_j);
				float *ptr1_x = addr2_s(ptr, y1+0, 0, stride_x_j, stride_y_j);

				for(int x = 0; x < offset; x++)
				{
					op4_fwd_sdl_prolog2_part_s(
						ptr0_x,
						ptr1_x,
						w, v,
						buff0_x0_lcr+0, buff0_x0_lcr+4, buff0_x0_lcr+8
					);
				}
				buff0_x0_lcr += offset*12;
				ptr0_x = addr1_s(ptr0_x, offset, stride_y_j);
				ptr1_x = addr1_s(ptr1_x, offset, stride_y_j);

				// FIXME: merge these into single function
				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);
				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr1_x, 0, stride_y_j),
					addr1_s(ptr1_x, 1, stride_y_j),
					addr1_s(ptr1_x, 2, stride_y_j),
					addr1_s(ptr1_x, 3, stride_y_j),
					addr1_s(ptr1_x, 4, stride_y_j),
					addr1_s(ptr1_x, 5, stride_y_j),
					addr1_s(ptr1_x, 6, stride_y_j),
					addr1_s(ptr1_x, 7, stride_y_j),
					addr1_s(ptr1_x, 8, stride_y_j),
					addr1_s(ptr1_x, 9, stride_y_j),
					w, v,
					y1_lcr
				);

				for(int s = 0; s < pairs_x-3; s++)
				{
					op4_fwd_sdl_core_prolog2_2x2_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						addr1_s(ptr1_x, 10, stride_y_j),
						addr1_s(ptr1_x, 11, stride_y_j),
						w, v,
						y0_lcr,
						y1_lcr,
						buff0_x0_lcr,
						buff0_x0_lcr+12
					);

					buff0_x0_lcr += 2*12;
					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
					ptr1_x = addr1_s(ptr1_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_prolog2_10x2_s(
					w, v,
					y0_lcr,
					y1_lcr,
					buff0_x0_lcr
				);
				buff0_x0_lcr += 10*12;
				ptr0_x = addr1_s(ptr0_x, +10, stride_y_j);
				ptr1_x = addr1_s(ptr1_x, +10, stride_y_j);

				if( is_even(size_x) )
				{
					op4_fwd_sdl_prolog2_part_s(
						ptr0_x,
						ptr1_x,
						w, v,
						buff0_x0_lcr+0, buff0_x0_lcr+4, buff0_x0_lcr+8
					);
				}
			}
			// part2
			{
				int row = 2;
				int idx = 2;

				int y0 = 0+offset+row;
				float *buff0_x0_lcr = x0_lcr;
				float *ptr0_x = addr2_s(ptr, y0+0, 0, stride_x_j, stride_y_j);

				for(int x = 0; x < offset; x++)
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
					buff0_x0_lcr += 1*12;
					ptr0_x = addr1_s(ptr0_x, +1, stride_y_j);
				}

				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);

				for(int s = 0; s < pairs_x-3; s++)
				{
					op4_fwd_sdl_core_prolog2_2x1_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						w, v,
						y0_lcr,
						buff0_x0_lcr,
						buff0_x0_lcr+12,
						idx
					);
					buff0_x0_lcr += 2*12;
					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_prolog2_10x1_s(
					w, v,
					y0_lcr,
					buff0_x0_lcr,
					idx
				);
				buff0_x0_lcr += 10*12;
				ptr0_x = addr1_s(ptr0_x, +10, stride_y_j);

				if( is_even(size_x) )
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
				}
			}
			// part3
			{
				int row0 = 6;
				int row1 = 7;

				int y0 = 0+offset+row0;
				int y1 = 0+offset+row1;
				float *buff0_x0_lcr = x0_lcr;
				float *ptr0_x = addr2_s(ptr, y0+0, 0, stride_x_j, stride_y_j);
				float *ptr1_x = addr2_s(ptr, y1+0, 0, stride_x_j, stride_y_j);

				for(int x = 0; x < offset; x++)
				{
					op4_fwd_sdl_prolog2_part_s(
						ptr0_x,
						ptr1_x,
						w, v,
						buff0_x0_lcr+0, buff0_x0_lcr+4, buff0_x0_lcr+8
					);
				}
				buff0_x0_lcr += offset*12;
				ptr0_x = addr1_s(ptr0_x, offset, stride_y_j);
				ptr1_x = addr1_s(ptr1_x, offset, stride_y_j);

				// FIXME: merge these into single function
				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);
				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr1_x, 0, stride_y_j),
					addr1_s(ptr1_x, 1, stride_y_j),
					addr1_s(ptr1_x, 2, stride_y_j),
					addr1_s(ptr1_x, 3, stride_y_j),
					addr1_s(ptr1_x, 4, stride_y_j),
					addr1_s(ptr1_x, 5, stride_y_j),
					addr1_s(ptr1_x, 6, stride_y_j),
					addr1_s(ptr1_x, 7, stride_y_j),
					addr1_s(ptr1_x, 8, stride_y_j),
					addr1_s(ptr1_x, 9, stride_y_j),
					w, v,
					y1_lcr
				);

				for(int s = 0; s < pairs_x-3; s++)
				{
					op4_fwd_sdl_core_prolog2_2x2_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						addr1_s(ptr1_x, 10, stride_y_j),
						addr1_s(ptr1_x, 11, stride_y_j),
						w, v,
						y0_lcr,
						y1_lcr,
						buff0_x0_lcr,
						buff0_x0_lcr+12
					);

					buff0_x0_lcr += 2*12;
					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
					ptr1_x = addr1_s(ptr1_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_prolog2_10x2_s(
					w, v,
					y0_lcr,
					y1_lcr,
					buff0_x0_lcr
				);
				buff0_x0_lcr += 10*12;
				ptr0_x = addr1_s(ptr0_x, +10, stride_y_j);
				ptr1_x = addr1_s(ptr1_x, +10, stride_y_j);

				if( is_even(size_x) )
				{
					op4_fwd_sdl_prolog2_part_s(
						ptr0_x,
						ptr1_x,
						w, v,
						buff0_x0_lcr+0, buff0_x0_lcr+4, buff0_x0_lcr+8
					);
				}
			}
			// part4
			{
				int row = 1;
				int idx = 1;

				int y0 = 0+offset+row;
				float *buff0_x0_lcr = x0_lcr;
				float *ptr0_x = addr2_s(ptr, y0+0, 0, stride_x_j, stride_y_j);

				for(int x = 0; x < offset; x++)
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
					buff0_x0_lcr += 1*12;
					ptr0_x = addr1_s(ptr0_x, +1, stride_y_j);
				}

				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);

				for(int s = 0; s < pairs_x-3; s++)
				{
					op4_fwd_sdl_core_prolog2_2x1_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						w, v,
						y0_lcr,
						buff0_x0_lcr,
						buff0_x0_lcr+12,
						idx
					);
					buff0_x0_lcr += 2*12;
					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_prolog2_10x1_s(
					w, v,
					y0_lcr,
					buff0_x0_lcr,
					idx
				);
				buff0_x0_lcr += 10*12;
				ptr0_x = addr1_s(ptr0_x, +10, stride_y_j);

				if( is_even(size_x) )
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
				}
			}
			// part5
			{
				int row0 = 8;
				int row1 = 9;

				int y0 = 0+offset+row0;
				int y1 = 0+offset+row1;
				float *buff0_x0_lcr = x0_lcr;
				float *ptr0_x = addr2_s(ptr, y0+0, 0, stride_x_j, stride_y_j);
				float *ptr1_x = addr2_s(ptr, y1+0, 0, stride_x_j, stride_y_j);

				for(int x = 0; x < offset; x++)
				{
					op4_fwd_sdl_prolog2_part_s(
						ptr0_x,
						ptr1_x,
						w, v,
						buff0_x0_lcr+0, buff0_x0_lcr+4, buff0_x0_lcr+8
					);
				}
				buff0_x0_lcr += offset*12;
				ptr0_x = addr1_s(ptr0_x, offset, stride_y_j);
				ptr1_x = addr1_s(ptr1_x, offset, stride_y_j);

				// FIXME: merge these into single function
				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);
				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr1_x, 0, stride_y_j),
					addr1_s(ptr1_x, 1, stride_y_j),
					addr1_s(ptr1_x, 2, stride_y_j),
					addr1_s(ptr1_x, 3, stride_y_j),
					addr1_s(ptr1_x, 4, stride_y_j),
					addr1_s(ptr1_x, 5, stride_y_j),
					addr1_s(ptr1_x, 6, stride_y_j),
					addr1_s(ptr1_x, 7, stride_y_j),
					addr1_s(ptr1_x, 8, stride_y_j),
					addr1_s(ptr1_x, 9, stride_y_j),
					w, v,
					y1_lcr
				);

				for(int s = 0; s < pairs_x-3; s++)
				{
					op4_fwd_sdl_core_prolog2_2x2_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						addr1_s(ptr1_x, 10, stride_y_j),
						addr1_s(ptr1_x, 11, stride_y_j),
						w, v,
						y0_lcr,
						y1_lcr,
						buff0_x0_lcr,
						buff0_x0_lcr+12
					);

					buff0_x0_lcr += 2*12;
					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
					ptr1_x = addr1_s(ptr1_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_prolog2_10x2_s(
					w, v,
					y0_lcr,
					y1_lcr,
					buff0_x0_lcr
				);
				buff0_x0_lcr += 10*12;
				ptr0_x = addr1_s(ptr0_x, +10, stride_y_j);
				ptr1_x = addr1_s(ptr1_x, +10, stride_y_j);

				if( is_even(size_x) )
				{
					op4_fwd_sdl_prolog2_part_s(
						ptr0_x,
						ptr1_x,
						w, v,
						buff0_x0_lcr+0, buff0_x0_lcr+4, buff0_x0_lcr+8
					);
				}
			}
			// part6
			{
				int row = 0;
				int idx = 0;

				int y0 = 0+offset+row;
				float *buff0_x0_lcr = x0_lcr;
				float *ptr0_x = addr2_s(ptr, y0+0, 0, stride_x_j, stride_y_j);

				for(int x = 0; x < offset; x++)
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
					buff0_x0_lcr += 1*12;
					ptr0_x = addr1_s(ptr0_x, +1, stride_y_j);
				}

				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);

				for(int s = 0; s < pairs_x-3; s++)
				{
					op4_fwd_sdl_core_prolog2_2x1_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						w, v,
						y0_lcr,
						buff0_x0_lcr,
						buff0_x0_lcr+12,
						idx
					);
					buff0_x0_lcr += 2*12;
					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_prolog2_10x1_s(
					w, v,
					y0_lcr,
					buff0_x0_lcr,
					idx
				);
				buff0_x0_lcr += 10*12;
				ptr0_x = addr1_s(ptr0_x, +10, stride_y_j);

				if( is_even(size_x) )
				{
					op4_fwd_sdl_prolog2_import_s(
						ptr0_x,
						buff0_x0_lcr,
						idx
					);
				}
			}
#endif

			// horizontal full + vertical core (*)
			for(int y = 0+offset+prolog2_coeffs; y < max_y; y+=2)
			{
				float *ptr0_x = addr2_s(ptr, y+0, offset, stride_x_j, stride_y_j);
				float *ptr1_x = addr2_s(ptr, y+1, offset, stride_x_j, stride_y_j);
				float *out0_x = addr2_s(ptr, y+0-10, offset, stride_x_j, stride_y_j);
				float *out1_x = addr2_s(ptr, y+1-10, offset, stride_x_j, stride_y_j);

				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);
				op4_fwd_sdl_prolog2_fast_s(
					addr1_s(ptr1_x, 0, stride_y_j),
					addr1_s(ptr1_x, 1, stride_y_j),
					addr1_s(ptr1_x, 2, stride_y_j),
					addr1_s(ptr1_x, 3, stride_y_j),
					addr1_s(ptr1_x, 4, stride_y_j),
					addr1_s(ptr1_x, 5, stride_y_j),
					addr1_s(ptr1_x, 6, stride_y_j),
					addr1_s(ptr1_x, 7, stride_y_j),
					addr1_s(ptr1_x, 8, stride_y_j),
					addr1_s(ptr1_x, 9, stride_y_j),
					w, v,
					y1_lcr
				);

				float *buff_x0_lcr = x0_lcr+12*(offset+0);

				for(int s = 0; s < pairs_x-3; s++)
				{
#ifndef DISABLE_CORE
					op4_fwd_sdl_2x2_s(
						addr1_s(ptr0_x, 10, stride_y_j),
						addr1_s(ptr0_x, 11, stride_y_j),
						addr1_s(ptr1_x, 10, stride_y_j),
						addr1_s(ptr1_x, 11, stride_y_j),
						addr1_s(out0_x,  0, stride_y_j),
						addr1_s(out0_x,  1, stride_y_j),
						addr1_s(out1_x,  0, stride_y_j),
						addr1_s(out1_x,  1, stride_y_j),
						w,
						v,
						y0_lcr,
						y1_lcr,
						buff_x0_lcr+0,
						buff_x0_lcr+12
					);
#endif

					buff_x0_lcr += 2*12;

					ptr0_x = addr1_s(ptr0_x, +2, stride_y_j);
					ptr1_x = addr1_s(ptr1_x, +2, stride_y_j);
					out0_x = addr1_s(out0_x, +2, stride_y_j);
					out1_x = addr1_s(out1_x, +2, stride_y_j);
				}

				op4_fwd_sdl_epilog2_fast_s(
					addr1_s(ptr0_x, 0, stride_y_j),
					addr1_s(ptr0_x, 1, stride_y_j),
					addr1_s(ptr0_x, 2, stride_y_j),
					addr1_s(ptr0_x, 3, stride_y_j),
					addr1_s(ptr0_x, 4, stride_y_j),
					addr1_s(ptr0_x, 5, stride_y_j),
					addr1_s(ptr0_x, 6, stride_y_j),
					addr1_s(ptr0_x, 7, stride_y_j),
					addr1_s(ptr0_x, 8, stride_y_j),
					addr1_s(ptr0_x, 9, stride_y_j),
					w, v,
					y0_lcr
				);
				op4_fwd_sdl_epilog2_fast_s(
					addr1_s(ptr1_x, 0, stride_y_j),
					addr1_s(ptr1_x, 1, stride_y_j),
					addr1_s(ptr1_x, 2, stride_y_j),
					addr1_s(ptr1_x, 3, stride_y_j),
					addr1_s(ptr1_x, 4, stride_y_j),
					addr1_s(ptr1_x, 5, stride_y_j),
					addr1_s(ptr1_x, 6, stride_y_j),
					addr1_s(ptr1_x, 7, stride_y_j),
					addr1_s(ptr1_x, 8, stride_y_j),
					addr1_s(ptr1_x, 9, stride_y_j),
					w, v,
					y1_lcr
				);

				// right border
				for(int i = 0; i < 10+is_even(size_x); i++)
				{
					op4_fwd_sdl_core_s(
						addr1_s(ptr0_x,  i, stride_y_j), // in
						addr1_s(ptr1_x,  i, stride_y_j), // in
						addr1_s(out0_x,  i, stride_y_j), // out
						addr1_s(out1_x,  i, stride_y_j), // out
						w, v,
						buff_x0_lcr+0, buff_x0_lcr+4, buff_x0_lcr+8
					);
					buff_x0_lcr += 1*12;
				}
			}
			// horizontal full + vertical core (last one)
			for(int y = max_y; y < size_y; y++)
			{
				float *ptr0_x = addr2_s(ptr, y, offset, stride_x_j, stride_y_j);

				op4_fwd_sdl_s(ptr0_x, stride_y_j, pairs_x, w, v); // FIXME: slow
			}
			// the left most column (only if offset == 1)
			for(int s = 0; s < pairs_y-3; s++)
			{
				for(int x = 0; x < offset; x++)
				{
					float *ptr0_y = addr2_s(ptr, offset, x, stride_x_j, stride_y_j);
					ptr0_y = addr1_s(ptr0_y, 2*s, stride_x_j);

					op4_fwd_sdl_core_s(
						addr1_s(ptr0_y, 10, stride_x_j),
						addr1_s(ptr0_y, 11, stride_x_j),
						addr1_s(ptr0_y,  0, stride_x_j),
						addr1_s(ptr0_y,  1, stride_x_j),
						w, v,
						&x0_lcr[12*x+0], &x0_lcr[12*x+4], &x0_lcr[12*x+8]
					);
				}
			}
			// horizontal none + vertical epilog2
			for(int x = 0; x < size_x; x++)
			{
				float *ptr0_y = addr2_s(ptr, offset, x, stride_x_j, stride_y_j);
				ptr0_y = addr1_s(ptr0_y, (pairs_y-3)*2, stride_x_j);
#ifndef DISABLE_SLOW
				op4_fwd_sdl_epilog2_fast_s(
					addr1_s(ptr0_y, 0, stride_x_j),
					addr1_s(ptr0_y, 1, stride_x_j),
					addr1_s(ptr0_y, 2, stride_x_j),
					addr1_s(ptr0_y, 3, stride_x_j),
					addr1_s(ptr0_y, 4, stride_x_j),
					addr1_s(ptr0_y, 5, stride_x_j),
					addr1_s(ptr0_y, 6, stride_x_j),
					addr1_s(ptr0_y, 7, stride_x_j),
					addr1_s(ptr0_y, 8, stride_x_j),
					addr1_s(ptr0_y, 9, stride_x_j),
					w, v,
					&x0_lcr[12*x]
				);
#endif
			}
		}
		else
		{
			// double
			if( size_x > 1 && size_x >= 5 )
			{
				for(int y = 0; y < size_y; y++)
				{
					dwt_cdf97_f_ex_stride_inplace_part_core_sdl_s(
						addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
						size_x, // N
						stride_y_j);
				}
			}
			if( size_y > 1 && size_y >= 5 )
			{
				for(int x = 0; x < size_x; x++)
				{
					dwt_cdf97_f_ex_stride_inplace_part_core_sdl_s(
						addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
						size_y, // N
						stride_x_j);
				}
			}
		}

		if( size_x > 1 && size_x >= 5 )
		{
			for(int y = 0; y < size_y; y++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					size_x, // N
					stride_y_j);
			}
		}
		if( size_y > 1 && size_y >= 5 )
		{
			for(int x = 0; x < size_x; x++)
			{
				dwt_cdf97_f_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					size_y, // N
					stride_x_j);
			}
		}

		j++;
	}

	FUNC_END;
}

void dwt_cdf97_1i_s(
	void *ptr,
	int stride_y,
	int size_o_big_x,
	int size_i_big_x,
	int j_max,
	int zero_padding
)
{
	FUNC_BEGIN;

	assert( 1 == dwt_util_get_num_workers() );

	const int threads = 1;

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SB);
#endif

	const int offset = 0;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_x, offset)
	);

	int j = ceil_log2( size_o_big_x );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);

		const int lines_x = size_o_dst_x;

		if( lines_x > 1 )
		{
				dwt_cdf97_i_ex_stride_s(
					addr1_const_s(ptr,0,stride_y),
					addr1_const_s(ptr,size_o_src_x,stride_y),
					addr1_s(ptr,0,stride_y),
					temp[0],
					size_i_dst_x,
					stride_y);
		}

		if(zero_padding)
		{
				dwt_zero_padding_i_stride_s(
					addr1_s(ptr,0,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
		}

		j--;
	}

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_cdf53_1i_s(
	void *ptr,
	int stride_y,
	int size_o_big_x,
	int size_i_big_x,
	int j_max,
	int zero_padding
)
{
	FUNC_BEGIN;

	assert( 1 == dwt_util_get_num_workers() );

	const int threads = 1;

	const int offset = 0;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_x, offset)
	);

	int j = ceil_log2( size_o_big_x );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);

		const int lines_x = size_o_dst_x;

		if( lines_x > 1 )
		{
				dwt_cdf53_i_ex_stride_s(
					addr1_const_s(ptr,0,stride_y),
					addr1_const_s(ptr,size_o_src_x,stride_y),
					addr1_s(ptr,0,stride_y),
					temp[0],
					size_i_dst_x,
					stride_y);
		}

		if(zero_padding)
		{
				dwt_zero_padding_i_stride_s(
					addr1_s(ptr,0,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
		}

		j--;
	}

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_interp53_1i_s(
	void *ptr,
	int stride_y,
	int size_o_big_x,
	int size_i_big_x,
	int j_max,
	int zero_padding
)
{
	FUNC_BEGIN;

	assert( 1 == dwt_util_get_num_workers() );

	const int threads = 1;

	const int offset = 0;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_x, offset)
	);

	int j = ceil_log2( size_o_big_x );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);

		const int lines_x = size_o_dst_x;

		if( lines_x > 1 )
		{
				dwt_interp53_i_ex_stride_s(
					addr1_const_s(ptr,0,stride_y),
					addr1_const_s(ptr,size_o_src_x,stride_y),
					addr1_s(ptr,0,stride_y),
					temp[0],
					size_i_dst_x,
					stride_y);
		}

		if(zero_padding)
		{
				dwt_zero_padding_i_stride_s(
					addr1_s(ptr,0,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
		}

		j--;
	}

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_cdf97_2f1_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int zero_padding
)
{
	UNUSED(size_o_big_y);

	for(int y = 0; y < size_i_big_y; y++)
	{
		int x = 0;
		void *y_ptr = addr2_s(ptr, y, x, stride_x, stride_y); // stride?

		dwt_cdf97_1f_s(
			y_ptr,
			stride_y, // stride?
			size_o_big_x,
			size_i_big_x,
			j_max_ptr,
			zero_padding
		);
	}
}

void dwt_cdf53_2f1_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int zero_padding
)
{
	UNUSED(size_o_big_y);

	for(int y = 0; y < size_i_big_y; y++)
	{
		int x = 0;
		void *y_ptr = addr2_s(ptr, y, x, stride_x, stride_y);

		dwt_cdf53_1f_s(
			y_ptr,
			stride_y,
			size_o_big_x,
			size_i_big_x,
			j_max_ptr,
			zero_padding
		);
	}
}

void dwt_cdf97_1f_s(
	void *ptr,
	int stride_y,
	int size_o_big_x,
	int size_i_big_x,
	int *j_max_ptr,
	int zero_padding
)
{
	FUNC_BEGIN;

	assert( 1 == dwt_util_get_num_workers() );

	const int threads = 1;

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SA);
#endif
	const int offset = 1;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_x, offset)
	);

	int j = 0;

	const int j_limit = ceil_log2( size_o_big_x );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );

		const int lines_x = size_o_src_x;

		if( lines_x > 1 )
		{
			dwt_cdf97_f_ex_stride_s(
				addr1_const_s(ptr,0,stride_y),
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				temp[0],
				size_i_src_x,
				stride_y);
		}

		if(zero_padding)
		{
			dwt_zero_padding_f_stride_s(
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				size_i_src_x,
				size_o_dst_x,
				size_o_src_x-size_o_dst_x,
				stride_y);
		}

		j++;
	}

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_cdf53_1f_s(
	void *ptr,
	int stride_y,
	int size_o_big_x,
	int size_i_big_x,
	int *j_max_ptr,
	int zero_padding
)
{
	FUNC_BEGIN;

	assert( 1 == dwt_util_get_num_workers() );

	const int threads = 1;

	const int offset = 1;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_x, offset)
	);

	int j = 0;

	const int j_limit = ceil_log2( size_o_big_x );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );

		const int lines_x = size_o_src_x;

		if( lines_x > 1 )
		{
			dwt_cdf53_f_ex_stride_s(
				addr1_const_s(ptr,0,stride_y),
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				temp[0],
				size_i_src_x,
				stride_y);
		}

		if(zero_padding)
		{
			dwt_zero_padding_f_stride_s(
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				size_i_src_x,
				size_o_dst_x,
				size_o_src_x-size_o_dst_x,
				stride_y);
		}

		j++;
	}

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_interp53_1f_s(
	void *ptr,
	int stride_y,
	int size_o_big_x,
	int size_i_big_x,
	int *j_max_ptr,
	int zero_padding
)
{
	FUNC_BEGIN;

	assert( 1 == dwt_util_get_num_workers() );

	const int threads = 1;

	const int offset = 1;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_x, offset)
	);

	int j = 0;

	const int j_limit = ceil_log2( size_o_big_x );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );

		const int lines_x = size_o_src_x;

		if( lines_x > 1 )
		{
			dwt_interp53_f_ex_stride_s(
				addr1_const_s(ptr,0,stride_y),
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				temp[0],
				size_i_src_x,
				stride_y);
		}

		if(zero_padding)
		{
			dwt_zero_padding_f_stride_s(
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				size_i_src_x,
				size_o_dst_x,
				size_o_src_x-size_o_dst_x,
				stride_y);
		}

		j++;
	}

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_interp2_1f_s(
	void *ptr,
	int stride_y,
	int size_o_big_x,
	int size_i_big_x,
	int *j_max_ptr,
	int zero_padding
)
{
	FUNC_BEGIN;

	assert( 1 == dwt_util_get_num_workers() );

	const int threads = 1;

	const int offset = 1;

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_x, offset)
	);

	int j = 0;

	const int j_limit = ceil_log2( size_o_big_x );

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j+1);
		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );

		const int lines_x = size_o_src_x;

		if( lines_x > 1 )
		{
			dwt_interp2_f_ex_stride_s(
				addr1_const_s(ptr,0,stride_y),
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				temp[0],
				size_i_src_x,
				stride_y);
		}

		if(zero_padding)
		{
			dwt_zero_padding_f_stride_s(
				addr1_s(ptr,0,stride_y),
				addr1_s(ptr,size_o_dst_x,stride_y),
				size_i_src_x,
				size_o_dst_x,
				size_o_src_x-size_o_dst_x,
				stride_y);
		}

		j++;
	}

	free_temp_s(threads, temp);

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

void dwt_cdf53_2f_inplace_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding
)
{
	const int size_o_big_min = min(size_o_big_x, size_o_big_y);
	const int size_o_big_max = max(size_o_big_x, size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		const int stride_y_j = stride_y * (1 << (j));
		const int stride_x_j = stride_x * (1 << (j));

		for(int y = 0; y < size_i_src_y; y++)
			dwt_cdf53_f_ex_stride_inplace_s(
				addr2_s(ptr,y,0,stride_x_j,stride_y_j),
				size_i_src_x,
				stride_y_j);
		for(int x = 0; x < size_i_src_x; x++)
			dwt_cdf53_f_ex_stride_inplace_s(
				addr2_s(ptr,0,x,stride_x_j,stride_y_j),
				size_i_src_y,
				stride_x_j);

		j++;
	}
}

void dwt_eaw53_2f_inplace_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding,
	float *wH[],
	float *wV[],
	float alpha
)
{
	const int size_o_big_min = min(size_o_big_x, size_o_big_y);
	const int size_o_big_max = max(size_o_big_x, size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_i_src_x = ceil_div_pow2(size_i_big_x, j  );
		const int size_i_src_y = ceil_div_pow2(size_i_big_y, j  );

		const int stride_y_j = stride_y * (1 << (j));
		const int stride_x_j = stride_x * (1 << (j));

		wH[j] = dwt_util_alloc(size_i_src_y * size_i_src_x, sizeof(float));
		wV[j] = dwt_util_alloc(size_i_src_x * size_i_src_y, sizeof(float));

		for(int y = 0; y < size_i_src_y; y++)
			dwt_eaw53_f_ex_stride_inplace_s(
				addr2_s(ptr,y,0,stride_x_j,stride_y_j),
				size_i_src_x,
				stride_y_j,
				&wH[j][y*size_i_src_x],
       				alpha
			);
		for(int x = 0; x < size_i_src_x; x++)
			dwt_eaw53_f_ex_stride_inplace_s(
				addr2_s(ptr,0,x,stride_x_j,stride_y_j),
				size_i_src_y,
				stride_x_j,
				&wV[j][x*size_i_src_y],
       				alpha
			);

		j++;
	}
}

void dwt_eaw53_2f_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding,
	float *wH[],
	float *wV[],
	float alpha
)
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

		wH[j] = dwt_util_alloc(size_o_src_y * size_i_src_x, sizeof(float));
		wV[j] = dwt_util_alloc(size_o_src_x * size_i_src_y, sizeof(float));

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_src_y; y++)
			dwt_eaw53_f_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x, // N
				stride_y,
				&wH[j][y*size_i_src_x],
				alpha
			);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_eaw53_f_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_dst_y,x,stride_x,stride_y),
				temp,
				size_i_src_y, // N
				stride_x,
				&wV[j][x*size_i_src_y],
				alpha
			);

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

void dwt_eaw53_2f_dummy_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;
}

void dwt_cdf53_2f_dummy_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one
)
{
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	const int j_limit = ceil_log2(decompose_one?size_o_big_max:size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;
}

void dwt_interp53_2f_s(
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
			dwt_interp53_f_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_dst_x,stride_x,stride_y),
				temp,
				size_i_src_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_src_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_src_x; x++)
			dwt_interp53_f_ex_stride_s(
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

	// FIXME(microblaze): align on 8 bytes boundary (GCC's __attribure__ is ignored)
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

	// FIXME(microblaze): align on 8 bytes boundary (GCC's __attribure__ is ignored)
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

	const int threads = dwt_util_get_num_threads();
	const int workers = dwt_util_get_num_workers();

	const int offset = 0;

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SB);
#endif
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_max, offset)
	);

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

			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < workers_lines_y; y += workers)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(ptr,y,0,stride_x,stride_y),
					addr2_const_s(ptr,y,size_o_src_x,stride_x,stride_y),
					addr2_s(ptr,y,0,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_dst_x,
					stride_y);
			}
			dwt_util_set_num_workers(1);
			for(int y = workers_lines_y; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(ptr,y,0,stride_x,stride_y),
					addr2_const_s(ptr,y,size_o_src_x,stride_x,stride_y),
					addr2_s(ptr,y,0,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_dst_x,
					stride_y);
			}
			dwt_util_set_num_workers(workers);
		}

		if( lines_y > 1 )
		{
			set_data_step_s( stride_y );

			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < workers_lines_x; x += workers)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(ptr,0,x,stride_x,stride_y),
					addr2_const_s(ptr,size_o_src_y,x,stride_x,stride_y),
					addr2_s(ptr,0,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_dst_y,
					stride_x);
			}
			dwt_util_set_num_workers(1);
			for(int x = workers_lines_x; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(ptr,0,x,stride_x,stride_y),
					addr2_const_s(ptr,size_o_src_y,x,stride_x,stride_y),
					addr2_s(ptr,0,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
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

	free_temp_s(threads, temp);

	FUNC_END;
}

void dwt_cdf97_1i_inplace_s(
	void *ptr,
	int stride,
	int size,
	int j_max
)
{
	int j = ceil_log2( size );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if( 0 == j )
			break;

		const int size_j = ceil_div_pow2(size, j-1);

		const int stride_j = stride * (1 << (j-1));

		if( size_j > 1 && size_j < 4 )
		{
			dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
				ptr,
				size_j,
				stride_j
			);
		}

		if( size_j >= 4 )
		{
			dwt_cdf97_i_ex_stride_inplace_part_prolog_s(
				ptr,
				size_j,
				stride_j
			);

			dwt_cdf97_i_ex_stride_inplace_part_core_s(
				ptr,
				size_j,
				stride_j
			);

			dwt_cdf97_i_ex_stride_inplace_part_epilog_s(
				ptr,
				size_j,
				stride_j
			);
		}

		j--;
	}
}

void dwt_cdf97_i_ex_stride_inplace_i(
	int *tmp,
	int N,
	int stride
)
{
	assert( N >= 0 && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
		return;
#if 0
	// backward update 2 + backward predict 2
	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( 1817*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;

	*addr1_i(tmp, 0, stride) -= ( 1817*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) -= ( 1817*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) += ( -113*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) - (1<<6) ) >> 7;

	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( -113*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) - (1<<6) ) >> 7;

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( -217*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;

	*addr1_i(tmp, 0, stride) -= ( -217*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) -= ( -217*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) += ( +203*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) - (1<<6) ) >> 7;

	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( +203*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) - (1<<6) ) >> 7;
#else
	// backward update 2 + backward predict 2
	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( 1817*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;

	*addr1_i(tmp, 0, stride) -= ( 1817*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) -= ( 1817*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) -= ( +113*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<6) ) >> 7;

	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( +113*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<6) ) >> 7;

	// backward update 1 + backward predict 1
	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( -217*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;

	*addr1_i(tmp, 0, stride) -= ( -217*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) -= ( -217*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) -= ( -203*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<6) ) >> 7;

	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( -203*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<6) ) >> 7;
#endif
}

// TODO: tested only with j=1
void dwt_cdf97_2i_inplace_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding
)
{
	const int size_o_big_min = min(size_o_big_x, size_o_big_y);
	const int size_o_big_max = max(size_o_big_x, size_o_big_y);

	int j = ceil_log2(decompose_one ? size_o_big_max : size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if( 0 == j )
			break;

		const int size_x_j = ceil_div_pow2(size_i_big_x, j-1);
		const int size_y_j = ceil_div_pow2(size_i_big_y, j-1);

		#pragma omp parallel for schedule(static, ceil_div(size_x_j, omp_get_num_threads()))
		for(int x = 0; x < size_x_j; x++)
			dwt_cdf97_i_ex_stride_inplace_i(
				addr2_i(ptr, 0, x, stride_x, stride_y),
				size_y_j,
				stride_x
			);
		#pragma omp parallel for schedule(static, ceil_div(size_y_j, omp_get_num_threads()))
		for(int y = 0; y < size_y_j; y++)
			dwt_cdf97_i_ex_stride_inplace_i(
				addr2_i(ptr, y, 0, stride_x, stride_y),
				size_x_j,
				stride_y
			);

		j--;
	}
}

void dwt_cdf97_f_ex_stride_inplace_i(
	int *tmp,
	int N,
	int stride
)
{
	assert( N >= 0 && NULL != tmp && 0 != stride );

	// fix for small N
	if(N < 2)
		return;

#if 0
	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( +203*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) - (1<<6) ) >> 7;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) += ( -217*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) -= ( +203*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) - (1<<6) ) >> 7;
	*addr1_i(tmp, 0, stride) += ( -217*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( -217*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;

	// predict 2 + update 2
	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) -= ( -113*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) - (1<<6) ) >> 7;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) += ( 1817*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) -= ( -113*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) - (1<<6) ) >> 7;
	*addr1_i(tmp, 0, stride) += ( 1817*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( 1817*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;
#else
	// predict 1 + update 1
	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( -203*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<6) ) >> 7;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) += ( -217*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) += ( -203*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<6) ) >> 7;
	*addr1_i(tmp, 0, stride) += ( -217*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( -217*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;

	// predict 2 + update 2
	for(int i=1; i<N-2+(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( +113*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<6) ) >> 7;

	if(is_odd(N))
		*addr1_i(tmp, N-1, stride) += ( 1817*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<11) ) >> 12;
	else
		*addr1_i(tmp, N-1, stride) += ( +113*(*addr1_i(tmp, N-2, stride)+*addr1_i(tmp, N-2, stride)) + (1<<6) ) >> 7;
	*addr1_i(tmp, 0, stride) += ( 1817*(*addr1_i(tmp, 1, stride)+*addr1_i(tmp, 1, stride)) + (1<<11) ) >> 12;

	for(int i=2; i<N-(N&1); i+=2)
		*addr1_i(tmp, i, stride) += ( 1817*(*addr1_i(tmp, i-1, stride)+*addr1_i(tmp, i+1, stride)) + (1<<11) ) >> 12;
#endif
}

// TODO: tested only with j=1
void dwt_cdf97_2f_inplace_i(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int *j_max_ptr,
	int decompose_one,
	int zero_padding
)
{
	const int size_o_big_min = min(size_o_big_x, size_o_big_y);
	const int size_o_big_max = max(size_o_big_x, size_o_big_y);

	int j = 0;

	const int j_limit = ceil_log2(decompose_one ? size_o_big_max : size_o_big_min);

	if( *j_max_ptr < 0 || *j_max_ptr > j_limit )
		*j_max_ptr = j_limit;

	for(;;)
	{
		if( *j_max_ptr == j )
			break;

		const int size_x_j = ceil_div_pow2(size_i_big_x, j  );
		const int size_y_j = ceil_div_pow2(size_i_big_y, j  );

		#pragma omp parallel for schedule(static, ceil_div(size_y_j, omp_get_num_threads()))
		for(int y = 0; y < size_y_j; y++)
			dwt_cdf97_f_ex_stride_inplace_i(
				addr2_i(ptr, y, 0, stride_x, stride_y),
				size_x_j,
				stride_y
			);
		#pragma omp parallel for schedule(static, ceil_div(size_x_j, omp_get_num_threads()))
		for(int x = 0; x < size_x_j; x++)
			dwt_cdf97_f_ex_stride_inplace_i(
				addr2_i(ptr, 0, x, stride_x, stride_y),
				size_y_j,
				stride_x
			);

		j++;
	}
}

void dwt_cdf97_2i_inplace_s(
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

	assert( 1 == dwt_util_get_num_workers() );

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

// 		const int size_o_src_x = ceil_div_pow2(size_o_big_x, j  );
// 		const int size_o_src_y = ceil_div_pow2(size_o_big_y, j  );
// 		const int size_o_dst_x = ceil_div_pow2(size_o_big_x, j-1);
// 		const int size_o_dst_y = ceil_div_pow2(size_o_big_y, j-1);
		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		const int lines_y = size_i_dst_y;
		const int lines_x = size_i_dst_x;

		const int stride_y_j = stride_y * (1 << (j-1));
		const int stride_x_j = stride_x * (1 << (j-1));

		if( lines_x > 1 && lines_x < 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x, // N
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y < 4 )
		{
			for(int x = 0; x < size_i_dst_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y, // N
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_prolog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_core_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_core_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_epilog_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

// 		if(zero_padding)
// 		{
// 			#pragma omp parallel for schedule(static, threads_segment_y)
// 			for(int y = 0; y < size_o_dst_y; y++)
// 				dwt_zero_padding_i_stride_s(
// 					addr2_s(ptr,y,0,stride_x,stride_y),
// 					size_i_dst_x,
// 					size_o_dst_x,
// 					stride_y);
// 			#pragma omp parallel for schedule(static, threads_segment_x)
// 			for(int x = 0; x < size_o_dst_x; x++)
// 				dwt_zero_padding_i_stride_s(
// 					addr2_s(ptr,0,x,stride_x,stride_y),
// 					size_i_dst_y,
// 					size_o_dst_y,
// 					stride_x);
// 		}

		j--;
	}

	FUNC_END;
}

// hole
void dwt_cdf97_2i_inplace_hole_s(
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

	assert( 1 == dwt_util_get_num_workers() );

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		const int lines_y = size_i_dst_y;
		const int lines_x = size_i_dst_x;

		const int stride_y_j = stride_y * (1 << (j-1));
		const int stride_x_j = stride_x * (1 << (j-1));

		if( lines_x > 1 && lines_x < 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				// FIXME: _hole
				dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x, // N
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y < 4 )
		{
			for(int x = 0; x < size_i_dst_x; x++)
			{
				// FIXME: _hole
				dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y, // N
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_prolog_hole_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_prolog_hole_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_core_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_core_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_epilog_hole_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_epilog_hole_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		j--;
	}

	FUNC_END;
}

// zero
void dwt_cdf97_2i_inplace_zero_s(
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

	assert( 1 == dwt_util_get_num_workers() );

	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	int j = ceil_log2( decompose_one ? size_o_big_max : size_o_big_min );

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		const int lines_y = size_i_dst_y;
		const int lines_x = size_i_dst_x;

		const int stride_y_j = stride_y * (1 << (j-1));
		const int stride_x_j = stride_x * (1 << (j-1));

		if( lines_x > 1 && lines_x < 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				// FIXME: _zero
				dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x, // N
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y < 4 )
		{
			for(int x = 0; x < size_i_dst_x; x++)
			{
				// FIXME: _zero
				dwt_cdf97_i_ex_stride_inplace_part_exceptions_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y, // N
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_prolog_zero_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_prolog_zero_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_core_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_core_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		if( lines_x > 1 && lines_x >= 4 )
		{
			for(int y = 0; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_epilog_zero_s(
					addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
					lines_x,
					stride_y_j);
			}
		}
		if( lines_y > 1 && lines_y >= 4 )
		{
			for(int x = 0; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_inplace_part_epilog_zero_s(
					addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
					lines_y,
					stride_x_j);
			}
		}

		j--;
	}

	FUNC_END;
}

void dwt_cdf53_2i_inplace_s(
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
	const int size_o_big_min = min(size_o_big_x, size_o_big_y);
	const int size_o_big_max = max(size_o_big_x, size_o_big_y);

	int j = ceil_log2(decompose_one ? size_o_big_max : size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		const int stride_y_j = stride_y * (1 << (j-1));
		const int stride_x_j = stride_x * (1 << (j-1));

		for(int y = 0; y < size_i_dst_y; y++)
			dwt_cdf53_i_ex_stride_inplace_s(
				addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
				size_i_dst_x,
				stride_y_j);
		for(int x = 0; x < size_i_dst_x; x++)
			dwt_cdf53_i_ex_stride_inplace_s(
				addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
				size_i_dst_y,
				stride_x_j);

		j--;
	}
}

void dwt_eaw53_2i_inplace_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding,
	float *wH[],
	float *wV[]
)
{
	const int size_o_big_min = min(size_o_big_x, size_o_big_y);
	const int size_o_big_max = max(size_o_big_x, size_o_big_y);

	int j = ceil_log2(decompose_one ? size_o_big_max : size_o_big_min);

	if( j_max >= 0 && j_max < j )
		j = j_max;

	for(;;)
	{
		if(0 == j)
			break;

		const int size_i_dst_x = ceil_div_pow2(size_i_big_x, j-1);
		const int size_i_dst_y = ceil_div_pow2(size_i_big_y, j-1);

		const int stride_y_j = stride_y * (1 << (j-1));
		const int stride_x_j = stride_x * (1 << (j-1));

		for(int x = 0; x < size_i_dst_x; x++)
			dwt_eaw53_i_ex_stride_inplace_s(
				addr2_s(ptr, 0, x, stride_x_j, stride_y_j),
				size_i_dst_y,
				stride_x_j,
				&wV[j-1][x*size_i_dst_y]
			);
		for(int y = 0; y < size_i_dst_y; y++)
			dwt_eaw53_i_ex_stride_inplace_s(
				addr2_s(ptr, y, 0, stride_x_j, stride_y_j),
				size_i_dst_x,
				stride_y_j,
				&wH[j-1][y*size_i_dst_x]
			);

		j--;
	}
}

void dwt_cdf97_2i_s2(
	const void *src,
	void *dst,
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

	// NOTE: this hack copies the input image into dst
	dwt_util_copy_i(
		src,
		dst,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y);

	const int threads = dwt_util_get_num_threads();
	const int workers = dwt_util_get_num_workers();

	const int offset = 0;

#ifdef microblaze
	dwt_util_switch_op(DWT_OP_LIFT4SB);
#endif
	const int size_o_big_min = min(size_o_big_x,size_o_big_y);
	const int size_o_big_max = max(size_o_big_x,size_o_big_y);

	float **temp = alloc_temp_s(threads,
		calc_and_set_temp_size_s(size_o_big_max, offset)
	);

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

			#pragma omp parallel for schedule(static, threads_segment_y)
			for(int y = 0; y < workers_lines_y; y += workers)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(dst,y,0,stride_x,stride_y),
					addr2_const_s(dst,y,size_o_src_x,stride_x,stride_y),
					addr2_s(dst,y,0,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_dst_x,
					stride_y);
			}
			dwt_util_set_num_workers(1);
			for(int y = workers_lines_y; y < lines_y; y++)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(dst,y,0,stride_x,stride_y),
					addr2_const_s(dst,y,size_o_src_x,stride_x,stride_y),
					addr2_s(dst,y,0,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_dst_x,
					stride_y);
			}
			dwt_util_set_num_workers(workers);
		}

		if( lines_y > 1 )
		{
			set_data_step_s( stride_y );

			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < workers_lines_x; x += workers)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(dst,0,x,stride_x,stride_y),
					addr2_const_s(dst,size_o_src_y,x,stride_x,stride_y),
					addr2_s(dst,0,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
					size_i_dst_y,
					stride_x);
			}
			dwt_util_set_num_workers(1);
			for(int x = workers_lines_x; x < lines_x; x++)
			{
				dwt_cdf97_i_ex_stride_s(
					addr2_const_s(dst,0,x,stride_x,stride_y),
					addr2_const_s(dst,size_o_src_y,x,stride_x,stride_y),
					addr2_s(dst,0,x,stride_x,stride_y),
					temp[dwt_util_get_thread_num()],
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
					addr2_s(dst,y,0,stride_x,stride_y),
					size_i_dst_x,
					size_o_dst_x,
					stride_y);
			#pragma omp parallel for schedule(static, threads_segment_x)
			for(int x = 0; x < size_o_dst_x; x++)
				dwt_zero_padding_i_stride_s(
					addr2_s(dst,0,x,stride_x,stride_y),
					size_i_dst_y,
					size_o_dst_y,
					stride_x);
		}

		j--;
	}

	free_temp_s(threads, temp);

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

void dwt_eaw53_2i_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding,
	float *wH[],
	float *wV[]
)
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

		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_eaw53_i_ex_stride_s(
				addr2_s(ptr,0,x,stride_x,stride_y),
				addr2_s(ptr,size_o_src_y,x,stride_x,stride_y),
				addr2_s(ptr,0,x,stride_x,stride_y),
				temp,
				size_i_dst_y, // N
				stride_x,
				&wV[j-1][x*size_i_dst_y]
			);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_y, omp_get_num_threads()))
		for(int y = 0; y < size_o_dst_y; y++)
			dwt_eaw53_i_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x, // N
				stride_y,
				&wH[j-1][y*size_i_dst_x]
			);

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

void dwt_interp53_2i_s(
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
			dwt_interp53_i_ex_stride_s(
				addr2_s(ptr,y,0,stride_x,stride_y),
				addr2_s(ptr,y,size_o_src_x,stride_x,stride_y),
				addr2_s(ptr,y,0,stride_x,stride_y),
				temp,
				size_i_dst_x,
				stride_y);
		#pragma omp parallel for private(temp) schedule(static, ceil_div(size_o_dst_x, omp_get_num_threads()))
		for(int x = 0; x < size_o_dst_x; x++)
			dwt_interp53_i_ex_stride_s(
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
#ifdef __asvp__
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

#ifdef __asvp__
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

#ifdef __asvp__
	for(int w = 0; w < get_total_workers(); w++)
	{
		WAL_CHECK( wal_done_worker(worker[w]) );
	}
#endif

	FUNC_END;
}

void dwt_util_abort()
{
	FUNC_BEGIN;

#ifdef __asvp__
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
					dwt_util_log(LOG_WARN, "%s: Maximum pixel intensity exceeded (%i > %i) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, max_value, y, x);
			}

			if( px > max_value )
			{
				val = target_max_value;
			}

			if( px < 0 )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Minimum pixel intensity exceeded (%i < %i) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, 0, y, x);
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

int dwt_util_save_log_to_pgm_s(
	const char *path,
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	// alloc temp
	void *temp;
	dwt_util_alloc_image(&temp, stride_x, stride_y, size_x, size_y);

	// temp = log(abs(input))
	dwt_util_conv_show_s(ptr, temp, stride_x, stride_y, size_x, size_y);

	// find min, max
	float minv, maxv;
	dwt_util_find_min_max_s(
		temp,
		size_x,
		size_y,
		stride_x,
		stride_y,
		&minv,
		&maxv
	);

#if 0
	// scale + save
	dwt_util_shift_s(
		temp,
		size_x,
		size_y,
		stride_x,
		stride_y,
		-minv
	);

	dwt_util_save_to_pgm_s(
		path,
		(-minv + maxv),
		temp,
		stride_x,
		stride_y,
		size_x,
		size_y
	);
#else
	// scale + save
	dwt_util_save_to_pgm_s(
		path,
		maxv,
		temp,
		stride_x,
		stride_y,
		size_x,
		size_y
	);
#endif

	// free
	dwt_util_free_image(&temp);

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
					dwt_util_log(LOG_WARN, "%s: Maximum pixel intensity exceeded (%f > %f) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, max_value, y, x);
			}

			// isnan
			if( px != px )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: NaN value (%f) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, y, x);
				val = 0;
			}

			if( px > max_value )
			{
				val = target_max_value;
			}

			if( px + 1e-3f < 0.0f )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Minimum pixel intensity exceeded (%f < %f) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, 0.0f, y, x);
			}

			if( px < 0.0f )
			{
				val = 0;
			}

			// minimum integer value
			if( abs((int)px) < 0 )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Wrong value (%f) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, y, x);
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
					dwt_util_log(LOG_WARN, "%s: Maximum pixel intensity exceeded (%f > %f) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, max_value, y, x);
			}

			if( px > max_value )
			{
				val = target_max_value;
			}

			if( px + 1e-6 < 0.0 )
			{
				if( !err++ )
					dwt_util_log(LOG_WARN, "%s: Minimum pixel intensity exceeded (%f < %f) at (y=%i, x=%i). Such an incident will be reported only once.\n", __FUNCTION__, px, 0.0f, y, x);
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

float *dwt_util_allocate_vec_s(int size)
{
	// FIXME: why must be even??? moreover, cannot allocate less elements than required
	size = to_even(size+1);

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

#ifdef __asvp__
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

enum dwt_color {
	DWT_COLOR_DEFAULT = 0,
	// styles
	DWT_COLOR_BOLD,
// 	DWT_COLOR_DIM,
// 	DWT_COLOR_UNDERLINED,
// 	DWT_COLOR_BLINK,
// 	DWT_COLOR_REVERSE,
// 	DWT_COLOR_HIDDEN,
	// colors
	DWT_COLOR_BLACK,
	DWT_COLOR_RED,
	DWT_COLOR_GREEN,
	DWT_COLOR_YELLOW,
	DWT_COLOR_BLUE,
	DWT_COLOR_MAGENTA,
	DWT_COLOR_CYAN,
	DWT_COLOR_LIGHTGRAY,
// 	DWT_COLOR_DARKGRAY,
// 	DWT_COLOR_LIGHTRED,
// 	DWT_COLOR_LIGHTGREEN,
// 	DWT_COLOR_LIGHTYELLOW,
// 	DWT_COLOR_LIGHTBLUE,
// 	DWT_COLOR_LIGHTMAGENTA,
// 	DWT_COLOR_LIGHTCYAN,
// 	DWT_COLOR_WHITE,
};

// TODO: http://misc.flogisoft.com/bash/tip_colors_and_formatting
// TODO: http://www.termsys.demon.co.uk/vtansi.htm
int dwt_util_color(FILE *stream, int style, int foreground, int background)
{
	int ret = 0;

	ret += dwt_util_fprintf(stream, "\e[0m");

	switch(style)
	{
		case DWT_COLOR_BOLD:
			ret += dwt_util_fprintf(stream, "\e[1m");
			break;
	}

	switch(foreground)
	{
		case DWT_COLOR_BLACK:
			ret += dwt_util_fprintf(stream, "\e[30m");
			break;
		case DWT_COLOR_RED:
			ret += dwt_util_fprintf(stream, "\e[31m");
			break;
		case DWT_COLOR_GREEN:
			ret += dwt_util_fprintf(stream, "\e[32m");
			break;
		case DWT_COLOR_YELLOW:
			ret += dwt_util_fprintf(stream, "\e[33m");
			break;
		case DWT_COLOR_BLUE:
			ret += dwt_util_fprintf(stream, "\e[34m");
			break;
		case DWT_COLOR_MAGENTA:
			ret += dwt_util_fprintf(stream, "\e[35m");
			break;
		case DWT_COLOR_CYAN:
			ret += dwt_util_fprintf(stream, "\e[36m");
			break;
		case DWT_COLOR_LIGHTGRAY:
			ret += dwt_util_fprintf(stream, "\e[37m");
			break;
	}

	switch(background)
	{
		case DWT_COLOR_BLACK:
			ret += dwt_util_fprintf(stream, "\e[40m");
			break;
		case DWT_COLOR_RED:
			ret += dwt_util_fprintf(stream, "\e[41m");
			break;
		case DWT_COLOR_GREEN:
			ret += dwt_util_fprintf(stream, "\e[42m");
			break;
		case DWT_COLOR_YELLOW:
			ret += dwt_util_fprintf(stream, "\e[43m");
			break;
		case DWT_COLOR_BLUE:
			ret += dwt_util_fprintf(stream, "\e[44m");
			break;
		case DWT_COLOR_MAGENTA:
			ret += dwt_util_fprintf(stream, "\e[45m");
			break;
		case DWT_COLOR_CYAN:
			ret += dwt_util_fprintf(stream, "\e[46m");
			break;
		case DWT_COLOR_LIGHTGRAY:
			ret += dwt_util_fprintf(stream, "\e[47m");
			break;
	}

	return ret;
}

int dwt_util_log(
	enum dwt_util_loglevel level,
	const char *format,
	...
)
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

	const int color[] = {
		[LOG_NONE] = DWT_COLOR_DEFAULT,
		[LOG_DBG]  = DWT_COLOR_DEFAULT,
		[LOG_INFO] = DWT_COLOR_BLUE,
		[LOG_WARN] = DWT_COLOR_YELLOW,
		[LOG_ERR]  = DWT_COLOR_RED,
		[LOG_TEST] = DWT_COLOR_MAGENTA,
	};

	flockfile(stream);

	ret += dwt_util_color(stream, DWT_COLOR_BOLD, color[level], DWT_COLOR_DEFAULT);
	ret += dwt_util_fprintf(stream, prefix[level]);
	ret += dwt_util_color(stream, DWT_COLOR_DEFAULT, DWT_COLOR_DEFAULT, DWT_COLOR_DEFAULT);

	va_list ap;

	va_start(ap, format);
	ret += dwt_util_vfprintf(stream, format, ap);
	va_end(ap);

	fflush(stream);

	funlockfile(stream);

	return ret;
}

int dwt_util_vlog(
	enum dwt_util_loglevel level,
	const char *format,
	va_list ap)
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

	ret += dwt_util_vfprintf(stream, format, ap);

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
	dwt_util_vlog(LOG_ERR, format, ap);
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

	int err = 0;

	for(int y = 0; y < size_i_big_y; y++)
	{
		for(int x = 0; x < size_i_big_x; x++)
		{
			const float coeff = *addr2_const_s(src, y, x, stride_x, stride_y);
			float *log_coeff = addr2_s(dst, y, x, stride_x, stride_y);

			float temp;
			log_i_s(&temp, 1.f+fabsf(coeff)*a);
			temp /= b;

			if( !isfinite(temp) )
			{
				if(!err)
					dwt_util_log(LOG_ERR, "either NaN or INFINITY; this error will be reported only once\n");
				err++;

				temp = 0.f;
			}

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

void dwt_util_copy3_s(
	const void *src,
	void *dst,
	int src_stride_x,
	int src_stride_y,
	int dst_stride_x,
	int dst_stride_y,
	int size_x,
	int size_y
)
{
	FUNC_BEGIN;

	assert( src != NULL && dst != NULL && size_x >= 0 && size_y >= 0 );

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			const float src_coeff = *addr2_const_s(src, y, x, src_stride_x, src_stride_y);
			float *dst_coeff = addr2_s(dst, y, x, dst_stride_x, dst_stride_y);

			*dst_coeff = src_coeff;
		}
	}

	FUNC_END;
}

void dwt_util_copy_d(
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
			const double src_coeff = *addr2_const_d(src, y, x, stride_x, stride_y);
			double *dst_coeff = addr2_d(dst, y, x, stride_x, stride_y);

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
void dwt_util_perf_cdf97_2_inplace_s(
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

	void *template;

	dwt_util_alloc_image(
			&template,
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y);

	dwt_util_test_image_fill_s(
			template,
			stride_x,
			stride_y,
			size_i_big_x,
			size_i_big_y,
			0);

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
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
		for(int m = 0; m < M; m++)
		{
			// fill with test pattern
			dwt_util_test_image_fill_s(
				ptr[m],
				stride_x,
				stride_y,
				size_i_big_x,
				size_i_big_y,
				0);
		}
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
			dwt_cdf97_2f_inplace_s(
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
			dwt_cdf97_2i_inplace_s(
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

		// compare
		for(int m = 0; m < M; m++)
		{
			if( dwt_util_compare_s(ptr[m], template, stride_x, stride_y, size_i_big_x, size_i_big_y) )
			{
				dwt_util_log(LOG_ERR, "images differ!\n");
			}
		}
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}
	dwt_util_free_image(&template);

	FUNC_END;
}

// TODO: propagate "flush"
void dwt_util_perf_cdf97_2_inplace_sep_s(
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

	void *template;

	dwt_util_alloc_image(
			&template,
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y);

	dwt_util_test_image_fill_s(
			template,
			stride_x,
			stride_y,
			size_i_big_x,
			size_i_big_y,
			0);

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
			size_o_big_y
		);
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
		for(int m = 0; m < M; m++)
		{
			// fill with test pattern
			dwt_util_test_image_fill_s(
				ptr[m],
				stride_x,
				stride_y,
				size_i_big_x,
				size_i_big_y,
				0);
		}
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
			dwt_cdf97_2f_inplace_sep_s(
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
			dwt_cdf97_2i_inplace_s(
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

		// compare
		for(int m = 0; m < M; m++)
		{
			if( dwt_util_compare_s(ptr[m], template, stride_x, stride_y, size_i_big_x, size_i_big_y) )
			{
				dwt_util_log(LOG_ERR, "images differ!\n");
			}
		}
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}
	dwt_util_free_image(&template);

	FUNC_END;
}

// TODO: propagate "flush"
void dwt_util_perf_cdf97_2_inplace_sep_sdl_s(
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

	void *template;

	dwt_util_alloc_image(
			&template,
			stride_x,
			stride_y,
			size_o_big_x,
			size_o_big_y);

	dwt_util_test_image_fill_s(
			template,
			stride_x,
			stride_y,
			size_i_big_x,
			size_i_big_y,
			0);

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
	}

	*fwd_secs = +INFINITY;
	*inv_secs = +INFINITY;

	// perform N test loops, select minimum
	for(int n = 0; n < N; n++)
	{
		for(int m = 0; m < M; m++)
		{
			// fill with test pattern
			dwt_util_test_image_fill_s(
				ptr[m],
				stride_x,
				stride_y,
				size_i_big_x,
				size_i_big_y,
				0);
		}
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
			dwt_cdf97_2f_inplace_sep_sdl_s(
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
			dwt_cdf97_2i_inplace_s(
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

		// compare
		for(int m = 0; m < M; m++)
		{
			if( dwt_util_compare_s(ptr[m], template, stride_x, stride_y, size_i_big_x, size_i_big_y) )
			{
				dwt_util_log(LOG_ERR, "images differ!\n");
			}
		}
	}

	// free M images
	for(int m = 0; m < M; m++)
	{
		dwt_util_free_image(&ptr[m]);
	}
	dwt_util_free_image(&template);

	FUNC_END;
}

// TODO: propagate "flush"
void dwt_util_perf_cdf97_2_inplace_sdl_s(
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
			dwt_cdf97_2f_inplace_sdl_s(
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
			// FIXME: use SDL version of inverse transform
			dwt_cdf97_2i_inplace_s(
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
	int *size_i_big_y
)
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
	int *size_i_big_y
)
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
	int *size_i_big_y
)
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
// float g_growth_factor_s = 1.28f;
// float g_growth_factor_d = 1.28;
float g_growth_factor_s = 1.13f;
float g_growth_factor_d = 1.13;

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
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
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
	for(int x = min_x; x <= max_x; x = ceil(x * growth_factor))
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
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
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

void dwt_util_measure_perf_cdf97_2_inplace_s(
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
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
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
		dwt_util_perf_cdf97_2_inplace_s(
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

void dwt_util_measure_perf_cdf97_2_inplace_sep_s(
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
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
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
		dwt_util_perf_cdf97_2_inplace_sep_s(
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

void dwt_util_measure_perf_cdf97_2_inplace_sep_sdl_s(
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
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
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
		dwt_util_perf_cdf97_2_inplace_sep_sdl_s(
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

void dwt_util_measure_perf_cdf97_2_inplace_sdl_s(
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
	for(int x = min_x; x <= max_x; x = ceilf(x * growth_factor))
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
		dwt_util_perf_cdf97_2_inplace_sdl_s(
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
	for(int x = min_x; x <= max_x; x = ceil(x * growth_factor))
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
int cmp_s(
	const void *p1,
	const void *p2
)
{
	if( *(const float *)p1 > *(const float *)p2 )
		return +1;
	if( *(const float *)p1 < *(const float *)p2 )
		return -1;
	return 0;
}

float dwt_util_band_med_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	const int size = size_x * size_y;

	//dwt_util_log(LOG_DBG, "size=%i size_x=%i size_y=%i\n", size, size_x, size_y);

	float *arr = dwt_util_allocate_vec_s(size);

#ifdef FV_ON_MAGNITUDES
	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
			arr[y*size_x+x] = fabsf(
				*dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y) );
#else
	for(int y = 0; y < size_y; y++)
		dwt_util_memcpy_stride_s(
			&arr[y*size_x],
			sizeof(float),
			dwt_util_addr_coeff_const_s(ptr, y, 0, stride_x, stride_y),
			stride_y,
			size_x);
#endif

	qsort(arr, size, sizeof(float), cmp_s);

	const float med = arr[size/2];

	free(arr);

	return med;
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
			fv[count++] = dwt_util_band_med_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_med_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_med_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

float dwt_util_band_maxidx_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
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
	int size_y
)
{
	float sum = 0.0f;

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
#ifdef FV_ON_MAGNITUDES
			float coeff = fabsf(*dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y));
#else
			float coeff = *dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y);
#endif
			sum += coeff;
		}

	sum /= size_x * size_y;

	return sum;
}

float dwt_util_band_moment_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	int n,
	float c
)
{
	const int size = size_x * size_y;

	float sum = 0.0f;

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
#ifdef FV_ON_MAGNITUDES
			float coeff = fabsf(*dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y));
#else
			float coeff = *dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y);
#endif
			sum += powf(coeff - c, n);
		}

	return sum/size;
}

float dwt_util_band_cmoment_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	int n
)
{
	const float mean = dwt_util_band_mean_s(ptr, stride_x, stride_y, size_x, size_y);

	return dwt_util_band_moment_s(ptr, stride_x, stride_y, size_x, size_y, n, mean);
}

float dwt_util_band_var_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	return dwt_util_band_cmoment_s(ptr, stride_x, stride_y, size_x, size_y, 2);
}

float dwt_util_band_stdev_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	const float var = dwt_util_band_var_s(ptr, stride_x, stride_y, size_x, size_y);

	return sqrtf(var);
}

float dwt_util_band_smoment_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	int n
)
{
	const float stdev = dwt_util_band_stdev_s(ptr, stride_x, stride_y, size_x, size_y);

	return dwt_util_band_cmoment_s(ptr, stride_x, stride_y, size_x, size_y, n) / powf(stdev, n);
}

float dwt_util_band_skew_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	return dwt_util_band_smoment_s(ptr, stride_x, stride_y, size_x, size_y, 3);
}

float dwt_util_band_kurt_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	return dwt_util_band_smoment_s(ptr, stride_x, stride_y, size_x, size_y, 4) - 3;
}

float dwt_util_band_maxnorm_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	float max = 0.0f;

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			const float c = fabsf(*dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y));

			if( c > max )
				max = c;
		}

	return max;
}

float dwt_util_band_lpnorm_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y,
	float p
)
{
	float sum = 0.0f;

	if( +INFINITY == p )
		return dwt_util_band_maxnorm_s(ptr, stride_x, stride_y, size_x, size_y);

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			const float c = *dwt_util_addr_coeff_const_s(ptr, y, x, stride_x, stride_y);

			sum += powf(fabsf(c), p);
		}

	return powf(sum, 1/p);
}

float dwt_util_band_norm_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	return dwt_util_band_lpnorm_s(ptr, stride_x, stride_y, size_x, size_y, 2);
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
			fv[count++] = dwt_util_band_maxidx_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_maxidx_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_maxidx_s(band_ptr, stride_x, stride_y, band_x, band_y);
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
			fv[count++] = dwt_util_band_mean_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_mean_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_mean_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

void dwt_util_var_s(
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
			fv[count++] = dwt_util_band_var_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_var_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_var_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

void dwt_util_stdev_s(
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
			fv[count++] = dwt_util_band_stdev_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_stdev_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_stdev_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

void dwt_util_skew_s(
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
			fv[count++] = dwt_util_band_skew_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_skew_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_skew_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

void dwt_util_kurt_s(
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
			fv[count++] = dwt_util_band_kurt_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_kurt_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_kurt_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

void dwt_util_maxnorm_s(
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
			fv[count++] = dwt_util_band_maxnorm_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_maxnorm_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_maxnorm_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

void dwt_util_lpnorm_s(
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	float *fv,
	float p
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
			fv[count++] = dwt_util_band_lpnorm_s(band_ptr, stride_x, stride_y, band_x, band_y, p);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_lpnorm_s(band_ptr, stride_x, stride_y, band_x, band_y, p);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_lpnorm_s(band_ptr, stride_x, stride_y, band_x, band_y, p);
	}
}

void dwt_util_norm_s(
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
			fv[count++] = dwt_util_band_norm_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_LH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_norm_s(band_ptr, stride_x, stride_y, band_x, band_y);

		dwt_util_subband_const_s(ptr, stride_x, stride_y, size_o_big_x, size_o_big_y, size_i_big_x, size_i_big_y, j, DWT_HH, &band_ptr, &band_x, &band_y);
		if( band_x && band_y )
			fv[count++] = dwt_util_band_norm_s(band_ptr, stride_x, stride_y, band_x, band_y);
	}
}

int dwt_util_test_cdf97_2_s(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding
)
{
	int j = j_max;
	void *data, *copy;

	// allocate image
	dwt_util_alloc_image(
		&data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// allocate copy
	dwt_util_alloc_image(
		&copy,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// fill with test pattern
	dwt_util_test_image_fill_s(
		data,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y,
		0);

	// copy test the image into the copy
	dwt_util_copy_s(
		data,
		copy,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y);

	// forward
	dwt_cdf97_2f_s(
		data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		&j,
		decompose_one,
		zero_padding);

	// inverse
	dwt_cdf97_2i_s(
		data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j,
		decompose_one,
		zero_padding);

	int ret;

	// compare
	if( dwt_util_compare_s(data, copy, stride_x, stride_y, size_i_big_x, size_i_big_y) )
		ret = 1;
	else
		ret = 0;

	dwt_util_free_image(&data);
	dwt_util_free_image(&copy);

	return ret;
}

int dwt_util_test_cdf97_2_s2(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding
)
{
	int j = j_max;
	void *data1, *data2, *data3, *copy;

	// allocate image
	dwt_util_alloc_image(
		&data1,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	dwt_util_alloc_image(
		&data2,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	dwt_util_alloc_image(
		&data3,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// allocate copy
	dwt_util_alloc_image(
		&copy,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// fill with test pattern
	dwt_util_test_image_fill_s(
		data1,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y,
		0);

	// copy test the image into the copy
	dwt_util_copy_s(
		data1,
		copy,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y);

	// forward
	dwt_cdf97_2f_s2(
		data1,
		data2,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		&j,
		decompose_one,
		zero_padding);

	// inverse
	dwt_cdf97_2i_s2(
		data2,
		data3,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j,
		decompose_one,
		zero_padding);

	int ret;

	// compare
	if( dwt_util_compare_s(data3, copy, stride_x, stride_y, size_i_big_x, size_i_big_y) )
		ret = 1;
	else
		ret = 0;

	dwt_util_free_image(&data1);
	dwt_util_free_image(&data2);
	dwt_util_free_image(&data3);
	dwt_util_free_image(&copy);

	return ret;
}

int dwt_util_test_cdf97_2_d(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding
)
{
	int j = j_max;
	void *data, *copy;

	// allocate image
	dwt_util_alloc_image(
		&data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// allocate copy
	dwt_util_alloc_image(
		&copy,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// fill with test pattern
	dwt_util_test_image_fill_d(
		data,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y,
		0);

	// copy test the image into the copy
	dwt_util_copy_d(
		data,
		copy,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y);

	// forward
	dwt_cdf97_2f_d(
		data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		&j,
		decompose_one,
		zero_padding);

	// inverse
	dwt_cdf97_2i_d(
		data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j,
		decompose_one,
		zero_padding);

	int ret;

	// compare
	if( dwt_util_compare_d(data, copy, stride_x, stride_y, size_i_big_x, size_i_big_y) )
		ret = 1;
	else
		ret = 0;

	dwt_util_free_image(&data);
	dwt_util_free_image(&copy);

	return ret;
}


int dwt_util_test_cdf97_2_i(
	int stride_x,
	int stride_y,
	int size_o_big_x,
	int size_o_big_y,
	int size_i_big_x,
	int size_i_big_y,
	int j_max,
	int decompose_one,
	int zero_padding
)
{
	int j = j_max;
	void *data, *copy;

	// allocate image
	dwt_util_alloc_image(
		&data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// allocate copy
	dwt_util_alloc_image(
		&copy,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y);

	// fill with test pattern
	dwt_util_test_image_fill_i(
		data,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y,
		0);

	// copy test the image into the copy
	dwt_util_copy_i(
		data,
		copy,
		stride_x,
		stride_y,
		size_i_big_x,
		size_i_big_y);

	// forward
	dwt_cdf97_2f_i(
		data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		&j,
		decompose_one,
		zero_padding);

	// inverse
	dwt_cdf97_2i_i(
		data,
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j,
		decompose_one,
		zero_padding);

	int ret;

	// compare
	if( dwt_util_compare_i(data, copy, stride_x, stride_y, size_i_big_x, size_i_big_y) )
		ret = 1;
	else
		ret = 0;

	dwt_util_free_image(&data);
	dwt_util_free_image(&copy);

	return ret;
}

int dwt_util_test2_cdf97_2_s(
	enum dwt_array array_type,
	int size_x,
	int size_y,
	int opt_stride,
	int j_max,
	int decompose_one
)
{
	int stride_x;
	int stride_y;
	int size_o_big_x;
	int size_o_big_y;
	int size_i_big_x;
	int size_i_big_y;

	// get sizes
	dwt_util_get_sizes_s(
		array_type,
		size_x,
		size_y,
		opt_stride,
		&stride_x,
		&stride_y,
		&size_o_big_x,
		&size_o_big_y,
		&size_i_big_x,
		&size_i_big_y
	);

	return dwt_util_test_cdf97_2_s(
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		decompose_one,
		0
	);
}

int dwt_util_test2_cdf97_2_s2(
	enum dwt_array array_type,
	int size_x,
	int size_y,
	int opt_stride,
	int j_max,
	int decompose_one
)
{
	int stride_x;
	int stride_y;
	int size_o_big_x;
	int size_o_big_y;
	int size_i_big_x;
	int size_i_big_y;

	// get sizes
	dwt_util_get_sizes_s(
		array_type,
		size_x,
		size_y,
		opt_stride,
		&stride_x,
		&stride_y,
		&size_o_big_x,
		&size_o_big_y,
		&size_i_big_x,
		&size_i_big_y
	);

	return dwt_util_test_cdf97_2_s2(
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		decompose_one,
		0
	);
}

int dwt_util_test2_cdf97_2_d(
	enum dwt_array array_type,
	int size_x,
	int size_y,
	int opt_stride,
	int j_max,
	int decompose_one
)
{
	int stride_x;
	int stride_y;
	int size_o_big_x;
	int size_o_big_y;
	int size_i_big_x;
	int size_i_big_y;

	// get sizes
	dwt_util_get_sizes_d(
		array_type,
		size_x,
		size_y,
		opt_stride,
		&stride_x,
		&stride_y,
		&size_o_big_x,
		&size_o_big_y,
		&size_i_big_x,
		&size_i_big_y
	);

	return dwt_util_test_cdf97_2_d(
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		decompose_one,
		0
	);
}

int dwt_util_test2_cdf97_2_i(
	enum dwt_array array_type,
	int size_x,
	int size_y,
	int opt_stride,
	int j_max,
	int decompose_one
)
{
	int stride_x;
	int stride_y;
	int size_o_big_x;
	int size_o_big_y;
	int size_i_big_x;
	int size_i_big_y;

	// get sizes
	dwt_util_get_sizes_i(
		array_type,
		size_x,
		size_y,
		opt_stride,
		&stride_x,
		&stride_y,
		&size_o_big_x,
		&size_o_big_y,
		&size_i_big_x,
		&size_i_big_y
	);

	return dwt_util_test_cdf97_2_i(
		stride_x,
		stride_y,
		size_o_big_x,
		size_o_big_y,
		size_i_big_x,
		size_i_big_y,
		j_max,
		decompose_one,
		0
	);
}

void dwt_util_abs_s(
	void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( ptr );

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			float *c = dwt_util_addr_coeff_s(ptr, y, x, stride_x, stride_y);

			*c = fabsf(*c);
		}
}

struct delta {
	int curr;
	int *symb;
	int symb_no;
	int next;
	void (*func)(void *, int, int *, int);
};

static
int fsm_symb_found(int symb, int *symb_group, int symb_no)
{
	int found = 0;

	for(int s = 0; s < symb_no; s++)
	{
		if( symb == symb_group[s] )
			found = 1;
	}

	return found;
}

/**
 * @brief Finite state machine.
 * @return Returns final or error state.
 */
static
int fsm(
	void *ctx,
	int (*get_symb)(void *),
	struct delta *delta,
	int count,
	int s_init,
	int s_final,
	int s_error
)
{
	int state = s_init;

	while(1)
	{
		int symb = get_symb(ctx);

		int d;

		for(d = 0; d < count; d++)
		{
			struct delta *row = &delta[d];

			if( state == row->curr )
			{
				if( fsm_symb_found(symb, row->symb, row->symb_no) )
				{
					if( row->func )
						row->func(ctx, symb, row->symb, row->symb_no);
					state = row->next;
					break;
				}
			}
		}

		if( count == d )
			state = s_error;

		if( s_error == state )
		{
			dwt_util_log(LOG_DBG, "FSM in error state: symb=%i(%c)\n", symb, (char)symb);
		}

		if( s_final == state || s_error == state )
			break;
	}

	return state;
}

int dwt_util_save_to_mat_s(
	const char *path,
	const void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y
)
{
	//dwt_util_log(LOG_DBG, "size_x=%i size_y=%i\n", size_x, size_y);

	FILE *file = fopen(path, "w");

	if( NULL == file )
		return 1;

	int symb_delim[] = { ',', ';', '\t', ' ' };
	int symb_newline[] = { '\n', '\r' };

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float coeff = *dwt_util_addr_coeff_const_s(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			);

			fprintf(file, "%f", coeff);

			if( x+1 != size_x )
				fprintf(file, "%c", (char)symb_delim[0]);
		}

		fprintf(file, "%c", (char)symb_newline[0]);
	}

	fclose(file);
	return 0;
}

struct mat_context
{
	FILE *file;

	int curr_cols;
	int min_cols;
	int rows;

	void **ptr;
	int *size_x;
	int *size_y;
	int *stride_x;
	int *stride_y;
};

static
void mat_context_init(struct mat_context *ctx, FILE *file, void **ptr, int *size_x, int *size_y, int *stride_x, int *stride_y)
{
	ctx->file = file;

	ctx->ptr = ptr;
	ctx->size_x = size_x;
	ctx->size_y = size_y;
	ctx->stride_x = stride_x;
	ctx->stride_y = stride_y;

	*ptr = NULL;
}

static
void mat_context_reset(struct mat_context *ctx)
{
	ctx->curr_cols = 0;
	ctx->min_cols = 0;
	ctx->rows = 0;
}

int mat_get_symb(void *ctx)
{
	struct mat_context *c = ctx;

	return fgetc(c->file);
}

int mat_unget_symb(void *ctx, int symb)
{
	struct mat_context *c = ctx;

	return ungetc(symb, c->file);
}

void mat_end_line(void *ctx, int symb, int *symb_group, int symb_no)
{
	struct mat_context *c = ctx;

	UNUSED(symb);
	UNUSED(symb_group);
	UNUSED(symb_no);

	//dwt_util_log(LOG_DBG, "end line: curr_cols=%i min_cols=%i\n", c->curr_cols, c->min_cols);

	c->min_cols = min(
		c->curr_cols?c->curr_cols:c->min_cols,
		c->min_cols?c->min_cols:c->curr_cols);
	c->curr_cols = 0;
}

static
int str_val_s(const char *buff, float *val)
{
	return 1 != sscanf(buff, "%f", val);
}

static
int str_val_i(const char *buff, int *val)
{
	return 1 != sscanf(buff, "%i", val);
}

#define CELL_MAX 256
void mat_cell_read_s(void *ctx, int symb, int *symb_group, int symb_no)
{
	struct mat_context *c = ctx;

	char buff[CELL_MAX];
	int cnt = 0;
	buff[cnt++] = (char)symb;
	do{
		int symb_new = mat_get_symb(ctx);

		if( fsm_symb_found(symb_new, symb_group, symb_no) )
		{
			buff[cnt++] = (char)symb_new;
		}
		else
		{
			mat_unget_symb(ctx, symb_new);
			break;
		}
	} while( cnt+1 < CELL_MAX );
	buff[cnt] = 0;

	int pos_x = c->curr_cols-1;
	int pos_y = c->rows;

	float val;
	if( str_val_s(buff, &val) )
	{
		dwt_util_log(LOG_WARN, "invalid cell content\n");
	}
	else
	{
		//dwt_util_log(LOG_DBG, "store %f at (y=%i,x=%i)\n", val, pos_y, pos_x);
		if( pos_x+1 > *c->size_x )
		{
			dwt_util_log(LOG_WARN, "x-coordinate is over limit\n");
		}
		else
		{
			float *coeff = dwt_util_addr_coeff_s(
				*c->ptr,
				pos_y,
				pos_x,
				*c->stride_x,
				*c->stride_y
			);

			*coeff = val;
		}
	}
}
#undef CELL_MAX

#define CELL_MAX 256
void mat_cell_read_i(void *ctx, int symb, int *symb_group, int symb_no)
{
	struct mat_context *c = ctx;

	char buff[CELL_MAX];
	int cnt = 0;
	buff[cnt++] = (char)symb;
	do{
		int symb_new = mat_get_symb(ctx);

		if( fsm_symb_found(symb_new, symb_group, symb_no) )
		{
			buff[cnt++] = (char)symb_new;
		}
		else
		{
			mat_unget_symb(ctx, symb_new);
			break;
		}
	} while( cnt+1 < CELL_MAX );
	buff[cnt] = 0;

	int pos_x = c->curr_cols-1;
	int pos_y = c->rows;

	int val;
	if( str_val_i(buff, &val) )
	{
		dwt_util_log(LOG_WARN, "invalid cell content\n");
	}
	else
	{
		//dwt_util_log(LOG_DBG, "store %f at (y=%i,x=%i)\n", val, pos_y, pos_x);
		if( pos_x+1 > *c->size_x )
		{
			dwt_util_log(LOG_WARN, "x-coordinate is over limit\n");
		}
		else
		{
			int *coeff = dwt_util_addr_coeff_i(
				*c->ptr,
				pos_y,
				pos_x,
				*c->stride_x,
				*c->stride_y
			);

			*coeff = val;
		}
	}
}
#undef CELL_MAX

void mat_new_cell_s(void *ctx, int symb, int *symb_group, int symb_no)
{
	struct mat_context *c = ctx;

	c->curr_cols++;

	// read a cell content when the matrix is allocated
	if( *c->ptr )
		mat_cell_read_s(ctx, symb, symb_group, symb_no);
}

void mat_new_cell_i(void *ctx, int symb, int *symb_group, int symb_no)
{
	struct mat_context *c = ctx;

	c->curr_cols++;

	// read a cell content when the matrix is allocated
	if( *c->ptr )
		mat_cell_read_i(ctx, symb, symb_group, symb_no);
}

void mat_new_line(void *ctx, int symb, int *symb_group, int symb_no)
{
	struct mat_context *c = ctx;

	//dwt_util_log(LOG_DBG, "new line on row %i\n", c->rows+1);

	mat_end_line(ctx, symb, symb_group, symb_no);

	c->rows++;
}

int dwt_util_load_from_mat_s(
	const char *path,
	void **ptr,
	int *size_x,
	int *size_y,
	int *stride_x,
	int *stride_y
)
{
	FILE *file = fopen(path, "r");

	if( NULL == file )
	{
		*ptr = NULL;
		return 1;
	}

	enum state {
		S_START,
		S_DELIM,
		S_CELL,
		S_FINAL,
		S_ERROR
	};

	int symb_delim[] = { ',', ';', '\t', ' ' };
	int symb_newline[] = { '\n', '\r' };
	int symb_number[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', 'e', '+' };
	int symb_eof[] = { EOF };

	struct delta delta[] = {
		{ S_START, symb_delim,   sizeof_arr(symb_delim),   S_DELIM, 0 },
		{ S_START, symb_newline, sizeof_arr(symb_newline), S_START, mat_end_line },
		{ S_START, symb_number,  sizeof_arr(symb_number),  S_CELL,  mat_new_cell_s },
		{ S_START, symb_eof,     sizeof_arr(symb_eof),     S_FINAL, mat_end_line },
		{ S_DELIM, symb_delim,   sizeof_arr(symb_delim),   S_DELIM, 0 },
		{ S_DELIM, symb_newline, sizeof_arr(symb_newline), S_START, mat_new_line },
		{ S_DELIM, symb_number,  sizeof_arr(symb_number),  S_CELL,  mat_new_cell_s },
		{ S_DELIM, symb_eof,     sizeof_arr(symb_eof),     S_FINAL, mat_end_line },
		{ S_CELL,  symb_number,  sizeof_arr(symb_number),  S_CELL,  0 },
		{ S_CELL,  symb_newline, sizeof_arr(symb_newline), S_START, mat_new_line },
		{ S_CELL,  symb_delim,   sizeof_arr(symb_delim),   S_DELIM, 0 },
		{ S_CELL,  symb_eof,     sizeof_arr(symb_eof),     S_FINAL, mat_end_line },
	};

	struct mat_context ctx;

	mat_context_init(&ctx, file, ptr, size_x, size_y, stride_x, stride_y);
	mat_context_reset(&ctx);

	if( S_ERROR == fsm(&ctx, mat_get_symb, delta, sizeof_arr(delta), S_START, S_FINAL, S_ERROR) )
	{
		fclose(file);
		*ptr = NULL;
		return 2;
	}

	//dwt_util_log(LOG_DBG, "y=%i x=%i\n", ctx.rows, ctx.min_cols);

	*size_x = ctx.min_cols;
	*size_y = ctx.rows;
	*stride_y = sizeof(float);
	*stride_x = dwt_util_get_opt_stride(*stride_y * *size_x);
	dwt_util_alloc_image(ctx.ptr, *ctx.stride_x, *ctx.stride_y, *ctx.size_x, *ctx.size_y);

	mat_context_reset(&ctx);

	rewind(file);

	if( S_ERROR == fsm(&ctx, mat_get_symb, delta, sizeof_arr(delta), S_START, S_FINAL, S_ERROR) )
	{
		fclose(file);
		// free allocated image
		dwt_util_free_image(ptr);
		*ptr = NULL;
		return 3;
	}

	fclose(file);
	return 0;
}

int dwt_util_load_from_mat_i(
	const char *path,
	void **ptr,
	int *size_x,
	int *size_y,
	int *stride_x,
	int *stride_y
)
{
	FILE *file = fopen(path, "r");

	if( NULL == file )
	{
		*ptr = NULL;
		return 1;
	}

	enum state {
		S_START,
		S_DELIM,
		S_CELL,
		S_FINAL,
		S_ERROR
	};

	int symb_delim[] = { ',', ';', '\t', ' ' };
	int symb_newline[] = { '\n', '\r' };
	int symb_number[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-' };
	int symb_eof[] = { EOF };

	struct delta delta[] = {
		{ S_START, symb_delim,   sizeof_arr(symb_delim),   S_DELIM, 0 },
		{ S_START, symb_newline, sizeof_arr(symb_newline), S_START, mat_end_line },
		{ S_START, symb_number,  sizeof_arr(symb_number),  S_CELL,  mat_new_cell_i },
		{ S_START, symb_eof,     sizeof_arr(symb_eof),     S_FINAL, mat_end_line },
		{ S_DELIM, symb_delim,   sizeof_arr(symb_delim),   S_DELIM, 0 },
		{ S_DELIM, symb_newline, sizeof_arr(symb_newline), S_START, mat_new_line },
		{ S_DELIM, symb_number,  sizeof_arr(symb_number),  S_CELL,  mat_new_cell_i },
		{ S_DELIM, symb_eof,     sizeof_arr(symb_eof),     S_FINAL, mat_end_line },
		{ S_CELL,  symb_number,  sizeof_arr(symb_number),  S_CELL,  0 },
		{ S_CELL,  symb_newline, sizeof_arr(symb_newline), S_START, mat_new_line },
		{ S_CELL,  symb_delim,   sizeof_arr(symb_delim),   S_DELIM, 0 },
		{ S_CELL,  symb_eof,     sizeof_arr(symb_eof),     S_FINAL, mat_end_line },
	};

	struct mat_context ctx;

	mat_context_init(&ctx, file, ptr, size_x, size_y, stride_x, stride_y);
	mat_context_reset(&ctx);

	if( S_ERROR == fsm(&ctx, mat_get_symb, delta, sizeof_arr(delta), S_START, S_FINAL, S_ERROR) )
	{
		fclose(file);
		*ptr = NULL;
		return 2;
	}

	//dwt_util_log(LOG_DBG, "y=%i x=%i\n", ctx.rows, ctx.min_cols);

	*size_x = ctx.min_cols;
	*size_y = ctx.rows;
	*stride_y = sizeof(int);
	*stride_x = dwt_util_get_opt_stride(*stride_y * *size_x);
	dwt_util_alloc_image(ctx.ptr, *ctx.stride_x, *ctx.stride_y, *ctx.size_x, *ctx.size_y);

	mat_context_reset(&ctx);

	rewind(file);

	if( S_ERROR == fsm(&ctx, mat_get_symb, delta, sizeof_arr(delta), S_START, S_FINAL, S_ERROR) )
	{
		fclose(file);
		// free allocated image
		dwt_util_free_image(ptr);
		*ptr = NULL;
		return 3;
	}

	fclose(file);
	return 0;
}

/**
 * @brief Saturation arithmetic.
 */
static
int saturate_i(int val, int lo, int hi)
{
	if( val < lo )
		return lo;
	if( val > hi )
		return hi;
	return val;
}

static
float dot_s(
	const void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
)
{
	float sum = 0.0f;

	for(int y1 = 0; y1 < size1_y; y1++)
		for(int x1 = 0; x1 < size1_x; x1++)
		{
			float val1 = *dwt_util_addr_coeff_const_s(
				ptr1,
				y1,
				x1,
				stride1_x,
				stride1_y
			);

			// saturation arithmetic
			int y2 = saturate_i(y1 + displ_y, 0, size2_y-1);
			int x2 = saturate_i(x1 + displ_x, 0, size2_x-1);

			float val2 = *dwt_util_addr_coeff_const_s(
				ptr2,
				y2,
				x2,
				stride2_x,
				stride2_y
			);

			sum += val1 * val2;
		}

	return sum;
}

float dwt_util_dot_s(
	const void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
)
{
	return dot_s(
		ptr1,
		size1_x,
		size1_y,
		stride1_x,
		stride1_y,
		displ_x,
		displ_y,
		ptr2,
		size2_x,
		size2_y,
		stride2_x,
		stride2_y
	);
}

static
void normalize_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float p
)
{
	float norm = dwt_util_band_lpnorm_s(
		ptr,
		stride_x,
		stride_y,
		size_x,
		size_y,
		p
	);

	for(int y = 0; y < size_y; y++)
		for(int x = 0; x < size_x; x++)
		{
			float *coeff = dwt_util_addr_coeff_s(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			);

			*coeff /= norm;
		}

	dwt_util_log(LOG_DBG, "normalize: p=%f, norm %f => %f\n",
		p,
		norm,
		dwt_util_band_lpnorm_s(
			ptr,
			stride_x,
			stride_y,
			size_x,
			size_y,
			p
		)
	);
}

void dwt_util_normalize_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float p
)
{
	normalize_s(
		ptr,
		size_x,
		size_y,
		stride_x,
		stride_y,
		p
	);
}

static
void add_s(
	void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
)
{
	for(int y1 = 0; y1 < size1_y; y1++)
		for(int x1 = 0; x1 < size1_x; x1++)
		{
			float *pdst = dwt_util_addr_coeff_s(
				ptr1,
				y1,
				x1,
				stride1_x,
				stride1_y
			);

			// saturation arithmetic
			int y2 = saturate_i(y1 + displ_y, 0, size2_y-1);
			int x2 = saturate_i(x1 + displ_x, 0, size2_x-1);

			float src = *dwt_util_addr_coeff_const_s(
				ptr2,
				y2,
				x2,
				stride2_x,
				stride2_y
			);

			*pdst += src;
		}
}

static
void mul_s(
	void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
)
{
	for(int y1 = 0; y1 < size1_y; y1++)
		for(int x1 = 0; x1 < size1_x; x1++)
		{
			float *pdst = dwt_util_addr_coeff_s(
				ptr1,
				y1,
				x1,
				stride1_x,
				stride1_y
			);

			// saturation arithmetic
			int y2 = saturate_i(y1 + displ_y, 0, size2_y-1);
			int x2 = saturate_i(x1 + displ_x, 0, size2_x-1);

			float src = *dwt_util_addr_coeff_const_s(
				ptr2,
				y2,
				x2,
				stride2_x,
				stride2_y
			);

			*pdst *= src;
		}
}

void dwt_util_add_s(
	void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
)
{
	add_s(
		ptr1,
		size1_x,
		size1_y,
		stride1_x,
		stride1_y,
		displ_x,
		displ_y,
		ptr2,
		size2_x,
		size2_y,
		stride2_x,
		stride2_y
	);
}

void dwt_util_mul_s(
	void *ptr1,
	int size1_x,
	int size1_y,
	int stride1_x,
	int stride1_y,
	int displ_x,
	int displ_y,
	const void *ptr2,
	int size2_x,
	int size2_y,
	int stride2_x,
	int stride2_y
)
{
	mul_s(
		ptr1,
		size1_x,
		size1_y,
		stride1_x,
		stride1_y,
		displ_x,
		displ_y,
		ptr2,
		size2_x,
		size2_y,
		stride2_x,
		stride2_y
	);
}

int dwt_util_save_to_svm_s(
	const char *path,
	const void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	const void *cls_ptr,
	int cls_size_x,
	int cls_size_y,
	int cls_stride_x,
	int cls_stride_y
)
{
	// assert
	assert( path && ptr && cls_ptr );
	assert( size_x > 0 && size_y > 0 && cls_size_x == 1 && cls_size_y == size_y );

	// fopen
	FILE *file = fopen(path, "w");

	if( NULL == file )
		return 1;

	// for each y:
	for(int y = 0; y < size_y; y++)
	{
		// get label(y)
		int label = *dwt_util_addr_coeff_const_i(
			cls_ptr,
			y, // y
			0, // x
			cls_stride_x,
			cls_stride_y
		);

		// put label(y)
		fprintf(file, "%i", label);

		// for each x:
		for(int x = 0; x < size_x; x++)
		{
			// get value(y,x)
			float coeff = *dwt_util_addr_coeff_const_s(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			);

			// put value(y,x)
			fprintf(file, " %i:%f", x+1, coeff);
		}

		// line end
		fprintf(file, "\n");
	}

	// fclose
	fclose(file);

	return 0;
}

int dwt_util_find_min_max_s(
	const void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float *min,
	float *max
)
{
	assert( ptr );
	assert( size_x > 0 && size_y > 0 );

	*min = *dwt_util_addr_coeff_const_s(
		ptr,
		0,
		0,
		stride_x,
		stride_y
	);

	*max = *dwt_util_addr_coeff_const_s(
		ptr,
		0,
		0,
		stride_x,
		stride_y
	);

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float coeff = *dwt_util_addr_coeff_const_s(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			);

			if( coeff > *max )
				*max = coeff;

			if( coeff < *min )
				*min = coeff;
		}
	}

	return 0;
}

int dwt_util_find_min_max_i(
	const void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int *min,
	int *max
)
{
	assert( ptr );
	assert( size_x > 0 && size_y > 0 );

	*min = *dwt_util_addr_coeff_const_i(
		ptr,
		0,
		0,
		stride_x,
		stride_y
	);

	*max = *dwt_util_addr_coeff_const_i(
		ptr,
		0,
		0,
		stride_x,
		stride_y
	);

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			float coeff = *dwt_util_addr_coeff_const_i(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			);

			if( coeff > *max )
				*max = coeff;

			if( coeff < *min )
				*min = coeff;
		}
	}

	return 0;
}

int dwt_util_shift_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float a
)
{
	assert( ptr );

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			*dwt_util_addr_coeff_s(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			) += a;
		}
	}

	return 0;
}

int dwt_util_scale_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float a
)
{
	assert( ptr );

	for(int y = 0; y < size_y; y++)
	{
		for(int x = 0; x < size_x; x++)
		{
			*dwt_util_addr_coeff_s(
				ptr,
				y,
				x,
				stride_x,
				stride_y
			) *= a;
		}
	}

	return 0;
}

int dwt_util_scale21_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	float lo,
	float hi
)
{
	assert( ptr );
	assert( hi > lo );

	float target_diff = hi - lo;

	// for each y:
	for(int y = 0; y < size_y; y++)
	{
		float min, max;

		// find min, max (on row)
		dwt_util_find_min_max_s(
			dwt_util_addr_coeff_const_s(
				ptr,
				y,
				0,
				stride_x,
				stride_y
			), // (y,0)
			size_x, // size_x
			1, // 1
			stride_x,
			stride_y,
			&min,
			&max
		);

		float diff = max - min;

		if( max == min )
		{
			dwt_util_log(LOG_WARN, "Cannot scale row y=%i (min=max=%f)\n", y, min);
			continue;
		}

		//dwt_util_log(LOG_DBG, "scale(y=%i) <%f..%f> => <%f..%f>\n", y, min, max, lo, hi);

		// shift min => lo (on row)
		dwt_util_shift_s(
			dwt_util_addr_coeff_s(
				ptr,
				y,
				0,
				stride_x,
				stride_y
			), // (y,0)
			size_x, // size_x
			1, // 1
			stride_x,
			stride_y,
			(lo-min)
		);

		// scale to hi (on row)
		dwt_util_scale_s(
			dwt_util_addr_coeff_s(
				ptr,
				y,
				0,
				stride_x,
				stride_y
			), // (y,0)
			size_x, // size_x
			1, // 1
			stride_x,
			stride_y,
			(target_diff/diff)
		);

		// check
		dwt_util_find_min_max_s(
			dwt_util_addr_coeff_const_s(
				ptr,
				y,
				0,
				stride_x,
				stride_y
			), // (y,0)
			size_x, // size_x
			1, // 1
			stride_x,
			stride_y,
			&min,
			&max
		);

		//dwt_util_log(LOG_DBG, "scale(y=%i) <%f..%f>\n", y, min, max);
	}

	return 0;
}

int dwt_util_displace1_s(
	void *ptr,
	int size_x,
	int stride_y,
	int displ_x
)
{
	// assert( ptr ); // not needed

	if( !displ_x )
		return 0;

	if( displ_x > 0 )
	{
		for(int x = 0; x < size_x; x++)
		{
			int src_x = saturate_i(x + displ_x, 0, size_x-1);

			*dwt_util_addr_coeff_s(
				ptr,
				0, // y
				x, // x
				0, // stride x
				stride_y // stride y
			) = *dwt_util_addr_coeff_const_s(
				ptr,
				0, // y
				src_x, // x
				0, // stride x
				stride_y // stride y
			);
		}
	}
	else // < 0
	{
		for(int x = size_x-1; x >= 0; x--)
		{
			int src_x = saturate_i(x + displ_x, 0, size_x-1);

			*dwt_util_addr_coeff_s(
				ptr,
				0, // y
				x, // x
				0, // stride x
				stride_y // stride y
			) = *dwt_util_addr_coeff_const_s(
				ptr,
				0, // y
				src_x, // x
				0, // stride x
				stride_y // stride y
			);
		}
	}

	return 0;
}

int dwt_util_displace1_zero_s(
	void *ptr,
	int size_x,
	int stride_y,
	int displ_x
)
{
	// assert( ptr ); // not needed

	if( !displ_x )
		return 0;

	if( displ_x > 0 )
	{
		for(int x = 0; x < size_x; x++)
		{
			int src_x = saturate_i(x + displ_x, 0, size_x-1);

			*dwt_util_addr_coeff_s(
				ptr,
				0, // y
				x, // x
				0, // stride x
				stride_y // stride y
			) = ( x + displ_x != src_x ) ? 0.0f :
			*dwt_util_addr_coeff_const_s(
				ptr,
				0, // y
				src_x, // x
				0, // stride x
				stride_y // stride y
			);
		}
	}
	else // < 0
	{
		for(int x = size_x-1; x >= 0; x--)
		{
			int src_x = saturate_i(x + displ_x, 0, size_x-1);

			*dwt_util_addr_coeff_s(
				ptr,
				0, // y
				x, // x
				0, // stride x
				stride_y // stride y
			) = ( x + displ_x != src_x ) ? 0.0f :
			*dwt_util_addr_coeff_const_s(
				ptr,
				0, // y
				src_x, // x
				0, // stride x
				stride_y // stride y
			);
		}
	}

	return 0;
}

int dwt_util_get_center1_s(
	const void *ptr,
	int size_x,
	int stride_y
)
{
	// assert( ptr ); // not needed
	assert( size_x > 0 );

	// TODO: as an argument
	const int p = 10;

	// total "norm"
	float norm = dwt_util_band_lpnorm_s(
		ptr,
		0, // stride x
		stride_y, // stride y
		size_x, // size x
		1, // size y
		p // p
	);

	if( 0.0f == norm )
	{
		dwt_util_log(LOG_WARN, "Cannot get a center of signal due to its zero norm!\n");
		return size_x/2;
	}

	// the value of p-norm raised to the power of p
	norm = powf(norm, p);

	float half = norm / 2;

	// indexes of center borders
	int lidx = -1;
	int ridx = -1;

	// "norm" accumulator
	float sum;

	sum = 0.0f;

	for(int x = 0; x < size_x; x++)
	{
		// get coeff
		float coeff = *dwt_util_addr_coeff_const_s(
			ptr,
			0, // y
			x, // x
			0, // stride x
			stride_y // stride y
		);

		// accumulate "norm"
		sum += powf(fabsf(coeff), p);

		if( sum > half )
		{
			ridx = x - 1;
			break;
		}
	}

	sum = 0.0f;

	for(int x = size_x-1; x >= 0; x--)
	{
		// get coeff
		float coeff = *dwt_util_addr_coeff_const_s(
			ptr,
			0, // y
			x, // x
			0, // stride x
			stride_y // stride y
		);

		// accumulate "norm"
		sum += powf(fabsf(coeff), p);

		if( sum > half )
		{
			lidx = x + 1;
			break;
		}
	}

	if( -1 == lidx || -1 == ridx )
	{
		dwt_util_log(LOG_WARN, "Cannot found center indexes! lidx=%i ridx=%i norm=%f half=%f size_x=%i\n", lidx, ridx, norm, half, size_x);

		if( -1 == lidx && -1 == ridx )
			return size_x / 2;

		if( -1 == lidx )
			lidx = ridx;
		else
			ridx = lidx;
	}

	int center = (lidx + ridx) / 2;

	// dwt_util_log(LOG_DBG, "center at %i (%i %i)\n", center, lidx, ridx);

	return center;
}

int dwt_util_center1_s(
	void *ptr,
	int size_x,
	int stride_y,
	int max_iters
)
{
	// iterations
	for(int i = 0; i < max_iters; i++)
	{
		int center = dwt_util_get_center1_s(
			ptr,
			size_x,
			stride_y
		);

		int exp_center = size_x / 2;
		int displ = exp_center - center;

#if 0
		dwt_util_log(LOG_DBG, "i=%i: real_center=%i expected_center=%i displacement=%i\n", i, center, exp_center, displ);
#endif

		if( !displ )
			break;

		dwt_util_displace1_zero_s(
			ptr,
			size_x,
			stride_y,
			-displ
		);
	}

	return 0;
}

int dwt_util_center21_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int max_iters
)
{
	for(int y = 0; y < size_y; y++)
	{
#if 0
		dwt_util_log(LOG_DBG, "Centering vector y=%i\n", y);
#endif
		dwt_util_center1_s(
			dwt_util_addr_coeff_s(
				ptr,
				y, // y
				0, // x
				stride_x,
				stride_y
			),
			size_x,
			stride_y,
			max_iters
		);
	}

	return 0;
}

void dwt_util_shift21_med_s(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y
)
{
	for(int y = 0; y < size_y; y++)
	{
		// single transformed vector
		void *src = dwt_util_addr_coeff_s(ptr, y, 0, stride_x, stride_y);
		int src_x = size_x;
		int src_y = 1;

		float med = dwt_util_band_med_s(
			src,
			stride_x,
			stride_y,
			src_x,
			src_y
		);

		//dwt_util_log(LOG_DBG, "shift21_med: y=%i med=%f\n", y, med);

		dwt_util_shift_s(
			src,
			src_x,
			src_y,
			stride_x,
			stride_y,
			-med
		);
	}
}

void *dwt_util_viewport(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int offset_x,
	int offset_y
)
{
	assert( offset_x < size_x && offset_y < size_y );

	return dwt_util_addr_coeff_s(ptr, offset_y, offset_x, stride_x, stride_y);
}

void *dwt_util_crop21(
	void *ptr,
	int size_x,
	int size_y,
	int stride_x,
	int stride_y,
	int len_x
)
{
	UNUSED(size_y);

	assert( len_x > 0 );

	assert( len_x < size_x );

	int center_x = size_x / 2;

	int offset_x = center_x - len_x/2;

	assert( offset_x + len_x <= size_x );

	//dwt_util_log(LOG_DBG, "crop: offset_x=%i len_x=%i\n", offset_x, len_x);

	return dwt_util_addr_coeff_s(ptr, 0, offset_x, stride_x, stride_y);
}

void dwt_util_unit_vec_s(
	float *addr,
	int size,
	int offset
)
{
	dwt_util_zero_vec_s(addr, size);

	addr[size/2+offset] = 1.f;
}

struct cbuff
{
	char *ptr;
	size_t size;
	char *pos;
};

static
struct cbuff *cbuff_create()
{
	struct cbuff *cbuff = (struct cbuff *)malloc(sizeof(struct cbuff));

	cbuff->size = 1;
	cbuff->ptr = malloc(1);
	if(!cbuff->ptr)
		dwt_util_error("unable to allocate memory!\n");

	cbuff->pos = cbuff->ptr;

	return cbuff;
}

static
void cbuff_destroy(struct cbuff *cbuff)
{
	free(cbuff->ptr);
	free(cbuff);
}

static
void cbuff_reset(struct cbuff *cbuff)
{
	cbuff->pos = cbuff->ptr;
}

static
int cbuff_offset(struct cbuff *cbuff)
{
	return cbuff->pos - cbuff->ptr;
}

static
int cbuff_avail(struct cbuff *cbuff)
{
	return cbuff->size - cbuff_offset(cbuff);
}

static
void cbuff_realloc(struct cbuff *cbuff)
{
	cbuff->size <<= 1;

	if(!cbuff->size)
		dwt_util_error("size_t is too small!\n");

	int offset = cbuff_offset(cbuff);

	cbuff->ptr = realloc(cbuff->ptr, cbuff->size);

	if(!cbuff->ptr)
		dwt_util_error("unable to allocate %i bytes!\n", cbuff->size);

	cbuff->pos = cbuff->ptr + offset;
}

static
char *cbuff_cstr(struct cbuff *cbuff)
{
	return cbuff->ptr;
}

static
void cbuff_sprintf(struct cbuff *cbuff, const char *format, ...)
{
	va_list ap;

	while(1)
	{
		va_start(ap, format);
		int n = vsnprintf(cbuff->pos, cbuff_avail(cbuff), format, ap);
		va_end(ap);

		if( n < 0 )
			dwt_util_error("vsnprintf returned negative value!\n");

		if( n >= cbuff_avail(cbuff) )
			cbuff_realloc(cbuff);
		else
		{
			cbuff->pos += n;
			break;
		}
	}
}

const char *dwt_util_str_vec_s(
	const float *vec,
	int size
)
{
	static struct cbuff *cbuff = NULL;
	if(!cbuff)
		cbuff = cbuff_create();

	cbuff_reset(cbuff);

	cbuff_sprintf(cbuff, "[ ");
	for(int i = 0; i < size; i++)
		cbuff_sprintf(cbuff, "%+11.8f ", vec[i]);
	cbuff_sprintf(cbuff, "] ");

	cbuff_sprintf(cbuff, "(%i)", size);

	return cbuff_cstr(cbuff);
}

int dwt_util_save_sym_to_pgm_s(
	const char *path,
	float max_value,
	const void *ptr,
	int stride_x,
	int stride_y,
	int size_x,
	int size_y
)
{
	assert( max_value > 0.f );

	// alloc "clone"
	void *clone = dwt_util_alloc_image2(stride_x, stride_y, size_x, size_y);

	// copy "ptr" into "clone"
	dwt_util_copy_s(ptr, clone, stride_x, stride_y, size_x, size_y);

	// shift "clone"
	dwt_util_shift_s(
		clone,
		size_x,
		size_y,
		stride_x,
		stride_y,
		+max_value
	);

	// save "clone"
	dwt_util_save_to_pgm_s(
		path,
		2.f*max_value,
		clone,
		stride_x,
		stride_y,
		size_x,
		size_y
	);

	// free "clone"
	dwt_util_free_image(&clone);

	return 0;
}
