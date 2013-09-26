/**
 * @brief signal_t infrastructure
 */

#ifndef SIGNAL_H
#define SIGNAL_H

struct signal_t;

struct signal_const_t;

typedef struct signal_t signal_t;
typedef struct signal_const_t signal_const_t;

signal_t *signal_create(
	void *ptr,
	int stride,
	unsigned int size,
	unsigned int center
);

signal_const_t *signal_const_create(
	const void *ptr,
	int stride,
	unsigned int size,
	unsigned int center
);

void signal_destroy(
	signal_t *signal
);

void signal_const_destroy(
	signal_const_t *signal
);

float *signal_get_s(
	const signal_t *signal,
	int index
);

const float *signal_const_get_s(
	const signal_const_t *signal,
	int index
);

int signal_left(
	const signal_t *signal
);

int signal_const_left(
	const signal_const_t *signal
);

int signal_right(
	const signal_t *signal
);

int signal_const_right(
	const signal_const_t *signal
);

#endif
