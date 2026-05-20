#pragma once

#include "bmpStruct.h"
#include <pthread.h>

typedef struct {
  struct BmpImage **buffer;
  int capacity;
  int head;
  int tail;
  int count;
  int closed;
  pthread_mutex_t mutex;
  pthread_cond_t not_empty;
  pthread_cond_t not_full;
} BoundedQueue;

int bq_init(BoundedQueue *q, int capacity);

int bq_push(BoundedQueue *q, struct BmpImage *item);

int bq_pop(BoundedQueue *q, struct BmpImage **out);

void bq_close(BoundedQueue *q);

void bq_destroy(BoundedQueue *q);
