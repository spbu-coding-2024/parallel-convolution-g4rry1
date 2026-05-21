#pragma once

#include <pthread.h>

struct PipelineTask;

typedef struct {
  struct PipelineTask **buffer;
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

int bq_push(BoundedQueue *q, struct PipelineTask *item);

int bq_pop(BoundedQueue *q, struct PipelineTask **out);

void bq_close(BoundedQueue *q);

void bq_destroy(BoundedQueue *q);
