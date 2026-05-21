#include "bounded_queue.h"
#include <stdlib.h>

int bq_init(BoundedQueue *q, int capacity) {
  q->buffer = malloc(capacity * sizeof(struct PipelineTask *));
  if (!q->buffer) {
    return -1;
  }
  q->capacity = capacity;
  q->head = 0;
  q->tail = 0;
  q->count = 0;
  q->closed = 0;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
  return 0;
}

int bq_push(BoundedQueue *q, struct PipelineTask *item) {
  pthread_mutex_lock(&q->mutex);

  while (q->count == q->capacity && !q->closed) {
    pthread_cond_wait(&q->not_full, &q->mutex);
  }

  if (q->closed) {
    pthread_mutex_unlock(&q->mutex);
    return -1;
  }

  q->buffer[q->tail] = item;
  q->tail = (q->tail + 1) % q->capacity;
  q->count++;

  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mutex);
  return 0;
}

int bq_pop(BoundedQueue *q, struct PipelineTask **out) {
  pthread_mutex_lock(&q->mutex);

  while (q->count == 0 && !q->closed) {
    pthread_cond_wait(&q->not_empty, &q->mutex);
  }

  if (q->count == 0) {
    pthread_mutex_unlock(&q->mutex);
    return -1;
  }

  *out = q->buffer[q->head];
  q->head = (q->head + 1) % q->capacity;
  q->count--;

  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->mutex);
  return 0;
}

void bq_close(BoundedQueue *q) {
  pthread_mutex_lock(&q->mutex);
  q->closed = 1;
  pthread_cond_broadcast(&q->not_empty);
  pthread_cond_broadcast(&q->not_full);
  pthread_mutex_unlock(&q->mutex);
}

void bq_destroy(BoundedQueue *q) {
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->not_empty);
  pthread_cond_destroy(&q->not_full);
  free(q->buffer);
  q->buffer = NULL;
}
