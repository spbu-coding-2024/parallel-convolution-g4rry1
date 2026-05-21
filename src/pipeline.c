#include "pipeline.h"
#include "bmpHandler.h"
#include "bounded_queue.h"
#include "convolution.h"
#include "convolutionParallel.h"
#include <dirent.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ReaderArgs {
  BoundedQueue *q1;
  const char *indir;
  const char *outdir;
};

struct WorkerArgs {
  BoundedQueue *q1;
  BoundedQueue *q2;
  const struct Filter *filter;
  int parallel;
  int numThreads;
  enum TypeParallel parallelType;
  atomic_int *activeWorkers;
};

struct WriterArgs {
  BoundedQueue *q2;
};

static void freeTask(struct PipelineTask *task) {
  free(task->image->palette);
  free(task->image->data);
  free(task->image);
  free(task->outpath);
  free(task);
}

static int hasBmpSuffix(const char *name) {
  size_t len = strlen(name);
  return len >= 4 && strcasecmp(name + len - 4, ".bmp") == 0;
}

static void *readerThread(void *arg) {
  struct ReaderArgs *args = arg;

  DIR *dir = opendir(args->indir);
  if (!dir) {
    perror(args->indir);
    bq_close(args->q1);
    return NULL;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (!hasBmpSuffix(entry->d_name)) {
      continue;
    }

    char inpath[PATH_MAX];
    snprintf(inpath, sizeof(inpath), "%s/%s", args->indir, entry->d_name);

    struct BmpImage *img = malloc(sizeof(struct BmpImage));
    if (readBmp(inpath, img) != 0) {
      free(img);
      continue;
    }

    char *outpath = malloc(PATH_MAX);
    snprintf(outpath, PATH_MAX, "%s/%s", args->outdir, entry->d_name);

    struct PipelineTask *task = malloc(sizeof(struct PipelineTask));
    task->image = img;
    task->outpath = outpath;

    if (bq_push(args->q1, task) != 0) {
      freeTask(task);
      break;
    }
  }

  closedir(dir);
  bq_close(args->q1);
  return NULL;
}

static void *workerThread(void *arg) {
  struct WorkerArgs *args = arg;
  struct PipelineTask *task;

  while (bq_pop(args->q1, &task) == 0) {
    if (args->parallel) {
      applyConvolutionParallel(task->image, args->filter, args->numThreads,
                               args->parallelType);
    } else {
      applyConvolution(task->image, args->filter);
    }

    if (bq_push(args->q2, task) != 0) {
      freeTask(task);
    }
  }

  if (atomic_fetch_sub(args->activeWorkers, 1) == 1) {
    bq_close(args->q2);
  }

  return NULL;
}

static void *writerThread(void *arg) {
  struct WriterArgs *args = arg;
  struct PipelineTask *task;

  while (bq_pop(args->q2, &task) == 0) {
    if (writeBmp(task->outpath, task->image) != 0) {
      fprintf(stderr, "Failed to write: %s\n", task->outpath);
    }
    freeTask(task);
  }

  return NULL;
}

void runPipeline(const char *indir, const char *outdir,
                 const struct Filter *filter, int workers, int queueSize,
                 int parallel, int numThreads, enum TypeParallel parallelType) {
  BoundedQueue q1;
  BoundedQueue q2;
  bq_init(&q1, queueSize);
  bq_init(&q2, queueSize);

  atomic_int activeWorkers = workers;

  pthread_t reader;
  struct ReaderArgs rargs = {&q1, indir, outdir};
  pthread_create(&reader, NULL, readerThread, &rargs);

  pthread_t *workerThreads = malloc(workers * sizeof(pthread_t));
  struct WorkerArgs *wargs = malloc(workers * sizeof(struct WorkerArgs));
  for (int i = 0; i < workers; i++) {
    wargs[i] = (struct WorkerArgs){
        &q1, &q2, filter, parallel, numThreads, parallelType, &activeWorkers};
    pthread_create(&workerThreads[i], NULL, workerThread, &wargs[i]);
  }

  pthread_t writer;
  struct WriterArgs wrargs = {&q2};
  pthread_create(&writer, NULL, writerThread, &wrargs);

  pthread_join(reader, NULL);
  for (int i = 0; i < workers; i++) {
    pthread_join(workerThreads[i], NULL);
  }
  pthread_join(writer, NULL);

  free(workerThreads);
  free(wargs);
  bq_destroy(&q1);
  bq_destroy(&q2);
}
