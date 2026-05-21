#include "convolutionParallel.h"
#include "bmpHandler.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static void *threadConvolution(void *args) {
  struct ThreadArgs *targs = (struct ThreadArgs *)args;

  for (int y = targs->y_start; y < targs->y_end; y++) {
    for (int x = targs->x_start; x < targs->x_end; x++) {
      double brightness = 0.0;
      for (int filterY = 0; filterY < targs->filter->height; filterY++) {
        for (int filterX = 0; filterX < targs->filter->width; filterX++) {
          int imageX =
              (x - (targs->filter->width / 2) + filterX + targs->width) %
              targs->width;
          int imageY =
              (y - (targs->filter->height / 2) + filterY + targs->height) %
              targs->height;
          brightness +=
              targs->tmp[(imageY * targs->stride) + imageX] *
              targs->filter->kernel[(filterY * targs->filter->width) + filterX];
        }
      }
      targs->image[(y * targs->stride) + x] = (uint8_t)fmin(
          fmax((targs->filter->factor * brightness) + targs->filter->bias, 0),
          255);
    }
  }
  return NULL;
}

static void *threadConvolutionPixel(void *args) {
  struct ThreadArgs *targs = (struct ThreadArgs *)args;
  int total = targs->width * targs->height;
  int id;
  while ((id = atomic_fetch_add(targs->counter, 1)) < total) {
    int x = id % targs->width;
    int y = id / targs->width;
    double brightness = 0.0;
    for (int filterY = 0; filterY < targs->filter->height; filterY++) {
      for (int filterX = 0; filterX < targs->filter->width; filterX++) {
        int imageX =
            (x - (targs->filter->width / 2) + filterX + targs->width) %
            targs->width;
        int imageY =
            (y - (targs->filter->height / 2) + filterY + targs->height) %
            targs->height;
        brightness +=
            targs->tmp[(imageY * targs->stride) + imageX] *
            targs->filter->kernel[(filterY * targs->filter->width) + filterX];
      }
    }
    targs->image[(y * targs->stride) + x] = (uint8_t)fmin(
        fmax((targs->filter->factor * brightness) + targs->filter->bias, 0),
        255);
  }
  return NULL;
}

void applyConvolutionParallel(struct BmpImage *image,
                              const struct Filter *filter, int numThreads,
                              enum TypeParallel type) {
  int stride = bmpStride(image);
  int w = (int)image->info.biWidth;
  int h = image->info.biHeight < 0 ? -image->info.biHeight
                                   : (int)image->info.biHeight;
  uint32_t dataSize = (uint32_t)(stride * h);

  uint8_t *tmp = malloc(dataSize);
  if (!tmp) {
    return;
  }
  memcpy(tmp, image->data, dataSize);

  atomic_int counter = 0;
  pthread_t *threads = malloc(numThreads * sizeof(pthread_t));
  struct ThreadArgs *targs = malloc(numThreads * sizeof(struct ThreadArgs));
  if (!threads || !targs) {
    free(threads);
    free(targs);
    free(tmp);
    return;
  }

  for (int i = 0; i < numThreads; i++) {
    targs[i].width = w;
    targs[i].height = h;
    targs[i].stride = stride;
    targs[i].image = image->data;
    targs[i].filter = filter;
    targs[i].tmp = tmp;
    if (type == horizontal) {
      targs[i].x_start = 0;
      targs[i].x_end = w;
      targs[i].y_start = i * (h / numThreads);
      targs[i].y_end = (i == numThreads - 1) ? h : (i + 1) * (h / numThreads);
      pthread_create(&threads[i], NULL, threadConvolution, &targs[i]);
    } else if (type == vertical) {
      targs[i].x_start = i * (w / numThreads);
      targs[i].x_end = (i == numThreads - 1) ? w : (i + 1) * (w / numThreads);
      targs[i].y_start = 0;
      targs[i].y_end = h;
      pthread_create(&threads[i], NULL, threadConvolution, &targs[i]);
    } else if (type == block) {
      int blocksPerRow = (int)sqrt(numThreads);
      int blockX = i % blocksPerRow;
      int blockY = i / blocksPerRow;
      targs[i].x_start = blockX * (w / blocksPerRow);
      targs[i].x_end =
          (blockX == blocksPerRow - 1) ? w : (blockX + 1) * (w / blocksPerRow);
      targs[i].y_start = blockY * (h / blocksPerRow);
      targs[i].y_end =
          (blockY == blocksPerRow - 1) ? h : (blockY + 1) * (h / blocksPerRow);
      pthread_create(&threads[i], NULL, threadConvolution, &targs[i]);
    } else {
      targs[i].counter = &counter;
      pthread_create(&threads[i], NULL, threadConvolutionPixel, &targs[i]);
    }
  }
  for (int i = 0; i < numThreads; i++) {
    pthread_join(threads[i], NULL);
  }

  free(threads);
  free(targs);
  free(tmp);
}
