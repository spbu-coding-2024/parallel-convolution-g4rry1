#pragma once

#include "bmpStruct.h"
#include "filter.h"
#include <stdatomic.h>

struct ThreadArgs {
  const uint8_t *tmp;
  uint8_t *image;
  const struct Filter *filter;
  int stride;
  int width;
  int height;
  int y_start, y_end;
  int x_start, x_end;
  atomic_int *counter;
};

enum TypeParallel { horizontal, vertical, block, pixel };

void applyConvolutionParallel(struct BmpImage *image,
                              const struct Filter *filter, int numThreads,
                              enum TypeParallel type);
