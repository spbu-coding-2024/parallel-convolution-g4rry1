#include "convolution.h"
#include "bmpHandler.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void applyConvolution(struct BmpImage *image, const struct Filter *filter) {
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

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      double brightness = 0.0;
      for (int filterY = 0; filterY < filter->height; filterY++) {
        for (int filterX = 0; filterX < filter->width; filterX++) {
          int imageX = (x - (filter->width / 2) + filterX + w) % w;
          int imageY = (y - (filter->height / 2) + filterY + h) % h;
          brightness += tmp[(imageY * stride) + imageX] *
                        filter->kernel[(filterY * filter->width) + filterX];
        }
      }
      image->data[(y * stride) + x] = (uint8_t)fmin(
          fmax((filter->factor * brightness) + filter->bias, 0), 255);
    }
  }

  free(tmp);
}
