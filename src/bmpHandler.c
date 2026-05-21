#include "bmpHandler.h"
#include <stdio.h>
#include <stdlib.h>

int bmpStride(const struct BmpImage *image) {
  return ((int)image->info.biWidth + 3) / 4 * 4;
}

static uint32_t bmpClrCount(const struct BmpImage *image) {
  if (image->info.biClrUsed) {
    return image->info.biClrUsed;
  }
  return 1U << image->info.biBitCount;
}

static uint32_t bmpDataSize(const struct BmpImage *image) {
  if (image->info.biSizeImage) {
    return image->info.biSizeImage;
  }
  int height =
      image->info.biHeight < 0 ? -image->info.biHeight : image->info.biHeight;
  return (uint32_t)(bmpStride(image) * height);
}

int readBmp(const char *filename, struct BmpImage *image) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Could not open file");
    return 1;
  }

  fread(&image->header, sizeof(struct BmpHeader), 1, file);
  fread(&image->info, sizeof(struct BmpInfoHeader), 1, file);

  uint32_t clrCount = bmpClrCount(image);
  image->palette = malloc(clrCount * sizeof(uint32_t));
  fread(image->palette, sizeof(uint32_t), clrCount, file);

  uint32_t dataSize = bmpDataSize(image);
  image->data = malloc(dataSize);
  fseek(file, image->header.bfOffBits, SEEK_SET);
  fread(image->data, dataSize, 1, file);

  fclose(file);
  return 0;
}

int writeBmp(const char *filename, const struct BmpImage *image) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    perror("Could not open file");
    return 1;
  }

  fwrite(&image->header, sizeof(struct BmpHeader), 1, file);
  fwrite(&image->info, sizeof(struct BmpInfoHeader), 1, file);
  fwrite(image->palette, sizeof(uint32_t), bmpClrCount(image), file);
  fseek(file, image->header.bfOffBits, SEEK_SET);
  fwrite(image->data, bmpDataSize(image), 1, file);

  fclose(file);
  return 0;
}
