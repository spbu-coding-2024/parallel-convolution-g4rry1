#include "bmpStruct.h"

int bmpStride(const struct BmpImage *image);
int readBmp(const char *filename, struct BmpImage *image);
int writeBmp(const char *filename, const struct BmpImage *image);