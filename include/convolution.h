#pragma once

#include "bmpStruct.h"
#include "filter.h"

void applyConvolution(struct BmpImage *image, const struct Filter *filter);
