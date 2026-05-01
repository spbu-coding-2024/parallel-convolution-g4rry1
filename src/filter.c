#include "filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double kernelIdentity[] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

static double kernelBoxBlur[] = {0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0};

static double kernelGaussianBlur[] = {1.0, 2.0, 1.0, 2.0, 4.0,
                                      2.0, 1.0, 2.0, 1.0};

static double kernelMotionBlur[] = {
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

static double kernelEdgeDetection[] = {-1.0, -1.0, -1.0, -1.0, 8.0,
                                       -1.0, -1.0, -1.0, -1.0};

static double kernelSharpen[] = {-1.0, -1.0, -1.0, -1.0, 9.0,
                                 -1.0, -1.0, -1.0, -1.0};

static double kernelEdgeEnhancement[] = {1.0, 1.0, 1.0, 1.0, -7.0,
                                         1.0, 1.0, 1.0, 1.0};

static double kernelEmboss[] = {-1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0};

static double kernelMeanFilter[] = {1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0};

const struct Filter filterIdentity = {3, 3, 1.0, 0.0, kernelIdentity};
const struct Filter filterEdgeDetection = {3, 3, 1.0, 0.0, kernelEdgeDetection};
const struct Filter filterSharpen = {3, 3, 1.0, 0.0, kernelSharpen};
const struct Filter filterBoxBlur = {3, 3, 1.0, 0.0, kernelBoxBlur};
const struct Filter filterGaussianBlur = {3, 3, 1.0 / 16, 0.0,
                                          kernelGaussianBlur};
const struct Filter filterMotionBlur = {9, 9, 1.0 / 9, 0.0, kernelMotionBlur};
const struct Filter filterEmboss = {3, 3, 1.0, 128.0, kernelEmboss};
const struct Filter filterEdgeEnhancement = {3, 3, 1.0, 0.0,
                                             kernelEdgeEnhancement};
const struct Filter filterMeanFilter = {3, 3, 1.0 / 9, 0.0, kernelMeanFilter};

const struct Filter *getFilter(enum convolutionType type) {
  switch (type) {
  case identity:
    return &filterIdentity;
  case edgeDetection:
    return &filterEdgeDetection;
  case sharpen:
    return &filterSharpen;
  case boxBlur:
    return &filterBoxBlur;
  case gaussianBlur:
    return &filterGaussianBlur;
  case motionBlur:
    return &filterMotionBlur;
  case emboss:
    return &filterEmboss;
  case edgeEnhancement:
    return &filterEdgeEnhancement;
  case meanFilter:
    return &filterMeanFilter;
  }
  return NULL;
}

int loadFilterFromFile(const char *path, struct Filter *out) {
  FILE *f = fopen(path, "r");
  if (!f) {
    perror(path);
    return 1;
  }

  int width, height;
  double factor, bias;
  if (fscanf(f, "%d %d %lf %lf", &width, &height, &factor, &bias) != 4 ||
      width <= 0 || height <= 0) {
    fprintf(stderr, "%s: invalid header (expected: width height factor bias)\n",
            path);
    fclose(f);
    return 1;
  }

  int count = width * height;
  double *kernel = malloc(count * sizeof(double));
  for (int i = 0; i < count; i++) {
    if (fscanf(f, "%lf", &kernel[i]) != 1) {
      fprintf(stderr, "%s: expected %d kernel values\n", path, count);
      free(kernel);
      fclose(f);
      return 1;
    }
  }

  fclose(f);
  out->width = width;
  out->height = height;
  out->factor = factor;
  out->bias = bias;
  out->kernel = kernel;
  return 0;
}
