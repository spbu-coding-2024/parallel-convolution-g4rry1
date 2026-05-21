#ifndef FILTER_H
#define FILTER_H

enum convolutionType {
  identity,
  edgeDetection,
  sharpen,
  boxBlur,
  gaussianBlur,
  motionBlur,
  emboss,
  edgeEnhancement,
  meanFilter
};

struct Filter {
  int width;
  int height;
  double factor;
  double bias;
  double *kernel;
};

extern const struct Filter filterIdentity;
extern const struct Filter filterEdgeDetection;
extern const struct Filter filterSharpen;
extern const struct Filter filterBoxBlur;
extern const struct Filter filterGaussianBlur;
extern const struct Filter filterMotionBlur;
extern const struct Filter filterEmboss;
extern const struct Filter filterEdgeEnhancement;
extern const struct Filter filterMeanFilter;

const struct Filter *getFilter(enum convolutionType type);
int loadFilterFromFile(const char *path, struct Filter *out);

#endif /* FILTER_H */
