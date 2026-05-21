#pragma once

#include "bmpStruct.h"
#include "convolutionParallel.h"
#include "filter.h"

struct PipelineTask {
  struct BmpImage *image;
  char *outpath;
};

void runPipeline(const char *indir, const char *outdir,
                 const struct Filter *filter, int workers, int queueSize,
                 int parallel, int numThreads, enum TypeParallel parallelType);
