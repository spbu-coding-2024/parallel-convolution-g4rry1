#include "bmpHandler.h"
#include "convolution.h"
#include "convolutionParallel.h"
#include "filter.h"
#include "pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void usage(const char *prog) {
  fprintf(
      stderr,
      "Usage:\n"
      "  Single image: %s <input> <output> <kernel> [options]\n"
      "  Pipeline:     %s <kernel> --pipeline[=N] --indir=<dir> "
      "--outdir=<dir>\n"
      "\n"
      "Built-in kernels: identity, edgeDetection, sharpen, boxBlur,\n"
      "                  gaussianBlur, motionBlur, emboss,\n"
      "                  edgeEnhancement, meanFilter\n"
      "\n"
      "  --parallel=type  parallel execution per image\n"
      "                   types: horizontal, vertical, block, pixel\n"
      "  --threads=N      number of threads (default: all CPUs)\n"
      "  --repeat=N       run filter N times and report mean (default: 1)\n"
      "  --kernel=file    load kernel from file instead of built-in name\n"
      "\n"
      "  --pipeline[=N]   pipeline mode: reader -> N workers -> writer\n"
      "                   N defaults to 1\n"
      "  --queue-size=N   bounded queue capacity (default: 8)\n"
      "  --indir=path     input directory (all .bmp files) for pipeline mode\n"
      "  --outdir=path    output directory for pipeline mode\n"
      "\n"
      "Kernel file format (first line: width height factor bias):\n"
      "  3 3 0.0625 0.0\n"
      "  1 2 1 2 4 2 1 2 1\n",
      prog, prog);
}

static const struct Filter *resolveNamedKernel(const char *name) {
  enum convolutionType type;
  if (strcmp(name, "identity") == 0)
    type = identity;
  else if (strcmp(name, "edgeDetection") == 0)
    type = edgeDetection;
  else if (strcmp(name, "sharpen") == 0)
    type = sharpen;
  else if (strcmp(name, "boxBlur") == 0)
    type = boxBlur;
  else if (strcmp(name, "gaussianBlur") == 0)
    type = gaussianBlur;
  else if (strcmp(name, "motionBlur") == 0)
    type = motionBlur;
  else if (strcmp(name, "emboss") == 0)
    type = emboss;
  else if (strcmp(name, "edgeEnhancement") == 0)
    type = edgeEnhancement;
  else if (strcmp(name, "meanFilter") == 0)
    type = meanFilter;
  else {
    fprintf(stderr, "Unknown kernel: %s\n", name);
    return NULL;
  }
  return getFilter(type);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }

  const char *kernelName = NULL;
  const char *kernelFile = NULL;
  const char *outdir = NULL;
  const char *indir = NULL;
  int parallel = 0;
  int numThreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
  int repeat = 1;
  int pipeline = 0;
  int pipelineWorkers = 1;
  int queueSize = 8;
  enum TypeParallel parallelType = horizontal;

  /* positional args: in single mode [input, output, kernel],
                      in pipeline mode [kernel, file1, file2, ...] */
  const char *positional[argc];
  int npos = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--parallel") == 0) {
      parallel = 1;
    } else if (strncmp(argv[i], "--parallel=", 11) == 0) {
      parallel = 1;
      if (strcmp(argv[i] + 11, "horizontal") == 0)
        parallelType = horizontal;
      else if (strcmp(argv[i] + 11, "vertical") == 0)
        parallelType = vertical;
      else if (strcmp(argv[i] + 11, "block") == 0)
        parallelType = block;
      else if (strcmp(argv[i] + 11, "pixel") == 0)
        parallelType = pixel;
      else {
        fprintf(stderr, "Unknown parallel type: %s\n", argv[i] + 11);
        usage(argv[0]);
        return 1;
      }
    } else if (strncmp(argv[i], "--threads=", 10) == 0) {
      numThreads = atoi(argv[i] + 10);
      if (numThreads < 1) {
        fprintf(stderr, "Invalid thread count: %s\n", argv[i] + 10);
        return 1;
      }
    } else if (strncmp(argv[i], "--repeat=", 9) == 0) {
      repeat = atoi(argv[i] + 9);
      if (repeat < 1)
        repeat = 1;
    } else if (strncmp(argv[i], "--kernel=", 9) == 0) {
      kernelFile = argv[i] + 9;
    } else if (strcmp(argv[i], "--pipeline") == 0) {
      pipeline = 1;
    } else if (strncmp(argv[i], "--pipeline=", 11) == 0) {
      pipeline = 1;
      pipelineWorkers = atoi(argv[i] + 11);
      if (pipelineWorkers < 1) {
        fprintf(stderr, "Invalid worker count: %s\n", argv[i] + 11);
        return 1;
      }
    } else if (strncmp(argv[i], "--queue-size=", 13) == 0) {
      queueSize = atoi(argv[i] + 13);
      if (queueSize < 1) {
        fprintf(stderr, "Invalid queue size: %s\n", argv[i] + 13);
        return 1;
      }
    } else if (strncmp(argv[i], "--outdir=", 9) == 0) {
      outdir = argv[i] + 9;
    } else if (strncmp(argv[i], "--indir=", 8) == 0) {
      indir = argv[i] + 8;
    } else if (argv[i][0] != '-') {
      positional[npos++] = argv[i];
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  if (kernelFile && kernelName) {
    fprintf(stderr, "Cannot use both a kernel name and --kernel=file\n");
    return 1;
  }

  const char *inputImage = NULL;
  const char *outputImage = NULL;

  if (pipeline) {
    if (!outdir) {
      fprintf(stderr, "--outdir is required in pipeline mode\n");
      return 1;
    }
    if (!indir) {
      fprintf(stderr, "--indir is required in pipeline mode\n");
      return 1;
    }
    if (!kernelFile) {
      if (npos < 1) {
        fprintf(stderr, "Pipeline mode requires a kernel name\n");
        usage(argv[0]);
        return 1;
      }
      kernelName = positional[0];
    }
  } else {
    int kernelPos = kernelFile ? 2 : 3;
    if (npos < kernelPos) {
      fprintf(stderr, "Single mode requires: input output kernel\n");
      usage(argv[0]);
      return 1;
    }
    inputImage = positional[0];
    outputImage = positional[1];
    if (!kernelFile)
      kernelName = positional[2];
  }

  if (!kernelName && !kernelFile) {
    fprintf(stderr, "No kernel specified\n");
    usage(argv[0]);
    return 1;
  }

  struct Filter customFilter = {0};
  const struct Filter *filter;

  if (kernelFile) {
    if (loadFilterFromFile(kernelFile, &customFilter) != 0)
      return 1;
    filter = &customFilter;
  } else {
    filter = resolveNamedKernel(kernelName);
    if (!filter)
      return 1;
  }

  if (pipeline) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    runPipeline(indir, outdir, filter, pipelineWorkers, queueSize, parallel,
                numThreads, parallelType);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms =
        (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("TIME_MS: %.3f\n", elapsed_ms);
    free(customFilter.kernel);
    return 0;
  }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  struct BmpImage image;
  if (readBmp(inputImage, &image) != 0) {
    free(customFilter.kernel);
    return 1;
  }

  for (int r = 0; r < repeat; r++) {
    if (parallel)
      applyConvolutionParallel(&image, filter, numThreads, parallelType);
    else
      applyConvolution(&image, filter);
  }

  int ret = 0;
  if (writeBmp(outputImage, &image) != 0)
    ret = 1;

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed_ms =
      ((t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6) /
      repeat;
  printf("TIME_MS: %.3f\n", elapsed_ms);

  free(image.palette);
  free(image.data);
  free(customFilter.kernel);
  return ret;
}
