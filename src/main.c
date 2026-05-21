#include "bmpHandler.h"
#include "convolution.h"
#include "convolutionParallel.h"
#include "filter.h"
#include "pipeline.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct Config {
  const char       *inputImage;
  const char       *outputImage;
  const char       *kernelName;
  const char       *kernelFile;
  const char       *outdir;
  const char       *indir;
  int               parallel;
  int               numThreads;
  int               repeat;
  int               pipeline;
  int               pipelineWorkers;
  int               queueSize;
  enum TypeParallel parallelType;
};

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
  if (strcmp(name, "identity") == 0) {
    type = identity;
  } else if (strcmp(name, "edgeDetection") == 0) {
    type = edgeDetection;
  } else if (strcmp(name, "sharpen") == 0) {
    type = sharpen;
  } else if (strcmp(name, "boxBlur") == 0) {
    type = boxBlur;
  } else if (strcmp(name, "gaussianBlur") == 0) {
    type = gaussianBlur;
  } else if (strcmp(name, "motionBlur") == 0) {
    type = motionBlur;
  } else if (strcmp(name, "emboss") == 0) {
    type = emboss;
  } else if (strcmp(name, "edgeEnhancement") == 0) {
    type = edgeEnhancement;
  } else if (strcmp(name, "meanFilter") == 0) {
    type = meanFilter;
  } else {
    fprintf(stderr, "Unknown kernel: %s\n", name);
    return NULL;
  }
  return getFilter(type);
}

static int parseParallelType(const char *str, enum TypeParallel *out) {
  if (strcmp(str, "horizontal") == 0) {
    *out = horizontal;
  } else if (strcmp(str, "vertical") == 0) {
    *out = vertical;
  } else if (strcmp(str, "block") == 0) {
    *out = block;
  } else if (strcmp(str, "pixel") == 0) {
    *out = pixel;
  } else {
    fprintf(stderr, "Unknown parallel type: %s\n", str);
    return 1;
  }
  return 0;
}

static int parseLong(const char *str, long *out) {
  char *end;
  errno = 0;
  *out = strtol(str, &end, 10);
  return (errno != 0 || end == str || *end != '\0') ? 1 : 0;
}

static int parseSingleOption(const char *arg, struct Config *cfg) {
  if (strcmp(arg, "--parallel") == 0) {
    cfg->parallel = 1;
  } else if (strncmp(arg, "--parallel=", 11) == 0) {
    cfg->parallel = 1;
    if (parseParallelType(arg + 11, &cfg->parallelType) != 0) {
      return 1;
    }
  } else if (strncmp(arg, "--threads=", 10) == 0) {
    long val;
    if (parseLong(arg + 10, &val) != 0 || val < 1) {
      fprintf(stderr, "Invalid thread count: %s\n", arg + 10);
      return 1;
    }
    cfg->numThreads = (int)val;
  } else if (strncmp(arg, "--repeat=", 9) == 0) {
    long val;
    cfg->repeat = (parseLong(arg + 9, &val) == 0 && val >= 1) ? (int)val : 1;
  } else if (strncmp(arg, "--kernel=", 9) == 0) {
    cfg->kernelFile = arg + 9;
  } else if (strcmp(arg, "--pipeline") == 0) {
    cfg->pipeline = 1;
  } else if (strncmp(arg, "--pipeline=", 11) == 0) {
    cfg->pipeline = 1;
    long val;
    if (parseLong(arg + 11, &val) != 0 || val < 1) {
      fprintf(stderr, "Invalid worker count: %s\n", arg + 11);
      return 1;
    }
    cfg->pipelineWorkers = (int)val;
  } else if (strncmp(arg, "--queue-size=", 13) == 0) {
    long val;
    if (parseLong(arg + 13, &val) != 0 || val < 1) {
      fprintf(stderr, "Invalid queue size: %s\n", arg + 13);
      return 1;
    }
    cfg->queueSize = (int)val;
  } else if (strncmp(arg, "--outdir=", 9) == 0) {
    cfg->outdir = arg + 9;
  } else if (strncmp(arg, "--indir=", 8) == 0) {
    cfg->indir = arg + 8;
  } else {
    fprintf(stderr, "Unknown option: %s\n", arg);
    return 1;
  }
  return 0;
}

static int parseArgs(int argc, char *argv[], struct Config *cfg,
                     const char **positional, int *npos) {
  *npos = 0;
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      if (parseSingleOption(argv[i], cfg) != 0) {
        return 1;
      }
    } else {
      positional[(*npos)++] = argv[i];
    }
  }
  return 0;
}

static int resolvePositional(struct Config *cfg, const char **positional,
                             int npos) {
  if (cfg->kernelFile && cfg->kernelName) {
    fprintf(stderr, "Cannot use both a kernel name and --kernel=file\n");
    return 1;
  }
  if (cfg->pipeline) {
    if (!cfg->outdir) {
      fprintf(stderr, "--outdir is required in pipeline mode\n");
      return 1;
    }
    if (!cfg->indir) {
      fprintf(stderr, "--indir is required in pipeline mode\n");
      return 1;
    }
    if (!cfg->kernelFile) {
      if (npos < 1) {
        fprintf(stderr, "Pipeline mode requires a kernel name\n");
        return 1;
      }
      cfg->kernelName = positional[0];
    }
  } else {
    int kernelPos = cfg->kernelFile ? 2 : 3;
    if (npos < kernelPos) {
      fprintf(stderr, "Single mode requires: input output kernel\n");
      return 1;
    }
    cfg->inputImage  = positional[0];
    cfg->outputImage = positional[1];
    if (!cfg->kernelFile) {
      cfg->kernelName = positional[2];
    }
  }
  if (!cfg->kernelName && !cfg->kernelFile) {
    fprintf(stderr, "No kernel specified\n");
    return 1;
  }
  return 0;
}

static double elapsedMs(struct timespec t0, struct timespec t1) {
  return ((double)(t1.tv_sec - t0.tv_sec) * 1000.0) +
         ((double)(t1.tv_nsec - t0.tv_nsec) / 1e6);
}

static int runSingle(const struct Config *cfg, const struct Filter *filter,
                     struct Filter *customFilter) {
  struct timespec t0;
  struct timespec t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  struct BmpImage image;
  if (readBmp(cfg->inputImage, &image) != 0) {
    free(customFilter->kernel);
    return 1;
  }

  for (int r = 0; r < cfg->repeat; r++) {
    if (cfg->parallel) {
      applyConvolutionParallel(&image, filter, cfg->numThreads,
                               cfg->parallelType);
    } else {
      applyConvolution(&image, filter);
    }
  }

  int ret = 0;
  if (writeBmp(cfg->outputImage, &image) != 0) {
    ret = 1;
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("TIME_MS: %.3f\n", elapsedMs(t0, t1) / (double)cfg->repeat);

  free(image.palette);
  free(image.data);
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }

  struct Config cfg = {
      .parallel        = 0,
      .numThreads      = (int)sysconf(_SC_NPROCESSORS_ONLN),
      .repeat          = 1,
      .pipeline        = 0,
      .pipelineWorkers = 1,
      .queueSize       = 8,
      .parallelType    = horizontal,
  };

  const char *positional[argc];
  int         npos = 0;

  if (parseArgs(argc, argv, &cfg, positional, &npos) != 0) {
    usage(argv[0]);
    return 1;
  }
  if (resolvePositional(&cfg, positional, npos) != 0) {
    usage(argv[0]);
    return 1;
  }

  struct Filter        customFilter = {0};
  const struct Filter *filter;

  if (cfg.kernelFile) {
    if (loadFilterFromFile(cfg.kernelFile, &customFilter) != 0) {
      return 1;
    }
    filter = &customFilter;
  } else {
    filter = resolveNamedKernel(cfg.kernelName);
    if (!filter) {
      return 1;
    }
  }

  if (cfg.pipeline) {
    struct timespec t0;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    runPipeline(cfg.indir, cfg.outdir, filter, cfg.pipelineWorkers,
                cfg.queueSize, cfg.parallel, cfg.numThreads, cfg.parallelType);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS: %.3f\n", elapsedMs(t0, t1));
    free(customFilter.kernel);
    return 0;
  }

  return runSingle(&cfg, filter, &customFilter);
}
