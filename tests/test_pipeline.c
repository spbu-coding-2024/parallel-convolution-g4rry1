#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "bmpHandler.h"
#include "convolution.h"
#include "filter.h"
#include "pipeline.h"
#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define INDIR TESTFILES_DIR "/pipeline_input"

static void freeImage(struct BmpImage *img) {
  free(img->palette);
  free(img->data);
}

/* Remove all .bmp files from dir, then rmdir. */
static void cleanDir(const char *dir) {
  DIR *d = opendir(dir);
  if (!d) {
    return;
}
  struct dirent *e;
  char path[PATH_MAX];
  while ((e = readdir(d)) != NULL) {
    size_t len = strlen(e->d_name);
    if (len < 4 || strcasecmp(e->d_name + len - 4, ".bmp") != 0) {
      continue;
}
    snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
    unlink(path);
  }
  closedir(d);
  rmdir(dir);
}

/*
 * Read indir/filename, apply sequential convolution → expected.
 * Read outdir/filename → actual.
 * Compare pixel by pixel.
 */
static void checkFile(const char *indir, const char *outdir,
                      const char *filename, const struct Filter *filter) {
  char inpath[PATH_MAX];
  char outpath[PATH_MAX];
  snprintf(inpath, sizeof(inpath), "%s/%s", indir, filename);
  snprintf(outpath, sizeof(outpath), "%s/%s", outdir, filename);

  struct BmpImage expected = {0};
  struct BmpImage actual = {0};
  assert_int_equal(0, readBmp(inpath, &expected));
  assert_int_equal(0, readBmp(outpath, &actual));

  applyConvolution(&expected, filter);

  int stride = bmpStride(&expected);
  int h = expected.info.biHeight < 0 ? -expected.info.biHeight
                                     : (int)expected.info.biHeight;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < (int)expected.info.biWidth; x++) {
      assert_int_equal(expected.data[(y * stride) + x],
                       actual.data[(y * stride) + x]);
}
}

  freeImage(&expected);
  freeImage(&actual);
}

/* --- tests --- */

static void test_single_worker(void **state) {
  (void)state;
  char outdir[] = "/tmp/pipe_out_XXXXXX";
  assert_non_null(mkdtemp(outdir));

  runPipeline(INDIR, outdir, &filterGaussianBlur, 1, 4, 0, 1, horizontal);

  checkFile(INDIR, outdir, "lena1.bmp", &filterGaussianBlur);
  checkFile(INDIR, outdir, "lena2.bmp", &filterGaussianBlur);
  checkFile(INDIR, outdir, "lena3.bmp", &filterGaussianBlur);
  checkFile(INDIR, outdir, "lena4.bmp", &filterGaussianBlur);

  cleanDir(outdir);
}

static void test_multi_worker(void **state) {
  (void)state;
  char outdir[] = "/tmp/pipe_out_XXXXXX";
  assert_non_null(mkdtemp(outdir));

  runPipeline(INDIR, outdir, &filterSharpen, 4, 4, 0, 1, horizontal);

  checkFile(INDIR, outdir, "lena1.bmp", &filterSharpen);
  checkFile(INDIR, outdir, "lena2.bmp", &filterSharpen);
  checkFile(INDIR, outdir, "lena3.bmp", &filterSharpen);
  checkFile(INDIR, outdir, "lena4.bmp", &filterSharpen);

  cleanDir(outdir);
}

/* queue_size=1 forces maximum backpressure. */
static void test_queue_size_one(void **state) {
  (void)state;
  char outdir[] = "/tmp/pipe_out_XXXXXX";
  assert_non_null(mkdtemp(outdir));

  runPipeline(INDIR, outdir, &filterEdgeDetection, 2, 1, 0, 1, horizontal);

  checkFile(INDIR, outdir, "lena1.bmp", &filterEdgeDetection);
  checkFile(INDIR, outdir, "lena2.bmp", &filterEdgeDetection);
  checkFile(INDIR, outdir, "lena3.bmp", &filterEdgeDetection);
  checkFile(INDIR, outdir, "lena4.bmp", &filterEdgeDetection);

  cleanDir(outdir);
}

/* Workers use parallel convolution internally. */
static void test_parallel_convolution_inside_workers(void **state) {
  (void)state;
  char outdir[] = "/tmp/pipe_out_XXXXXX";
  assert_non_null(mkdtemp(outdir));

  runPipeline(INDIR, outdir, &filterGaussianBlur, 2, 4, 1, 2, horizontal);

  checkFile(INDIR, outdir, "lena1.bmp", &filterGaussianBlur);
  checkFile(INDIR, outdir, "lena2.bmp", &filterGaussianBlur);
  checkFile(INDIR, outdir, "lena3.bmp", &filterGaussianBlur);
  checkFile(INDIR, outdir, "lena4.bmp", &filterGaussianBlur);

  cleanDir(outdir);
}

int main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_single_worker),
      cmocka_unit_test(test_multi_worker),
      cmocka_unit_test(test_queue_size_one),
      cmocka_unit_test(test_parallel_convolution_inside_workers),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
