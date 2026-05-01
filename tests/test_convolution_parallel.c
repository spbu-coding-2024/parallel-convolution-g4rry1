#include "bmpHandler.h"
#include "convolution.h"
#include "convolutionParallel.h"
#include "filter.h"
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <string.h>

static struct BmpImage makeRandomImage(int w, int h, unsigned int seed) {
  struct BmpImage img = {0};
  img.info.biWidth = w;
  img.info.biHeight = h;
  int stride = bmpStride(&img);
  img.data = malloc(stride * h);
  srand(seed);
  for (int i = 0; i < stride * h; i++)
    img.data[i] = (uint8_t)(rand() % 256);
  return img;
}

static void freeImage(struct BmpImage *img) { free(img->data); }
static void freeBmpImage(struct BmpImage *img) {
  free(img->palette);
  free(img->data);
}

// applies filter sequentially and in parallel (same seed → same input),
// then compares every pixel
static void check_parallel_equals_sequential(const struct Filter *filter,
                                             enum TypeParallel type,
                                             int numThreads) {
  int w = 64, h = 64;
  struct BmpImage seq = makeRandomImage(w, h, 42);
  struct BmpImage par = makeRandomImage(w, h, 42);
  int stride = bmpStride(&seq);

  applyConvolution(&seq, filter);
  applyConvolutionParallel(&par, filter, numThreads, type);

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(seq.data[y * stride + x], par.data[y * stride + x]);
  freeImage(&seq);
  freeImage(&par);
}

// parametric variant: explicit size, seed and thread count
static void check_par_eq_seq(const struct Filter *filter,
                             enum TypeParallel type, int numThreads, int w,
                             int h, unsigned int seed) {
  struct BmpImage seq = makeRandomImage(w, h, seed);
  struct BmpImage par = makeRandomImage(w, h, seed);
  int stride = bmpStride(&seq);

  applyConvolution(&seq, filter);
  applyConvolutionParallel(&par, filter, numThreads, type);

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(seq.data[y * stride + x], par.data[y * stride + x]);
  freeImage(&seq);
  freeImage(&par);
}

// --- horizontal ---

static void test_horizontal_identity(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterIdentity, horizontal, 4);
}
static void test_horizontal_gaussian_blur(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterGaussianBlur, horizontal, 4);
}
static void test_horizontal_edge_detection(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEdgeDetection, horizontal, 4);
}
static void test_horizontal_sharpen(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterSharpen, horizontal, 4);
}
static void test_horizontal_emboss(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEmboss, horizontal, 4);
}

// --- vertical ---

static void test_vertical_identity(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterIdentity, vertical, 4);
}
static void test_vertical_gaussian_blur(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterGaussianBlur, vertical, 4);
}
static void test_vertical_edge_detection(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEdgeDetection, vertical, 4);
}
static void test_vertical_sharpen(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterSharpen, vertical, 4);
}
static void test_vertical_emboss(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEmboss, vertical, 4);
}

// --- block (numThreads must be a perfect square: 4 = 2x2) ---

static void test_block_identity(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterIdentity, block, 4);
}
static void test_block_gaussian_blur(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterGaussianBlur, block, 4);
}
static void test_block_edge_detection(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEdgeDetection, block, 4);
}
static void test_block_sharpen(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterSharpen, block, 4);
}
static void test_block_emboss(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEmboss, block, 4);
}

// --- pixel (work-stealing) ---

static void test_pixel_identity(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterIdentity, pixel, 4);
}
static void test_pixel_gaussian_blur(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterGaussianBlur, pixel, 4);
}
static void test_pixel_edge_detection(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEdgeDetection, pixel, 4);
}
static void test_pixel_sharpen(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterSharpen, pixel, 4);
}
static void test_pixel_emboss(void **state) {
  (void)state;
  check_parallel_equals_sequential(&filterEmboss, pixel, 4);
}

// --- thread-count and image-size variation ---

static const int kLinearThreads[] = {1, 2, 4, 8, 16};

static const struct {
  int w, h;
  unsigned int seed;
} kParSizes[] = {
    {1, 1, 3}, {2, 3, 5}, {4, 4, 7}, {32, 32, 42}, {128, 128, 99},
};

typedef struct {
  const char *name;
  const struct Filter *filter;
} FilterCase;

static const FilterCase kFilters[] = {
    {"identity", &filterIdentity},
    {"gaussian_blur", &filterGaussianBlur},
    {"edge_detection", &filterEdgeDetection},
    {"sharpen", &filterSharpen},
    {"emboss", &filterEmboss},
};

#define N_LINEAR_THREADS                                                       \
  ((int)(sizeof(kLinearThreads) / sizeof(kLinearThreads[0])))
#define N_FILTERS ((int)(sizeof(kFilters) / sizeof(kFilters[0])))
#define N_PAR_SIZES ((int)(sizeof(kParSizes) / sizeof(kParSizes[0])))

static void test_horizontal_thread_counts(void **state) {
  (void)state;
  for (int fi = 0; fi < N_FILTERS; fi++)
    for (int ti = 0; ti < N_LINEAR_THREADS; ti++)
      for (int si = 0; si < N_PAR_SIZES; si++)
        check_par_eq_seq(kFilters[fi].filter, horizontal, kLinearThreads[ti],
                         kParSizes[si].w, kParSizes[si].h, kParSizes[si].seed);
}

static void test_vertical_thread_counts(void **state) {
  (void)state;
  for (int fi = 0; fi < N_FILTERS; fi++)
    for (int ti = 0; ti < N_LINEAR_THREADS; ti++)
      for (int si = 0; si < N_PAR_SIZES; si++)
        check_par_eq_seq(kFilters[fi].filter, vertical, kLinearThreads[ti],
                         kParSizes[si].w, kParSizes[si].h, kParSizes[si].seed);
}

static void test_pixel_thread_counts(void **state) {
  (void)state;
  for (int fi = 0; fi < N_FILTERS; fi++)
    for (int ti = 0; ti < N_LINEAR_THREADS; ti++)
      for (int si = 0; si < N_PAR_SIZES; si++)
        check_par_eq_seq(kFilters[fi].filter, pixel, kLinearThreads[ti],
                         kParSizes[si].w, kParSizes[si].h, kParSizes[si].seed);
}

// block requires a perfect-square numThreads; 16 = 4x4 grid stresses small
// images
static const int kBlockThreads[] = {1, 4, 9, 16};
#define N_BLOCK_THREADS                                                        \
  ((int)(sizeof(kBlockThreads) / sizeof(kBlockThreads[0])))

static void test_block_thread_counts(void **state) {
  (void)state;
  for (int fi = 0; fi < N_FILTERS; fi++)
    for (int ti = 0; ti < N_BLOCK_THREADS; ti++)
      for (int si = 0; si < N_PAR_SIZES; si++)
        check_par_eq_seq(kFilters[fi].filter, block, kBlockThreads[ti],
                         kParSizes[si].w, kParSizes[si].h, kParSizes[si].seed);
}

static void test_bmp_file_parallel_matches_sequential(void **state) {
  (void)state;
  struct BmpImage seq = {0};
  struct BmpImage par = {0};
  assert_int_equal(0, readBmp(TESTFILES_DIR "/lena.bmp", &seq));
  assert_int_equal(0, readBmp(TESTFILES_DIR "/lena.bmp", &par));

  int stride = bmpStride(&seq);
  int height = seq.info.biHeight < 0 ? -seq.info.biHeight : seq.info.biHeight;

  applyConvolution(&seq, &filterSharpen);
  applyConvolutionParallel(&par, &filterSharpen, 4, block);

  assert_int_equal(seq.info.biWidth, par.info.biWidth);
  assert_int_equal(seq.info.biHeight, par.info.biHeight);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < seq.info.biWidth; x++)
      assert_int_equal(seq.data[y * stride + x], par.data[y * stride + x]);

  freeBmpImage(&seq);
  freeBmpImage(&par);
}

int main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_horizontal_identity),
      cmocka_unit_test(test_horizontal_gaussian_blur),
      cmocka_unit_test(test_horizontal_edge_detection),
      cmocka_unit_test(test_horizontal_sharpen),
      cmocka_unit_test(test_horizontal_emboss),
      cmocka_unit_test(test_vertical_identity),
      cmocka_unit_test(test_vertical_gaussian_blur),
      cmocka_unit_test(test_vertical_edge_detection),
      cmocka_unit_test(test_vertical_sharpen),
      cmocka_unit_test(test_vertical_emboss),
      cmocka_unit_test(test_block_identity),
      cmocka_unit_test(test_block_gaussian_blur),
      cmocka_unit_test(test_block_edge_detection),
      cmocka_unit_test(test_block_sharpen),
      cmocka_unit_test(test_block_emboss),
      cmocka_unit_test(test_pixel_identity),
      cmocka_unit_test(test_pixel_gaussian_blur),
      cmocka_unit_test(test_pixel_edge_detection),
      cmocka_unit_test(test_pixel_sharpen),
      cmocka_unit_test(test_pixel_emboss),
      cmocka_unit_test(test_horizontal_thread_counts),
      cmocka_unit_test(test_vertical_thread_counts),
      cmocka_unit_test(test_pixel_thread_counts),
      cmocka_unit_test(test_block_thread_counts),
      cmocka_unit_test(test_bmp_file_parallel_matches_sequential),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
