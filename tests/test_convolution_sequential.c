#include "bmpHandler.h"
#include "convolution.h"
#include "filter.h"
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <string.h>

static struct BmpImage makeUniformImage(int w, int h, uint8_t fill) {
  struct BmpImage img = {0};
  img.info.biWidth = w;
  img.info.biHeight = h;
  int stride = bmpStride(&img);
  img.data = malloc(stride * h);
  memset(img.data, fill, stride * h);
  return img;
}

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

// identity filter must not change any pixel of a random image
static void test_identity_preserves_random_image(void **state) {
  (void)state;
  int w = 64, h = 64;
  struct BmpImage img = makeRandomImage(w, h, 42);
  int stride = bmpStride(&img);

  uint8_t *expected = malloc(stride * h);
  memcpy(expected, img.data, stride * h);

  applyConvolution(&img, &filterIdentity);

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(expected[y * stride + x], img.data[y * stride + x]);

  free(expected);
  freeImage(&img);
}

// uniform image + gaussian blur = same value
// (kernel sum = 16, factor = 1/16, 128 * 16 * (1/16) = 128)
static void test_gaussian_blur_uniform_unchanged(void **state) {
  (void)state;
  int w = 32, h = 32;
  struct BmpImage img = makeUniformImage(w, h, 128);
  int stride = bmpStride(&img);

  applyConvolution(&img, &filterGaussianBlur);

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(128, img.data[y * stride + x]);

  freeImage(&img);
}

// uniform image + edge detection = all zeros
// (kernel sum = 0, so brightness = 0 for any uniform input)
static void test_edge_detection_uniform_gives_zero(void **state) {
  (void)state;
  int w = 32, h = 32;
  struct BmpImage img = makeUniformImage(w, h, 200);
  int stride = bmpStride(&img);

  applyConvolution(&img, &filterEdgeDetection);

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(0, img.data[y * stride + x]);

  freeImage(&img);
}

// uniform image + sharpen = same value
// (kernel sum = 1, factor = 1.0, no division — result is exact)
static void test_sharpen_uniform_unchanged(void **state) {
  (void)state;
  int w = 32, h = 32;
  struct BmpImage img = makeUniformImage(w, h, 150);
  int stride = bmpStride(&img);

  applyConvolution(&img, &filterSharpen);

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(150, img.data[y * stride + x]);

  freeImage(&img);
}

// 3x3 image: only center = 255, rest = 0
// edge detection: center → 8*255 = 2040 → clamp → 255
//                 others: -255 → clamp → 0
static void test_edge_detection_single_bright_center_3x3(void **state) {
  (void)state;
  int w = 3, h = 3;
  struct BmpImage img = makeUniformImage(w, h, 0);
  int stride = bmpStride(&img);
  img.data[stride + 1] = 255;

  applyConvolution(&img, &filterEdgeDetection);

  assert_int_equal(255, img.data[stride + 1]);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      if (!(x == 1 && y == 1))
        assert_int_equal(0, img.data[y * stride + x]);

  freeImage(&img);
}

// boundary values: all-black and all-white images must not change under
// identity
static void test_identity_boundary_values(void **state) {
  (void)state;
  int w = 16, h = 16;
  int stride;

  struct BmpImage black = makeUniformImage(w, h, 0);
  applyConvolution(&black, &filterIdentity);
  stride = bmpStride(&black);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(0, black.data[y * stride + x]);
  freeImage(&black);

  struct BmpImage white = makeUniformImage(w, h, 255);
  applyConvolution(&white, &filterIdentity);
  stride = bmpStride(&white);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(255, white.data[y * stride + x]);
  freeImage(&white);
}

// shift right then shift left = original image (composition giving identity)
// kernel[1][0]=1 shifts right: output[y][x] = input[y][(x-1+w)%w]
// kernel[1][2]=1 shifts left:  output[y][x] = input[y][(x+1)%w]
static void test_composition_shift_right_then_left(void **state) {
  (void)state;
  int w = 16, h = 16;
  struct BmpImage img = makeRandomImage(w, h, 42);
  int stride = bmpStride(&img);

  uint8_t *original = malloc(stride * h);
  memcpy(original, img.data, stride * h);

  double kernelRight[] = {0, 0, 0, 1, 0, 0, 0, 0, 0};
  double kernelLeft[] = {0, 0, 0, 0, 0, 1, 0, 0, 0};
  struct Filter shiftRight = {3, 3, 1.0, 0.0, kernelRight};
  struct Filter shiftLeft = {3, 3, 1.0, 0.0, kernelLeft};

  applyConvolution(&img, &shiftRight);
  applyConvolution(&img, &shiftLeft);

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(original[y * stride + x], img.data[y * stride + x]);
  free(original);
  freeImage(&img);
}

// zero kernel: all coefficients 0 → brightness = 0 → result = clamp(bias, 0,
// 255)
static void test_zero_kernel_gives_zero(void **state) {
  (void)state;
  double kernelZero[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  struct Filter zeroFilter = {3, 3, 1.0, 0.0, kernelZero};
  int w = 32, h = 32;
  struct BmpImage img = makeRandomImage(w, h, 77);
  int stride = bmpStride(&img);
  applyConvolution(&img, &zeroFilter);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(0, img.data[y * stride + x]);
  freeImage(&img);
}

// zero kernel + bias=100: every pixel → 100 regardless of input
static void test_zero_kernel_with_bias(void **state) {
  (void)state;
  double kernelZero[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  struct Filter biasOnly = {3, 3, 1.0, 100.0, kernelZero};
  int w = 32, h = 32;
  struct BmpImage img = makeRandomImage(w, h, 88);
  int stride = bmpStride(&img);
  applyConvolution(&img, &biasOnly);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      assert_int_equal(100, img.data[y * stride + x]);
  freeImage(&img);
}

// --- wide-range parametric tests ---

static const struct {
  int w, h;
  unsigned int seed;
} kSizes[] = {
    {3, 3, 0}, {7, 11, 1}, {16, 16, 42}, {64, 64, 123}, {100, 50, 999},
};
#define N_SIZES ((int)(sizeof(kSizes) / sizeof(kSizes[0])))

// identity preserves every random image across many sizes and seeds
static void test_identity_various_sizes(void **state) {
  (void)state;
  for (int i = 0; i < N_SIZES; i++) {
    int w = kSizes[i].w, h = kSizes[i].h;
    struct BmpImage img = makeRandomImage(w, h, kSizes[i].seed);
    int stride = bmpStride(&img);
    uint8_t *orig = malloc(stride * h);
    memcpy(orig, img.data, stride * h);
    applyConvolution(&img, &filterIdentity);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        assert_int_equal(orig[y * stride + x], img.data[y * stride + x]);
    free(orig);
    freeImage(&img);
  }
}

// edge detection on any uniform image gives zero (kernel sum = 0),
// tested over multiple sizes and fill values
static void test_edge_detection_various_sizes(void **state) {
  (void)state;
  const uint8_t fills[] = {0, 1, 100, 200, 255};
  for (int fi = 0; fi < (int)(sizeof(fills) / sizeof(fills[0])); fi++) {
    for (int i = 0; i < N_SIZES; i++) {
      int w = kSizes[i].w, h = kSizes[i].h;
      struct BmpImage img = makeUniformImage(w, h, fills[fi]);
      int stride = bmpStride(&img);
      applyConvolution(&img, &filterEdgeDetection);
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
          assert_int_equal(0, img.data[y * stride + x]);
      freeImage(&img);
    }
  }
}

// factor=2 doubles pixel values exactly for values in [0,127] (no clamping)
static void test_factor_doubles_output(void **state) {
  (void)state;
  double kernelId[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  struct Filter doubleFilter = {3, 3, 2.0, 0.0, kernelId};
  const uint8_t vals[] = {0, 1, 64, 100, 127};
  for (int vi = 0; vi < (int)(sizeof(vals) / sizeof(vals[0])); vi++) {
    for (int i = 0; i < N_SIZES; i++) {
      int w = kSizes[i].w, h = kSizes[i].h;
      struct BmpImage img = makeUniformImage(w, h, vals[vi]);
      int stride = bmpStride(&img);
      applyConvolution(&img, &doubleFilter);
      uint8_t expected = (uint8_t)(vals[vi] * 2);
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
          assert_int_equal(expected, img.data[y * stride + x]);
      freeImage(&img);
    }
  }
}

// bias=128 shifts a black image to grey (factor=1, kernel=identity)
static void test_bias_shifts_output(void **state) {
  (void)state;
  double kernelId[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  struct Filter biasFilter = {3, 3, 1.0, 128.0, kernelId};
  for (int i = 0; i < N_SIZES; i++) {
    int w = kSizes[i].w, h = kSizes[i].h;
    struct BmpImage img = makeUniformImage(w, h, 0);
    int stride = bmpStride(&img);
    applyConvolution(&img, &biasFilter);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        assert_int_equal(128, img.data[y * stride + x]);
    freeImage(&img);
  }
}

// horizontal composition (shift right + shift left = identity) over many sizes
static void test_composition_horizontal_various_sizes(void **state) {
  (void)state;
  double kernelRight[] = {0, 0, 0, 1, 0, 0, 0, 0, 0};
  double kernelLeft[] = {0, 0, 0, 0, 0, 1, 0, 0, 0};
  struct Filter shiftRight = {3, 3, 1.0, 0.0, kernelRight};
  struct Filter shiftLeft = {3, 3, 1.0, 0.0, kernelLeft};
  for (int i = 0; i < N_SIZES; i++) {
    int w = kSizes[i].w, h = kSizes[i].h;
    struct BmpImage img = makeRandomImage(w, h, kSizes[i].seed);
    int stride = bmpStride(&img);
    uint8_t *orig = malloc(stride * h);
    memcpy(orig, img.data, stride * h);
    applyConvolution(&img, &shiftRight);
    applyConvolution(&img, &shiftLeft);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        assert_int_equal(orig[y * stride + x], img.data[y * stride + x]);
    free(orig);
    freeImage(&img);
  }
}

// vertical composition: shift down + shift up = identity
// kernel[0][1]=1: output[y][x] = input[(y-1+h)%h][x]  (shifts content down)
// kernel[2][1]=1: output[y][x] = input[(y+1)%h][x]    (shifts content up)
static void test_composition_shift_vertical(void **state) {
  (void)state;
  double kernelDown[] = {0, 1, 0, 0, 0, 0, 0, 0, 0};
  double kernelUp[] = {0, 0, 0, 0, 0, 0, 0, 1, 0};
  struct Filter shiftDown = {3, 3, 1.0, 0.0, kernelDown};
  struct Filter shiftUp = {3, 3, 1.0, 0.0, kernelUp};
  for (int i = 0; i < N_SIZES; i++) {
    int w = kSizes[i].w, h = kSizes[i].h;
    struct BmpImage img = makeRandomImage(w, h, kSizes[i].seed);
    int stride = bmpStride(&img);
    uint8_t *orig = malloc(stride * h);
    memcpy(orig, img.data, stride * h);
    applyConvolution(&img, &shiftDown);
    applyConvolution(&img, &shiftUp);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        assert_int_equal(orig[y * stride + x], img.data[y * stride + x]);
    free(orig);
    freeImage(&img);
  }
}

int main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_identity_preserves_random_image),
      cmocka_unit_test(test_gaussian_blur_uniform_unchanged),
      cmocka_unit_test(test_edge_detection_uniform_gives_zero),
      cmocka_unit_test(test_sharpen_uniform_unchanged),
      cmocka_unit_test(test_edge_detection_single_bright_center_3x3),
      cmocka_unit_test(test_identity_boundary_values),
      cmocka_unit_test(test_zero_kernel_gives_zero),
      cmocka_unit_test(test_zero_kernel_with_bias),
      cmocka_unit_test(test_composition_shift_right_then_left),
      cmocka_unit_test(test_identity_various_sizes),
      cmocka_unit_test(test_edge_detection_various_sizes),
      cmocka_unit_test(test_factor_doubles_output),
      cmocka_unit_test(test_bias_shifts_output),
      cmocka_unit_test(test_composition_horizontal_various_sizes),
      cmocka_unit_test(test_composition_shift_vertical),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
