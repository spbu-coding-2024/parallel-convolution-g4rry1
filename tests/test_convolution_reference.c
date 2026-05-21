#include "bmpHandler.h"
#include "convolution.h"
#include "filter.h"
#include "fixtures.h"
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <string.h>

static void assert_uint8_near(uint8_t expected, uint8_t actual, int delta) {
  assert_true(abs((int)actual - (int)expected) <= delta);
}

typedef struct {
  int w, h;
  const uint8_t *input;
  const uint8_t *expected_identity;
  const uint8_t *expected_gaussian_blur;
  const uint8_t *expected_edge_detection;
  const uint8_t *expected_sharpen;
  const uint8_t *expected_emboss;
} FixtureSet;

static const FixtureSet kFixtures[] = {
    {FIXTURE_4X4_W, FIXTURE_4X4_H, fixture_4x4_input,
     fixture_4x4_expected_identity, fixture_4x4_expected_gaussian_blur,
     fixture_4x4_expected_edge_detection, fixture_4x4_expected_sharpen,
     fixture_4x4_expected_emboss},
    {FIXTURE_16X16_W, FIXTURE_16X16_H, fixture_16x16_input,
     fixture_16x16_expected_identity, fixture_16x16_expected_gaussian_blur,
     fixture_16x16_expected_edge_detection, fixture_16x16_expected_sharpen,
     fixture_16x16_expected_emboss},
    {FIXTURE_32X32_W, FIXTURE_32X32_H, fixture_32x32_input,
     fixture_32x32_expected_identity, fixture_32x32_expected_gaussian_blur,
     fixture_32x32_expected_edge_detection, fixture_32x32_expected_sharpen,
     fixture_32x32_expected_emboss},
};
#define N_FIXTURES ((int)(sizeof(kFixtures) / sizeof(kFixtures[0])))

static struct BmpImage makeImageFromPixels(int w, int h,
                                           const uint8_t *pixels) {
  struct BmpImage img = {0};
  img.info.biWidth = w;
  img.info.biHeight = h;
  int stride = bmpStride(&img);
  img.data = malloc((size_t)stride * h);
  for (int y = 0; y < h; y++) {
    memcpy(&img.data[(size_t)y * stride], &pixels[(size_t)y * w], w);
}
  return img;
}

static void freeImage(struct BmpImage *img) { free(img->data); }

static void check_vs_reference(const struct Filter *filter,
                               const FixtureSet *fs, const uint8_t *expected) {
  struct BmpImage img = makeImageFromPixels(fs->w, fs->h, fs->input);
  int stride = bmpStride(&img);

  applyConvolution(&img, filter);

  for (int y = 0; y < fs->h; y++) {
    for (int x = 0; x < fs->w; x++) {
      assert_uint8_near(expected[(y * fs->w) + x], img.data[(y * stride) + x], 1);
}
}
  freeImage(&img);
}

static void test_reference_identity(void **state) {
  (void)state;
  for (int i = 0; i < N_FIXTURES; i++) {
    check_vs_reference(&filterIdentity, &kFixtures[i],
                       kFixtures[i].expected_identity);
}
}

static void test_reference_gaussian_blur(void **state) {
  (void)state;
  for (int i = 0; i < N_FIXTURES; i++) {
    check_vs_reference(&filterGaussianBlur, &kFixtures[i],
                       kFixtures[i].expected_gaussian_blur);
}
}

static void test_reference_edge_detection(void **state) {
  (void)state;
  for (int i = 0; i < N_FIXTURES; i++) {
    check_vs_reference(&filterEdgeDetection, &kFixtures[i],
                       kFixtures[i].expected_edge_detection);
}
}

static void test_reference_sharpen(void **state) {
  (void)state;
  for (int i = 0; i < N_FIXTURES; i++) {
    check_vs_reference(&filterSharpen, &kFixtures[i],
                       kFixtures[i].expected_sharpen);
}
}

static void test_reference_emboss(void **state) {
  (void)state;
  for (int i = 0; i < N_FIXTURES; i++) {
    check_vs_reference(&filterEmboss, &kFixtures[i],
                       kFixtures[i].expected_emboss);
}
}

int main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_reference_identity),
      cmocka_unit_test(test_reference_gaussian_blur),
      cmocka_unit_test(test_reference_edge_detection),
      cmocka_unit_test(test_reference_sharpen),
      cmocka_unit_test(test_reference_emboss),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
