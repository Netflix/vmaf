/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

/* integer_adm_scale0 of adm_cuda against the scalar CPU adm. The CPU rounds the
 * contrast masking sum of each row once with shift_inner_accum; the structured
 * content keeps most row sums small, where rounding every warp tile instead
 * of every row is a large relative error. Needs a CUDA device. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/feature.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/picture.h"

#define MAX_FRAMES 4
#define NUM_SCORES 5

enum content {
  CONTENT_RANDOM,
  CONTENT_STRUCTURED,
};

static const char *const score_name[NUM_SCORES] = {
    "VMAF_integer_feature_adm2_score",
    "integer_adm_scale0",
    "integer_adm_scale1",
    "integer_adm_scale2",
    "integer_adm_scale3",
};

static uint64_t rng_state;

static uint32_t rng_next(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (uint32_t)(rng_state >> 33);
}

static int clip_sample(int v, int max) {
  return v < 0 ? 0 : (v > max ? max : v);
}

/* Plane of the reference picture. Random: independent samples. Structured: a
 * ramp with a 3x3 checkerboard and a mirrored lower right quadrant, whose
 * smooth areas leave near-zero contrast masking accumulators. */
static void fill_reference(int *ref, unsigned w, unsigned h, int mx,
                           enum content content) {
  for (unsigned y = 0; y < h; y++) {
    for (unsigned x = 0; x < w; x++) {
      int v;
      if (content == CONTENT_RANDOM) {
        v = (int)(rng_next() % (unsigned)(mx + 1));
      } else {
        const int ramp = ((int)(x * (unsigned)mx / (w > 1 ? w - 1 : 1)) +
                          (int)(y * (unsigned)mx / (h > 1 ? h - 1 : 1))) /
                         2;
        const int check = (((x / 3) + (y / 3)) & 1) ? mx / 8 : 0;
        v = ramp / 2 + check + (int)(rng_next() % (unsigned)(mx / 32 + 1));
        if (x > w / 2 && y > h / 3)
          v = mx - v;
      }
      ref[y * w + x] = clip_sample(v, mx);
    }
  }
}

/* Distorted plane: random noise on the reference, or its 3x3 box blur plus
 * a little noise. */
static void fill_distorted(const int *ref, int *dis, unsigned w, unsigned h,
                           int mx, enum content content) {
  for (unsigned y = 0; y < h; y++) {
    for (unsigned x = 0; x < w; x++) {
      int v;
      if (content == CONTENT_RANDOM) {
        v = ref[y * w + x] + (int)(rng_next() % (unsigned)(mx / 4 + 1)) -
            mx / 8;
      } else {
        int sum = 0, n = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            const int yy = (int)y + dy, xx = (int)x + dx;
            if (yy < 0 || xx < 0 || yy >= (int)h || xx >= (int)w)
              continue;
            sum += ref[yy * w + xx];
            n++;
          }
        }
        v = sum / n + (int)(rng_next() % (unsigned)(mx / 64 + 1));
      }
      dis[y * w + x] = clip_sample(v, mx);
    }
  }
}

static void store_plane(VmafPicture *pic, unsigned plane, const int *src) {
  const unsigned w = pic->w[plane], h = pic->h[plane];
  for (unsigned y = 0; y < h; y++) {
    uint8_t *row = (uint8_t *)pic->data[plane] + y * pic->stride[plane];
    for (unsigned x = 0; x < w; x++) {
      if (pic->bpc == 8)
        row[x] = (uint8_t)src[y * w + x];
      else
        ((uint16_t *)row)[x] = (uint16_t)src[y * w + x];
    }
  }
}

static int make_pictures(VmafPicture *ref, VmafPicture *dis, unsigned w,
                         unsigned h, unsigned bpc, enum content content) {
  int err = vmaf_picture_alloc(ref, VMAF_PIX_FMT_YUV420P, bpc, w, h);
  err |= vmaf_picture_alloc(dis, VMAF_PIX_FMT_YUV420P, bpc, w, h);
  if (err)
    return err;
  const int mx = (1 << bpc) - 1;
  for (unsigned p = 0; p < 3; p++) {
    const unsigned n = ref->w[p] * ref->h[p];
    int *r = malloc(n * sizeof(*r));
    int *d = malloc(n * sizeof(*d));
    if (!r || !d) {
      free(r);
      free(d);
      return -1;
    }
    fill_reference(r, ref->w[p], ref->h[p], mx, content);
    fill_distorted(r, d, ref->w[p], ref->h[p], mx, content);
    store_plane(ref, p, r);
    store_plane(dis, p, d);
    free(r);
    free(d);
  }
  return 0;
}

/* Scores of `frames` frames, from the CUDA extractor or from the scalar CPU
 * extractor. */
static int run_adm(int use_cuda, unsigned w, unsigned h, unsigned bpc,
                   enum content content, unsigned seed, unsigned frames,
                   double scores[MAX_FRAMES][NUM_SCORES]) {
  VmafConfiguration cfg = {0};
  if (!use_cuda)
    cfg.cpumask = ~(uint64_t)0; /* scalar C only */

  VmafContext *vmaf;
  if (vmaf_init(&vmaf, cfg))
    return -1;

  int err = 0;
  if (use_cuda) {
    VmafCudaState *cu_state;
    VmafCudaConfiguration cu_cfg = {0};
    err = vmaf_cuda_state_init(&cu_state, cu_cfg);
    if (!err)
      err = vmaf_cuda_import_state(vmaf, cu_state);
  }
  if (!err)
    err = vmaf_use_feature(vmaf, use_cuda ? "adm_cuda" : "adm", NULL);

  rng_state = (uint64_t)seed * 7919 + w * 131 + h;
  for (unsigned i = 0; !err && i < frames; i++) {
    VmafPicture ref, dis;
    err = make_pictures(&ref, &dis, w, h, bpc, content);
    if (!err)
      err = vmaf_read_pictures(vmaf, &ref, &dis, i);
  }
  if (!err)
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
  for (unsigned i = 0; !err && i < frames; i++) {
    for (unsigned s = 0; s < NUM_SCORES; s++)
      err |= vmaf_feature_score_at_index(vmaf, score_name[s], &scores[i][s], i);
  }
  err |= vmaf_close(vmaf);
  return err;
}

static int have_cuda_device(void) {
  VmafContext *vmaf;
  VmafConfiguration cfg = {0};
  if (vmaf_init(&vmaf, cfg))
    return 0;
  VmafCudaState *cu_state;
  VmafCudaConfiguration cu_cfg = {0};
  const int ok = !vmaf_cuda_state_init(&cu_state, cu_cfg);
  vmaf_close(vmaf);
  return ok;
}

/* Largest absolute difference between the CUDA and the CPU score over the
 * frames, for the scores first..last. */
static int max_difference(unsigned w, unsigned h, unsigned bpc,
                          enum content content, unsigned seed, unsigned frames,
                          unsigned first, unsigned last, double *worst) {
  double cuda[MAX_FRAMES][NUM_SCORES], cpu[MAX_FRAMES][NUM_SCORES];
  int err = run_adm(1, w, h, bpc, content, seed, frames, cuda);
  err |= run_adm(0, w, h, bpc, content, seed, frames, cpu);
  if (err)
    return err;
  *worst = 0.;
  for (unsigned i = 0; i < frames; i++) {
    for (unsigned s = first; s <= last; s++) {
      const double d = fabs(cuda[i][s] - cpu[i][s]);
      if (!(d <= *worst))
        *worst = d; /* also records NaN */
    }
  }
  return 0;
}

#define TOLERANCE 1e-9

static const struct {
    unsigned w, h;
} sizes[] = {{38, 38}, {48, 48}, {64, 64}, {130, 40}, {200, 120}};

static char *test_scale_0_matches_cpu(void)
{
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        for (unsigned bpc = 8; bpc <= 10; bpc += 2) {
            for (int content = CONTENT_RANDOM; content <= CONTENT_STRUCTURED; content++) {
                double worst;
                const int err = max_difference(sizes[i].w, sizes[i].h, bpc, content, 1, 2, 1, 1,
                                               &worst);
                mu_assert("problem during the CUDA / CPU comparison", !err);
                if (worst > TOLERANCE)
                    fprintf(stderr, "%ux%u %u-bit content %d: |cuda - cpu| = %g\n", sizes[i].w,
                            sizes[i].h, bpc, content, worst);
                mu_assert("integer_adm_scale0 of adm_cuda differs from the CPU", worst <= TOLERANCE);
            }
        }
    }
    return NULL;
}

char *run_tests(void)
{
    if (!have_cuda_device()) {
        fprintf(stderr, "no CUDA device: skipped\n");
        return NULL;
    }
    mu_run_test(test_scale_0_matches_cpu);
    return NULL;
}

