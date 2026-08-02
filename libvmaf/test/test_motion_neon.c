#include <stdint.h>
#include <stdlib.h>

#include "cpu.h"
#include "test.h"
#include "libvmaf/picture.h"
#include "feature/feature_extractor.h"
#include "feature/feature_collector.h"

static void fill_random_luma(VmafPicture *pic, unsigned seed)
{
    srand(seed);
    uint8_t *data = pic->data[0];
    for (unsigned i = 0; i < pic->h[0]; i++) {
        for (unsigned j = 0; j < pic->w[0]; j++) {
            data[i * pic->stride[0] + j] = (uint8_t)(rand() % 256);
        }
    }
}

static int compute_motion_sad(unsigned w, unsigned h,
                              unsigned seed_prev, unsigned seed_cur,
                              unsigned cpu_mask, double *score)
{
    int err;
    vmaf_set_cpu_flags_mask(cpu_mask);

    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion");
    if (!fex) return -1;

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    if (err) return err;

    VmafPicture prev_pic, cur_pic;
    err = vmaf_picture_alloc(&prev_pic, VMAF_PIX_FMT_YUV420P, 8, w, h);
    if (err) return err;
    err = vmaf_picture_alloc(&cur_pic, VMAF_PIX_FMT_YUV420P, 8, w, h);
    if (err) return err;

    fill_random_luma(&prev_pic, seed_prev);
    fill_random_luma(&cur_pic, seed_cur);

    VmafFeatureCollector *vfc;
    err = vmaf_feature_collector_init(&vfc);
    if (err) return err;

    err = vmaf_feature_extractor_context_extract(fex_ctx, &prev_pic, NULL,
                                                 &prev_pic, NULL, 0, vfc);
    if (err) return err;

    if (fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_PREV_REF)
        fex_ctx->fex->prev_ref = prev_pic;

    err = vmaf_feature_extractor_context_extract(fex_ctx, &cur_pic, NULL,
                                                 &cur_pic, NULL, 1, vfc);
    if (err) return err;

    err = vmaf_feature_collector_get_score(vfc,
            "VMAF_integer_feature_motion_sad_score", score, 1);
    if (err) return err;

    err = vmaf_feature_extractor_context_close(fex_ctx);
    if (err) return err;
    err = vmaf_feature_extractor_context_destroy(fex_ctx);
    if (err) return err;

    vmaf_feature_collector_destroy(vfc);
    vmaf_picture_unref(&prev_pic);
    vmaf_picture_unref(&cur_pic);

    return 0;
}

static char *test_motion_neon_matches_scalar()
{
    // w/h < 3 excluded: mirror()'s radius-2 reflection goes out of bounds there
    // (same bug in scalar/AVX2), so those sizes compare garbage against garbage.
    static const struct { unsigned w, h; } sizes[] = {
        {3, 3}, {4, 4}, {5, 5}, {7, 7}, {9, 9},
        {15, 15}, {16, 16}, {17, 17}, {20, 4}, {33, 9}, {64, 48}, {65, 63},
    };
    const unsigned n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    vmaf_init_cpu();

    for (unsigned s = 0; s < n_sizes; s++) {
        for (unsigned seed = 0; seed < 3; seed++) {
            double scalar_score = -1.0, neon_score = -1.0;
            int err;

            err = compute_motion_sad(sizes[s].w, sizes[s].h,
                                     100 + seed, 200 + seed,
                                     0, &scalar_score);
            mu_assert("scalar motion extraction failed", !err);

            err = compute_motion_sad(sizes[s].w, sizes[s].h,
                                     100 + seed, 200 + seed,
                                     ~0u, &neon_score);
            mu_assert("neon motion extraction failed", !err);

            mu_assert("NEON motion SAD score must bit-exactly match scalar",
                      scalar_score == neon_score);
        }
    }

    vmaf_set_cpu_flags_mask(~0u);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_motion_neon_matches_scalar);
    return NULL;
}
