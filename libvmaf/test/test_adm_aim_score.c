/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <math.h>

#include "test.h"
#include "feature/integer_adm.c"

#define W 64
#define H 64
#define POISON (-12345.0)

/* A reference without any detail and adm_noise_weight=0 give a zero ADM
 * denominator, the case in which integer_compute_adm() used to leave
 * *score_aim unwritten. */
static int run_adm(unsigned char ref_val, unsigned char dis_val,
                   int dis_stripes, double *score, double *score_aim)
{
    AdmState s;
    memset(&s, 0, sizeof(s));
    VmafFeatureExtractor fex = vmaf_fex_integer_adm;
    fex.priv = &s;
    for (unsigned i = 0; fex.options[i].name; i++) {
        const VmafOption *opt = &fex.options[i];
        if (vmaf_option_set(opt, &s,
                            strcmp(opt->name, "adm_noise_weight") ? NULL : "0"))
            return -1;
    }
    if (init(&fex, VMAF_PIX_FMT_YUV420P, 8, W, H)) return -1;

    VmafPicture ref, dis;
    if (vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, W, H)) return -1;
    if (vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, 8, W, H)) return -1;
    for (unsigned y = 0; y < H; y++) {
        for (unsigned x = 0; x < W; x++) {
            ((uint8_t *) ref.data[0])[y * ref.stride[0] + x] = ref_val;
            ((uint8_t *) dis.data[0])[y * dis.stride[0] + x] =
                (dis_stripes && (x & 1)) ? dis_val : ref_val;
        }
    }

    double score_num, score_den, scores[8];
    *score_aim = POISON;
    integer_compute_adm(&s, &ref, &dis, score, &score_num, &score_den, scores,
                        &s.buf, s.adm_enhn_gain_limit, s.adm_norm_view_dist,
                        s.adm_ref_display_height, score_aim, s.adm_csf_mode,
                        s.adm_csf_scale, s.adm_csf_diag_scale,
                        s.adm_noise_weight, s.adm_skip_aim,
                        s.adm_skip_scale0);

    int err = score_den != 0.0;
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dis);
    close(&fex);
    return err;
}

static char *test_flat_reference_flat_distorted()
{
    double score, score_aim;
    mu_assert("flat/flat: zero denominator expected",
              run_adm(128, 128, 0, &score, &score_aim) == 0);
    mu_assert("flat/flat: score", score == 1.0);
    mu_assert("flat/flat: score_aim left unwritten", score_aim == 0.0);
    return NULL;
}

static char *test_flat_reference_distorted_detail()
{
    double score, score_aim;
    mu_assert("flat/striped: zero denominator expected",
              run_adm(128, 255, 1, &score, &score_aim) == 0);
    mu_assert("flat/striped: score", score == 1.0);
    mu_assert("flat/striped: score_aim left unwritten", score_aim == 1.0);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_flat_reference_flat_distorted);
    mu_run_test(test_flat_reference_distorted_detail);
    return NULL;
}
