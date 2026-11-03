/**
 *
 *  Copyright 2026 ChipLab.
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

#include <stdint.h>
#include <stdio.h>

#include "test.h"
#include "vidinput.h"

static char *test_c422_direct_read()
{
    FILE *fin = tmpfile();
    mu_assert("could not create Y4M input", fin);
    mu_assert("could not write Y4M header",
              fputs("YUV4MPEG2 W32 H24 F30:1 Ip A1:1 C422\n", fin) >= 0);
    for (unsigned f = 0; f < 2; f++) {
        mu_assert("could not write frame header", fputs("FRAME\n", fin) >= 0);
        for (unsigned p = 0; p < 3; p++) {
            const unsigned w = p ? 16 : 32;
            for (unsigned y = 0; y < 24; y++) {
                for (unsigned x = 0; x < w; x++) {
                    const uint8_t sample = f * 53 + p * 71 + y * 17 + x * 3;
                    mu_assert("could not write frame sample",
                              fputc(sample, fin) != EOF);
                }
            }
        }
    }
    rewind(fin);

    video_input vid;
    int err = video_input_open(&vid, fin);
    mu_assert("could not open C422 input", !err);
    video_input_info info;
    video_input_get_info(&vid, &info);
    mu_assert("incorrect C422 format", info.pixel_fmt == PF_422);
    mu_assert("incorrect C422 bit depth", info.depth == 8);
    mu_assert("incorrect C422 dimensions", info.pic_w == 32 && info.pic_h == 24);

    VmafPicture pic;
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV422P, info.depth,
                             info.pic_w, info.pic_h);
    mu_assert("could not allocate picture", !err);
    for (unsigned f = 0; f < 2; f++) {
        int ret = video_input_fetch_into_vmaf_picture(&vid, &pic);
        mu_assert("could not read C422 frame directly", ret == 1);
        for (unsigned p = 0; p < 3; p++) {
            const unsigned w = p ? 16 : 32;
            const uint8_t *row = pic.data[p];
            for (unsigned y = 0; y < 24; y++) {
                for (unsigned x = 0; x < w; x++) {
                    const uint8_t sample = f * 53 + p * 71 + y * 17 + x * 3;
                    mu_assert("incorrect C422 sample", row[x] == sample);
                }
                row += pic.stride[p];
            }
        }
    }
    mu_assert("expected EOF after two C422 frames",
              video_input_fetch_into_vmaf_picture(&vid, &pic) == 0);

    vmaf_picture_unref(&pic);
    video_input_close(&vid);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_c422_direct_read);
    return NULL;
}
