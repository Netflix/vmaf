/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
#include <string.h>

#include "test.h"
#include "vidinput.h"
#include "libvmaf/picture.h"

#define FRAME_CNT 2

/* A sample is a function of its plane, its position in the plane as the file
 * stores it, and the frame index, so that a plane read at the wrong offset or
 * with the wrong row pitch cannot match by accident. */
static uint8_t sample_of(unsigned plane, unsigned row, unsigned col,
                         unsigned frame)
{
    return (uint8_t) (17 * plane + 31 * row + 7 * col + 101 * frame + 1);
}

static void plane_dims(unsigned plane, unsigned w, unsigned h,
                       unsigned dec_h, unsigned dec_v,
                       unsigned *pw, unsigned *ph)
{
    const unsigned dh = plane ? dec_h : 1;
    const unsigned dv = plane ? dec_v : 1;
    *pw = (w + dh - 1) / dh;
    *ph = (h + dv - 1) / dv;
}

/* Writes FRAME_CNT frames in the layout a raw file uses: every plane holds
 * ceil(dimension / decimation) samples. */
static FILE *write_clip(unsigned w, unsigned h, unsigned dec_h,
                        unsigned dec_v, const char *y4m_header)
{
    FILE *f = tmpfile();
    if (!f) return NULL;

    if (y4m_header) {
        const size_t len = strlen(y4m_header);
        if (fwrite(y4m_header, 1, len, f) != len) goto fail;
    }

    for (unsigned n = 0; n < FRAME_CNT; n++) {
        if (y4m_header && fwrite("FRAME\n", 1, 6, f) != 6) goto fail;
        for (unsigned p = 0; p < 3; p++) {
            unsigned pw, ph;
            plane_dims(p, w, h, dec_h, dec_v, &pw, &ph);
            for (unsigned i = 0; i < ph; i++) {
                for (unsigned j = 0; j < pw; j++) {
                    const uint8_t s = sample_of(p, i, j, n);
                    if (fwrite(&s, 1, 1, f) != 1) goto fail;
                }
            }
        }
    }

    rewind(f);
    return f;

fail:
    fclose(f);
    return NULL;
}

/* Every sample the picture carries must be the sample the file holds at the
 * same position of the same plane, for every frame of the clip. */
static char *check_clip(video_input *vid, unsigned w, unsigned h,
                        unsigned dec_h, unsigned dec_v, const char *what)
{
    for (unsigned n = 0; n < FRAME_CNT; n++) {
        VmafPicture pic;
        int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, w, h);
        mu_assert("problem during vmaf_picture_alloc", !err);

        const int ret = video_input_fetch_into_vmaf_picture(vid, &pic);
        if (ret != 1) {
            vmaf_picture_unref(&pic);
            mu_assert(what, ret == 1);
        }

        for (unsigned p = 0; p < 3; p++) {
            unsigned pw, ph;
            plane_dims(p, w, h, dec_h, dec_v, &pw, &ph);
            /* The picture carries the floor of the decimation and the file
               the ceiling, so the picture holds the leading part of a row. */
            mu_assert("picture plane is larger than the file plane",
                      pic.w[p] <= pw && pic.h[p] <= ph);
            const uint8_t *data = pic.data[p];
            for (unsigned i = 0; i < pic.h[p]; i++) {
                for (unsigned j = 0; j < pic.w[p]; j++) {
                    if (data[i * pic.stride[p] + j] != sample_of(p, i, j, n)) {
                        vmaf_picture_unref(&pic);
                        mu_assert(what, 0);
                    }
                }
            }
        }
        vmaf_picture_unref(&pic);
    }

    /* The clip is exhausted, not merely out of step with the reader. */
    VmafPicture pic;
    int err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, w, h);
    mu_assert("problem during vmaf_picture_alloc", !err);
    const int ret = video_input_fetch_into_vmaf_picture(vid, &pic);
    vmaf_picture_unref(&pic);
    mu_assert("the reader found a frame past the end of the clip", ret == 0);

    return NULL;
}

static char *test_yuv_even_dimensions()
{
    FILE *f = write_clip(20, 20, 2, 2, NULL);
    mu_assert("could not create the test clip", f);

    video_input vid;
    const int err = raw_input_open(&vid, f, 20, 20, VMAF_PIX_FMT_YUV420P, 8);
    mu_assert("problem during raw_input_open", !err);

    char *msg = check_clip(&vid, 20, 20, 2, 2,
                           "a 20x20 4:2:0 raw clip did not read back");
    video_input_close(&vid);
    return msg;
}

static char *test_yuv_odd_dimensions()
{
    /* 19x19 4:2:0: the chroma planes are 10x10 in the file and 9x9 in the
       picture, so every frame holds 38 samples the picture does not. */
    FILE *f = write_clip(19, 19, 2, 2, NULL);
    mu_assert("could not create the test clip", f);

    video_input vid;
    const int err = raw_input_open(&vid, f, 19, 19, VMAF_PIX_FMT_YUV420P, 8);
    mu_assert("problem during raw_input_open", !err);

    char *msg = check_clip(&vid, 19, 19, 2, 2,
                           "a 19x19 4:2:0 raw clip did not read back");
    video_input_close(&vid);
    return msg;
}

static char *test_y4m_odd_dimensions()
{
    FILE *f = write_clip(19, 19, 2, 2,
                         "YUV4MPEG2 W19 H19 F25:1 Ip A1:1 C420jpeg\n");
    mu_assert("could not create the test clip", f);

    video_input vid;
    const int err = video_input_open(&vid, f);
    mu_assert("problem during video_input_open", !err);

    char *msg = check_clip(&vid, 19, 19, 2, 2,
                           "a 19x19 4:2:0 y4m clip did not read back");
    video_input_close(&vid);
    return msg;
}

char *run_tests()
{
    mu_run_test(test_yuv_even_dimensions);
    mu_run_test(test_yuv_odd_dimensions);
    mu_run_test(test_y4m_odd_dimensions);
    return NULL;
}
