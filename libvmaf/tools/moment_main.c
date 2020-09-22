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

#include <stdio.h>
#include <stdlib.h>

#include "read_frame.h"

int moment(int (*read_noref_frame)(float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt, int order);

static void usage(void)
{
    puts("usage: moment order fmt video w h\n"
         "order:\n"
         "\t1\n"
         "\t2\n"
         "fmts:\n"
         "\tyuv420p\n"
         "\tyuv422p\n"
         "\tyuv444p\n"
         "\tyuv420p10le\n"
         "\tyuv422p10le\n"
         "\tyuv444p10le\n"
         "\tyuv420p12le\n"
         "\tyuv422p12le\n"
         "\tyuv444p12le\n"
         "\tyuv420p16le\n"
         "\tyuv422p16le\n"
         "\tyuv444p16le"
    );
}

int run_moment(int order, const char *fmt, const char *video_path, int w, int h)
{
    int ret = 0;
    struct noref_data *s;
    s = (struct noref_data *)malloc(sizeof(struct noref_data));
    s->format = fmt;
    s->width = w;
    s->height = h;

    ret = get_frame_offset(fmt, w, h, &(s->offset));
    if (ret)
    {
        goto fail_or_end;
    }

    if (!(s->dis_rfile = fopen(video_path, "rb")))
    {
        fprintf(stderr, "fopen video_path %s failed.\n", video_path);
        ret = 1;
        goto fail_or_end;
    }

    ret = moment(read_noref_frame, s, w, h, fmt, order);

fail_or_end:
    if (s->dis_rfile)
    {
        fclose(s->dis_rfile);
    }
    if (s)
    {
        free(s);
    }
    return ret;
}

int main(int argc, const char **argv)
{
    const char *video_path;
    int order;
    const char *fmt;
    int w;
    int h;

    if (argc < 6) {
        usage();
        return 2;
    }

    order     = atoi(argv[1]);
    fmt         = argv[2];
    video_path = argv[3];
    w        = atoi(argv[4]);
    h        = atoi(argv[5]);

    if (w <= 0 || h <= 0) {
        usage();
        return 2;
    }

    if (!(order == 1 || order == 2))
    {
        usage();
        return 2;
    }

    return run_moment(order, fmt, video_path, w, h);
}
