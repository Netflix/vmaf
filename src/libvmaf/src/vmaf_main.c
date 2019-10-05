/**
 *
 *  Copyright 2016-2019 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
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
#include <string.h>

#include "common/frame.h"
#include "common/cpu.h"

int adm(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt);
int ansnr(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt);
int vif(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt);
int vifdiff(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt);
int motion(int (*read_noref_frame)(float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt);
int all(int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), void *user_data, int w, int h, const char *fmt);

enum vmaf_cpu cpu; // global

static void usage(void)
{
    puts("usage: vmaf app fmt ref dis w h\n"
         "apps:\n"
         "\tadm\n"
         "\tansnr\n"
         "\tmotion\n"
         "\tvif\n"
         "\tvifdiff\n"
         "\tall\n"
         "fmts:\n"
         "\tyuv420p\n"
         "\tyuv422p\n"
         "\tyuv444p\n"
         "\tyuv420p10le\n"
         "\tyuv422p10le\n"
         "\tyuv444p10le"
    );
}

int run_vmaf(const char *app, const char *fmt, const char *ref_path, const char *dis_path, int w, int h)
{
    int ret = 0;
    cpu = cpu_autodetect();

    if (!strcmp(app, "motion"))
    {
        struct noref_data *s;
        s = (struct noref_data *)malloc(sizeof(struct noref_data));
        s->format = fmt;
        s->width = w;
        s->height = h;

        ret = get_frame_offset(fmt, w, h, &(s->offset));
        if (ret)
        {
            goto fail_or_end_noref;
        }

        /* NOTE: below is legitimate: noref_data has a field called dis_rfile,
         * but what's needed to be passed to motion is ref_path!
         */
        if (!(s->dis_rfile = fopen(ref_path, "rb")))
        {
            fprintf(stderr, "fopen ref_path %s failed.\n", ref_path);
            ret = 1;
            goto fail_or_end_noref;
        }

        ret = motion(read_noref_frame, s, w, h, fmt);

fail_or_end_noref:
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
    else
    {
        struct data *s;
        s = (struct data *)malloc(sizeof(struct data));
        s->format = fmt;
        s->width = w;
        s->height = h;

        ret = get_frame_offset(fmt, w, h, &(s->offset));
        if (ret)
        {
            goto fail_or_end;
        }

        if (!(s->ref_rfile = fopen(ref_path, "rb")))
        {
            fprintf(stderr, "fopen ref_path %s failed.\n", ref_path);
            ret = 1;
            goto fail_or_end;
        }
        if (!(s->dis_rfile = fopen(dis_path, "rb")))
        {
            fprintf(stderr, "fopen ref_path %s failed.\n", dis_path);
            ret = 1;
            goto fail_or_end;
        }

        if (!strcmp(app, "adm"))
            ret = adm(read_frame, s,  w, h, fmt);
        else if (!strcmp(app, "ansnr"))
            ret = ansnr(read_frame, s, w, h, fmt);
        else if (!strcmp(app, "vif"))
            ret = vif(read_frame, s, w, h, fmt);
        else if (!strcmp(app, "all"))
            ret = all(read_frame, s, w, h, fmt);
        else if (!strcmp(app, "vifdiff"))
            ret = vifdiff(read_frame, s, w, h, fmt);
        else
            ret = 2;

fail_or_end:
        if (s->ref_rfile)
        {
            fclose(s->ref_rfile);
        }
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
}

int main(int argc, const char **argv)
{
    const char *app;
    const char *ref_path;
    const char *dis_path;
    const char *fmt;
    int w;
    int h;

    if (argc < 7) {
        usage();
        return 2;
    }

    app      = argv[1];
    fmt         = argv[2];
    ref_path = argv[3];
    dis_path = argv[4];
    w        = atoi(argv[5]);
    h        = atoi(argv[6]);

    if (w <= 0 || h <= 0) {
        usage();
        return 2;
    }

    return run_vmaf(app, fmt, ref_path, dis_path, w, h);
}
