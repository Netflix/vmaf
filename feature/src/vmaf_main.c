/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
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

#include "common/cpu.h"

int adm(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int ansnr(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int vif(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int motion(const char *dis_path, int w, int h, const char *fmt);
int all(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

enum vmaf_cpu cpu; // global

static void usage(void)
{
    puts("usage: vmaf app fmt ref dis w h\n"
         "apps:\n"
         "\tadm\n"
         "\tansnr\n"
         "\tmotion\n"
         "\tvif\n"
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

int main(int argc, const char **argv)
{
    const char *app;
    const char *ref_path;
    const char *dis_path;
    const char *fmt;
    int w;
    int h;
    int ret;

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

    cpu = cpu_autodetect();

    if (!strcmp(app, "adm"))
        ret = adm(ref_path, dis_path, w, h, fmt);
    else if (!strcmp(app, "ansnr"))
        ret = ansnr(ref_path, dis_path, w, h, fmt);
    else if (!strcmp(app, "vif"))
        ret = vif(ref_path, dis_path, w, h, fmt);
    else if (!strcmp(app, "motion"))
        ret = motion(ref_path, w, h, fmt);
    else if (!strcmp(app, "all"))
        ret = all(ref_path, dis_path, w, h, fmt);
    else
        return 2;

    if (ret)
        return ret;

    return 0;
}
