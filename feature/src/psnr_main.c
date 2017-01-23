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

#include <stdlib.h>
#include <stdio.h>

int psnr(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

static void usage(void)
{
    puts("usage: psnr fmt ref dis w h\n"
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
    const char *ref_path;
    const char *dis_path;
    const char *fmt;
    int w;
    int h;
    int ret;

    if (argc < 6) {
        usage();
        return 2;
    }

    fmt         = argv[1];
    ref_path = argv[2];
    dis_path = argv[3];
    w        = atoi(argv[4]);
    h        = atoi(argv[5]);

    if (w <= 0 || h <= 0) {
        usage();
        return 2;
    }

    ret = psnr(ref_path, dis_path, w, h, fmt);

    if (ret)
        return ret;

    return 0;
}
