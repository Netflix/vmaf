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

int moment(const char *path, int w, int h, const char *fmt, int order);

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
         "\tyuv444p10le"
    );
}

int main(int argc, const char **argv)
{
    const char *video_path;
    int order;
    const char *fmt;
    int w;
    int h;
    int ret;

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

    ret = moment(video_path, w, h, fmt, order);

    if (ret)
        return ret;

    return 0;
}
