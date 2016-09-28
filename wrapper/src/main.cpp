/**
 *
 *  Copyright 2016 Netflix, Inc.
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

#include <cstdio>
#include <exception>
#include <string>
#include <algorithm>

#include "vmaf.h"

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[])
{
    double score;
    char* fmt;
    int width;
    int height;
    char *ref_path;
    char *dis_path;
    char *model_path;
    char *log_path = NULL;
    char *log_fmt = NULL;
    bool disable_clip = false;
    bool do_psnr = false;
    bool do_ssim = false;
    bool do_ms_ssim = false;

    /* Check parameters */

    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s fmt width height ref_path dis_path model_path [--log log_path] [--log-fmt log_fmt] [--disable-clip] [--psnr] [--ssim] [--ms-ssim]\n", argv[0]);
        fprintf(stderr, "fmt:\n\tyuv420p\n\tyuv422p\n\tyuv444p\n\tyuv420p10le\n\tyuv422p10le\n\tyuv444p10le\n\n");
        fprintf(stderr, "log_fmt:\n\tjson (default)\n\txml\n\n");
        return -1;
    }

    try
    {
        fmt = argv[1];
        width = std::stoi(argv[2]);
        height = std::stoi(argv[3]);
        ref_path = argv[4];
        dis_path = argv[5];
        model_path = argv[6];

        if (width <= 0 || height <= 0)
        {
            fprintf(stderr, "%s: Invalid frame resolution %dx%d\n", argv[0], width, height);
        }

        log_path = getCmdOption(argv + 7, argv + argc, "--log");

        log_fmt = getCmdOption(argv + 7, argv + argc, "--log-fmt");
        if (log_fmt != NULL && !(strcmp(log_fmt, "xml")==0 || strcmp(log_fmt, "json")==0))
        {
            fprintf(stderr, "error: log_fmt must be xml or json, but is %s\n", log_fmt);
            return -1;
        }

        if (cmdOptionExists(argv + 7, argv + argc, "--disable-clip"))
        {
            disable_clip = true;
        }

        if (cmdOptionExists(argv + 7, argv + argc, "--psnr"))
        {
            do_psnr = true;
        }

        if (cmdOptionExists(argv + 7, argv + argc, "--ssim"))
        {
            do_ssim = true;
        }

        if (cmdOptionExists(argv + 7, argv + argc, "--ms-ssim"))
        {
            do_ms_ssim = true;
        }

        /* Run VMAF */
        score = RunVmaf(fmt, width, height, ref_path, dis_path, model_path, log_path, log_fmt, disable_clip, do_psnr, do_ssim, do_ms_ssim);

    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return -1;
    }

    return 0;
}
