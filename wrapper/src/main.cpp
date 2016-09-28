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
    char *log_file_path = NULL;
    bool disable_clip = false;
    bool do_psnr = false;
    bool do_ssim = false;
    bool do_ms_ssim = false;

    /* Check parameters */

    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s width height input_ref input_dis model [-f log] [--disable-clip] [--psnr] [--ssim] [--ms-ssim]\n", argv[0]);
        return -1;
    }

    try
    {
        int width = std::stoi(argv[1]);
        int height = std::stoi(argv[2]);
        char *ref_path = argv[3];
        char *dis_path = argv[4];
        char *svm_model_path = argv[5];

        if (width <= 0 || height <= 0)
        {
            fprintf(stderr, "%s: Invalid frame resolution %dx%d\n", argv[0], width, height);
        }

        log_file_path = getCmdOption(argv + 6, argv + argc, "-f");

        if (cmdOptionExists(argv + 6, argv + argc, "--disable-clip"))
        {
            disable_clip = true;
        }

        if (cmdOptionExists(argv + 6, argv + argc, "--psnr"))
        {
            do_psnr = true;
        }

        if (cmdOptionExists(argv + 6, argv + argc, "--ssim"))
        {
            do_ssim = true;
        }

        if (cmdOptionExists(argv + 6, argv + argc, "--ms-ssim"))
        {
            do_ms_ssim = true;
        }

        /* Run VMAF */

        printf("Start calculating VMAF score\n");
        score = RunVmaf(width, height, ref_path, dis_path, svm_model_path, log_file_path, disable_clip, do_psnr, do_ssim, do_ms_ssim);
        printf("VMAF score = %f\n", score);

    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return -1;
    }

    return 0;
}
