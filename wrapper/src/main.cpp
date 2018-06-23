/**
 *
 *  Copyright 2016-2018 Netflix, Inc.
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
#include <stdlib.h>
#include <stdexcept>
#include <exception>
#include <string>
#include <algorithm>
#include <cstring>
#include "libvmaf.h"

extern "C" {
#include "common/frame.h"
}

#define read_image_b       read_image_b2s
#define read_image_w       read_image_w2s

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

void print_usage(int argc, char *argv[])
{
    fprintf(stderr, "Usage: %s fmt width height ref_path dis_path model_path [--log log_path] [--log-fmt log_fmt] [--thread n_thread] [--subsample n_subsample] [--disable-clip] [--disable-avx] [--psnr] [--ssim] [--ms-ssim] [--phone-model] [--ci]\n", argv[0]);
    fprintf(stderr, "fmt:\n\tyuv420p\n\tyuv422p\n\tyuv444p\n\tyuv420p10le\n\tyuv422p10le\n\tyuv444p10le\n\n");
    fprintf(stderr, "log_fmt:\n\txml (default)\n\tjson\n\n");
    fprintf(stderr, "n_thread:\n\tmaximum threads to use (default 0 - use all threads)\n\n");
    fprintf(stderr, "n_subsample:\n\tn indicates computing on one of every n frames (default 1)\n\n");
}

int run_wrapper(char *fmt, int width, int height, char *ref_path, char *dis_path, char *model_path,
        char *log_path, char *log_fmt, bool disable_clip, bool disable_avx, bool enable_transform, bool phone_model,
        bool do_psnr, bool do_ssim, bool do_ms_ssim, char *pool_method, int n_thread, int n_subsample, bool enable_conf_interval)
{
    double score;

    int ret = 0;
    struct data *s;
    s = (struct data *)malloc(sizeof(struct data));
    s->format = fmt;
    s->width = width;
    s->height = height;
    s->ref_rfile = NULL;
    s->dis_rfile = NULL;

    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv420p10le"))
    {
        if ((width * height) % 2 != 0)
        {
            fprintf(stderr, "(width * height) %% 2 != 0, width = %d, height = %d.\n", width, height);
            ret = 1;
            goto fail_or_end;
        }
        s->offset = width * height / 2;
    }
    else if (!strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv422p10le"))
    {
        s->offset = width * height;
    }
    else if (!strcmp(fmt, "yuv444p") || !strcmp(fmt, "yuv444p10le"))
    {
        s->offset = width * height * 2;
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
        ret = 1;
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

    /* Run VMAF */
    ret = compute_vmaf(&score, fmt, width, height, read_frame, s, model_path, log_path, log_fmt,
                       disable_clip, disable_avx, enable_transform, phone_model, do_psnr, do_ssim,
                       do_ms_ssim, pool_method, n_thread, n_subsample, enable_conf_interval);

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

int main(int argc, char *argv[])
{
    char* fmt;
    int width;
    int height;
    char* ref_path;
    char* dis_path;
    char *model_path;
    char *log_path = NULL;
    char *log_fmt = NULL;
    bool disable_clip = false;
    bool disable_avx = false;
    bool enable_transform = false;
    bool phone_model = false;
    bool do_psnr = false;
    bool do_ssim = false;
    bool do_ms_ssim = false;
    char *pool_method = NULL;
    int n_thread = 0;
    int n_subsample = 1;
    bool enable_conf_interval = false;
    char *temp;

    /* Check parameters */

    if (argc < 7)
    {
        print_usage(argc, argv);
        return -1;
    }

    fmt = argv[1];

    try
    {
        width = std::stoi(argv[2]);
        height = std::stoi(argv[3]);
    }
    catch (std::logic_error& e)
    {
        fprintf(stderr, "Error: Invalid width/height format: %s\n", e.what());
        print_usage(argc, argv);
        return -1;
    }

    if (width <= 0 || height <= 0)
    {
        fprintf(stderr, "Error: Invalid width/height value: %d, %d\n", width, height);
        print_usage(argc, argv);
        return -1;
    }

    ref_path = argv[4];
    dis_path = argv[5];
    model_path = argv[6];

    log_path = getCmdOption(argv + 7, argv + argc, "--log");

    log_fmt = getCmdOption(argv + 7, argv + argc, "--log-fmt");
    if (log_fmt != NULL && !(strcmp(log_fmt, "xml")==0 || strcmp(log_fmt, "json")==0))
    {
        fprintf(stderr, "Error: log_fmt must be xml or json, but is %s\n", log_fmt);
        return -1;
    }

    temp = getCmdOption(argv + 7, argv + argc, "--thread");
    if (temp)
    {
        try
        {
            n_thread = std::stoi(temp);
        }
        catch (std::logic_error& e)
        {
            fprintf(stderr, "Error: Invalid n_thread format: %s\n", e.what());
            print_usage(argc, argv);
            return -1;
        }
    }
    if (n_thread < 0)
    {
        fprintf(stderr, "Error: Invalid n_thread value: %d\n", n_thread);
        print_usage(argc, argv);
        return -1;
    }

    temp = getCmdOption(argv + 7, argv + argc, "--subsample");
    if (temp)
    {
        try
        {
            n_subsample = std::stoi(temp);
        }
        catch (std::logic_error& e)
        {
            fprintf(stderr, "Error: Invalid n_subsample format: %s\n", e.what());
            print_usage(argc, argv);
            return -1;
        }
    }
    if (n_subsample <= 0)
    {
        fprintf(stderr, "Error: Invalid n_subsample value: %d\n", n_subsample);
        print_usage(argc, argv);
        return -1;
    }

    if (cmdOptionExists(argv + 7, argv + argc, "--disable-clip"))
    {
        disable_clip = true;
    }

    if (cmdOptionExists(argv + 7, argv + argc, "--disable-avx"))
    {
        disable_avx = true;
    }

    if (cmdOptionExists(argv + 7, argv + argc, "--enable-transform"))
    {
        enable_transform = true;
    }

    if (cmdOptionExists(argv + 7, argv + argc, "--phone-model"))
    {
        phone_model = true;
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

    pool_method = getCmdOption(argv + 7, argv + argc, "--pool");
    if (pool_method != NULL && !(strcmp(pool_method, "min")==0 || strcmp(pool_method, "harmonic_mean")==0 || strcmp(pool_method, "mean")==0))
    {
        fprintf(stderr, "Error: pool_method must be min, harmonic_mean or mean, but is %s\n", pool_method);
        return -1;
    }

    if (cmdOptionExists(argv + 7, argv + argc, "--ci"))
    {
        enable_conf_interval = true;
    }

    try
    {
        return run_wrapper(fmt, width, height, ref_path, dis_path, model_path,
                log_path, log_fmt, disable_clip, disable_avx, enable_transform, phone_model,
                do_psnr, do_ssim, do_ms_ssim, pool_method, n_thread, n_subsample, enable_conf_interval);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Error: %s\n", e.what());
        print_usage(argc, argv);
        return -1;
    }
    
}
