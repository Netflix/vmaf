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

#include "libvmaf/libvmaf.h"

#include <stdarg.h>
#include <unistd.h>

static enum VmafLogLevel vmaf_log_level = VMAF_LOG_LEVEL_INFO;
static int istty = 0;

void vmaf_set_log_level(enum VmafLogLevel level)
{
    level = level < VMAF_LOG_LEVEL_NONE ? VMAF_LOG_LEVEL_NONE : level;
    level = level > VMAF_LOG_LEVEL_DEBUG ? VMAF_LOG_LEVEL_DEBUG : level;
    vmaf_log_level = level;
    istty = isatty(fileno(stderr));
}

static const char *level_str[] = {
    [VMAF_LOG_LEVEL_ERROR] = "ERROR",
    [VMAF_LOG_LEVEL_WARNING] = "WARNING",
    [VMAF_LOG_LEVEL_INFO] = "INFO",
    [VMAF_LOG_LEVEL_DEBUG] = "DEBUG",
};

static const char *level_str_color[] = {
    [VMAF_LOG_LEVEL_ERROR] = "\x1B[31m",
    [VMAF_LOG_LEVEL_WARNING] = "\x1B[33m",
    [VMAF_LOG_LEVEL_INFO] = "\x1B[32m",
    [VMAF_LOG_LEVEL_DEBUG] = "\x1B[34m",
};

void vmaf_log(enum VmafLogLevel level, const char *fmt, ...)
{
    if (level <= VMAF_LOG_LEVEL_NONE) return;
    if (level > vmaf_log_level) return;

    va_list(args);
    fprintf(stderr, "%slibvmaf%s %s%s%s ",
            istty ? "\x1B[35m" : "",
            istty ? "\x1B[0m" : "",
            istty ? level_str_color[level] : "",
            level_str[level],
            istty ? "\x1B[0m" : "");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
}
