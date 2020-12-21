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

#ifndef __VMAF_SRC_LOG_H__
#define __VMAF_SRC_LOG_H__

#include "libvmaf/libvmaf.h"

void vmaf_set_log_level(enum VmafLogLevel log_level);
void vmaf_log(enum VmafLogLevel log_level, const char *fmt, ...);

#endif /* __VMAF_SRC_LOG_H__ */
