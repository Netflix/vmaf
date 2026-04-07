/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
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

#ifndef __VMAF_SRC_PICTURE_POOL_H__
#define __VMAF_SRC_PICTURE_POOL_H__

#include "picture.h"

typedef struct VmafPicturePoolConfig {
    unsigned pic_cnt;
    unsigned w;
    unsigned h;
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
} VmafPicturePoolConfig;

typedef struct VmafPicturePool VmafPicturePool;

int vmaf_picture_pool_init(VmafPicturePool **pool,
                           VmafPicturePoolConfig cfg);

int vmaf_picture_pool_close(VmafPicturePool *pool);

int vmaf_picture_pool_fetch(VmafPicturePool *pool, VmafPicture *pic);

#endif /* __VMAF_SRC_PICTURE_POOL_H__ */
