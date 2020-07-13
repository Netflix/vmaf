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

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

int vmaf_ref_init(VmafRef **ref)
{
    VmafRef *const r = *ref = malloc(sizeof(*r));
    if (!r) return -ENOMEM;
    memset(r, 0, sizeof(*r));
    atomic_init(&r->cnt, 1);
    return 0;
}

void vmaf_ref_fetch_increment(VmafRef *ref)
{
    atomic_fetch_add(&ref->cnt, 1);
}

void vmaf_ref_fetch_decrement(VmafRef *ref)
{
    atomic_fetch_sub(&ref->cnt, 1);
}

long vmaf_ref_load(VmafRef *ref)
{
    return atomic_load(&ref->cnt);
}

int vmaf_ref_close(VmafRef *ref)
{
    free(ref);
    return 0;
}
