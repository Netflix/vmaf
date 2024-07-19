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

#include "metadata_handler.h"

int vmaf_metadata_init(VmafCallbackList **const metadata)
{
    if (!metadata) return -EINVAL;

    VmafCallbackList *const metadata_s = *metadata =
        malloc(sizeof(*metadata_s));
    if (!metadata_s) goto fail;

    metadata_s->head = NULL;

    return 0;

fail:
    return -ENOMEM;
}

int vmaf_metadata_append(VmafCallbackList *metadata, const VmafMetadataConfig *metadata_config)
{
    if (!metadata) return -EINVAL;
    if (!metadata_config) return -EINVAL;
    if (!metadata_config->callback) return -EINVAL;

    VmafCallbackItem *node = malloc(sizeof(*node));
    if (!node) goto fail;
    memset(node, 0, sizeof(*node));

    node->data = metadata_config->data;
    node->callback = metadata_config->callback;

    if (!metadata->head) {
        metadata->head = node;
    } else {
        VmafCallbackItem *iter = metadata->head;
        while (iter->next) iter = iter->next;
        iter->next = node;
    }

    return 0;

fail:
    return -ENOMEM;
}

int vmaf_metadata_destroy(VmafCallbackList *metadata)
{
    if (!metadata) return -EINVAL;

    VmafCallbackItem *iter = metadata->head;
    while (iter) {
        VmafCallbackItem *next = iter->next;
        free(iter);
        iter = next;
    }

    free(metadata);

    return 0;
}
