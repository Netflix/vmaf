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

#include "metadata_handler.h"
#include "test.h"

void set_meta() {}

static char *test_propagate_metadata_init()
{
    VmafCallbackList *propagate_metadata;
    int err = vmaf_metadata_init(&propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_init", !err);
    mu_assert("problem during vmaf_propagate_metadata_init, metadata is NULL",
              propagate_metadata);

    vmaf_metadata_destroy(propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_destroy", !err);

    return NULL;
}

static char *test_propagate_metadata_destroy()
{
    VmafCallbackList *propagate_metadata;
    int err = vmaf_metadata_init(&propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_init", !err);
    mu_assert("problem during vmaf_propagate_metadata_init, metadata is NULL",
              propagate_metadata);

    err = vmaf_metadata_destroy(propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_destroy", !err);

  return NULL;
}

static char *test_propagate_metadata_append()
{
    VmafCallbackList *propagate_metadata;
    int err = vmaf_metadata_init(&propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_init", !err);

    VmafMetadataConfig metadata_config;
    metadata_config.callback = set_meta;
    metadata_config.data = NULL;

    err = vmaf_metadata_append(propagate_metadata, &metadata_config);
    mu_assert("problem during vmaf_propagate_metadata_append", !err);
    mu_assert("problem during vmaf_propagate_metadata_append, metadata->head is NULL",
              propagate_metadata->head);
    mu_assert("problem during vmaf_propagate_metadata_append, metadata->head->next is not NULL",
              !propagate_metadata->head->next);

    err = vmaf_metadata_append(propagate_metadata, &metadata_config);
    mu_assert("problem during vmaf_propagate_metadata_append", !err);
    mu_assert("problem during vmaf_propagate_metadata_append, metadata->head->next is NULL",
              propagate_metadata->head->next);

    err = vmaf_metadata_destroy(propagate_metadata);
    mu_assert("problem during vmaf_propagate_metadata_destroy", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_propagate_metadata_init);
    mu_run_test(test_propagate_metadata_destroy);
    mu_run_test(test_propagate_metadata_append);
    return NULL;
}
