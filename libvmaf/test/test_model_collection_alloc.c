/**
 *
 *  Copyright 2026 Lusoris
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

#include <stdlib.h>
#include <string.h>

#include "test.h"

static int fail_realloc;
static void *test_realloc(void *ptr, size_t size)
{
    return fail_realloc ? NULL : realloc(ptr, size);
}

/* Exercise the real append operation with a deterministic growth failure. */
#define realloc test_realloc
#include "model.c"
#undef realloc

static VmafModel *new_model(void)
{
    VmafModel *model = calloc(1, sizeof(*model));
    if (!model) return NULL;
    model->name = strdup("test_0000");
    if (!model->name) {
        free(model);
        return NULL;
    }
    model->type = VMAF_MODEL_BOOTSTRAP_SVM_NUSVR;
    return model;
}

static char *test_collection_growth_failure_preserves_ownership(void)
{
    VmafModelCollection *collection = NULL;
    for (unsigned i = 0; i < 8; i++) {
        VmafModel *model = new_model();
        mu_assert("model allocation failed", model);
        mu_assert("initial append failed",
                  !vmaf_model_collection_append(&collection, model));
    }
    mu_assert("fixture did not fill initial capacity", collection->cnt == collection->size);
    VmafModelCollection *original = collection;
    VmafModel **members = collection->model;
    VmafModel *first = members[0];
    const unsigned count = collection->cnt, capacity = collection->size;
    VmafModel *next = new_model();
    mu_assert("model allocation failed", next);

    fail_realloc = 1;
    int err = vmaf_model_collection_append(&collection, next);
    fail_realloc = 0;
    mu_assert("allocation failure not reported", err == -ENOMEM);
    mu_assert("allocation failure lost the existing collection", collection == original);
    mu_assert("allocation failure changed the model array", collection->model == members);
    mu_assert("allocation failure changed the count", collection->cnt == count);
    mu_assert("allocation failure changed the capacity", collection->size == capacity);
    mu_assert("allocation failure changed an existing member", collection->model[0] == first);

    /* The failed append retains caller ownership of next. Retrying must
     * append it exactly once, preserving all earlier models for destruction. */
    mu_assert("retry failed", !vmaf_model_collection_append(&collection, next));
    mu_assert("retry appended the wrong model", collection->model[count] == next);
    mu_assert("retry appended more than once", collection->cnt == count + 1);
    vmaf_model_collection_destroy(collection);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_collection_growth_failure_preserves_ownership);
    return NULL;
}
