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

#include "test.h"
#include "dict.h"

#include "dict.c"

static char *test_vmaf_dictionary()
{
    int err = 0;

    VmafDictionary *dict = NULL;
    const char *test_key_1 = "key_1";
    const char *test_val_1 = "val_1";
    err |= vmaf_dictionary_set(&dict, test_key_1, test_val_1, 0);
    mu_assert("problem during vmaf_dictionary_set", !err);
    mu_assert("new dictionary should have been created", dict);
    const unsigned initial_size = dict->size;
    mu_assert("this test asserts that initial_size is 8", initial_size == 8);
    const char *test_key_2 = "key_2";
    const char *test_val_2 = "val_2";
    err |= vmaf_dictionary_set(&dict, test_key_2, test_val_2, 0);
    const char *test_key_3 = "key_3";
    const char *test_val_3 = "val_3";
    err |= vmaf_dictionary_set(&dict, test_key_3, test_val_3, 0);
    const char *test_key_4 = "key_4";
    const char *test_val_4 = "val_4";
    err |= vmaf_dictionary_set(&dict, test_key_4, test_val_4, 0);
    const char *test_key_5 = "key_5";
    const char *test_val_5 = "val_5";
    err |= vmaf_dictionary_set(&dict, test_key_5, test_val_5, 0);
    const char *test_key_6 = "key_6";
    const char *test_val_6 = "val_6";
    err |= vmaf_dictionary_set(&dict, test_key_6, test_val_6, 0);
    const char *test_key_7 = "key_7";
    const char *test_val_7 = "val_7";
    err |= vmaf_dictionary_set(&dict, test_key_7, test_val_7, 0);
    const char *test_key_8 = "key_8";
    const char *test_val_8 = "val_8";
    err |= vmaf_dictionary_set(&dict, test_key_8, test_val_8, 0);
    mu_assert("problem during vmaf_dictionary_set", !err);
    mu_assert("dictionary should be completely full",
              dict->size == initial_size && dict->size == dict->cnt);
    const char *test_key_9 = "key_9";
    const char *test_val_9 = "val_9";
    err |= vmaf_dictionary_set(&dict, test_key_9, test_val_9, 0);
    mu_assert("problem during vmaf_dictionary_set", !err);
    mu_assert("dictionary capacity should have doubled",
              dict->cnt == initial_size + 1 && dict->size == initial_size * 2);

    VmafDictionaryEntry *entry = NULL;
    entry = vmaf_dictionary_get(&dict, "key_5", 0);
    mu_assert("dictionary should return an entry with valid key", entry);

    entry = vmaf_dictionary_get(&dict, "invalid_key", 0);
    mu_assert("dictionary should return NULL with invalid key", !entry);

    const char *pre_existing_key = "key_9";
    const char *new_value = "new_value";
    err = vmaf_dictionary_set(&dict, pre_existing_key, new_value,
                              VMAF_DICT_DO_NOT_OVERWRITE);
    mu_assert("vmaf_dictionary_set should fail with pre-existing key", err);
    entry = vmaf_dictionary_get(&dict, pre_existing_key, 0);
    mu_assert("dictionary should return original value with pre-existing key",
              !strcmp(entry->key, pre_existing_key) &&
              !strcmp(entry->val, "val_9"));
    err = vmaf_dictionary_set(&dict, pre_existing_key, "val_9",
                              VMAF_DICT_DO_NOT_OVERWRITE);
    mu_assert("vmaf_dictionary_set should not fail when pre-existing key "
              "matches pre-existing value", !err);
    err = vmaf_dictionary_set(&dict, pre_existing_key, new_value, 0);
    mu_assert("problem during vmaf_dictionary_set", !err);
    entry = vmaf_dictionary_get(&dict, pre_existing_key, 0);
    mu_assert("dictionary should return new value with pre-existing key",
              !strcmp(entry->key, pre_existing_key) &&
              !strcmp(entry->val, new_value));

    VmafDictionary *new_dict = NULL;
    err = vmaf_dictionary_copy(&dict, &new_dict);
    mu_assert("problem during vmaf_dictionary_copy", !err);
    mu_assert("new_dict should no longer be NULL", new_dict);
    mu_assert("new_dict should have a matching cnt",
              dict->cnt == new_dict->cnt);

    vmaf_dictionary_free(&dict);
    mu_assert("dictionary should be NULL after free", !dict);
    vmaf_dictionary_free(&new_dict);
    mu_assert("dictionary should be NULL after free", !new_dict);

    return NULL;
}

static char *test_vmaf_dictionary_merge()
{
    int err = 0;
    VmafDictionary *a = NULL;
    VmafDictionary *b = NULL;
    VmafDictionary *d = NULL;
    VmafDictionaryEntry *entry = NULL;

    d = vmaf_dictionary_merge(&a, &b, 0);
    mu_assert("merging two NULL dicts should result in a NULL dict", !d);

    err = vmaf_dictionary_set(&a, "key_a", "val_a", 0);
    mu_assert("problem during vmaf_dictionary_set", !err);
    d = vmaf_dictionary_merge(&a, &b, 0);
    mu_assert("merging one NULL and one non-NULL dict should work", d);
    entry = vmaf_dictionary_get(&d, "key_a", 0);
    mu_assert("dictionary should return an entry with valid key", entry);
    mu_assert("entry should have correct value", !strcmp(entry->val, "val_a"));
    vmaf_dictionary_free(&d);
    mu_assert("dictionary should be NULL after free", !d);
    d = vmaf_dictionary_merge(&b, &a, 0);
    mu_assert("merging one NULL and one non-NULL dict should work", d);
    entry = vmaf_dictionary_get(&d, "key_a", 0);
    mu_assert("dictionary should return an entry with valid key", entry);
    mu_assert("entry should have correct value", !strcmp(entry->val, "val_a"));
    vmaf_dictionary_free(&d);
    mu_assert("dictionary should be NULL after free", !d);

    err = vmaf_dictionary_set(&b, "key_b", "val_b", 0);
    mu_assert("problem during vmaf_dictionary_set", !err);
    d = vmaf_dictionary_merge(&b, &a, 0);
    mu_assert("merging two non-NULL dicts should work", d);
    entry = vmaf_dictionary_get(&d, "key_a", 0);
    mu_assert("dictionary should return an entry with valid key", entry);
    mu_assert("entry should have correct value", !strcmp(entry->val, "val_a"));
    entry = vmaf_dictionary_get(&d, "key_b", 0);
    mu_assert("dictionary should return an entry with valid key", entry);
    mu_assert("entry should have correct value", !strcmp(entry->val, "val_b"));
    vmaf_dictionary_free(&d);
    mu_assert("dictionary should be NULL after free", !d);

    err = vmaf_dictionary_set(&a, "duplicate_key", "val_a", 0);
    mu_assert("problem during vmaf_dictionary_set", !err);
    err = vmaf_dictionary_set(&a, "duplicate_key", "val_b", 0);
    mu_assert("problem during vmaf_dictionary_set", !err);

    d = vmaf_dictionary_merge(&b, &a, 0);
    mu_assert("merging two non-NULL dicts with duplicate keys should work", d);
    entry = vmaf_dictionary_get(&d, "duplicate_key", 0);
    mu_assert("dictionary should return an entry with valid key", entry);
    mu_assert("entry should have expected value", !strcmp(entry->val, "val_b"));
    vmaf_dictionary_free(&d);
    mu_assert("dictionary should be NULL after free", !d);

    err = vmaf_dictionary_set(&b, "duplicate_key", "val_c", 0);
    d = vmaf_dictionary_merge(&b, &a, VMAF_DICT_DO_NOT_OVERWRITE);
    mu_assert("dictionary should be NULL for duplicated key but different values", !d);

    vmaf_dictionary_free(&a);
    mu_assert("dictionary should be NULL after free", !a);
    vmaf_dictionary_free(&b);
    mu_assert("dictionary should be NULL after free", !b);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_vmaf_dictionary);
    mu_run_test(test_vmaf_dictionary_merge);
    return NULL;
}
