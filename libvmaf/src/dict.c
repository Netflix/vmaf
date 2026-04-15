#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"
#include "libvmaf/feature.h"

VmafDictionaryEntry *vmaf_dictionary_get(VmafDictionary **dict,
                                         const char *key, uint64_t flags)
{
    if (!dict) return NULL;
    if (!(*dict)) return NULL;
    if (!key) return NULL;

    (void) flags; // available for possible future use

    VmafDictionary *d = *dict;
    for (unsigned i = 0; i < d->cnt; i++) {
       if (!strcmp(key, d->entry[i].key))
           return &d->entry[i];
    }

    return NULL;
}

/* NOLINTNEXTLINE(readability-function-size) */
int vmaf_dictionary_set(VmafDictionary **dict, const char *key, const char *val,
                        uint64_t flags)
{
    if (!dict) return -EINVAL;
    if (!key) return -EINVAL;
    if (!val) return -EINVAL;

    VmafDictionary *d = *dict;
    char *buf = NULL;
    const char *val_copy = NULL;
    const char *key_copy = NULL;

    if (!d) {
        d = *dict = malloc(sizeof(*d));
        if (!d) return -ENOMEM;
        memset(d, 0, sizeof(*d));
        const size_t initial_sz = 8 * sizeof(*d->entry);
        d->entry = malloc(initial_sz);
        if (!d->entry) {
            free(d);
            *dict = NULL;
            return -ENOMEM;
        }
        memset(d->entry, 0, initial_sz);
        d->size = 8;
    }

    if (flags & VMAF_DICT_NORMALIZE_NUMERICAL_VALUES) {
        char *end = NULL;
        double dv = strtof(val, &end);
        if (!(dv == 0 && val == end)) {
            const char *fmt = "%g";
            const int snp = snprintf(NULL, 0, fmt, dv);
            if (snp < 0) return -EINVAL;
            const size_t buf_sz = (size_t) snp + 1;
            buf = malloc(buf_sz);
            if (!buf) return -ENOMEM;
            (void) snprintf(buf, buf_sz, fmt, dv);
        }
    }

    val = buf ? buf : val;
    VmafDictionaryEntry *existing_entry = vmaf_dictionary_get(&d, key, 0);
    if (existing_entry && (flags & VMAF_DICT_DO_NOT_OVERWRITE)) {
        int ret = !strcmp(existing_entry->val, val) ? 0 : -EINVAL;
        free(buf);
        return ret;
    }

    /* Overwrite path — no realloc, so existing_entry stays valid. */
    if (existing_entry) {
        val_copy = strdup(val);
        if (!val_copy) goto fail;
        free((char*)existing_entry->val);
        existing_entry->val = val_copy;
        free(buf);
        return 0;
    }

    if (d->cnt == d->size) {
        assert(d->size > 0);
        const size_t sz = d->size * sizeof(*d->entry) * 2;
        VmafDictionaryEntry *entry =
            (VmafDictionaryEntry *) realloc(d->entry, sz);
        if (!entry) goto fail;
        d->entry = entry;
        d->size *= 2;
    }

    val_copy = strdup(val);
    if (!val_copy) goto fail;
    key_copy = strdup(key);
    if (!key_copy) goto fail;

    free(buf);

    VmafDictionaryEntry entry = {
        .key = key_copy,
        .val = val_copy,
    };

    d->entry[d->cnt++] = entry;

    return 0;

fail:
    free(buf);
    free((char*)val_copy);
    free((char*)key_copy);
    return -ENOMEM;
}

int vmaf_dictionary_copy(VmafDictionary **src, VmafDictionary **dst)
{
    if (!src) return -EINVAL;
    if (!(*src)) return -EINVAL;
    if (!dst) return -EINVAL;

    int err = 0;

    VmafDictionary *d = *src;
    for (unsigned i = 0; i < d->cnt; i++)
        err |= vmaf_dictionary_set(dst, d->entry[i].key, d->entry[i].val, 0);

    return err;
}

int vmaf_dictionary_free(VmafDictionary **dict)
{
    if (!dict) return -EINVAL;
    if (!(*dict)) return 0;

    VmafDictionary *d = *dict;
    for (unsigned i = 0; i < d->cnt; i++) {
       if (d->entry[i].key) free((char*)d->entry[i].key);
       if (d->entry[i].val) free((char*)d->entry[i].val);
    }
    free(d->entry);
    free(d);
    *dict = NULL;

    return 0;
}

VmafDictionary *vmaf_dictionary_merge(VmafDictionary **dict_a,
                                      VmafDictionary **dict_b,
                                      uint64_t flags)
{
    int err = 0;
    VmafDictionary *a = *dict_a;
    VmafDictionary *b = *dict_b;
    VmafDictionary *d = NULL;

    if (a) {
        err = vmaf_dictionary_copy(&a, &d);
        if (err) goto fail;
    }

    if (b) {
        for (unsigned i = 0; i < b->cnt; i++)
            err |= vmaf_dictionary_set(&d, b->entry[i].key, b->entry[i].val, flags);
        if (err) goto fail;
    }

    return d;

fail:
    (void) vmaf_dictionary_free(&d);
    return NULL;
}

int vmaf_dictionary_compare(VmafDictionary *a, VmafDictionary *b)
{
    if (!a && !b) return 0;
    if (!a != !b) return -EINVAL;
    if (a->cnt != b->cnt) return -EINVAL;

    for (unsigned i = 0; i < a->cnt; i++) {
        const VmafDictionaryEntry *e =
            vmaf_dictionary_get(&b, a->entry[i].key, 0);
        if (!e) return -EINVAL;
        if (strcmp(e->val, a->entry[i].val) != 0) return -EINVAL;
    }

    return 0;
}

static int alphabetical_compare(const void* a, const void* b)
{
    const VmafDictionaryEntry *entry_a = a;
    const VmafDictionaryEntry *entry_b = b;
    return strcmp(entry_a->key, entry_b->key);
}

void vmaf_dictionary_alphabetical_sort(VmafDictionary *dict)
{
    if (!dict) return;
    qsort(dict->entry, dict->cnt, sizeof(*dict->entry), alphabetical_compare);
}

static int isnumeric(const char *str)
{
    char *end = NULL;
    (void) strtof(str, &end);
    if (end == str) return 0;
    while (*end == ' ' || *end == '\t' || *end == '\n') end++;
    return *end == '\0';
}

int vmaf_feature_dictionary_set(VmafFeatureDictionary **dict, const char *key,
                                const char *val)
{
    uint64_t flags = 0;
    if (isnumeric(val)) flags |= VMAF_DICT_NORMALIZE_NUMERICAL_VALUES;
    return vmaf_dictionary_set((VmafDictionary**)dict, key, val, flags);
}

int vmaf_feature_dictionary_free(VmafFeatureDictionary **dict)
{
    return vmaf_dictionary_free((VmafDictionary**)dict);
}
