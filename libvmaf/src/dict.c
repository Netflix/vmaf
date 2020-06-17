#include <errno.h>
#include <string.h>

#include "dict.h"

typedef struct VmafDictionary {
    VmafDictionaryEntry *entry;
    unsigned size, cnt;
} VmafDictionary;

const VmafDictionaryEntry *vmaf_dictionary_get(VmafDictionary **dict,
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

int vmaf_dictionary_set(VmafDictionary **dict, const char *key, const char *val,
                        uint64_t flags)
{
    if (!dict) return -EINVAL;
    if (!key) return -EINVAL;
    if (!val) return -EINVAL;

    VmafDictionary *d = *dict;

    if (!d) {
        d = *dict = malloc(sizeof(*d));
        if (!d) goto fail;
        memset(d, 0, sizeof(*d));
        const size_t initial_sz = 8 * sizeof(*d->entry);
        d->entry = malloc(initial_sz);
        if (!d->entry) {
            free(d);
            *dict = NULL;
            goto fail;
        }
        memset(d->entry, 0, initial_sz);
        d->size = 8;
    }

    VmafDictionaryEntry *existing_entry = vmaf_dictionary_get(&d, key, 0);
    if (existing_entry && (flags & VMAF_DICT_DO_NOT_OVERWRITE))
        return -EINVAL;

    if (d->cnt == d->size) {
        const size_t sz = d->size * sizeof(*d->entry) * 2;
        VmafDictionaryEntry *entry = realloc(d->entry, sz);
        if (!entry) goto fail;
        d->entry = entry;
        d->size *= 2;
    }

    const char *val_copy = strdup(val);
    if (!val_copy) goto fail;

    if (existing_entry && !(flags & VMAF_DICT_DO_NOT_OVERWRITE)) {
        free(existing_entry->val);
        existing_entry->val = val_copy;
        return 0;
    }

    const char *key_copy = strdup(key);
    if (!key_copy) goto free_val_copy;

    VmafDictionaryEntry entry = {
        .key = key_copy,
        .val = val_copy,
    };

    d->entry[d->cnt++] = entry;

    return 0;

free_val_copy:
    free(val_copy);
fail:
    return -ENOMEM;
}

int vmaf_dictionary_free(VmafDictionary **dict)
{
    if (!dict) return -EINVAL;
    if (!(*dict)) return 0;

    VmafDictionary *d = *dict;
    for (unsigned i = 0; i < d->cnt; i++) {
       if (d->entry[i].key) free(d->entry[i].key);
       if (d->entry[i].val) free(d->entry[i].val);
    }
    free(d->entry);
    free(d);
    *dict = NULL;

    return 0;
}
