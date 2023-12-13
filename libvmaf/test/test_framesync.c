#include "test.h"
#include "../src/framesync.c"

#include <stdint.h>

static int most_recent_callback_index = -1;

struct my_cookie {
    uint8_t my_data[256];
};

int my_framesync_callback(void *data, unsigned data_cnt, unsigned index)
{
    struct my_cookie* my_data = (struct my_cookie*)data;

    (void)my_data;
    (void)data_cnt;

    most_recent_callback_index = index;
    return 0;
}

static char *test_framesync()
{
    int err = 0;

    VmafFrameSyncContext *fs_ctx;

    int index_offsets[] = { -1, 0, +1 };
    unsigned index_offsets_cnt = sizeof(index_offsets) / sizeof(*index_offsets);

    VmafFrameSyncConfiguration cfg = {
        .data_sz = sizeof(struct my_cookie),
        .framesync_callback = my_framesync_callback,
        .index_offsets = index_offsets,
        .index_offsets_cnt = index_offsets_cnt,
    };

    err = vmaf_framesync_init(&fs_ctx, cfg);
    mu_assert("problem during vmaf_framesync_init", !err);

    mu_assert("no callbacks should have been executed yet, "
              "vmaf_framesync_init",
              most_recent_callback_index == -1);

    struct my_cookie *my_data = NULL;
    err = vmaf_framesync_fetch_data(fs_ctx, (void**)&my_data);
    mu_assert("problem during vmaf_framesync_fetch_data", !err);
    mu_assert("fetched data pointer is still null", my_data);

    err = vmaf_framesync_submit_data(fs_ctx, my_data, 0);
    mu_assert("problem during vmaf_framesync_submit_data", !err);

    mu_assert("no callbacks should have been executed yet, "
              "callback dependencies unmet",
              most_recent_callback_index == -1);

    err = vmaf_framesync_fetch_data(fs_ctx, (void**)&my_data);
    mu_assert("problem during vmaf_framesync_fetch_data", !err);
    mu_assert("fetched data pointer is still null", my_data);

    err = vmaf_framesync_submit_data(fs_ctx, my_data, 1);
    mu_assert("problem during vmaf_framesync_submit_data", !err);

    mu_assert("no callbacks should have been executed yet, "
              "callback dependencies unmet",
              most_recent_callback_index == -1);

    err = vmaf_framesync_fetch_data(fs_ctx, (void**)&my_data);
    mu_assert("problem during vmaf_framesync_fetch_data", !err);
    mu_assert("fetched data pointer is still null", my_data);

    err = vmaf_framesync_submit_data(fs_ctx, my_data, 2);
    mu_assert("problem during vmaf_framesync_submit_data", !err);

    mu_assert("callback should have been executed for index == 1, "
              "callback dependencies are met: (0, 1, 2)",
              most_recent_callback_index == 1);

    err = vmaf_framesync_close(fs_ctx);
    mu_assert("problem during vmaf_framesync_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_framesync);
    return NULL;
}
