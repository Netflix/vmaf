#include "framesync.h"

typedef struct VmafFrameSyncContext {
    VmafFrameSyncConfiguration cfg;
} VmafFrameSyncContext;

int vmaf_framesync_init(VmafFrameSyncContext **fs_ctx, VmafFrameSyncConfiguration cfg)
{
    return 0;
}

int vmaf_framesync_fetch_data(VmafFrameSyncContext *fs_ctx, void **data)
{
    return 0;
}

int vmaf_framesync_submit_data(VmafFrameSyncContext *fs_ctx, void *data, unsigned index)
{
    return 0;
}

int vmaf_framesync_close(VmafFrameSyncContext *fs_ctx)
{
    return 0;
}
