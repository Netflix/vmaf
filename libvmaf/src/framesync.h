typedef struct VmafFrameSyncConfiguration {
    size_t data_sz;
    int (*framesync_callback)(void *data, unsigned data_cnt, unsigned index);
    int *index_offsets;
    unsigned index_offsets_cnt;
} VmafFrameSyncConfiguration;

typedef struct VmafFrameSyncContext VmafFrameSyncContext;

int vmaf_framesync_init(VmafFrameSyncContext **fs_ctx, VmafFrameSyncConfiguration cfg);

int vmaf_framesync_fetch_data(VmafFrameSyncContext *fs_ctx, void **data);

int vmaf_framesync_submit_data(VmafFrameSyncContext *fs_ctx, void *data, unsigned index);

int vmaf_framesync_close(VmafFrameSyncContext *fs_ctx);
