#include "test.h"
#include "libvmaf/libvmaf.rc.h"

static char *test_context_init_and_close()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg;

    vmaf_default_configuration(&cfg);

    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_context_init_and_close);
    return NULL;
}
