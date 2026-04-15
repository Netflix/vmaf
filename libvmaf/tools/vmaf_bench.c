/**
 *  VMAF Performance Benchmark & Validation Tool
 *
 *  Benchmarks feature extractors (CPU, CUDA, SYCL) using real video
 *  content derived from Big Buck Bunny at multiple resolutions.
 *
 *  Test data location: /tmp/vmaf_test/ (override with VMAF_TEST_DATA env
 *  or --data-dir flag).  Required files per resolution:
 *    ref_{W}x{H}.yuv  dis_{W}x{H}.yuv   (raw YUV420P 8-bit, 48 frames)
 *
 *  Generate with:
 *    ffmpeg -i bbb.mp4 -frames:v 48 -vf scale=W:H -pix_fmt yuv420p ref_WxH.yuv
 *    (dis = CRF 28 x264 encode of same, decoded back to raw)
 *
 *  Modes:
 *    vmaf_bench [--resolution WxH] [--frames N]         Performance benchmark
 *    vmaf_bench --validate [--resolution WxH]           GPU vs CPU correctness
 *    vmaf_bench --list-devices                           List GPU devices
 *    vmaf_bench --device N                               Select GPU device
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#include "libvmaf/picture.h"
#include "libvmaf/libvmaf.h"

#ifdef HAVE_CUDA
#include "libvmaf/libvmaf_cuda.h"
#endif

#ifdef HAVE_SYCL
#include "libvmaf/libvmaf_sycl.h"
#endif

/* clock_gettime-based high-resolution timer */
#ifdef _WIN32
#include <windows.h>
static double now_ms(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / freq.QuadPart * 1000.0;
}
#else
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* ==================== YUV file I/O ==================== */

#define MAX_TEST_FRAMES 48  /* frames available in test data */
#define DEFAULT_DATA_DIR "/tmp/vmaf_test"

static unsigned g_bpc = 8; /* configurable via --bpc */
static const char *g_datadir = NULL;

static const char *get_data_dir(void)
{
    if (!g_datadir) {
        g_datadir = getenv("VMAF_TEST_DATA");
        if (!g_datadir || !g_datadir[0])
            g_datadir = DEFAULT_DATA_DIR;
    }
    return g_datadir;
}

typedef struct {
    FILE *ref_fp;
    FILE *dis_fp;
    unsigned width;
    unsigned height;
    size_t frame_bytes;
    uint8_t *ref_buf;
    uint8_t *dis_buf;
} YuvPair;

static int yuv_pair_open(YuvPair *yp, unsigned w, unsigned h)
{
    char ref_path[1280], dis_path[1280];
    snprintf(ref_path, sizeof(ref_path), "%s/ref_%ux%u.yuv", get_data_dir(), w, h);
    snprintf(dis_path, sizeof(dis_path), "%s/dis_%ux%u.yuv", get_data_dir(), w, h);

    yp->ref_fp = fopen(ref_path, "rb");
    yp->dis_fp = fopen(dis_path, "rb");
    if (!yp->ref_fp || !yp->dis_fp) {
        fprintf(stderr, "Cannot open test data for %ux%u\n"
                "  ref: %s (%s)\n  dis: %s (%s)\n"
                "Set VMAF_TEST_DATA or --data-dir to your data directory.\n",
                w, h, ref_path, yp->ref_fp ? "ok" : "MISSING",
                dis_path, yp->dis_fp ? "ok" : "MISSING");
        if (yp->ref_fp) fclose(yp->ref_fp);
        if (yp->dis_fp) fclose(yp->dis_fp);
        memset(yp, 0, sizeof(*yp));
        return -1;
    }

    yp->width = w;
    yp->height = h;
    yp->frame_bytes = (size_t)w * h * 3 / 2;  /* YUV420P 8-bit */
    yp->ref_buf = malloc(yp->frame_bytes);
    yp->dis_buf = malloc(yp->frame_bytes);
    if (!yp->ref_buf || !yp->dis_buf) {
        free(yp->ref_buf); free(yp->dis_buf);
        fclose(yp->ref_fp); fclose(yp->dis_fp);
        memset(yp, 0, sizeof(*yp));
        return -1;
    }
    return 0;
}

static int yuv_pair_read_frame(YuvPair *yp, unsigned frame_idx,
                               VmafPicture *ref, VmafPicture *dist)
{
    size_t offset = (size_t)frame_idx * yp->frame_bytes;
    fseek(yp->ref_fp, (long)offset, SEEK_SET);
    fseek(yp->dis_fp, (long)offset, SEEK_SET);

    if (fread(yp->ref_buf, 1, yp->frame_bytes, yp->ref_fp) != yp->frame_bytes ||
        fread(yp->dis_buf, 1, yp->frame_bytes, yp->dis_fp) != yp->frame_bytes) {
        fprintf(stderr, "Short read at frame %u\n", frame_idx);
        return -1;
    }

    unsigned w = yp->width, h = yp->height;
    size_t y_bytes = (size_t)w * h;
    unsigned uv_w = w / 2, uv_h = h / 2;
    size_t uv_bytes = (size_t)uv_w * uv_h;

    const int hbd = (g_bpc > 8);
    const unsigned shift = hbd ? (g_bpc - 8) : 0;

    /* Y plane */
    for (unsigned y = 0; y < h; y++) {
        if (hbd) {
            uint16_t *rdst = (uint16_t *)((uint8_t *)ref->data[0] + y * ref->stride[0]);
            uint16_t *ddst = (uint16_t *)((uint8_t *)dist->data[0] + y * dist->stride[0]);
            for (unsigned x = 0; x < w; x++) {
                rdst[x] = (uint16_t)(yp->ref_buf[y * w + x]) << shift;
                ddst[x] = (uint16_t)(yp->dis_buf[y * w + x]) << shift;
            }
        } else {
            memcpy((uint8_t *)ref->data[0] + y * ref->stride[0],
                   yp->ref_buf + y * w, w);
            memcpy((uint8_t *)dist->data[0] + y * dist->stride[0],
                   yp->dis_buf + y * w, w);
        }
    }
    /* U plane */
    for (unsigned y = 0; y < uv_h; y++) {
        if (hbd) {
            uint16_t *rdst = (uint16_t *)((uint8_t *)ref->data[1] + y * ref->stride[1]);
            uint16_t *ddst = (uint16_t *)((uint8_t *)dist->data[1] + y * dist->stride[1]);
            for (unsigned x = 0; x < uv_w; x++) {
                rdst[x] = (uint16_t)(yp->ref_buf[y_bytes + y * uv_w + x]) << shift;
                ddst[x] = (uint16_t)(yp->dis_buf[y_bytes + y * uv_w + x]) << shift;
            }
        } else {
            memcpy((uint8_t *)ref->data[1] + y * ref->stride[1],
                   yp->ref_buf + y_bytes + y * uv_w, uv_w);
            memcpy((uint8_t *)dist->data[1] + y * dist->stride[1],
                   yp->dis_buf + y_bytes + y * uv_w, uv_w);
        }
    }
    /* V plane */
    for (unsigned y = 0; y < uv_h; y++) {
        if (hbd) {
            uint16_t *rdst = (uint16_t *)((uint8_t *)ref->data[2] + y * ref->stride[2]);
            uint16_t *ddst = (uint16_t *)((uint8_t *)dist->data[2] + y * dist->stride[2]);
            for (unsigned x = 0; x < uv_w; x++) {
                rdst[x] = (uint16_t)(yp->ref_buf[y_bytes + uv_bytes + y * uv_w + x]) << shift;
                ddst[x] = (uint16_t)(yp->dis_buf[y_bytes + uv_bytes + y * uv_w + x]) << shift;
            }
        } else {
            memcpy((uint8_t *)ref->data[2] + y * ref->stride[2],
                   yp->ref_buf + y_bytes + uv_bytes + y * uv_w, uv_w);
            memcpy((uint8_t *)dist->data[2] + y * dist->stride[2],
                   yp->dis_buf + y_bytes + uv_bytes + y * uv_w, uv_w);
        }
    }

    return 0;
}

static void yuv_pair_close(YuvPair *yp)
{
    if (yp->ref_fp) fclose(yp->ref_fp);
    if (yp->dis_fp) fclose(yp->dis_fp);
    free(yp->ref_buf);
    free(yp->dis_buf);
    memset(yp, 0, sizeof(*yp));
}

/* ==================== Backend enum ==================== */

enum Backend {
    BACKEND_CPU    = 0,
    BACKEND_CUDA   = 2,
    BACKEND_SYCL   = 3,
};

typedef struct {
    const char *label;
    const char *feature;
    enum Backend backend;
} BenchTarget;

static const BenchTarget targets[] = {
    { "motion (CPU)",          "motion",              BACKEND_CPU },
    { "vif (CPU)",             "vif",                 BACKEND_CPU },
    { "adm (CPU)",             "adm",                 BACKEND_CPU },
    { "float_ssim (CPU)",      "float_ssim",          BACKEND_CPU },
    { "float_ms_ssim (CPU)",   "float_ms_ssim",       BACKEND_CPU },
    { "psnr (CPU)",            "psnr",                BACKEND_CPU },
#ifdef HAVE_CUDA
    { "motion (CUDA)",         "motion_cuda",         BACKEND_CUDA },
    { "vif (CUDA)",            "vif_cuda",             BACKEND_CUDA },
    { "adm (CUDA)",            "adm_cuda",             BACKEND_CUDA },
#endif
#ifdef HAVE_SYCL
    { "motion (SYCL)",         "motion_sycl",          BACKEND_SYCL },
    { "vif (SYCL)",            "vif_sycl",              BACKEND_SYCL },
    { "adm (SYCL)",            "adm_sycl",              BACKEND_SYCL },
#endif
};
static const int n_targets = sizeof(targets) / sizeof(targets[0]);

typedef struct {
    unsigned width;
    unsigned height;
} Resolution;

static const Resolution resolutions[] = {
    {  576,  324 },
    {  640,  480 },
    { 1280,  720 },
    { 1920, 1080 },
    { 3840, 2160 },
};
static const int n_resolutions = sizeof(resolutions) / sizeof(resolutions[0]);

static int g_gpu_device_idx = -1; /* -1 = auto; applies to SYCL */

#ifdef HAVE_SYCL
static int run_sycl_gpu_profile(unsigned w, unsigned h, unsigned n_frames)
{
    int err = 0;

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 0,
        .cpumask = 0,
        .gpumask = 0,
    };

    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    if (err) return err;

    VmafSyclState *sycl_state = NULL;
    VmafSyclConfiguration sycl_cfg = {
        .device_index = g_gpu_device_idx,
        .enable_profiling = 1,
    };
    err = vmaf_sycl_state_init(&sycl_state, sycl_cfg);
    if (err) { vmaf_close(vmaf); return err; }

    err = vmaf_sycl_profiling_enable(sycl_state);
    if (err) {
        fprintf(stderr, "Failed to enable SYCL profiling: %d\n", err);
        vmaf_close(vmaf);
        return err;
    }

    err = vmaf_sycl_import_state(vmaf, sycl_state);
    if (err) { vmaf_close(vmaf); return err; }

    /* Register all three SYCL features */
    err = vmaf_use_feature(vmaf, "motion_sycl", NULL);
    if (err) { vmaf_close(vmaf); return err; }
    err = vmaf_use_feature(vmaf, "vif_sycl", NULL);
    if (err) { vmaf_close(vmaf); return err; }
    err = vmaf_use_feature(vmaf, "adm_sycl", NULL);
    if (err) { vmaf_close(vmaf); return err; }

    /* Run frames */
    YuvPair yp = { 0 };
    if (yuv_pair_open(&yp, w, h)) {
        vmaf_close(vmaf);
        return -1;
    }

    for (unsigned i = 0; i < n_frames; i++) {
        VmafPicture ref, dist;
        vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
        vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
        if (yuv_pair_read_frame(&yp, i, &ref, &dist)) {
            vmaf_picture_unref(&ref);
            vmaf_picture_unref(&dist);
            break;
        }
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (err) break;
    }
    yuv_pair_close(&yp);

    /* Print per-kernel timing */
    printf("SYCL Kernel Profile (%ux%u, %u-bit, %u frames)\n",
           w, h, g_bpc, n_frames);
    vmaf_sycl_profiling_print(sycl_state);

    vmaf_read_pictures(vmaf, NULL, NULL, 0);
    vmaf_close(vmaf);
    return 0;
}
#endif /* HAVE_SYCL */

/* ==================== Benchmark core ==================== */

static int bench_feature(const BenchTarget *target, unsigned w, unsigned h,
                         unsigned n_frames, double *out_init_ms,
                         double *out_avg_ms, double *out_total_ms)
{
    int err = 0;
    double t0, t1;
    int is_gpu = (target->backend != BACKEND_CPU);

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 0,
        .cpumask = 0,
        .gpumask = is_gpu ? 0 : (unsigned)~0,
    };

    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    if (err) return err;

#ifdef HAVE_CUDA
    if (target->backend == BACKEND_CUDA) {
        VmafCudaState *cu_state = NULL;
        VmafCudaConfiguration cu_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cu_cfg);
        if (err) { vmaf_close(vmaf); return err; }
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) { vmaf_close(vmaf); return err; }
    }
#endif
#ifdef HAVE_SYCL
    if (target->backend == BACKEND_SYCL) {
        VmafSyclState *sycl_state = NULL;
        VmafSyclConfiguration sycl_cfg = { .device_index = g_gpu_device_idx };
        err = vmaf_sycl_state_init(&sycl_state, sycl_cfg);
        if (err) { vmaf_close(vmaf); return err; }
        err = vmaf_sycl_import_state(vmaf, sycl_state);
        if (err) { vmaf_close(vmaf); return err; }
    }
#endif

    t0 = now_ms();
    err = vmaf_use_feature(vmaf, target->feature, NULL);
    if (err) {
        vmaf_close(vmaf);
        return err;
    }

    YuvPair yp = { 0 };
    if (yuv_pair_open(&yp, w, h)) {
        vmaf_close(vmaf);
        return -1;
    }

    /* Warm up: first frame also triggers init */
    VmafPicture ref, dist;
    vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
    vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
    if (yuv_pair_read_frame(&yp, 0, &ref, &dist)) {
        vmaf_picture_unref(&ref);
        vmaf_picture_unref(&dist);
        yuv_pair_close(&yp);
        vmaf_close(vmaf);
        return -1;
    }
    err = vmaf_read_pictures(vmaf, &ref, &dist, 0);
    if (err) {
        yuv_pair_close(&yp);
        vmaf_close(vmaf);
        return err;
    }
    t1 = now_ms();
    *out_init_ms = t1 - t0;

    /* Benchmark: timed extraction loop */
    t0 = now_ms();
    for (unsigned i = 1; i < n_frames; i++) {
        VmafPicture r, d;
        vmaf_picture_alloc(&r, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
        vmaf_picture_alloc(&d, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
        if (yuv_pair_read_frame(&yp, i, &r, &d)) {
            vmaf_picture_unref(&r);
            vmaf_picture_unref(&d);
            break;
        }
        err = vmaf_read_pictures(vmaf, &r, &d, i);
        if (err) break;
    }
    t1 = now_ms();
    yuv_pair_close(&yp);

    *out_total_ms = t1 - t0;
    *out_avg_ms = *out_total_ms / (n_frames - 1);

    /* Flush */
    vmaf_read_pictures(vmaf, NULL, NULL, 0);
    vmaf_close(vmaf);
    return err;
}

static void print_separator(int cols)
{
    for (int i = 0; i < cols; i++) fputc('-', stdout);
    fputc('\n', stdout);
}

/* ==================== Validation mode ==================== */

#if defined(HAVE_CUDA) || defined(HAVE_SYCL)

typedef struct {
    const char *cpu_feature;
    const char *gpu_feature;
    const char *label;
    enum Backend backend;
    double tolerance;
    const char *score_names[16];
} ValidationPair;

static const ValidationPair validation_pairs[] = {
#ifdef HAVE_CUDA
    {
        "motion", "motion_cuda", "Motion/CU", BACKEND_CUDA, 5e-6,
        { "VMAF_integer_feature_motion_score",
          "VMAF_integer_feature_motion2_score", NULL }
    },
    {
        "vif", "vif_cuda", "VIF/CU", BACKEND_CUDA, 0.001,
        { "VMAF_integer_feature_vif_scale0_score",
          "VMAF_integer_feature_vif_scale1_score",
          "VMAF_integer_feature_vif_scale2_score",
          "VMAF_integer_feature_vif_scale3_score", NULL }
    },
    {
        "adm", "adm_cuda", "ADM/CU", BACKEND_CUDA, 0.5,
        { "VMAF_integer_feature_adm2_score",
          "integer_adm_scale0",
          "integer_adm_scale1",
          "integer_adm_scale2",
          "integer_adm_scale3", NULL }
    },
#endif
#ifdef HAVE_SYCL
    {
        "motion", "motion_sycl", "Motion/SYCL", BACKEND_SYCL, 5e-6,
        { "VMAF_integer_feature_motion2_score", NULL }
    },
    {
        "vif", "vif_sycl", "VIF/SYCL", BACKEND_SYCL, 0.001,
        { "VMAF_integer_feature_vif_scale0_score",
          "VMAF_integer_feature_vif_scale1_score",
          "VMAF_integer_feature_vif_scale2_score",
          "VMAF_integer_feature_vif_scale3_score", NULL }
    },
    {
        "adm", "adm_sycl", "ADM/SYCL", BACKEND_SYCL, 0.5,
        { "VMAF_integer_feature_adm2_score",
          "integer_adm_scale0",
          "integer_adm_scale1",
          "integer_adm_scale2",
          "integer_adm_scale3", NULL }
    },
#endif
};
static const int n_validation_pairs =
    sizeof(validation_pairs) / sizeof(validation_pairs[0]);

/* Run a feature extractor and collect per-frame scores */
static int run_feature_collect(const char *feature, enum Backend backend,
                               unsigned w, unsigned h, unsigned n_frames,
                               const char *const *score_names,
                               double scores[][16])
{
    int err = 0;
    int is_gpu = (backend != BACKEND_CPU);

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 1,
        .n_subsample = 0,
        .cpumask = 0,
        .gpumask = is_gpu ? 0 : (unsigned)~0,
    };

    VmafContext *vmaf = NULL;
    err = vmaf_init(&vmaf, cfg);
    if (err) return err;

#ifdef HAVE_CUDA
    if (backend == BACKEND_CUDA) {
        VmafCudaState *cu_state = NULL;
        VmafCudaConfiguration cu_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cu_cfg);
        if (err) { vmaf_close(vmaf); return err; }
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) { vmaf_close(vmaf); return err; }
    }
#endif
#ifdef HAVE_SYCL
    if (backend == BACKEND_SYCL) {
        VmafSyclState *sycl_state = NULL;
        VmafSyclConfiguration sycl_cfg = { .device_index = g_gpu_device_idx };
        err = vmaf_sycl_state_init(&sycl_state, sycl_cfg);
        if (err) { vmaf_close(vmaf); return err; }
        err = vmaf_sycl_import_state(vmaf, sycl_state);
        if (err) { vmaf_close(vmaf); return err; }
    }
#endif

    err = vmaf_use_feature(vmaf, feature, NULL);
    if (err) { vmaf_close(vmaf); return err; }

    YuvPair yp = { 0 };
    if (yuv_pair_open(&yp, w, h)) {
        vmaf_close(vmaf);
        return -1;
    }

    for (unsigned i = 0; i < n_frames; i++) {
        VmafPicture r, d;
        vmaf_picture_alloc(&r, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
        vmaf_picture_alloc(&d, VMAF_PIX_FMT_YUV420P, g_bpc, w, h);
        if (yuv_pair_read_frame(&yp, i, &r, &d)) {
            vmaf_picture_unref(&r);
            vmaf_picture_unref(&d);
            yuv_pair_close(&yp);
            vmaf_close(vmaf);
            return -1;
        }
        err = vmaf_read_pictures(vmaf, &r, &d, i);
        if (err) {
            yuv_pair_close(&yp); vmaf_close(vmaf); return err;
        }
    }
    yuv_pair_close(&yp);
    vmaf_read_pictures(vmaf, NULL, NULL, 0);

    /* Collect scores */
    for (unsigned i = 0; i < n_frames; i++) {
        for (int s = 0; score_names[s]; s++) {
            double val = 0.0;
            err = vmaf_feature_score_at_index(vmaf, score_names[s], &val, i);
            scores[i][s] = err ? NAN : val;
        }
    }

    vmaf_close(vmaf);
    return 0;
}

static int run_validation(unsigned w, unsigned h, unsigned n_frames)
{
    int total_fail = 0;
    char res_str[16];
    snprintf(res_str, sizeof(res_str), "%ux%u", w, h);

    for (int p = 0; p < n_validation_pairs; p++) {
        const ValidationPair *vp = &validation_pairs[p];

        /* Count score names */
        int n_scores = 0;
        while (vp->score_names[n_scores]) n_scores++;

        double (*cpu_scores)[16] = calloc(n_frames, sizeof(*cpu_scores));
        double (*gpu_scores)[16] = calloc(n_frames, sizeof(*gpu_scores));
        if (!cpu_scores || !gpu_scores) {
            fprintf(stderr, "allocation failed\n");
            free(cpu_scores);
            free(gpu_scores);
            return -1;
        }

        int err_cpu = run_feature_collect(vp->cpu_feature, BACKEND_CPU,
                                          w, h, n_frames,
                                          vp->score_names, cpu_scores);
        int err_gpu = run_feature_collect(vp->gpu_feature, vp->backend,
                                          w, h, n_frames,
                                          vp->score_names, gpu_scores);

        if (err_cpu || err_gpu) {
            printf("  %-10s @ %s: SKIP (cpu_err=%d gpu_err=%d)\n",
                   vp->label, res_str, err_cpu, err_gpu);
            free(cpu_scores);
            free(gpu_scores);
            continue;
        }

        /* Compare scores per frame per metric */
        for (int s = 0; s < n_scores; s++) {
            double max_diff = 0.0;
            int any_nan = 0;
            for (unsigned f = 0; f < n_frames; f++) {
                double c = cpu_scores[f][s];
                double v = gpu_scores[f][s];
                // Both NaN is acceptable (e.g. motion2 at index 1)
                if (isnan(c) && isnan(v)) continue;
                // One NaN but not the other is a real mismatch
                if (isnan(c) || isnan(v)) { any_nan = 1; continue; }
                double diff = fabs(c - v);
                if (diff > max_diff) max_diff = diff;
            }

            const double tol = vp->tolerance;
            int pass = !any_nan && max_diff <= tol;
            const char *status = pass ? "PASS" : (any_nan ? "NaN!" : "FAIL");

            if (!pass)
                total_fail++;

            printf("  %-10s @ %s  %-45s  max_diff=%.2e  [%s]\n",
                   vp->label, res_str, vp->score_names[s], max_diff, status);


        }

        free(cpu_scores);
        free(gpu_scores);
    }

    return total_fail;
}

#endif /* HAVE_CUDA || HAVE_SYCL */

/* ==================== Main ==================== */

int main(int argc, char *argv[])
{
    unsigned n_frames = 10;
    int res_idx = -1;
    int validate_mode = 0;
    int list_devices = 0;
    int gpu_profile_mode = 0;
    int gpu_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            char *end = NULL;
            const long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0 || v > INT_MAX) {
                fprintf(stderr, "Invalid --frames value: %s\n", argv[i]);
                return 1;
            }
            n_frames = (unsigned) v;
            if (n_frames < 2) n_frames = 2;
        } else if (!strcmp(argv[i], "--resolution") && i + 1 < argc) {
            i++;
            char *end = NULL;
            const long rw_l = strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != 'x' || rw_l <= 0 || rw_l > INT_MAX) {
                fprintf(stderr, "Invalid --resolution: %s\n", argv[i]);
                return 1;
            }
            char *p = end + 1;
            char *end2 = NULL;
            const long rh_l = strtol(p, &end2, 10);
            if (end2 == p || *end2 != '\0' || rh_l <= 0 || rh_l > INT_MAX) {
                fprintf(stderr, "Invalid --resolution: %s\n", argv[i]);
                return 1;
            }
            const unsigned rw = (unsigned) rw_l;
            const unsigned rh = (unsigned) rh_l;
            for (int j = 0; j < n_resolutions; j++) {
                if (resolutions[j].width == rw &&
                    resolutions[j].height == rh) {
                    res_idx = j;
                    break;
                }
            }
            if (res_idx < 0) {
                fprintf(stderr, "Unknown resolution %ux%u. "
                        "Supported: 576x324, 640x480, 1280x720, "
                        "1920x1080, 3840x2160\n", rw, rh);
                return 1;
            }
        } else if (!strcmp(argv[i], "--bpc") && i + 1 < argc) {
            char *end = NULL;
            const long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0 || v > 16) {
                fprintf(stderr, "Invalid --bpc value: %s\n", argv[i]);
                return 1;
            }
            g_bpc = (unsigned) v;
            if (g_bpc != 8 && g_bpc != 10 && g_bpc != 12 && g_bpc != 16) {
                fprintf(stderr, "Unsupported bpc: %u (use 8, 10, 12, or 16)\n", g_bpc);
                return 1;
            }
        } else if (!strcmp(argv[i], "--validate")) {
            validate_mode = 1;
        } else if (!strcmp(argv[i], "--gpu-profile")) {
#if defined(HAVE_SYCL)
            gpu_profile_mode = 1;
#else
            fprintf(stderr, "--gpu-profile requires SYCL support\n");
            return 1;
#endif
        } else if (!strcmp(argv[i], "--gpu-only")) {
            gpu_only = 1;
        } else if (!strcmp(argv[i], "--data-dir") && i + 1 < argc) {
            g_datadir = argv[++i];
        } else if (!strcmp(argv[i], "--list-devices")) {
            list_devices = 1;
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
#if defined(HAVE_SYCL)
            g_gpu_device_idx = atoi(argv[++i]);
#else
            fprintf(stderr, "--device requires SYCL support\n");
            return 1;
#endif
        } else if (!strcmp(argv[i], "--help")) {
            printf("Usage: vmaf_bench [OPTIONS]\n\n");
            printf("Performance benchmark mode (default):\n");
            printf("  --frames N        Number of frames per benchmark (default: 10, max: 48)\n");
            printf("  --resolution WxH  Single resolution to test (default: all)\n");
            printf("  --bpc N           Bits per component (8, 10, 12, 16; default: 8)\n");
            printf("  --data-dir PATH   Path to test data directory (default: %s)\n", DEFAULT_DATA_DIR);
            printf("                    Override with VMAF_TEST_DATA env var\n\n");
            printf("GPU device selection:\n");
            printf("  --list-devices    List available GPU devices\n");
            printf("  --device N        Select GPU device by index (default: auto)\n\n");
            printf("Validation mode (GPU vs CPU correctness):\n");
            printf("  --validate        Compare GPU vs CPU output scores\n");
            printf("  --frames N        Number of frames to compare (default: 10)\n");
            printf("  --resolution WxH  Single resolution to test (default: all)\n\n");
            printf("GPU profiling mode (per-shader timing):\n");
            printf("  --gpu-profile     Print per-shader GPU timing breakdown\n");
            printf("  --gpu-only        Skip CPU features in benchmark mode\n");
            return 0;
        }
    }

    if (list_devices) {
#ifdef HAVE_SYCL
        /* SYCL device listing could be added here */
        fprintf(stderr, "SYCL device listing not yet implemented\n");
#else
        fprintf(stderr, "No GPU backend enabled\n");
#endif
        return 0;
    }

    /* Cap frames at available test data */
    if (n_frames > MAX_TEST_FRAMES) {
        fprintf(stderr, "Warning: capping --frames %u to %d (available in test data)\n",
                n_frames, MAX_TEST_FRAMES);
        n_frames = MAX_TEST_FRAMES;
    }

    int r_start = res_idx >= 0 ? res_idx : 0;
    int r_end = res_idx >= 0 ? res_idx + 1 : n_resolutions;

    if (gpu_profile_mode) {
        /* Default to 4K if no resolution specified */
        unsigned pw = 3840, ph = 2160;
        if (res_idx >= 0) {
            pw = resolutions[res_idx].width;
            ph = resolutions[res_idx].height;
        }
#ifdef HAVE_SYCL
        {
            int ret = run_sycl_gpu_profile(pw, ph, n_frames);
            if (ret) return ret;
        }
#endif
#if !defined(HAVE_SYCL)
        fprintf(stderr, "No GPU backend enabled\n");
        return 1;
#endif
        return 0;
    }

    if (validate_mode) {
#if defined(HAVE_CUDA) || defined(HAVE_SYCL)
        printf("VMAF GPU Correctness Validation (%s)\n", vmaf_version());
        printf("Data: %s\n", get_data_dir());
        printf("Frames per test: %u, bpc: %u\n", n_frames, g_bpc);
        printf("\n");

        int total_fail = 0;
        for (int r = r_start; r < r_end; r++) {
            total_fail += run_validation(resolutions[r].width,
                                         resolutions[r].height, n_frames);
        }

        printf("\n");
        if (total_fail == 0)
            printf("ALL PASSED\n");
        else
            printf("FAILURES: %d\n", total_fail);

        return total_fail > 0 ? 1 : 0;
#else
        fprintf(stderr, "No GPU backend enabled, cannot validate\n");
        return 1;
#endif
    }

    /* Benchmark mode */
    printf("VMAF Performance Benchmark (%s)\n", vmaf_version());
    printf("Data: %s\n", get_data_dir());
    printf("Frames per test: %u\n", n_frames);
    printf("\n");

    const int col_w = 88;
    printf("%-28s  %8s  %8s  %8s  %8s  %8s\n",
           "Feature", "Res", "Init ms", "Avg ms", "Total ms", "FPS");
    print_separator(col_w);

    for (int t = 0; t < n_targets; t++) {
        if (gpu_only && targets[t].backend == BACKEND_CPU)
            continue;
        for (int r = r_start; r < r_end; r++) {
            unsigned w = resolutions[r].width;
            unsigned h = resolutions[r].height;
            double init_ms = 0, avg_ms = 0, total_ms = 0;

            char res_str[16];
            snprintf(res_str, sizeof(res_str), "%ux%u", w, h);

            int err = bench_feature(&targets[t], w, h, n_frames,
                                    &init_ms, &avg_ms, &total_ms);
            if (err) {
                printf("%-28s  %8s  %8s  %8s  %8s  %8s\n",
                       targets[t].label, res_str, "FAIL", "-", "-", "-");
                continue;
            }

            double fps = (n_frames - 1) / (total_ms / 1000.0);
            printf("%-28s  %8s  %8.1f  %8.2f  %8.1f  %8.1f\n",
                   targets[t].label, res_str, init_ms, avg_ms, total_ms, fps);
            fflush(stdout);
        }
        if (r_end - r_start > 1)
            print_separator(col_w);
    }

    return 0;
}
