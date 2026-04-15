/**
 *  VMAF VPL Zero-Copy Pipeline Demo
 *
 *  Decodes two video files with Intel VPL (hardware decode on Intel GPU)
 *  and computes VMAF scores using SYCL feature extractors.
 *
 *  Pipeline (zero-copy):
 *    VPL decode → VA surface → DMA-BUF → Level Zero → SYCL → VMAF
 *
 *  Usage:
 *    vmaf_vpl --ref ref.mp4 --dis dis.mp4 [--model vmaf_v0.6.1]
 *             [--frames N] [--device N] [--render-node /dev/dri/renderD128]
 *
 *  Requirements:
 *    - Intel GPU with hardware decode support
 *    - Intel VPL runtime (libmfx-gen / vpl-gpu-rt)
 *    - libva + libva-drm + Level Zero
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>

#include <vpl/mfx.h>

#include "libvmaf/picture.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_sycl.h"

/* SYCL surface import: DMA-BUF/VA-API on Linux */
#include "../src/sycl/dmabuf_import.h"

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void print_usage(const char *argv0)
{
    fprintf(stderr,
        "Usage: %s --ref <file> --dis <file> [options]\n"
        "\n"
        "Options:\n"
        "  --ref <file>        Reference video file\n"
        "  --dis <file>        Distorted video file\n"
        "  --model <name>      VMAF model name (default: vmaf_v0.6.1)\n"
        "  --frames <N>        Max frames to process (0 = all)\n"
        "  --device <N>        SYCL device index (default: 0)\n"
        "  --render-node <path> VA-API render node (default: /dev/dri/renderD128)\n"
        "  --fallback          Use host upload if zero-copy import fails\n"
        "  --help              Show this help\n"
        "\n"
        "Pipeline: VPL decode → VA surface → DMA-BUF → Level Zero → SYCL → VMAF\n",
        argv0);
}

/* ------------------------------------------------------------------ */
/* GPU acceleration + VPL setup                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    int          drm_fd;
    VADisplay    va_display;
    mfxLoader    loader;
    mfxSession   session;
    mfxVideoParam decode_params;
    int          width;
    int          height;
    int          bpc;         /* 8 or 10 */
    int          eof;
    /* Bitstream buffer */
    mfxBitstream bs;
    uint8_t     *bs_buf;
    size_t       bs_buf_size;
    FILE        *fp;
} VplDecoder;

static void vpl_cleanup_gpu(VplDecoder *dec)
{
    if (dec->va_display) vaTerminate(dec->va_display);
    dec->va_display = NULL;
    if (dec->drm_fd >= 0) close(dec->drm_fd);
    dec->drm_fd = -1;
}

static int vpl_decoder_open(VplDecoder *dec, const char *filename,
                             const char *render_node)
{
    memset(dec, 0, sizeof(*dec));
    dec->drm_fd = -1;

    /* ---- VA-API GPU init ---- */
    dec->drm_fd = open(render_node, O_RDWR);
    if (dec->drm_fd < 0) {
        fprintf(stderr, "Cannot open %s\n", render_node);
        return -1;
    }

    dec->va_display = vaGetDisplayDRM(dec->drm_fd);
    if (!dec->va_display) {
        fprintf(stderr, "vaGetDisplayDRM failed\n");
        close(dec->drm_fd);
        return -1;
    }

    int va_major, va_minor;
    VAStatus va_st = vaInitialize(dec->va_display, &va_major, &va_minor);
    if (va_st != VA_STATUS_SUCCESS) {
        fprintf(stderr, "vaInitialize failed: %s\n", vaErrorStr(va_st));
        close(dec->drm_fd);
        return -1;
    }

    printf("VA-API %d.%d on %s\n", va_major, va_minor, render_node);

    /* ---- VPL loader + session ---- */
    dec->loader = MFXLoad();
    if (!dec->loader) {
        fprintf(stderr, "MFXLoad failed\n");
        vpl_cleanup_gpu(dec);
        return -1;
    }

    /* Require hardware implementation */
    mfxConfig cfg1 = MFXCreateConfig(dec->loader);
    mfxVariant val;
    val.Type = MFX_VARIANT_TYPE_U32;
    val.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    MFXSetConfigFilterProperty(cfg1,
        (mfxU8 *)"mfxImplDescription.Impl", val);

    /* Set VA-API acceleration mode */
    mfxConfig cfg2 = MFXCreateConfig(dec->loader);
    val.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI;
    MFXSetConfigFilterProperty(cfg2,
        (mfxU8 *)"mfxImplDescription.AccelerationMode", val);

    mfxStatus sts = MFXCreateSession(dec->loader, 0, &dec->session);
    if (sts != MFX_ERR_NONE) {
        fprintf(stderr, "MFXCreateSession failed: %d\n", sts);
        MFXUnload(dec->loader);
        vpl_cleanup_gpu(dec);
        return -1;
    }

    /* Pass VA display to VPL */
    sts = MFXVideoCORE_SetHandle(dec->session, MFX_HANDLE_VA_DISPLAY,
                                  dec->va_display);
    if (sts != MFX_ERR_NONE) {
        fprintf(stderr, "SetHandle(VA_DISPLAY) failed: %d\n", sts);
        goto fail_session;
    }

    /* Open input file */
    dec->fp = fopen(filename, "rb");
    if (!dec->fp) {
        fprintf(stderr, "Cannot open %s\n", filename);
        goto fail_session;
    }

    /* Allocate bitstream buffer (2 MB) */
    dec->bs_buf_size = 2 * 1024 * 1024;
    dec->bs_buf = (uint8_t *)malloc(dec->bs_buf_size);
    if (!dec->bs_buf) {
        fclose(dec->fp);
        dec->fp = NULL;
        goto fail_session;
    }

    dec->bs.Data = dec->bs_buf;
    dec->bs.MaxLength = (mfxU32)dec->bs_buf_size;

    return 0;

fail_session:
    MFXClose(dec->session);
    MFXUnload(dec->loader);
    vpl_cleanup_gpu(dec);
    return -1;
}

/* Read more data into the bitstream buffer */
static int vpl_read_bitstream(VplDecoder *dec)
{
    if (dec->eof) return 0;

    /* Move unprocessed data to the beginning of the buffer */
    if (dec->bs.DataOffset > 0) {
        memmove(dec->bs.Data,
                dec->bs.Data + dec->bs.DataOffset,
                dec->bs.DataLength);
        dec->bs.DataOffset = 0;
    }

    /* Fill the rest of the buffer */
    size_t space = dec->bs_buf_size - dec->bs.DataLength;
    if (space == 0) return 0;

    size_t nread = fread(dec->bs.Data + dec->bs.DataLength,
                         1, space, dec->fp);
    if (nread == 0 && feof(dec->fp)) {
        dec->eof = 1;
        return 0;
    }

    dec->bs.DataLength += (mfxU32)nread;
    return (int)nread;
}

/* Probe stream to determine codec and resolution */
static int vpl_probe_and_init(VplDecoder *dec, mfxU32 codec_id)
{
    /* Read initial data */
    vpl_read_bitstream(dec);

    /* Set up decode header query params */
    memset(&dec->decode_params, 0, sizeof(dec->decode_params));
    dec->decode_params.mfx.CodecId = codec_id;
    dec->decode_params.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    /* Decode header to get stream info */
    mfxStatus sts = MFXVideoDECODE_DecodeHeader(dec->session, &dec->bs,
                                                 &dec->decode_params);
    if (sts != MFX_ERR_NONE) {
        fprintf(stderr, "DecodeHeader failed: %d\n", sts);
        return -1;
    }

    dec->width  = dec->decode_params.mfx.FrameInfo.CropW
                ? dec->decode_params.mfx.FrameInfo.CropW
                : dec->decode_params.mfx.FrameInfo.Width;
    dec->height = dec->decode_params.mfx.FrameInfo.CropH
                ? dec->decode_params.mfx.FrameInfo.CropH
                : dec->decode_params.mfx.FrameInfo.Height;

    /* Determine bit depth from fourcc */
    mfxU32 fourcc = dec->decode_params.mfx.FrameInfo.FourCC;
    if (fourcc == MFX_FOURCC_P010 || fourcc == MFX_FOURCC_Y210 ||
        fourcc == MFX_FOURCC_Y410) {
        dec->bpc = 10;
    } else {
        dec->bpc = 8;
    }

    printf("Stream: %dx%d %d-bit (fourcc=0x%08x)\n",
           dec->width, dec->height, dec->bpc, fourcc);

    /* Initialize decoder */
    sts = MFXVideoDECODE_Init(dec->session, &dec->decode_params);
    if (sts != MFX_ERR_NONE && sts != MFX_WRN_PARTIAL_ACCELERATION) {
        fprintf(stderr, "DECODE_Init failed: %d\n", sts);
        return -1;
    }

    if (sts == MFX_WRN_PARTIAL_ACCELERATION) {
        printf("Warning: partial HW acceleration\n");
    }

    return 0;
}

/**
 * Decode one frame.
 *
 * Returns VASurfaceID via out_va_surface.
 * Caller must release via vpl_release_surface().
 * Returns 0 on success, 1 on EOF, negative on error.
 */
static int vpl_decode_frame(VplDecoder *dec, VASurfaceID *out_surface,
                             mfxFrameSurface1 **out_held_surf)
{
    mfxStatus sts;
    mfxSyncPoint sync = NULL;
    mfxFrameSurface1 *out_surf = NULL;

    *out_held_surf = NULL;

    for (;;) {
        /* Refill bitstream if needed */
        if (dec->bs.DataLength < dec->bs_buf_size / 2 && !dec->eof) {
            vpl_read_bitstream(dec);
        }

        int passing_null = (dec->bs.DataLength == 0 && dec->eof);
        sts = MFXVideoDECODE_DecodeFrameAsync(
            dec->session,
            passing_null ? NULL : &dec->bs,
            NULL, /* internal allocation */
            &out_surf,
            &sync);

        if (sts == MFX_ERR_NONE && sync) {
            /* Got a frame — sync and return */
            sts = MFXVideoCORE_SyncOperation(dec->session, sync,
                                              60000 /* 60s timeout */);
            if (sts != MFX_ERR_NONE) {
                fprintf(stderr, "SyncOperation failed: %d\n", sts);
                if (out_surf) out_surf->FrameInterface->Release(out_surf);
                return -1;
            }

            /* Extract VA surface handle */
            mfxHDL resource = NULL;
            mfxResourceType res_type = MFX_RESOURCE_VA_SURFACE;
            mfxStatus gnh_sts = out_surf->FrameInterface->GetNativeHandle(
                out_surf, &resource, &res_type);

            if (gnh_sts == MFX_ERR_NONE && resource) {
                *out_surface = *(VASurfaceID *)resource;
            } else if (out_surf->Data.MemId) {
                *out_surface = *(VASurfaceID *)out_surf->Data.MemId;
            } else {
                fprintf(stderr, "No VA surface in decoded frame\n");
                out_surf->FrameInterface->Release(out_surf);
                return -1;
            }

            /* Keep the surface reference — caller must release */
            *out_held_surf = out_surf;
            return 0;
        }

        if (sts == MFX_ERR_MORE_DATA) {
            if (passing_null)
                return 1; /* drain call returned no data → truly done */
            continue; /* consumed input or need more — loop to drain */
        }

        if (sts == MFX_ERR_MORE_SURFACE || sts == MFX_WRN_DEVICE_BUSY) {
            /* Wait a bit and retry */
            usleep(1000);
            continue;
        }

        if (sts < 0) {
            fprintf(stderr, "DecodeFrameAsync failed: %d\n", sts);
            return -1;
        }

        /* Other warnings — retry */
        if (!sync) continue;
    }
}

static void vpl_release_surface(mfxFrameSurface1 *surf)
{
    if (surf && surf->FrameInterface)
        surf->FrameInterface->Release(surf);
}

static void vpl_decoder_close(VplDecoder *dec)
{
    if (dec->session) {
        MFXVideoDECODE_Close(dec->session);
        MFXClose(dec->session);
    }
    if (dec->loader) MFXUnload(dec->loader);
    if (dec->fp) fclose(dec->fp);
    free(dec->bs_buf);
    vpl_cleanup_gpu(dec);
    memset(dec, 0, sizeof(*dec));
    dec->drm_fd = -1;
}

/* ------------------------------------------------------------------ */
/* Guess codec from file extension                                     */
/* ------------------------------------------------------------------ */

static mfxU32 guess_codec(const char *filename)
{
    const char *ext = strrchr(filename, '.');
    if (!ext) return MFX_CODEC_HEVC;
    if (strcasecmp(ext, ".h264") == 0 ||
        strcasecmp(ext, ".264") == 0 ||
        strcasecmp(ext, ".avc") == 0)
        return MFX_CODEC_AVC;
    if (strcasecmp(ext, ".h265") == 0 ||
        strcasecmp(ext, ".265") == 0 ||
        strcasecmp(ext, ".hevc") == 0)
        return MFX_CODEC_HEVC;
    if (strcasecmp(ext, ".mp4") == 0 ||
        strcasecmp(ext, ".mkv") == 0 ||
        strcasecmp(ext, ".webm") == 0) {
        /* Container → need a demuxer; we only handle elementary streams.
         * Suggest ffmpeg pre-extraction. */
        fprintf(stderr,
            "Warning: %s appears to be a container file.\n"
            "  This tool only handles elementary streams (H.264/H.265).\n"
            "  Extract with: ffmpeg -i %s -c:v copy -bsf:v hevc_mp4toannexb output.h265\n"
            "  Or:           ffmpeg -i %s -c:v copy -bsf:v h264_mp4toannexb output.h264\n",
            filename, filename, filename);
        return MFX_CODEC_HEVC; /* try anyway */
    }
    if (strcasecmp(ext, ".av1") == 0 ||
        strcasecmp(ext, ".ivf") == 0 ||
        strcasecmp(ext, ".obu") == 0)
        return MFX_CODEC_AV1;
    if (strcasecmp(ext, ".vp9") == 0)
        return MFX_CODEC_VP9;

    return MFX_CODEC_HEVC; /* default */
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    const char *ref_file = NULL;
    const char *dis_file = NULL;
    const char *model_name = "vmaf_v0.6.1";
    const char *render_node = "/dev/dri/renderD128";
    int max_frames = 0;
    int device_idx = 0;
    int use_fallback = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref") && i + 1 < argc)
            ref_file = argv[++i];
        else if (!strcmp(argv[i], "--dis") && i + 1 < argc)
            dis_file = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model_name = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            char *end = NULL;
            const long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0 || v > INT_MAX) {
                fprintf(stderr, "Invalid --frames value: %s\n", argv[i]);
                return 1;
            }
            max_frames = (int) v;
        }
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            char *end = NULL;
            const long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0 || v > INT_MAX) {
                fprintf(stderr, "Invalid --device value: %s\n", argv[i]);
                return 1;
            }
            device_idx = (int) v;
        }
        else if (!strcmp(argv[i], "--render-node") && i + 1 < argc)
            render_node = argv[++i];
        else if (!strcmp(argv[i], "--fallback"))
            use_fallback = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!ref_file || !dis_file) {
        fprintf(stderr, "Error: --ref and --dis are required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ---- Open VPL decoders ---- */
    VplDecoder ref_dec, dis_dec;

    mfxU32 ref_codec = guess_codec(ref_file);
    mfxU32 dis_codec = guess_codec(dis_file);

    printf("Opening reference: %s (codec=0x%08x)\n", ref_file, ref_codec);
    if (vpl_decoder_open(&ref_dec, ref_file, render_node) < 0)
        return 1;
    if (vpl_probe_and_init(&ref_dec, ref_codec) < 0) {
        vpl_decoder_close(&ref_dec);
        return 1;
    }

    printf("Opening distorted: %s (codec=0x%08x)\n", dis_file, dis_codec);
    if (vpl_decoder_open(&dis_dec, dis_file, render_node) < 0) {
        vpl_decoder_close(&ref_dec);
        return 1;
    }
    if (vpl_probe_and_init(&dis_dec, dis_codec) < 0) {
        vpl_decoder_close(&dis_dec);
        vpl_decoder_close(&ref_dec);
        return 1;
    }

    /* Check dimensions match */
    if (ref_dec.width != dis_dec.width || ref_dec.height != dis_dec.height) {
        fprintf(stderr, "Error: resolution mismatch: ref=%dx%d dis=%dx%d\n",
                ref_dec.width, ref_dec.height, dis_dec.width, dis_dec.height);
        vpl_decoder_close(&dis_dec);
        vpl_decoder_close(&ref_dec);
        return 1;
    }

    int w = ref_dec.width;
    int h = ref_dec.height;
    int bpc = ref_dec.bpc > dis_dec.bpc ? ref_dec.bpc : dis_dec.bpc;

    printf("Resolution: %dx%d @ %d-bit\n", w, h, bpc);

    /* ---- Set up SYCL state ---- */
    VmafSyclState *sycl_state = NULL;
    VmafSyclConfiguration sycl_cfg = { .device_index = device_idx, .enable_profiling = 0 };
    int err = vmaf_sycl_state_init(&sycl_state, sycl_cfg);
    if (err) {
        fprintf(stderr, "vmaf_sycl_state_init failed: %d\n", err);
        vpl_decoder_close(&dis_dec);
        vpl_decoder_close(&ref_dec);
        return 1;
    }

    /* ---- Set up VMAF context ---- */
    VmafContext *vmaf = NULL;
    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 1,
        .n_subsample = 0,
        .cpumask = 0,
    };
    err = vmaf_init(&vmaf, vmaf_cfg);
    if (err) {
        fprintf(stderr, "vmaf_init failed: %d\n", err);
        vmaf_sycl_state_free(&sycl_state);
        vpl_decoder_close(&dis_dec);
        vpl_decoder_close(&ref_dec);
        return 1;
    }

    err = vmaf_sycl_import_state(vmaf, sycl_state);
    if (err) {
        fprintf(stderr, "vmaf_sycl_import_state failed: %d\n", err);
        vmaf_close(vmaf);
        vmaf_sycl_state_free(&sycl_state);
        vpl_decoder_close(&dis_dec);
        vpl_decoder_close(&ref_dec);
        return 1;
    }

    /* Register SYCL feature extractors.
     * These provide the same feature scores as CPU extractors
     * (e.g. VMAF_integer_feature_vif_scale0_score), so the VMAF model
     * can consume them directly without registering CPU extractors. */
    const char *features[] = { "vif_sycl", "adm_sycl", "motion_sycl" };
    for (int i = 0; i < 3; i++) {
        err = vmaf_use_feature(vmaf, features[i], NULL);
        if (err) {
            fprintf(stderr, "vmaf_use_feature(%s) failed: %d\n", features[i], err);
        }
    }

    /* Load VMAF model (for final score computation).
     * We do NOT call vmaf_use_features_from_model() because the SYCL
     * extractors already provide the required features. */
    VmafModel *model = NULL;
    VmafModelConfig model_cfg = {
        .name = model_name,
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load(&model, &model_cfg, model_name);
    if (err) {
        /* Try as file path if built-in name lookup fails */
        err = vmaf_model_load_from_path(&model, &model_cfg, model_name);
    }
    if (err) {
        fprintf(stderr, "vmaf_model_load(%s) failed: %d\n", model_name, err);
        /* Continue without model — features still compute */
    }

    /* ---- Init frame buffers for zero-copy ---- */
    err = vmaf_sycl_init_frame_buffers(vmaf, w, h, bpc);
    if (err) {
        fprintf(stderr, "vmaf_sycl_init_frame_buffers failed: %d\n", err);
        if (model) vmaf_model_destroy(model);
        vmaf_close(vmaf);
        vmaf_sycl_state_free(&sycl_state);
        vpl_decoder_close(&dis_dec);
        vpl_decoder_close(&ref_dec);
        return 1;
    }

    /* ---- Decode + compute loop ---- */
    printf("\nStarting VPL decode → SYCL VMAF pipeline...\n\n");
    double t_start = now_ms();
    int frame_idx = 0;
    int dmabuf_ok = 1; /* initially assume DMA-BUF import works */

    while (max_frames == 0 || frame_idx < max_frames) {
        mfxFrameSurface1 *ref_held = NULL, *dis_held = NULL;
        VASurfaceID ref_surf, dis_surf;

        int r1 = vpl_decode_frame(&ref_dec, &ref_surf, &ref_held);
        int r2 = vpl_decode_frame(&dis_dec, &dis_surf, &dis_held);

        if (r1 == 1 || r2 == 1) {
            printf("End of stream at frame %d\n", frame_idx);
            vpl_release_surface(ref_held);
            vpl_release_surface(dis_held);
            break;
        }
        if (r1 < 0 || r2 < 0) {
            fprintf(stderr, "Decode error at frame %d\n", frame_idx);
            vpl_release_surface(ref_held);
            vpl_release_surface(dis_held);
            break;
        }

        /* Import decoded surfaces into SYCL shared buffers */
        /* VA-API surface → DMA-BUF → Level Zero → SYCL (zero-copy) */
        if (dmabuf_ok) {
            err = vmaf_sycl_import_va_surface(sycl_state,
                                               ref_dec.va_display,
                                               ref_surf, 1, w, h, bpc);
            if (!err)
                err = vmaf_sycl_import_va_surface(sycl_state,
                                                   dis_dec.va_display,
                                                   dis_surf, 0, w, h, bpc);
            if (err) {
                fprintf(stderr, "DMA-BUF import failed at frame %d: %d\n",
                        frame_idx, err);
                if (use_fallback) {
                    fprintf(stderr, "Falling back to host upload path\n");
                    dmabuf_ok = 0;
                } else {
                    vpl_release_surface(ref_held);
                    vpl_release_surface(dis_held);
                    break;
                }
            }
        }

        if (!dmabuf_ok) {
            /* Fallback: map VA surface to host, upload via standard path */
            /* TODO: implement vaCopySurfaceToHost + vmaf_sycl_shared_frame_upload */
            fprintf(stderr, "Fallback host upload not yet implemented\n");
            vpl_release_surface(ref_held);
            vpl_release_surface(dis_held);
            break;
        }

        /* Release VPL surface references now that import is done */
        vpl_release_surface(ref_held);
        vpl_release_surface(dis_held);

        /* Submit frame to VMAF (zero-copy path) */
        err = vmaf_read_pictures_sycl(vmaf, frame_idx);
        if (err) {
            fprintf(stderr, "vmaf_read_pictures_sycl failed at frame %d: %d\n",
                    frame_idx, err);
            break;
        }

        frame_idx++;
        if (frame_idx % 10 == 0) {
            printf("  Processed %d frames...\n", frame_idx);
        }
    }

    /* Flush SYCL pipeline */
    err = vmaf_flush_sycl(vmaf);
    if (err)
        fprintf(stderr, "vmaf_flush_sycl failed: %d\n", err);

    double t_end = now_ms();
    double elapsed_s = (t_end - t_start) / 1000.0;

    printf("\n--- Results ---\n");
    printf("Frames: %d\n", frame_idx);
    printf("Time:   %.3f s (%.1f FPS)\n", elapsed_s,
           elapsed_s > 0 ? frame_idx / elapsed_s : 0);

    /* Print VMAF scores */
    if (model && frame_idx > 0) {
        double vmaf_score;
        err = vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN,
                                &vmaf_score, 0, frame_idx - 1);
        if (!err)
            printf("VMAF:   %.6f (mean)\n", vmaf_score);
        else
            fprintf(stderr, "vmaf_score_pooled failed: %d\n", err);
    }

    /* Print per-frame feature scores */
    for (int f = 0; f < frame_idx && f < 5; f++) {
        double vif0 = 0, adm2 = 0, motion = 0;
        vmaf_feature_score_at_index(vmaf,
            "VMAF_integer_feature_vif_scale0_score", &vif0, f);
        vmaf_feature_score_at_index(vmaf,
            "VMAF_integer_feature_adm2_score", &adm2, f);
        vmaf_feature_score_at_index(vmaf,
            "VMAF_integer_feature_motion2_score", &motion, f);
        printf("  Frame %d: vif0=%.6f adm2=%.6f motion2=%.6f\n",
               f, vif0, adm2, motion);
    }
    if (frame_idx > 5)
        printf("  ... (%d more frames)\n", frame_idx - 5);

    /* Cleanup */
    if (model) vmaf_model_destroy(model);
    vmaf_close(vmaf);
    vmaf_sycl_state_free(&sycl_state);
    vpl_decoder_close(&dis_dec);
    vpl_decoder_close(&ref_dec);

    return 0;
}
