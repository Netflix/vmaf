#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "cli_parse.h"
#include "spinner.h"
#include "vidinput.h"

#include <libvmaf/picture.h>
#include <libvmaf/libvmaf.rc.h>

static enum VmafPixelFormat pix_fmt_map(int pf)
{
    switch (pf) {
    case PF_420:
        return VMAF_PIX_FMT_YUV420P;
    case PF_422:
        return VMAF_PIX_FMT_YUV422P;
    case PF_444:
        return VMAF_PIX_FMT_YUV444P;
    default:
        return VMAF_PIX_FMT_UNKNOWN;
    }
}

static int validate_videos(video_input *vid1, video_input *vid2)
{
    int err_cnt = 0;

    video_input_info info1, info2;
    video_input_get_info(vid1, &info1);
    video_input_get_info(vid2, &info2);

    if ((info1.frame_w != info2.frame_w) || (info1.frame_h != info2.frame_h)) {
        fprintf(stderr, "dimensions do not match: %dx%d, %dx%d\n",
                info1.frame_w, info1.frame_h, info2.frame_w, info2.frame_h);
        err_cnt++;
    }

    if (info1.pixel_fmt != info2.pixel_fmt) {
        fprintf(stderr, "pixel formats do not match: %d, %d\n",
                info1.pixel_fmt, info2.pixel_fmt);
        err_cnt++;
    }

    if (!pix_fmt_map(info1.pixel_fmt) || !pix_fmt_map(info2.pixel_fmt)) {
        fprintf(stderr, "unsupported pixel format: %d\n", info1.pixel_fmt);
        err_cnt++;
    }

    if (info1.depth != info2.depth) {
        fprintf(stderr, "bitdepths do not match: %d, %d\n",
                info1.depth, info2.depth);
        err_cnt++;
    }

    if (info1.depth < 8 || info1.depth > 16) {
        fprintf(stderr, "unsupported bitdepth: %d\n", info1.depth);
        err_cnt++;
    }

    //TODO: more validations are possible.

    return err_cnt;
}

static int fetch_picture(video_input *vid, VmafPicture *pic)
{
    int ret;
    video_input_ycbcr ycbcr;
    video_input_info info;

    ret = video_input_fetch_frame(vid, ycbcr, NULL);
    if (ret < 1) return !ret;

    video_input_get_info(vid, &info);
    ret = vmaf_picture_alloc(pic, pix_fmt_map(info.pixel_fmt), info.depth,
                             info.pic_w, info.pic_h);
    if (ret) {
        fprintf(stderr, "problem allocating picture.\n");
        return -1;
    }

    assert(pic->w[0] == ycbcr[0].width);
    assert(pic->w[1] == ycbcr[1].width);
    assert(pic->w[2] == ycbcr[2].width);

    if (info.depth == 8) {
        for (unsigned i = 0; i < 3; i++) {
            int xdec = i&&!(info.pixel_fmt&1);
            int ydec = i&&!(info.pixel_fmt&2);
            int xstride = info.depth > 8 ? 2 : 1;
            uint8_t *ycbcr_data = ycbcr[i].data +
                (info.pic_y >> ydec) * ycbcr[i].stride +
                (info.pic_x * xstride >> xdec);
            // ^ gross, but this is how the daala y4m API works. FIXME.
            uint8_t *pic_data = pic->data[i];

            for (unsigned j = 0; j < pic->h[i]; j++) {
                memcpy(pic_data, ycbcr_data, sizeof(*pic_data) * pic->w[i]);
                pic_data += pic->stride[i];
                ycbcr_data += ycbcr[i].stride;
            }
        }
    } else {
        for (unsigned i = 0; i < 3; i++) {
            int xdec = i&&!(info.pixel_fmt&1);
            int ydec = i&&!(info.pixel_fmt&2);
            int xstride = info.depth > 8 ? 2 : 1;
            uint16_t *ycbcr_data = (uint16_t*) ycbcr[i].data +
                (info.pic_y >> ydec) * (ycbcr[i].stride / 2) +
                (info.pic_x * xstride >> xdec);
            // ^ gross, but this is how the daala y4m API works. FIXME.
            uint16_t *pic_data = pic->data[i];

            for (unsigned j = 0; j < pic->h[i]; j++) {
                memcpy(pic_data, ycbcr_data, sizeof(*pic_data) * pic->w[i]);
                pic_data += pic->stride[i] / 2;
                ycbcr_data += ycbcr[i].stride / 2;
            }
        }
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int err = 0;
    const int istty = isatty(fileno(stderr));

    fprintf(stderr, "VMAF version %s\n", vmaf_version());

    CLISettings c;
    cli_parse(argc, argv, &c);

    FILE *file_ref = fopen(c.path_ref, "rb");
    if (!file_ref) {
        fprintf(stderr, "could not open file: %s\n", c.path_ref);
        return -1;
    }

    FILE *file_dist = fopen(c.path_dist, "rb");
    if (!file_dist) {
        fprintf(stderr, "could not open file: %s\n", c.path_dist);
        return -1;
    }

    video_input vid_ref;
    if (c.use_yuv) {
        err = raw_input_open(&vid_ref, file_ref,
                             c.width, c.height, c.pix_fmt, c.bitdepth);
    } else {
        err = video_input_open(&vid_ref, file_ref);
    }
    if (err) {
        fprintf(stderr, "problem with reference file: %s\n", c.path_ref);
        return -1;
    }

    video_input vid_dist;
    if (c.use_yuv) {
        err = raw_input_open(&vid_dist, file_dist,
                             c.width, c.height, c.pix_fmt, c.bitdepth);
    } else {
        err = video_input_open(&vid_dist, file_dist);
    }
    if (err) {
        fprintf(stderr, "problem with distorted file: %s\n", c.path_dist);
        return -1;
    }

    err = validate_videos(&vid_ref, &vid_dist);
    if (err) {
        fprintf(stderr, "videos are incompatible, %d %s.\n",
                err, err == 1 ? "problem" : "problems");
        return -1;
    }

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = c.thread_cnt,
        .n_subsample = c.subsample,
        .cpumask = c.cpumask,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    if (err) {
        fprintf(stderr, "problem initializing VMAF context\n");
        return -1;
    }

    VmafModel *model[c.model_cnt];
    for (unsigned i = 0; i < c.model_cnt; i++) {
        err = vmaf_model_load_from_path(&model[i], &c.model_config[i]);
        if (err) {
            fprintf(stderr, "problem loading model file: %s\n",
                    c.model_config[i].path);
            return -1;
        }
        err = vmaf_use_features_from_model(vmaf, model[i]);
        if (err) {
            fprintf(stderr,
                    "problem loading feature extractors from model file: %s\n",
                    c.model_config[i].path);
            return -1;
        }
    }

    for (unsigned i = 0; i < c.feature_cnt; i++) {
        err = vmaf_use_feature(vmaf, c.feature_cfg[i].name,
                               c.feature_cfg[i].opts_dict);
        if (err) {
            fprintf(stderr, "problem loading feature extractor: %s\n",
                    c.feature_cfg[i].name);
            return -1;
        }
    }

    const time_t t = clock();
    unsigned picture_index;
    for (picture_index = 0 ;; picture_index++) {
        VmafPicture pic_ref, pic_dist;
        int ret1 = fetch_picture(&vid_ref, &pic_ref);
        int ret2 = fetch_picture(&vid_dist, &pic_dist);

        if (ret1 && ret2) {
            break;
        } else if (ret1 < 0 || ret2 < 0) {
            fprintf(stderr, "\nproblem while reading pictures\n",
                    c.path_ref, c.path_dist);
            break;
        } else if (ret1) {
            fprintf(stderr, "\n\"%s\" ended before \"%s\".\n",
                    c.path_ref, c.path_dist);
            break;
        } else if (ret2) {
            fprintf(stderr, "\n\"%s\" ended before \"%s\".\n",
                    c.path_dist, c.path_ref);
            break;
        }

        if (istty) {
            fprintf(stderr, "\r%d frame%s %s %.2f FPS\033[K",
                    picture_index + 1, picture_index ? "s" : " ",
                    spinner[picture_index % spinner_length],
                    (picture_index + 1) /
                    (((float)clock() - t) / CLOCKS_PER_SEC));
            fflush(stderr);
        }

        err = vmaf_read_pictures(vmaf, &pic_ref, &pic_dist, picture_index);
        if (err) {
            fprintf(stderr, "\nproblem reading pictures\n");
            break;
        }
    }
    fprintf(stderr, "\n");

    for (unsigned i = 0; i < c.model_cnt; i++) {
        double vmaf_score;
        err = vmaf_score_pooled(vmaf, model[i], VMAF_POOL_METHOD_MEAN,
                                &vmaf_score, 0, picture_index - 1);
        if (err) {
            fprintf(stderr, "problem generating pooled VMAF score\n");
            return -1;
        }

        fprintf(stderr, "%s: %f\n", c.model_config[i].path, vmaf_score);
    }

    if (c.output_path) {
        FILE *outfile = fopen(c.output_path, "w");
        if (!outfile) {
            fprintf(stderr, "could not open file: %s\n", c.output_path);
            return -1;
        }
        vmaf_write_output(vmaf, outfile, c.output_fmt);
        fclose(outfile);
    }

    for (unsigned i = 0; i < c.model_cnt; i++) {
        vmaf_model_destroy(model[i]);
    }

    video_input_close(&vid_ref);
    video_input_close(&vid_dist);
    vmaf_close(vmaf);
    return err;
}
