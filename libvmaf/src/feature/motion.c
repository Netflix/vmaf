/**
 *
 *  Copyright 2016-2019 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "offset.h"
#include "motion_options.h"
#include "mem.h"
#include "common/convolution.h"
#include "common/convolution_internal.h"
#include "motion_tools.h"

#define convolution_f32_c convolution_f32_c_s
#define FILTER_5           FILTER_5_s
#define offset_image       offset_image_s

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
// My extra #defines
// frames per second
#define FPS 4
// minimum (seconds) of frame gap
#define MIN_GAP 2
// maximum (seconds) of frame gap
#define MAX_GAP 4
// discovered multiplier for correct frame indexing
#define FRAME_INDEX_OFFSET 1.5

/**
 * Note: img1_stride and img2_stride are in terms of (sizeof(float) bytes)
 */
float vmaf_image_sad_c(const float *img1, const float *img2, int width, int height, int img1_stride, int img2_stride, int pass)
{
    float accum = (float)0.0;
    for (int i = 0; i < height; ++i) {
        float accum_line = (float)0.0;
        for (int j = 0; j < width; ++j) {
            float img1px = img1[i * img1_stride + j];
            float img2px = img2[i * img2_stride + j];  
            // if running through the first time, print out all the motion scores
            // so that I can use them in the alpha masking.
            if (pass == 0){
                if(j == width - 1 && i == height - 1){     
                    // no comma after the final score for parsing in cinemagraph.py
                    printf("%f", fabs(img1px - img2px));   
                } else {
                    printf("%f,", fabs(img1px - img2px));  
                }
            }                    
            accum_line += fabs(img1px - img2px);
        }
        accum += accum_line;
    }
    float res = (float) (accum / (width * height));
    if (pass == 0) { 
        // marking the end of a frame with a newline for cinemagraph.py 
        printf("\n"); 
    }
    return res;
}

float check_frame(const float *img1, int w_h)
{
    float accum = (float)0.0;

    for (int i = 0; i < w_h; ++i) {
        accum += img1[i];
    }
    return accum;
}

/** 
 * Note: ref_stride and dis_stride are in terms of bytes
 */
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, int pass)
{

    if (ref_stride % sizeof(float) != 0)
    {
        printf("error: ref_stride %% sizeof(float) != 0, ref_stride = %d, sizeof(float) = %zu.\n", ref_stride, sizeof(float));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(float) != 0)
    {
        printf("error: dis_stride %% sizeof(float) != 0, dis_stride = %d, sizeof(float) = %zu.\n", dis_stride, sizeof(float));
        fflush(stdout);
        goto fail;
    }
    // stride for vmaf_image_sad_c is in terms of (sizeof(float) bytes)
    *score = vmaf_image_sad_c(ref, dis, w, h, ref_stride / sizeof(float), dis_stride / sizeof(float), pass);

    return 0;

fail:
    return 1;
}

int motion(int (*read_noref_frame)(float *main_data, float *temp_data, int stride, void *user_data, int offset), void *user_data, int w, int h, const char *fmt)
{
    double score = 0;
    float *ref_buf = 0;
    float *prev_blur_buf = 0;
    float *blur_buf = 0;
    float *next_ref_buf = 0;
    float *next_blur_buf = 0;
    float *temp_buf = 0;
    size_t data_sz;
    int stride;
    int ret = 1;
    bool next_frame_read;
    int global_frm_idx = 0; // map to thread_data->frm_idx in combo.c
    // Pass is the mode to run the motion computation in. If it is zero, 
    // it prints all the individual cell scores (used for masking). 
    // If it is 1, we print the motion between the frames (used for N^2 comparions).
    int pass = 0;

    if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(float)) { 
        goto fail_or_end; 
    }
    stride = ALIGN_CEIL(w * sizeof(float));
    if ((size_t)h > SIZE_MAX / stride) { 
        goto fail_or_end; 
    }

    data_sz = (size_t)stride * h;

    if (!(ref_buf = aligned_malloc(data_sz, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for ref_buf.\n");
        fflush(stdout); 
        goto fail_or_end;
    }
    if (!(prev_blur_buf = aligned_malloc(data_sz, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for prev_blur_buf.\n");
        fflush(stdout); 
        goto fail_or_end;
    }
    if (!(blur_buf = aligned_malloc(data_sz, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for blur_buf.\n");
        fflush(stdout); 
        goto fail_or_end;
    }
    if (!(next_ref_buf = aligned_malloc(data_sz, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for next_ref_buf.\n");
        fflush(stdout); 
        goto fail_or_end;
    }
    if (!(next_blur_buf = aligned_malloc(data_sz, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for next_blur_buf.\n");
        fflush(stdout); 
        goto fail_or_end;
    }
    if (!(temp_buf = aligned_malloc(data_sz, MAX_ALIGN))){
        printf("error: aligned_malloc failed for temp_buf.\n");
        fflush(stdout); 
        goto fail_or_end;
    }

    int frm_idx = -1;
    while (1) {
        // the next frame
        frm_idx = global_frm_idx;
        global_frm_idx++;
        if (frm_idx == 0) {
            // using -1 for read_noref_frame call to indicate that no seeking is required
            ret = read_noref_frame(ref_buf, temp_buf, stride, user_data, -1);
            if (ret == 1) { 
                goto fail_or_end; 
            }
            if (ret == 2) { 
                break; 
            }
            // ===============================================================
            // offset pixel by OPT_RANGE_PIXEL_OFFSET
            // ===============================================================
            offset_image(ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
            convolution_f32_c(FILTER_5, 5, ref_buf, blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));
        } 

        // reading a buffer ahead, important for knowing if the last iteration or not
        ret = read_noref_frame(next_ref_buf, temp_buf, stride, user_data, -1);
        if (ret == 1) { 
            goto fail_or_end;
        }
        if (ret == 2) { 
            next_frame_read = false; 
        } else { 
            next_frame_read = true; 
        }
        // ===============================================================
        // offset pixel by OPT_RANGE_PIXEL_OFFSET
        // ===============================================================
        if (next_frame_read) { 
            offset_image(next_ref_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride); 
        }
        // ===============================================================
        // apply filtering (to eliminate effects film grain)
        // stride input to convolution_f32_c is in terms of (sizeof(float) bytes)
        // since stride = ALIGN_CEIL(w * sizeof(float)), stride divides sizeof(float)
        // ===============================================================
        if (next_frame_read) { 
            convolution_f32_c(FILTER_5, 5, next_ref_buf, next_blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float)); 
        }
        
        /* =========== motion ============== */
        // compute
        if (frm_idx == 0){
            score = 0.0; 
        } else {     
            if ((ret = compute_motion(prev_blur_buf, blur_buf, w, h, stride, stride, &score, pass))){
                printf("error: compute_motion (prev) failed.\n");
                fflush(stdout); 
                goto fail_or_end;
            }      
        }
        fflush(stdout);
        memcpy(prev_blur_buf, blur_buf, data_sz);
        memcpy(ref_buf, next_ref_buf, data_sz);
        memcpy(blur_buf, next_blur_buf, data_sz);

        if (!next_frame_read) { 
            break; 
        }
    }

    // The second pass (pass 1) is an N^2 comparison of relative motion between all frames
    // in the input video file. The outer loop frame is referred to as 'b_frame_buf' and 
    // the inner loop frame is 'c_frame_buf', you can think of 'b' as the reference frame
    // for all the iteration of 'c' frames to be compared to. Blur buf's serve the same purpose
    // but have blurring & convolutioin applied to eliminate noise to give more accurate motion scores
    // b   c
    // ------
    // 0   1   
    // 0   2
    //  ...
    // 2   2
    // 2   3
    //  ...
    // 52  52
    pass = 1;
    float *c_frame_buf = 0;
    float *c_blur_buf = 0;
    float *b_blur_buf = 0;
    float *b_frame_buf = 0;
    if (!(c_frame_buf = aligned_malloc(data_sz, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for c frame.\n");
        fflush(stdout); goto fail_or_end;
    }
    if (!(c_blur_buf = aligned_malloc(data_sz, MAX_ALIGN))){
        printf("error: aligned_malloc failed for c blur.\n");
        fflush(stdout); goto fail_or_end;
    }
    if (!(b_frame_buf = aligned_malloc(data_sz, MAX_ALIGN))) {
        printf("error: aligned_malloc failed for b_buf.\n");
        fflush(stdout); 
        goto fail_or_end;
    }
    if (!(b_blur_buf = aligned_malloc(data_sz, MAX_ALIGN))){
        printf("error: aligned_malloc failed for b blur.\n");
        fflush(stdout); goto fail_or_end;
    }
    
    // Initialisation of scores and indices. The min_lower and min_upper are
    // initialised to give the entire video length as opposed to flag values 
    // that would pass errors back to the python process, so that if an error 
    // occurs, as has happened with very short videos, we use the whole video.
    float min = -1.0;
    int min_lower_idx = 0;
    int min_upper_idx = global_frm_idx-1;
    // loop until all frames have been iterated over for comparison
    for (int b_idx = 0; b_idx < global_frm_idx; b_idx++){   
        // read in the b frame to be the frame of reference  
        read_noref_frame(b_frame_buf, temp_buf, stride, user_data, b_idx * w * h * FRAME_INDEX_OFFSET);
        // offset and blur b_frame in preparation for comparison
        offset_image(b_frame_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
        convolution_f32_c(FILTER_5, 5, b_frame_buf, b_blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));      
        // loop from the frame index immediately after the current 'b' frame index until
        // the end of the frames
        for (int c_idx = b_idx + 1; c_idx < global_frm_idx; c_idx++){        
            // read the frame given by the 'c' index offset as the new comparison frame
            read_noref_frame(c_frame_buf, temp_buf, stride, user_data, c_idx * w * h * FRAME_INDEX_OFFSET);
            // offset and blur the 'c' frame in preparation for motion calculation
            offset_image(c_frame_buf, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
            convolution_f32_c(FILTER_5, 5, c_frame_buf, c_blur_buf, temp_buf, w, h, stride / sizeof(float), stride / sizeof(float));
            // compute the motion from b -> c with into score         
            compute_motion(b_blur_buf, c_blur_buf, w, h, stride, stride, &score, pass);   
            // min -1.0 is the condition that shows no genuine minimum has been found yet.
            // Otherwise the motion must be less than the current minimum and the index gap
            // must meet the #define'd acceptable cinemagraph length as measured in frames
            if((min == -1.0 || score < min) && MIN_GAP * FPS < (c_idx - b_idx) && (c_idx - b_idx) < MAX_GAP * FPS){
                min = score;
                min_lower_idx = b_idx;
                min_upper_idx = c_idx;
            }           
        } 
    }
    // give result to cinemagraphs.py in expected format
    printf("%f,%d,%d\n", min, min_lower_idx, min_upper_idx);  
    // cleanup
    aligned_free(b_blur_buf);
    aligned_free(b_frame_buf);
    aligned_free(c_frame_buf);
    aligned_free(c_blur_buf);
fail_or_end:
    aligned_free(ref_buf);
    aligned_free(prev_blur_buf);
    aligned_free(blur_buf);
    aligned_free(next_ref_buf);
    aligned_free(next_blur_buf);
    aligned_free(temp_buf);

    return ret;
}


