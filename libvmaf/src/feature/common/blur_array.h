/*
 * blur_array.h
 *
 *  Created on: 12.02.2018
 *      Author: thomas
 */

#ifndef VMAF_FEATURE_SRC_BLUR_ARRAY_H_
#define VMAF_FEATURE_SRC_BLUR_ARRAY_H_

#include <stdlib.h>
#include <pthread.h>
#include "mem.h"

#define MAX_NUM_THREADS 128
typedef struct
{
    int frame_idx;
    float *blur_buf;
    int reference_count;


} BLUR_BUF_STRUCT;

typedef struct
{
    BLUR_BUF_STRUCT blur_buf_array[MAX_NUM_THREADS];
    int actual_length;
    size_t buffer_size;
    pthread_mutex_t block;

} BLUR_BUF_ARRAY;

int init_blur_array(BLUR_BUF_ARRAY* arr, int array_length, size_t size, size_t alignement);

float* get_free_blur_buf_slot(BLUR_BUF_ARRAY* arr, int frame_idx);

int get_blur_buf_reference_count(BLUR_BUF_ARRAY* arr, int frame_idx);

int release_blur_buf_slot(BLUR_BUF_ARRAY* arr, int search_frame_idx);

int release_blur_buf_reference(BLUR_BUF_ARRAY* arr, int search_frame_idx);

float* get_blur_buf(BLUR_BUF_ARRAY* arr, int search_frame_idx);

int put_blur_buf(BLUR_BUF_ARRAY* arr, int frame_idx, float* blur_buf);

void free_blur_buf(BLUR_BUF_ARRAY* arr);

#endif /* VMAF_FEATURE_SRC_BLUR_ARRAY_H_ */
