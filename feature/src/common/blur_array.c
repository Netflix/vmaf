/*
 * blur_array.c
 *
 *  Created on: 12.02.2018
 *      Author: thomas
 */

#include <string.h>
#include "blur_array.h"

/*
 * initializes an array of blurred buffers
 */
int init_blur_array(BLUR_BUF_ARRAY* arr, int array_length, size_t size, size_t alignement)
{
    // we can't go beyond the max number of threads
    if (array_length > MAX_NUM_THREADS)
        return 0;

    for (int i = 0; i < array_length; i++)
    {
        arr->blur_buf_array[i].frame_idx = -1;
        arr->blur_buf_array[i].blur_buf = aligned_malloc(size, alignement);

        if (arr->blur_buf_array[i].blur_buf == 0)
            return 0;

        arr->buffer_size = size;
        arr->actual_length = i + 1;
    }

    pthread_mutex_init(&arr->block, NULL);

    return 1;
}

/*
 * returns the blurred buffer for the frame index or 0 if the frame index was not found
 */
float* get_blur_buf(BLUR_BUF_ARRAY* arr, int search_frame_idx)
{
    // find item for the search_frame_idx
    while (1)
    {
        pthread_mutex_lock(&arr->block);

        int array_length = arr->actual_length;
        BLUR_BUF_STRUCT* s = arr->blur_buf_array;

        for (int i = 0; i < array_length; i++)
        {
            if (s->frame_idx == search_frame_idx)
            {
                pthread_mutex_unlock(&arr->block);
                return s->blur_buf;
            }

            // next array item
            s++;
        }

        pthread_mutex_unlock(&arr->block);
    }

    return 0;
}

/*
 * finds a free slot in the array, assigns the new frame index and copies the buffer
 */
int put_blur_buf(BLUR_BUF_ARRAY* arr, int frame_idx, float* blur_buf)
{
    int ret = 0;
    int array_length = arr->actual_length;
    size_t buf_size = arr->buffer_size;
    BLUR_BUF_STRUCT* s = arr->blur_buf_array;

    pthread_mutex_lock(&arr->block);

    for (int i = 0; i < array_length; i++)
    {
        if (s->frame_idx == -1)
        {
            memcpy(s->blur_buf, blur_buf, buf_size);
            s->frame_idx = frame_idx;
            ret = 1;
            break;
        }

        // next array item
        s++;
    }

    pthread_mutex_unlock(&arr->block);

    return ret;
}

/*
 * resets the slot in the array to -1 to indicate that the buffer can be used again
 */
int release_blur_buf(BLUR_BUF_ARRAY* arr, int search_frame_idx)
{
    int ret = 0;
    int array_length = arr->actual_length;
    BLUR_BUF_STRUCT* s = arr->blur_buf_array;

    pthread_mutex_lock(&arr->block);

    for (int i = 0; i < array_length; i++)
    {
        if (s->frame_idx == search_frame_idx)
        {
            s->frame_idx = -1;
            ret = 1;
            break;
        }

        // next struct
        s++;
    }

    pthread_mutex_unlock(&arr->block);

    return ret;
}

/*
 * gives the memory buffers of the complete array free again
 */
void free_blur_buf(BLUR_BUF_ARRAY* arr)
{
    int array_length = arr->actual_length;
    size_t buf_size = arr->buffer_size;
    BLUR_BUF_STRUCT* s = arr->blur_buf_array;

    for (int i = 0; i < array_length; i++)
    {
        aligned_free(s->blur_buf);

        // next struct
        s++;
    }

    pthread_mutex_destroy(&arr->block);
}
