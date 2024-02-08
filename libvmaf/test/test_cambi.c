/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include "test.h"
#include "ref.h"
#include "feature/cambi.c"

#define EPS 0.00001

/* Test support function */
static int almost_equal(double a, double b)
{
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

static bool pic_data_equality(VmafPicture *pic, VmafPicture *pic2)
{
    uint16_t *data = pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;
    uint16_t *data2 = pic2->data[0];
    ptrdiff_t stride2 = pic2->stride[0] >> 1;

    for (unsigned i = 0; i < pic->h[0]; i++) {
        for (unsigned j = 0; j < pic->w[0]; j++) {
            if(data[i * stride + j] != data2[i * stride2 + j]) {
                return 0;
            }
        }
    }
    return 1;
}

static int data_pic_sum(VmafPicture *pic)
{
    int sum = 0;
    uint16_t *data = pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;
    for (unsigned i = 0; i < pic->h[0]; i++) {
        for (unsigned j = 0; j < pic->w[0]; j++) {
            sum += data[i * stride + j];
        }
    }
    return sum;
}

static int get_sample_image(VmafPicture *pic, int pic_index)
{
    int count = 0;
    uint16_t sample_pic[9][16] = {
        {1, 2, 0, 100, 0, 2, 0, 100, 0, 2, 0, 100, 4, 2, 0, 100},
        {1, 1, 50, 100, 1, 1, 50, 100, 2, 1, 50, 100, 3, 1, 50, 100},
        {4, 8, 0, 400, 0, 8, 0, 400, 0, 8, 0, 400, 16, 8, 0, 400},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
        {1, 2, 1, 1, 1, 0, 1, 5, 0, 0, 1, 0, 0, 0, 1, 1},
        {1, 2, 0, 1, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0, 0, 1},
        {6, 7, 5, 1, 6, 5, 5, 5, 0, 0, 1, 0, 0, 0, 0, 1},
        {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    };

    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV400P, 10, 4, 4);
    if (err) return err;
    uint16_t *data = (uint16_t *) pic->data[0];
    int stride = pic->stride[0] >> 1;
    for (unsigned i = 0; i < pic->h[0]; i++) {
        for (unsigned j = 0; j < pic->w[0]; j++) {
            data[i * stride + j] = sample_pic[pic_index][count++];
        }
    }
    return 0;
}

static int get_sample_image_8b(VmafPicture *pic)
{
    int count = 0;
    uint16_t sample_pic[16] = {1, 2, 0, 100, 0, 2, 0, 100, 0, 2, 0, 100, 4, 2, 0, 100};
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV400P, 8, 4, 4);
    if (err) return err;
    uint8_t *data = (uint8_t *) pic->data[0];
    int stride = pic->stride[0];
    for (unsigned i = 0; i < pic->h[0]; i++) {
        for (unsigned j = 0; j < pic->w[0]; j++) {
            data[i * stride + j] = sample_pic[count++];
        }
    }
    return 0;
}

static int get_sample_image_8x8(VmafPicture *pic, int pic_index)
{
    int count = 0;
    uint16_t sample_pic[2][64] = {
        {1, 2, 0, 100, 101, 100, 0, 1,
         0, 2, 0, 100, 101, 100, 1, 0,
         1, 2, 0, 100, 101, 100, 0, 1,
         0, 2, 0, 100, 101, 100, 1, 0,
         0, 2, 0, 100, 101, 100, 1, 0,
         4, 2, 0, 100, 101, 100, 0, 0,
         1, 2, 0, 100, 101, 100, 1, 0,
         0, 2, 0, 100, 101, 100, 0, 0},
        {1, 1, 0, 0, 0, 1, 0, 0,
         1, 1, 0, 0, 0, 1, 0, 0,
         1, 1, 0, 0, 0, 1, 1, 0,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 0, 0, 0, 1, 1, 1}};

    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV400P, 10, 8, 8);
    if (err) return err;
    uint16_t *data = (uint16_t *) pic->data[0];
    int stride = pic->stride[0] >> 1;
    for (unsigned i = 0; i < pic->h[0]; i++) {
        for (unsigned j = 0; j < pic->w[0]; j++) {
            data[i * stride + j] = sample_pic[pic_index][count++];
        }
    }
    return 0;
}


/* Preprocessing functions */
static char *test_anti_dithering_filter()
{
    VmafPicture pic, filtered_pic;

    int err = 0;
    err |= get_sample_image(&pic, 0);
    err |= get_sample_image(&filtered_pic, 1);
    mu_assert("test_anti_dithering_filter alloc error", !err);
    anti_dithering_filter(&pic, pic.w[0], pic.h[0]);
    bool equal = pic_data_equality(&pic, &filtered_pic);
    mu_assert("anti_dithering_filter output pic wrong", equal);

    vmaf_picture_unref(&pic);
    vmaf_picture_unref(&filtered_pic);

    return NULL;
}

/* Banding detection functions */
static char *test_decimate()
{
    VmafPicture pic;
    int err = get_sample_image(&pic, 0);
    mu_assert("test_decimate alloc error", !err);

    uint16_t *data = pic.data[0];
    ptrdiff_t stride = pic.stride[0] >> 1;
    uint16_t width = pic.w[0]>>1;
    uint16_t height = pic.h[0]>>1;

    decimate(&pic, width, height);

    mu_assert("decimate pic wrong pixel value (0,0)", data[0]==1);
    mu_assert("decimate pic wrong pixel value (1,0)", data[1]==0);
    mu_assert("decimate pic wrong pixel value (0,1)", data[stride]==0);
    mu_assert("decimate pic wrong pixel value (1,1)", data[1+stride]==0);

    vmaf_picture_unref(&pic);

    return NULL;
}

/* Banding detection functions */
static char *test_decimate_generic()
{
    VmafPicture pic;
    int err = 0;
    err |= get_sample_image(&pic, 0);
    mu_assert("test_decimate_generic alloc #1 error", !err);

    VmafPicture out_pic;
    err |= vmaf_picture_alloc(&out_pic, VMAF_PIX_FMT_YUV400P, 10, 2, 2);
    mu_assert("test_decimate_generic alloc #2 error", !err);

    pic.bpc = 10;
    decimate_generic_uint16_and_convert_to_10b(&pic, &out_pic, out_pic.w[0], out_pic.h[0]);

    uint16_t *data = out_pic.data[0];
    ptrdiff_t stride = out_pic.stride[0] >> 1;

    mu_assert("decimate generic 10b wrong pixel value (0,0)", data[0]==2);
    mu_assert("decimate generic 10b wrong pixel value (0,1)", data[1]==100);
    mu_assert("decimate generic 10b wrong pixel value (1,0)", data[stride]==2);
    mu_assert("decimate generic 10b wrong pixel value (1,1)", data[1+stride]==100);

    pic.bpc = 16;
    decimate_generic_uint16_and_convert_to_10b(&pic, &out_pic, out_pic.w[0], out_pic.h[0]);

    mu_assert("decimate generic 16b wrong pixel value (0,0)", data[0]==0);
    mu_assert("decimate generic 16b wrong pixel value (0,1)", data[1]==2);
    mu_assert("decimate generic 16b wrong pixel value (1,0)", data[stride]==0);
    mu_assert("decimate generic 16b wrong pixel value (1,1)", data[1+stride]==2);

    pic.bpc = 12;
    decimate_generic_uint16_and_convert_to_10b(&pic, &out_pic, out_pic.w[0], out_pic.h[0]);

    mu_assert("decimate generic 12b wrong pixel value (0,0)", data[0]==1);
    mu_assert("decimate generic 12b wrong pixel value (0,1)", data[1]==25);
    mu_assert("decimate generic 12b wrong pixel value (1,0)", data[stride]==1);
    mu_assert("decimate generic 12b wrong pixel value (1,1)", data[1+stride]==25);

    pic.bpc = 9;
    decimate_generic_9b_and_convert_to_10b(&pic, &out_pic, out_pic.w[0], out_pic.h[0]);

    mu_assert("decimate generic 9b to 10b wrong pixel value (0,0)", data[0]==4);
    mu_assert("decimate generic 9b to 10b wrong pixel value (0,1)", data[1]==200);
    mu_assert("decimate generic 9b to 10b wrong pixel value (1,0)", data[stride]==4);
    mu_assert("decimate generic 9b to 10b wrong pixel value (1,1)", data[1+stride]==200);

    VmafPicture out_pic_4x4;
    err |= vmaf_picture_alloc(&out_pic_4x4, VMAF_PIX_FMT_YUV400P, 10, 4, 4);
    mu_assert("test_decimate_generic alloc #3 error", !err);

    pic.bpc = 10;
    decimate_generic_uint16_and_convert_to_10b(&pic, &out_pic_4x4, out_pic_4x4.w[0], out_pic_4x4.h[0]);

    mu_assert("decimate generic 10b wrong for same dimensions", pic_data_equality(&pic, &out_pic_4x4));

    VmafPicture pic_8b;
    err |= get_sample_image_8b(&pic_8b);
    mu_assert("test_decimate_generic alloc #4 error", !err);

    pic_8b.bpc = 8;
    decimate_generic_uint8_and_convert_to_10b(&pic_8b, &out_pic, out_pic.w[0], out_pic.h[0]);

    mu_assert("decimate generic 8b to 10b wrong pixel value (0,0)", data[0]==8);
    mu_assert("decimate generic 8b to 10b wrong pixel value (0,1)", data[1]==400);
    mu_assert("decimate generic 8b to 10b wrong pixel value (1,0)", data[stride]==8);
    mu_assert("decimate generic 8b to 10b wrong pixel value (1,1)", data[1+stride]==400);

    vmaf_picture_unref(&pic);
    vmaf_picture_unref(&pic_8b);
    vmaf_picture_unref(&out_pic);
    vmaf_picture_unref(&out_pic_4x4);

    return NULL;
}

static char *test_filter_mode()
{
    VmafPicture filtered_image, image;
    unsigned w = 5, h = 5;
    uint16_t buffer[3 * w];

    int err = 0;
    err |= vmaf_picture_alloc(&filtered_image, VMAF_PIX_FMT_YUV400P, 10, w, h);
    err |= vmaf_picture_alloc(&image, VMAF_PIX_FMT_YUV400P, 10, w, h);
    mu_assert("test_filter_mode alloc error", !err);

    uint16_t *data = image.data[0];
    ptrdiff_t stride = image.stride[0]>>1;
    uint16_t *filtered_data = filtered_image.data[0];
    ptrdiff_t output_stride = filtered_image.stride[0]>>1;

    data[1 * stride + 2] = 1; data[2 * stride + 2] = 1;
    data[1 * stride + 3] = 1; data[3 * stride + 3] = 1;
    memcpy(filtered_data, data, stride * h * sizeof(uint16_t));
    filter_mode(&filtered_image, w, h, buffer);
    mu_assert("filter_mode: all zeros", data_pic_sum(&filtered_image)==0);

    data[3 * stride + 4] = 1;
    memcpy(filtered_data, data, stride * h * sizeof(uint16_t));
    filter_mode(&filtered_image, w, h, buffer);

    mu_assert("filter_mode: one one sum check", data_pic_sum(&filtered_image)==1);
    mu_assert("filter_mode: zero (3,3) check", filtered_data[3 * output_stride + 3]==0);
    mu_assert("filter_mode: one (2,3) check", filtered_data[2 * output_stride + 3]==1);

    data[0 * stride + 0] = 2;
    data[0 * stride + 1] = 1;
    memcpy(filtered_data, data, stride * h * sizeof(uint16_t));
    filter_mode(&filtered_image, w, h, buffer);
    mu_assert("filter_mode: two in the corner check", filtered_data[0 * output_stride + 0]==2);
    data[1 * stride + 0] = 1;
    memcpy(filtered_data, data, stride * h * sizeof(uint16_t));
    filter_mode(&filtered_image, w, h, buffer);
    mu_assert("filter_mode: two in the corner and adjacent one check", filtered_data[0 * output_stride + 1]==1);
    data[2 * stride + 0] = 2;
    memcpy(filtered_data, data, stride * h * sizeof(uint16_t));
    filter_mode(&filtered_image, w, h, buffer);
    mu_assert("filter_mode: two in corner and edge check", filtered_data[1 * output_stride + 0]==2);

    vmaf_picture_unref(&image);
    vmaf_picture_unref(&filtered_image);

    return NULL;
}

static char *test_get_mask_index()
{
    uint16_t index = get_mask_index(3840, 2160, 7);
    mu_assert("get_mask_index wrong index for (3840, 2160)", index==24);
    index = get_mask_index(2560, 1440, 7);
    mu_assert("get_mask_index wrong index for (2560, 1440)", index==22);
    index = get_mask_index(1980, 1080, 7);
    mu_assert("get_mask_index wrong index for (1980, 1080)", index==21);
    index = get_mask_index(1280, 720, 7);
    mu_assert("get_mask_index wrong index for (1280, 720)", index==19);
    index = get_mask_index(960, 540, 7);
    mu_assert("get_mask_index wrong index for (960, 540)", index==18);
    index = get_mask_index(640, 360, 7);
    mu_assert("get_mask_index wrong index for (640, 360)", index==16);
    index = get_mask_index(480, 270, 7);
    mu_assert("get_mask_index wrong index for (480, 270)", index==15);
    index = get_mask_index(320, 180, 7);
    mu_assert("get_mask_index wrong index for (320, 180)", index==13);
    index = get_mask_index(6000, 4000, 7);
    mu_assert("get_mask_index wrong index for (6000, 4000)", index==27);
    index = get_mask_index(960, 540, 5);
    mu_assert("get_mask_index wrong index for (960, 540)", index==6);
    return NULL;
}

static char *test_get_spatial_mask_for_index()
{
    VmafPicture image, mask;
    uint16_t filter_size = 3;
    unsigned width = 4, height = 4;
    // dp_width = width + 2 * (filter_size >> 2) + 1
    // dp_height = 2 * (filter_size >> 2) + 2
    uint32_t mask_dp[7*4];
    int err = 0;
    uint16_t derivative_buffer[4];

    err |= get_sample_image(&image, 3);
    mu_assert("test_get_spatial_mask_for_index alloc #1 error", !err);

    err |= get_sample_image(&mask, 3);
    mu_assert("test_get_spatial_mask_for_index alloc #2 error", !err);

    get_spatial_mask_for_index(&image, &mask, mask_dp, derivative_buffer, 2, filter_size, width, height, get_derivative_data_for_row);
    mu_assert("spatial_mask_for_index wrong mask for index=2, image=3", data_pic_sum(&mask)==14);
    get_spatial_mask_for_index(&image, &mask, mask_dp, derivative_buffer, 1, filter_size, width, height, get_derivative_data_for_row);
    mu_assert("spatial_mask_for_index wrong mask for index=1, image=3", data_pic_sum(&mask)==16);
    get_spatial_mask_for_index(&image, &mask, mask_dp, derivative_buffer, 0, filter_size, width, height, get_derivative_data_for_row);
    mu_assert("spatial_mask_for_index wrong mask for index=0, image=3", data_pic_sum(&mask)==16);

    vmaf_picture_unref(&image);

    err |= get_sample_image(&image, 4);
    mu_assert("test_get_spatial_mask_for_index alloc #3 error", !err);

    get_spatial_mask_for_index(&image, &mask, mask_dp, derivative_buffer, 3, filter_size, width, height, get_derivative_data_for_row);
    mu_assert("spatial_mask_for_index wrong mask for index=3, image=4", data_pic_sum(&mask)==0);
    get_spatial_mask_for_index(&image, &mask, mask_dp, derivative_buffer, 2, filter_size, width, height, get_derivative_data_for_row);
    mu_assert("spatial_mask_for_index wrong mask for index=2, image=4", data_pic_sum(&mask)==6);
    get_spatial_mask_for_index(&image, &mask, mask_dp, derivative_buffer, 1, filter_size, width, height, get_derivative_data_for_row);
    mu_assert("spatial_mask_for_index wrong mask for index=1, image=4", data_pic_sum(&mask)==9);

    vmaf_picture_unref(&image);
    vmaf_picture_unref(&mask);

    return NULL;
}

static char *test_calculate_c_values()
{
    VmafPicture input, mask;
    float combined_c_values[16];
    float expected_values[16] = {0.6666667, 2.0, 0.0, 0.0, 2.4, 3.4285715, 2.4, 0.0,
                                 2.6666667, 3.75, 3.0, 0.0, 2.0, 2.4, 2.0, 0.0};
    unsigned width = 4, height = 4;
    uint16_t tvi_for_diff[4] = {178, 305, 432, 559};
    uint16_t window_size = 3;
    const uint16_t num_diffs = 4;
    uint16_t histograms[4*1032];

    uint16_t *diffs_to_consider = NULL;
    int *diff_weights = NULL;
    int *all_diffs = NULL;
    int err = 0;

    set_contrast_arrays(num_diffs, &diffs_to_consider, &diff_weights, &all_diffs);
    err |= get_sample_image(&input, 0);
    mu_assert("test_calculate_c_values alloc #1 error", !err);
    err |= get_sample_image(&mask, 8);
    mu_assert("test_calculate_c_values alloc #2 error", !err);

    calculate_c_values(&input, &mask, combined_c_values, histograms, window_size,
                       num_diffs, tvi_for_diff, diff_weights, all_diffs, width, height, 
                       increment_range, decrement_range);

    for (unsigned i=0; i<16; i++) {
        mu_assert("calculate_c_values error ws=3",
            almost_equal(combined_c_values[i], expected_values[i]));
    }

    VmafPicture input_8x8, mask_8x8;
    float combined_c_values_8x8[64];
    err |= get_sample_image_8x8(&input_8x8, 0);
    mu_assert("test_calculate_c_values alloc #3 error", !err);
    err |= get_sample_image_8x8(&mask_8x8, 1);
    mu_assert("test_calculate_c_values alloc #4 error", !err);
    window_size = 9;
    uint16_t histograms_8x8[8*1032];
    calculate_c_values(&input_8x8, &mask_8x8, combined_c_values_8x8, histograms_8x8,
                       window_size, num_diffs, tvi_for_diff, diff_weights, all_diffs, 8, 8, 
                       increment_range, decrement_range);

    double sum = 0;
    for (unsigned i = 0; i < 64; i++) {
        sum += combined_c_values_8x8[i];
    }
    mu_assert("combined_c_values 8x8 error", almost_equal(sum, 195.382527));

    vmaf_picture_unref(&input);
    vmaf_picture_unref(&mask);
    vmaf_picture_unref(&input_8x8);
    vmaf_picture_unref(&mask_8x8);

    return NULL;
}

static char *test_c_value_pixel()
{
    uint16_t histogram[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint16_t value = 2;
    int diffs[5] = {-2, -1, 0, 1, 2};
    uint16_t tvi_thresholds[2] = {2, 3};
    int diff_weights[2] = {1, 2};
    uint16_t num_diffs = 2;
    float c_value;

    c_value = c_value_pixel(histogram, value, diff_weights, diffs, num_diffs, tvi_thresholds, 0, 1);
    mu_assert("c_value_all_diffs for value=2, weights=2,3", almost_equal(c_value, 2.6666667));

    diff_weights[0] = 4;
    diff_weights[1] = 5;
    c_value = c_value_pixel(histogram, value, diff_weights, diffs, num_diffs, tvi_thresholds, 0, 1);
    mu_assert("c_value_all_diffs for value=2, weights=4,5", almost_equal(c_value, 6.6666667));

    value = 4;
    c_value = c_value_pixel(histogram, value, diff_weights, diffs, num_diffs, tvi_thresholds, 0, 1);
    mu_assert("c_value_all_diffs for value=4, weights=4,5", almost_equal(c_value, 0));
    return NULL;
}

static char *test_update_range()
{
    uint16_t arr[15] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    increment_range(arr, 5, 10);
    mu_assert("increment_range i=5", arr[5] == 6);
    mu_assert("increment_range i=7", arr[7] == 6);
    mu_assert("increment_range i=9", arr[9] == 6);
    mu_assert("increment_range i=10", arr[10] == 5);

    decrement_range(arr, 2, 6);
    mu_assert("decrement_range i=2", arr[2] == 4);
    mu_assert("decrement_range i=4", arr[4] == 4);
    mu_assert("decrement_range i=5", arr[5] == 5);
    mu_assert("decrement_range i=8", arr[8] == 6);

    return NULL;
}

static char *test_spatial_pooling()
{
    float arr[12] = {0, 1, 2, 3, 4, 5, 10, 7, 8, 9, 6, 11};

    double average = spatial_pooling(arr, 0, 4, 3);
    mu_assert("spatial_pooling for topk=0", average==11);

    average = spatial_pooling(arr, 0.1, 4, 3);
    mu_assert("spatial_pooling for topk=0.1", average==11);

    average = spatial_pooling(arr, 0.2, 4, 3);
    mu_assert("spatial_pooling for topk=0.2", average==10.5);

    average = spatial_pooling(arr, 1.0, 4, 3);
    mu_assert("spatial_pooling for topk=1.0", average==5.5);

    return NULL;
}

static char *test_quick_select()
{
    float arr[12] = {0, 1, 2, 3, 4, 5, 10, 7, 8, 9, 6, 11};
    int kth = 5;

    quick_select(arr, 12, 5);

    mu_assert("quick_select error value entry kth=5", arr[kth]==6);
    for (int i=0; i<kth; i++)
        mu_assert("quick_select larger values for index<kth", arr[i]>=arr[kth]);

    for (int i=kth+1; i<12; i++)
        mu_assert("quick_select smaller values for index>kth", arr[i]<=arr[kth]);


    return NULL;
}

static char *test_average_topk_elements()
{
    float arr[12] = {11, 10, 9, 8, 7, 6, 1, 2, 3, 4, 5, 0};
    double average;

    average = average_topk_elements(arr, 1);
    mu_assert("average_topk_elements topk_elements=1", average==11);

    average = average_topk_elements(arr, 2);
    mu_assert("average_topk_elements topk_elements=2", average==10.5);

    average = average_topk_elements(arr, 12);
    mu_assert("average_topk_elements topk_elements=12", average==5.5);

    return NULL;
}

static char *test_get_pixels_in_window()
{
    uint16_t pixels_in_window;
    pixels_in_window = get_pixels_in_window(62);
    mu_assert("pixels_in_window for length 62", pixels_in_window==3969);
    pixels_in_window = get_pixels_in_window(63);
    mu_assert("pixels_in_window for length 63", pixels_in_window==3969);
    pixels_in_window = get_pixels_in_window(65);
    mu_assert("pixels_in_window for length 65", pixels_in_window==4225);
    return NULL;
}

static char *test_weight_scores_per_scale()
{
    double scores_per_scale[NUM_SCALES] = {10000, 1000, 100, 10, 1};
    double score = weight_scores_per_scale(scores_per_scale, (uint16_t) 10);
    mu_assert("weight_scores_per_scale cambi score", almost_equal(score, 16842.1));
    return NULL;
}

static char *test_adjust_window_size()
{
    uint16_t window_size = 63;
    adjust_window_size(&window_size, 3840, 2160);
    mu_assert("adjusted window size for input=(3840, 2160), ws=63", window_size==63);

    window_size = 63;
    adjust_window_size(&window_size, 2560, 1440);
    mu_assert("adjusted window size for input=(2560, 1440), ws=63", window_size==42);

    window_size = 63;
    adjust_window_size(&window_size, 1920, 1080);
    mu_assert("adjusted window size for input=(1920, 1080), ws=63", window_size==31);

    window_size = 63;
    adjust_window_size(&window_size, 1280, 720);
    mu_assert("adjusted window size for input=(1280, 720), ws=63", window_size==21);

    window_size = 63;
    adjust_window_size(&window_size, 960, 540);
    mu_assert("adjusted window size for input=(960, 540), ws=63", window_size==15);

    window_size = 63;
    adjust_window_size(&window_size, 640, 360);
    mu_assert("adjusted window size for input=(640, 360), ws=63", window_size==10);

    window_size = 63;
    adjust_window_size(&window_size, 480, 270);
    mu_assert("adjusted window size for input=(480, 270), ws=63", window_size==7);

    window_size = 63;
    adjust_window_size(&window_size, 320, 180);
    mu_assert("adjusted window size for input=(320, 180), ws=63", window_size==5);

    window_size = 63;
    adjust_window_size(&window_size, 6000, 4000);
    mu_assert("adjusted window size for input=(6000, 4000), ws=63", window_size==105);

    window_size = 60;
    adjust_window_size(&window_size, 1920, 1080);
    mu_assert("adjusted window size for input=(1920, 1080), ws=60", window_size==30);

    window_size = 31;
    adjust_window_size(&window_size, 1280, 720);
    mu_assert("adjusted window size for input=(1280, 720), ws=31", window_size==10);

    return NULL;
}


/* Visibility threshold functions */
static char *test_get_tvi_for_diff()
{
    VmafLumaRange range_10b_limited;
    vmaf_luminance_init_luma_range(&range_10b_limited, 10, VMAF_PIXEL_RANGE_LIMITED);

    int tvi = get_tvi_for_diff(1, 0.019, 10, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_for_diff 1 and bd=10", tvi==178);
    tvi = get_tvi_for_diff(2, 0.019, 10, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_for_diff 2 and bd=10", tvi==305);
    tvi = get_tvi_for_diff(3, 0.019, 10, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_for_diff 3 and bd=10", tvi==432);
    tvi = get_tvi_for_diff(4, 0.019, 10, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_for_diff 4 and bd=10", tvi==559);

    return NULL;
}


static char *test_tvi_condition()
{
    VmafLumaRange range_10b_limited;
    vmaf_luminance_init_luma_range(&range_10b_limited, 10, VMAF_PIXEL_RANGE_LIMITED);

    bool condition;
    condition = tvi_condition(177, 1, 0.019, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_condition for bitdepth 10 and diff 1", condition);
    condition = tvi_condition(178, 1, 0.019, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_condition for bitdepth 10 and diff 1", condition);
    condition = tvi_condition(179, 1, 0.019, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_condition for bitdepth 10 and diff 4", !condition);
    condition = tvi_condition(935, 4, 0.01, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_condition for bitdepth 10 and diff 4", condition);
    condition = tvi_condition(936, 4, 0.01, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("tvi_condition for bitdepth 10 and diff 4", condition);
    return NULL;
}

static char *test_set_contrast_arrays()
{
    uint16_t *diffs_to_consider = NULL;
    int *diffs_weights = NULL;
    int *all_diffs = NULL;

    int max_log_diff = 2;
    int expected_diffs_to_consider_4[4] = {1, 2, 3, 4};
    int expected_diffs_weights_4[4] = {1, 2, 3, 4};
    int expected_all_diffs_4[9] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

    int num_diffs = (1<<max_log_diff);
    set_contrast_arrays(num_diffs, &diffs_to_consider, &diffs_weights, &all_diffs);

    for (int i=0; i < num_diffs; i++) {
        mu_assert("set_contrast_arrays max_log_diff 2, error at diffs_to_consider",
                   expected_diffs_to_consider_4[i] == diffs_to_consider[i]);
        mu_assert("set_contrast_arrays max_log_diff 2, error at diffs_weights",
                   expected_diffs_weights_4[i] == diffs_weights[i]);
    }

    for (int i=0; i < 2*num_diffs + 1; i++) {
        mu_assert("set_contrast_arrays max_log_diff 2, error at all_diffs",
                   expected_all_diffs_4[i] == all_diffs[i]);
    }

    aligned_free(diffs_to_consider);
    aligned_free(diffs_weights);
    aligned_free(all_diffs);

    max_log_diff = 3;
    int expected_diffs_to_consider_8[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int expected_diffs_weights_8[8] = {1, 2, 3, 4, 4, 5, 5, 6};
    int expected_all_diffs_8[17] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    num_diffs = (1<<max_log_diff);
    set_contrast_arrays(num_diffs, &diffs_to_consider, &diffs_weights, &all_diffs);

    for (int i=0; i < num_diffs; i++) {
        mu_assert("set_contrast_arrays max_log_diff 3, error at diffs_to_consider",
                   expected_diffs_to_consider_8[i] == diffs_to_consider[i]);
        mu_assert("set_contrast_arrays max_log_diff 3, error at diffs_weights",
                   expected_diffs_weights_8[i] == diffs_weights[i]);
    }

    for (int i=0; i < 2*num_diffs + 1; i++) {
        mu_assert("set_contrast_arrays max_log_diff 3, error at all_diffs",
                   expected_all_diffs_8[i] == all_diffs[i]);
    }

    aligned_free(diffs_to_consider);
    aligned_free(diffs_weights);
    aligned_free(all_diffs);

    return NULL;
}

static char *test_tvi_hard_threshold_condition()
{
    VmafLumaRange range_10b_limited;
    vmaf_luminance_init_luma_range(&range_10b_limited, 10, VMAF_PIXEL_RANGE_LIMITED);

    enum CambiTVIBisectFlag result;
    result = tvi_hard_threshold_condition(177, 1, 0.019, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("hard threshold error for bd=10 and diff=1", result==CAMBI_TVI_BISECT_TOO_SMALL);
    result = tvi_hard_threshold_condition(178, 1, 0.019, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("hard threshold error for bd=10 and diff=1", result==CAMBI_TVI_BISECT_CORRECT);
    result = tvi_hard_threshold_condition(179, 1, 0.019, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("hard threshold error for bd=10 and diff=1", result==CAMBI_TVI_BISECT_TOO_BIG);
    result = tvi_hard_threshold_condition(305, 2, 0.019, range_10b_limited, vmaf_luminance_bt1886_eotf);
    mu_assert("hard threshold error for bd=10 and diff=2", result==CAMBI_TVI_BISECT_CORRECT);
    return NULL;
}

char *run_tests()
{
    /* Preprocessing functions */
    mu_run_test(test_anti_dithering_filter);
    mu_run_test(test_decimate_generic);

    /* Banding detection functions */
    mu_run_test(test_decimate);
    mu_run_test(test_filter_mode);

    mu_run_test(test_get_mask_index);
    mu_run_test(test_get_spatial_mask_for_index);

    mu_run_test(test_calculate_c_values);
    mu_run_test(test_c_value_pixel);
    mu_run_test(test_update_range);

    mu_run_test(test_spatial_pooling);
    mu_run_test(test_quick_select);
    mu_run_test(test_average_topk_elements);

    mu_run_test(test_get_pixels_in_window);
    mu_run_test(test_weight_scores_per_scale);
    mu_run_test(test_adjust_window_size);

    /* Visibility threshold functions */
    mu_run_test(test_get_tvi_for_diff);
    mu_run_test(test_tvi_condition);
    mu_run_test(test_set_contrast_arrays);
    mu_run_test(test_tvi_hard_threshold_condition);

    return NULL;
}