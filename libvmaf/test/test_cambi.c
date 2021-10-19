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
int almost_equal(double a, double b)
{
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

bool pic_data_equality(VmafPicture *pic, VmafPicture *pic2)
{
    uint16_t *data = pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;
    uint16_t *data2 = pic2->data[0];
    ptrdiff_t stride2 = pic2->stride[0] >> 1;

    for (unsigned i=0; i<pic->h[0]; i++)
        for (unsigned j=0; j<pic->w[0]; j++)
            if(data[i * stride + j]!=data2[i * stride2 + j])
                return 0;
    return 1;
}

int data_pic_sum(VmafPicture *pic)
{
    int sum = 0;
    uint16_t *data = pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;
    for (unsigned i=0; i<pic->h[0]; i++)
        for (unsigned j=0; j<pic->w[0]; j++)
            sum += data[i * stride + j];
    return sum;
}

void get_sample_image(VmafPicture *pic, int pic_index)
{
    int err, count = 0;
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

    err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV400P, 10, 4, 4);
    uint16_t *data = (uint16_t *) pic->data[0];
    int stride = pic->stride[0] >> 1;
    for (unsigned i=0; i<pic->h[0]; i++)
        for (unsigned j=0; j<pic->w[0]; j++)
            data[i * stride + j] = sample_pic[pic_index][count++];
}

void get_sample_image_8b(VmafPicture *pic)
{
    int err, count = 0;
    uint16_t sample_pic[16] = {1, 2, 0, 100, 0, 2, 0, 100, 0, 2, 0, 100, 4, 2, 0, 100};
    err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV400P, 8, 4, 4);
    uint8_t *data = (uint8_t *) pic->data[0];
    int stride = pic->stride[0];
    for (unsigned i=0; i<pic->h[0]; i++)
        for (unsigned j=0; j<pic->w[0]; j++)
            data[i * stride + j] = sample_pic[count++];
}

void get_sample_image_8x8(VmafPicture *pic, int pic_index)
{
    int err, count = 0;
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

    err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV400P, 10, 8, 8);
    uint16_t *data = (uint16_t *) pic->data[0];
    int stride = pic->stride[0] >> 1;
    for (unsigned i=0; i<pic->h[0]; i++)
        for (unsigned j=0; j<pic->w[0]; j++)
            data[i * stride + j] = sample_pic[pic_index][count++];
}


/* Preprocessing functions */
static char *test_anti_dithering_filter()
{
    VmafPicture pic, filtered_pic;

    get_sample_image(&pic, 0);
    get_sample_image(&filtered_pic, 1);
    anti_dithering_filter(&pic);
    bool equal = pic_data_equality(&pic, &filtered_pic);
    mu_assert("anti_dithering_filter output pic wrong", equal);

    return NULL;
}

static char *test_copy_10b_luma()
{
    VmafPicture pic, copy_pic, expected;
    get_sample_image(&pic, 0);
    get_sample_image(&copy_pic, 0);
    get_sample_image(&expected, 0);

    copy_10b_luma(&pic, &copy_pic);
    bool equal = pic_data_equality(&copy_pic, &expected);
    mu_assert("convert_to_10b: wrong 10b luma copy", equal);

    return NULL;
}

/* Banding detection functions */
static char *test_decimate()
{
    VmafPicture pic;
    get_sample_image(&pic, 0);

    uint16_t *data = pic.data[0];
    ptrdiff_t stride = pic.stride[0] >> 1;
    uint16_t width = pic.w[0]>>1;
    uint16_t height = pic.h[0]>>1;

    decimate(&pic, width, height);

    mu_assert("decimate pic wrong pixel value (0,0)", data[0]==1);
    mu_assert("decimate pic wrong pixel value (1,0)", data[1]==0);
    mu_assert("decimate pic wrong pixel value (0,1)", data[stride]==0);
    mu_assert("decimate pic wrong pixel value (1,1)", data[1+stride]==0);

    return NULL;
}

/* Banding detection functions */
static char *test_decimate_generic()
{
    VmafPicture pic;
    get_sample_image(&pic, 0);

    VmafPicture out_pic;
    int err = vmaf_picture_alloc(&out_pic, VMAF_PIX_FMT_YUV400P, 10, 2, 2);
    (void)err;

    decimate_generic_10b(&pic, &out_pic);

    uint16_t *data = out_pic.data[0];
    ptrdiff_t stride = out_pic.stride[0] >> 1;

    mu_assert("decimate generic 10b wrong pixel value (0,0)", data[0]==2);
    mu_assert("decimate generic 10b wrong pixel value (0,1)", data[1]==100);
    mu_assert("decimate generic 10b wrong pixel value (1,0)", data[stride]==2);
    mu_assert("decimate generic 10b wrong pixel value (1,1)", data[1+stride]==100);

    VmafPicture pic_8b;
    get_sample_image_8b(&pic_8b);

    decimate_generic_8b_and_convert_to_10b(&pic_8b, &out_pic);

    mu_assert("decimate generic 8b to 10b wrong pixel value (0,0)", data[0]==8);
    mu_assert("decimate generic 8b to 10b wrong pixel value (0,1)", data[1]==400);
    mu_assert("decimate generic 8b to 10b wrong pixel value (1,0)", data[stride]==8);
    mu_assert("decimate generic 8b to 10b wrong pixel value (1,1)", data[1+stride]==400);

    return NULL;
}

static char *test_filter_mode()
{
    VmafPicture filtered_image, image;
    unsigned w = 5, h = 5;

    int err = vmaf_picture_alloc(&filtered_image, VMAF_PIX_FMT_YUV400P, 10, w, h);
    err |= vmaf_picture_alloc(&image, VMAF_PIX_FMT_YUV400P, 10, w, h);
    mu_assert("problem during vmaf_picture_alloc", !err);

    uint16_t *data = image.data[0];
    ptrdiff_t stride = image.stride[0]>>1;
    uint16_t *output_data = filtered_image.data[0];
    ptrdiff_t output_stride = filtered_image.stride[0]>>1;

    data[2 * stride + 2] = 1; data[3 * stride + 2] = 1;
    data[2 * stride + 3] = 1; data[3 * stride + 3] = 1;
    filter_mode(&image, &filtered_image, w, h);
    mu_assert("filter_mode: all zeros", data_pic_sum(&filtered_image)==0);

    data[3 * stride + 4] = 1;
    filter_mode(&image, &filtered_image, w, h);
    mu_assert("filter_mode: two ones sum check", data_pic_sum(&filtered_image)==2);
    mu_assert("filter_mode: two ones (3,3) check", output_data[3 * output_stride + 3]==1);
    mu_assert("filter_mode: two ones (2,3) check", output_data[2 * output_stride + 3]==1);

    data[0 * stride + 0] = 2;
    data[0 * stride + 1] = 1;
    filter_mode(&image, &filtered_image, w, h);
    mu_assert("filter_mode: two in the corner check", output_data[0 * output_stride + 0]==2);
    data[1 * stride + 0] = 1;
    filter_mode(&image, &filtered_image, w, h);
    mu_assert("filter_mode: two in the corner and adjacent ones check", output_data[0 * output_stride + 0]==1);
    data[2 * stride + 0] = 2;
    filter_mode(&image, &filtered_image, w, h);
    mu_assert("filter_mode: two in corner and edge check", output_data[1 * output_stride + 0]==2);

    return NULL;
}

static char *test_get_zero_derivative()
{
    VmafPicture pic, expected, zero_derivative;
    get_sample_image(&pic, 3);
    get_sample_image(&zero_derivative, 3);
    get_sample_image(&expected, 4);

    get_zero_derivative(&pic, &zero_derivative, 4, 4);
    bool equal = pic_data_equality(&expected, &zero_derivative);
    mu_assert("zero derivative output pic wrong", equal);

    return NULL;
}

static char *test_get_mask_index()
{
    uint16_t index = get_mask_index(1980, 1080, 7);
    mu_assert("get_mask_index wrong index for (1980, 1080)", index==21);
    index = get_mask_index(3840, 2160, 7);
    mu_assert("get_mask_index wrong index for (3840, 2160)", index==24);
    index = get_mask_index(960, 540, 5);
    mu_assert("get_mask_index wrong index for (960, 540)", index==3);
    return NULL;
}

static char *test_get_spatial_mask_for_index()
{
    VmafPicture image, mask;
    uint16_t filter_size = 3;
    unsigned width = 4, height = 4;

    get_sample_image(&image, 3);
    get_sample_image(&mask, 3);

    get_spatial_mask_for_index(&image, &mask, 2, filter_size, width, height);
    mu_assert("spatial_mask_for_index wrong mask for index=2, image=3", data_pic_sum(&mask)==0);
    get_spatial_mask_for_index(&image, &mask, 1, filter_size, width, height);
    mu_assert("spatial_mask_for_index wrong mask for index=1, image=3", data_pic_sum(&mask)==2);
    get_spatial_mask_for_index(&image, &mask, 0, filter_size, width, height);
    mu_assert("spatial_mask_for_index wrong mask for index=0, image=3", data_pic_sum(&mask)==11);

    get_sample_image(&image, 4);
    get_sample_image(&image, 4);

    get_spatial_mask_for_index(&image, &mask, 5, filter_size, width, height);
    mu_assert("spatial_mask_for_index wrong mask for index=5, image=4", data_pic_sum(&mask)==2);
    get_spatial_mask_for_index(&image, &mask, 4, filter_size, width, height);
    mu_assert("spatial_mask_for_index wrong mask for index=4, image=4", data_pic_sum(&mask)==7);
    get_spatial_mask_for_index(&image, &mask, 3, filter_size, width, height);
    mu_assert("spatial_mask_for_index wrong mask for index=3, image=4", data_pic_sum(&mask)==9);

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

    get_sample_image(&input, 0);
    get_sample_image(&mask, 8);
    calculate_c_values(&input, &mask, combined_c_values,
                       window_size, tvi_for_diff, width, height);

    for (unsigned i=0; i<16; i++)
        mu_assert("calculate_c_values error ws=3",
            almost_equal(combined_c_values[i], expected_values[i]));

    VmafPicture input_8x8, mask_8x8;
    float combined_c_values_8x8[64];
    get_sample_image_8x8(&input_8x8, 0);
    get_sample_image_8x8(&mask_8x8, 1);
    window_size = 9;
    calculate_c_values(&input_8x8, &mask_8x8, combined_c_values_8x8,
                       window_size, tvi_for_diff, 8, 8);

    double sum = 0;
    for (unsigned i=0; i<64; i++)
        sum += combined_c_values_8x8[i];
    mu_assert("combined_c_values 8x8 error", almost_equal(sum, 195.382527));

    return NULL;
}

static char *test_pic_add_offset()
{
    VmafPicture image, expected;
    get_sample_image(&image, 6);
    get_sample_image(&expected, 7);

    uint16_t const offset = 5;
    pic_add_offset(&image, offset, 3, 2);
    mu_assert("pic_add_offset output pic wrong", pic_data_equality(&expected, &image));

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

    c_value = c_value_pixel(histogram, value, diff_weights, diffs, num_diffs, tvi_thresholds);
    mu_assert("c_value_all_diffs for value=2, weights=2,3", almost_equal(c_value, 2.6666667));


    diff_weights[0] = 4;
    diff_weights[1] = 5;
    c_value = c_value_pixel(histogram, value, diff_weights, diffs, num_diffs, tvi_thresholds);
    mu_assert("c_value_all_diffs for value=2, weights=4,5", almost_equal(c_value, 6.6666667));

    value = 4;
    c_value = c_value_pixel(histogram, value, diff_weights, diffs, num_diffs, tvi_thresholds);
    mu_assert("c_value_all_diffs for value=4, weights=4,5", almost_equal(c_value, 0));

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
    adjust_window_size(&window_size, 3840);
    mu_assert("adjusted window size for input=3840, ws=63", window_size==63);

    window_size = 63;
    adjust_window_size(&window_size, 1920);
    mu_assert("adjusted window size for input=1920, ws=63", window_size==31);

    window_size = 60;
    adjust_window_size(&window_size, 1920);
    mu_assert("adjusted window size for input=1920, ws=60", window_size==30);

    window_size = 31;
    adjust_window_size(&window_size, 1280);
    mu_assert("adjusted window size for input=1280, ws=31", window_size==10);

    return NULL;
}


/* Visibility threshold functions */
static char *test_get_tvi_for_diff()
{
    int tvi = get_tvi_for_diff(1, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("tvi_for_diff 1 and bd=10", tvi==178);
    tvi = get_tvi_for_diff(2, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("tvi_for_diff 2 and bd=10", tvi==305);
    tvi = get_tvi_for_diff(3, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("tvi_for_diff 3 and bd=10", tvi==432);
    tvi = get_tvi_for_diff(4, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("tvi_for_diff 4 and bd=10", tvi==559);

    tvi = get_tvi_for_diff(1, 0.01, 10, 200.0, 0.02, "standard");
    mu_assert("tvi_for_diff 1, non-default params", tvi==285);
    tvi = get_tvi_for_diff(2, 0.01, 10, 200.0, 0.02, "standard");
    mu_assert("tvi_for_diff 2, non-default params", tvi==526);

    tvi = get_tvi_for_diff(2, 0.01, 8, 200.0, 0.02, "standard");
    mu_assert("tvi_for_diff 2, bd=8, limit 255", tvi==255);

    tvi = get_tvi_for_diff(4, 0.01, 10, 200.0, 0.02, "standard");
    mu_assert("tvi_for_diff 4, bd=10, limit 1023", tvi==1023);
    return NULL;
}

static char *test_tvi_condition()
{
    bool condition;
    condition = tvi_condition(177, 1, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("tvi_condition for bitdepth 10 and diff 1", condition);
    condition = tvi_condition(178, 1, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("tvi_condition for bitdepth 10 and diff 1", condition);
    condition = tvi_condition(179, 1, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("tvi_condition for bitdepth 10 and diff 4", !condition);
    condition = tvi_condition(935, 4, 0.01, 10, 200.0, 0.02, "standard");
    mu_assert("tvi_condition for bitdepth 10 and diff 4", condition);
    condition = tvi_condition(936, 4, 0.01, 10, 200.0, 0.02, "standard");
    mu_assert("tvi_condition for bitdepth 10 and diff 4", condition);
    return NULL;
}

static char *test_tvi_hard_threshold_condition()
{
    enum CambiTVIBisectFlag result;
    result = tvi_hard_threshold_condition(177, 1, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("hard threshold error for bd=10 and diff=1", result==CAMBI_TVI_BISECT_TOO_SMALL);
    result = tvi_hard_threshold_condition(178, 1, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("hard threshold error for bd=10 and diff=1", result==CAMBI_TVI_BISECT_CORRECT);
    result = tvi_hard_threshold_condition(179, 1, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("hard threshold error for bd=10 and diff=1", result==CAMBI_TVI_BISECT_TOO_BIG);
    result = tvi_hard_threshold_condition(305, 2, 0.019, 10, 300.0, 0.01, "standard");
    mu_assert("hard threshold error for bd=10 and diff=2", result==CAMBI_TVI_BISECT_CORRECT);
    return NULL;
}

static char *test_luminance_bt1886()
{
    double L;
    L = luminance_bt1886(100, 8, 100.0, 0.005, "standard");
    mu_assert("wrong 'standard' 8b luminance bt1886", almost_equal(L, 10.663418019592129));
    L = luminance_bt1886(100, 8, 300.0, 0.01, "standard");
    mu_assert("wrong 'standard' 8b luminance bt1886", almost_equal(L, 31.68933962217197));
    L = luminance_bt1886(400, 10, 100.0, 0.005, "standard");
    mu_assert("wrong 'standard' 10b luminance bt1886", almost_equal(L, 10.663418019592129));
    L = luminance_bt1886(400, 10, 300.0, 0.01, "standard");
    mu_assert("wrong 'standard' 10b luminance bt1886", almost_equal(L, 31.68933962217197));
    L = luminance_bt1886(400, 10, 100.0, 0.005, "full");
    mu_assert("wrong 'full' 10b luminance bt1886", almost_equal(L, 11.221687443343862));

    return NULL;
}

static char *test_normalize_range()
{
    double n = normalize_range(0, 8, "full");
    mu_assert("wrong 'full' 8b normalize range", almost_equal(n, 0.0));
    n = normalize_range(128, 8, "standard");
    mu_assert("wrong 'standard' 8b normalize range", almost_equal(n, 0.5114155251141552));
    n = normalize_range(255, 8, "standard");
    mu_assert("wrong 'standard' 8b normalize range", almost_equal(n, 1.0));

    n = normalize_range(65, 10, "standard");
    mu_assert("wrong 'standard' 10b normalize range", almost_equal(n, 0.001141552511415525));
    n = normalize_range(512, 10, "standard");
    mu_assert("wrong 'standard' 10b normalize range", almost_equal(n, 0.5114155251141552));
    n = normalize_range(939, 10, "standard");
    mu_assert("wrong 'standard' 10b normalize range", almost_equal(n, 0.9988584474885844));

    return NULL;
}

static char *test_bt1886_eotf()
{
    double L = bt1886_eotf(0.5, 2.4, 1.0, 0.0);
    mu_assert("wrong bt1886_eotf result", almost_equal(L, 0.18946457081379978));
    L = bt1886_eotf(0.5, 2.0, 1.0, 0.0);
    mu_assert("wrong bt1886_eotf result", almost_equal(L, 0.25));
    L = bt1886_eotf(0.2, 2.0, 1.0, 0.0);
    mu_assert("wrong bt1886_eotf result", almost_equal(L, 0.04));

    return NULL;
}

static char *test_range_foot_head()
{
    int foot, head;

    range_foot_head(8, "standard", &foot, &head);
    mu_assert("wrong 'standard' 8b range computation", (foot==16 && head==235));
    range_foot_head(8, "full", &foot, &head);
    mu_assert("wrong 'full' 8b range computation", (foot==0 && head==255));
    range_foot_head(10, "standard", &foot, &head);
    mu_assert("wrong 'standard' 10b range computation", (foot==64 && head==940));

    return NULL;
}


char *run_tests()
{
    /* Preprocessing functions */
    mu_run_test(test_anti_dithering_filter);
    mu_run_test(test_copy_10b_luma);
    mu_run_test(test_decimate_generic);

    /* Banding detection functions */
    mu_run_test(test_decimate);
    mu_run_test(test_filter_mode);
    mu_run_test(test_get_zero_derivative);

    mu_run_test(test_get_mask_index);
    mu_run_test(test_get_spatial_mask_for_index);

    mu_run_test(test_calculate_c_values);
    mu_run_test(test_pic_add_offset);
    mu_run_test(test_c_value_pixel);

    mu_run_test(test_spatial_pooling);
    mu_run_test(test_quick_select);
    mu_run_test(test_average_topk_elements);

    mu_run_test(test_get_pixels_in_window);
    mu_run_test(test_weight_scores_per_scale);
    mu_run_test(test_adjust_window_size);

    /* Visibility threshold functions */
    mu_run_test(test_get_tvi_for_diff);
    mu_run_test(test_tvi_condition);
    mu_run_test(test_tvi_hard_threshold_condition);
    mu_run_test(test_luminance_bt1886);
    mu_run_test(test_normalize_range);
    mu_run_test(test_bt1886_eotf);
    mu_run_test(test_range_foot_head);

    return NULL;
}