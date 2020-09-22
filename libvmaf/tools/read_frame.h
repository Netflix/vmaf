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

#ifndef READ_FRAME_H_
#define READ_FRAME_H_

struct data
{
    char* format; /* yuv420p, yuv422p, yuv444p, yuv420p10le, yuv422p10le, yuv444p10le, yuv420p12le, yuv422p12le, yuv444p12le, yuv420p16le, yuv422p16le, yuv444p16le */
    int width;
    int height;
    size_t offset;
    FILE *ref_rfile;
    FILE *dis_rfile;
    int num_frames;
};

struct noref_data
{
    char* format; /* yuv420p, yuv422p, yuv444p, yuv420p10le, yuv422p10le, yuv444p10le, yuv420p12le, yuv422p12le, yuv444p12le, yuv420p16le, yuv422p16le, yuv444p16le */
    int width;
    int height;
    size_t offset;
    FILE *dis_rfile;
};

int read_frame(float *ref_data, float *dis_data, float *temp_data, int stride_byte, void *s);

int read_noref_frame(float *dis_data, float *temp_data, int stride_byte, void *s);

int get_frame_offset(const char *fmt, int w, int h, size_t *offset);

#endif /* READ_FRAME_H_ */

