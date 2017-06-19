/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
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

#pragma once

#ifndef FILE_IO_H_
#define FILE_IO_H_

/* Whether to use [0,255] or [-128,127] input pixel range. */
//#define OPT_RANGE_PIXEL_OFFSET 0
#define OPT_RANGE_PIXEL_OFFSET (-128)

int read_image(FILE *rfile, void *buf, int width, int height, int stride, int elem_size);
int write_image(FILE *wfile, const void *buf, int width, int height, int stride, int elem_size);

int read_image_b2s(FILE *rfile, float *buf, float off, int width, int height, int stride);
int read_image_b2d(FILE *rfile, double *buf, double off, int width, int height, int stride);

int read_image_w2s(FILE *rfile, float *buf, float off, int width, int height, int stride);
int read_image_w2d(FILE *rfile, double *buf, double off, int width, int height, int stride);

int offset_image_s(float *buf, float off, int width, int height, int stride);
int offset_image_d(double *buf, double off, int width, int height, int stride);

#endif /* FILE_IO_H_ */
