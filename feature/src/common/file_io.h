#pragma once

#ifndef FILE_IO_H_
#define FILE_IO_H_

int read_image(FILE *rfile, void *buf, int width, int height, int stride, int elem_size);
int write_image(FILE *wfile, const void *buf, int width, int height, int stride, int elem_size);

int read_image_b2s(FILE *rfile, float *buf, float off, int width, int height, int stride);
int read_image_b2d(FILE *rfile, double *buf, double off, int width, int height, int stride);

#endif /* FILE_IO_H_ */
