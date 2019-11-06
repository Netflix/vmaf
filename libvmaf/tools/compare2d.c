#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
  #define DATA_TYPE double
#endif

typedef DATA_TYPE data_t;

int main(int argc, const char **argv)
{
	FILE *ref_file = 0;
	FILE *dis_file = 0;
	int w, h;
	double ref_accum = 0;
	double dis_accum = 0;
	double mse_accum = 0;
	double xsq_accum = 0;

	double abs_err = 0;
	double rel_err = 0;

	int abs_err_i = 0;
	int abs_err_j = 0;
	int rel_err_i = 0;
	int rel_err_j = 0;

	int i, j;
	int ret = 1;

	if (argc < 5)
	{
		goto fail;
	}

	ref_file = fopen(argv[1], "rb");
	dis_file = fopen(argv[2], "rb");
	w = atoi(argv[3]);
	h = atoi(argv[4]);

	if (!ref_file || !dis_file || w <= 0 || h <= 0)
	{
		goto fail;
	}

	for (i = 0; i < h; ++i) {
		data_t x, y;
		double abs_err_inner;
		double rel_err_inner;
		double ref_inner = 0;
		double dis_inner = 0;
		double mse_inner = 0;
		double xsq_inner = 0;

		for (j = 0; j < w; ++j)
		{
			if (fread(&x, sizeof(data_t), 1, ref_file) != 1)
			{
				goto fail;
			}
			if (fread(&y, sizeof(data_t), 1, dis_file) != 1)
			{
				goto fail;
			}

			abs_err_inner = fabs(x - y);
			rel_err_inner = abs_err_inner / (x == 0 ? nextafter(x, INFINITY) : x);

			ref_inner += x;
			dis_inner += y;
			mse_inner += (x - y) * (x - y);
			xsq_inner += x * x;

			if (abs_err_inner > abs_err) {
				abs_err = abs_err_inner;
				abs_err_i = i;
				abs_err_j = j;
			}
			if (rel_err_inner > rel_err) {
				rel_err = rel_err_inner;
				rel_err_i = i;
				rel_err_j = j;
			}
		}

		ref_accum += ref_inner;
		dis_accum += dis_inner;
		mse_accum += mse_inner;
		xsq_accum += xsq_inner;
	}

	ref_accum /= (double)w * h;
	dis_accum /= (double)w * h;

	printf("ref mean: %f\n", ref_accum);
	printf("dis mean: %f\n", dis_accum);
	printf("snr: %f\n", 10 * log10(xsq_accum / mse_accum));
	printf("max abs err: %e @ (%d, %d)\n", abs_err, abs_err_i, abs_err_j);
	printf("max rel err: %e @ (%d, %d)\n", rel_err, rel_err_i, rel_err_j);

fail:
	if (ref_file)
		fclose(ref_file);
	if (dis_file)
		fclose(dis_file);
	return ret;
}
