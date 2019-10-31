#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILTER_MAX 17

typedef double (*filter_func_1d)(double x, double param);

double gauss_func(double x, double sigma)
{
	return exp(-(x * x) / (2 * (sigma * sigma)));
}

int cmp_d(const void *a, const void *b)
{
	double da = *(const double *)a;
	double db = *(const double *)b;

	if (da < db)
		return -1;
	else if (da > db)
		return 1;
	else
		return 0;
}

int cmp_s(const void *a, const void *b)
{
	float fa = *(const float *)a;
	float fb = *(const float *)b;

	if (fa < fb)
		return -1;
	else if (fa > fb)
		return 1;
	else
		return 0;
}

double norm_sum_d(const double *x, int n)
{
	double accum = 0;
	int i;

	for (i = 0; i < n; ++i) {
		accum += x[i];
	}
	return accum;
}

float norm_sum_s(const float *x, int n)
{
	float accum = 0;
	int i;

	for (i = 0; i < n; ++i) {
		accum += x[i];
	}
	return accum;
}

double accurate_sum_d(const double *x, int n)
{
	double filter[FILTER_MAX] = { 0 };

	memcpy(filter, x, sizeof(double) * n);
	qsort(filter, n, sizeof(double), cmp_d);

	return norm_sum_d(filter, n);
}

double accurate_sum_s(const float *x, int n)
{
	float filter[FILTER_MAX] = { 0 };

	memcpy(filter, x, sizeof(float) * n);
	qsort(filter, n, sizeof(float), cmp_s);

	return norm_sum_s(filter, n);
}

void normalize_filt_d(const double *in, double *out, int n)
{
	double filter[FILTER_MAX] = { 0 };
	double sum = accurate_sum_d(in, n);
	int center = n / 2;
	int i;

	for (i = 0; i < center; ++i) {
		filter[i] = in[i] / sum;
	}
	for (i = center + 1; i < n; ++i) {
		filter[i] = in[i] / sum;
	}

	sum = norm_sum_d(filter, n);
	filter[center] = 1.0 - sum;

	memcpy(out, filter, sizeof(double) * n);
}

void normalize_filt_s(const double *in, float *out, int n)
{
	float filter[FILTER_MAX] = { 0 };
	double isum = accurate_sum_d(in, n);
	float osum = 0;
	int center = n / 2;
	int i;

	for (i = 0; i < center; ++i) {
		filter[i] = (float)(in[i] / isum);
	}
	for (i = center + 1; i < n; ++i) {
		filter[i] = (float)(in[i] / isum);
	}

	osum = norm_sum_s(filter, n);
	filter[center] = 1.0f - osum;

	memcpy(out, filter, sizeof(float) * n);
}

void eval_filt(int support, filter_func func, double func_param)
{
	double filter[FILTER_MAX] = { 0 };

	double filter_norm_d[FILTER_MAX] = { 0 };
	float filter_norm_s[FILTER_MAX] = { 0 };

	double sum_d = 0;
	float sum_s = 0;

	int center = support;
	int n = support * 2 + 1;
	int i;

	for (i = 0; i < n; ++i) {
		double x = i - center;
		filter[i] = func(x, func_param);
	}

	normalize_filt_d(filter, filter_norm_d, n);
	normalize_filt_s(filter, filter_norm_s, n);

	puts("double precision:");
	for (i = 0; i < n; ++i) {
		printf("%a, ", filter_norm_d[i]);
	}
	puts("");
	puts("");

	puts("single precision:");
	for (i = 0; i < n; ++i) {
		printf("%a, ", filter_norm_s[i]);
	}
	puts("");
	puts("");

	sum_d = norm_sum_d(filter_norm_d, n);
	sum_s = norm_sum_s(filter_norm_s, n);
	printf("err1: %a %a\n", 1.0 - sum_d, 1.0f - sum_s);

	sum_d = accurate_sum_d(filter_norm_d, n);
	sum_s = accurate_sum_s(filter_norm_s, n);
	printf("err2: %a %a\n", 1.0 - accurate_sum_d(filter_norm_d, n), 1.0f - accurate_sum_s(filter_norm_s, n));

	puts("");
}

int main()
{
	puts("filter 17");
	eval_filt(8, gauss_func, 17.0 / 5.0);
	puts("");

	puts("filter 9");
	eval_filt(4, gauss_func, 9.0 / 5.0);
	puts("");

	puts("filter 5");
	eval_filt(2, gauss_func, 5.0 / 5.0);
	puts("");

	puts("filter 3");
	eval_filt(1, gauss_func, 3.0 / 5.0);
	puts("");
}
