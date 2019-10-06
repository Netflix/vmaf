#define _POSIX_C_SOURCE 199309L

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <time.h>

#ifdef __MACH__
  #include <mach/clock.h>
  #include <mach/mach.h>
#endif

static const float log2_poly[9] = { -0.012671635276421, 0.064841182402670, -0.157048836463065, 0.257167726303123, -0.353800560300520, 0.480131410397451, -0.721314327952201, 1.442694803896991, 0 };

#ifdef USE_FMA
  #define FMAF(a, b, c) fmaf(a, b, c)
#else
  #define FMAF(a, b, c) ((a) * (b) + (c))
#endif

static float horner(const float *coeffs, float x, int n)
{
	float val = 0;
	int i;

	for (i = 0; i < n; ++i) {
		val = FMAF(val, x, coeffs[i]);
	}

	return val;
}

static float log2f_remez(float x)
{
	x = x - 1.0f;

	return horner(log2_poly, x, sizeof(log2_poly) / sizeof(log2_poly[0]));
}

static float log2f_approx(float x)
{
	const uint32_t f_one_const = 0x3F800000UL;
	const uint32_t f_expo_mask = 0x7F800000UL;
	const uint32_t f_mant_mask = 0x007FFFFFUL;

	float remain, log2_remain;
	uint32_t u32, u32_remain;
	uint32_t expo, mant;

	if (x <= 0.0f)
		return NAN;
	if (x == 0.0f)
		return -INFINITY;

	memcpy(&u32, &x, sizeof(float));
	expo = ((u32 & f_expo_mask) >> 23) - 127;
	mant = u32 & f_mant_mask;

	u32_remain = mant | f_one_const;
	memcpy(&remain, &u32_remain, sizeof(float));

	log2_remain = log2f_remez(remain);

	return (float)(int32_t)expo + log2_remain;
}

static long long sub_timespec(struct timespec a, struct timespec b)
{
	return (long long)(a.tv_sec - b.tv_sec) * 1000000000 + a.tv_nsec - b.tv_nsec;
}

__attribute__((noinline))
static void check_accuracy(void)
{
	double abs_err = 0.0;
	double rel_err = 0.0;

	double abs_err_x = 0.0;
	double rel_err_x = 0.0;

	float x;

	for (x = 1.0f; x < 2.0f; x = nextafterf(x, 2.0f)) {
		float exact  = log2f(x);
		float approx = log2f_approx(x);

		double abs_err_curr = fabsf(exact - approx);
		double rel_err_curr = abs_err_curr / (exact <= 0 ? DBL_EPSILON : exact);

		if (abs_err_curr > abs_err) {
			abs_err   = abs_err_curr;
			abs_err_x = x;
		}
		if (rel_err_curr > rel_err) {
			rel_err   = rel_err_curr;
			rel_err_x = x;
		}
	}

	printf("log2(1) = %e (%e)\n", log2f_approx(1.0f), log2f(1.0f));
	printf("log2(2) = %e (%e)\n", log2f_approx(2.0f), log2f(2.0f));

	printf("log2(1-eps) = %e (%e)\n", log2f_approx(1.0f - FLT_EPSILON / 2.0f), log2f(1.0f - FLT_EPSILON / 2.0f));
	printf("log2(1+eps) = %e (%e)\n", log2f_approx(1.0f + FLT_EPSILON), log2f(1.0f + FLT_EPSILON));
	printf("log2(2-eps) = %e (%e)\n", log2f_approx(2.0f - FLT_EPSILON), log2f(2.0f - FLT_EPSILON));

	printf("log2(1.5) = %f (%f)\n", log2f_approx(1.5f), log2f(1.5f));
	printf("log2(6.0) = %f (%f)\n", log2f_approx(6.0f), log2f(6.0f));

	printf("abs err: %e @ %e (%a)\n", abs_err, abs_err_x, abs_err_x);
	printf("rel err: %e @ %e (%a)\n", rel_err, rel_err_x, rel_err_x);
}

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  #define clock_gettime(id, tp) \
	do { \
		clock_serv_t cclock; \
		mach_timespec_t mts; \
		host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock); \
		clock_get_time(cclock, &mts); \
		mach_port_deallocate(mach_task_self(), cclock); \
		(tp)->tv_sec = mts.tv_sec; \
		(tp)->tv_nsec = mts.tv_nsec; \
	} while (0)
#endif

__attribute__((noinline))
static void benchmark_approx(void)
{
	struct timespec tp1;
	struct timespec tp2;
	long long diff;

	float x;

	clock_gettime(CLOCK_REALTIME, &tp1);

	for (x = 1.0f; x < 2.0f; x += FLT_EPSILON) {
		volatile float q = log2f_approx(x);
	}

	clock_gettime(CLOCK_REALTIME, &tp2);

	diff = sub_timespec(tp2, tp1);
	printf("log2f (approx): %lld (%e log/s)\n", diff, (1.0 / FLT_EPSILON) * 1000000000 / diff);
}

__attribute__((noinline))
static void benchmark_exact(void)
{
	struct timespec tp1;
	struct timespec tp2;
	long long diff;

	float x;

	clock_gettime(CLOCK_REALTIME, &tp1);

	for (x = 1.0f; x < 2.0f; x += FLT_EPSILON) {
		volatile float q = log2f(x);
	}

	clock_gettime(CLOCK_REALTIME, &tp2);

	diff = sub_timespec(tp2, tp1);
	printf("log2f (exact):  %lld (%e log/s)\n", diff, (1.0 / FLT_EPSILON) * 1000000000 / diff);
}

int main()
{
	check_accuracy();

	benchmark_approx();
	benchmark_exact();
}
