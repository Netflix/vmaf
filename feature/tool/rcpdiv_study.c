#define _POSIX_C_SOURCE 199309L

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifdef __MACH__
  #include <mach/clock.h>
  #include <mach/mach.h>
#endif

#include <emmintrin.h>

static float rcp_ieee(float x)
{
	return 1.0f / x;
}

static float rcp_sse(float x)
{
	return _mm_cvtss_f32(_mm_rcp_ss(_mm_set1_ps(x)));
}

static float rcp_newton(float x)
{
	float xi;

	xi  = rcp_sse(x);
	xi = xi + xi * (1.0f - x * xi);

	return xi;
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
		float exact  = rcp_ieee(x);
		float approx = rcp_newton(x);

		double abs_err_curr = fabsf(exact - approx);
		double rel_err_curr = abs_err_curr / exact;

		if (abs_err_curr > abs_err) {
			abs_err   = abs_err_curr;
			abs_err_x = x;
		}
		if (rel_err_curr > rel_err) {
			rel_err   = rel_err_curr;
			rel_err_x = x;
		}
	}

	printf("rcp(1) = %e (%e)\n", rcp_newton(1.0f), 1.0);
	printf("rcp(2) = %e (%e)\n", rcp_newton(2.0f), 0.5);

	printf("rcp(1-eps) = %e (%e)\n", rcp_newton(1.0f - FLT_EPSILON / 2.0f), rcp_ieee(1.0f - FLT_EPSILON / 2.0f));
	printf("rcp(1+eps) = %e (%e)\n", rcp_newton(1.0f + FLT_EPSILON), rcp_ieee(1.0f - FLT_EPSILON));
	printf("rcp(2-eps) = %e (%e)\n", rcp_newton(2.0f - FLT_EPSILON), rcp_ieee(2.0f - FLT_EPSILON));

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
		volatile float q = rcp_newton(x);
	}

	clock_gettime(CLOCK_REALTIME, &tp2);

	diff = sub_timespec(tp2, tp1);
	printf("rcp (approx): %lld (%e rcp/s)\n", diff, (1.0 / FLT_EPSILON) * 1000000000 / diff);
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
		volatile float q = rcp_ieee(x);
	}

	clock_gettime(CLOCK_REALTIME, &tp2);

	diff = sub_timespec(tp2, tp1);
	printf("rcp (exact):  %lld (%e rcp/s)\n", diff, (1.0 / FLT_EPSILON) * 1000000000 / diff);
}

int main()
{
	check_accuracy();

	benchmark_approx();
	benchmark_exact();
}
