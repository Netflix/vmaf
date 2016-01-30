#include <fenv.h>
#include <stdio.h>

volatile double x = 0x1.ac4bdd6e3f184p-1;
volatile double y = 255.0;

const char *fegetround_str(void)
{
	int mode = fegetround();

	switch (mode) {
	case FE_TONEAREST:
		return "FE_TONEAREST";
	case FE_DOWNWARD:
		return "FE_DOWNARD";
	case FE_UPWARD:
		return "FE_UPWARD";
	case FE_TOWARDZERO:
		return "FE_TOWARDZERO";
	default:
		return "ERROR";
	}
}

int main()
{
	int mode = fegetround();
	printf("default rounding mode: %s\n", fegetround_str());

	printf("result of %a * %a (%.20e * %f)\n", x, y, x, y);

	fesetround(FE_TONEAREST);
	printf("%s:\t%a (%.20e)\n", fegetround_str(), x * y, x * y);

	fesetround(FE_DOWNWARD);
	printf("%s:\t%a (%.20e)\n", fegetround_str(), x * y, x * y);

	fesetround(FE_UPWARD);
	printf("%s:\t%a (%.20e)\n", fegetround_str(), x * y, x * y);

	fesetround(FE_TOWARDZERO);
	printf("%s:\t%a (%.20e)\n", fegetround_str(), x * y, x * y);

	fesetround(mode);
}
