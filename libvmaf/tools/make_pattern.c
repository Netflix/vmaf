#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef enum pattern_type {
	PATTERN_FIXED,
	PATTERN_CHECKER_EVEN,
	PATTERN_CHECKER_ODD,
	PATTERN_URANDOM,
	PATTERN_BAD_PATTERN
} pattern_type;

unsigned char get_char(int i, int j, pattern_type pattern, int param)
{
	switch (pattern) {
	case PATTERN_FIXED:
		return param;
	case PATTERN_CHECKER_EVEN:
		return (i % 2 == j % 2) ? param : 0;
	case PATTERN_CHECKER_ODD:
		return (i % 2 == j % 2) ? 0 : param;
	case PATTERN_URANDOM:
		return (unsigned)rand() >> 8;
	default:
		return 0;
	}
}

int main(int argc, const char **argv)
{
	const char *path;
	FILE *file;
	int w;
	int h;
	int type;
	int param;

	int i, j;
	int ret = 1;

	if (argc < 5)
		return 2;

	path  = argv[1];
	w     = atoi(argv[2]);
	h     = atoi(argv[3]);
	type  = atoi(argv[4]);
	param = argc > 5 ? atoi(argv[5]) : 0;

	if (w <= 0 || h <= 0 || type < 0 || type >= PATTERN_BAD_PATTERN)
		return 2;

	if (!(file = fopen(path, "wb")))
	{
		goto fail;
	}

	srand(time(0));

	for (i = 0; i < h; ++i) {
		for (j = 0; j < w; ++j) {
			unsigned char x = get_char(i, j, type, param);

			if (fwrite(&x, 1, 1, file) != 1)
			{
				goto fail;
			}
		}
	}
fail:
	if (file)
		fclose(file);
	return ret;
}
