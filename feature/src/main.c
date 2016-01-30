#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"

static void usage(void)
{
	puts("usage: vmaf_study app fmt ref dis w h\n"
	     "apps:\n"
	     "\tadm\n"
	     "\tansnr\n"
		 "\tmotion\n"
	     "\tvif\n"
		 "fmts:\n"
		 "\tyuv420\n"
		 "\tyuv422\n"
		 "\tyuv444"
	);
}

int main(int argc, const char **argv)
{
	const char *app;
	const char *ref_path;
	const char *dis_path;
	const char *fmt;
	int w;
	int h;
	int ret;

	if (argc < 7) {
		usage();
		return 2;
	}

	app      = argv[1];
	fmt		 = argv[2];
	ref_path = argv[3];
	dis_path = argv[4];
	w        = atoi(argv[5]);
	h        = atoi(argv[6]);

	if (w <= 0 || h <= 0)
		return 2;

	if (!strcmp(app, "adm"))
		ret = adm(ref_path, dis_path, w, h, fmt);
	else if (!strcmp(app, "ansnr"))
		ret = ansnr(ref_path, dis_path, w, h, fmt);
	else if (!strcmp(app, "vif"))
		ret = vif(ref_path, dis_path, w, h, fmt);
	else if (!strcmp(app, "motion"))
		ret = motion(ref_path, w, h, fmt);
	else
		return 2;

	if (ret)
		return ret;

	return 0;
}
