#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_TYPE
  #define DATA_TYPE double
#endif

typedef DATA_TYPE data_t;

int main(int argc, const char **argv)
{
	FILE *file = 0;
	char buf[1024];
	int n;
	int i, j;

	if (argc != 3)
		return 1;

	file = fopen(argv[1], "rb");
	n = atoi(argv[2]);

	for (i = 0; i < n; ++i) {
		data_t d;
		char *ptr = buf;

		memset(buf, 0, sizeof(buf));

		for (j = 0; j < n; ++j) {
			if (fread(&d, sizeof(data_t), 1, file) != 1) {
				fclose(file);
				return 1;
			}

			ptr += sprintf(ptr, "%a%s", d, j == n - 1 ? "" : ", ");
		}
		puts(buf);
	}
	fclose(file);
}
