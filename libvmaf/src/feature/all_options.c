/**
 * Note: stride is in terms of bytes
 */
int offset_image_s(float *buf, float off, int width, int height, int stride)
{
	char *byte_ptr = (char *)buf;
	int ret = 1;
	int i, j;

	for (i = 0; i < height; ++i)
	{
		float *row_ptr = (float *)byte_ptr;

		for (j = 0; j < width; ++j)
		{
			row_ptr[j] += off;
		}

		byte_ptr += stride;
	}

	ret = 0;

	return ret;
}
