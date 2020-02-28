/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/**
 * Note: stride is in terms of bytes
 */
int write_image(FILE *wfile, const void *buf, int width, int height, int stride, int elem_size)
{
	const char *byte_ptr = buf;
	int i;
	int ret = 1;

	if (width <= 0 || height <= 0 || elem_size <= 0)
	{
		goto fail_or_end;
	}

	for (i = 0; i < height; ++i)
	{
		if (fwrite(byte_ptr, elem_size, width, wfile) != (size_t)width)
		{
			goto fail_or_end;
		}

		byte_ptr += stride;
	}

	ret = 0;

fail_or_end:
	return ret;
}
