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

#define _POSIX_C_SOURCE 200112L

#include <stddef.h>
#include <stdlib.h>
#include "mem.h"

void *aligned_malloc(size_t size, size_t alignment)
{
	void *ptr;

#if defined(_MSC_VER) || defined(__MINGW32__)
    if ((ptr = _aligned_malloc(size, alignment)) == NULL)
#else
    if (posix_memalign(&ptr, alignment, size))
#endif
		return 0;
	else
		return ptr;
}

void aligned_free(void *ptr)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
	free(ptr);
#endif
}
