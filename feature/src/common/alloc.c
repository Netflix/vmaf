#define _POSIX_C_SOURCE 200112L

#include <stddef.h>
#include <stdlib.h>
#include "alloc.h"

void *aligned_malloc(size_t size, size_t alignment)
{
	void *ptr;

	if (posix_memalign(&ptr, alignment, size))
		return 0;
	else
		return ptr;
}

void aligned_free(void *ptr)
{
	free(ptr);
}
