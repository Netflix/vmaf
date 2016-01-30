#pragma once

#ifndef ALLOC_H_
#define ALLOC_H_

#define MAX_ALIGN 32

#define ALIGN_FLOOR(x) ((x) - (x) % MAX_ALIGN)
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))

void *aligned_malloc(size_t size, size_t alignment);

void aligned_free(void *ptr);

#endif /* ALLOC_H_ */
