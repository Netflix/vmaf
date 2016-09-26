#ifndef OC_STRING_H_
#define OC_STRING_H_

// Thin thin wrapper to allow us to use OCString without needing the
// Allocators (or their definitions) until link time.

#include "ocstring_impl.h"
//#include "ocstreamingpool.h"

//OC_BEGIN_NAMESPACE
//inline char* Allocate (Allocator* a, int len) { return a->allocate(len); }
//inline void DeAllocate (Allocator* a, char* m) { a->deallocate(m); }
//OC_END_NAMESPACE



#endif // OC_STRING_H_



