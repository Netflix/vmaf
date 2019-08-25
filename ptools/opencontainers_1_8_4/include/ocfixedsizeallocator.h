#ifndef OCFIXEDSIZEALLOCATOR_H_
#define OCFIXEDSIZEALLOCATOR_H_

#include "ocspinfo.h"
#include "ocsynchronizer.h"

OC_BEGIN_NAMESPACE

// This is a simple fixed size allocator (with minimal allocation
// and deallocation overhead) that can be overlain onto any memory.
//
// FixedSizeAllocator<64, 256> is pretty typical:  Allocations of 64
// bytes, and there are 256 (only of which 255 are available) allocated.
// [Why only 256?  To keep overhead minimal, we are using int_u1
//  which means we can't represent all 0-255 drawers AND an empty value.
//  Thus "0" becomes the empty value, and the very first entry
//  can'be used.]
template <size_t SIZE, size_t DRAWERS>
class FixedSizeAllocator {

 public :

  FixedSizeAllocator (bool shared_across_processes = false) :
    lock_(shared_across_processes) // Locks in cross-process SHM need to be told
  {
    if (DRAWERS < 2 || DRAWERS>256) {
      throw runtime_error("FixedSizeAllocator can only supports DRAWERS >=2 and <= 256");
    }
    // First drawer used for management
    freelist_() = 1;  // Front of free list is node 0
    int_u1 *drawer = &storage_[SIZE];
    for (unsigned char ii=1; ii<DRAWERS-1; ii++) {
      drawer[0] = ii+1; // Next on free list is the next node in the array
      drawer += SIZE;   // Move to next drawer
    }
    // Last drawer, end of list marker (0)
    drawer[0] = 0;
  }


  // Allocate a chunk of memory of size SIZE.  Returns
  // the adress of the memory is succeeds, otherwise 0.
  char *allocate ()
  {
    if (freelist_()==0) return NULL;  // All full!

    // Get index of next free entry
    int index_of_first_free = freelist_();
    unsigned char *mem_of_next_free = &storage_[0] + index_of_first_free * SIZE;

    // update freelist with pointer to next index
    int index_of_next_free = mem_of_next_free[0];
    freelist_() = index_of_next_free;
    char *mem = reinterpret_cast<char*>(mem_of_next_free);
    return mem;
  }

  // Return a piece of memory allocated by allocate back to
  // the fixed size allocator pool.
  void deallocate (char *user_mem)
  {
    unsigned char *mem = reinterpret_cast<unsigned char*>(user_mem);
    int index_of_this_drawer = (mem - &storage_[0])/SIZE;
    mem[0] = freelist_();
    freelist_() = index_of_this_drawer;
  }

  // Return true if the piece of memory is managed by this FixedSizeAllocator
  // This allows the user to check and make sure before a deallocation
  bool isMemoryFromHere (char *memory)
  {
    unsigned char* user_mem = reinterpret_cast<unsigned char*>(memory);
    return (user_mem >= &storage_[SIZE]) &&  // because drawer 0 is freelist
           (user_mem < &storage_[SIZE*DRAWERS]);
  }


  // Print some repr of the allocator: not guaranteed to show anything
  // except debugging info
  void print (ostream& os)
  {
    os << "FREELIST:" << int(freelist_()) << endl;

    // Find all free nodes;
    unsigned char freed[DRAWERS] = { 0 };
    unsigned char index = freelist_();
    while (index != 0) {
      // mark this one inuse:
      freed[index] = 1;

      // Get index of next free entry
      unsigned char *mem_of_next_free = &storage_[0] + index*SIZE;
      index = mem_of_next_free[0];
    }

    // Print em all
    for (size_t ii=1; ii<DRAWERS; ii++) {
      os << ii;
      if (freed[ii]) os << ":F ";
      else os << ":I ";
    }
    os << endl;
  }

  bool isPristine ()
  {
    // Find all free nodes;
    unsigned char freed[DRAWERS] = { 0 };
    unsigned char index = freelist_();
    while (index != 0) {
      // mark this one inuse:
      freed[index] = 1;

      // Get index of next free entry
      unsigned char *mem_of_next_free = &storage_[0] + index*SIZE;
      index = mem_of_next_free[0];
    }
    for (size_t ii=1; ii<DRAWERS; ii++) {
      if (!freed[ii]) return false;
    }
    return true;
  }

  // Lock for mutual exclusion
  Mutex& lock () { return lock_; }

 protected:

  // Overlaid onto memory: the first one is used an management,
  // only can store 255 in here.  A 0 on the freelist means nothing
  // is free.
  unsigned char storage_[SIZE * DRAWERS];
  unsigned char& freelist_ () { return storage_[0]; }

  // Mutex
  Mutex lock_;

}; // FixedSizeAllocator

OC_END_NAMESPACE

#endif // OCFIXEDSIZEALLOCATOR_H_
