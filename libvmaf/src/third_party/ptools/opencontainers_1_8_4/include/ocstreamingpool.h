#ifndef OCSTREAMINGPOOL_H_

// The StreamingPool manages a memory pool, allowing a user to
// allocate and deallocate pieces of memory from the pool of memory
// (like C with mallocs and free).  This is "optimized" for reqeusts
// that tend to have a FIFO quality.  The allocation schedule doesn't
// have to be FIFO (and in many uses won't be), but the idea is that
// data is streaming through a system. The data gets allocated at the
// start of a chain, streams through the system, then gets deallocated
// at the end.  Although malloc and free could handle this, we would
// use this allocator to avoid prossible fragmentation.

// This pool has a extra quality: it is completely self-contained.
// All information needs to be stored IN the pool because this pool
// needs to be usable in a shared memory environment. Imagine multiple
// processes where each process looks into the pool: all relevant
// StreamingPool info has to be stored in the pool itself.  In other
// words, no external bookkeeping: all bookkeeping has to be in the
// memory region itself.

#include "ocport.h"
#include "ocspinfo.h"
#include "ocsynchronizer.h"


OC_BEGIN_NAMESPACE

// The number of small allocators to create at startup.
#define SP_SMALL_ALLOCATORS 2
template <size_t, size_t> class FixedSizeAllocator; // Forward


// A memory alloctor for Streaming Data and shared memory allocations
class StreamingPool {
  
 public:

  // The "worst" alignment restriction you can have: 8 is good enough
  // for just about everything, but some MMX instructions need 16
  // byte aligned data.  If something in the future needs even
  // stricter alignments, you can set this to some larger power of two
  enum { ALIGN_MAX = 16 };

  // When constructing a StreamingPool, this will be constructed OVER
  // a piece of given memory, which is why there is no constructor nor
  // destructor: You must use the static method below for
  // construction.  When you are "done" with the pool, you schedule it
  // for deletion (see method below).
  static StreamingPool* CreateStreamingPool (char* memory, int_ptr bytes,
					     int alignment_restrictions=4,
					     bool small_allocators=false)
  { 
    if (bytes<=int_ptr(sizeof(StreamingPool))) {
      throw logic_error("Not really enough memory to manage");
    }
    return  new (memory) StreamingPool(memory, bytes, alignment_restrictions,
				       small_allocators);
  }

  // When the pool is pristine and you want to "clean up".  This
  // doesn't actually return the memory (we don't know it if was
  // allocated in process shared memory, from the free store, etc.)
  // to the system, only clean UP the memory so that the pool can
  // be drained.
  // (1) "Anybody in the pool?"  (isPristine)
  // (2) "Clean all the crap out of the pool"  (CleanStreamingPool)
  // (3) "Drain Pool"  (somehow return memory to system:  delete mem?)
  static void Clean (StreamingPool* sp)
  { 
    if (!sp->isPristine()) {
      throw logic_error("The pool still has things in it, can't clean");
    }
    sp->~StreamingPool();
  }
  
  // Ask for a piece of memory of the given size from the pool.
  OC_INLINE char* allocate (int_ptr bytes_requested);
  
  // Return a piece of memory given by allocate 
  OC_INLINE void deallocate (char* start_of_user_data);
  
  // Indicate we are done with the this pool.  When the last piece of
  // memory is returned to the pool (via some thread calling
  // deallocate), this will release all ressources of the memory (but
  // not the memory itself: that must be explicitly deallocated by the
  // user, reversing how it was allocated).  TODO: Allow hook for
  // destruction of memory?
  OC_INLINE void scheduleForDeletion ();
  
  // See if the pool is in pristine state, meaning all 
  // allocations have been returned to the pool
  OC_INLINE bool isPristine () const;
  
  // See if the pool is completely full, meaning no more memory
  // available to give: This means there are NO free nodes at all of
  // any size
  OC_INLINE bool isFull () const;
  
  // How big is the biggest free object in the pool?  Don't allocate,
  // just tell me how big it is (i.e., an "allocate" of this size
  // would get the free piece in question).
  OC_INLINE int_ptr biggestFree ();

  // Prepare some memory, given with allocate above, marking it with
  // magic cookies so that we can tell it came from this allocator.
  // NOTE: Once a user "fills" memory returned by allocate, the magic
  // cookie is gone.  This is just a programmatic way to magic cookie
  // uninitialized memory.
  OC_INLINE void setMagicCookie (char* memory);

  // Check to see if a piece of memory was prepared by this allocator
  // and had its magic cookie set.  If the magic cookie was set,
  // return the allocator that prepared this memory, otherwise returns
  // 0.
  OC_INLINE static StreamingPool* CheckMagicCookie (char* memory);
					  

  // Make sure magic cookie is gone from allocated memory (ONLY use this
  // on memory that is still allocated!).
  OC_INLINE void cleanMagicCookie (char* memory);


  // If needed, give back the StreamingPool from a piece of memory
  // allocated from the StreamingPool.  This is useful in some
  // circumstances when you want to allocate memory from the same
  // pool as another piece of memory you have.
  OC_INLINE static StreamingPool& GetStreamingPool (char* memory_from_allocate);

  // Somewhat esorteric fuction to get a pointer to first allocated
  // thing in pool
  OC_INLINE SPInfo_* firstAllocated (); 

  // Number of TOTAL bytes allocated when pool was created
  int_ptr bytes () const { return bytes_; }

 protected :
    
   OC_INLINE friend SPInfo_* GetFreeList (StreamingPool*);
 
   // Constructor and destructor protected so that they only way it can
   // be constructed is over a piece of unitialized memory
   StreamingPool (char* memory, int_ptr bytes, int alignment_restrict,
		  bool supports_small_allocators);
   ~StreamingPool () { }

   // Don't allow copies.
   StreamingPool (const StreamingPool&); // NO IMPL
   StreamingPool& operator= (const StreamingPool&);  // NO IMPL
   
   // The pool has to start at a certain place (not just
   // &poolData_[0]) so the user memory of the first node meets
   // alignment restictions
   SPInfo_* pool_() const { return (SPInfo_*) &poolData_[poolOffset_]; }   


   // Figure out how big the maximum SPInfo block is for the given
   // amount of memory
   OC_INLINE int_ptr maximumBlockSize_ ();

   // If small enpugh, we can defer to small Memory allocator
   OC_INLINE char *handleSmallAllocation_ (int_ptr bytes_requested);

   // Check and see if the memory is allocated from the general
   // heap or one of the small memory allocators that are
   // managed by the general heap (and still in the heap proper)
   OC_INLINE bool handleSmallDeallocation_ (char* start_of_user_data);


   // ****** The order of these data members is pretty important: The
   // first few bytes are overlaid onto the memory as the data members
   // for the StreamingPool, and the rest of memory is managed as the
   // actual memory pool.

   // What are the alignment restrictions on memory from the pool?
   // Usually user memory has to be given on 8 byte boundaries, but
   // sometimes (like MMX instructions) alignment has to be 16 bytes
   int_4 alignment_;

   // When setting up the very first "free" node, we have to align
   // it in memory, so this will be an offset.   In other words,
   // the first SPINFO in memory starts at &pool_[poolOffset_]
   int_ptr poolOffset_;

   // When this VectorPool is ready for deletion, schedule it so that
   // when its last Vector comes, it deletes everything (including
   // itself) then.
   int_4 scheduledForDeletion_;

   // Total number of bytes of memory:  this includes all the overhead
   // of Mutexes, etc.  If you want the amount of memory the user can
   // actually use, compute that below
   int_ptr bytes_;

   // All allocates and deallocates must use this lock so we keep 
   // our data structures consistent
   Mutex lock_;

   // Pointers to some small FixedAllocators, and some locks
   // so each one has its own: this way small allocations don't
   // hold up big allocations.
   FixedSizeAllocator<64,256> *smallAllocators_[SP_SMALL_ALLOCATORS];

   // The freelist in memory.  it is really only used as a dummy node
   // on the circular, doubly-linked list for all free nodes in
   // memory.
   SPInfo_ freelist_;

   // The start of "managed memory":  really, a bunch of SPInfo_s
   // laid out in memory.   Because we have to lay out the first one
   // in memory at a certain alignment, this may be adjusted upto
   // "alignment_" extra byte
   char poolData_[sizeof(SPInfo_)+ALIGN_MAX];
     // SPInfo_ pool_;

}; // StreamingPool


// Right now, just a typedef.  This may become more complex as it has more
// implementation.  We have to be careful to have no virtual functions ...
// (see the Proxy class for reasons why)
typedef StreamingPool Allocator;

OC_END_NAMESPACE

OC_BEGIN_NAMESPACE
inline char* Allocate (Allocator* a, size_t len) { return a->allocate(int_ptr(len)); }
inline void DeAllocate (Allocator* a, char* m) { a->deallocate(m); }
OC_END_NAMESPACE

// The implementation: can be put into a .o if you don't want
// everything inlined.
#if !defined(OC_FACTOR_INTO_H_AND_CC)
# include "ocstreamingpool.cc"
#endif 



#define OCSTREAMINGPOOL_H_
#endif // OCSTREAMINGPOOL_H_
