#ifndef OCSER_H_

#include "ochashtable.h"
#include "ocval.h"

OC_BEGIN_NAMESPACE

// The purpose of this class is to Serialize the Tab/Vals/Arrs in the
// same way as BytesToSerialize/Serialize.  This is mostly a test
// class so we can compare implementations.  *****You should prefer the
// routines in ocserialize.h,cc because they are almost always faster.

// HISTORY: basically, this class was an experiment to see if the
// paradigm of (1) BytesToSerialize/Serialize was faster than a (2)
// OCSerializer with dynamically resizing buffer.  (1) was typically
// 15% faster, and used less memory/memory footprint (which also
// affects speed).  This is still useful as "another way" of
// serializing for testing purposes (comparing implementations to make
// sure we get the same answer), which is why we keep it.

class OCSerializer {

  enum { SER_START_LEN=1024 };

 public:

  // Create a "serialization session" with an "expected amount of
  // memory" need to serialize some Val/Tab/Arr.
  OCSerializer (size_t start_capac=SER_START_LEN) :
    mem_(new char[start_capac]),
    len_(0),
    cap_(start_capac)
  { }

  // All done
  ~OCSerializer () { delete [] mem_; }

  // Serialize the given constructs into current buffer
  OC_INLINE void put (const Val& v);
  OC_INLINE void put (const Str& v);
  OC_INLINE void put (const Tab& t);
  OC_INLINE void put (const Arr& a);
  template <class T> 
    OC_INLINE void put (const Array<T>& a);
  OC_INLINE void put (const Proxy& p);

#if !defined(OC_USE_OC_STRING)
  OC_INLINE void put (const OCString& s);
#endif

  // Return the memory to the user to delete on his own
  char* abandon () { char* m = mem_; mem_ = 0; return m; }

  // Peek at the memory, don't have to delete
  char* peek (size_t& len) { len = len_; return mem_; }

 protected:

  // Lookup table for looking up the markers:  When a proxy comes
  // in, we need to know if we have seen it before, and if so,
  // what proxy it refers to.  Note that we hold each proxy by
  // pointer because we are ASSUMING that nothing is changing the
  // table in question (and avoids extra locking)
  AVLHashT<void*, int_4, 8> lookup_;

  // Memory we serialize into. 
  char* mem_;
  size_t len_;
  size_t cap_;

  // Make sure there is enough space (in bytes) for the next few puts.
  // Returns a pointer to the first of the next n bytes that are
  // reserved (so you are allowed to write into what's returned for
  // next bytes)
  char* reserve_ (size_t bytes)
  {   
    // 99% of the time, this is all we do
    // len_+=bytes;

    // The other 1%, we can do more work
    const size_t old_len = len_;
    if ((len_+=bytes)>cap_) {
      // Allocate on side, in case fails, still exception safe

      // cerr << "bytes" << bytes << " len_" << len_ << "cap_ " << cap_ << endl;
      // By keeping memory "smaller", we speed up the serialization as
      // memory doesn't grind as hard.
      size_t shift = 1;
      if (cap_>1024) shift = 2;
      if (cap_>65536) shift = 3;
      if (cap_>1000000) shift = 4;
      size_t new_cap = cap_ + (cap_<<shift);
      if (new_cap < len_) new_cap = len_ + (len_<<shift);

      char* new_mem = new char[new_cap];
      memcpy(new_mem, mem_, len_-bytes);
      
      // Install so exception safe
      cap_ = new_cap;
      delete [] mem_;
      mem_ = new_mem;
    }
    return mem_+old_len;
  }

}; // OCSerializer


// Need a bit more state as we deserialize Proxies
class OCDeserializer {
 public:
  
  // Create a "session" for Deserialization: the given piece of memory
  // has some values serialized into it: if we adopt the memory, the
  // OCDeserializer has responsibility to delete [] it (so it assumes
  // the memory was allocated with new char[length]).
  OCDeserializer (char* memory, bool adopt_memory=false) :
    adoptMemory_(adopt_memory),
    mem_(memory),
    memStart_(memory)
    { }
  
  // All done, clean up deserilization
  ~OCDeserializer () { if (adoptMemory_) delete [] memStart_; }

  // Deserialize into the given Val: ASSUMPTION: v is initially None
  OC_INLINE void load (Val& v);

 protected:

  // See if we adopt the memory we are reading out off
  bool adoptMemory_;

  // The memory we ar reading from (always being updated)
  char* mem_;
  char* memStart_;  // Where it started from (so we can delete)

  // When we see a marker, see if we have already deserialized it.  If
  // we see it, there is some Proxys already out there
  AVLHashT<int_4, Proxy, 8> lookup_;  

  // Helper routines: these do inplace construction on the piece
  // of memory given
  OC_INLINE void load (OCString* tp, char tag='a');   
                 // Load and create a Str in raw memory 'a':int_u4, 'A':int_u8
  OC_INLINE void load (Tab* tp, char tag='t'); 
                 // Load and create a Tab in raw memory 't':int_u4, 'T':int_u8
  OC_INLINE void load (Arr* tp, char tag='n'); 
                 // Load and create a Arr in raw memory 'n':int_u4, 'N':int_u8
  OC_INLINE void load (Proxy* pp); // Load and create a Proxy in raw memory

  template <class T>
    OC_INLINE void load (Array<T>* tp, char tag='n'); // Load and create an Array in raw memory

  
}; // OCDeserializer

OC_END_NAMESPACE

// The implementation: can be put into a .o if you don't want
// everything inlined.
#if !defined(OC_FACTOR_INTO_H_AND_CC)
# include "ocser.cc"
#endif

#define OCSER_H_
#endif // OCSER_H_
