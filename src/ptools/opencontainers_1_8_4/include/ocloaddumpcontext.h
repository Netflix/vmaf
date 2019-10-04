#ifndef OCLOADDUMPCONTEXT_H_
#define OCLOADDUMPCONTEXT_H_

// These are helper functions for ocserialize.h/.cc and the OCPy serializtion
// routines in PythonCExt dir.  We factored them out so both sets of files 
// could see the implementation, but most people shouldn't care this file.

#include "ochashtable.h"  // Gives me the right HashFunction for OCDumpContext
#include "ocavlhasht.h"
// #include "arraydisposition.h"

OC_BEGIN_NAMESPACE

// For serialization
#define VALCOPY(T,N) { memcpy(mem,&N,sizeof(T));mem+=sizeof(T); }

// For Deserialization
#define VALDECOPY(T,N) { memcpy(&N,mem,sizeof(T)); mem+=sizeof(T); }


// This is an implementation class:  It allows us to track proxies so
// we don't serialize them twice.
struct OCDumpContext_ {
  OCDumpContext_ (char* start_mem, bool compat) : 
    mem(start_mem), compat_(compat) { }

  char* mem;  // Where we currently are in the buffer we are dumping into

  // Lookup table for looking up the markers:  When a proxy comes
  // in, we need to know if we have seen it before, and if so,
  // what proxy it refers to.  Note that we hold each proxy by
  // pointer because we are ASSUMING that nothing is changing the
  // table in question (and avoids extra locking)
  AVLHashT<void*, int_4, 8> lookup_;

  // Compatibility mode: Starting with PicklingTools 1.2.0, we support
  // OTab and Tup: if previous sytems don't support these, we need to
  // be able to turn OTabs->Tab and Tup->Arr.
  bool compat_;  // true means convert OTab->Tab, Tup->Arr

}; // OCDumpContext_


// Helper class to keep track of all the Proxy's we've seen so we don't have
// to unserialize again
struct OCLoadContext_ {
  OCLoadContext_ (char* start_mem, bool compat) :
    mem(start_mem), compat_(compat) { }

  char* mem; // Where we are in the buffer
  // When we see a marker, see if we have already deserialized it.  If
  // we see it, there is some Proxys already out there
  AVLHashT<int_4, Proxy, 8> lookup_;
  
  // See OCDumpContext for discussion of compat_
  bool compat_;

}; // OCLoadContext_


OC_END_NAMESPACE


#endif //  OCLOADDUMPCONTEXT_H_
