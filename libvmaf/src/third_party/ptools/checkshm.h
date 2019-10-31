#ifndef CHECKSHM_H_
#define CHECKSHM_H_

// Call InSHM to check and see if a data structure
// (and all its constituents) are in Shared Memory (as specified
// by the given allocator).  This "tool" is 
// most useful when using cross-process shared memory,
// so you can check if a data structure will "cross"
// process boundaries (a false means: NO! this won't
// work in the process you want to send to).  By default, 
// this will just return true or false for a data structure:
// true meaning all of a data structure has been placed
// in shared memory, false meaning "some piece" was not
// (and you probably shouldn't use it for cross-process SHM).
// If you want more info about where the failure was,
// change "throw_on_error" to true, and an exception with
// more context (in the .what() of the exeception) will
// help debug it.

#include "ocval.h"

PTOOLS_BEGIN_NAMESPACE

// ///////////////////////////////////////////////// In SHM

// Choose whether you want a silent invalid return (with a false
// returned) or an explicit error from an exception (thrown runtime_error)
// when we discover something not in memory.
//
// We use the macro version so the string is ONLY constructed if we are
// trying to return an error. 
#define NOTINSHMERROR(errs, throw_on_error) ((throw_on_error) ? (throw runtime_error(errs)), false  : (false))

// ///// Forwards
inline bool InSHM (const Val& a, StreamingPool* sp, 
		   bool throw_on_error=false, const char* error_context="");

// Check to see if the the address (and its bytes length) are
// contained within the given streaming pool.
inline bool IsContainedMemory (const void* addr, int bytes, StreamingPool *sp)
{
  char* mem = (char*)sp;
  char* thing = (char*)addr;  // where in memory is object
  if ((mem < thing) && (thing < mem+sp->bytes()) &&
      (mem < thing+bytes) && (thing+bytes < mem+sp->bytes())) {
    return true;
  } else {
    return false;
  }
}

template <class T>
inline bool TopLevelSHMCheck_ (const T& t, StreamingPool *sp,
			       bool throw_on_error, const char* error_context)
{
  // Check top-level table
  if (t.allocator() == 0) {
    string repr = Stringize(t);
    return NOTINSHMERROR("The value "+repr+" is not using the shared "
			 "memory allocator. "+string(error_context), 
			 throw_on_error);
  } else if (t.allocator() != sp) {
    string repr = Stringize(t);
    return NOTINSHMERROR("The value "+repr+" is using a different shared "
			 "memory? " + string(error_context), 
			 throw_on_error);
  } else {
    return true;
  }
}

template <class T>
inline bool InSHM (const Array<T>& a, StreamingPool* sp, 
		   bool throw_on_error=false,
		   const char* error_context="")
{ return TopLevelSHMCheck_(a, sp, throw_on_error, "array"); }


inline bool InSHM (const Arr& a, StreamingPool* sp, bool throw_on_error=false,
		   const char* error_context="")
{
  const Array<Val>& av = a;
  bool ok = InSHM(av, sp, throw_on_error);
  if (!ok) {
    return false;
  } 
  // Assertion: basics okay, check each individual member
  int len = int(a.length());
  for (int ii=0; ii<len; ii++) {
    bool single_check = 
      InSHM(a[ii], sp, throw_on_error, "element of an array");
    if (!single_check) return false;
  }
  return true;
}

inline bool InSHM (const Tup& u, StreamingPool* sp, bool throw_on_error=false,
		   const char* error_context="")
{ return InSHM(u.impl(), sp, throw_on_error, "tuple"); }


template <class T>
inline bool InSHMTable_ (const T& t, StreamingPool* sp, 
			 bool throw_on_error=false, 
			 const char* error_context="")
{

  // Recursively check
  for (It ii(t); ii();) {
    const Val& key = ii.key();
    Val& value = ii.value();
    if (!InSHM(key, sp, throw_on_error, error_context)) {
      return NOTINSHMERROR("key:"+string(key) + ". " + error_context, 
			   throw_on_error);
    }
    if (!InSHM(ii.value(), sp)) {
      return NOTINSHMERROR("key: "+string(key)+" and value:" +string(value) + 
			   ". " + error_context, 
			   throw_on_error);
    }
  }
  return true;
}

inline bool InSHM (const Tab& t, StreamingPool* sp, bool throw_on_error=false,
		   const char* error_context="")
{ return InSHMTable_(t, sp, throw_on_error, "Tab"); }

inline bool InSHM (const OTab& t, StreamingPool* sp, bool throw_on_error=false,
		   const char* error_context="")
{ return InSHMTable_(t, sp, throw_on_error, "OTab"); }

inline bool InSHM (const OCString& s, StreamingPool* sp, 
		   bool throw_on_error=false, const char* error_context="")
{ 
  // Regardless, string has to be in memory!
  if (!IsContainedMemory(s.c_str(), s.length(), sp)) {
    return NOTINSHMERROR(s.c_str(), throw_on_error);
  } 
  
  // Still, check allocator
  if (s.allocator() == 0) {
    // String uses internal space, so not allocator ... that's okay
    // in this case as string just fits in memory.  Since we've
    // already checked string, we are okay
    return true;
  } else if (s.allocator() != sp) {
    return NOTINSHMERROR("using wrong allocator for string", throw_on_error);
  } else { // allocators the same, already checked string in mem
    return true;
  }
  return true;
}


#if !defined(OC_USE_OC_STRING)
inline bool InSHM (const string& s, StreamingPool* sp, 
		   bool throw_on_error=false, const char *error_context="")
{
  if (IsContainedMemory(s.data(), s.length(), sp)) {
    return NOTINSHMERROR(s.data(), throw_on_error);
  }
}
#endif

inline bool InSHM (const int_n& t, StreamingPool* sp, 
		   bool throw_on_error=false, const char *error_context="")
{ return TopLevelSHMCheck_(t, sp, throw_on_error, "int_n"); }

inline bool InSHM (const int_un& t, StreamingPool* sp, 
		   bool throw_on_error=false, const char *error_context="")
{ return TopLevelSHMCheck_(t, sp, throw_on_error, "int_n"); }


// Check and see if given Val is in shared memory
inline bool InSHM (const Val& v, StreamingPool* sp, 
		   bool throw_on_error, const char *error_context)
{
  // Base case: Check top-level container.
  if (!v.isproxy) {
    bool ok = TopLevelSHMCheck_(v, sp, throw_on_error, "plain Val.");
    if (!ok) return false;
  }

  // Recursive case:
  // Proxy handled by base check, internals by checks below ...
  switch (v.tag) {
  case 's': 
  case 'S': 
  case 'i': 
  case 'I': 
  case 'l': 
  case 'L': 
  case 'x': 
  case 'X': 
  case 'b': 
  case 'f': 
  case 'd': 
  case 'F': 
  case 'D': 
  case 'Z': // All bit-blit types
    return IsContainedMemory(&v.u.D, sizeof(complex_16), sp); 
  
  // Complex containers
  case 'a': { OCString*p=(OCString*)&v.u.a;return InSHM(*p,sp,throw_on_error); }

    // Use the implict conversion so it'll handle Proxys okay
  case 't': { Tab&  t=v;  return InSHM(t,sp,throw_on_error);break; }
  case 'o': { OTab& o=v;  return InSHM(o,sp,throw_on_error);break; }
  case 'u': { Tup&  u=v;  return InSHM(u,sp,throw_on_error);break; }
  case 'q': { int_n* ip=(int_n*)&v.u.q;  return InSHM(*ip,sp,throw_on_error);break; }
  case 'Q': { int_un*ip=(int_un*)&v.u.Q; return InSHM(*ip,sp,throw_on_error);break; }
  case 'n': {
    switch(v.subtype) {
    case 'Z': {
      Arr& a = v; return InSHM(a, sp, throw_on_error); break;
    } 
    case 's': 
    case 'S': 
    case 'i': 
    case 'I': 
    case 'l': 
    case 'L': 
    case 'x': 
    case 'X': 
    case 'b': 
    case 'f': 
    case 'd': 
    case 'F': 
    case 'D': // Type doesn't matter, just need the allocator, as that'll
              //  check everything for us
    {
      if (v.isproxy) { throw runtime_error("not handled yet!"); }
      Array<int_1> *ap = (Array<int_1>*)&v.u.n;
      return InSHM(*ap, sp, throw_on_error);
      break;
    }
    default:
      throw runtime_error("Can't handle arrays of "+string(1,v.tag));
    }
  }
  default:
    throw runtime_error("Unknown type tag:"+string(1, v.tag));
  }

  return true;
}

PTOOLS_END_NAMESPACE

#endif // CHECKSHM_H_
