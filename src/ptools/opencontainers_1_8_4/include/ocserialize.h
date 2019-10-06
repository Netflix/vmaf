#ifndef OCSERIALIZE_H_

#include "ocval.h"

OC_BEGIN_NAMESPACE

// Routines for a very fast serialization routines. These are not
// compatible with Python Pickling: there are routines available that
// do that.  These routines also do not support cross-platform
// serialializations.  NOTE: There is no signature for supporting
// Proxies, but the "const Val&" signature recognizes Proxies.

// Note that there are two serialization routines: the ones below
// (where you have to call BytesToSerialize first) and the
// ones in m2ser.h,cc.  The routines in here are _preferred_
// because they are faster.  

// NOTES on OC_SERIALIZE_COMPAT (set from command-line or ocport.h) A
// note on compatibility: As of PicklingTools 1.2.0, we support OTab
// and Tup and int_n and int_un, but previous systems may not
// understand how to support those constructs, so we allow
// "compatibility" mode, which, when true, converts all OTabs->Tabs
// and Tup->Arr and int_n,int_un ->Str so that previous systems
// understand them.

// The number of bytes it will take to serialize the given.  Note that
// there are special signatures for Tabs and strings to avoid extra
// copies (so it won't get converted to a Val first!).
OC_INLINE size_t BytesToSerialize (const Val& v, 
				   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE size_t BytesToSerialize (const Tab& t, 
				   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE size_t BytesToSerialize (const OTab& t, 
				   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE size_t BytesToSerialize (const Tup& t, 
				   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE size_t BytesToSerialize (const int_n& i, 
				   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE size_t BytesToSerialize (const int_un& i, 
				   bool compatibility=OC_SERIALIZE_COMPAT);
//OC_INLINE size_t BytesToSerialize (const Str& s, 
//				   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE size_t BytesToSerialize (const Arr& a, 
				   bool compatibility=OC_SERIALIZE_COMPAT);
template <class T>
OC_INLINE size_t BytesToSerialize (const Array<T>& a);

// Serialize into a memory buffer: this assumes mem is big enough to
// hold the data (it can be computed with BytesToSerialize).  Note
// that we have extra signatures for Tabs and strings to avoid extra
// construction that may not be necessary. This returns one past where
// the next serialize would begin (so that the total bytes serialized
// can be computed from mem-Serialize(...))
OC_INLINE char* Serialize (const Val& v, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE char* Serialize (const Tab& t, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE char* Serialize (const OTab& t, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE char* Serialize (const Tup& t, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE char* Serialize (const int_n& t, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
OC_INLINE char* Serialize (const int_un& t, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
#if !defined(OC_USE_OC_STRING)
OC_INLINE char* Serialize (const Str& s, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
#endif
OC_INLINE char* Serialize (const Arr& s, char* mem, 
			   bool compatibility=OC_SERIALIZE_COMPAT);
template <class T>
OC_INLINE char* Serialize (const Array<T>& s, char* mem);

// Deserialize into the given memory.  It assumes there is enough
// memory (computed with BytesToSerialize) to deserialize from. This
// returns one byte beyond where it serialized (so mem-return value is
// how many bytes it serialized).  The into val has to be an "empty"
// Val or it throws a logic_error exception.
OC_INLINE char* Deserialize (Val& into, char* mem, 
			     bool compatibility=OC_SERIALIZE_COMPAT);


//#if defined(OC_USE_OC_STRING)
// Still have to be able handle OCStrings even if not using...
OC_INLINE char* Serialize (const OCString&, char *mem);
OC_INLINE size_t BytesToSerialize (const OCString&);
//#endif

OC_END_NAMESPACE

// The implementation: can be put into a .o if you don't want
// everything inlined.
#if !defined(OC_FACTOR_INTO_H_AND_CC)
# include "ocserialize.cc"
#endif


#define OCSERIALIZE_H_
#endif // OCSERIALIZE_H_
