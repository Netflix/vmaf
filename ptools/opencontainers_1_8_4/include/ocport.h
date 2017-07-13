#ifndef OCPORT_H_

// A define which indicates we are using open containers to compile
#define OC_USE_OC 1_8_4

// This file exists so you can make a few portability decisions.
// Here are the current portability decisions you need to decide
// to define
// #define OC_DBL_DIGITS (16 by default), OC_FLT_DIGITS (7 by default)
// #define OC_NEEDS_BOOL
// #define OC_NEW_STYLE_INCLUDES
// #define OC_NO_STD_CHAR
// #define OC_LONG_INT_IS_64BIT
// #define OC_BYTES_IN_POINTER || #define OC_BYTES_IN_POINTER
// #define OC_NON_INCLUSION_TEMPLATE_MODEL
// #define OC_NAMESPACE_EXPERT__NO_NAMESPACES
// #define OC_NAMESPACE_FORCE_NAMESPACE
// #define OC_SERIALIZE_COMPAT true
// #define OC_BIGINT_OUTCONVERT_AS true

// They are all turned off by default, and your code can include it
// directly and turn on the portability features you need with
// -D on the command line.
//
// For example, on Solaris (CC: WorkShop Compilers 4.2) we invoke:
//  CC -I/home/opencontainers/include -D OC_NEEDS_BOOL string_test.cc
// On DECUnix (Compaq C++ V6.5-033) we invoke
//  cxx -I /home/opencontainers/include -D OC_NEW_STYLE_INCLUDES array_test.cc

// Common
#include <stddef.h>      // For size_t
#include <stdlib.h>      // For malloc/free
#include <limits.h>    

// DECISION: 
//
// Older versions of PicklingTools DO NOT support OTab and Tup: thus,
// older systems won't work with PicklingTools >1.2.0 unless we force
// the conversion of OTab ->> Tab and Tup->>Arr when talking with
// older systems.  By setting this OC_SERIALIZE_COMPAT to true, we are
// changing the default of all serialization routines to convert OTabs
// to Tabs and Tups to Arrs: Note that this is just changes the
// default: Every serialize type can decide on a per instance basis whther
// to be in compatibility mode or not.
#if !defined(OC_SERIALIZE_COMPAT)
// This needs to be either true or false: true means BY DEFAULT
// (unless the user excplicitly changes the comapatibility mode on the
// serialization) all OTabs are converted to Tabs and Tups to Arrs
// during serialization. false means BY DEFAULT no conversion:
// serialize as-is.
#define OC_SERIALIZE_COMPAT true
#endif

// DECISION: 
//
// The older versions of the OC library (and PTOOLS) were in the
// "global" space, not protected by a C++ namespace.  We have added
// namespaces to OC as of version 1.6.7 (with PicklingTools 1.1.1),
// but in a backwards compatible way.  Old code should be completely
// backwards compatible: it should still compile and and link and run
// the same.  The namespaces have the extra benefit that all the link
// symbols are encapsulated in the the OC namespace so as not to
// conflict with similar symbols.
//
// The way this works: by default, namespaces are turned on in the new
// version, but with a default "using namespace oc;" here in ocport.h:
// Thus the names are all available at the global level as before, but
// link symbols are protected inside the OC namespace.  If you wish to
// force the user to deal with the OC namespace (so the user has to do
// "using"s manually), you can:
//
//    #define OC_FORCE_NAMESPACE
//
// You can completely also turn off the namespace declarations 
// to completely eliminate any namespace constructs (for severe backwards 
// compatibility constraints) using 
//
//    #define OC_NAMESPACE_EXPERT__NO_NAMESPACES
//
// but you are very very unlikely to need to do that.  The default
// (with the namespace constructs in-place and the "using namespace
// OC") should work with old code without any code changes (just a
// recompile/relink).  This is a mechanism for when that doesn't work.
#if defined(OC_NAMESPACE_EXPERT__NO_NAMESPACES)

// Don't use namespaces, default to old way
#  define OC_BEGIN_NAMESPACE 
#  define OC_END_NAMESPACE

#  define PTOOLS_BEGIN_NAMESPACE 
#  define PTOOLS_END_NAMESPACE 

#  define OC_NAMESPACED   

#else

// Embracing namespaces on!
#  define OC_BEGIN_NAMESPACE namespace OC {
#  define OC_END_NAMESPACE }; 

    // ... for now, keep all the pickling tools stuff in the OC namespace:
    // we could use a 'using', but this seems simpler to keep them 
    // co-mingled (since they are very intertwined).  If we need to,
    // we can always separate them out since we use a different namespace
    // for all the C++ PicklingTools work.
#  define PTOOLS_BEGIN_NAMESPACE namespace OC {
#  define PTOOLS_END_NAMESPACE }; 

    // When the ::swap is used in the old ways, we have to say where it comes
    // from
#  define OC_NAMESPACED OC

#endif


// Put "something" into OC namespace, otherwise using below may have
// problems: here is the version!
OC_BEGIN_NAMESPACE
const int OCMajorVersion=1;
const int OCMinorVersion=7;
const int OCBugVersion  =9;
OC_END_NAMESPACE


// EXPERT USERS: By default, we insert a "using namespace OC" into a
// .h file (this file:ocport.h), which is usually a no-no in C++
// circles: we only do it to preserve backwards compatibility with
// previous versions of the library.  We do, however, allow control
// over the 'using' via a special macro to turn "off" the using:
//
//   #define OC_FORCE_NAMESPACE   // force user to do 'using' himself  
//
// Really, you should only use this if you are trying to combine this
// library with another library (MITE, Midas 2k) in the same .cc file,
// but you also may just want to force the user to use namespaces and
// have them deal with them appropriately. In a perfect world, all the
// PTOOLS code would have been namespaced already and this wouldn't be
// an issue: a good chunk of this code, however, migrated from ARM C++
// code.  You can, for example, turn off the 'using' on a file-by-file 
// basis like:
//
//   #include "mite.h"      // Has conflicting symbols, like HashTableT
//   #define OC_FORCE_NAMESPACE
//   #include "ocport.h"
//   using OC::HashTableT;  // Forces user to manually use/qualify OC  
//
// In general, the default should work fine: you probably only need
// this if (a) you want to force your user to deal with manual
// namespaces or (b) link OC with MITE or M2k in the same process/file.
#if defined(OC_NAMESPACE_EXPERT__NO_NAMESPACES) || defined(OC_FORCE_NAMESPACE)
  // If no namespaces, nothing to open!

  // OC_FORCE_NAMESPACE is a forcing the user to use/deal with namespaces
  // and namespace qualifications.
#else
  using namespace OC; // By default, OC namespace open 
#endif


// DECISION:
// Most compilers support bools, but not all
#if defined(OC_NEEDS_BOOL)
   typedef int bool;
   enum { false=0, true=1 };
#endif



// DECISION:
// Do we support X-Midas?  There are a few typedefs that typeclash
// when using X-MIdas, so we have to turn them off (mostly in complex)
// #define OC_SUPPORT_XM

// DECISION: 
// What kind of includes do we use?  #include <iostream.h>
// or the newer style #include <stream>  (without the .h)

#if defined(OC_NEW_STYLE_INCLUDES)

#if defined(OSF1_)
#  define __USE_STD_IOSTREAM
#endif

#  include <iostream>    // For a few streams
#  include <new>         // For placement new
#  include <sstream>
#  include <fstream>

  OC_BEGIN_NAMESPACE

  // Note: previous users expect all of std, and for backwards
  // compatility we allow that, but we also give the users a chance
  // to "just" use what they need with a macro.
#    if defined(OC_ONLY_NEEDED_STL)
  using std::cout; using std::cerr; using std::endl;
  using std::ifstream; using std::ofstream;
  using std::istream; using std::ostream;
  using std::istringstream; 
  using std::ios; using std::ostringstream;
using std::swap; using std::max; 
# else
  using namespace std;
#endif

  typedef istringstream istrstream;  // So we can use istrstream in both worlds
  OC_END_NAMESPACE

#else  // defined(OC_NEW_STYLE_INCLUDES)

#  include <iostream.h>    // For a few streams
#  include <new.h>         // For placement new
#  include <strstream.h> 
#  include <fstream.h>

#endif // defined(OC_NEW_STYLE_INCLUDES)



// ... and right away, most of these constructs need to be namespaced



// ///////////////////////////////////////////// Type definitions

#if defined(OC_SUPPORT_XM)  
#  include "cdefs.h"           // X-Midas already defines these for us
#else

OC_BEGIN_NAMESPACE 

// On platforms that do not allow overloading on "char" and "signed char"
// we typedef to "char".
// DECISION:
#ifdef OC_NO_STD_CHAR
  typedef char          int_1;                  // int_1, int_u1
#else
  typedef signed char   int_1;
#endif
typedef unsigned char   int_u1;

typedef short           int_2;                  // int_2, int_u2
typedef unsigned short  int_u2;
typedef int             int_4;                  // int_4, int_u4
typedef unsigned int    int_u4;

// DECISION:
#if defined(OC_LONG_INT_IS_64BIT)
  // Force an int_8/int_u8 to be a long
  typedef long          int_8;
  typedef unsigned long int_u8;
#else

#  if defined(OC_BYTES_IN_POINTER)
    // If someone defined OC_BYTES_IN_POINTER, then they are relying on the
    // old behavior, where you have to specify the OC_LONG_INT_IS_64BIT
    // to get a long, otherwise you get a long long.
    typedef long long             int_8;
    typedef unsigned long long    int_u8;
#  else 
    // If you DON'T specify, you are using new techniques which
    // automatically figure these out for you.  Otherwise, let's
    // figure out if a long or a long long makes more sense using
    // automatic techniques.  Using a technique from "Modern C++
    // Design", have the system at compile time figure out how many
    // bytes are in a long long.  This trick only works if you have a
    // modern C++ compiler (which is most, frankly).
    template <bool flag, typename T, typename U>
    struct Select1
    { typedef T Result; };

    template<typename T, typename U>
    struct Select1<false, T, U>
    { typedef U Result; };

    // If you have to convert a pointer to an int, here's a typedef
    // that will make sure the "int" (AVLP) is big enough to hold the pointer:
    // either 4 bytes or 8 bytes
    typedef Select1<sizeof(long)==8, long, long long>::Result int_8;
    typedef Select1<sizeof(long)==8, unsigned long,  unsigned long long>::Result int_u8;
#  endif // OC_BYTES_IN_POINTER 

#endif // OC_LONG_INT_IS_64BIT

typedef float   real_4;                         // real_4
typedef double  real_8;                         // real_8

OC_END_NAMESPACE

#endif // OC_SUPPORT_XM


OC_BEGIN_NAMESPACE

// DECISION: Are size_t distinct from both int_u4 and int_u8?  On
// X-Midas 64-bit platforms, size_t and int_u8 are NOT typedefed the
// same, so code like 'size_t a = 1; Val v = a;' would cause compiler
// errors (as the compiler couldn't choose what to overload on
// ... strictly speaking, size_t and unsigned long long are different
// types even though they are exactly the same size and have exactly
// the same behavior).  Thus, we add a special type which "becomes"
// int_u8 when passed a size_t, but becomes something that doesn't get
// in the way if size_t and int_u8 are typedeffed the same.
struct OC_UNUSED_SIZE_T { 
  OC_UNUSED_SIZE_T (int_u8 init=0) : xx(init) { }
  operator int_u8 () { return xx; }
  int_u8 xx;
};// Strictly speaking, this can be empty
#if defined(OC_SIZE_T_ADD)    // Do it manually for older systems

#   if OC_SIZE_T_ADD==1
       typedef OC_UNUSED_SIZE_T ALLOW_SIZE_T;
#   else
       typedef size_t ALLOW_SIZE_T;
#   endif

#else                         // Automatically detect

    // Use specialization, if size_t is same as int_u4 and int_u8,
    // no problem, we make ALLOW_SIZE_T some unused type.  Otherwise,
    // if we can distinguisj between int_u4, int_u8 and size_t,
    // we make ALLOW_SIZE_T a size_t
    template <class T>
    struct FindSizeT {
      typedef size_t Result;
    };
    template <>
    struct FindSizeT<int_u4> {
      typedef OC_UNUSED_SIZE_T Result;
    };
    template <>
    struct FindSizeT<int_u8> {
      typedef OC_UNUSED_SIZE_T Result;
    };
    typedef FindSizeT<size_t>::Result ALLOW_SIZE_T;

#endif

// DECISION: Are long distinct from both int_4 and int_8?  
struct OC_UNUSED_LONG { 
  OC_UNUSED_LONG (long init=0) : xx(init) { }
  operator int_8 () { return xx; }
  int_8 xx;
};// Strictly speaking, this can be empty
#if defined(OC_LONG_ADD)    // Do it manually for older systems

#   if OC_LONG_ADD==1
       typedef OC_UNUSED_LONG ALLOW_LONG;
#   else
       typedef long ALLOW_LONG;
#   endif

#else                         // Automatically detect

    // Use specialization, if size_t is same as int_u4 and int_u8,
    // no problem, we make ALLOW_SIZE_T some unused type.  Otherwise,
    // if we can distinguisj between int_u4, int_u8 and size_t,
    // we make ALLOW_SIZE_T a size_t
    template <class T>
    struct FindLong {
      typedef long Result;
    };
    template <>
    struct FindLong<int_4> {
      typedef OC_UNUSED_LONG Result;
    };
    template <>
    struct FindLong<int_8> {
      typedef OC_UNUSED_LONG Result;
    };
    typedef FindLong<long>::Result ALLOW_LONG;

#endif

// DECISION: Are long distinct from both int_4 and int_8?  
struct OC_UNUSED_UNSIGNED_LONG { 
  OC_UNUSED_UNSIGNED_LONG (unsigned long init=0) : xx(init) { }
  operator int_u8 () { return xx; }
  int_u8 xx;
};// Strictly speaking, this can be empty
#if defined(OC_UNSIGNED_LONG_ADD)    // Do it manually for older systems

#   if OC_UNSIGNED_LONG_ADD==1
       typedef OC_UNUSED_UNSIGNED_LONG ALLOW_UNSIGNED_LONG;
#   else
       typedef unsigned long ALLOW_UNSIGNED_LONG;
#   endif

#else                         // Automatically detect

    // Use specialization, if size_t is same as int_u4 and int_u8,
    // no problem, we make ALLOW_SIZE_T some unused type.  Otherwise,
    // if we can distinguisj between int_u4, int_u8 and size_t,
    // we make ALLOW_SIZE_T a size_t
    template <class T>
    struct FindUnsignedLong {
      typedef unsigned long Result;
    };
    template <>
    struct FindUnsignedLong<int_u4> {
      typedef OC_UNUSED_UNSIGNED_LONG Result;
    };
    template <>
    struct FindUnsignedLong<int_u8> {
      typedef OC_UNUSED_UNSIGNED_LONG Result;
    };
    template <>
    struct FindUnsignedLong<ALLOW_SIZE_T> {
      typedef OC_UNUSED_UNSIGNED_LONG Result;
    };
    typedef FindUnsignedLong<unsigned long>::Result ALLOW_UNSIGNED_LONG;

#endif

// DECISION: Are long long distinct from both int_4 and int_8?  
struct OC_UNUSED_LONG_LONG { 
  OC_UNUSED_LONG_LONG (long long init=0) : xx(init) { }
  operator int_8 () { return xx; }
  int_8 xx;
};// Strictly speaking, this can be empty
#if defined(OC_LONG_LONG_ADD)    // Do it manually for older systems

#   if OC_LONG_LONG_ADD==1
       typedef OC_UNUSED_LONG_LONG ALLOW_LONG_LONG;
#   else
       typedef long long ALLOW_LONG_LONG;
#   endif

#else                         // Automatically detect

    template <class T>
    struct FindLongLong {
      typedef long long Result;
    };
    template <>
    struct FindLongLong<int_4> {
      typedef OC_UNUSED_LONG_LONG Result;
    };
    template <>
    struct FindLongLong<int_8> {
      typedef OC_UNUSED_LONG_LONG Result;
    };
    template <>
    struct FindLongLong<OC_UNUSED_LONG> {
      typedef OC_UNUSED_LONG_LONG Result;
    };
    typedef FindLongLong<long long>::Result ALLOW_LONG_LONG;

#endif

// DECISION: Are unsigned long long distinct from both int_4 and int_8?  
struct OC_UNUSED_UNSIGNED_LONG_LONG { 
  OC_UNUSED_UNSIGNED_LONG_LONG (unsigned long long init=0) : xx(init) { }
  operator int_u8 () { return xx; }
  int_u8 xx;
};// Strictly speaking, this can be empty
#if defined(OC_UNSIGNED_LONG_LONG_ADD)    // Do it manually for older systems

#   if OC_UNSIGNED_LONG_LONG_ADD==1
       typedef OC_UNUSED_UNSIGNED_LONG_LONG ALLOW_UNSIGNED_LONG_LONG;
#   else
       typedef unsigned long long ALLOW_UNSIGNED_LONG_LONG;
#   endif

#else                         // Automatically detect

    // Use specialization, if size_t is same as int_u4 and int_u8,
    // no problem, we make ALLOW_UNSIGNED_LONG_LONG some unused type.  
    // Otherwise,
    // if we can distinguisj between int_u4, int_u8 and size_t,
    // we make ALLOW_UNSIGNED_LONG_LONG a size_t
    template <class T>
    struct FindUnsignedLongLong {
      typedef unsigned long long Result;
    };
    template <>
    struct FindUnsignedLongLong<int_u4> {
      typedef OC_UNUSED_UNSIGNED_LONG_LONG Result;
    };
    template <>
    struct FindUnsignedLongLong<int_u8> {
      typedef OC_UNUSED_UNSIGNED_LONG_LONG Result;
    };
    template <>
    struct FindUnsignedLongLong<OC_UNUSED_SIZE_T> {
      typedef OC_UNUSED_UNSIGNED_LONG_LONG Result;
    };
    template <>
    struct FindUnsignedLongLong<OC_UNUSED_UNSIGNED_LONG> {
      typedef OC_UNUSED_UNSIGNED_LONG_LONG Result;
    };
    typedef FindUnsignedLongLong<unsigned long long>::Result ALLOW_UNSIGNED_LONG_LONG;

#endif


// DECISION:
// How big are your pointers?
// #define OC_BYTES_IN_POINTER 4
// #define OC_BYTES_IN_POINTER 8
// If you don't define, system will figure it out for you.

// Used for both AVLTreeT and AVLHashT classes
#if OC_BYTES_IN_POINTER==8
    typedef int_u8 AVLP;
    typedef int_8   int_ptr;
    typedef int_u8  int_uptr;
#   define AVL_EMPTY_MASK 0xfffffffffffffffe
#elif OC_BYTES_IN_POINTER==4
    typedef int_u4 AVLP;
    typedef int_4   int_ptr;
    typedef int_u4  int_uptr;
#   define AVL_EMPTY_MASK 0xfffffffe
#else
    // System figures out for you.

    // Using a technique from "Modern C++ Design", have the system at
    // compile time figure out how many bytes are in a pointer.  This
    // makes it so you don't have to set BYTES_IN_POINTER.  This trick
    // only works if you have a modern C++ compiler, so this #define is
    // still available for those who need to
    template <bool flag, typename T, typename U>
    struct Select2
    { typedef T Result; };

    template<typename T, typename U>
    struct Select2<false, T, U>
    { typedef U Result; };

    // If you have to convert a pointer to an int, here's a typedef
    // that will make sure the "int" (AVLP) is big enough to hold the pointer:
    // either 4 bytes or 8 bytes
    typedef Select2<sizeof(void*)==4, int_u4, int_u8>::Result AVLP;
    typedef Select2<sizeof(void*)==4, int_4,  int_8 >::Result int_ptr;
    typedef Select2<sizeof(void*)==4, int_u4, int_u8>::Result int_uptr;;
#   define AVL_EMPTY_MASK (~AVLP(1))

#endif

OC_END_NAMESPACE

// DECISION: Do we support BigUInt.operator int_u8()? or AS
#define OC_BIGINT_OUTCONVERT_AS true



// DECISION: Do we use STL or OC strings? Note we have to do this
// before the exceptions, because they need to know what a string is.
#if defined(OC_USE_OC_STRING)
# include "ocstring_impl.h"
OC_BEGIN_NAMESPACE
  typedef OCString string; 
OC_END_NAMESPACE
#else
  #include <string>
OC_BEGIN_NAMESPACE
  using std::string;
OC_END_NAMESPACE
#endif

// TODO: Should the exceptions be in the OC namespace?

// DECISION:  
// Do we use STL exceptions?
#if defined(OC_USE_OC_EXCEPTIONS)
# include "ocexception.h"
#else 
# include <stdexcept>

  OC_BEGIN_NAMESPACE

#if !defined(OSF1_)
  using std::exception;
#endif
  using std::logic_error;
  using std::length_error;
  using std::runtime_error;
  using std::out_of_range;
  
  OC_END_NAMESPACE

#endif 






OC_BEGIN_NAMESPACE

// DECISION:
// Is everything inlined, or do we factor out into .h and .cc files?
#if defined(OC_FACTOR_INTO_H_AND_CC)
# define OC_INLINE 
#else
# define OC_INLINE inline
#endif


// DECISION:
// How many digits of precision do we use when we print doubles 
// and floats?
#if !defined(OC_DBL_DIGITS)
# define OC_DBL_DIGITS 16
#endif

#if !defined(OC_FLT_DIGITS)
# define OC_FLT_DIGITS 7
#endif

// A very standard way to Stringize items
template <class T>
inline string GenericStringize (const T& v) 
{
#if defined(OC_NEW_STYLE_INCLUDES)
  ostringstream os;
  os << v;
#if defined(OC_USE_OC_STRING)
  std::string s = os.str();
  string ret_val(s.data(), s.length());
#else 
  string ret_val = os.str();
#endif
  return ret_val;
#else
  ostrstream os;
  os << v << ends; // operator<< will do precision correctly
  string ret_val = string(os.str());
  delete [] os.str(); // Clean up ostrstream's mess
  return ret_val;
#endif
}

// Use the generic way, UNLESS you want something very specfic!
template <class T>
inline string Stringize (const T& v) { return GenericStringize(v); }

// Optimization
template <>
inline string Stringize<string> (const string& s) 
{ return s; }

// Stringize of floats and doubles should look like Python more, with
// .0 added to distinguish it from an int.  This isn't super fast
// right now: it needs to be cleaned up a little.
template <>
inline string Stringize<real_4> (const real_4& orig)
{
#if defined(OC_NEW_STYLE_INCLUDES)
  ostringstream os;

  os.precision(OC_FLT_DIGITS);
  if (orig<0) {
    int_8 con=int_8(orig);
    real_4 convert_back = con;
    if (convert_back==orig) {
      os << con << +".0";
    } else {
      os << orig;
    }
  } else {
    int_u8 con=int_u8(orig);
    real_4 convert_back = con;
    if (convert_back==orig) {
      os << con << ".0";
    } else {
      os << orig;
    }
  }

#if defined(OC_USE_OC_STRING)
  std::string s = os.str();
  string ret_val(s.data(), s.length());
#else
  string ret_val = os.str();
#endif
  return ret_val;
#else
  ostrstream os;

  os.precision(OC_FLT_DIGITS);
  if (orig<0) {
    int_8 con=int_8(orig);
    real_4 convert_back = con;
    if (convert_back==orig) {
      os << con << +".0";
    } else {
      os << orig;
    }
  } else {
    int_u8 con=int_u8(orig);
    real_4 convert_back = con;
    if (convert_back==orig) {
      os << con << ".0";
    } else {
      os << orig;
    }
  }
  // os << v << ends; // operator<< will do precision correctly
  os << ends;
  string ret_val = string(os.str());
  delete [] os.str(); // Clean up ostrstream's mess
  return ret_val;
#endif
}

template <>
inline string Stringize<real_8> (const real_8& orig)
{
#if defined(OC_NEW_STYLE_INCLUDES)
  ostringstream os;

  os.precision(OC_DBL_DIGITS);
  if (orig<0) {
    int_8 con=int_8(orig);
    real_8 convert_back = con;
    if (convert_back==orig) {
      os << con << +".0";
    } else {
      os << orig;
    }
  } else {
    int_u8 con=int_u8(orig);
    real_8 convert_back = con;
    if (convert_back==orig) {
      os << con << ".0";
    } else {
      os << orig;
    }
  }
#if defined(OC_USE_OC_STRING)
  std::string s = os.str();
  string ret_val(s.data(), s.length());
#else
  string ret_val = os.str();
#endif

  return ret_val;
#else
  ostrstream os;

  os.precision(OC_DBL_DIGITS);
  if (orig<0) {
    int_8 con=int_8(orig);
    real_8 convert_back = con;
    if (convert_back==orig) {
      os << con << +".0";
    } else {
      return os << orig;
    }
  } else {
    int_u8 con=int_u8(orig);
    real_8 convert_back = con;
    if (convert_back==orig) {
      os << con << ".0";
    } else {
      os << orig;
    }
  }
  // os << v << ends; // operator<< will do precision correctly
  os << ends;
  string ret_val = string(os.str());
  delete [] os.str(); // Clean up ostrstream's mess
  return ret_val;
#endif
}

// Very optimized way to stringize an integer.  Note that this will
// output numbers for chars, when maybe you want the ASCII value.  The
// slowest part of this routine is converting the char buff to a
// string!  This works for any 1,2,4,8,16 byte integer, signed or
// unsigned.
template <class INT>
inline string StringizeUInt (INT n)
{
  const int LEN = sizeof(n)*4;
  char a[LEN]; // Enough space to stringize into based on size of int: don't waste space!
  int ii=LEN-1;
  do {
    a[ii--] = n%10 + '0';
    n/=10;
  } while (n);
  return string(&a[ii+1], LEN-ii-1);
}

// Note this is self-contained
template <class INT>
inline string StringizeInt (INT n)
{
  const int LEN = sizeof(n)*4;
  char a[LEN]; // Enough space to stringize into based on size of int: don't waste space!
  int ii=LEN-1;
  if (n<0) {
    INT z = INT(-1)<<((sizeof(INT)<<3)-1); // Largest negative int
    int correction = (z==n) ? 1 : 0; 
    n = -(n+correction); // safe to negate, as long as not largest
    a[ii--] = n%10 + '0' + correction; // only corrects last char ... and no, there can't be a nine at the end
    n/=10;
    while (n) {
      a[ii--] = n%10 + '0';
      n/=10;
    } 
    a[ii--] = '-'; // mark as negative
  } else {
    do {
      a[ii--] = n%10 + '0';
      n/=10;
    } while (n);
  }
  return string(&a[ii+1], LEN-ii-1);
}

// Why not int_1, int_u1?  Too frequently wnat the CHAR, not the number
template <>
inline string Stringize<int_2> (const int_2& n)   { return StringizeInt(n); }
template <>
inline string Stringize<int_4> (const int_4& n)   { return StringizeInt(n); }
template <>
inline string Stringize<int_8> (const int_8& n)   { return StringizeInt(n); }
template <>
inline string Stringize<int_u2> (const int_u2& n) { return StringizeUInt(n); }
template <>
inline string Stringize<int_u4> (const int_u4& n) { return StringizeUInt(n); }
template <>
inline string Stringize<int_u8> (const int_u8& n) { return StringizeUInt(n); }


// Convert a string to an int
template <class I>
inline I StringToInt (const char* data, int len)
{
  I result=0;  // RVO
  char c = ' ';
  char sign = '\0';
  int ii;
  // Skip white space
  for (ii=0; ii<len; ii++) {
    c = data[ii];
    if (isspace(c)) continue;
    else if (isdigit(c) || c=='-' || c=='+') break;
    else ii=len; // Done
  }
  // Only accept sign after white space
  if (c=='+' || c=='-') {
    ii++;
    sign=c;
  }
  for (; ii<len; ii++) {
    c = data[ii];
    if ( !isdigit(c) ) break; // Only keep going if digit
    result*= 10;
    result+= (c-'0');
  }
  if (sign=='-') {
    result = -(signed(result));
  }
  return result;
}


OC_END_NAMESPACE

#if defined(OC_NEW_STYLE_INCLUDES)
#  include <algorithm>

OC_BEGIN_NAMESPACE
  using std::swap;
OC_END_NAMESPACE

#else

OC_BEGIN_NAMESPACE

template <class T>
void swap (T& t1, T& t2)
{
  T temp = t1;
  t1 = t2;
  t2 = temp;
}
OC_END_NAMESPACE

#endif 




#define OCPORT_H_

#if defined(OC_USE_OC_STRING)
#include "ocstring.h" // pick up rest...
#endif




#endif // OCPORT_H_
