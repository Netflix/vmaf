#ifndef M2CONVERTREP_H
#define M2CONVERTREP_H
//
//
// ///// Authors: Pat Gioannini

// The MachineRep_e utilities.

// /////////////////////////////////////////////////////// Include Files

#include "ocarray.h"
#include "ocval.h"

PTOOLS_BEGIN_NAMESPACE

// M2k things
typedef int_u8 Size;
typedef int_u8 Index;

// This is Normally in an include file in M2k:  3.23.1 probably the
// last version of M2k
#define MIDAS_VERSION "3.23.1"  

// This is how M2k does casting (keep original code as much as possible)
#  define          reinterpret_cast(TYPE, OBJ) reinterpret_cast<TYPE>(OBJ)
#  define          static_cast(TYPE, OBJ)      static_cast<TYPE>(OBJ)

// M2k portability thing. Usually empty
#  define  M2BOGUS_BREAK  

// From m2types.h

// ////////////////////////////////////////////////// The Numeric_e type

// The Numeric_e type describes all of the official Midas 2k numeric
// types for an "element".

// NOTE: From a user's perspective, the values of these enumerations
// (as with any enumeration) should have no significance other than
// being unique.  Any reliance on their integer values would be
// foolish.  However, the values have been carefully selected to allow
// efficient identification, conversion, etc. via bit manipulation in
// the routines of this module.  Specifically, methods rely on
// UNDEFINED being 0 and that all integer types have bit patterns in
// the first 8 bits that are less than the bit pattern for FLOAT, and
// conversely if they are >= FLOAT they are floating point types.
//
// TODO (cll 18-AUG-00): 
// M2Time and M2Duration do not fit gracefully into this enumeration.
// They will not, however, break anything.  Hence things like
// PromoteNumericType do not yet support M2Time and M2Duration.  This
// will be resolved simulaneously with MathOps to support both M2Time
// and M2Duration (along with the other numeric types).

// These constants are used to determine what type a number is below.

const int M2REALMASK_            = 0x10;
const int M2COMPLEXMASK_         = 0x20;
const int M2UNSIGNEDMASK_        = 0x40;

enum Numeric_e {

  M2_UNDEFINED = 0,

  M2_BYTE       = M2REALMASK_ | 0x02,
  M2_INTEGER,
  M2_LONG,
  M2_XLONG,
  M2_FLOAT,
  M2_DOUBLE,
  M2_DURATION,


  M2_CX_BYTE     = M2COMPLEXMASK_ | 0x2,
  M2_CX_INTEGER,
  M2_CX_LONG,
  M2_CX_XLONG,
  M2_CX_FLOAT,
  M2_CX_DOUBLE,


  M2_BIT         = M2REALMASK_ | M2UNSIGNEDMASK_ | 0x1,
  M2_UBYTE,
  M2_UINTEGER,
  M2_ULONG,
  M2_UXLONG,
  M2_TIME,


  M2_CX_BIT      = M2COMPLEXMASK_ | M2UNSIGNEDMASK_ | 0x1,
  M2_CX_UBYTE,
  M2_CX_UINTEGER,
  M2_CX_ULONG,
  M2_CX_UXLONG

};				// Numeric_e


// The global aliases defined below are for backwards compatibility
// and will probably have to remain forever.  These very generic names
// were formerly the Numeric_e enum names.  They are now #defines so
// that they can be #undef'd where necessary to prevent collisions
// with 3rd party software (Motif, for example.)  At the time of this
// writing, namespaces are still not working on all compilers to
// resolve this issue.

#define UNDEFINED	M2_UNDEFINED
#define BYTE		M2_BYTE
#define INTEGER		M2_INTEGER
#define LONG		M2_LONG
#define XLONG		M2_XLONG
#define FLOAT		M2_FLOAT
#define DOUBLE		M2_DOUBLE
#define CX_BYTE		M2_CX_BYTE
#define CX_INTEGER	M2_CX_INTEGER
#define CX_LONG		M2_CX_LONG
#define CX_XLONG	M2_CX_XLONG
#define CX_FLOAT	M2_CX_FLOAT
#define CX_DOUBLE	M2_CX_DOUBLE
#define BIT		M2_BIT
#define UBYTE		M2_UBYTE
#define UINTEGER	M2_UINTEGER
#define ULONG		M2_ULONG
#define UXLONG		M2_UXLONG
#define CX_BIT		M2_CX_BIT
#define CX_UBYTE	M2_CX_UBYTE
#define CX_UINTEGER	M2_CX_UINTEGER
#define CX_ULONG	M2_CX_ULONG
#define CX_UXLONG	M2_CX_UXLONG


// Returns whether values of the given type can contain only a real
// component (i.e. it does not allow for an imaginary component as do
// complex numeric types).  Whether the value is floating point versus
// integer, is answered by the methods isFloatingPoint() and
// isInteger() below.
//
// Caution: if the Numeric_e has the value UNDEFINED, !isReal() is not
// the same as isComplex().  UNDEFINED is neither real nor complex.
// Check for this value explicitly if necessary.

inline bool isReal (Numeric_e n)
{
  return (n & M2REALMASK_) == M2REALMASK_;
}

inline Numeric_e toReal (Numeric_e n)
{
  if (n == M2_UNDEFINED) {
    throw runtime_error("toReal()"+Stringize(n)+"real");
  }
  return (Numeric_e)((n & ~M2COMPLEXMASK_) | M2REALMASK_);
}


// Returns whether values of the given type can contain an imaginary
// component.
//
// Caution: if the Numeric_e has the value UNDEFINED, !isComplex() is
// not the same as isReal().  UNDEFINED is neither real nor complex.
// Check for this value explicitly if necessary.

inline bool isComplex (Numeric_e n)
{
  return (n & M2COMPLEXMASK_) == M2COMPLEXMASK_;
}


inline int ByteLength (Numeric_e format)
{
  switch (format) {
    case DOUBLE:
    case XLONG:
    case UXLONG:
    case M2_TIME:
    case M2_DURATION:
      return 8;
      M2BOGUS_BREAK;
    case FLOAT:
    case LONG:
    case ULONG:
      return 4;
      M2BOGUS_BREAK;
    case INTEGER:
    case UINTEGER:
      return 2;
      M2BOGUS_BREAK;
    case BYTE:
    case UBYTE:
      return 1;
      M2BOGUS_BREAK;
    case CX_DOUBLE:
    case CX_XLONG:
    case CX_UXLONG:
      return 16;
      M2BOGUS_BREAK;
    case CX_FLOAT:
    case CX_LONG:
    case CX_ULONG:
      return 8;
      M2BOGUS_BREAK;
    case CX_INTEGER:
    case CX_UINTEGER:
      return 4;
      M2BOGUS_BREAK;
    case CX_BYTE:
    case CX_UBYTE:
      return 2;
      M2BOGUS_BREAK;
    case UNDEFINED:
      return 0;
      M2BOGUS_BREAK;
    case BIT:
      return 1; // sizeof(Bit);
      M2BOGUS_BREAK;
    default:
      throw runtime_error("ByteLength()"+Stringize(format)+
			  "a known byte length");
      M2BOGUS_BREAK;
  }                     // switch
}

// From m2k  m2typeconvt.h
#  define SelectOn(typename)      const typename*
#  define Selector(typename)    ((const typename *)0)


// ///// Map compiler types to the enumerated type

//inline Numeric_e NumericTypeLookup (SelectOn(Bit))      { return M2_BIT; }
inline Numeric_e NumericTypeLookup (SelectOn(int_1))    { return M2_BYTE; }
inline Numeric_e NumericTypeLookup (SelectOn(int_2))    { return M2_INTEGER; }
inline Numeric_e NumericTypeLookup (SelectOn(int_4))    { return M2_LONG; }
inline Numeric_e NumericTypeLookup (SelectOn(int_8))    { return M2_XLONG; }
inline Numeric_e NumericTypeLookup (SelectOn(real_4))   { return M2_FLOAT; }
inline Numeric_e NumericTypeLookup (SelectOn(real_8))   { return M2_DOUBLE; }
inline Numeric_e NumericTypeLookup (SelectOn(int_u1))   { return M2_UBYTE; }
inline Numeric_e NumericTypeLookup (SelectOn(int_u2))   { return M2_UINTEGER; }
inline Numeric_e NumericTypeLookup (SelectOn(int_u4))   { return M2_ULONG; }
inline Numeric_e NumericTypeLookup (SelectOn(int_u8))   { return M2_UXLONG; }
//inline Numeric_e NumericTypeLookup (SelectOn(M2Time))   { return M2_TIME; }
//inline Numeric_e NumericTypeLookup (SelectOn(M2Duration))
//{ return M2_DURATION; }


#if !defined(OC_BYTES_IN_POINTER)
  // Someone is depending on system to find  
  // BYTES_IN_POINTER and LONG_INT_IS_64BIT

struct OC_SOME_UNUSED1 { };
struct OC_SOME_UNUSED2 { };
typedef Select2<sizeof(long)==8, OC_SOME_UNUSED1, long>::Result OC_LONG_NUMERIC_TYPE;
typedef Select2<sizeof(long)==8, OC_SOME_UNUSED2, unsigned long>::Result OC_ULONG_NUMERIC_TYPE;

// Add the missing conversions for native integral type "long"
inline Numeric_e NumericTypeLookup (SelectOn(OC_LONG_NUMERIC_TYPE))
        { return M2_LONG; }
inline Numeric_e NumericTypeLookup (SelectOn(OC_ULONG_NUMERIC_TYPE))
        { return M2_ULONG; }

#else 

  // Old style way, someone sets it by hand, make sure we still support
#   if ! defined(OC_LONG_INT_IS_64BIT)
// Add the missing conversions for native integral type "long"
inline Numeric_e NumericTypeLookup (SelectOn(long))
        { return M2_LONG; }
inline Numeric_e NumericTypeLookup (SelectOn(unsigned long))
        { return M2_ULONG; }
#   endif

#endif

//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_1>))
//        { return M2_CX_BYTE; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_2>))
//        { return M2_CX_INTEGER; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_4>))
//       { return M2_CX_LONG; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_8>))
//        { return M2_CX_XLONG; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<real_4>))
//        { return M2_CX_FLOAT; }
inline Numeric_e NumericTypeLookup (SelectOn(complex_8))
        { return M2_CX_FLOAT; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<real_8>))
//        { return M2_CX_DOUBLE; }
inline Numeric_e NumericTypeLookup (SelectOn(complex_16))
        { return M2_CX_DOUBLE; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_u1>))
//        { return M2_CX_UBYTE; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_u2>))
//        { return M2_CX_UINTEGER; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_u4>))
//        { return M2_CX_ULONG; }
//inline Numeric_e NumericTypeLookup (SelectOn(ComplexT<int_u8>))
//        { return M2_CX_UXLONG; }

// Catch non-numeric types

inline Numeric_e NumericTypeLookup (...)
        { return M2_UNDEFINED; }

// The following macro removes the need for the caller to provide the
// Selector macro when calling NumericTypeLookup

#define NumericType(typename) \
        (NumericTypeLookup(Selector(typename)))




// ////////////////////////////////////////////////// The MachineRep_e type

// The MachineRep_e type describes Midas 2k's supported machine
// representations.
//
// NOTE: MachineRep_VAX describes VMS's VAX floating point representations
//       AS USED BY X-MIDAS.  We assume D_FLOAT (as opposed to G_FLOAT)
//       double precision representation.

enum MachineRep_e {

  MachineRep_UNDEFINED = 0,

  MachineRep_BIG_ENDIAN = 0x10,    // bit used by IsBigEndian
  MachineRep_IEEE,
  MachineRep_NETWORK = MachineRep_IEEE,
  MachineRep_CRAY,

  MachineRep_LITTLE_ENDIAN = 0x20, // bit used by IsBigEndian
  MachineRep_EEEI,
  MachineRep_VAX

};				// MachineRep_e

// M2ENUM_STREAM_OPS_DECLS(MachineRep_e);



// ///// Convert between MachineRep_e enumerations and strings

MachineRep_e	DecodeMachineRep (const string& str);
string		EncodeMachineRep (MachineRep_e rep);

#define M2_EEEI_
//#define M2_IEEE_

inline MachineRep_e NativeMachineRep ()
{
#ifdef M2_IEEE_
  return MachineRep_IEEE;
#elif defined(M2_EEEI_)
  return MachineRep_EEEI;
#elif defined(M2_VAX_)
  return MachineRep_VAX;
#elif defined(M2_CRAY_)
  return MachineRep_CRAY;
#else
  return MachineRep_UNDEFINED;
#endif
}


inline bool IsBigEndian (MachineRep_e rep)
{
  return (rep & MachineRep_BIG_ENDIAN) != 0;
}


inline bool IsLittleEndian (MachineRep_e rep)
{
  return (rep & MachineRep_LITTLE_ENDIAN) != 0;
}




// ///// Functions for converting buffers between Machine Reps

// Converts the given input buffer of the specified input machine
// representation to a new output buffer of the given output machine
// representation.  The caller must allocate the output buffer.

void ConvertBufferRep (MachineRep_e in_rep, MachineRep_e out_rep,
		       const void* in_buf, void* out_buf, 
		       Numeric_e format, int_4 elements);


// The implementation for ConvertBufferRep above.

void ConvertBufferRepInPlace (MachineRep_e in_rep, MachineRep_e out_rep,
			      void* buf, Numeric_e format, int_4 elements);


template <class T>
inline T NativeMachineRep (MachineRep_e in_rep, T value)
{
  ConvertBufferRepInPlace(in_rep, NativeMachineRep(), &value,
			  NumericTypeLookup(Selector(T)), 1);
  return value;
}


template <class T>
inline T NativeToNetworkMachineRep (T value)
{
  ConvertBufferRepInPlace(NativeMachineRep(), MachineRep_NETWORK, &value,
			  NumericTypeLookup(Selector(T)), 1);
  return value;
}


template <class T>
inline T NetworkToNativeMachineRep (T value)
{
  ConvertBufferRepInPlace(MachineRep_NETWORK, NativeMachineRep(), &value,
			  NumericTypeLookup(Selector(T)), 1);
  return value;
}


PTOOLS_END_NAMESPACE

#endif


