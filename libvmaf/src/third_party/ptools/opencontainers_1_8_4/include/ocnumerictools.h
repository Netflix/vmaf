#ifndef OC_NUMERIC_TOOLS_H_

// A few helper functions/defines for help for dealing with Numeric
// (Python Numeric)
#include "ocport.h"

OC_BEGIN_NAMESPACE

// Convert from a Val tag to a Python Numeric Tab
inline char OCTagToNumeric (char tag, bool allow_cx_int=false)
{
  char c;
  switch (tag) {
  case 's': c='1'; break;
  case 'S': c='b'; break; 
  case 'i': c='s'; break; 
  case 'I': c='w'; break; 
  case 'l': c='i'; break; 
  case 'L': c='u'; break; 
  case 'x': c='l'; break; 
  case 'X': c='l'; break; 
  case 'b': c='b'; break; 
  case 'f': c='f'; break; 
  case 'd': c='d'; break; 
  case 'F': c='F'; break; 
  case 'D': c='D'; break; 
  default: 
    if (allow_cx_int) {
      c = tag; // propagate it through
      break;
    }
    throw runtime_error("No corresponding Numeric type for Val type");
  }
  return c;
}

// Convert from Numeric tag to OC Tag
inline char NumericTagToOC (char tag, bool allow_cx_int=false)
{
  char c;
  switch (tag) {
  case '1': c='s'; break;
  case 'b': c='S'; break; 
  case 's': c='i'; break; 
  case 'w': c='I'; break; 
  case 'i': c='l'; break; 
  case 'u': c='L'; break; 
  case 'l': c='x'; break; 
    //case 'l': c='X'; break; 
    //case 'b': c='b'; break; 
  case 'f': c='f'; break; 
  case 'd': c='d'; break; 
  case 'F': c='F'; break; 
  case 'D': c='D'; break; 
  default: 
    if (allow_cx_int) {
      c = tag; // just propagate tag through: it's not in the same space
      break;
    }
    throw runtime_error("No corresponding Numeric type for Val type");
  }
  return c;
}


// Convert low-level bytes to another type: this assumes both input
// and output have the right number of elements
template <class IT, class OT>
inline void BufferConvert (const IT* in, OT* out, int len)
{
  for (int ii=0; ii<len; ii++) {
    out[ii] = OT(in[ii]);
  }
} 

// Convert low-level bytes to another type: this assumes both input
// and output have the right number of elements:  (OT is the complex
// class, ST is base class: I.e., OT=cx_t<int_1>, ST=int_1
template <class IT, class OT, class ST>
inline void BufferConvertCX (const IT* in, OT* out, int len, ST*)
{
  for (int ii=0; ii<len; ii++) {
    out[ii] = OT(ST(in[ii].re), ST(in[ii].im));
  }
} 

// helper function to convert from POD array to a different type of POD array
#define OCCVTARRT(T) { result=Array<T>(len); Array<T>& c=result; c.expandTo(len); T* cdata=c.data(); BufferConvert(a,cdata,len); }
template <class T>
inline void ConvertArrayT_ (const T* a, const int len, char to_type,
			    Val& result, bool /*is_complex*/=false)
{
  switch (to_type) {
  case 's': OCCVTARRT(int_1); break;
  case 'S': OCCVTARRT(int_u1); break; 
  case 'i': OCCVTARRT(int_2 ); break; 
  case 'I': OCCVTARRT(int_u2); break; 
  case 'l': OCCVTARRT(int_4); break; 
  case 'L': OCCVTARRT(int_u4); break; 
    // Take care with these ... longs and int_8s not necessarily the same
  case 'x': OCCVTARRT(int_8); break; // TODo exception?
  case 'X': OCCVTARRT(int_u8); break; 
  case 'b': OCCVTARRT(bool); break; 
  case 'f': OCCVTARRT(real_4); break; 
  case 'd': OCCVTARRT(real_8); break; 

    // These complexes don't convert ...
    //case 'c': OCCVTARRT(cx_t<int_1>); break; 
    //case 'C': OCCVTARRT(cx_t<int_u1>); break; 
    //case 'e': OCCVTARRT(cx_t<int_2>); break; 
    //case 'E': OCCVTARRT(cx_t<int_u2>); break; 
    //case 'g': OCCVTARRT(cx_t<int_4>); break; 
    //case 'G': OCCVTARRT(cx_t<int_u4>); break; 
    //case 'h': OCCVTARRT(cx_t<int_8>); break; 
    //case 'H': OCCVTARRT(cx_t<int_u8>); break; 

  case 'F': OCCVTARRT(complex_8); break; 
  case 'D': OCCVTARRT(complex_16); break; 
  default: throw runtime_error("Array not a POD type");
  }
}

#define OCCVTARRTCX(T) { result=Array<cx_t<T> >(len); Array<cx_t<T> >& c=result; c.expandTo(len); cx_t<T>* cdata=c.data(); BufferConvertCX(a,cdata,len, (T*)0);}
#define OCCVTARRTCX2(CT,T) { result=Array<CT>(len); Array<CT>& c=result; c.expandTo(len); CT* cdata=c.data(); BufferConvertCX(a,cdata,len, (T*)0);}
template <class T>
inline void ConvertCXArrayT_ (const T* a, const int len, char to_type,
			      Val& result, bool /*is_complex*/=false)
{
  switch (to_type) {
  case 'c': OCCVTARRTCX(int_1);  break; 
  case 'C': OCCVTARRTCX(int_u1); break; 
  case 'e': OCCVTARRTCX(int_2);  break; 
  case 'E': OCCVTARRTCX(int_u2); break; 
  case 'g': OCCVTARRTCX(int_4);  break; 
  case 'G': OCCVTARRTCX(int_u4); break; 
  case 'h': OCCVTARRTCX(int_8);  break; 
  case 'H': OCCVTARRTCX(int_u8); break; 
  case 'F': OCCVTARRTCX2(complex_8, real_4); break;
  case 'D': OCCVTARRTCX2(complex_16, real_8); break;
  default: throw runtime_error("Cannot convert array of complex to anything but another complex: Use mag2? mag? re? im?"); break;
  }
}

// Convert a pre-existing array from one type to the given type: this
// completely installs the new converted array into a.
#define OCCONVERTARRAY(T, A, TAG) { Array<T>& aa=A; ConvertArrayT_<T>(aa.data(), aa.length(), TAG, result); }
#define OCCONVERTCXARRAY(T, A, TAG) { Array<T>& aa=A; ConvertCXArrayT_<T>(aa.data(), aa.length(), TAG, result); }
inline void ConvertArray (Val& a, char to_type)
{
  Val result;
  if (a.tag != 'n') throw runtime_error("ConvertArray: input not array");
  char this_tag = a.subtype;
  if (this_tag == to_type) return;  // Both same, no work

  switch (this_tag) {
  case 's': OCCONVERTARRAY(int_1,  a, to_type); break;
  case 'S': OCCONVERTARRAY(int_u1, a, to_type);; break; 
  case 'i': OCCONVERTARRAY(int_2,  a, to_type);; break; 
  case 'I': OCCONVERTARRAY(int_u2, a, to_type);; break; 
  case 'l': OCCONVERTARRAY(int_4,  a, to_type);; break; 
  case 'L': OCCONVERTARRAY(int_u4, a, to_type);; break; 
  case 'x': OCCONVERTARRAY(int_8,  a, to_type);; break; 
  case 'X': OCCONVERTARRAY(int_u8, a, to_type);; break; 
  case 'b': OCCONVERTARRAY(bool,   a, to_type);; break; 
  case 'f': OCCONVERTARRAY(real_4, a, to_type);; break; 
  case 'd': OCCONVERTARRAY(real_8, a, to_type);; break; 

  case 'c': OCCONVERTCXARRAY(cx_t<int_1>,  a, to_type);; break; 
  case 'C': OCCONVERTCXARRAY(cx_t<int_u1>, a, to_type);; break; 
  case 'e': OCCONVERTCXARRAY(cx_t<int_2>,  a, to_type);; break; 
  case 'E': OCCONVERTCXARRAY(cx_t<int_u2>, a, to_type);; break; 
  case 'g': OCCONVERTCXARRAY(cx_t<int_4>,  a, to_type);; break; 
  case 'G': OCCONVERTCXARRAY(cx_t<int_u4>, a, to_type);; break; 
  case 'h': OCCONVERTCXARRAY(cx_t<int_8>,  a, to_type);; break; 
  case 'H': OCCONVERTCXARRAY(cx_t<int_u8>, a, to_type);; break; 

  case 'F': OCCONVERTCXARRAY(complex_8, a, to_type); break; 
  case 'D': OCCONVERTCXARRAY(complex_16, a,to_type); break; 
  default: throw runtime_error("Array not a POD type");
  }
  a.swap(result); // Install new array
}

OC_END_NAMESPACE

#define OC_NUMERIC_TOOLS_H_
#endif //  OC_NUMERIC_TOOLS_H_
