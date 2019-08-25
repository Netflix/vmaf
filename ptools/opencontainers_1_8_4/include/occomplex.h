#ifndef OCCOMPLEX_H_

#include "ocport.h"
#include "ocmove.h"

// A templatized class for most of the complex math.  The complex_8
// and complex_16 are simply typedefs for cx_t<real_4> and
// cx_t<real_8), respectively.

OC_BEGIN_NAMESPACE

template <class T> class cx_union_t;

template <class T>
struct cx_t {

  T re, im;
  explicit cx_t (const T& real=0, const T& imag=0):re(real),im(imag) { }

  // use default constructor, default op=, default destructor


  cx_t<T>& operator+= (const cx_t<T>& r) { re+=r.re; im+=r.im; return *this; }
  cx_t<T>& operator+= (const T& scalar)  { re+=scalar; return *this; }
  cx_t<T>& operator-= (const cx_t<T>& r) { re-=r.re; im-=r.im; return *this; }
  cx_t<T>& operator-= (const T& scalar)  { re-=scalar; return *this; }

  cx_t<T>& operator*= (const cx_t<T>& r) 
  { 
    T t1=re*r.re-im*r.im; 
    T t2=im=re*r.im+im*r.re; 
    re = t1; im = t2;
    return *this; 
  }
  cx_t<T>& operator*= (const T& scalr) { re*=scalr; im*=scalr; return *this; } 
  
  cx_t<T>& operator/= (const cx_t<T>& r) 
  { 
    T den=r.re*r.re+r.im*r.im; 
    T t1 = (re*r.re + im*r.im)/den;  
    T t2 = (im*r.re - re*r.im)/den;  
    re=t1; im=t2;
    return *this; 
  }
  cx_t<T>& operator/= (const T& scalr) { re/=scalr; im/=scalr; return *this; }

  // These are so other cx_ts can op= and construct easily
  template <class T2> cx_t (const cx_t<T2>& rhs) : re(rhs.re), im(rhs.im) { }
  template <class T2> cx_t (const T2& rhs) : re(T(rhs)), im(0) { }
  template <class T2> cx_t operator= (const cx_t<T2>& rhs) 
  { re=rhs.re; im=rhs.im; return *this; }
  template <class T2> cx_t operator= (const T2& rhs) 
  { re=rhs; im=0; return *this; }

}; // cx_t

template <class T> 
inline cx_t<T> operator+ (const cx_t<T>& l, const cx_t<T>& r)
{ cx_t<T> retval(l); return retval+=r; }
template <class T, class S> 
inline cx_t<T> operator+ (const S& scalar, const cx_t<T>& r)
{ cx_t<T> retval(scalar); return retval+=r; }
template <class T, class S> 
inline cx_t<T> operator+ (const cx_t<T>& l, const S& scalar)
{ cx_t<T> retval(l); return retval+=scalar; }

template <class T> 
inline cx_t<T> operator- (const cx_t<T>& l, const cx_t<T>& r)
{ cx_t<T> retval(l); return retval-=r; }
template <class T, class S> 
inline cx_t<T> operator- (const S& scalar, const cx_t<T>& r)
{ cx_t<T> retval(scalar); return retval-=r; }
template <class T, class S> 
inline cx_t<T> operator- (const cx_t<T>& l, const S& scalar)
{ cx_t<T> retval(l); return retval-=scalar; }

template <class T> 
inline cx_t<T> operator* (const cx_t<T>& l, const cx_t<T>& r)
{ cx_t<T> retval(l); return retval*=r; }
template <class T, class S> 
inline cx_t<T> operator* (const S& scalar, const cx_t<T>& r)
{ cx_t<T> retval(scalar); return retval*=r; }
template <class T, class S> 
inline cx_t<T> operator* (const cx_t<T>& l, const S& scalar)
{ cx_t<T> retval(l); return retval*=scalar; }

template <class T> 
inline cx_t<T> operator/ (const cx_t<T>& l, const cx_t<T>& r)
{ cx_t<T> retval(l); return retval/=r; }
template <class T, class S> 
inline cx_t<T> operator/ (const S& scalar, const cx_t<T>& r)
{ cx_t<T> retval(scalar); return retval/=r; }
template <class T, class S> 
inline cx_t<T> operator/ (const cx_t<T>& l, const S& scalar)
{ cx_t<T> retval(l); return retval/=scalar; }

template <class T>
inline bool operator== (const cx_t<T>& lhs, const cx_t<T>& rhs) 
{ return lhs.re==rhs.re && lhs.im==rhs.im; }

template <class T>
inline bool operator!= (const cx_t<T>& lhs, const cx_t<T>& rhs) 
{ return !(lhs==rhs); }



// ///// Specialize


#if !defined(OC_SUPPORT_XM)
typedef cx_t<real_4> complex_8;
typedef cx_t<real_8> complex_16;

template <class T>
inline ostream& operator<< (ostream& os, cx_t<T> a)
{ return os<<"("<<a.re<<((a.im<0)?"":"+")<<a.im<<"j)"; }

// No sign to look at 
#define CXTUNSIGNED(T) template <> inline ostream& operator<< <T>(ostream&os,cx_t<T>a) \
{ return os<<"("<<a.re<<"+"<<a.im<<"j)"; }

CXTUNSIGNED(int_u2)
CXTUNSIGNED(int_u4)
CXTUNSIGNED(int_u8)

// Signed needs to look at the signed and decide if negative or pos
#define CXTSIGNED(T) template <> inline ostream& operator<< <T>(ostream&os,cx_t<T>a) \
{ return os<<"("<<a.re<<((a.im<0)?"":"+")<<a.im<<"j)"; }

CXTSIGNED(int_2)
CXTSIGNED(int_4)
CXTSIGNED(int_8)

// int_u1 and int_1 print chars, so we have to force it up to int_2
template <> inline ostream& operator<< <int_1>(ostream& os, cx_t<int_1> a)
{ return os<<"("<<int_2(a.re)<<((a.im<0)?"":"+")<<int_2(a.im)<<"j)"; }

template <> inline ostream& operator<< <int_u1>(ostream& os, cx_t<int_u1> a)
{ return os<<"("<<int_u2(a.re)<<"+"<<int_u2(a.im)<<"j)"; }

// reals need precision set
template <> inline ostream& operator<< <real_4>(ostream& os, cx_t<real_4> a)
{ os.precision(OC_FLT_DIGITS); return os<<"("<<a.re<<((a.im<0)?"":"+")<<a.im<<"j)"; }

template <> inline ostream& operator<< <real_8>(ostream& os, cx_t<real_8> a)
{ os.precision(OC_DBL_DIGITS); return os<<"("<<a.re<<((a.im<0)?"":"+")<<a.im<<"j)"; }




// TODO:  Ya, this probably needs to be made more robust
template <class T>
inline istream& operator>> (istream& is, cx_t<T>& a)
{
  char c; is >> c; is >> a.re; // is >> c; 
  is >> a.im; is >> c; is>>c; return is;
}

#else // X-Midas support ... already has complex_8 and complex_16


OC_END_NAMESPACE
#include <complex_n.h>

OC_BEGIN_NAMESPACE

inline istream& operator>> (istream& is, complex_8& a)
{ char c; is >> c; is >> a.re; is >> c; is >> a.im; is >> c; is>>c; return is; }

inline istream& operator>> (istream& is, complex_16& a)
{ char c; is >> c; is >> a.re; is >> c; is >> a.im; is >> c; is>>c; return is; }

#define OCCOMPLEXISTREAM(T) inline istream& operator>> (istream& is, T& a) \
{ char c; is >> c; is >> a.re; is >> c; is >> a.im; is >> c; is>>c; return is; }

OCCOMPLEXISTREAM(cx_t<int_1>)
OCCOMPLEXISTREAM(cx_t<int_u1>)
OCCOMPLEXISTREAM(cx_t<int_2>)
OCCOMPLEXISTREAM(cx_t<int_u2>)
OCCOMPLEXISTREAM(cx_t<int_4>)
OCCOMPLEXISTREAM(cx_t<int_u4>)
OCCOMPLEXISTREAM(cx_t<int_8>)
OCCOMPLEXISTREAM(cx_t<int_u8>)

#endif

// For converting out to scalars, we tend to return the magnitude (squared).
inline real_4 mag2 (const complex_8& v)  { return (v.re*v.re + v.im*v.im); }
inline real_8 mag2 (const complex_16& v) { return (v.re*v.re + v.im*v.im); }

template <class T>
T mag2 (const cx_t<T>& v) { return T(v.re*v.re + v.im*v.im); }



// What should we should to compare complexes?  SHOULD be a compiler
// error, but because of all the type conversion interactions, this is
// harder to do.  So, we defer to the Val class and have it throw an
// exception if you try to compare complexes.
//inline bool operator< (const complex_8& v1, const complex_8& v2) 
//{ return mag2(v1)<mag2(v2); }
//inline bool operator<= (const complex_8& v1, const complex_8& v2) 
//{ return mag2(v1)<=mag2(v2); }
//inline bool operator> (const complex_8& v1, const complex_8& v2) 
//{ return mag2(v1)>mag2(v2); }
//inline bool operator>= (const complex_8& v1, const complex_8& v2) 
//{ return mag2(v1)>=mag2(v2); }
//
//inline bool operator< (const complex_16& v1, const complex_16& v2) 
//{ return mag2(v1)<mag2(v2); }
//inline bool operator<= (const complex_16& v1, const complex_16& v2) 
//{ return mag2(v1)<=mag2(v2); }
//inline bool operator> (const complex_16& v1, const complex_16& v2) 
//{ return mag2(v1)>mag2(v2); }
//inline bool operator>= (const complex_16& v1, const complex_16& v2) 
//{ return mag2(v1)>=mag2(v2); }

// Intel compiler workaround (10.0.12) for overly aggressive template
// instatiation for complex_8 and complex_16 comparisons.
#if defined(__ICC)
#  if __ICC==1010 && __INTEL_COMPILER_BUILD_DATE==20080112 // LAME! Can't find 10.0.12 anywhere in INTEL's defines, only if build date???

   inline real_4 mymag2 (const complex_8& v) { return v.re*v.re+v.im*v.im; }
   inline real_8 mymag2 (const complex_16& v) { return v.re*v.re+v.im*v.im; }

   template <class T>
   T mag2 (const cx_t<T>& v) { return T(v.re*v.re + v.im*v.im); }

   inline bool operator< (complex_16 a,complex_8 b) 
   { return mymag2(a)<mymag2(b); }
   inline bool operator< (complex_16 a,complex_16 b)
   { return mymag2(a)<mymag2(b); }
#  endif
#endif

// ///// cx_union_t: This class is bitwise compatible with cx_t, but
// exists only to fit into a Union and play at least somewhat nice
// with cx_t

template <class T>
struct cx_union_t {
  T re, im;
  operator cx_t<T> () const { cx_t<T> retval(re, im); return retval; }
}; // cx_union_t

template <class T> 
cx_t<T> cx_union_t_cx (const cx_union_t<T>& x) { return cx_t<T>(x.re, x.im); }

template <class T>
inline bool operator== (const cx_union_t<T>& lhs, const cx_union_t<T>& rhs) 
{ return lhs.re==rhs.re && lhs.im==rhs.im; }

template <class T>
inline bool operator!= (const cx_union_t<T>& lhs, const cx_union_t<T>& rhs) 
{ return !(lhs==rhs); }

template <class T>
T mag2 (const cx_union_t<T>& v) { return T(v.re*v.re + v.im*v.im); }

MOVEARRAYPOD(cx_t<int_1>);   // has to be right after defined
MOVEARRAYPOD(cx_t<int_u1>);  // has to be right after defined
MOVEARRAYPOD(cx_t<int_2>);   // has to be right after defined
MOVEARRAYPOD(cx_t<int_u2>);  // has to be right after defined
MOVEARRAYPOD(cx_t<int_4>);   // has to be right after defined
MOVEARRAYPOD(cx_t<int_u4>);  // has to be right after defined
MOVEARRAYPOD(cx_t<int_8>);   // has to be right after defined
MOVEARRAYPOD(cx_t<int_u8>);  // has to be right after defined
MOVEARRAYPOD(complex_8);  // has to be right after defined
MOVEARRAYPOD(complex_16); // has to be right after defined

OC_END_NAMESPACE

#define OCCOMPLEX_H_
#endif // OCCOMPLEX_H_
