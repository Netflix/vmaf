#ifndef OCVAL_H_

// Val is a recursive, heterogeneuous container class, without virtual
// function overhead.  Vals can contains ints and reals, as well as
// arrays (Arr or Array<int_type> or Array<real_type) and Tabs
// (AVLHashT tables).  For example:

// Val v = 123.567890123456;  // v contains a real_8 (tag='d' for double)
// v = 56;                    // v contains an int   (tag='l' for int)
// v = Tab();                 // now a table         (tag='t')
// v[10] = 100;               // insert key:10 with value 100 into table
// Val result = v[10];        // Get value out
// int_4 ii = result;         // Convert result into int 

#include "ocport.h"
OC_BEGIN_NAMESPACE 

// ///// Forwards
struct Val;
class Arr;
struct Tab;
struct OTab;
class Tup;
// To make sure we get the proper swaps defined: because inclusion
// order is so important to which swap gets selected by the compiler:
// we "forward declare" these swaps so the test cases get the right
// swap.
void swap (Arr& lhs, Arr& rhs) ;
void swap (Val& lhs, Val& rhs) ;
void swap (Tab& lhs, Tab& rhs) ;
void swap (OTab& lhs, OTab& rhs) ;
void swap (Tup& lhs, Tup& rhs); 


OC_END_NAMESPACE

#include "ochashtable.h" 
#include "ocavlhasht.h"
#include "ocordavlhasht.h"
#include "ocbigint.h" // both int_un and int_n
#include "ocproxy.h"  // Weird depends ...


#ifndef VALTAB
# define VALTAB 32 // sizeof(Tab)
#endif
#ifndef VALSTR
# define VALSTR 32 // sizeof(Str)
#endif
#ifndef VALARR 
# define VALARR 32 // sizeof(Arr)
#endif
#ifndef VALOTAB
# define VALOTAB 32 // sizeof(OTab)
#endif
#ifndef VALTUP
# define VALTUP 32 // sizeof(Tup)
#endif
#ifndef VALBIGINT
# define VALBIGINT 32 // sizeof(int_n) 
#endif
#ifndef VALBIGUINT
# define VALBIGUINT 32 // should be sizeof(int_un)
#endif

OC_BEGIN_NAMESPACE

// Correspondance between tags and types
static const char* ValTags [] = { 
  "s", "S", // int_1, int_u1 
  "i", "I", // int_2, int_u2
  "l", "L", // int_4, int_u4
  "x", "X", // int_8, int_u8 and essentially size_t
  "q", "Q", // int_n, int_un (BigInt, BigUInt)
  "b",      // bool
  "f",      // real_4
  "d",      // real_8
  "c", "C", // cx<int_1>, cx<int_u1>
  "e", "E", // cx<int_2>, cx<int_u2>
  "g", "G", // cx<int_4>, cx<int_u4>
  "h", "H", // cx<int_8>, cx<int_u8>
  "F",      // complex_8 (2 real_4)
  "D",      // complex_16 (2 real_8)
  "a",      // ascii Str, like strings
  "t",      // table
  "o",      // OTab, ordered (by insertion) table
  "u",      // Tup, like a Python tuple
  "n",      // Array of Vals or primitives
  "Z",      // None, the empty value
  0
}; // ValTags

inline void printTags_ () { for (int ii=0; ValTags[ii]!=0; ii++) cout << ValTags[ii] << " "; }

// How do you print ODicts?  By default, we prefer to do things like
// Python does, but we are hopeful Python will adopt a better syntax
// with either the [] or o{}
enum ODictPrint_e { ODICT_LIKE_PYTHON, ODICT_SMALL_O, ODICT_BRACKETS };


// Find the Tag for the given Type:  usually called like 
// char tag = TagFor((int_1*)0);
inline char TagFor (int_n*)  { return 'q'; }
inline char TagFor (int_un*) { return 'Q'; }
inline char TagFor (int_1*)  { return 's'; }
inline char TagFor (int_u1*) { return 'S'; }
inline char TagFor (int_2*)  { return 'i'; }
inline char TagFor (int_u2*) { return 'I'; }
inline char TagFor (int_4*)  { return 'l'; }
inline char TagFor (int_u4*) { return 'L'; }
inline char TagFor (int_8*)  { return 'x'; }
inline char TagFor (int_u8*) { return 'X'; }
inline char TagFor (ALLOW_SIZE_T*)  { return 'X'; }
inline char TagFor (ALLOW_LONG*)  { return 'x'; }
inline char TagFor (ALLOW_UNSIGNED_LONG*)  { return 'X'; }
inline char TagFor (ALLOW_LONG_LONG*)  { return 'x'; }
inline char TagFor (ALLOW_UNSIGNED_LONG_LONG*)  { return 'X'; }
inline char TagFor (bool*)   { return 'b'; }
inline char TagFor (real_4*) { return 'f'; }
inline char TagFor (real_8*) { return 'd'; }
inline char TagFor (cx_t<int_1>*)   { return 'c'; }
inline char TagFor (cx_t<int_u1>*)  { return 'C'; }
inline char TagFor (cx_t<int_2>*)   { return 'e'; }
inline char TagFor (cx_t<int_u2>*)  { return 'E'; }
inline char TagFor (cx_t<int_4>*)   { return 'g'; }
inline char TagFor (cx_t<int_u4>*)  { return 'G'; }
inline char TagFor (cx_t<int_8>*)   { return 'h'; }
inline char TagFor (cx_t<int_u8>*)  { return 'H'; }
inline char TagFor (complex_8*)  { return 'F'; }
inline char TagFor (complex_16*) { return 'D'; }
inline char TagFor (string*)   { return 'a'; }
#if !defined(OC_USE_OC_STRING)
inline char TagFor (OCString*) { return 'a'; }
#endif
inline char TagFor (Tab*) { return 't'; }
inline char TagFor (OTab*) { return 'o'; }
inline char TagFor (Tup*) { return 'u'; }
inline char TagFor (Arr*) { return 'n'; }
inline char TagFor (Array<int_1>*)  { return 'n'; }
inline char TagFor (Array<int_u1>*) { return 'n'; }
inline char TagFor (Array<int_2>*)  { return 'n'; }
inline char TagFor (Array<int_u2>*) { return 'n'; }
inline char TagFor (Array<int_4>*)  { return 'n'; }
inline char TagFor (Array<int_u4>*) { return 'n'; }
inline char TagFor (Array<int_8>*)  { return 'n'; }
inline char TagFor (Array<int_u8>*) { return 'n'; }
inline char TagFor (Array<cx_t<int_1> >  *) { return 'n'; }
inline char TagFor (Array<cx_t<int_u1> > *) { return 'n'; }
inline char TagFor (Array<cx_t<int_2> >  *) { return 'n'; }
inline char TagFor (Array<cx_t<int_u2> > *) { return 'n'; }
inline char TagFor (Array<cx_t<int_4> >  *) { return 'n'; }
inline char TagFor (Array<cx_t<int_u4> > *) { return 'n'; }
inline char TagFor (Array<cx_t<int_8> >  *) { return 'n'; }
inline char TagFor (Array<cx_t<int_u8> > *) { return 'n'; }
inline char TagFor (Val*) { return 'Z'; }

inline int ByteLength (char s) {
  int ret = 0;
  switch (s) {
  case 's': ret=sizeof(int_1); break;  
  case 'S': ret=sizeof(int_u1); break; 
  case 'i': ret=sizeof(int_2); break;  
  case 'I': ret=sizeof(int_u2); break; 
  case 'l': ret=sizeof(int_4); break;  
  case 'L': ret=sizeof(int_u4); break; 
  case 'x': ret=sizeof(int_8); break;  
  case 'X': ret=sizeof(int_u8); break; 
  case 'b': ret=sizeof(bool); break;   
  case 'f': ret=sizeof(real_4); break; 
  case 'd': ret=sizeof(real_8); break; 
  case 'c': ret=sizeof(cx_t<int_1>); break; 
  case 'C': ret=sizeof(cx_t<int_u1>); break; 
  case 'e': ret=sizeof(cx_t<int_2>); break; 
  case 'E': ret=sizeof(cx_t<int_u2>); break; 
  case 'g': ret=sizeof(cx_t<int_4>); break; 
  case 'G': ret=sizeof(cx_t<int_u4>); break; 
  case 'h': ret=sizeof(cx_t<int_8>); break; 
  case 'H': ret=sizeof(cx_t<int_u8>); break; 
  case 'F': ret=sizeof(complex_8); break;  
  case 'D': ret=sizeof(complex_16); break;
  default : throw logic_error("Only POD data for ByteLength");
  }
  return ret; 
}

// Verified these work with the new codes 'o' and 'u' and 'q' and 'Q'
// Note that 'q' and 'Q' are not counted as POD ints
#define OC_IS_REAL(v) ((((v.tag) ^ 0x60)& 0xfc)==0x04) //'fd'
static char is_int_type[16] = { 0,0,1,0,0,0,0,0,1,1,0,0,0,1,0,0 };
#define OC_TAG_SOME_INT(tag) is_int_type[(((tag)>>6)^(tag))&0x0f] // silxSILX
#define OC_IS_INT(v)  (OC_TAG_SOME_INT(v.tag)&((v.tag)>0x60)) // silx
#define OC_IS_UINT(v) (OC_TAG_SOME_INT(v.tag)&((v.tag)<0x60)) // SILX
inline char GetRidOfWarningOfUnused() { return is_int_type[0]; } 

// Let compiler figure out best way to do this
inline bool oc_is_numeric (char t) { switch (t) { case 'a': case 't': case 'o': case 'u': case 'n': return false; default: return true; } }
#define OC_IS_NUMERIC(v) (oc_is_numeric(v.tag))
inline bool oc_is_cx (char t) { switch (t) { case 'c': case 'e': case 'g': case 'h': case 'C': case 'D': case 'E': case 'F': case 'G': case 'H': return true; default: return false; } }
#define OC_IS_CX(v) (oc_is_cx(v.tag))


// A value container for heterogeneous types: Note that we can
// recursively contain other Vals.
struct Val {

    char tag; char subtype; bool isproxy; char pad[1];
    Allocator* a;

    inline Allocator* allocator () const { return a; }

    // Constructors for every explicit type
    Val (void*) { throw logic_error("Cannot handle proxies except for Arr, Tab, OTab and Array<T>"); }

    Val () : tag('Z'), isproxy(false), a(0) { }   // None value
    Val (int_1 v)  : tag('s'), isproxy(false), a(0) { u.s = v; } 
    Val (int_u1 v) : tag('S'), isproxy(false), a(0) { u.S = v; }
    Val (int_2 v)  : tag('i'), isproxy(false), a(0) { u.i = v; }
    Val (int_u2 v) : tag('I'), isproxy(false), a(0) { u.I = v; }
    Val (int_4 v)  : tag('l'), isproxy(false), a(0) { u.l = v; }
    Val (int_u4 v) : tag('L'), isproxy(false), a(0) { u.L = v; }
    Val (int_8 v)  : tag('x'), isproxy(false), a(0) { u.x = v; }
    Val (int_u8 v) : tag('X'), isproxy(false), a(0) { u.X = v; }
    Val (ALLOW_SIZE_T v) : tag('X'), isproxy(false), a(0) { u.X = v; }
    Val (ALLOW_LONG v) : tag('x'), isproxy(false), a(0) { u.X = v; }
    Val (ALLOW_UNSIGNED_LONG v) : tag('X'), isproxy(false), a(0) { u.X = v; }
    Val (ALLOW_LONG_LONG v) : tag('x'), isproxy(false), a(0) { u.X = v; }
    Val (ALLOW_UNSIGNED_LONG_LONG v) : tag('X'), isproxy(false), a(0) { u.X = v; }
    Val (bool v)   : tag('b'), isproxy(false), a(0) { u.b = v; }
    Val (real_4 v) : tag('f'), isproxy(false), a(0) { u.f = v; }
    Val (real_8 v) : tag('d'), isproxy(false), a(0) { u.d = v; }
    Val (cx_t<int_1> v):tag('c'), isproxy(false),a(0) { new (&u.c) cx_t<int_1>(v);}
    Val (cx_t<int_u1> v):tag('C'), isproxy(false),a(0) { new (&u.C) cx_t<int_u1>(v);}
    Val (cx_t<int_2> v):tag('e'), isproxy(false),a(0) { new (&u.e) cx_t<int_2>(v);}
    Val (cx_t<int_u2> v):tag('E'), isproxy(false),a(0) { new (&u.E) cx_t<int_u2>(v);}
    Val (cx_t<int_4> v):tag('g'), isproxy(false),a(0) { new (&u.g) cx_t<int_4>(v);}
    Val (cx_t<int_u4> v):tag('G'), isproxy(false),a(0) { new (&u.G) cx_t<int_u4>(v);}
    Val (cx_t<int_8> v):tag('h'), isproxy(false),a(0) { new (&u.h) cx_t<int_8>(v);}
    Val (cx_t<int_u8> v):tag('H'), isproxy(false),a(0) { new (&u.H) cx_t<int_u8>(v);}
    Val (complex_8 v):tag('F'), isproxy(false),a(0) { new (&u.F) complex_8(v);}
    Val (complex_16 v):tag('D'), isproxy(false),a(0) {new (&u.D)complex_16(v);}
    Val (const char* cc) : tag('a'), isproxy(false), a(0) { new (&u.a) OCString(cc); }
    Val (const Str& s)   : tag('a'), isproxy(false), a(0) { new (&u.a) OCString(s.data(), s.length()); }
#if !defined(OC_USE_OC_STRING)
    Val (const OCString& s)   : tag('a'), isproxy(false), a(0) { new (&u.a) OCString(s); }
#endif
    OC_INLINE Val (const Tab& v, Allocator* a=0);
    OC_INLINE Val (const OTab& ot, Allocator* a=0);
    OC_INLINE Val (const Tup& tu, Allocator* a=0);
    OC_INLINE Val (const int_n& tu, Allocator* a=0);
    OC_INLINE Val (const int_un& tu, Allocator* a=0);
    Val (const Proxy& p) : tag(p.tag), subtype(p.subtype), isproxy(true), 
                           a(0) { new (&u.P) Proxy(p); }
    Val (Tab* adopt_table) : tag('t'), isproxy(true), a(0) 
                                          { new (&u.P) Proxy(adopt_table); }
    Val (OTab* adopt_table) : tag('o'), isproxy(true), a(0) 
                                          { new (&u.P) Proxy(adopt_table); }
    Val (Tup* adopt_tuple) : tag('u'), isproxy(true), a(0) 
                                          { new (&u.P) Proxy(adopt_tuple); }

    // Construct an Array of any primitive type Val can hold,
    // Array<int_4>, Array<real_8>, Array<Val> (aka Arr).  No Arrays
    // of Arrays.
    template <class T> OC_INLINE Val (const Array<T>& a, Allocator* al=0);

    // Construct an Array of any primitive type Val can hold,
    // Array<int_4>, Array<real_8>, Array<Val> (aka Arr).  No Arrays
    // of Arrays.  This ADOPTS a new Array<T>
    template <class T>
    Val (Array<T>* adopt_arr, Allocator*aa=0):tag('n'), subtype(TagFor((T*)0)), 
                   isproxy(true), a(aa) { new (&u.P) Proxy(adopt_arr); }


    // Copy constructor:  Have to write because of string, table, array cases
    OC_INLINE Val (const Val& r, Allocator*a = 0);

    // Copy: Have to write for string, table, array classes
    OC_INLINE Val& operator= (const Val& v); 

    // Destructor: Have to write for string, table, array classes
    OC_INLINE ~Val ();

    // Useful for cascading lookups s["key"][2], etc.  Like python,
    // you can only use integer types for indices to arrays, but Tabs
    // can use others type as lookup keys. Str currently don't support
    // lookup (TODO: Should they? No. We return Val&)
    OC_INLINE Val& operator [] (int_1  v);
    OC_INLINE Val& operator [] (int_u1 v);
    OC_INLINE Val& operator [] (int_2  v);
    OC_INLINE Val& operator [] (int_u2 v);
    OC_INLINE Val& operator [] (int_4  v);
    OC_INLINE Val& operator [] (int_u4 v);
    OC_INLINE Val& operator [] (int_8  v);
    OC_INLINE Val& operator [] (int_u8 v);
    OC_INLINE Val& operator [] (int_n v);
    OC_INLINE Val& operator [] (int_un v);
    OC_INLINE Val& operator [] (ALLOW_SIZE_T v);
    OC_INLINE Val& operator [] (ALLOW_LONG v);
    OC_INLINE Val& operator [] (ALLOW_UNSIGNED_LONG v);
    OC_INLINE Val& operator [] (ALLOW_LONG_LONG v);
    OC_INLINE Val& operator [] (ALLOW_UNSIGNED_LONG_LONG v);
    OC_INLINE Val& operator [] (bool v);
    OC_INLINE Val& operator [] (real_4 v);
    OC_INLINE Val& operator [] (real_8 v);
    //inline Val& operator [] (complex_8 v);
    //inline Val& operator [] (complex_16 v);
    OC_INLINE Val& operator [] (const char* cc);
    OC_INLINE Val& operator [] (const Str& s);
    OC_INLINE Val& operator [] (const Val& v);
    //OC_INLINE Val& operator [] (Tab& t);      // doesn't make sense
    //OC_INLINE Val& operator [] (Array<T>& t); // doesn't make sense

    // Like operator[] above, but the () operation WILL throw an
    // out_of_range_error if the key is not in the table.
    OC_INLINE Val& operator () (int_1  v) const;
    OC_INLINE Val& operator () (int_u1 v) const;
    OC_INLINE Val& operator () (int_2  v) const;
    OC_INLINE Val& operator () (int_u2 v) const;
    OC_INLINE Val& operator () (int_4  v) const;
    OC_INLINE Val& operator () (int_u4 v) const;
    OC_INLINE Val& operator () (int_8  v) const;
    OC_INLINE Val& operator () (int_u8 v) const;
    OC_INLINE Val& operator () (int_n  v) const;
    OC_INLINE Val& operator () (int_un v) const;
    OC_INLINE Val& operator () (ALLOW_SIZE_T v) const;
    OC_INLINE Val& operator () (ALLOW_LONG v) const;
    OC_INLINE Val& operator () (ALLOW_UNSIGNED_LONG v) const;
    OC_INLINE Val& operator () (ALLOW_LONG_LONG v) const;
    OC_INLINE Val& operator () (ALLOW_UNSIGNED_LONG_LONG v) const;
    OC_INLINE Val& operator () (bool v) const;
    OC_INLINE Val& operator () (real_4 v) const;
    OC_INLINE Val& operator () (real_8 v) const;
    //inline Val& operator () (complex_8 v) const;
    //inline Val& operator () (complex_16 v) const;
    OC_INLINE Val& operator () (const char* cc) const;
    OC_INLINE Val& operator () (const Str& s) const;
    OC_INLINE Val& operator () (const Val& v) const;
    //OC_INLINE Val& operator () (Tab& t);      // doesn't make sense
    //OC_INLINE Val& operator () (Array<T>& t); // doesn't make sense

    // Out converters for every type: so you can ask for a value
    // easily: 
    // Val somevalue = ...;
    // int get_value_out = somevalue; // Get out as int if possible
    OC_INLINE operator int_1 ()  const;
    OC_INLINE operator int_u1 () const;
    OC_INLINE operator int_2 ()  const;
    OC_INLINE operator int_u2 () const;
    OC_INLINE operator int_4 ()  const;
    OC_INLINE operator int_u4 () const;
    OC_INLINE operator int_8 ()  const;
    OC_INLINE operator int_u8 () const;
    OC_INLINE operator ALLOW_SIZE_T () const;
    OC_INLINE operator ALLOW_LONG () const;
    OC_INLINE operator ALLOW_UNSIGNED_LONG () const;
    OC_INLINE operator ALLOW_LONG_LONG () const;
    OC_INLINE operator ALLOW_UNSIGNED_LONG_LONG () const;
    OC_INLINE operator bool () const;
    OC_INLINE operator real_4 () const;
    OC_INLINE operator real_8 () const;
    OC_INLINE operator cx_t<int_1> () const;
    OC_INLINE operator cx_t<int_u1> () const;
    OC_INLINE operator cx_t<int_2> () const;
    OC_INLINE operator cx_t<int_u2> () const;
    OC_INLINE operator cx_t<int_4> () const;
    OC_INLINE operator cx_t<int_u4> () const;
    OC_INLINE operator cx_t<int_8> () const;
    OC_INLINE operator cx_t<int_u8> () const;
    OC_INLINE operator complex_8 () const;
    OC_INLINE operator complex_16 () const;
    OC_INLINE operator Str ()    const;
    OC_INLINE operator Tab& () const; // Throw logic_error if not really table
    OC_INLINE operator Arr& () const; // Throw logic_error if not really Arr
    OC_INLINE operator Proxy& () const; //Throw logic_error if not really Proxy
    OC_INLINE operator OTab& () const; //Throw logic_error if not really OTab
    OC_INLINE operator Tup& () const; //Throw logic_error if not really Tup
  //OC_INLINE operator int_n& () const; //Throw logic_error if not really int_n
    OC_INLINE operator int_n () const; //Throw logic_error if not really int_n
  //OC_INLINE operator int_un& () const;//Throw logic_error if not really int_n
    OC_INLINE operator int_un () const;//Throw logic_error if not really int_n
  //
  //#if !defined(OC_USE_OC_STRING)
  //  OC_INLINE operator OCString& () const; // Throws logic_error if not really Proxy
  //#endif

#if defined(TEMPLATE_OUTCONVERSION_SUPPORTED)
    // Interestingly, this "feature" to outcast isn't supported by all
    // compilers: if that is the case, you may have to revert to OC1.3.4
    template <class T> operator Array<T>& () const 
    {
      if (isproxy) {
	Proxy* p = (Proxy*)&u.P;
	Array<T>& pa = *p; // Have proxy do conversion
	return pa;
      } else if ('n'!=tag! || TagFor((T*)0) != subtype) {
	Str mesg = "No conversion from:"+Str(*this)+" to array.";
	throw logic_error(mesg.c_str());
      } 
      Array<T>* n = (Array<T>*)&u.n;
      return *n;
    }
#else

// Same code as above ... squiched to fit in macro
#  define TEMPLATE_OUTCONV(T) operator Array<T>& () const { \
if (isproxy) { Proxy*p=(Proxy*)&u.P;Array<T>& pa=*p;return pa;}\
else if ('n'!=tag || TagFor((T*)0)!= subtype) { Str mesg = "No conversion from:"+Str(*this)+" to array.";throw logic_error(mesg.c_str()); } \
Array<T>*n=(Array<T>*)&u.n;return *n;}

    TEMPLATE_OUTCONV(int_1)
    TEMPLATE_OUTCONV(int_u1)
    TEMPLATE_OUTCONV(int_2)
    TEMPLATE_OUTCONV(int_u2)
    TEMPLATE_OUTCONV(int_4)
    TEMPLATE_OUTCONV(int_u4)
    TEMPLATE_OUTCONV(int_8)
    TEMPLATE_OUTCONV(int_u8)
    TEMPLATE_OUTCONV(ALLOW_SIZE_T)
    TEMPLATE_OUTCONV(ALLOW_LONG)
    TEMPLATE_OUTCONV(ALLOW_UNSIGNED_LONG)
    TEMPLATE_OUTCONV(ALLOW_LONG_LONG)
    TEMPLATE_OUTCONV(ALLOW_UNSIGNED_LONG_LONG)
    TEMPLATE_OUTCONV(bool)
    TEMPLATE_OUTCONV(real_4)
    TEMPLATE_OUTCONV(real_8)
    TEMPLATE_OUTCONV(cx_t<int_1>)
    TEMPLATE_OUTCONV(cx_t<int_u1>)
    TEMPLATE_OUTCONV(cx_t<int_2>)
    TEMPLATE_OUTCONV(cx_t<int_u2>)
    TEMPLATE_OUTCONV(cx_t<int_4>)
    TEMPLATE_OUTCONV(cx_t<int_u4>)
    TEMPLATE_OUTCONV(cx_t<int_8>)
    TEMPLATE_OUTCONV(cx_t<int_u8>)
    TEMPLATE_OUTCONV(complex_8)
    TEMPLATE_OUTCONV(complex_16)
    TEMPLATE_OUTCONV(Str)  
    TEMPLATE_OUTCONV(Tab)
    TEMPLATE_OUTCONV(OTab)    
    TEMPLATE_OUTCONV(Tup)
    TEMPLATE_OUTCONV(Val)

#endif // TEMPLATE_OUTCONVERSION_SUPPORTED

    union valunion {
	int_1  s;
	int_u1 S;
	int_2  i;
	int_u2 I;
	int_4  l;
	int_u4 L;
	int_8  x;
	int_u8 X;
        bool   b;
	real_4 f;
	real_8 d;       // in a union with a real_8, so no need to worry 
        cx_union_t<int_1>  c;
        cx_union_t<int_u1> C;
        cx_union_t<int_2>  e;
        cx_union_t<int_u2> E;
        cx_union_t<int_4>  g;
        cx_union_t<int_u4> G;
        cx_union_t<int_8>  h;
        cx_union_t<int_u8> H;
        cx_union_t<real_4> F;
        cx_union_t<real_8> D;
	char a[VALSTR]; // sizeof(Str)
	char t[VALTAB]; // sizeof(Tab) 
        char n[VALARR]; // sizeof(Array<T>): Note, the Array container always same size regardless of type
        char P[VALARR]; // sizeof(Proxy)
        char o[VALOTAB]; // sizeof(OTab)
        char u[VALTUP]; // sizeof(Tup)
        char q[VALBIGINT]; // sizeof(int_n)
        char Q[VALBIGUINT]; // sizeof(int_n)
      
      // NOTE: u.a is ALWAYS an OCString (because they coopy faster
      // and are friendlier on the heap. The Array<> of string is
      // ALWAYS and Array<Str>.  This is a subtle difference, because
      // on most systems Str will be string to support easier usage.
      // The OCString is an implementation string that helps use
      // preserve out heap resources.
    } u;

    // Swap in O(1) time.
    OC_INLINE void swap (Val& rhs);

    // Print this Val with nesting
    OC_INLINE void prettyPrint (ostream& os, int starting_indent=0, 
				int additive_indent=4) const;

    // Append to the end of the Tab or Arr.  If the append would overwrite
    // an entry that's already there, it will throw a runtime_error.
    OC_INLINE void append (const Val& v);
    OC_INLINE void appendStr (const Val& v);

    // Frequently want the length.  If this is some Array<T>, Arr or
    // Tab, (or Proxy for one of those), it returns the length.
    //  Note, this is is the number of SURFACE ENTRIES at the top
    // level, not the number of recursive entries.
    // Otherwise, throws an exception indicating this is NOT a composite
    // entity
    OC_INLINE size_t entries () const;
    inline size_t length () const { return entries(); }

    // Frequently want to be able to do a contains check (right before
    // accessing the key) from a Val.  Note that all of Arr/Tab/Array<POD>
    // support contains:  anything else throws a logic_error.
    OC_INLINE bool contains (const Val& k) const;
    
    // Frequently want to be able to do a removes from some table: 
    // If not a table, then throws a logic_error
    OC_INLINE bool remove (const Val& k);

    // Run this to make sure everything is compiled and working okay
    OC_INLINE void test ();

    // Generic function: If this is contained by value (and not a
    // Proxy), we want to turn it INTO a Proxy, with minimal cost.
    // This is actually a constant time operation, as does not involve
    // copying the data, just moving a few things around.  If this is
    // already a proxy, then it immediately returns.  If the Val
    // contains something that can't be Proxyized, then a logic_error
    // is thrown.  The parameters indicate what kind of proxy this is.
    OC_INLINE void Proxyize (bool adopt=true, bool locked=false);
  
    // This is the inverse operation of Proxyize: turn a Proxy inside into
    // something that is contained by Val.  TODO:  Worried about copyies?
  // OC_INLINE void UnProxyize ();


}; // Val

// When resizing Arr, make as fast as possible
MOVEARRAYPOD(Val)

// If two items are "essentially" the same POD type, they are
// converted to the bigger type and compared in the bigger type.  If
// they are "incomparable", then they are both converted to strings
// and compared that way: Two things are ALWAYS unequal (except for
// Tab and OTab comparisons) if they have different non-POD types.
OC_INLINE bool operator== (const Val& v1, const Val& v2);
OC_INLINE bool operator!= (const Val& v1, const Val& v2);
OC_INLINE bool operator< (const Val& v1, const Val& v2);
OC_INLINE bool operator<= (const Val&v1, const Val&v2);
OC_INLINE bool operator> (const Val& v1, const Val& v2);
OC_INLINE bool operator>= (const Val&v1, const Val&v2);

static const Val None;  // None constant, default constructed


// Specialization, since Val supports allocators
template <>
inline void PlaceCopyCons_ (void* location, const Val& copy, Allocator* a)
{ 
  new (location) Val(copy, a); 
}

// Specialization: This allows us to construct things using the
// allocator for Tabs Vals and Arrs
template <> 
inline AVLNode_<Val, Val>* 
AVLCreateNode<Val, Val> (void* memory_to_construct_into,
			 AVLNode_<Val,Val>*l, 
			 AVLNode_<Val,Val>*r, 
			 AVLNode_<Val,Val>*p, 
			 const Val& key, int_u4 keyhash, const Val& value,
			 Allocator* a)
{
  // Construct default Val first
  AVLNode_<Val, Val>* result = new (memory_to_construct_into) AVLNode_<Val, Val>(l,r,p,Val(),keyhash,Val());
  // No need to destruct Vals, there are just Nones
  
  // Now, construct using the allocator
  new (&result->key) Val(key, a);
  new (&result->value) Val(value, a);
  return result;
}


// ///////////////////////////////////////////// Class Tab

// Table of Vals: same interface as HashTable, AVLHash, etc.  An
// AVLHash is an extendible HashTable: performs just about as well as
// a HashTable, but can grow without having to rehash or reduce
// performance.
struct Tab : public AVLHashT<Val, Val, 8> {
  
  // Return the total number of elements: entries only returns the
  // total number of entries at the surface level.  In fact, most of
  // the routines only operate at the "surface" of the table.
  OC_INLINE size_t total_elements() const;
  size_t length () const { return entries_; }

  // Constructor allows easier way to build Tabs from literal strings
  // Tab t = " { 'a':1, 'b':2 }"
  OC_INLINE Tab (const char* s, Allocator* a=0); 
  OC_INLINE Tab (const Str& s, Allocator* a=0);

  // Empty Table
  OC_INLINE Tab (Allocator* a=0);
  
  OC_INLINE Tab (const Tab& t, Allocator* a = 0);

  // Have to allow nesting here:  Hide the one from AVLhashT
  OC_INLINE Tab& operator= (const Tab& rhs);

  // Merge right table into this table
  OC_INLINE Tab& operator+= (const Tab& rhs);

  // Append a new entry, based on the current number of elements.  If
  // we are going to overwrite an entry that is already there, we
  // throw a runtime_error
  OC_INLINE void append (const Val& v);     // int version
  OC_INLINE void appendStr (const Val& v);  // string version

  // Swap in O(1) time.
  inline void swap (Tab& rhs) { AVLHashT<Val,Val,8>*me=this;me->swap(rhs); }

  // Pretty print the table (with indentation)
  OC_INLINE void prettyPrint (ostream& os, int starting_indent=0, 
			      int additive_indent=4) const;
  OC_INLINE ostream& prettyPrintHelper_ (ostream& os, int indent, bool pretty=true, int additive_indent=4) const;

  // Give access to allocator
  inline Allocator* allocator () const { return allocator_; }

  // If the key is in the Tab, then return the corresponding
  // value: if it's not, then return the given default value.
  // This does NOT change the Tab underneath.  Note that this
  // returns a Val (not a Val reference) since it's possible
  // to return a default value, which may not anywhere in particular.
  OC_INLINE Val get (const Val& key, const Val& def=None) const;

  // Get a list of all keys or values in an Arr, 
  // also items returns an Arr of 2-tuples with key, value pairts
  OC_INLINE Arr keys () const;
  OC_INLINE Arr values () const;
  OC_INLINE Arr items () const;

 protected:
 OC_INLINE void appendHelper_ (const Val& key, const Val& value); 
}; // Tab

// Specialization, since Tab supports allocators
template <>
inline void PlaceCopyCons_<Tab> (void* location, const Tab& copy, Allocator* a)
{ 
  new (location) Tab(copy, a); 
}

// ///////////////////////////////////////////// Class TabIt and TabSit

// Iterator through Tabs.  Syntax:
// Tab some_table = ...
// for (TabIt it(some_table); it(); ) {
//   const Val& key = it.key();      
//   Val& val = it.value();    
//   cerr << key << " " << value << endl;
// }
// NOTE: Keys and values are give to you as references so no copying 
// is required unless you specifically request it.

// Iterators specfically for Tabs
struct TabIt : public AVLHashTIterator<Val, Val, 8> {
    TabIt (const Tab& t) : AVLHashTIterator<Val, Val, 8>(t) { }
    TabIt () : AVLHashTIterator<Val,Val,8>() { }
}; // TabIt

// Sorted iterator WILL give you the keys in sorted order
struct TabSit : public AVLHashTSortedIterator<Val, Val, 8> {
    TabSit (const Tab& t) : AVLHashTSortedIterator<Val, Val, 8>(t) { }
    TabSit () : AVLHashTSortedIterator<Val, Val, 8>() { }
}; // TabSit

// ///////////////////////////////////////////// Class OTab

// Similar to Tabs, but iterator preserves insertion order.  Keys that
// are updated DO NOT change insertion order, and two OTabs must have
// EXACT SAME INSERTION order to compare equally.
struct OTab : public OrdAVLHashT<Val, Val, 8> {
  
  // Return the total number of elements: entries only returns the
  // total number of entries at the surface level.  In fact, most of
  // the routines only operate at the "surface" of the table.
  OC_INLINE size_t total_elements() const;

  // Constructor allows easier way to build OTabs from literal strings
  // OTab t = " o{ 'a':1, 'b':2 }";
  // OTab tt= "  [ 'a':1, 'b':2 ]";
  // OTab tt= " OrderedDict([('a',1), ('b',2)])";
  OC_INLINE OTab (const char* s, Allocator* a=0); 
  OC_INLINE OTab (const Str& s, Allocator* a=0);

  // Empty OTable
  OC_INLINE OTab (Allocator* a=0);
  
  OC_INLINE OTab (const OTab& t, Allocator* a = 0);

  // Have to allow nesting here:  Hide the one from AVLhashT
  OC_INLINE OTab& operator= (const OTab& rhs);

  // Merge right table into this table
  OC_INLINE OTab& operator+= (const OTab& rhs);

  // Append a new entry, based on the current number of elements.  If
  // we are going to overwrite an entry that is already there, we
  // throw a runtime_error
  OC_INLINE void append (const Val& v);     // int version
  OC_INLINE void appendStr (const Val& v);  // string version

  // Swap in O(1) time.
  inline void swap (OTab& rhs) { OrdAVLHashT<Val,Val,8>*me=this;me->swap(rhs);}

  // Pretty print the table (with indentation)
  OC_INLINE void prettyPrint (ostream& os, int starting_indent=0, 
			      int additive_indent=4) const;
  OC_INLINE ostream& prettyPrintHelper_ (ostream& os, int indent, bool pretty=true, int additive_indent=4) const;

  // Give access to allocator
  inline Allocator* allocator () const { return allocator_; }

  // If the key is in the OTab, then return the corresponding
  // value: if it's not, then return the given default value.
  // This does NOT change the OTab underneath.  Note that this
  // returns a Val (not a Val reference) since it's possible
  // to return a default value, which may not anywhere in particular.
  OC_INLINE Val get (const Val& key, const Val& def=None) const;

  // Get a list of all keys or values in an Arr, 
  // also items returns an Arr of 2-tuples with key, value pairts
  OC_INLINE Arr keys () const;
  OC_INLINE Arr values () const;
  OC_INLINE Arr items () const;

 protected:
 OC_INLINE void appendHelper_ (const Val& key, const Val& value); 
}; // OTab

// Specialization, since OTab supports allocators
template <>
inline void PlaceCopyCons_<OTab> (void* location,const OTab& copy,Allocator* a)
{ 
  new (location) OTab(copy, a); 
}


// ///////////////////////////////////////////// Class OTabIt and OTabSit

// The main difference between OTab and Tab: the OTab preserves
// insertion order.  Thus:
//    OTab tt = " o{ 'a':1, 'b':2, 'c':3 } ";
// will preserve the order (a,b,c) of the insertion into the table
// so that the iterator below will see them in insertion order.

// Iterator through OTabs.  Syntax:
// OTab some_table = ...
// for (OTabIt it(some_table); it(); ) {
//   const Val& key = it.key();      
//   Val& val = it.value();    
//   cerr << key << " " << value << endl;
// }
// NOTE: Keys and values are give to you as references so no copying 
// is required unless you specifically request it.

// Iterators specfically for OTab: This will iterate in
// insertion order
struct OTabIt : public OrdAVLHashTIterator<Val, Val, 8> {
    OTabIt (const OTab& t) : OrdAVLHashTIterator<Val, Val, 8>(t) { }
    OTabIt () : OrdAVLHashTIterator<Val,Val,8>() { }
}; // OTabIt

// Sorted iterator WILL give you the keys in sorted order
struct OTabSit : public OrdAVLHashTSortedIterator<Val, Val, 8> {
    OTabSit (const OTab& t) : OrdAVLHashTSortedIterator<Val, Val, 8>(t) { }
    OTabSit () : OrdAVLHashTSortedIterator<Val, Val, 8>() { }
}; // OTabSit


// ///////////////////////////////////////////// Class Tup

// Like a python Tuple: contains a constant-length list of
// items.  A tuple is very similar to a list in a lot of ways,
// but the way they are constructed with makes them feel more like
// Python tuples.  For example:
//
//   Tup empty;     // empty tuple
//   Tup one(1.1);    // one entry: a real_8
//   Tup another(1);  // one entry: an int_4
//
//   Tup pair(1, 2.2); // two entries: an int_4 and real_8
//
//   Tup t(1, 2.2, "three", Tab("{ 'a':1 }"), Arr("[1,2,3]"));
//   int_u4 ii = t[0];
//   real_8 rr = t[1];
//   string ss = t[2];
//   Tab&   tt = t[3];
//   Arr&   aa = t[4];
//   cout << t << endl;           // (1, 2.2, "three", {'a':1}, [1,2,3])
//   cout << t.entries() << endl;  // 5
//   cerr << t[6];                // Exception!  Only 5 in this tuple
//
//
// The real purpose of tuples is to have a list-like construct that
// is closer to the language: with Arr and Tab, the construct literals
// are strings: with Tups, you can use C++ values directly.  
//
// Things in tuples are constructed as is: they don't have a single
// string literal syntax like Tabs and Arrs.
//
//   Tup t1("(1,2,3)"); 
//         // SURPRISING AT FIRST! t1 is tuple with 1 element: a string
//  
// You can get the same effect using Eval (AFTER the Tup is constructed)
//   Tup t2;
//   t2 = Eval("(1,2,3")); // t2 is a tuple with 3 elements:int * 3
//
// Unfortunately, this can be slightly confusing when:
//   Tup t3 = Eval("(1,2,3)") // t3 is ((1,2,3)) .. tuple within tuple
//      Why???  Because that is EQUIVALENT to 
//   Tup t3(Eval("(1,2,3)")); // Where t3[0] is set to first argument
// 
// A way around this on the command line would be:
//   Tup t4 = (Tup&)Eval("(1,2,3)"); // t4 is (1,2,3)
//      or simply:  
//   Val uu = Eval("(1,2,3)");
// 
// Using Tups with Vals is very similar to Tab& and Arr&: you just
// ask for tuple out.
//
//   Val v = Tup(1,2,3);
//   Tup& t = v;          // Get a reference to the Tup inside
//   Tup copy = v;        // Get my own copy
//
// You can also use them with Proxys: 
//   Proxy p = new Tup(1,2,3);
//   Val v = p;
//     or
//   Val vv = new Tup(3,4,5);  // easier syntax
// 
// If you absoultely have to get the underlying array out:
//   Tup t;
//   Array<Val>& a = t.impl();
//   a.append(1);  // Can extend tuple

class Tup {
 public:

  // Constructor and copy constructor
  OC_INLINE Tup (Allocator* alloc);
  OC_INLINE Tup (const Tup& t, Allocator* a);

  // Allow the user to specify a tuple of any size desired: the number
  // of entries in the constructor is how big the tuple is.
  // Currently, we can handle tuples up to 10.
  Tup ()
    : a_(0) { }
  Tup (const Val& a)
    : a_(1) { a_.append(a); }
  Tup (const Val& a, const Val& b)
    : a_(2) { a_.append(a); a_.append(b); }
  Tup (const Val& a, const Val& b, const Val& c)
    : a_(3) { a_.append(a); a_.append(b); a_.append(c); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d)
    : a_(4) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e)
    : a_(5) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e, const Val& f)
    : a_(6) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); a_.append(f); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e, const Val& f, const Val& g)
    : a_(7) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); a_.append(f); a_.append(g); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e, const Val& f, const Val& g, const Val& h)
    : a_(8) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); a_.append(f); a_.append(g); a_.append(h); }

  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e, const Val& f, const Val& g, const Val& h, const Val& i)
    : a_(9) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); a_.append(f); a_.append(g); a_.append(h); a_.append(i); }

  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e, const Val& f, const Val& g, const Val& h, const Val& i, const Val& j)
    : a_(10) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); a_.append(f); a_.append(g); a_.append(h); a_.append(i); a_.append(j); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e, const Val& f, const Val& g, const Val& h, const Val& i, const Val& j, const Val& k)
    : a_(11) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); a_.append(f); a_.append(g); a_.append(h); a_.append(i); a_.append(j); a_.append(k); }
  Tup (const Val& a, const Val& b, const Val& c, const Val& d, const Val& e, const Val& f, const Val& g, const Val& h, const Val& i, const Val& j, const Val& k, const Val& l)
    : a_(12) { a_.append(a); a_.append(b); a_.append(c); a_.append(d); a_.append(e); a_.append(f); a_.append(g); a_.append(h); a_.append(i); a_.append(j); a_.append(k); a_.append(l); }


  // Like the Array operations inspection operations
  inline size_t length () const { return a_.length(); }
  inline size_t entries () const { return a_.entries(); }
  inline bool   contains (const Val& v) const { return a_.contains(v); }

  // Like the Array operations: can't do non-const operations on Tup,
  // just like Python tuple
  void   append (const Val&)
  { throw runtime_error("Can't do appendStr on Tup, use Arr instead"); }
  void   appendStr (const Val&)
  { throw runtime_error("Can't do append on Tup, use Arr instead"); }
  bool   remove (const Val&)
  { throw runtime_error("Can't do remove on Tup, use Arr instead"); }

  // Helper functions

  // Swap in O(1) time.
  inline void swap (Tup& rhs) { a_.swap(rhs.a_);}

  // Pretty print the table (with indentation)
  OC_INLINE void prettyPrint (ostream& os, int starting_indent=0,
                              int additive_indent=4) const;
  OC_INLINE ostream& prettyPrintHelper_ (ostream& os, int indent, bool pretty=true, int additive_indent=4) const; 

  // Give access to allocator
  inline Allocator* allocator () const { return a_.allocator(); }

  // Like the array operarions
  const Val& operator[] (size_t i) const { return a_[i]; }
  const Val& operator() (size_t i) const { return a_[i]; }  // because () is unsafe for Arrays: longstanding issue
        Val& operator[] (size_t i) { return a_[i]; }
        Val& operator() (size_t i) { return a_[i]; } // because () is unsafe


  // Have to allow nesting here
  OC_INLINE Tup& operator= (const Tup& rhs);

  // Allow access to array
  const Array<Val>& impl () const { return a_; }
        Array<Val>& impl () { return a_; }

 protected:
  // Implementation: just an array!
  Array<Val> a_;

}; // Tup

// ///// Global Tup functions
inline bool operator== (const Tup& t1, const Tup& t2)
{ return t1.impl()==t2.impl(); }
inline bool operator!= (const Tup& t1, const Tup& t2)
{ return t1.impl()!=t2.impl(); }
inline bool operator< (const Tup& t1, const Tup& t2)
{ return t1.impl()<t2.impl(); }
inline bool operator<= (const Tup& t1, const Tup& t2)
{ return t1.impl()<=t2.impl(); }
inline bool operator> (const Tup& t1, const Tup& t2)
{ return t1.impl()>t2.impl(); }
inline bool operator>= (const Tup& t1, const Tup& t2)
{ return t1.impl()>=t2.impl(); }


// ///////////////////////////////////////////// Class Arr

// An array of Vals.  Very similar to Python List
class Arr : public Array<Val> {
 public:
  // Constructor for Arrs that allows easy literal construction
  // Arr a = "[1,'two',3.0, [1,2,3]]";
  OC_INLINE Arr (const char* s, Allocator* al=0);
  OC_INLINE Arr (const Str& s,  Allocator* al=0);
  
  Arr (const Array<Val>& a, Allocator* al =0) : Array<Val>(a, al) { } 

  // Constructor for expected length.
  OC_INLINE Arr (size_t expected_len, Allocator*a = 0);
  
  // Empty List
  OC_INLINE Arr (Allocator* a=0);
  
  // Swap in O(1) time
  inline void swap (Arr& rhs) { Array<Val>* me=this;me->swap(rhs); }

  // Have to allow nesting here:  Hide the one from Array
  OC_INLINE Arr& operator= (const Arr& rhs);

  // Print a more nested structure
  OC_INLINE void prettyPrint (ostream& os, int starting_indent = 0,
			      int additive_indent=4) const;
  OC_INLINE ostream& prettyPrintHelper_ (ostream& os, int indent, bool pretty=true, int additive_indent=4) const;
}; // Arr

// Specialization since Arr supports allocators
template <>
inline void PlaceCopyCons_ (void* location, const Arr& copy, Allocator* a)
{ 
  new (location) Arr(copy, a); 
}

// /// Strictly speaking, you should iterate through Arrs with
// /// for loops, like below: 
// for (size_t ii=0; ii<some_arr.length(); ii++) {
//     cerr << ii << " " << some_arr[ii] << endl;
// }
// ///// But sometimes it's nice to be able to use the plain iterator
// ///// syntax
// Arr some_arr = ...
// for (ArrIt it(some_arr); ii(); ) {  
//    const Val& key = it.key(); /// always 0..some_arr.length()-1
//    Val& val = it.value();     //  like some_arr[ii]
// }
// NOTE: Keys and values are give to you as references so no copying 
// is required unless you specifically request it.

// Iterators specifically for Arr
template <class T>
class ArrIt {
 public:
  // Arr to iterate over
  ArrIt (const Array<T>& a): ii_(~size_t(0)), len_(a.length()), data_((T*)a.data()) { }

  // Plain constructor ... nothing to iterate over
  ArrIt () : ii_(~size_t(0)), len_(0), data_(0) { }

  // Advance the iterator one position.  Returns true if the new
  // position is valid, false otherwise.
  bool next () { return ++ii_ < len_; }

  // Syntactic sugar for next
  bool operator++ () { return next(); }
  bool operator() () { return next(); }

  // Get a reference to the key and value
  const T& key () const { key_=ii_; return key_; }
  T& value () const { return data_[ii_]; }

  // Resets the iterator to the state it had immediately after
  // construction.
  void reset () { ii_=~size_t(0); }
    
  // Resets the iterator to iterate over collection c
  void reset (const Array<T>& c) { reset(); len_=c.length(); data_=c.data(); }
  
 protected:
  mutable T key_;   // If the object is an Arr, then we have to have a "key"
  size_t ii_;  // where in the Arr
  size_t len_;
  T* data_;  // Arr we refer to
  
}; // ArrIt

// An Array Sorted Iterator:  the values are sorted (but not in place):
// This allows you to iterate through an Array in sorted order without
// changing the array underneath

template <class T>  // Helper class so easy to call sort routines
struct Deref_ {
  Deref_ (T* dl) : data(dl) { }
  bool operator== (Deref_ rhs) { return (*data)==*(rhs.data); }
  bool operator< (Deref_ rhs) { return (*data)<*(rhs.data); }
  bool operator> (Deref_ rhs) { return (*data)>*(rhs.data); }
  // Use default destructor (no adoption), copy-cons, op=
  T* data;
}; // Deref_

template <class T>
class ArrSit {
 public:
  // Arr to iterate over
  ArrSit (const Array<T>& a) : ii_(~size_t(0)), arp_(a.length())
  {
    // Copy the pointers to data
    const size_t len = a.length();
    T* data = (T*)a.data();
    for (size_t ii=0; ii<len; ii++) {
      arp_.append(Deref_<T>(&data[ii]));
    }
    // Now sort the pointers, based on the values underneath
    Deref_<T>* arp_data = arp_.data();
    OCQuickSort(arp_data, 0, len);
  }

  // Plain constructor ... nothing to iterate over
  ArrSit () : key_(0), ii_(~size_t(0)), arp_() { }

  // Advance the iterator one position.  Returns true if the new
  // position is valid, false otherwise.
  bool next () { return ++ii_ < arp_.length(); }

  // Syntactic sugar for next
  bool operator++ () { return next(); }
  bool operator() () { return next(); }

  // Get a reference to the key and value
  const T& key () const { return key_=ii_; }
  T& value () const { return *(arp_(ii_).data); }

  // Resets the iterator to the state it had immediately after
  // construction.
  void reset () { ii_=~size_t(0); }
    
  // Resets the iterator to iterate over collection c
  void reset (const Array<T>& c) { reset(); arp_=&c; }
  
 protected:
  mutable T key_;   // If the object is an Arr, then we have to have a "key"
  size_t ii_;  // where in the Arr
  Array<Deref_<T> > arp_;  // Pointers into the Array to "sort"
  
}; // ArrSit

// You can iterate over the Tabs, OTabs and Arr the same way: it's slightly
// slower, but it uses the same syntax
// Val v = ...some Tab or some Arr...
// for (It it(v); it(); ) {
//    const Val& key = it.key(); 
//          Val& value = it.value();
//    cerr << key << " " << value << endl;
// }

// RANT: What C++ needs is a way to "infer" the type given.
// Generic iterator for Tabs and Arrs and OTabs
template <class AIT, class TIT, class OIT>
class GenericIT {

  enum Tab_e { OC_ARR, OC_TAB, OC_OTAB };

 public:
  // The type of object to iterate through: Can only iterate through
  // Tabs and Arrs (or Proxies to). This throws an exception if we try
  // to iterate through anything other than a Tab or Arr.
  GenericIT (const Tab& t)  :tit_(t),isTab_(OC_TAB)  { }
  GenericIT (const Arr& a)  :ait_(a),isTab_(OC_ARR) { }
  GenericIT (const Proxy& p) { Val v=p;cons_(v); }
  GenericIT (const OTab& ot):oit_(ot), isTab_(OC_OTAB) { }
  GenericIT (const Tup& ot) :ait_((Arr&)ot.impl()), isTab_(OC_ARR) { }
  GenericIT (const Val& v)   { cons_(v); }
  
  // Advance the iterator one position.  Returns true if the new
  // position is valid, false otherwise.
  bool next () 
  { 
    switch (isTab()) { 
    case OC_ARR:  return ait_.next(); 
    case OC_TAB:  return tit_.next(); 
    case OC_OTAB: return oit_.next(); 
    }
    throw runtime_error("Unknown type for Iterator");
  }

  // Syntactic sugar for next
  bool operator++ () { return next(); }
  bool operator() () { return next(); }

  // Get a reference to the key and value
  const Val& key () const 
  { 
    switch (isTab()) { 
    case OC_ARR:  return ait_.key(); 
    case OC_TAB:  return tit_.key(); 
    case OC_OTAB: return oit_.key(); 
    }
    throw runtime_error("Unknown type for Iterator");
  }

  Val& value () const 
  { 
    switch (isTab()) { 
    case OC_ARR:  return ait_.value(); break;
    case OC_TAB:  return tit_.value(); break;
    case OC_OTAB: return oit_.value(); break;
    }
    throw runtime_error("Unknown type for Iterator");
  }

  // Resets the iterator to the state it had immediately after
  // construction.
  void reset () 
  { 
    switch (isTab()) { 
    case OC_ARR:  ait_.reset(); break;
    case OC_TAB:  tit_.reset(); break;
    case OC_OTAB: oit_.reset(); break;
    }
    throw runtime_error("Unknown type for Iterator");
  } 
  // Resets the iterator to iterate over collection c
  void reset (const Tab& c)  { tit_ = TIT(c); isTab_ = OC_TAB; }
  void reset (const Arr& c)  { ait_ = AIT(c); isTab_ = OC_ARR; }
  void reset (const OTab& c) { oit_ = TIT(c); isTab_ = OC_OTAB; }
  void reset (const Proxy& c) { cons_(c); } 
  void reset (const Val& c)   { cons_(c); }

  // Are we iterating through a Tab or Arr?
  Tab_e isTab () const { return isTab_; }

 protected:
  AIT ait_;    // Default constructor does little
  TIT tit_;    // Default constructor does little
  OIT oit_;    // Default constructor does little
  Tab_e isTab_;

  // Centralized construction
  void cons_ (const Val& c) 
  {
    if (c.tag=='n' && c.subtype=='Z') {
      Arr& a = c;
      ait_ = AIT(a);
      isTab_ = OC_ARR;
    } else if (c.tag=='t') {
      Tab& t = c;
      tit_ = TIT(t);
      isTab_ = OC_TAB;
    } else if (c.tag=='o') {
      OTab& t = c;
      oit_ = OIT(t);
      isTab_ = OC_OTAB;
    } else if (c.tag=='u') {
      Tup& t = c;
      Array<Val>& a = t.impl();
      Arr& ar = (Arr&)a;
      ait_ = AIT(ar);
      isTab_ = OC_ARR;
    } else {
      throw runtime_error("Can only iterate through Tab, OTab, Tup & Arr:  "
			  "If you need to iterate through an Array<POD>, then "
			  " use a for-loop");
    }
  }

}; // GenericIT


// The plain It and Sit iterators share enough code to templatize them
typedef GenericIT<ArrIt<Val>, TabIt, OTabIt> It;
typedef GenericIT<ArrSit<Val>, TabSit, OTabSit> Sit;


// HashFunction on Vals: If a string, use Pearson's hash, otherwise
// just return the value as an int_u4.
OC_INLINE int_u4 HashFunction (const Val& v);

OC_INLINE void unknownType_ (const char* routine, char tag);


// In Python, dict and OrderedDict can compare and be equivalent if
// they have the same keys and values, even though two OrderedDicts
// compare differently if the order is different.
OC_INLINE bool operator== (const OTab& o, const Tab& t);
OC_INLINE bool operator== (const Tab& o, const OTab& t);
OC_INLINE bool operator!= (const OTab& o, const Tab& t); 
OC_INLINE bool operator!= (const Tab& o, const OTab& t); 

OC_INLINE bool operator== (const Tab& o, const OTab& t); 
OC_INLINE bool operator== (const Tab& o, const OTab& t); 

// Output: Note that default constructed ('Z') comes out as "None".
OC_INLINE ostream& operator<< (ostream& os, const Val& v);
OC_INLINE ostream& operator<< (ostream& os, const Tab& v);
OC_INLINE ostream& operator<< (ostream& os, const OTab& v);
OC_INLINE ostream& operator<< (ostream& os, const Tup& v);
//template <class T>
//OC_INLINE ostream& operator<< (ostream& os, const Array<T>& a);
template <> OC_INLINE ostream& operator<< <Val>(ostream& os, const Array<Val>& a);

template <class T> 
OC_INLINE ostream& PrintArray (ostream& os, const Array<T>& a);
template <>
OC_INLINE ostream& PrintArray <Val> (ostream& os, const Array<Val>& a);

inline void swap (Arr& lhs, Arr& rhs) { lhs.swap(rhs); }
inline void swap (Val& lhs, Val& rhs) { lhs.swap(rhs); }
inline void swap (Tab& lhs, Tab& rhs) { lhs.swap(rhs); }
inline void swap (OTab& lhs, OTab& rhs) { lhs.swap(rhs); }
inline void swap (Tup& lhs, Tup& rhs) { lhs.swap(rhs); }


inline bool is (const Val& l, const Val& r) 
{ 
  if (l.isproxy && r.isproxy) {
    Proxy& p1 = l;  Proxy& p2 = r; return is(p1,p2);
  } 
  return false;
}
inline bool IsProxy (const Val& v) { return v.isproxy; }

// **Very unsafe** method for converting an OTab to a Tab and
// vice-versa: only use this if you KNOW to_drop will immediately
// destruct after usage: to_drop is in an "unsafe" state for anything
// except destruction after this call.  However, this can be VERY FAST
// way to convert from one type to another!
template <class FROM, class TO>
OC_INLINE void SwapInto (FROM& to_drop, TO& to_gain)
{
  for (It ii(to_drop); ii(); ) {
    Val& key   = const_cast<Val&>(ii.key());
    Val& value = ii.value();
    to_gain.swapInto(key, value);
  }
}


OC_END_NAMESPACE


// The implementation: can be put into a .o if you don't want
// everything inlined.
#if !defined(OC_FACTOR_INTO_H_AND_CC)
# include "ocval.cc"
#endif 



#define OCVAL_H_
#endif // OCVAL_H_
