


#if defined(OC_FACTOR_INTO_H_AND_CC)

# include "ocval.h"
#include "ocnumpytools.h"
#include "ocitems.h"

OC_BEGIN_NAMESPACE

  // Explicit instantiations
#define OC_INST_PRINT(T) template ostream& PrintArray<T>(ostream&, const Array<T>& a);

OC_INST_PRINT(int_1)
OC_INST_PRINT(int_u1)
OC_INST_PRINT(int_2)
OC_INST_PRINT(int_u2)
OC_INST_PRINT(int_4)
OC_INST_PRINT(int_u4)
OC_INST_PRINT(int_8)
OC_INST_PRINT(int_u8)
OC_INST_PRINT(ALLOW_SIZE_T)
OC_INST_PRINT(ALLOW_LONG)
OC_INST_PRINT(ALLOW_UNSIGNED_LONG)
OC_INST_PRINT(ALLOW_LONG_LONG)
OC_INST_PRINT(ALLOW_UNSIGNED_LONG_LONG)
OC_INST_PRINT(real_4)
OC_INST_PRINT(real_8)
OC_INST_PRINT(cx_t<int_1>)
OC_INST_PRINT(cx_t<int_u1>)
OC_INST_PRINT(cx_t<int_2>)
OC_INST_PRINT(cx_t<int_u2>)
OC_INST_PRINT(cx_t<int_4>)
OC_INST_PRINT(cx_t<int_u4>)
OC_INST_PRINT(cx_t<int_8>)
OC_INST_PRINT(cx_t<int_u8>)
OC_INST_PRINT(complex_8)
OC_INST_PRINT(complex_16)
OC_INST_PRINT(bool)
OC_INST_PRINT(Str)
OC_INST_PRINT(Tab)
OC_INST_PRINT(OTab)
OC_INST_PRINT(Tup)

#define OC_INST_VAL(T) template Val::Val (const Array<T>&, Allocator*);

OC_INST_VAL(int_1)
OC_INST_VAL(int_u1)
OC_INST_VAL(int_2)
OC_INST_VAL(int_u2)
OC_INST_VAL(int_4)
OC_INST_VAL(int_u4)
OC_INST_VAL(int_8)
OC_INST_VAL(int_u8)
OC_INST_VAL(ALLOW_SIZE_T)
OC_INST_VAL(ALLOW_LONG)
OC_INST_VAL(ALLOW_UNSIGNED_LONG)
OC_INST_VAL(ALLOW_LONG_LONG)
OC_INST_VAL(ALLOW_UNSIGNED_LONG_LONG)
OC_INST_VAL(real_4)
OC_INST_VAL(real_8)
OC_INST_VAL(cx_t<int_1>)
OC_INST_VAL(cx_t<int_u1>)
OC_INST_VAL(cx_t<int_2>)
OC_INST_VAL(cx_t<int_u2>)
OC_INST_VAL(cx_t<int_4>)
OC_INST_VAL(cx_t<int_u4>)
OC_INST_VAL(cx_t<int_8>)
OC_INST_VAL(cx_t<int_u8>)
OC_INST_VAL(complex_8)
OC_INST_VAL(complex_16)
OC_INST_VAL(bool)
OC_INST_VAL(Str)
OC_INST_VAL(Tab)
OC_INST_VAL(OTab)
OC_INST_VAL(Tup)
OC_INST_VAL(Val)

OC_END_NAMESPACE
#else 

// We pick up everything anyways, but ocitems
#include "ocitems.h"

#endif



OC_BEGIN_NAMESPACE

///////////////////////////////////////////////////////////////////////////
// Most of the useful interface is in the .h file, below be
// implementation.  If you are worried about code bloat, #define
// OC_FACTOR_INTO_H_AND_CC and you can compile this .cc into an object
// file.  Otherwise, this .cc file is just "included" and handled as
// if it were inlined code.

// Discussion: if you look at the implementation of Val, you may notice
// that I explicitly break rules [9] from Exceptional C++.  For
// scalability and performance, two of the fundamental rules of Midas
// 2k were "stay out of the heap" and "use a lookaside cache for
// locality".  So, you'll notice that the Str and Tab are constructed
// in-place in the 32 bytes of the Val.  This keeps most small Strs
// from going to the Heap, it makes one less call to the heap for
// Tabs, and keeps all numeric types in 32 bytes of val so they don't
// go the heap at all!  Herb complains that this is a bad idea, but
// (a) it works on 3 different platforms (b) I am using it with a
// real_8 in a union so the alignments are not problematic (c) there
// are no polymorphic types, so shouldn't be object slicing (d) it's
// great for performance: stay in a lookaside cache for scalability,
// locality.  And honestly, the code isn't that bad.  I break the rule
// on purpose, for performance and scalabilty.

OC_INLINE int_u4 HashFunction (const Val& v) 
{
  int_u4 retval;
  if (v.tag=='a') {
    OCString* sp = (OCString*)v.u.a;
    retval = HashFunction(*sp);
  } else {
    retval = v;
  }
  return retval;
}

OC_INLINE bool operator== (const Val& v1, const Val& v2);
OC_INLINE bool operator!= (const Val& v1, const Val& v2) { return !(v1==v2); }
OC_INLINE bool operator< (const Val& v1, const Val& v2); 
OC_INLINE bool operator<= (const Val&v1, const Val&v2)
                                                 { return (v1==v2)||(v1<v2); }
OC_INLINE bool operator> (const Val& v1, const Val& v2)  { return !(v1<=v2); }
OC_INLINE bool operator>= (const Val&v1, const Val& v2)  { return !(v1<v2); }

// Because Python dicts and OrderedDict can compare the same, soto OTab/Tab
OC_INLINE bool operator== (const Tab&v1, const OTab& v2)  
{
  if (v1.entries() != v2.entries()) return false;
  for (It ii(v1); ii(); ) {
    const Val& key = ii.key();
    Val& value = ii.value();
    if (!(v2.contains(key) && v2(key)==value)) return false;  
  }
  return true;
} 
OC_INLINE bool operator== (const OTab&v1, const Tab& v2) { return v2==v1; }
OC_INLINE bool operator!= (const Tab&v1, const OTab& v2) { return !(v1==v2); }
OC_INLINE bool operator!= (const OTab&v1, const Tab& v2) { return !(v1==v2); }

// Output: Note that default constructed ('Z') comes out as "None".
OC_INLINE ostream& operator<< (ostream& os, const Val& v);
OC_INLINE ostream& operator<< (ostream& os, const Tab& v);
OC_INLINE ostream& operator<< (ostream& os, const OTab& v);
OC_INLINE ostream& operator<< (ostream& os, const Tup& v);

//template <class T>
//OC_INLINE ostream& operator<< (ostream& os, const Array<T>& a);
template <> OC_INLINE ostream& operator<< <Val>(ostream& os, const Array<Val>& a);

OC_INLINE size_t Tab::total_elements () const
{
  size_t surface_entries = 0;
  for (It ii(*this); ii(); ) {
    Val& val(ii.value());
    if (val.tag=='o') {
      OTab& t = val; // Tab converts out as a reference, so no copying
      // +1 if we want to include this entry in the count
      surface_entries += t.total_elements() + 1;  
    } else if (val.tag=='t') {
      Tab& t = val; // Tab converts out as a reference, so no copying
      // +1 if we want to include this entry in the count
      surface_entries += t.total_elements() + 1;  
    } else {
      surface_entries++;
    }
  }
  return surface_entries;
}

OC_INLINE Val Tab::get (const Val& key, const Val& def) const
{
  // Use the implementation where you only have to lookup the value
  // once.
  Found_e found_where; 
  AVLNode_<Val,Val>* node = this->lookup_(key, found_where);
  if (found_where==FOUND) {
    return node->value; // no need for two lookups
  } else {
    return def;
  }
}

// Get a list of all keys or values in an Arr, 
// also items returns an Arr of 2-tuples with key, value pairts
OC_INLINE Arr Tab::keys () const   { return OC_NAMESPACED::keys(*this); }
OC_INLINE Arr Tab::values () const { return OC_NAMESPACED::values(*this); }
OC_INLINE Arr Tab::items () const  { return OC_NAMESPACED::items(*this); }

OC_INLINE size_t OTab::total_elements () const
{
  size_t surface_entries = 0;
  for (It ii(*this); ii(); ) {
    Val& val(ii.value());
    if (val.tag=='o') {
      OTab& t = val; // Tab converts out as a reference, so no copying
      // +1 if we want to include this entry in the count
      surface_entries += t.total_elements() + 1;  
    } else if (val.tag=='t') {
      Tab& t = val; // Tab converts out as a reference, so no copying
      // +1 if we want to include this entry in the count
      surface_entries += t.total_elements() + 1;  
    } else {
      surface_entries++;
    }
  }
  return surface_entries;
}

OC_INLINE Val OTab::get (const Val& key, const Val& def) const
{
  // Use the implementation where you only have to lookup the value
  // once.
  Found_e found_where; 
  AVLOrdNode_<Val,Val>* node = this->lookup_(key, found_where);
  if (found_where==FOUND) {
    return node->value; // no need for two lookups
  } else {
    return def;
  }
}

// Get a list of all keys or values in an Arr, 
// also items returns an Arr of 2-tuples with key, value pairts
OC_INLINE Arr OTab::keys () const   { return OC_NAMESPACED::keys(*this); }
OC_INLINE Arr OTab::values () const { return OC_NAMESPACED::values(*this); }
OC_INLINE Arr OTab::items () const  { return OC_NAMESPACED::items(*this); }

OC_INLINE void Val::test ()
{
  // Make sure Tab can fit into union
  if (sizeof(Tab)>VALTAB) {
    Val sz = int_u4(sizeof(Tab));
    Str mesg="Tab is too big to fit in the union: VALTAB should be:"+Str(sz);
    cerr << mesg << endl;
    throw logic_error(mesg.c_str());
  }
  // Make Strs can fit in union
  if (sizeof(Str)>VALSTR) {
    Val sz = int_u4(sizeof(Str));
    Str mesg="Tab is too big to fit in the union: VALSTR should be:"+Str(sz);
    cerr << mesg << endl;
    throw logic_error(mesg.c_str());
  }
  // Make Arrs can fit in union
  if (sizeof(Array<Val>)>VALARR) {
    Val sz = int_u4(sizeof(Arr));
    Str mesg="Arr is too big to fit in the union: VALSTR should be:"+Str(sz);
    cerr << mesg << endl;
    throw logic_error(mesg.c_str());
  }

  // Make Proxies can fit in union
  if (sizeof(Proxy)>VALARR) {
    Val sz = int_u4(sizeof(Proxy));
    Str mesg="Proxy is too big to fit in the union: VALSTR should be:"+Str(sz);
    cerr << mesg << endl;
    throw logic_error(mesg.c_str());
  }

  // Make Proxies can fit in union
  if (sizeof(OTab)>VALOTAB) {
    Val sz = int_u4(sizeof(OTab));
    Str mesg="OTab is too big to fit in the union: VALSTR should be:"+Str(sz);
    cerr << mesg << endl;
    throw logic_error(mesg.c_str());
  }

  // Make Proxies can fit in union
  if (sizeof(int_n)>VALBIGINT) {
    Val sz = int_u4(sizeof(int_n));
    Str mesg="int_n is too big to fit in the union: VALSTR should be:"+Str(sz);
    cerr << mesg << endl;
    throw logic_error(mesg.c_str());
  }

  // Make Proxies can fit in union
  if (sizeof(int_un)>VALBIGUINT) {
    Val sz = int_u4(sizeof(int_un));
    Str mesg="int_un is too big to fit in the union: VALSTR should be:"+Str(sz);
    cerr << mesg << endl;
    throw logic_error(mesg.c_str());
  }
}


// For unknown types, throw same exception
OC_INLINE void unknownType_ (const char* routine, char tag)
{
  char name[2]; name[0] = tag; name[1] = '\0';
  Str mesg = "Unknown type:"+ Str(name)+ " in routine:" + Str(routine);
  throw logic_error(mesg.c_str());
}

      
OC_INLINE Val::Val (const Tab& t, Allocator* alloc) : 
  tag('t'), 
  isproxy(false),
  a(alloc)  
{ new (&u.t) Tab(t, alloc); }

OC_INLINE Val::Val (const OTab& t, Allocator* alloc) : 
  tag('o'), 
  isproxy(false),
  a(alloc)  
{ new (&u.o) OTab(t, alloc); }

OC_INLINE Val::Val (const Tup& t, Allocator* alloc) : 
  tag('u'), 
  isproxy(false),
  a(alloc)  
{ new (&u.u) Tup(t, alloc); }

OC_INLINE Val::Val (const int_n& t, Allocator* alloc) : 
  tag('q'), 
  isproxy(false),
  a(alloc)  
{ new (&u.q) int_n(t, alloc); }

OC_INLINE Val::Val (const int_un& t, Allocator* alloc) : 
  tag('Q'), 
  isproxy(false),
  a(alloc)  
{ new (&u.Q) int_un(t, alloc); }

template <class T> 
OC_INLINE Val::Val (const Array<T>& a, Allocator* alloc) : 
  tag('n'),
  isproxy(false),
  a(alloc)
{
  subtype = TagFor((T*)0);
  if (subtype=='n') {
    throw logic_error("Arrays of Arrays not currently supported");
  } 
  new (&u.n) Array<T>(a,alloc);
}

#define VALDESTR(T) { Array<T>*ap=(Array<T>*)&u.n;ap->~Array<T>(); }
OC_INLINE Val::~Val ()
{
  if (isproxy) { Proxy* pp = (Proxy*)&u.P; pp->~Proxy(); return; }
  switch(tag) { 
  case 'a': { OCString* sp = (OCString*)&u.a; sp->~OCString(); break; }
  case 't': { Tab* tp = (Tab*)&u.t; tp->~Tab(); break; }
  case 'o': { OTab* tp = (OTab*)&u.o; tp->~OTab(); break; }
  case 'u': { Tup* tp = (Tup*)&u.o; tp->~Tup(); break; }
  case 'q': { int_n* tp = (int_n*)&u.q; tp->~int_n(); break; }
  case 'Q': { int_un* tp = (int_un*)&u.Q; tp->~int_un(); break; }
  case 'n': { 
    switch(subtype) { 
    case 's': VALDESTR(int_1);  break;
    case 'S': VALDESTR(int_u1); break;
    case 'i': VALDESTR(int_2);  break;
    case 'I': VALDESTR(int_u2); break;
    case 'l': VALDESTR(int_4);  break;
    case 'L': VALDESTR(int_u4); break;
    case 'x': VALDESTR(int_8);  break;
    case 'X': VALDESTR(int_u8); break;
    case 'b': VALDESTR(bool);   break;
    case 'f': VALDESTR(real_4); break;
    case 'd': VALDESTR(real_8); break;
    case 'c': VALDESTR(cx_t<int_1>); break;
    case 'C': VALDESTR(cx_t<int_u1>); break;
    case 'e': VALDESTR(cx_t<int_2>); break;
    case 'E': VALDESTR(cx_t<int_u2>); break;
    case 'g': VALDESTR(cx_t<int_4>); break;
    case 'G': VALDESTR(cx_t<int_u4>); break;
    case 'h': VALDESTR(cx_t<int_8>); break;
    case 'H': VALDESTR(cx_t<int_u8>); break;
    case 'F': VALDESTR(complex_8); break;
    case 'D': VALDESTR(complex_16); break;
    case 'a': VALDESTR(Str);    break;
    case 't': VALDESTR(Tab);    break;
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': VALDESTR(Val);    break;
    default: unknownType_("destructor", subtype);
    }
  }
  }
}


#define VALCOPYCONS(T) {Array<T>*ap=(Array<T>*)&r.u.n;new(&u.n)Array<T>(*ap,alloc);}
OC_INLINE Val::Val (const Val& r, Allocator* alloc) : 
  tag(r.tag),
  subtype(r.subtype),
  isproxy(r.isproxy),
  a(alloc)
{ 
  if (isproxy) { Proxy* pp=(Proxy*)&r.u.P; new (&u.P) Proxy(*pp); return; }
  // Copy constructor: Have to write because of Str and table cases.
  // Although we could copy in less code, we want to be purify clean
  // (for now).
  switch(tag) { 
  case 's': u.s = r.u.s; break;
  case 'S': u.S = r.u.S; break;
  case 'i': u.i = r.u.i; break;
  case 'I': u.I = r.u.I; break;
  case 'l': u.l = r.u.l; break;
  case 'L': u.L = r.u.L; break;
  case 'x': u.x = r.u.x; break;
  case 'X': u.X = r.u.X; break;
  case 'b': u.b = r.u.b; break;
  case 'f': u.f = r.u.f; break;
  case 'd': u.d = r.u.d; break;
  case 'c': u.c = r.u.c; break;
  case 'C': u.C = r.u.C; break;
  case 'e': u.e = r.u.e; break;
  case 'E': u.E = r.u.E; break;
  case 'g': u.g = r.u.g; break;
  case 'G': u.G = r.u.G; break;
  case 'h': u.h = r.u.h; break;
  case 'H': u.H = r.u.H; break;
  case 'F': u.F = r.u.F; break;
  case 'D': u.D = r.u.D; break;
  case 'a': { OCString* sp=(OCString*)&r.u.a; new (&u.a) OCString(*sp,alloc); break; }
  case 't': { Tab* tp=(Tab*)&r.u.t; new (&u.t) Tab(*tp,alloc); break; }
  case 'o': { OTab*tp=(OTab*)&r.u.o;new (&u.o)OTab(*tp,alloc); break; }
  case 'u': { Tup* tp=(Tup*)&r.u.u; new (&u.u) Tup(*tp,alloc); break; }
  case 'q': { int_n* tp=(int_n*)&r.u.q; new (&u.q) int_n(*tp,alloc); break; }
  case 'Q': { int_un* tp=(int_un*)&r.u.Q;new (&u.Q) int_un(*tp,alloc); break; }
  case 'n': {
    switch(r.subtype) { 
    case 's': VALCOPYCONS(int_1);  break;
    case 'S': VALCOPYCONS(int_u1); break;
    case 'i': VALCOPYCONS(int_2);  break;
    case 'I': VALCOPYCONS(int_u2); break;
    case 'l': VALCOPYCONS(int_4);  break;
    case 'L': VALCOPYCONS(int_u4); break;
    case 'x': VALCOPYCONS(int_8);  break;
    case 'X': VALCOPYCONS(int_u8); break;
    case 'b': VALCOPYCONS(bool);   break;
    case 'f': VALCOPYCONS(real_4); break;
    case 'd': VALCOPYCONS(real_8); break;
    case 'c': VALCOPYCONS(cx_t<int_1>); break;
    case 'C': VALCOPYCONS(cx_t<int_u1>); break;
    case 'e': VALCOPYCONS(cx_t<int_2>); break;
    case 'E': VALCOPYCONS(cx_t<int_u2>); break;
    case 'g': VALCOPYCONS(cx_t<int_4>); break;
    case 'G': VALCOPYCONS(cx_t<int_u4>); break;
    case 'h': VALCOPYCONS(cx_t<int_8>); break;
    case 'H': VALCOPYCONS(cx_t<int_u8>); break;
    case 'F': VALCOPYCONS(complex_8); break;
    case 'D': VALCOPYCONS(complex_16); break;
    case 'a': VALCOPYCONS(Str);    break;
    case 't': VALCOPYCONS(Tab);    break;
    case 'o': VALCOPYCONS(OTab);    break;
    case 'u': VALCOPYCONS(Tup);    break;
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': VALCOPYCONS(Val);    break;
    default: unknownType_("copy constructor", subtype);
    }
  }
  case 'Z': break; // stay uninitialized
  default: unknownType_("copy constructor", tag);
  }
}


OC_INLINE Val& Val::operator= (const Val& rhs)
{ 
  if (&rhs!=this) { // Below will work if &rhs==this, but unnecessary work
    // To keep the lhs (this) alive, we hold onto it until
    // the assignment is done: we do this for nesting
    Allocator* alloc = a;
    Val temp;
    this->swap(temp);
    
    // temp holds the full rhs, we need to destruct the current Tab
    // (it's just empty() so we can construct into its memory.
    //    this->~Val();
    new (this) Val(rhs, alloc);
  }
  return *this; 
}

// For all the "convert out" operations: Since they are so similar,
// use macros to generate the code to avoid multiple maintenance.

// Code block to get the proper union member and cast it out correctly
#define VALSWITCHME(T,M) switch(tag) { case's':return T(u.s);case'S':return T(u.S);case'i':return T(u.i);case'I':return T(u.I);\
case'l':return T(u.l);case'L':return T(u.L);case'x':return T(u.x);case'X':return T(u.X);case 'b':return T(u.b); case'f':return T(u.f);case'd':return T(u.d); \
case'c':return T(M(u.c));case'e':return T(M(u.e));case'g':return T(M(u.g));case'h':return T(M(u.h));case'C':return T(M(u.C));case'E':return T(M(u.E));case'G':return T(M(u.G));case'H':return T(M(u.H));\
case'F':return T(M(u.F)); case'D':return T(M(u.D)); case'q':{int_n*qp=(int_n*)&u.q; return T(qp->as());} case'Q':{int_un*qp=(int_un*)&u.Q;return T(qp->as());}}

#define VALSWITCHME2(T,ST) switch(tag) { case's':return T(u.s);case'S':return T(u.S);case'i':return T(u.i);case'I':return T(u.I);	\
  case'l':return T(u.l);case'L':return T(u.L);case'x':return T(u.x);case'X':return T(u.X);case 'b':return T(u.b); case'f':return T(ST(u.f));case'd':return T(ST(u.d)); \
case'c':return T(u.c.re,u.c.im);case'e':return T(u.e.re,u.e.im);case'g':return T(u.g.re,u.g.im);case'h':return T(u.h.re,u.h.im);case'C':return T(u.C.re,u.C.im);case'E':return T(u.E.re,u.E.im);case'G':return T(u.G.re,u.G.im);case'H':return T(u.H.re,u.H.im);\
  case'F':return T(ST(u.F.re), ST(u.F.im)); case'D':return T(ST(u.D.re), ST(u.D.im)); case'q':{int_n*qp=(int_n*)&u.q;return T(real_8(int_8(qp->as())));} case'Q':{int_un*qp=(int_un*)&u.Q;return T(real_8(int_u8(qp->as())));} }


// Function tempplate for convert outs for the numeric types
#define CREATEVALOPBODY(T) \
{ if (tag=='a') { OCString* sp=(OCString*)&u.a;T tmp=0; istrstream is(sp->c_str());is.precision(OC_DBL_DIGITS);is>>tmp;return tmp;} \
else if (tag=='t'){Tab&t=*this;return T(t.entries());} else if (tag=='n'){Arr&ar=*this;return T(ar.length());} \
else if(tag=='o'){OTab&o=*this;return T(o.entries());} else if (tag=='u'){Tup&t=*this;return T(t.length());} return T(0); }

#define CREATEVALOP(T) \
OC_INLINE Val::operator T () const { VALSWITCHME(T,mag2); CREATEVALOPBODY(T) }

#define CREATEVALOP2(T, ST)						\
  OC_INLINE Val::operator T () const { VALSWITCHME2(T, ST); CREATEVALOPBODY(T) }

// CREATEVALOP(int_1)  op>> doesn't seem to work for this?
OC_INLINE Val::operator int_1 () const 
{ 
  VALSWITCHME(int_1,mag2); 
  CREATEVALOPBODY(int_u1);  // Same bits of int_u1 and int_1
}

// Force the call of "some size_t" to int_u8 code
OC_INLINE Val::operator ALLOW_SIZE_T () const { VALSWITCHME(int_u8,mag2); CREATEVALOPBODY(int_u8); }
OC_INLINE Val::operator ALLOW_LONG () const { VALSWITCHME(int_8,mag2); CREATEVALOPBODY(int_8); }
OC_INLINE Val::operator ALLOW_UNSIGNED_LONG () const { VALSWITCHME(int_u8,mag2); CREATEVALOPBODY(int_u8); }
OC_INLINE Val::operator ALLOW_LONG_LONG () const { VALSWITCHME(int_8,mag2); CREATEVALOPBODY(int_8); }
OC_INLINE Val::operator ALLOW_UNSIGNED_LONG_LONG () const { VALSWITCHME(int_u8,mag2); CREATEVALOPBODY(int_u8); }



CREATEVALOP(int_u1)
CREATEVALOP(int_2)
CREATEVALOP(int_u2)
CREATEVALOP(int_4)
CREATEVALOP(int_u4)
CREATEVALOP(int_8)
CREATEVALOP(int_u8)
//CREATEVALOP(ALLOW_SIZE_T)
CREATEVALOP(bool)
CREATEVALOP(real_4)
CREATEVALOP(real_8)


CREATEVALOP2(cx_t<int_1>,  int_1)
CREATEVALOP2(cx_t<int_u1>, int_u1)
CREATEVALOP2(cx_t<int_2>,  int_2)
CREATEVALOP2(cx_t<int_u2>, int_u2)
CREATEVALOP2(cx_t<int_4>,  int_4)
CREATEVALOP2(cx_t<int_u4>, int_u4)
CREATEVALOP2(cx_t<int_8>,  int_8)
CREATEVALOP2(cx_t<int_u8>, int_u8)

CREATEVALOP2(complex_8, real_4)
CREATEVALOP2(complex_16, real_8)

OC_INLINE Val::operator Str () const
{
  if (tag=='a') {
    const OCString* sp = reinterpret_cast<const OCString*>(&u.a);
#if defined(OC_USE_OC_STRING)
    return *sp;
#else
    return Str(sp->data(), sp->length());
#endif
  } else {
    return Stringize(*this);
  }
}


OC_INLINE Val::operator Tab& () const
{
  Tab *tp = (Tab*)&u.t;
  if (tag != 't') {
    NoConversion_(tag, *this, "table");
  }
  if (isproxy) {
    Proxy* pp = (Proxy*)&u.P;
    Tab& t = *pp;
    tp = &t;
  }
  return *tp;
}

OC_INLINE Val::operator OTab& () const
{
  OTab *tp = (OTab*)&u.t;
  if (tag != 'o') {
    NoConversion_(tag, *this, "ordered table");
  }
  if (isproxy) {
    Proxy* pp = (Proxy*)&u.P;
    OTab& t = *pp;
    tp = &t;
  }
  return *tp;
}

OC_INLINE Val::operator Tup& () const
{
  Tup *tp = (Tup*)&u.t;
  if (tag != 'u') {
    NoConversion_(tag, *this, "tuple");
  }
  // Does it make sense to have proxies for Tuples???
  if (isproxy) {
    Proxy* pp = (Proxy*)&u.P;
    Tup& t = *pp;
    tp = &t;
  }
  return *tp;
}

OC_INLINE Val::operator Arr& () const
{
  Arr* ap = (Arr*)&u.n;
  if (tag!='n' || subtype!='Z') {
    NoConversion_(tag, *this, "Arr");
  }
  if (isproxy) {
    Proxy* pp = (Proxy*)&u.P;
    Arr& a = *pp;
    ap = &a;
  }
  return *ap;
}

OC_INLINE Val::operator Proxy& () const
{
  Proxy* tp= (Proxy*)&u.P;
  if (!isproxy) {
    NoConversion_(tag, *this, "Proxy");
  }
  return *tp;
}

//OC_INLINE Val::operator int_n& () const
//{
//  int_n* ap = (int_n*)&u.q;
//  if (tag!='q') {
//    NoConversion_(tag, *this, "int_n&");
//  }
//  return *ap;
//}

OC_INLINE Val::operator int_n () const 
{
  if (tag=='q') {
    int_n* ap = (int_n*)&u.q;
    return *ap;
  } else if (tag=='Q') {
    const int_n* ap = (int_n*)&u.q; // int_un and int_n has same impl! VERY SKETHCY!
    return *ap;
  } else if (tag=='f' || tag=='d') {
    int_n result;
    real_8 r = *this;
    MakeBigIntFromReal(r, result);
    return result;
  } else if (tag=='a') {
    OCString* ocsp = (OCString*)&u.a;
    return StringToBigInt(ocsp->data(), ocsp->length());
  }
  int_8 value = *this; // let this convert, or not convert appropriately
  return int_n(value);

  // TODO: Handle proxies of bigints??
  //if (isproxy) {
  //  Proxy* pp = (Proxy*)&u.P;
  //  Arr& a = *pp;
  //  ap = &a;
  //}
  //return *ap;
}

//OC_INLINE Val::operator int_un& () const
//{
//  int_un* ap = (int_un*)&u.Q;
//  if (tag!='Q') {
//    NoConversion_(tag, *this, "int_un&");
//  }
//  return *ap;
//}

OC_INLINE Val::operator int_un () const
{
  if (tag=='Q') {
    int_un* ap = (int_un*)&u.Q;
    return *ap;
  } else if (tag == 'q') {
    int_un* ap = (int_un*)&u.q; // SKETCHY: int_un and int_n same impl
    return int_un(*ap); 
  } else if (tag=='f' || tag=='d') {
    int_un result;
    real_8 r = *this;
    MakeBigUIntFromReal(r, result);
    return result;
  } else if (tag=='a') {
    OCString* ocsp = (OCString*)&u.a;
    return StringToBigUInt(ocsp->data(), ocsp->length());
  }
  int_u8 value = *this; // let this convert, or not convert appropriately
  return int_un(value);

  // TODO: Handle proxies of bigints??
  //if (isproxy) {
  //  Proxy* pp = (Proxy*)&u.P;
  //  Arr& a = *pp;
  //  ap = &a;
  //}
  //return *ap;
}

// because of complex iteractions between strings, we don't want this
//
//#if !defined(OC_USE_OC_STRING)
//OC_INLINE Val::operator OCString& () const
//{
//  if (tag=='a') {
//    OCString* ocp= (OCString*)&u.a;
//    return *ocp;
//  } else {
//    NoConversion_(tag, *this, "OCString&");
//  }
//}
//#endif

// You might wonder why we instatiate so many copies rather than
// having operator[](const Val& v).  Because this way we can pay most
// cost at compile time ... otherwise, we'd have to do it at runtime
// and slow us down a little more (not much, but every little bit
// helps).

template <class LOOKUP> Val& ValLookUpBody_ (const Val& v, LOOKUP ii)
{ 
  // If there are proxies, the outcast will do the proper thing
  if (v.tag=='n') {
    if (v.subtype=='Z') { // For Array<Val>, we can return a reference
      Array<Val>& a = v;
      return a[ii];
    }
    // ... but for Arrays of other types, can't return a reference:
    // Force someone to get a contig array, thus force an exception  
    throw logic_error("Only Array<Val> can subscript, Array<T> cannot");
  } else if (v.tag=='o') {
    OTab& ot=v;
    return ot[ii];
  } else if (v.tag=='u') {
    Tup& t=v;
    return t[ii];
  } else { // Note that this will throw an exception if v is NOT a Tab
    Tab& t = v;
    return t[ii];
  }
}

#if defined(OC_FACTOR_INTO_H_AND_CC)
// Explicit instantiation
#define OC_INST_VALLOOKUP(T) template Val& ValLookUpBody_<T>(const Val&, T);
OC_INST_VALLOOKUP(int_1) 
OC_INST_VALLOOKUP(int_u1) 
OC_INST_VALLOOKUP(int_2) 
OC_INST_VALLOOKUP(int_u2) 
OC_INST_VALLOOKUP(int_4) 
OC_INST_VALLOOKUP(int_u4) 
OC_INST_VALLOOKUP(int_8) 
OC_INST_VALLOOKUP(int_u8) 
OC_INST_VALLOOKUP(ALLOW_SIZE_T) 
OC_INST_VALLOOKUP(ALLOW_LONG) 
OC_INST_VALLOOKUP(ALLOW_UNSIGNED_LONG) 
OC_INST_VALLOOKUP(ALLOW_LONG_LONG) 
OC_INST_VALLOOKUP(ALLOW_UNSIGNED_LONG_LONG) 
//OC_INST_VALLOOKUP(int_n) 
//OC_INST_VALLOOKUP(int_un) 
OC_INST_VALLOOKUP(bool) 
#endif


// Tab and Arr lookups
OC_INLINE Val& Val::operator[] (int_1  ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_u1 ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_2  ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_u2 ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_4  ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_u4 ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_8  ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_u8 ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (int_n  ii) { return ValLookUpBody_(*this,ii.as());}
OC_INLINE Val& Val::operator[] (int_un ii) { return ValLookUpBody_(*this,ii.as());}
OC_INLINE Val& Val::operator[] (ALLOW_SIZE_T ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (ALLOW_LONG ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (ALLOW_UNSIGNED_LONG ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (ALLOW_LONG_LONG ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (ALLOW_UNSIGNED_LONG_LONG ii) { return ValLookUpBody_(*this,ii);}
OC_INLINE Val& Val::operator[] (bool ii)   { return ValLookUpBody_(*this,ii);}

// Tab and OTab only lookups
#define OC_TABOTABLOOK() if (tag=='o'){OTab&t=*this; return t[v];}Tab&t=*this;return t[v];
OC_INLINE Val& Val::operator[] (real_4 v) { OC_TABOTABLOOK(); }
OC_INLINE Val& Val::operator[] (real_8 v) { OC_TABOTABLOOK(); }
//OC_INLINE Val& Val::operator[] (complex_8 v) { OC_TABOTABLOOK(); }
//OC_INLINE Val& Val::operator[] (complex_16 v) { OC_TABOTABLOOK(); }
  OC_INLINE Val& Val::operator[] (const char* v) { OC_TABOTABLOOK(); }
OC_INLINE Val& Val::operator[] (const Str& v)  { OC_TABOTABLOOK(); }
OC_INLINE Val& Val::operator[] (const Val& v)  { OC_TABOTABLOOK(); }


template <class LOOKUP> Val& ValLookUpBodyThrow_ (const Val& v, LOOKUP ii)
{
  // Proxyies handled via outcast
  if (v.tag=='n') {
    if (v.subtype=='Z') { // For Array<Val>, we can return a reference
      Array<Val>& a = v;
      return a[ii];  // Possibly should use (), but makes more sense to use the version that throws an exception if there's a problem
    }
    // ... but for Arrays of other types, can't return a reference:
    // Force someone to get a contig array, thus force an exception  
    throw logic_error("Only Array<Val> can subscript, Array<T> cannot");
  } else if (v.tag=='o') {
    OTab& ot=v;
    return ot(ii);
  } else if (v.tag=='u') {
    Tup& t=v;
    return t[ii]; // Possibly should use (), but makes more sense to use the version that throws an exception if there's a problem
  } else { // Note that this will throw an exception if v is NOT a Tab
    Tab& t = v;
    return t(ii);
  }
}

#if defined(OC_FACTOR_INTO_H_AND_CC)
// Explicit instantiation
#define OC_INST_VALLOOKUP_THROW(T) template Val& ValLookUpBodyThrow_<T>(const Val&, T);
OC_INST_VALLOOKUP_THROW(int_1) 
OC_INST_VALLOOKUP_THROW(int_u1) 
OC_INST_VALLOOKUP_THROW(int_2) 
OC_INST_VALLOOKUP_THROW(int_u2) 
OC_INST_VALLOOKUP_THROW(int_4) 
OC_INST_VALLOOKUP_THROW(int_u4) 
OC_INST_VALLOOKUP_THROW(int_8) 
OC_INST_VALLOOKUP_THROW(int_u8) 
//OC_INST_VALLOOKUP_THROW(int_n) 
//OC_INST_VALLOOKUP_THROW(int_un) 
OC_INST_VALLOOKUP_THROW(ALLOW_SIZE_T) 
OC_INST_VALLOOKUP_THROW(ALLOW_LONG) 
OC_INST_VALLOOKUP_THROW(ALLOW_UNSIGNED_LONG) 
OC_INST_VALLOOKUP_THROW(ALLOW_LONG_LONG) 
OC_INST_VALLOOKUP_THROW(ALLOW_UNSIGNED_LONG_LONG) 
OC_INST_VALLOOKUP_THROW(bool) 
#endif


// Tab and Arr lookups
OC_INLINE Val& Val::operator() (int_1  ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_u1 ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_2  ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_u2 ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_4  ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_u4 ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_8  ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_u8 ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (int_n  ii) const { return ValLookUpBodyThrow_(*this,ii.as());}
OC_INLINE Val& Val::operator() (int_un ii) const { return ValLookUpBodyThrow_(*this,ii.as());}
OC_INLINE Val& Val::operator() (ALLOW_SIZE_T ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (ALLOW_LONG ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (ALLOW_UNSIGNED_LONG ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (ALLOW_LONG_LONG ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (ALLOW_UNSIGNED_LONG_LONG ii) const { return ValLookUpBodyThrow_(*this,ii);}
OC_INLINE Val& Val::operator() (bool ii)   const { return ValLookUpBodyThrow_(*this,ii);}

// Tab and OTab only lookups
#define OC_OTABLOOKER() if (tag=='o'){OTab&t=*this; return t(v);}Tab&t=*this;return t(v);
OC_INLINE Val& Val::operator() (real_4 v) const { OC_OTABLOOKER(); }
OC_INLINE Val& Val::operator() (real_8 v) const { OC_OTABLOOKER(); }
//OC_INLINE Val& Val::operator() (complex_8 v)  { OC_OTABLOOKER(); }
//OC_INLINE Val& Val::operator() (complex_16 v) { OTABLOOKER(); }
OC_INLINE Val& Val::operator() (const char* v) const { OC_OTABLOOKER(); }
OC_INLINE Val& Val::operator() (const Str& v)   const { OC_OTABLOOKER(); }
OC_INLINE Val& Val::operator() (const Val& v)   const { OC_OTABLOOKER(); }


inline bool ProxyEq (const Proxy& p, const Val& v);

#define ARRAYEQUALITY(T) { Array<T>*vp1,*vp2;vp1=(Array<T>*)&v1.u.n;vp2=(Array<T>*)&v2.u.n; return *vp1==*vp2; }
OC_INLINE bool same_type_equal (const Val& v1, const Val& v2)
{ 
  if (v1.tag != v2.tag) return false; 
  // Proxy handling
  if (v1.isproxy && v2.isproxy) {
    Proxy* p1=(Proxy*)&v1.u.P; Proxy* p2=(Proxy*)&v2.u.P; return *p1==*p2;
  } 
  if (v1.isproxy) {     // One is a proxy, one is not ...
    return ProxyEq(v1,v2);
  } else if (v2.isproxy) {
    return ProxyEq(v2,v1);
  }
  // Assertion: No proxies involved in the comparison
  switch (v1.tag) {
  case 's': return v1.u.s==v2.u.s;
  case 'S': return v1.u.S==v2.u.S;
  case 'i': return v1.u.i==v2.u.i;
  case 'I': return v1.u.I==v2.u.I;
  case 'l': return v1.u.l==v2.u.l;
  case 'L': return v1.u.L==v2.u.L;
  case 'x': return v1.u.x==v2.u.x;
  case 'X': return v1.u.X==v2.u.X;
  case 'b': return v1.u.b==v2.u.b;
  case 'f': return v1.u.f==v2.u.f;
  case 'd': return v1.u.d==v2.u.d;
  case 'c': return v1.u.c==v2.u.c;
  case 'C': return v1.u.C==v2.u.C;
  case 'e': return v1.u.e==v2.u.e;
  case 'E': return v1.u.E==v2.u.E;
  case 'g': return v1.u.g==v2.u.g;
  case 'G': return v1.u.G==v2.u.G;
  case 'h': return v1.u.h==v2.u.h;
  case 'H': return v1.u.H==v2.u.H;
  case 'F': return v1.u.F==v2.u.F;
  case 'D': return v1.u.D==v2.u.D;
  case 'a': { OCString* s1=(OCString*)&v1.u.a; OCString* s2=(OCString*)&v2.u.a; return *s1==*s2; }
  case 't': { Tab* t1=(Tab*)&v1.u.t; Tab* t2=(Tab*)&v2.u.t; return *t1==*t2; }
  case 'o': {OTab*t1=(OTab*)&v1.u.o;OTab*t2=(OTab*)&v2.u.o; return *t1==*t2; }
  case 'u': { Tup* t1=(Tup*)&v1.u.u; Tup* t2=(Tup*)&v2.u.t; return *t1==*t2; }
  case 'q': {int_n*t1=(int_n*)&v1.u.q; int_n*t2=(int_n*)&v2.u.q; return *t1==*t2; }
  case 'Q': {int_un*t1=(int_un*)&v1.u.Q;int_un*t2=(int_un*)&v2.u.Q;return *t1==*t2; }
  case 'n': {
    // Two arrays of different types always have to compare (< and ==)
    // as strings so they are totally-ordered.
    if (v1.subtype != v2.subtype) return Str(v1)==Str(v2); // false; 
    switch(v1.subtype) { 
    case 's': ARRAYEQUALITY(int_1);  
    case 'S': ARRAYEQUALITY(int_u1); 
    case 'i': ARRAYEQUALITY(int_2);  
    case 'I': ARRAYEQUALITY(int_u2); 
    case 'l': ARRAYEQUALITY(int_4);  
    case 'L': ARRAYEQUALITY(int_u4); 
    case 'x': ARRAYEQUALITY(int_8);  
    case 'X': ARRAYEQUALITY(int_u8); 
    case 'b': ARRAYEQUALITY(bool); 
    case 'f': ARRAYEQUALITY(real_4); 
    case 'd': ARRAYEQUALITY(real_8); 
    case 'c': ARRAYEQUALITY(cx_t<int_1>); 
    case 'C': ARRAYEQUALITY(cx_t<int_u1>); 
    case 'e': ARRAYEQUALITY(cx_t<int_2>); 
    case 'E': ARRAYEQUALITY(cx_t<int_u2>); 
    case 'g': ARRAYEQUALITY(cx_t<int_4>); 
    case 'G': ARRAYEQUALITY(cx_t<int_u4>); 
    case 'h': ARRAYEQUALITY(cx_t<int_8>); 
    case 'H': ARRAYEQUALITY(cx_t<int_u8>); 
    case 'F': ARRAYEQUALITY(complex_8); 
    case 'D': ARRAYEQUALITY(complex_16); 
    case 'a': ARRAYEQUALITY(Str);    
    case 't': ARRAYEQUALITY(Tab);    
    case 'o': ARRAYEQUALITY(OTab);    
    case 'u': ARRAYEQUALITY(Tup);    
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': ARRAYEQUALITY(Val);    
    default:  unknownType_("operator==", v1.subtype);
    }
    break;
  }
  case 'Z': return true;
  default:  unknownType_("operator==", v1.tag);
  }
  return false;
}

#define PROXYCOMPEQ(T) { T& t1=p; T& t2=v; return t1==t2; }
#define PROXYCOMPARREQ(T) { Array<T>& a1=p; Array<T>& a2=v; return a1==a2; }
inline bool ProxyEq (const Proxy& p, const Val& v)
{
  // Not equal if not the same type!
  if (p.tag != v.tag) return false;
  if (p.tag=='t') {
    PROXYCOMPEQ(Tab);
  } else if (p.tag=='o') {
    PROXYCOMPEQ(OTab);
  } else if (p.tag=='u') {
    PROXYCOMPEQ(Tup);
  } else if (p.tag=='n') {
    // Keep total ordered, have to compare if different arrays
    if (p.subtype != v.subtype) return Stringize(p)==Str(v);
    switch(p.subtype) { 
    case 's': PROXYCOMPARREQ(int_1);  
    case 'S': PROXYCOMPARREQ(int_u1); 
    case 'i': PROXYCOMPARREQ(int_2);  
    case 'I': PROXYCOMPARREQ(int_u2); 
    case 'l': PROXYCOMPARREQ(int_4);  
    case 'L': PROXYCOMPARREQ(int_u4); 
    case 'x': PROXYCOMPARREQ(int_8);  
    case 'X': PROXYCOMPARREQ(int_u8); 
    case 'b': PROXYCOMPARREQ(bool); 
    case 'f': PROXYCOMPARREQ(real_4); 
    case 'd': PROXYCOMPARREQ(real_8); 
    case 'c': PROXYCOMPARREQ(cx_t<int_1>); 
    case 'C': PROXYCOMPARREQ(cx_t<int_u1>); 
    case 'e': PROXYCOMPARREQ(cx_t<int_2>); 
    case 'E': PROXYCOMPARREQ(cx_t<int_u2>); 
    case 'g': PROXYCOMPARREQ(cx_t<int_4>); 
    case 'G': PROXYCOMPARREQ(cx_t<int_u4>); 
    case 'h': PROXYCOMPARREQ(cx_t<int_8>); 
    case 'H': PROXYCOMPARREQ(cx_t<int_u8>); 
    case 'F': PROXYCOMPARREQ(complex_8); 
    case 'D': PROXYCOMPARREQ(complex_16); 
    case 'a': PROXYCOMPARREQ(Str);    
    case 't': PROXYCOMPARREQ(Tab);    
    case 'o': PROXYCOMPARREQ(OTab);    
    case 'u': PROXYCOMPARREQ(Tup);    
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': PROXYCOMPEQ(Arr);    
    default:  unknownType_("ProxyEq", p.subtype);
    }
  } else  {
    throw logic_error("Internal error");
  }
  return false;
}

#define PROXYCOMPLT(T) { T& t1=p; T& t2=v; return invert ? t2<t1 : t1<t2; }
#define PROXYCOMPARRLT(T) { Array<T>& a1=p; Array<T>& a2=v; return invert ? a2<a1 : a1<a2; }
inline bool ProxyLT (const Proxy& p, const Val& v, bool invert)
{
  // Default kind of compare (like Val compares for total order)
  if (p.tag != v.tag) return invert ? Str(v)<Stringize(p) : Stringize(p)<Str(v); 
  if (p.tag=='t') {
    PROXYCOMPLT(Tab);
  } else if (p.tag=='n') {
    // Keep total ordered, have to compare if different arrays
    if (p.subtype != v.subtype) return invert ? Str(v)<Stringize(p) : Stringize(p)<Str(v);
    switch(p.subtype) { 
    case 's': PROXYCOMPARRLT(int_1);  
    case 'S': PROXYCOMPARRLT(int_u1); 
    case 'i': PROXYCOMPARRLT(int_2);  
    case 'I': PROXYCOMPARRLT(int_u2); 
    case 'l': PROXYCOMPARRLT(int_4);  
    case 'L': PROXYCOMPARRLT(int_u4); 
    case 'x': PROXYCOMPARRLT(int_8);  
    case 'X': PROXYCOMPARRLT(int_u8); 
    case 'b': PROXYCOMPARRLT(bool); 
    case 'f': PROXYCOMPARRLT(real_4); 
    case 'd': PROXYCOMPARRLT(real_8); 
    case 'c': PROXYCOMPARRLT(cx_t<int_1>); 
    case 'C': PROXYCOMPARRLT(cx_t<int_u1>); 
    case 'e': PROXYCOMPARRLT(cx_t<int_2>); 
    case 'E': PROXYCOMPARRLT(cx_t<int_u2>); 
    case 'g': PROXYCOMPARRLT(cx_t<int_4>); 
    case 'G': PROXYCOMPARRLT(cx_t<int_u4>); 
    case 'h': PROXYCOMPARRLT(cx_t<int_8>); 
    case 'H': PROXYCOMPARRLT(cx_t<int_u8>); 
    case 'F': PROXYCOMPARRLT(complex_8); 
    case 'D': PROXYCOMPARRLT(complex_16); 
    case 'a': PROXYCOMPARRLT(Str);    
    case 't': PROXYCOMPARRLT(Tab);    
    case 'o': PROXYCOMPARRLT(OTab);    
    case 'u': PROXYCOMPARRLT(Tup);    
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': PROXYCOMPLT(Arr);    
    default:  unknownType_("ProxyLT", p.subtype);
    }
  } else {
    throw logic_error("Internal error");
  }
  return false;
}


OC_INLINE bool operator== (const Val& v1, const Val& v2)
{
  if (v1.tag==v2.tag) 
    return same_type_equal(v1,v2);

  // So that dicts can compare like in Python
  if (v1.tag=='o' && v2.tag=='t') {
    OTab& o = v1; 
    Tab&  t = v2;
    return bool(o==t);
  } else if (v1.tag=='t' && v2.tag=='o') {
    Tab& t = v1; 
    OTab& o = v2;
    return bool(t==o);
  }

  // If they are both Numeric types, and they are equal when cast up
  // to a bigger type, then they are equal.  If they are different
  // types and NOT Numeric types, they are not equal.  Note that two
  // arrays of different types will NOT be equal BUT two arrays of
  // Vals can be equal if their Numeric types are the same.  BTW, this
  // is basically how Python works: we are doing what they do.

  switch (v1.tag) {
  case 'q': { // int_n types, but with hugers
    int_n* l1 = (int_n*)&v1.u.q;
    switch (v2.tag) {
    case'b':                         
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': { int_8 t2=v2; return *l1==int_un(t2); } 
    case'f':case'd':           { return MakeRealFromBigInt(*l1)==real_8(v2); }
    case'F':case'D':           { real_8 r1 = MakeRealFromBigInt(*l1); complex_16 t1=r1; complex_16 t2=v2; return t1==t2; }
    case'c':case'e':case'g':case'h': 
                       {cx_t<int_8>t2=v2;return(t2.im)?0:*l1==int_n(t2.re); }
    case'C':case'E':case'G':case'H': 
                       {cx_t<int_u8>t2=v2;return(t2.im)?0:*l1==int_un(t2.re); }
    case'Q':                   { int_un* l2=(int_un*)&v2.u.Q; return *l1==*l2;}
    default:                   { return false; }
    }
  }
  case 'Q': { // int_un
    int_un* l1 = (int_un*)&v1.u.Q;
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x': { int_8 t2=v2; return *l1==int_n(t2); }
    case'S':case'I':case'L':case'X': { int_u8 tu2=v2; return *l1==int_un(tu2); }
    case'f':case'd':           { return MakeRealFromBigUInt(*l1)==real_8(v2); }
    case'F':case'D':           { real_8 r1 = MakeRealFromBigUInt(*l1); complex_16 t1=r1; complex_16 t2=v2; return t1==t2; } 
    case'c':case'e':case'g':case'h': 
                       {cx_t<int_8>t2=v2;return(t2.im)?0:*l1==int_n(t2.re); }
    case'C':case'E':case'G':case'H': 
                       {cx_t<int_u8>t2=v2;return(t2.im)?0:*l1==int_un(t2.re); }
    case'q':                   { int_n* l2=(int_n*)&v2.u.q; return *l1==*l2; } 
    default:                   { return false; }
    }
  }
  case 's': case 'i': case 'l': case 'x': { // int types
    switch (v2.tag) {
    case'b':                         
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': { return int_8(v1)==int_8(v2); }
    case'f':case'd':                 { return real_8(v1)==real_8(v2); }
    case'F':case'D':     { complex_16 t1=v1; complex_16 t2=v2; return t1==t2; }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'q':case'Q': { return v2==v1; }
    default:                         { return false; }
    }
  } 
  case 'S': case 'I': case 'L': case 'X': { // unsigned int types
    switch (v2.tag) {
    case's':case'i':case'l':case'x': { return int_8(v1)==int_8(v2); }

    case'b':            
    case'S':case'I':case'L':case'X': { return int_u8(v1)==int_u8(v2); }
    case'f':case'd':                 { return real_8(v1)==real_8(v2); }
    case'F':case'D':    { complex_16 t1=v1; complex_16 t2=v2; return t1==t2; }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'q':case'Q': { return v2==v1; }
    default:            { return false; }
    }
  } 
  case 'f': case 'd': { // real types
    switch(v2.tag) {
    case'b':                         { return int_8(v1)==int_8(v2); }
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': 
    case'f':case'd':                 { return real_8(v1)==real_8(v2); }
    case'F':case'D':     { complex_16 t1=v1; complex_16 t2=v2; return t1==t2; }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'q':case'Q': { return v2==v1; }
    default:                         { return false; }
    }
  } 
  case 'c': case 'e': case 'g': case 'h': { // cx int types
    cx_t<int_8> t1 = v1;
    switch (v2.tag) {
    case'b':                         
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': { return(t1.im)?false:(int_8(v2)==t1.re);}
    case'f':case'd':                 { return(t1.im)?false:(real_8(v2)==t1.re);}
    case'c':case'e':case'g':case'h': { cx_t<int_8> t2=v2; return t1==t2; }
    case'C':case'E':case'G':case'H': { cx_t<int_u8> t2=v2; return t1==t2; }
    case'F':case'D':     { complex_16 t1=v1; complex_16 t2=v2; return t1==t2; }
    case'q':case'Q':                 { return v2==v1; }
    default:                         { return false; }
    }
  }
  case 'C': case 'E': case 'G': case 'H': { // cx unsigned int types
    cx_t<int_u8> t1 = v1;
    switch (v2.tag) {
    case'b':                         
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': { return(t1.im)?false:(int_u8(v2)==t1.re);}
    case'f':case'd':                 { return(t1.im)?false:(real_8(v2)==t1.re);}
    case'c':case'e':case'g':case'h': { cx_t<int_8> t2=v2; return t1==t2; }
    case'C':case'E':case'G':case'H': { cx_t<int_u8> t2=v2; return t1==t2; }
    case'F':case'D':     { complex_16 t1=v1; complex_16 t2=v2; return t1==t2; }
    case'q':case'Q':                 { return v2==v1; }
    default:                         { return false; }
    }
  }
  case'F':case'D': {   // cx types
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x':
    case'S':case'I':case'L':case'X':
    case'f':case'd':
    case'F':case'D':     { complex_16 t1=v1; complex_16 t2=v2; return t1==t2; }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'q':case'Q': { return v2==v1; }
    default:            { return false; }
    }
  } 
  case 'b': {    // bool types
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x':
    case'S':case'I':case'L':case'X':
    case'f':case'd':
    case'F':case'D':                { return bool(v1)==bool(v2); }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'q':case'Q': { return v2==v1; }
    default:            { return false; }
    } 
  }
  default: return false;
  }
}


#define ARRAYLESSTHAN(T) { Array<T>*vp1,*vp2;vp1=(Array<T>*)&v1.u.n;vp2=(Array<T>*)&v2.u.n; return *vp1<*vp2; }
OC_INLINE bool same_type_lt (const Val& v1, const Val& v2)
{   
  if (v1.tag != v2.tag) return Str(v1)<Str(v2); // false;
  // Proxy handling
  if (v1.isproxy && v2.isproxy) {
    Proxy& p1=v1; Proxy& p2=v2; return p1<p2;
  } 
  if (v1.isproxy) {     // One is a proxy, one is not ...
    return ProxyLT(v1,v2, false);
  } else if (v2.isproxy) {
    return ProxyLT(v2,v1, true);
  }
  // Assertion: No proxies involved in compare, so can get value straight out
  switch (v1.tag) {
  case 's': return v1.u.s<v2.u.s;
  case 'S': return v1.u.S<v2.u.S;
  case 'i': return v1.u.i<v2.u.i;
  case 'I': return v1.u.I<v2.u.I;
  case 'l': return v1.u.l<v2.u.l;
  case 'L': return v1.u.L<v2.u.L;
  case 'x': return v1.u.x<v2.u.x;
  case 'X': return v1.u.X<v2.u.X;
  case 'b': return v1.u.b<v2.u.b;
  case 'f': return v1.u.f<v2.u.f;
  case 'd': return v1.u.d<v2.u.d;
  case 'c': case 'C': case 'e': case 'E' : case 'g': case 'G': case 'h': case 'H':
  case 'F': //return complex_8(v1.u.F.re,v1.u.F.im) < 
	    //  complex_8(v2.u.F.re,v2.u.F.im);
  case 'D': //return complex_16(v1.u.D.re,v1.u.D.im) < 
	    //  complex_16(v2.u.D.re,v2.u.D.im);
           throw runtime_error("Complex types are not ordered");
  case 'a': { OCString* s1=(OCString*)&v1.u.a; OCString* s2=(OCString*)&v2.u.a; return *s1<*s2; }
  case 't': {Tab* t1=(Tab*)&v1.u.t; Tab* t2=(Tab*)&v2.u.t; return (*t1)<(*t2);}
  case 'o':{ OTab*t1=(OTab*)&v1.u.o;OTab* t2=(OTab*)&v2.u.o;return(*t1)<(*t2);}
  case 'u': { Tup* t1=(Tup*)&v1.u.u;Tup* t2=(Tup*)&v2.u.u; return (*t1)<(*t2);}
  case 'n': {
    if (v1.subtype != v2.subtype) return Str(v1)<Str(v2); // false;
    switch(v1.subtype) { 
    case 's': ARRAYLESSTHAN(int_1);  
    case 'S': ARRAYLESSTHAN(int_u1); 
    case 'i': ARRAYLESSTHAN(int_2);  
    case 'I': ARRAYLESSTHAN(int_u2); 
    case 'l': ARRAYLESSTHAN(int_4);  
    case 'L': ARRAYLESSTHAN(int_u4); 
    case 'x': ARRAYLESSTHAN(int_8);  
    case 'X': ARRAYLESSTHAN(int_u8); 
    case 'b': ARRAYLESSTHAN(bool); 
    case 'f': ARRAYLESSTHAN(real_4); 
    case 'd': ARRAYLESSTHAN(real_8); 
    case 'c': ARRAYLESSTHAN(cx_t<int_1>); 
    case 'C': ARRAYLESSTHAN(cx_t<int_u1>); 
    case 'e': ARRAYLESSTHAN(cx_t<int_2>); 
    case 'E': ARRAYLESSTHAN(cx_t<int_u2>); 
    case 'g': ARRAYLESSTHAN(cx_t<int_4>); 
    case 'G': ARRAYLESSTHAN(cx_t<int_u4>); 
    case 'h': ARRAYLESSTHAN(cx_t<int_8>); 
    case 'H': ARRAYLESSTHAN(cx_t<int_u8>); 
    case 'F': ARRAYLESSTHAN(complex_8); 
    case 'D': ARRAYLESSTHAN(complex_16); 
    case 'a': ARRAYLESSTHAN(Str);    
    case 't': ARRAYLESSTHAN(Tab);    
    case 'o': ARRAYLESSTHAN(OTab);    
    case 'u': ARRAYLESSTHAN(Tup);    
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': ARRAYLESSTHAN(Val);    
    default:  unknownType_("operator<", v1.subtype);
    }
  }
  case 'Z': return false;
  default:  unknownType_("operator<", v1.tag);
  }
  return false;
}

OC_INLINE bool operator< (const Val& v1, const Val& v2)
{
  if (v1.tag == v2.tag) 
    return same_type_lt(v1,v2);
  else if (v1.tag == 'Z') 
    return true;
  else if (v2.tag == 'Z') // None is less than everything
    return false;

  // Numeric types are always less than string types, which is what
  // things get converted to if they have no obvious comparison.
  // This seems to be what Python does.
  switch (v1.tag) {
  case 'q': { // int_n types, but with hugers
    int_n* l1 = (int_n*)&v1.u.q;
    switch (v2.tag) {
    case's':case'i':case'l':case'x': { int_8 t2=v2; return *l1<int_n(t2); }
    case'S':case'I':case'L':case'X': { int_u8 t2=v2;   return *l1<int_un(t2); }
    case'f':case'd':             { return MakeRealFromBigInt(*l1)<real_8(v2); }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D': { throw runtime_error("Complex types are not ordered"); } 
    case'b':                     { return int_8(v1)<int_8(v2); }
    case'Q':                     { int_un*l2=(int_un*)&v2.u.Q;return *l1<*l2; } 
    default:                     { return true; }
    }
  }
  case 'Q': { // int_un
    int_un* l1 = (int_un*)&v1.u.Q;
    switch (v2.tag) {
    case'b':                         
    case's':case'i':case'l':case'x': { int_8 t2=v2; return *l1<int_n(t2); }
    case'S':case'I':case'L':case'X': { int_u8 t2=v2;   return *l1<int_un(t2); }
    case'f':case'd':             { return MakeRealFromBigUInt(*l1)<real_8(v2); }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':  { throw runtime_error("Complex types are not ordered"); } 
    case'q':                     { int_n* l2=(int_n*)&v2.u.q; return *l1<*l2; } 
    default:                     { return true; }
    }
  }
  case 's': case 'i': case 'l': case 'x': { // int types
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x':  
    case'S':case'I':case'L':case'X':  { return int_8(v1)<int_8(v2); }
    case'f':case'd':                  { return real_8(v1)<real_8(v2); }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':  { throw runtime_error("Complex types are not ordered"); } 
    case'q':case'Q':                  { return !(v2<v1 || v2==v1); } // Reverse
    default:                          { return true; }
    }
  } 
  case 'S': case 'I': case 'L': case 'X': { // unsigned int types
    switch (v2.tag) {
    case'b':                          { return int_u8(v1)<int_u8(v2); }
    case's':case'i':case'l':case'x':  { return int_8(v1)<int_8(v2); }
    case'S':case'I':case'L':case'X':  
    case'f':case'd':                  { return real_8(v1)<real_8(v2); }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':  { throw runtime_error("Complex types are not ordered"); } 
    case'q':case'Q':                  { return !(v2<v1 || v2==v1); } // Reverse
    default:                          { return true; }
    }
  } 
  case 'f': case 'd': { // real types
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': 
    case'f':case'd':                 { return real_8(v1)<real_8(v2); }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':  { throw runtime_error("Complex types are not ordered"); } 
    case'q':case'Q':                  { return !(v2<v1 || v2==v1); } // Reverse
    default:                         { return true; }
    }
  } 
  case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':  { 
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': 
    case'f':case'd':
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':
    case'q':case'Q':  { throw runtime_error("Complex types are not ordered"); } 
    default:                         { return true; }
    }
  }
  case 'b':  { // bool
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': 
    case'f':case'd':           { return bool(v1)<bool(v2); }
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':  { throw runtime_error("Complex types are not ordered"); } 
    case'q':case'Q':                  { return !(v2<v1 || v2==v1); } // Reverse
    default:                       { return true; }
   }

  } 
  default: {
    switch (v2.tag) {
    case'b':
    case's':case'i':case'l':case'x': 
    case'S':case'I':case'L':case'X': 
    case'f':case'd':           
    case'c':case'e':case'g':case'h':case'C':case'E':case'G':case'H':case'F':case'D':
      return false; // Numeric types always less-than string types 
   default:
     return Str(v1)<Str(v2);  // Compare as string types if no obvious compare
    }
  }

  } // end v1.tag switch
}  

OC_END_NAMESPACE

#include "ocstringtools.h"

#include "ocnumerictools.h"
#include "ocnumpytools.h"

OC_BEGIN_NAMESPACE

// Choose how prettyPrint prints out the POD arrays
enum ArrayOutputOptions_e { NATURAL=1, LIKE_NUMPY=2, LIKE_NUMERIC=3 };
#if !defined(OCARRAY_OPTIONS_DEFAULT)
# define OCARRAY_OPTIONS_DEFAULT LIKE_NUMERIC  // Sigh ... until we feel numpy
#endif


// Make Array<T> look like Python Numeric arrays
template <class T> 
OC_INLINE ostream& PrintArray (ostream& os, const Array<T>& a)
{
  const size_t len = a.length();
  os << "array([";
  if (len) {
    for (size_t ii=0; ii<len-1; ++ii) {
      os << Val(a[ii]) << ",";
    }
    os << Val(a[len-1]);
  }
  os << "], ";
  static ArrayOutputOptions_e ArrayOutputOptions = OCARRAY_OPTIONS_DEFAULT;
  if (ArrayOutputOptions == LIKE_NUMERIC) {
    os << "'" << OCTagToNumeric(TagFor((T*)0), true) << "')";
  } else {
    os << "dtype=" << OCTagToNumPy(TagFor((T*)0), true) << ")";
  }
  return os;
}

// Specialization for Array<Val>: Make them look like pythons lists
template <>
OC_INLINE ostream& PrintArray <Val> (ostream& os, const Array<Val>& a)
{
  const size_t len = a.length();
  os << "[";
  if (len) {
    for (size_t ii=0; ii<len-1; ii++) {
      const Val& v = a[ii];
      os << v << ", ";
    }
    os << a[len-1];
  }
  return os << "]";
}

// Specialization for Array<Val>: Make them look like pythons lists
template <>
OC_INLINE ostream& operator<< <Val> (ostream& os, const Array<Val>& a)
{ return PrintArray(os,a); }

// Make this look similar to python
#define PRINTARRAY(T) {Array<T>*ap=(Array<T>*)&v.u.n; return PrintArray(os, *ap);}
#define OUTPUTCX(T,MEMBER) { T*cxp=(T*)&v.u.MEMBER; os << *cxp; break; }
OC_INLINE ostream& operator<< (ostream& os, const Val& v)
{
  if (v.isproxy) { 
    Proxy *p = (Proxy*)&v.u.P; 
    return os << *p; 
  }
  switch(v.tag) {
  case 's': os << int_4(v.u.s); break;  // because python doesn't have char
  case 'S': os << int_u4(v.u.S); break; // because python doesn't have char
  case 'i': os << v.u.i; break;
  case 'I': os << v.u.I; break;
  case 'l': os << v.u.l; break;
  case 'L': os << v.u.L; break;
  case 'x': os << v.u.x; break;
  case 'X': os << v.u.X; break;
  case 'q': { int_n* temp = (int_n*)&v.u.q; os << (*temp) << "L"; break; }
  case 'Q': { int_un* temp = (int_un*)&v.u.Q; os << (*temp) << "L"; break; }
  case 'b': { Str res = "False"; if (v.u.b) res="True"; os << res; break; }
  case 'f': os << Stringize(v.u.f); break; 
  case 'd': os << Stringize(v.u.d); break;

    // TODO: is this X-Midas problematic? 
  case 'c': OUTPUTCX(cx_t<int_1>, c); 
  case 'C': OUTPUTCX(cx_t<int_u1>,C); 
  case 'e': OUTPUTCX(cx_t<int_2>, e); 
  case 'E': OUTPUTCX(cx_t<int_u2>,E); 
  case 'g': OUTPUTCX(cx_t<int_4> ,g); 
  case 'G': OUTPUTCX(cx_t<int_u4>,G); 
  case 'h': OUTPUTCX(cx_t<int_8>, h); 
  case 'H': OUTPUTCX(cx_t<int_u8>,H);
 
#if defined(OC_SUPPORT_XM)
  case 'F': os.precision(OC_FLT_DIGITS);  os<<"("<<v.u.F.re<<((v.u.F.im<0)?"":"+")<<v.u.F.im<<"j)"; break;
  case 'D': os.precision(OC_DBL_DIGITS); os<<"("<<v.u.D.re<<((v.u.D.im<0)?"":"+")<<v.u.D.im<<"j)"; break;
#else
  case 'F': os << complex_8(v.u.F.re, v.u.F.im); break;
  case 'D': os << complex_16(v.u.D.re, v.u.D.im); break;
#endif
  case 'a': { OCString *s = (OCString*)&v.u.a; os << PyImage(*s);  break; }
  case 't': { Tab *t = (Tab*)&v.u.t; os << *t; break; }
  case 'o': { OTab*t = (OTab*)&v.u.o; os << *t; break; }
  case 'u': { Tup *t = (Tup*)&v.u.u; os << *t; break; }
  case 'n': { 
    switch(v.subtype) { 
    case 's': PRINTARRAY(int_1);  
    case 'S': PRINTARRAY(int_u1); 
    case 'i': PRINTARRAY(int_2);  
    case 'I': PRINTARRAY(int_u2); 
    case 'l': PRINTARRAY(int_4);  
    case 'L': PRINTARRAY(int_u4); 
    case 'x': PRINTARRAY(int_8);  
    case 'X': PRINTARRAY(int_u8); 
    case 'b': PRINTARRAY(bool);   
    case 'f': PRINTARRAY(real_4); 
    case 'd': PRINTARRAY(real_8); 
    case 'c': PRINTARRAY(cx_t<int_1>); 
    case 'C': PRINTARRAY(cx_t<int_u1>); 
    case 'e': PRINTARRAY(cx_t<int_2>); 
    case 'E': PRINTARRAY(cx_t<int_u2>); 
    case 'g': PRINTARRAY(cx_t<int_4>); 
    case 'G': PRINTARRAY(cx_t<int_u4>); 
    case 'h': PRINTARRAY(cx_t<int_8>); 
    case 'H': PRINTARRAY(cx_t<int_u8>);
    case 'F': PRINTARRAY(complex_8); 
    case 'D': PRINTARRAY(complex_16); 
    case 'a': PRINTARRAY(Str);    
    case 't': PRINTARRAY(Tab);    
    case 'o': PRINTARRAY(OTab);    
    case 'u': PRINTARRAY(Tup);    
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': PRINTARRAY(Val);    
    default:  unknownType_("operator<<", v.subtype);
    }
  }
  case 'Z': os << "None"; break;
  default: unknownType_("operator<<", v.tag);
  }
  return os;
}

// Make this look like pythons dictionaries
OC_INLINE ostream& operator<< (ostream& os, const Tab& t)
{
  os << "{";
  for (It it(t); it(); ) {
    // os << it.key() << ": " << it.value();
    const Val& key = it.key();
    const Val& value = it.value();
    os << key << ": " << value;
 
    // See if we want to output a comma
    It next(it); 
    if (next()) { 
      os << ", ";
    }
  }
  return os << "}";
}

// Make this look like pythons ordered dictionaries
OC_INLINE ostream& operator<< (ostream& os, const OTab& t)
{
  os << "OrderedDict([";
  for (It it(t); it(); ) {
    // os << it.key() << ": " << it.value();
    const Val& key = it.key();
    const Val& value = it.value();
    os << "(" << key << ", " << value << ")";
 
    // See if we want to output a comma
    It next(it); 
    if (next()) { 
      os << ", ";
    }
  }
  return os << "])";
}

// Make this look like pythons tuples
OC_INLINE ostream& operator<< (ostream& os, const Tup& t)
{
  os << "(";
  const size_t len = t.length();
  for (size_t ii=0; ii<len; ii++) {
    os << t[ii];
    if (ii!=len-1) os << ", ";
  }
  return os << ")";
}

OC_END_NAMESPACE

// Due to crazy include dependencies, we do this last.
#include "ocvalreader.h"

OC_BEGIN_NAMESPACE


OC_INLINE Tab::Tab (const char* cc, Allocator*) 
{ 
  ValReader r(cc); 
  r.expectTab(*this);
} 
OC_INLINE Tab::Tab (const Str& s, Allocator*) 
{
  ValReader r(s.c_str());
  r.expectTab(*this);
}

OC_INLINE Tab::Tab (Allocator* alloc) : AVLHashT<Val, Val, 8>(alloc) { }

OC_INLINE Tab::Tab (const Tab& t, Allocator* a) : 
  AVLHashT<Val, Val, 8>(t, a) 
{ }


OC_INLINE void Tab::appendHelper_ (const Val& key, const Val& value)
{
  int_u4 hashkey = HashFunction(key);
  Found_e found_where; N* node = this->lookup_(key, hashkey, found_where);
  if (found_where==FOUND) {
    string err = 
      "Appending key:"+Stringize(key)+" with value:"+Stringize(value)+
      " into table:"+Stringize(*this)+" will cause an overwrite of the "
      "previous key of that value: That's probably not what you want append "
      "to do.";
    throw runtime_error(err);
  } else {
    (void)notInTableInsert_(node, key, hashkey, value, found_where);
  }
}

OC_INLINE void Tab::append (const Val& value) 
{ 
  Val key=entries();
  appendHelper_(key, value);
}

OC_INLINE void Tab::appendStr (const Val& value) 
{
  Val key=Stringize(entries()); 
  appendHelper_(key, value);
}

OC_INLINE Tab& Tab::operator= (const Tab& rhs)
{ 
  if (&rhs!=this) { // Below will work if &rhs==this, but unnecessary work
    // To keep the lhs (this) alive, we hold onto it until
    // the assignment is done: we do this for nesting
    Tab temp;
    this->swap(temp);
    
    // temp holds the full rhs, we need to destruct the current Tab
    // (it's just empty() so we can construct into its memory.
    this->~Tab();
    new (this) Tab(rhs);
  }
  return *this; 
}

OC_INLINE Tab& Tab::operator+= (const Tab& rhs)
{ 
  for (It ii(rhs); ii();) { 
    this->insertKeyAndValue(ii.key(), ii.value()); 
  }
  return *this; 
}

OC_INLINE ostream& Tab::prettyPrintHelper_ (ostream& os, int indent, 
					    bool pretty, int indent_additive) const
{
  // Base case, empty table
  if (entries()==0) return os << "{ }"; 

  // Recursive case
  os << "{";
  if (pretty) os << endl;

  // Iterate through, printing out each element
  Sit sii(*this);
  for (size_t ii=0; sii(); ii++) {
    const Val& key = sii.key();
    const Val& value = sii.value();
    
    if (pretty) indentOut_(os, indent+indent_additive);
    os << key << ":";
    
    // For most values, use default output method
    switch (value.tag) {
    case 'a': { 
      OCString* ap = (OCString*)&value.u.a;
      os << PyImage(*ap);
      break; 
    }
    case 't': { 
      Tab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'o': { 
      OTab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'u': { 
      Tup& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'n': { 
      if (value.subtype=='Z') {
	Arr& arr = value;
	arr.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			       pretty, indent_additive);
	break;
      } // else fall thru for other array types
    }
    default: os << value; break;
    }

    if (entries()>1 && ii!=(entries())-1) os << ","; // commas on all but last
    if (pretty) os << endl;
  }

  if (pretty) indentOut_(os, indent);
  return os << "}";
}

OC_INLINE void Tab::prettyPrint (ostream& os, int starting_indent, 
				 int indent_additive) const
{
  indentOut_(os, starting_indent);
  prettyPrintHelper_(os, starting_indent, true, 
		     indent_additive) << endl;
}



OC_INLINE OTab::OTab (const char* cc, Allocator*) 
{ 
  ValReader r(cc); 
  r.expectOTab(*this);
} 
OC_INLINE OTab::OTab (const Str& s, Allocator*) 
{
  ValReader r(s.c_str());
  r.expectOTab(*this);
}

OC_INLINE OTab::OTab (Allocator* alloc) : OrdAVLHashT<Val, Val, 8>(alloc) { }

OC_INLINE OTab::OTab (const OTab& t, Allocator* a) : 
  OrdAVLHashT<Val, Val, 8>(t, a) 
{ }


OC_INLINE void OTab::appendHelper_ (const Val& key, const Val& value)
{
  int_u4 hashkey = HashFunction(key);
  Found_e found_where; N* node = this->lookup_(key, hashkey, found_where);
  if (found_where==FOUND) {
    string err = 
      "Appending key:"+Stringize(key)+" with value:"+Stringize(value)+
      " into table:"+Stringize(*this)+" will cause an overwrite of the "
      "previous key of that value: That's probably not what you want append "
      "to do.";
    throw runtime_error(err);
  } else {
    (void)notInTableInsert_(node, key, hashkey, value, found_where);
  }
}

OC_INLINE void OTab::append (const Val& value) 
{ 
  Val key=entries();
  appendHelper_(key, value);
}

OC_INLINE void OTab::appendStr (const Val& value) 
{
  Val key=Stringize(entries()); 
  appendHelper_(key, value);
}

OC_INLINE OTab& OTab::operator= (const OTab& rhs)
{ 
  if (&rhs!=this) { // Below will work if &rhs==this, but unnecessary work
    // To keep the lhs (this) alive, we hold onto it until
    // the assignment is done: we do this for nesting
    OTab temp;
    this->swap(temp);
    
    // temp holds the full rhs, we need to destruct the current OTab
    // (it's just empty() so we can construct into its memory.
    this->~OTab();
    new (this) OTab(rhs);
  }
  return *this; 
}

OC_INLINE OTab& OTab::operator+= (const OTab& rhs)
{ 
  for (It ii(rhs); ii();) { 
    this->insertKeyAndValue(ii.key(), ii.value()); 
  }
  return *this; 
}

// TODO: What should the default of OTab pretty print be?
// o{ 'a': 1, 'b':1 } 
// ['a':1, 'b':2]
// OrderedDict([('a',1), ('b':2)])
// Easiest right now is o{ }, but will revisit
// I also like odict() instead of dict.
static const char* OTabEmpty[]={ "OrderedDict([])", "o{ }","OrderedDict([])" };
static const char* OTabLeft[] ={ "OrderedDict([", "o{", "[" };
static const char* OTabRight[]={ "])", "}", "]" };
//#define OC_DEFAULT_OTAB_REPR 1
#if !defined(OC_DEFAULT_OTAB_REPR) 
#  define OC_DEFAULT_OTAB_REPR 1
#endif
static int  OTabRepr = OC_DEFAULT_OTAB_REPR;

OC_INLINE ostream& OTab::prettyPrintHelper_ (ostream& os, int indent, 
					     bool pretty, int indent_additive) const
{
  // Base case, empty table
  if (entries()==0) { 
    return os << OTabEmpty[OTabRepr]; // "o{ }"; 
  }

  // Recursive case
  os << OTabLeft[OTabRepr]; // "o{";
  if (pretty) os << endl;

  // Iterate through, printing out each element
  It iii(*this);
  for (size_t ii=0; iii(); ii++) {
    const Val& key = iii.key();
    const Val& value = iii.value();
    
    if (pretty) indentOut_(os, indent+indent_additive);
    if (OTabRepr==0) {
      os << "(" << key << ", ";
    } else {
      os << key << ":";
    }
    
    // For most values, use default output method
    switch (value.tag) {
    case 'a': { 
      OCString* ap = (OCString*)&value.u.a;
      os << PyImage(*ap);
      break; 
    }
    case 't': { 
      Tab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'o': { 
      OTab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'u': { 
      Tup& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'n': { 
      if (value.subtype=='Z') {
	Arr& arr = value;
	arr.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			       pretty, indent_additive);
	break;
      } // else fall thru for other array types
    }
    default: os << value; break;
    }
    if (OTabRepr==0) {
      os << ")";
    } 

    if (entries()>1 && ii!=(entries())-1) os << ","; // commas on all but last
    if (pretty) os << endl;
  }

  if (pretty) indentOut_(os, indent);
  return os << OTabRight[OTabRepr]; // "}";
}

OC_INLINE void OTab::prettyPrint (ostream& os, int starting_indent, 
				  int indent_additive) const
{
  indentOut_(os, starting_indent);
  prettyPrintHelper_(os, starting_indent, true, 
		     indent_additive) << endl;
}



OC_INLINE Tup::Tup (Allocator* a):a_(Array<Val>::ARRAY_DEFAULT_CAPACITY,a) { }

OC_INLINE Tup::Tup (const Tup& t, Allocator* a) :
  a_(t.a_, a) 
{ }

OC_INLINE Tup& Tup::operator= (const Tup& rhs)
{ 
  if (&rhs!=this) { // Below will work if &rhs==this, but unnecessary work
    // To keep the lhs (this) alive, we hold onto it until
    // the assignment is done: we do this for nesting
    Tup temp;
    this->swap(temp);
    
    // temp holds the full rhs, we need to destruct the current Tup
    // (it's just empty() so we can construct into its memory.
    this->~Tup();
    new (this) Tup(rhs);
  }
  return *this; 
}


OC_INLINE ostream& Tup::prettyPrintHelper_ (ostream& os, int indent,
                                            bool pretty, int indent_additive) const
{
  // Base case, empty
  if (entries()==0) return os << "( )";

  // Recursive case
  os << "(";
  if (pretty) os << endl;

  // Iterate through, printing out each element
  size_t ent = entries();
  for (size_t ii=0; ii<ent; ++ii) {
    const Val& value = (*this)[ii];

    if (pretty) indentOut_(os, indent+indent_additive);

    // For most values, use default output method
    switch (value.tag) {
    case 'a': {
      OCString* ap = (OCString*)&value.u.a;
      os << PyImage(*ap);
      break;
    }
    case 't': {
      Tab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0,
                           pretty, indent_additive);
      break;
    }
    case 'o': {
      OTab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0,
                           pretty, indent_additive);
      break;
    }
    case 'u': {
      Tup& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0,
                           pretty, indent_additive);
      break;
    }
    case 'n': {
      if (value.subtype=='Z') {
        Arr& arr = value;
        arr.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0,
                               pretty, indent_additive);
        break;
      } // else fall thru for other array types
    }
    default: os << value; break;
    }

    if (entries()>1 && ii!=(entries())-1) os << ","; // commas on all but last
    if (pretty) os << endl;
  }

  if (pretty) indentOut_(os, indent);
  return os << ")";
}

OC_INLINE void Tup::prettyPrint (ostream& os, int starting_indent, 
				 int indent_additive) const
{
  indentOut_(os, starting_indent);
  prettyPrintHelper_(os, starting_indent, true, 
		     indent_additive) << endl;
}




OC_INLINE Arr::Arr (const char* cc, Allocator* al) : 
  Array<Val>(ARRAY_DEFAULT_CAPACITY, al) 
{ ValReader r(cc); r.expectArr(*this); } 
OC_INLINE Arr::Arr (const Str& s, Allocator* al) : 
  Array<Val>(ARRAY_DEFAULT_CAPACITY, al) 
{ ValReader r(s.c_str());r.expectArr(*this); }
OC_INLINE Arr::Arr (Allocator* al) : 
  Array<Val>(ARRAY_DEFAULT_CAPACITY, al) 
{ }
OC_INLINE Arr::Arr (size_t len, Allocator* al) : 
  Array<Val>(len, al) { }

OC_INLINE Arr& Arr::operator= (const Arr& rhs)
{ 
  if (&rhs!=this) { // Below will work if &rhs==this, but unnecessary work
    // To keep the lhs (this) alive, we hold onto it until
    // the assignment is done: we do this for nesting
    Arr temp;
    this->swap(temp);
    
    // temp holds the full rhs, we need to destruct the current Tab
    // (it's just empty() so we can construct into its memory.
    this->~Arr();
    new (this) Arr(rhs);
  }
  return *this; 
}

OC_INLINE ostream& Arr::prettyPrintHelper_ (ostream& os, int indent, 
					    bool pretty, int indent_additive) const
{
  // Base case, empty 
  if (entries()==0) return os << "[ ]"; 

  // Recursive case
  os << "[";
  if (pretty) os << endl;

  // Iterate through, printing out each element
  size_t ent = entries();
  for (size_t ii=0; ii<ent; ++ii) {
    const Val& value = (*this)[ii];
    
    if (pretty) indentOut_(os, indent+indent_additive);
    
    // For most values, use default output method
    switch (value.tag) {
    case 'a': {
      OCString* ap = (OCString*)&value.u.a; 
      os << PyImage(*ap);
      break; 
    }
    case 't': { 
      Tab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'o': { 
      OTab& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'u': { 
      Tup& t = value;
      t.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'n': {
      if (value.subtype=='Z') {
	Arr& arr = value;
	arr.prettyPrintHelper_(os, pretty ? indent+indent_additive : 0, 
			       pretty, indent_additive);
	break;
      } // else fall thru for other array types
    }
    default: os << value; break;
    }
    
    if (entries()>1 && ii!=(entries())-1) os << ","; // commas on all but last
    if (pretty) os << endl;
  }

  if (pretty) indentOut_(os, indent);
  return os << "]";
}

OC_INLINE void Arr::prettyPrint (ostream& os, int starting_indent, 
				 int indent_additive) const
{
  indentOut_(os, starting_indent);
  prettyPrintHelper_(os, starting_indent, true, 
		     indent_additive) << endl;
}

OC_INLINE void Val::swap (Val& rhs)
{ 
  OC_NAMESPACED::swap(this->tag,     rhs.tag);
  OC_NAMESPACED::swap(this->subtype, rhs.subtype);
  OC_NAMESPACED::swap(this->isproxy, rhs.isproxy);
  OC_NAMESPACED::swap(this->a,       rhs.a);
  OC_NAMESPACED::swap(this->u,       rhs.u); 
}

OC_INLINE size_t Val::entries () const
{
  size_t ent = ~size_t(0);
  if (isproxy) {
    Proxy* pp=(Proxy*)&(u.P);
    if (pp->tag=='t') {
      Tab& t = *pp;
      ent = t.entries();
    } else {
      Arr* ap = (Arr*)pp->data_(); // May be different type, layout the same
      ent = ap->length();
    }
  } else {
    if (tag=='t') {
      Tab* tp = (Tab*)&u.t;
      ent = tp->entries();
    } else if (tag=='o') {
      OTab* tp = (OTab*)&u.o;
      ent = tp->entries();
    } else if (tag=='u') {
      Tup* tp = (Tup*)&u.u;
      ent = tp->entries();
    } else if (tag=='n') {  // May be different type, but layout the same
      Arr* ap = (Arr*)&u.n;
      ent = ap->length();
    } else if (tag=='a') {
      OCString* ocp = (OCString*)&u.a;
      ent = ocp->length();
    } 
  }

  if (ent==~size_t(0)) {
    throw logic_error("Can't take length of a non-container "+
		      Stringize(*this));
  }
  return ent;
}


#define OC_CONTAINS(T) { Array<T>&arr=*this; con=arr.contains(key); break; }
OC_INLINE bool Val::contains (const Val& key) const
{
  int con = -17;
  if (tag=='t') {
    Tab& t = *this;
    con = t.contains(key);
  } else if (tag=='o') {
    OTab& t = *this;
    con = t.contains(key);
  } else if (tag=='u') {
    Tup& t = *this;
    con = t.contains(key);
  } else if (tag=='n') {
    switch (subtype) {
    case 's': OC_CONTAINS(int_1);  
    case 'S': OC_CONTAINS(int_u1); 
    case 'i': OC_CONTAINS(int_2);  
    case 'I': OC_CONTAINS(int_u2); 
    case 'l': OC_CONTAINS(int_4);  
    case 'L': OC_CONTAINS(int_u4); 
    case 'x': OC_CONTAINS(int_8);  
    case 'X': OC_CONTAINS(int_u8); 
    case 'b': OC_CONTAINS(bool);   
    case 'f': OC_CONTAINS(real_4); 
    case 'd': OC_CONTAINS(real_8); 
    case 'c': OC_CONTAINS(cx_t<int_1>); 
    case 'C': OC_CONTAINS(cx_t<int_u1>); 
    case 'e': OC_CONTAINS(cx_t<int_2>); 
    case 'E': OC_CONTAINS(cx_t<int_u2>); 
    case 'g': OC_CONTAINS(cx_t<int_4>); 
    case 'G': OC_CONTAINS(cx_t<int_u4>); 
    case 'h': OC_CONTAINS(cx_t<int_8>); 
    case 'H': OC_CONTAINS(cx_t<int_u8>);
    case 'F': OC_CONTAINS(complex_8); 
    case 'D': OC_CONTAINS(complex_16); 
    case 'a': OC_CONTAINS(string); 
    case 't': OC_CONTAINS(Tab); 
    case 'o': OC_CONTAINS(OTab); 
    case 'u': OC_CONTAINS(Tup); 
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': OC_CONTAINS(Val);    
    default:  unknownType_("contains", subtype);
    }
  }

  if (con==-17) {
    throw logic_error("Can't ask for contains of a non-container "+
		      Stringize(*this));
  }
  return bool(con);
}

#define OC_REMOVE(T) { Array<T>&arr=*this; con=arr.remove(key); break; }
OC_INLINE bool Val::remove (const Val& key)
{
  int con = -17;
  if (tag=='t') {
    Tab& t = *this;
    con = t.remove(key);
  } else if (tag=='o') {
    OTab& t = *this;
    con = t.remove(key);
  } else if (tag=='u') {
    Tup& t = *this;
    con = t.remove(key);
  } else if (tag=='n') {
    switch (subtype) {
    case 's': OC_REMOVE(int_1);  
    case 'S': OC_REMOVE(int_u1); 
    case 'i': OC_REMOVE(int_2);  
    case 'I': OC_REMOVE(int_u2); 
    case 'l': OC_REMOVE(int_4);  
    case 'L': OC_REMOVE(int_u4); 
    case 'x': OC_REMOVE(int_8);  
    case 'X': OC_REMOVE(int_u8); 
    case 'b': OC_REMOVE(bool);   
    case 'f': OC_REMOVE(real_4); 
    case 'd': OC_REMOVE(real_8); 
    case 'c': OC_REMOVE(cx_t<int_1>); 
    case 'C': OC_REMOVE(cx_t<int_u1>); 
    case 'e': OC_REMOVE(cx_t<int_2>); 
    case 'E': OC_REMOVE(cx_t<int_u2>); 
    case 'g': OC_REMOVE(cx_t<int_4>); 
    case 'G': OC_REMOVE(cx_t<int_u4>); 
    case 'h': OC_REMOVE(cx_t<int_8>); 
    case 'H': OC_REMOVE(cx_t<int_u8>);
    case 'F': OC_REMOVE(complex_8); 
    case 'D': OC_REMOVE(complex_16); 
    case 'a': OC_REMOVE(string); 
    case 't': OC_REMOVE(Tab); 
    case 'o': OC_REMOVE(OTab); 
    case 'u': OC_REMOVE(Tup); 
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': OC_REMOVE(Val);    
    default:  unknownType_("contains", subtype);
    }
  }

  if (con==-17) {
    throw logic_error("Can't ask for contains of a non-container "+
		      Stringize(*this));
  }
  return bool(con);
}


OC_INLINE void Val::prettyPrint (ostream& os, int starting_indent,
				 int additive_indent) const
{
  if (tag=='t') {
    Tab& t = *this;
    t.prettyPrint(os, starting_indent, additive_indent);
  } else if (tag=='o') {
    OTab& t = *this;
    t.prettyPrint(os, starting_indent, additive_indent);
  } else if (tag=='u') {
    Tup& t = *this;
    t.prettyPrint(os, starting_indent, additive_indent);
  } else if (tag=='n' && subtype=='Z') {
    Arr& a = *this;
    a.prettyPrint(os, starting_indent, additive_indent);
  } else {
    os << (*this);
  }
}


#define OC_APPEND(T) {Array<T>&a=*this;a.append(v); break;}
OC_INLINE void Val::append (const Val& v) 
{ 
  if (tag=='t') {
    Tab& t = *this;
    t.append(v);
  } else if (tag=='o') {
    OTab& t = *this;
    t.append(v);
  } else if (tag=='u') {
    Tup& t = *this;
    t.append(v);
  } else if (tag=='n') {
    switch (subtype) {
    case 's': OC_APPEND(int_1);  
    case 'S': OC_APPEND(int_u1); 
    case 'i': OC_APPEND(int_2);  
    case 'I': OC_APPEND(int_u2); 
    case 'l': OC_APPEND(int_4);  
    case 'L': OC_APPEND(int_u4); 
    case 'x': OC_APPEND(int_8);  
    case 'X': OC_APPEND(int_u8); 
    case 'b': OC_APPEND(bool);   
    case 'f': OC_APPEND(real_4); 
    case 'd': OC_APPEND(real_8); 
    case 'c': OC_APPEND(cx_t<int_1>); 
    case 'C': OC_APPEND(cx_t<int_u1>); 
    case 'e': OC_APPEND(cx_t<int_2>); 
    case 'E': OC_APPEND(cx_t<int_u2>); 
    case 'g': OC_APPEND(cx_t<int_4>); 
    case 'G': OC_APPEND(cx_t<int_u4>); 
    case 'h': OC_APPEND(cx_t<int_8>); 
    case 'H': OC_APPEND(cx_t<int_u8>);
    case 'F': OC_APPEND(complex_8); 
    case 'D': OC_APPEND(complex_16); 
    case 'a': {Array<Str>*ap=(Array<Str>*)&v.u.a;ap->append(v); break;}
    case 't': {Array<Tab>*ap=(Array<Tab>*)&v.u.a;ap->append(v); break;}
    case 'o': {Array<OTab>*ap=(Array<OTab>*)&v.u.o;ap->append(v); break;}
    case 'u': {Array<Tup>*ap=(Array<Tup>*)&v.u.u;ap->append(v); break;}
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': OC_APPEND(Val);    
    default:  unknownType_("append", subtype);
    }
  } else {
    throw logic_error("Only Array<T>, Arr and Tab (and Proxies) can append");
  }
}

OC_INLINE void Val::appendStr (const Val& v)
{
  if (tag=='t') {
    Tab& t = *this;
    t.appendStr(v);
  } else if (tag=='o') {
    OTab& t = *this;
    t.appendStr(v);
  } else {
    append(v); // call the normal append
  }
}


template <class T>
inline void OCHelpProxize_ (Val& v, T* impl, 
			    bool adopt, bool locked)
{
  // Turn into a Proxy by swapping the implementations! 
  T* new_thing = new T();  // TODO: Allocator?
  impl->swap(*new_thing);
  v = Proxy(new_thing, adopt, locked);  // creates proxy for us
}


#define OC_HELPPROX(T) {Array<T>*ap=(Array<T>*)&u.n;OCHelpProxize_(*this,ap,adopt, locked); break; }
OC_INLINE void Val::Proxyize (bool adopt, bool locked)
{
  if (isproxy) return; 
  if (tag=='t') {
    OCHelpProxize_(*this,(Tab*)&u.t, adopt, locked);
  } else if (tag=='o') {
    OCHelpProxize_(*this,(OTab*)&u.o, adopt, locked);
  } else if (tag=='u') {
    OCHelpProxize_(*this,(Tup*)&u.u, adopt, locked);
  } else if (tag=='n') {
    switch (subtype) {
    case 's': OC_HELPPROX(int_1);  
    case 'S': OC_HELPPROX(int_u1); 
    case 'i': OC_HELPPROX(int_2);  
    case 'I': OC_HELPPROX(int_u2); 
    case 'l': OC_HELPPROX(int_4);  
    case 'L': OC_HELPPROX(int_u4); 
    case 'x': OC_HELPPROX(int_8);  
    case 'X': OC_HELPPROX(int_u8); 
    case 'b': OC_HELPPROX(bool);   
    case 'f': OC_HELPPROX(real_4); 
    case 'd': OC_HELPPROX(real_8); 
    case 'c': OC_HELPPROX(cx_t<int_1>); 
    case 'C': OC_HELPPROX(cx_t<int_u1>); 
    case 'e': OC_HELPPROX(cx_t<int_2>); 
    case 'E': OC_HELPPROX(cx_t<int_u2>); 
    case 'g': OC_HELPPROX(cx_t<int_4>); 
    case 'G': OC_HELPPROX(cx_t<int_u4>); 
    case 'h': OC_HELPPROX(cx_t<int_8>); 
    case 'H': OC_HELPPROX(cx_t<int_u8>);
    case 'F': OC_HELPPROX(complex_8); 
    case 'D': OC_HELPPROX(complex_16); 
    case 'a': throw logic_error("Arrays of Strings not currently supported");
    case 't': throw logic_error("Arrays of Tabs not currently supported");
    case 'o': throw logic_error("Arrays of OTabs not currently supported");
    case 'u': throw logic_error("Arrays of Tups not currently supported");
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': OC_HELPPROX(Val);    
    default:  unknownType_("Proxyize", subtype);
    }
  } else {
    throw logic_error("Can't Proxyize the value inside:Proxies don't exist");
  }
}



OC_END_NAMESPACE


// Because of twisted dependencies (Proxy depends on Val depends on
// Tab depends on Val), we include the proxy code LAST here.  This is
// not a problem when you factor into .h and .cc, only when you are in
// a "completely inline" environment.
#if !defined(OC_FACTOR_INTO_H_AND_CC)
# include "ocproxy.cc"
#endif
