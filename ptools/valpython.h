#ifndef VALPYTHON_H_

// These (mostly) inline functions bind Val to Python Serialization.

#include "ochashtable.h" 
#include "ocval.h"
#include "occonvert.h"
#include "ocnumerictools.h"
#include "arraydisposition.h"

PTOOLS_BEGIN_NAMESPACE

// This first part is for Python Depickling

inline Val MakeNone () { return None; }
inline Val MakeBool (int bb) { return Val(bool(bb)); } 

// Make Val ints/floats from strings
inline Val MakeInt4 (const char* ascii_int) 
{ return int_4(Val(ascii_int)); }
inline Val MakeInt8 (const char* long_int) 
{ return int_8(Val(long_int)); }
inline Val MakeIntu8 (const char* long_int) 
{ return int_u8(Val(long_int)); }

inline Val MakeDouble (const char* ascii_float) 
{ return real_8(Val(ascii_float)); }
inline Val MakeString (const char* ascii_string)
{ return Val(ascii_string); }
inline Val MakeString (const char* ascii_string, int len)
{ return Val(string(ascii_string, len)); }

inline int_4 GetInt4 (const Val& val) { int_4 rv=val; return rv; }
inline Val PutInt4 (int_4 i) { return Val(i); }

inline Val MakeComplex (const Val& real, const Val& imag) 
{ real_8 r = real; real_8 i = imag; return complex_8(r, i); }

// List operations
inline Val MakeList () { return Arr(); }
inline void ListAppend (Val& l, const Val& value)
{ Arr& a = l; a.append(value); }

// Table operations
inline Val MakeTable () { return Tab(); }
inline void TableSet (Val& table, const Val& key, const Val& value) 
{ Tab& t = table; t[key] = value; } 

// Tuple operations: Basically a list: only ever seen in REDUCE
// mostly.
inline Val MakeTuple (bool compat) 
{ if (compat) return Arr(); else return Tup(); }
inline void TupleAppend (const Val& tuple, const Val& value)
{ 
  if (tuple.tag=='u') { // Allows both compatibility modes
    Tup& a=tuple; a.impl().append(value); 
  } else {
    Arr& a = tuple; a.append(value); 
  }
}
inline Val TupleGet (const Val& tuple, int ii) 
{ return tuple(ii); } 
inline int TupleLength (Val& tuple) { return tuple.length(); }

// OrderedDict operations 
inline Val MakeOrderedDict (bool compat)
{
  if (compat) {
    return Tab();
  } else {
    return OTab();
  }
}

inline void OrderedDictSet (Val& table, const Val& key, const Val& value) 
{ table[key] = value; } 

// Get the Proxy handle out
inline void* GetProxyHandle (const Val& v)
{
  Proxy& p=v;
  void* ptr_handle = p.handle_;
  return ptr_handle;
}

// String
inline string GetString (const Val& v) 
{ Str s = v; 
 return string(s.data(), s.length()); 
}

// Exceptions
inline void MakeException (const string& err_messg) 
{ throw logic_error(err_messg); }

// Warnings
inline void MakeWarning (const string& err_msg) 
{ cerr << err_msg << endl; }

// Vector
Val MakeVector (const string& typecode, size_t elements,
		const char* mem, size_t bytes);

// These functions are for pickling


inline bool IsNone (const Val& v) { return v.tag=='Z'; }
inline bool IsBool (const Val& v) { return v.tag=='b'; }
inline bool IsNumber (const Val& v)  
{
  char number_typecodes[]   = "sSiIlLxXfdFDqQ";
  char*where = strchr(number_typecodes, v.tag);
  return where!=NULL;
}
inline bool IsString (const Val& v) { return v.tag=='a'; }
inline bool IsVector (const Val& v) { return v.tag=='n' && v.subtype!='Z'; }
inline bool IsTable (const Val& v)  { return v.tag=='t'; }
inline bool IsProxyTable (const Val& v)  { if (IsProxy(v)) { Proxy& p=v; return p.tag=='t'; } return false; }
inline bool IsProxyList (const Val& v)  { if (IsProxy(v)) { Proxy& p=v; return p.tag=='n' && p.subtype=='Z';} return false; }
inline bool IsProxyVector (const Val& v)  { if (IsProxy(v)) { Proxy& p=v; return p.tag=='n' && p.subtype!='Z';} return false; }
inline bool IsProxyTuple (const Val& v)  { if (IsProxy(v)) { Proxy& p=v; return p.tag=='u';} return false; }
inline bool IsProxyOrderedDict (const Val& v)  { if (IsProxy(v)) { Proxy& p=v; return p.tag=='o';} return false; }
inline bool IsList (const Val& v)   { return v.tag=='n' && v.subtype=='Z'; }
inline bool IsTuple (const Val& v)   { return v.tag=='u'; }
inline bool IsOrderedDict (const Val& v)   { return v.tag=='o'; }
inline bool IsBigInt (const Val& v) { return v.tag=='q'; }

inline void Tablify (const Val& src, Val& dest) 
{ OTab& u=src; Tab& t=dest=Tab(); ConvertOTabToTab(u,t); }

inline string TypeOf (const Val& ov) { string s; s=ov.tag; return s; }

inline bool IsTrue (const Val& v) { return v; }

inline int VectorElements (const Val& v)
{ return v.length(); }

#define VALPYTHONARRAYGET(T) {Array<T>&a=v;return a[ii];} 
inline Val VectorGet (const Val& v, int ii)
{ 
  if (v.tag!='n') MakeException("Not a Vector");
  switch (v.subtype) {
  case 's': VALPYTHONARRAYGET(int_1);  // int_1
  case 'S': VALPYTHONARRAYGET(int_u1); // int_u1
  case 'i': VALPYTHONARRAYGET(int_2);  // int_2
  case 'I': VALPYTHONARRAYGET(int_u2); // int_u2
  case 'l': VALPYTHONARRAYGET(int_4);  // int_4
  case 'L': VALPYTHONARRAYGET(int_u4); // int_u4
  case 'x': VALPYTHONARRAYGET(int_8);  // int_8
  case 'X': VALPYTHONARRAYGET(int_u8); // int_u8
  case 'b': VALPYTHONARRAYGET(bool);   // bool
  case 'f': VALPYTHONARRAYGET(real_4); // real_4
  case 'd': VALPYTHONARRAYGET(real_8); // real_8
  case 'F': VALPYTHONARRAYGET(complex_8); // complex_8
  case 'D': VALPYTHONARRAYGET(complex_16); // complex_16
  case 'a': {Array<Str>*ap=(Array<Str>*)&v.u.n;return (*ap)[ii];}//ascii strigs
  case 't': {Array<Tab>*ap=(Array<Tab>*)&v.u.n;return (*ap)[ii];}//Tab
  case 'n': MakeException("Arrays of Arrays not supported");
  case 'Z': VALPYTHONARRAYGET(Val);    // array of None (tag for flag)
  default:  string s; s=v.subtype; MakeException("Unknown Type Tag:"+s);
  }
  return None;
}

inline int ListElements (const Val& v)    { return v.length(); } 
inline Val ListGet (const Val& v, int ii) { return v(ii); }

class TableIterator {
 public:
  TableIterator (const Val& ot) :
    it_(ot)
  { }
  bool operator() () { return it_(); } 
  Val key () const { return it_.key(); }
  Val value () const { return it_.value(); }
 protected:
  It it_;
}; // TableIterator

 


// Return the python Numeric typecode for this vector
inline string BestFitForVector (const Val& v)
{
  if (v.tag!='n') MakeException("Not a Vector");
  switch(v.subtype) {
  case 's': // int_1
  case 'S': // int_u1
  case 'i': // int_2
  case 'I': // int_u2
  case 'l': // int_4
    return "l"; break;
  case 'f': // real_4
  case 'd': // real_8
    return "d"; break;
  case 'F' : // complex_8
    return "F"; break;
  case 'D' : // complex_16
    return "D"; break;
  }
  // Want to do this, but because of issues with 'l' in Numeric
  // on 32-bit and 64-bit, we stay the way we were for backwards 
  // compatibilibily.
  //try {
  //  char t = OCTagToNumeric(v.subtype);
  //  string s; s=t; return s;
  //} catch (const exception& e) {
  //}
  string string_type; string_type = v.subtype;
  MakeException("Don't know how serialize Vector of type:"+string_type);
  return "";
}

char ChooseNumber (const Val& ov, string& arg1, string& arg2);
string BuildVector (const Val& ov, const string& python_typecode);

inline void Proxyize (Val& obj) { obj.Proxyize(); }

PTOOLS_END_NAMESPACE


#define MIDAS_COMPILER_TEMPLATES
#include "m2pythonpickler.h"

PTOOLS_BEGIN_NAMESPACE
void UnknownType (PythonPicklerA<Val>& pickler, const Val& v);


void PickleIntoBuffer (const Val& v, Array<char>& array_to_dump_to, 
		       bool uses_numeric=false);
Val DepickleFromBuffer (int buff_len, char* buff, bool uses_numeric=false);

string OBJToNumPyCode (const Val& obj);

string BuildNumPyVector (const  Val& v, const string& endian);

PTOOLS_END_NAMESPACE

#define VALPYTHON_H_
#endif // VALPYTHON_H_
