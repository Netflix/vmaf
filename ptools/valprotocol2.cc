
#include "valprotocol2.h"
#include <stdint.h>

PTOOLS_BEGIN_NAMESPACE

// Little helper function for error reporting (so errors centralized)
inline uintptr_t p2error_ (string s) 
{
  throw runtime_error(s.c_str());
  return 0;
} 

PTOOLS_END_NAMESPACE

#include "p2common.h"


PTOOLS_BEGIN_NAMESPACE

// Forwards
inline void P2DumpValue (const Val& v, DumpContext_& dc);
inline void P2DumpArr (const Arr& a, DumpContext_& dc, void* memoize_self);
inline void P2DumpTab (const Tab& a, DumpContext_& dc, void* memoize_self);
inline void P2DumpOTab (const OTab& a, DumpContext_& dc, void* memoize_self);
inline void P2DumpTup (const Tup& a, DumpContext_& dc, void* memoize_self);
inline void P2DumpBigInt (const int_n& a, DumpContext_& dc, void* memoize_self);
inline void P2DumpProxy (const Proxy& p, DumpContext_& dc);

inline size_t BytesToDumpVal (const Val& t, DumpContext_& dc);
inline size_t BytesToDumpTab (const Tab& t, DumpContext_& dc, void* memoize_self);
inline size_t BytesToDumpOTab (const OTab& t, DumpContext_& dc, void* memoize_self);
inline size_t BytesToDumpTup (const Tup& t, DumpContext_& dc, void* memoize_self);
inline size_t BytesToDumpBigInt (const int_n& t, DumpContext_& dc, void* memoize_self);
inline size_t BytesToDumpArr (const Arr& t, DumpContext_& dc, void* memoize_self);
inline size_t BytesToDumpProxy (const Proxy& t, DumpContext_& dc);

inline bool P2LoadValue (Val& v, LoadContext_& lc);


char* P2TopLevelDumpVal (const Val& v, char* mem, 
			 ArrayDisposition_e dis,
			 PicklingIssues_e issues)
{ return topleveldump_(v, mem, dis, issues); }

char* P2DumpVal (const Val& v, char* mem, 
		 ArrayDisposition_e dis,
		 PicklingIssues_e issues)
{ return P2DumpSingleVal(v, mem, dis, issues); }

size_t P2TopLevelBytesToDumpVal (const Val& v, ArrayDisposition_e dis,
			      PicklingIssues_e issues)
{ return TopLevelBytesToDump(v, dis, issues); }

size_t P2BytesToDumpVal (const Val& v, ArrayDisposition_e dis,
			     PicklingIssues_e issues)
{ return P2BytesToDumpSingleVal(v, dis, issues); }


char* P2TopLevelLoadVal (Val& v, char* mem)
{ return toplevelload_(v, mem); }

char* P2LoadVal (Val& v, char* mem)
{ return P2LoadSingleVal(v, mem); }


// When a dumper is ready to memoize itself (it knows where it needs
// to do the memo), it creates a new handle and inserts it into the table
// then places the the PUT on
void MemoizeSelf_ (void* ptr_handle, DumpContext_& dc)
{
  int_u4 int_handle = dc.current_handle++;
  dc.handles[ptr_handle] = int_handle;
  P2DumpCodeAndInt_(int_handle, PY_BINPUT, PY_LONG_BINPUT, dc);
}

int_u4 BytesToMemoizeSelf_ (void* ptr_handle, DumpContext_& dc)
{
  int_u4 int_handle = dc.current_handle++;
  dc.handles[ptr_handle] = int_handle;
  return (int_handle<256) ? 2 : 5;
}

#define P2PLAINARRAYDUMP(T, FUN) { Array<T>&ap=*(Array<T>*)arr_data; size_t len=ap.length(); T*d=ap.data(); for (size_t ii=0;ii<len;ii++) { FUN(d[ii], dc); } }
#define P2PLAINARRAYDUMP2(T, FUN, ARG) { Array<T>&ap=*(Array<T>*)arr_data; size_t len=ap.length(); T*d=ap.data(); for (size_t ii=0;ii<len;ii++) { FUN(d[ii], dc, ARG); } }
// The user doesn't have Numeric installed and Array is an older version
// of python that doesn't work (!), so we have to be able to give them
// back something:  an Array of Val.
inline void dumpArrayAsList_ (void* arr_data, char subtype, DumpContext_& dc, 
			      void* memoize_self)
{
  // Assume v.tag = 'n'

  // dumpArr does the mark list already, and is usual case anyway
  if (subtype=='Z') {
    Arr*ap=(Arr*)arr_data;
    P2DumpArr(*ap, dc, memoize_self);
    return;
  }

  // Other arrays ...
  Arr*aaap=(Arr*)arr_data;
  size_t length = aaap->length();
  *(dc.mem)++ = PY_EMPTY_LIST;
  if (memoize_self) MemoizeSelf_(memoize_self, dc);
  if (length==0) return;
  if (length!=1) *(dc.mem)++ = '(';

  switch (subtype) {
  case 's': P2PLAINARRAYDUMP(int_1,     dump4ByteInteger);  break;
  case 'S': P2PLAINARRAYDUMP(int_u1,    dump4ByteInteger);  break;
  case 'i': P2PLAINARRAYDUMP(int_2,     dump4ByteInteger);  break;
  case 'I': P2PLAINARRAYDUMP(int_u2,    dump4ByteInteger);  break;
  case 'l': P2PLAINARRAYDUMP(int_4,     dump4ByteInteger);  break;
  case 'L': P2PLAINARRAYDUMP(int_u4,    dump8ByteUnsignedInteger);  break;
  case 'x': P2PLAINARRAYDUMP(int_8,     dump8ByteInteger);  break;
  case 'X': P2PLAINARRAYDUMP(int_u8,    dump8ByteUnsignedInteger);  break;
  case 'b': P2PLAINARRAYDUMP(bool,      dumpBool);          break;
  case 'f': P2PLAINARRAYDUMP(real_4,    dump8ByteFloat);    break;
  case 'd': P2PLAINARRAYDUMP(real_8,    dump8ByteFloat);    break;
  case 'F': P2PLAINARRAYDUMP(complex_8, dump16ByteComplex); break;
  case 'D': P2PLAINARRAYDUMP(complex_16,dump16ByteComplex); break;
  case 'a': P2PLAINARRAYDUMP(Str,       dumpString);        break;
  case 't': P2PLAINARRAYDUMP2(Tab, P2DumpTab, 0); break;
  case 'o': P2PLAINARRAYDUMP2(OTab, P2DumpOTab, 0); break;
  case 'u': P2PLAINARRAYDUMP2(Tup, P2DumpTup, 0); break;
  case 'n': throw logic_error("Arrays of Arrays not currently supported");
  case 'Z': // { Arr*ap=(Arr*)arr_data; mem = P2DumpArr(*ap, mem) ; break; }
  default: p2error_("unknown type in dumpArrayAsList_");
  }
  
  *(dc.mem)++ = (length==1) ? PY_APPEND : PY_APPENDS;
}

inline void dumpArrayAsList_ (const Val& v, DumpContext_& dc)
{
  dumpArrayAsList_((void*)&v.u.n, v.subtype, dc, 0);
}


size_t BytesToDumpArrayAsList (void* arr_data, char subtype, DumpContext_& dc,
			       void* memoize_self)
{
  if (subtype=='Z') {
    Arr*ap=(Arr*)arr_data;
    return BytesToDumpArr(*ap, dc, memoize_self);
  }

  size_t bytes = 1 + 1 + 1;
  if (memoize_self) bytes += BytesToMemoizeSelf_(memoize_self, dc);
  Arr*ap = (Arr*)arr_data;//&(v.u.n);
  const size_t len = ap->length();
  size_t element_size = 0;
  switch (subtype) {
  case 's': 
  case 'S': 
  case 'i': 
  case 'I': 
  case 'l': element_size = BytesToDump4ByteInteger(); break;
  case 'L': element_size = BytesToDump8ByteUnsignedInteger(dc);  break;
  case 'x': element_size = BytesToDump8ByteInteger(dc);  break;
  case 'X': element_size = BytesToDump8ByteUnsignedInteger(dc);  break;
  case 'b': element_size = BytesToDumpBool();          break;
  case 'f': 
  case 'd': element_size = BytesToDump8ByteFloat();    break;
  case 'F': 
  case 'D': element_size = BytesToDump16ByteComplex(dc); break;
  case 'a': {
    string* data = (string*)ap->data();
    for (size_t ii=0; ii<len; ii++) {
      bytes += BytesToDumpString(data[ii]);
    }
    return bytes;
  }

    //#error 666 // I don't think this code has ever been tested!
  case 't': {
    Tab* data = (Tab*)ap->data();
    for (size_t ii=0; ii<len; ii++) {
      bytes += BytesToDumpTab(data[ii], dc,0);
    }
    return bytes;
  }
  case 'o': {
    OTab* data = (OTab*)ap->data();
    for (size_t ii=0; ii<len; ii++) {
      bytes += BytesToDumpOTab(data[ii], dc,0);
    }
    return bytes;
  }
  case 'u': {
    Tup* data = (Tup*)ap->data();
    for (size_t ii=0; ii<len; ii++) {
      bytes += BytesToDumpTup(data[ii], dc,0);
    }
    return bytes;
  }
    //#error 666 // I don't think this code has ever been tested!

  case 'n': throw logic_error("Arrays of Arrays not currently supported");
  // case 'Z': return BytesToDumpArr(*ap, dc, memoize_self) ; break; 
  default: p2error_("unknown type in BytesToDumpArrayAsList");
  }
  bytes += element_size*len;
  return bytes;
}

inline size_t BytesToDumpArrayAsList (const Val& v, DumpContext_& dc)
{
  return BytesToDumpArrayAsList((void*)&v.u.n, v.subtype, dc, 0);
}


// This is a newer feature, allowing us to dump arrays as Python
// package array, but it is broken in some version of Pythons, so we
// have to make sure it is a newer version of Python
inline void dumpArray_ (void* arr_data, char subtype, DumpContext_& dc,
			void* memoize_self)
{
  // Assume v is an Array, figure out what its Python typecode will be
  int sz = -1;
  const char* c = " "; // some value
  switch (subtype) {
  case 's': sz=sizeof(int_1);  c="c"; break;
  case 'S': sz=sizeof(int_u1); c="B"; break;
  case 'i': sz=sizeof(int_2);  c="h"; break;
  case 'I': sz=sizeof(int_u2); c="H"; break;
  case 'l': sz=sizeof(int_4);  c="i"; break;
  case 'L': sz=sizeof(int_u4); c="I"; break;
  case 'x': sz=sizeof(int_8);  c="l"; break;
  case 'X': sz=sizeof(int_u8); c="L"; break;
  case 'b': sz=sizeof(bool);   c="b"; break;
  case 'f': sz=sizeof(real_4); c="f"; break;
  case 'd': sz=sizeof(real_8); c="d"; break;
  case 'F': sz=sizeof(complex_8);  c="F"; break;
  case 'D': sz=sizeof(complex_16); c="D"; break;
  }

  if (sz != -1) { // THIS IS A POD Array!  can bit blit

    // Dump the memoize to make it faster
    if (dc.array_preamble_dumped) {
      P2DumpCodeAndInt_(dc.array_handle, PY_BINGET, PY_LONG_BINGET, dc);
    } else {
      // Not dumped, figure out what handle should be
      dc.array_handle = dc.current_handle++;
      dc.array_preamble_dumped = true;

      // Then dump it (with memo to annotate it)
      memcpy(dc.mem, ArrayPreamble, sizeof(ArrayPreamble)-1);
      dc.mem += sizeof(ArrayPreamble)-1;
      P2DumpCodeAndInt_(dc.array_handle, PY_BINPUT, PY_LONG_BINPUT, dc);
    }

    // Same layout, regardless of type.
    // TODO:  Will we have to reendiaze this?
    Arr*ap=(Arr*)arr_data; 
    char* dat = (char*)ap->data();

    // Dump the format before the data
    dumpCString(c, 1, dc);
    dumpCString(dat, ap->length()*sz, dc);
    *(dc.mem)++ = PY_TUPLE2;
    *(dc.mem)++ = PY_REDUCE;
    if (memoize_self)  MemoizeSelf_(memoize_self, dc); 
    return;
  }

  // If it's not a numeric or POD type, we still have to dump
  // as an Array of Vals, lame but will work
  dumpArrayAsList_(arr_data, subtype, dc, memoize_self);
}

inline void dumpArray_ (const Val& v, DumpContext_& dc)
{
  dumpArray_((void*)&v.u.n, v.subtype, dc, 0);
}

inline size_t BytesToDumpArray (void* arr_data, char subtype, DumpContext_& dc,
				void* memoize_self)
{
  // Assume v is an Array: If it's not a numeric or POD type, we
  // still have to dump as an Array of Vals, lame but will work
  switch (subtype) {
  case 'Z' :
  case 'a': case 't': case 'o': case 'u': case 'n': 
    return BytesToDumpArrayAsList(arr_data, subtype, dc, memoize_self);
  }

  Arr*ap=(Arr*)arr_data;
  size_t bytes = ByteLength(subtype) * ap->length() + 26;
  if (dc.array_preamble_dumped) {
    bytes += 2;
  } else {
    dc.array_handle = dc.current_handle++;
    dc.array_preamble_dumped = true;
    bytes += sizeof(ArrayPreamble)+2;
  }
  if (memoize_self) bytes += BytesToMemoizeSelf_(memoize_self, dc);
  return bytes;
}

inline size_t BytesToDumpArray (const Val& v, DumpContext_& dc)
{
  return BytesToDumpArray((void*)&v.u.n, v.subtype, dc, 0);
}

// Helper dunction so we can debug dumps to memory
template <class PT, class T>
void dumpNumericHelp_(PT*, T*, const char*c, DumpContext_& dc, void* arr_data)
{
  dumpCString(c, 1, dc); // Type string

  // Get necessary data
  Array<T>*oa=((Array<T>*)arr_data);//((Array<T>*)(&v.u.n));
  T* od = oa->data();
  size_t elements = oa->length();

  // Dump string header
  size_t len = elements*sizeof(PT);
  P2DumpCodeAndInt_(len, PY_SHORT_BINSTRING, PY_BINSTRING, dc);

  // Dump actual data for string
  PT* sm=(PT*)dc.mem;
  for (size_t ii=0; ii<elements; ii++) {
    sm[ii] = od[ii];
  }
  dc.mem += len;
}

// If someone is using XMPY, they may want their Numeric Arrays
void dumpNumericArray_ (void* arr_data, char subtype, DumpContext_& dc,
			void* memoize_self)
{
  // Assume v is an Array: If it's not a numeric or POD type, we
  // still have to dump as an Array of Vals, lame but will work
  switch (subtype) {
  case 'Z':
  case 'a': case 't': case 'o': case 'u': case 'n': 
    dumpArrayAsList_(arr_data, subtype, dc, memoize_self); return; break;
  }

  // THIS IS A POD Array!  can bit blit

  // If already memoized, just dump the memo for faster serialization
  if (dc.numeric_preamble_dumped) {
    P2DumpCodeAndInt_(dc.numeric_handle, PY_BINGET, PY_LONG_BINGET, dc);
  } else {
    // Get new handle
    dc.numeric_handle = dc.current_handle++;
    dc.numeric_preamble_dumped = true;

    // Dump the original data
    memcpy(dc.mem, NumericPreamble, sizeof(NumericPreamble)-1);
    dc.mem += sizeof(NumericPreamble)-1;
    P2DumpCodeAndInt_(dc.numeric_handle, PY_BINPUT, PY_LONG_BINPUT, dc);
  } 


  // Same layout, regardless of type.
  // TODO:  Will we have to reendiaze this?
  Arr*ap=(Arr*)arr_data;

  // Dump (, then length of array, then tuple
  *(dc.mem)++ = '(';
  if (dc.pickling_issues==AS_PYTHON_2_2) {
    *(dc.mem)++ = '(';
  }
  size_t elements = ap->length();
  dump4ByteInteger(elements, dc);
  *(dc.mem)++ = (dc.pickling_issues==AS_PYTHON_2_2) ? 't' : PY_TUPLE1;

  // Dump string with typecode, then dump as data
  const char *c;
  switch (subtype) {
  case 's': c="1"; dumpNumericHelp_((int_1*)0,  (int_1*)0, c, dc, arr_data);  break;
  case 'S': c="b"; dumpNumericHelp_((int_u1*)0, (int_u1*)0, c, dc, arr_data);  break;
  case 'i': c="s"; dumpNumericHelp_((int_2*)0,  (int_2*)0, c, dc, arr_data);  break;
  case 'I': c="w"; dumpNumericHelp_((int_u2*)0, (int_u2*)0, c, dc, arr_data);  break;
  case 'l': c="i"; dumpNumericHelp_((int_4*)0,  (int_4*)0, c, dc, arr_data); break;
  case 'L': c="u"; dumpNumericHelp_((int_u4*)0, (int_u4*)0, c, dc, arr_data); break;

    // Take care with these ... longs and int_8s not necessarily the same
  case 'x': c="l"; dumpNumericHelp_((long*)0,  (int_8*)0, c, dc, arr_data); break;
  case 'X': c="l"; dumpNumericHelp_((long*)0, (int_u8*)0, c, dc, arr_data); break;

  case 'b': c="b"; dumpNumericHelp_((bool*)0, (bool*)0, c, dc, arr_data);   break;
  case 'f': c="f"; dumpNumericHelp_((real_4*)0, (real_4*)0, c, dc, arr_data); break;
  case 'd': c="d"; dumpNumericHelp_((real_8*)0, (real_8*)0, c, dc, arr_data); break;  
  case 'F': c="F"; dumpNumericHelp_((complex_8*)0, (complex_8*)0, c, dc, arr_data); break;
  case 'D': c="D"; dumpNumericHelp_((complex_16*)0, (complex_16*)0, c, dc, arr_data); break;
  default: p2error_("Can't use type in dumpOpalNumericArray");
  }

  if (dc.pickling_issues==AS_PYTHON_2_2) {
    dump4ByteInteger(1, dc);
  } else { 
    *(dc.mem)++ = PY_NEWTRUE;
  }

  *(dc.mem)++ = 't';
  *(dc.mem)++ = PY_REDUCE;
  if (memoize_self) { MemoizeSelf_(memoize_self, dc); }
}

void dumpNumericArray_ (const Val& v, DumpContext_& dc)
{
  dumpNumericArray_((void*)&v.u.n, v.subtype, dc, 0);
}

inline size_t BytesToDumpNumericArray (void* arr_data, char subtype, 
				       DumpContext_& dc, void* memoize_self)
{
  // Assume v is an Array: If it's not a numeric or POD type, we
  // still have to dump as an Array of Vals, lame but will work
  switch (subtype) {
  case 'Z':
  case 'a': case 't': case 'o': case 'u': case 'n': 
    return BytesToDumpArrayAsList(arr_data, subtype, dc, memoize_self);
  }

  Arr*ap=(Arr*)arr_data;
  size_t bytes=ByteLength(subtype)*ap->length()+14+2*BytesToDump4ByteInteger();
  if (dc.numeric_preamble_dumped) {
    bytes += 2;
  } else {
    dc.numeric_handle = dc.current_handle++;
    dc.numeric_preamble_dumped = true;
    bytes += sizeof(NumericPreamble)+2;
  }
  if (memoize_self) 
    bytes += BytesToMemoizeSelf_(memoize_self, dc);
  return bytes;
}

inline size_t BytesToDumpNumericArray (const Val& v, DumpContext_& dc)
{
  return BytesToDumpNumericArray((void*)&v.u.n, v.subtype, dc, 0);
}

// //////////////////////// NumPy dumping

// Helper: convert Val type codes into NumPy type strings
string ValToNumPyCode (char subtype)
{
  switch (subtype) {
  case 's': return "i1";
  case 'S': return "u1";
  case 'i': return "i2";
  case 'I': return "u2";
  case 'l': return "i4";
  case 'L': return "u4";
  case 'x': return "i8";
  case 'X': return "u8";
  case 'f': return "f4";
  case 'd': return "f8";
  case 'F': return "c8";
  case 'D': return "c16";
  case 'b': return "b1";
  
  default: {
    string ty; ty=subtype; throw runtime_error("Cannot handle arrays of "+ty);
  }
  }
}

// If someone is using XMPY, they may want their Numeric Arrays
void dumpNumPyArray_ (void* arr_data, char subtype, DumpContext_& dc,
		      void* memoize_self)
{
  // Assume v is an Array: If it's not a NumPy or POD type, we
  // still have to dump as an Array of Vals, lame but will work
  switch (subtype) {
  case 'Z':
  case 'a': case 't': case 'o': case 'u': case 'n': 
    dumpArrayAsList_(arr_data, subtype, dc, memoize_self); return; break;
  }

  // Assertion: raw data to dump

  // layout same, regardless type
  Array<char>* ap = (Array<char>*)arr_data;  
  size_t shape = ap->length();

  // PY_GLOBAL reconstruct ...  
  PreambleDumperNumPyReconstruct(dc);

  // Initial arguments to a "prototype" NDArray
  PreambleDumperNumPyNDArray(dc); // 1: GLOBAL
  P2DumpTup(Tup(Val(0)), dc, 0);  // 2: initial unused shape
  dumpCString("b",1,dc);          // 3: initial unused type
  *(dc.mem++) = PY_TUPLE3;        // (1,2,3)
  *(dc.mem)++ = PY_REDUCE;

  // Starting argumements for "prototype" multiarray BUILD
  {
    *(dc.mem)++ = PY_MARK;

    P2DumpValue(1, dc);
    P2DumpValue(Tup(shape), dc);

    // Starting DTYPE
    {
      // Initial args to a "prototype" dtype
      PreambleDumperNumPyDtype(dc);
      string numpy_code = ValToNumPyCode(subtype);
      Tup dtype_initial(numpy_code, 0, 1);
      P2DumpValue(dtype_initial, dc);
      *(dc.mem)++ = PY_REDUCE;

      // Tuple of arguments that get applied to "prototype" dtype BUILD
      string endian = ByteLength(subtype)==1 ? "|" : IsLittleEndian() ? "<" : ">";
      Tup dtype_args(3, endian, None, None, None, -1, -1, 0);
      P2DumpValue(dtype_args, dc);
      *(dc.mem)++ = PY_BUILD;
    }
    // Assertion: Dtype top thing on values stack

    *(dc.mem)++ = PY_NEWFALSE;

    // Dump the actual data
    const char* raw_data = ap->data();
    dumpCString(raw_data, ByteLength(subtype)*ap->length(), dc);
    
    *(dc.mem)++ = PY_TUPLE;
  }
  *(dc.mem)++ = PY_BUILD;
  if (memoize_self) { MemoizeSelf_(memoize_self, dc); }
}

void dumpNumPyArray_ (const Val& v, DumpContext_& dc)
{
  dumpNumPyArray_((void*)&v.u.n, v.subtype, dc, 0);
}


inline size_t BytesToDumpNumPyArray (void* arr_data, char subtype, 
				     DumpContext_& dc, void* memoize_self)
{
  // Assume v is an Array: If it's not a numeric or POD type, we
  // still have to dump as an Array of Vals, lame but will work
  switch (subtype) {
  case 'Z':
  case 'a': case 't': case 'o': case 'u': case 'n': 
    return BytesToDumpArrayAsList(arr_data, subtype, dc, memoize_self);
  }

  // Otherwise, POD data
  Arr*ap=(Arr*)arr_data;
  size_t bytes=ByteLength(subtype)*ap->length()+50+2*BytesToDump4ByteInteger();
  bytes += BytesPreambleNumPyReconstruct(dc);
  bytes += BytesPreambleNumPyNDArray(dc);
  bytes += BytesPreambleNumPyDtype(dc);
  if (memoize_self) 
    bytes += BytesToMemoizeSelf_(memoize_self, dc);
  return bytes;
}

inline size_t BytesToDumpNumPyArray (const Val& v, DumpContext_& dc)
{
  return BytesToDumpNumPyArray((void*)&v.u.n, v.subtype, dc, 0);
}


template <class T>
inline void DumpTableHelper_ (const T& t, DumpContext_& dc, void* memoize_self)
{
  const size_t len = t.entries();
  *(dc.mem)++ = PY_EMPTY_DICT;
  if (memoize_self) MemoizeSelf_(memoize_self, dc); // Memo JUST AFTER {

  if (len!=0)  { // Empty Tab just single }

    // Otherwise, some work has to be done
    if (len!=1) *(dc.mem)++ = '(';

    for (It ii(t); ii();) {
      const Val& key = ii.key();
      const Val& value = ii.value();
      P2DumpValue(key, dc);
      P2DumpValue(value, dc);
    }

    *(dc.mem)++ = (len==1) ? PY_SETITEM : PY_SETITEMS;
  }
}

// Dump a Tab as a Python Dictionary (pickling protocol 2)
inline void P2DumpTab (const Tab& t, DumpContext_& dc, void* memoize_self)
{
  DumpTableHelper_(t, dc, memoize_self);
}

template <class T>
inline size_t BytesToDumpTable_ (const T& t, DumpContext_& dc, void* memoize_self)
{
  size_t bytes = 1 + 1 + 1; // EMPTY_DICT, ( SETITEMS
  if (memoize_self) bytes += BytesToMemoizeSelf_(memoize_self, dc);

  for (It ii(t); ii();) {
    const Val& key = ii.key();
    const Val& value = ii.value();
    bytes += BytesToDumpVal(key, dc);
    bytes += BytesToDumpVal(value, dc);
  }
  return bytes;
}

inline size_t BytesToDumpTab (const Tab& t, DumpContext_& dc, void* memoize_self)
{ return BytesToDumpTable_(t, dc, memoize_self); }


// Dump OTab as a Python OrderedDict 
inline void P2DumpOTab (const OTab& t, DumpContext_& dc, void* memoize_self)
{
  if (dc.compat_) {
    DumpTableHelper_(t, dc, memoize_self);
    return;
  }

  // Preamble:
  // If already memoized, just dump the memo for faster serialization
  if (dc.ordereddict_preamble_dumped) {
    P2DumpCodeAndInt_(dc.ordereddict_handle, PY_BINGET, PY_LONG_BINGET, dc);
  } else {
    // Get new handle
    dc.ordereddict_handle = dc.current_handle++;
    dc.ordereddict_preamble_dumped = true;

    // Dump the original data
    memcpy(dc.mem, OrderedDictPreamble, sizeof(OrderedDictPreamble)-1);
    dc.mem += sizeof(OrderedDictPreamble)-1;
    P2DumpCodeAndInt_(dc.ordereddict_handle, PY_BINPUT, PY_LONG_BINPUT, dc);
  } 

  const size_t len = t.entries();
  *(dc.mem)++ = PY_EMPTY_LIST;


  if (len!=0)  { // Empty OTab just single ]

    // Otherwise, some work has to be done
    if (len!=1) *(dc.mem)++ = '(';

    for (It ii(t); ii();) {
      *(dc.mem)++ = PY_EMPTY_LIST;  // This is the tuple of key-value
      *(dc.mem)++ = PY_MARK;

      const Val& key = ii.key();
      const Val& value = ii.value();
      P2DumpValue(key, dc);
      P2DumpValue(value, dc);

      *(dc.mem)++ = PY_APPENDS;     // Append two items of tuple
    }
    
    *(dc.mem)++ = (len==1) ? PY_APPEND : PY_APPENDS;
  }

  *(dc.mem)++ = PY_TUPLE1;
  *(dc.mem)++ = PY_REDUCE;
  if (memoize_self) MemoizeSelf_(memoize_self, dc); // Memo JUST AFTER Reduce
}

inline size_t BytesToDumpOTab (const OTab& t, DumpContext_& dc,void* memoize_self)
{
  if (dc.compat_) {
    return BytesToDumpTable_(t, dc, memoize_self);
  }

  size_t bytes = 1; // ]
  if (dc.ordereddict_preamble_dumped) {
    bytes += 2;
  } else {
    dc.ordereddict_handle = dc.current_handle++;
    dc.ordereddict_preamble_dumped = true;
    bytes += sizeof(OrderedDictPreamble)+2; // 2 for q\x01
  }

  for (It ii(t); ii();) {
    bytes += 4;  // [, mark, mark, append
    const Val& key = ii.key();
    const Val& value = ii.value();
    bytes += BytesToDumpVal(key, dc);
    bytes += BytesToDumpVal(value, dc);
  }

  bytes += 1+1; // \x85 R
  if (memoize_self) bytes += BytesToMemoizeSelf_(memoize_self, dc);
  return bytes;
}


// Dump a Tuple (pickling protocol 2)
inline void P2DumpTup (const Tup& u, DumpContext_& dc, void* memoize_self)
{
  if (dc.compat_) { // Tup is an Arr for backwards compatibility
    P2DumpArr((Arr&)u.impl(), dc, memoize_self);
    return;
  }

  const size_t len = u.entries();

  if (len==0) {          // Empty tuple, just a mark
    *(dc.mem)++ = PY_EMPTY_TUPLE;
  } else if (len==1) {   // Special mark
    P2DumpValue(u(0), dc);
    *(dc.mem)++ = PY_TUPLE1;
  } else if (len==2) {
    P2DumpValue(u(0), dc);
    P2DumpValue(u(1), dc);
    *(dc.mem)++ = PY_TUPLE2;
  } else if (len==3) {
    P2DumpValue(u(0), dc);
    P2DumpValue(u(1), dc);
    P2DumpValue(u(2), dc);
    *(dc.mem)++ = PY_TUPLE3;
  } else {
    // General case
    *(dc.mem)++ = PY_MARK;
    for (size_t ii=0; ii<len; ii++) {
      P2DumpValue(u(ii), dc);
    }
    *(dc.mem)++ = PY_TUPLE;
  }
  if (memoize_self) MemoizeSelf_(memoize_self, dc);
}

inline size_t BytesToDumpTup (const Tup& u, DumpContext_& dc, void* memoize_self)
{
  if (dc.compat_) { // Tup is an Arr for backwards compatibility
    return BytesToDumpArr((Arr&)u.impl(), dc, memoize_self);
  }

  size_t bytes = 0;
  const size_t len = u.entries();
  if (len==0) {          // Empty tuple, just a mark
    bytes += 1;
  } else if (len==1) {   // Special mark
    bytes += BytesToDumpVal(u(0), dc) + 1;
  } else if (len==2) {
    bytes += BytesToDumpVal(u(0), dc);
    bytes += BytesToDumpVal(u(1), dc);
    bytes += 1;
  } else if (len==3) {
    bytes += BytesToDumpVal(u(0), dc);
    bytes += BytesToDumpVal(u(1), dc);
    bytes += BytesToDumpVal(u(2), dc);
    bytes += 1;
  } else {
    // General case
    bytes += 1;
    for (size_t ii=0; ii<len; ii++) {
      bytes += BytesToDumpVal(u(ii), dc);
    }
    bytes += 1;
  }
  if (memoize_self) bytes += BytesToMemoizeSelf_(memoize_self, dc);
  return bytes;
}

inline void P2DumpBigInt (const int_n& i, DumpContext_& dc, void* memoize_self)
{
  if (dc.compat_) {
    string s = i.stringize();
    dumpString(s, dc);
    return;
  }

  string s = MakeBinaryFromBigInt(i);
  size_t len = s.size();
  if (len > 255) {
    *(dc.mem)++ = PY_LONG4;
    dump4ByteInteger(len,dc);  
  } else {
    *(dc.mem)++ = PY_LONG1;
    int_u1 small_len = int_u1(len);
    *(dc.mem)++ = small_len;
  }
  memcpy(dc.mem, s.data(), len);
  dc.mem += len;
  if (memoize_self) MemoizeSelf_(memoize_self, dc);
}

inline size_t BytesToDumpBigInt (const int_n& u, DumpContext_& dc, void*memoize_self)
{
  if (dc.compat_) {
    string s = u.stringize();
    return BytesToDumpString(s);
  }

  size_t bytes = 5; // \x8a char  or \x8b 4 chars  ... overestimate
  bytes += u.bytes();
  if (memoize_self) bytes += BytesToMemoizeSelf_(memoize_self, dc);
  return bytes;
}

// Dump a Arr as a Python List (pickling protocol 2)
inline void P2DumpArr (const Arr& a, DumpContext_& dc, void* memoize_self)
{
  const size_t len = a.entries();
  *(dc.mem)++ = PY_EMPTY_LIST;
  if (memoize_self) MemoizeSelf_(memoize_self, dc);

  if (len!=0) { // Empty List just single ]

    // Otherwise, some work has to be done
    if (len!=1) *(dc.mem)++ = '(';
    for (size_t ii=0; ii<len; ii++) {
      P2DumpValue(a[ii], dc);
    }

    *(dc.mem)++ = (len==1) ? PY_APPEND : PY_APPENDS;
  }
}

inline size_t BytesToDumpArr (const Arr& ca, DumpContext_& dc, void* memoize_self)
{
  size_t bytes = 1 + 1 + 1; // PY_EMPTY_LIST, ( PY_APPENDS
  if (memoize_self) bytes += BytesToMemoizeSelf_(memoize_self, dc);

  Arr* ap = (Arr*)&ca;
  Val* data = ap->data();
  const size_t len = ap->length();
  for (size_t ii=0; ii<len; ii++) {
    bytes += BytesToDumpVal(data[ii], dc);
  }
  return bytes;
}


// Dump a Proxy: Dump what it points to "normally", then label it with
// an int_handle, so it comes up again, we simply do a put of the
// handle.
inline void P2DumpProxy (const Proxy& p, DumpContext_& dc)
{
  // Lookup proxys by pointer to see if we have dumped them already
  void* ptr_handle = p.handle_;
  if (dc.handles.contains(ptr_handle)) {
    // Already placed, just an integer and a PY_BINGET
    int_u4 int_handle = dc.handles(ptr_handle);
    P2DumpCodeAndInt_(int_handle, PY_BINGET, PY_LONG_BINGET, dc);
    return;
  } 

  // Assertion: Proxy not dumped yet. 
  
  // Dump what the proxy points to as normal (but force the label
  // where appropriate)
  if (p.tag=='t') { 
    Tab& t = p;  
    P2DumpTab(t, dc, ptr_handle); // Memoize happens in here
  } else if (p.tag=='o') { 
    OTab& ot = p;  
    P2DumpOTab(ot, dc, ptr_handle); // Memoize happens in here
  } else if (p.tag=='u') { 
    Tup& t = p;  
    P2DumpTup(t, dc, ptr_handle); // Memoize happens in here

  } else if (p.tag=='n') {

    // Invade the proxy and get the underlying array ...
    RefCount_<void>* invade = (RefCount_<void>*)ptr_handle;
    void* arr_data = invade->data_;
    char subtype = p.subtype;

    switch (dc.disposition) {
    case AS_LIST:  dumpArrayAsList_(arr_data,subtype,dc, ptr_handle); break;
    case AS_PYTHON_ARRAY: dumpArray_(arr_data,subtype,dc, ptr_handle); break;
    case AS_NUMERIC: dumpNumericArray_(arr_data,subtype,dc, ptr_handle); break;
    case AS_NUMPY: dumpNumPyArray_(arr_data, subtype, dc, ptr_handle); break;
    default: p2error_("unknown disposition for dumping arrays");
    }
  } else {
    throw logic_error("No such Proxy type");
  }
}

inline size_t BytesToDumpProxy (const Proxy& p, DumpContext_& dc)
{
  size_t bytes = 0;

  void* ptr_handle = p.handle_;
  if (dc.handles.contains(ptr_handle)) {
    // Already placed, just an integer and a BINPUT
    int_u4 int_handle = dc.handles(ptr_handle);
    return (int_handle<256) ? 1+1 : 4+1; // 1 or 4-byte int, plus BINPUT tag
  }

  // Assertion: Proxy not dumped yet.  

  // Bytes to dump?
  if (p.tag=='t') {
    Tab& t = p;  
    bytes += BytesToDumpTab(t, dc, ptr_handle);
  } else if (p.tag=='o') {
    OTab& ot = p;  
    bytes += BytesToDumpOTab(ot, dc, ptr_handle);
  } else if (p.tag=='u') {
    Tup& t = p;  
    bytes += BytesToDumpTup(t, dc, ptr_handle);

  } else if (p.tag=='n') {

    // Invade the proxy and get the underlying array ...
    RefCount_<void>* invade = (RefCount_<void>*)ptr_handle;
    void* arr_data = invade->data_;
    char subtype = p.subtype;

    switch (dc.disposition) {
    case AS_LIST: 
      bytes += BytesToDumpArrayAsList(arr_data,subtype,dc,ptr_handle);  break;
    case AS_PYTHON_ARRAY: 
      bytes += BytesToDumpArray(arr_data,subtype,dc,ptr_handle);        break;
    case AS_NUMERIC: 
      bytes += BytesToDumpNumericArray(arr_data,subtype,dc,ptr_handle); break;
    case AS_NUMPY: 
      bytes += BytesToDumpNumPyArray(arr_data,subtype,dc,ptr_handle); break;
    default: p2error_("unknown disposition for dumping arrays");
    }
  } else {
    throw logic_error("No such Proxy type");
  }
  return bytes;
}


// Generic dump
inline void P2DumpValue (const Val& v, DumpContext_& dc)
{
  if (IsProxy(v)) { Proxy*pp=(Proxy*)&v.u.P; P2DumpProxy(*pp, dc); return; }

  switch(v.tag) {

  case 's':
  case 'S':
  case 'i':
  case 'I':
  case 'l': dump4ByteInteger(v,dc);   break;

  case 'x': dump8ByteInteger(v, dc);  break;

  case 'L': 
  case 'X': dump8ByteUnsignedInteger(v, dc);  break;

  case 'b': dumpBool(v, dc);          break;

  case 'f':
  case 'd': dump8ByteFloat(v, dc);    break;

  case 'F':
  case 'D': dump16ByteComplex(v, dc); break;

    // After Proxy checks, so can get impl directly out
  case 'a': { OCString*sp=(OCString*)&v.u.a; dumpString(*sp, dc); break;}
  case 't': { Tab*tp=(Tab*)&v.u.t; P2DumpTab(*tp, dc, 0);    break;}
  case 'o': { OTab*op=(OTab*)&v.u.o; P2DumpOTab(*op, dc, 0);    break;}
  case 'u': { Tup*tp=(Tup*)&v.u.u; P2DumpTup(*tp, dc, 0);    break;}
  case 'q': { int_n*np=(int_n*)&v.u.q; P2DumpBigInt(*np, dc, 0); break;}
  case 'Q': { int_n*np=(int_n*)&v.u.Q; P2DumpBigInt(*np, dc, 0); break;}
  case 'n': {
    switch (dc.disposition) {
    case AS_LIST: dumpArrayAsList_(v,dc); break;
    case AS_PYTHON_ARRAY: dumpArray_(v,dc); break;
    case AS_NUMERIC: dumpNumericArray_(v,dc); break;
    case AS_NUMPY: dumpNumPyArray_(v,dc); break;
    default: p2error_("unknown disposition for dumping arrays");
    }
    break;
  }

  case 'Z' : *(dc.mem)++ = 'N'; break;
  default:  p2error_("Unknown type in dump");
  }

}

// Note that these are all over estimates by a few bytes 
inline size_t BytesToDumpVal (const Val& v, DumpContext_& dc)
{
  if (IsProxy(v)) { Proxy*pp=(Proxy*)&v.u.P;return BytesToDumpProxy(*pp, dc); }

  switch(v.tag) {

  case 's':
  case 'S':
  case 'i':
  case 'I':
  case 'l': return BytesToDump4ByteInteger();  
  case 'x': return BytesToDump8ByteInteger(dc);  
  case 'L': 
  case 'X': return BytesToDump8ByteUnsignedInteger(dc);  

  case 'b': return BytesToDumpBool();

  case 'f':
  case 'd': return BytesToDump8ByteFloat();

  case 'F':
  case 'D': return BytesToDump16ByteComplex(dc); 

    // After Proxy checks, so can get impl directly out
  case 'a': { OCString*sp=(OCString*)&v.u.a; return BytesToDumpString(*sp); }
  case 't': { Tab*tp=(Tab*)&v.u.t; return BytesToDumpTab(*tp, dc, 0); }
  case 'o': { OTab*tp=(OTab*)&v.u.o; return BytesToDumpOTab(*tp, dc, 0); }
  case 'u': { Tup*tp=(Tup*)&v.u.u; return BytesToDumpTup(*tp, dc, 0); }
  case 'q': { int_n*tp=(int_n*)&v.u.q; return BytesToDumpBigInt(*tp, dc, 0); }
  case 'Q': { int_n*tp=(int_n*)&v.u.Q; return BytesToDumpBigInt(*tp, dc, 0); }
  case 'n': {
    switch (dc.disposition) {
    case AS_LIST: return BytesToDumpArrayAsList(v,dc); break;
    case AS_PYTHON_ARRAY: return BytesToDumpArray(v,dc); break;
    case AS_NUMERIC: return BytesToDumpNumericArray(v,dc); break;
    case AS_NUMPY: return BytesToDumpNumPyArray(v,dc); break;
    default: p2error_("unknown disposition for dumping arrays");
    }
    break;
  }

  case 'Z' : return 1;
  default:  p2error_("Unknown type in dump");
  }
  return 0;
}

// ********************* Loads ************************

// The routines below are needed for LOADING, but we need the
// When doing a get, if we have to GET an item that is held by value,
// then we have to turn that value (held by value) into an item that
// is held by reference.

// When doing a "get" of previously found values on the load stack,
// find the previous value in the load context and do the right thing
// with it.  I.e. lhs=rhs, but with more work.
inline void HandleGetAssignment (Val& lhs, Val& rhs)
{
  // THESE items can be turned into Proxies and TRULY shared
  switch (rhs.tag) {
  case 't':  // Tab was memoized, AND an on demand proxy needs to be created
    rhs.Proxyize(); break;
  case 'o':  // OTab was memoized, AND an on demand proxy needs to be created
    rhs.Proxyize(); break;
  case 'u':  // Tup was memoized, AND an on demand proxy needs to be created
    rhs.Proxyize(); break;
  case 'n': // Array<x> was memoized AND on demand proxy needs to be created
    rhs.Proxyize(); break;
  }
  lhs = rhs;
}


#define LOADARRAYHELP(T, SUB) { v.tag='n'; v.subtype=SUB; size_t len=byte_len/sizeof(T); Array<T>*ap=(Array<T>*)(&v.u.n); new (ap) Array<T>(len); ap->expandTo(len); memcpy(ap->data(), lc.mem, byte_len); }


void P2LoadArray(Val& v, LoadContext_& lc, bool saw_memo=false)
{
  // If the preamble was memoized, then we saw the preamble already as a get
  if (!saw_memo) {
    lc.mem = P2EXPECT_(ArrayPreamble, lc.mem);
    lc.handle = P2_ARRAY_HANDLE;
    handleAPut_(0, lc);
  }

  // Get a string describes the type of the array: it is
  // a simple string with a typecode
  OCString desc;
  P2LoadString(&desc, lc);
  char array_tag = desc[0];

  // Start getting the length of the data array .. in bytes:
  // by getting the length first, we can preallocate the
  // Array<T>
  size_t byte_len = loadStringLength_(lc);

  switch (array_tag) {
  case 'c': LOADARRAYHELP(int_1, 's'); break;
  case 'B': LOADARRAYHELP(int_u1,'S'); break;
  case 'h': LOADARRAYHELP(int_2, 'i'); break;
  case 'H': LOADARRAYHELP(int_u2,'I'); break;
  case 'i': LOADARRAYHELP(int_4, 'l'); break;
  case 'I': LOADARRAYHELP(int_u4,'L'); break;
  case 'l': LOADARRAYHELP(int_8, 'x'); break;
  case 'L': LOADARRAYHELP(int_u8,'X'); break;
  case 'b': LOADARRAYHELP(bool,  'b'); break;
  case 'f': LOADARRAYHELP(real_4,'f'); break;
  case 'd': LOADARRAYHELP(real_8,'d'); break;
  case 'F': LOADARRAYHELP(complex_8, 'F'); break;
  case 'D': LOADARRAYHELP(complex_16,'D'); break;
  default: p2error_("Can't have anything other than POD data in an array");
  }

  // .. with length known, normally we'd finish "loading the string",
  //  but don't need to  ... we already memcpy the data from
  // the buffer!
  ///  -> NO NEEED! finishLoadingString_(v, dc);
  lc.mem += byte_len;  // Advance to end of string
  handleAPut_(&v, lc); // BINPUT, the handle

  lc.mem = P2EXPECT_("\x86R", lc.mem);// TUPLE, PY_REDUCE
  handleAPut_(&v, lc); // BINPUT, the handle
}

// Currently same as above
#define LOADNUMERICHELP(T, SUB) { v.tag='n'; v.subtype=SUB; size_t len=byte_len/sizeof(T); Array<T>*ap=(Array<T>*)(&v.u.n); new (ap) Array<T>(len); ap->expandTo(len); memcpy(ap->data(), lc.mem, byte_len); }

void P2LoadNumeric(Val& v, LoadContext_& lc, bool saw_memo=false)
{
  // If the preamble was memoized, then we saw the preamble already as a get
  if (!saw_memo) {
    lc.mem = P2EXPECT_(NumericPreamble, lc.mem);
    lc.handle = P2_NUMERIC_HANDLE;
    handleAPut_(0, lc);
  }

  if (*(lc.mem)++!='(') p2error_("expected ( in Numeric Array");
  // .. and there might be another if this was Python 2.2!
  if (*lc.mem=='(') {
    lc.mem++;
  }

  // It's possible to send a shapeless array: Numeric.zeros((),'i') which
  // causes there to be a closing ')' instead of a number.  This causes
  // us to 'skip' the number of elements in the array and go straight to the
  // type code string.
  if (*lc.mem==')') {
    lc.mem++;
  } else {
    // get an integer:  this is the number of elements in the array
    Val length_of_array;
    P2LoadValue(length_of_array, lc);  // TODO: Should check against byte len

    char tp = *(lc.mem)++; // 't' is in Python 2.2, TUPLE1 in > Python 2.2
    if (tp!='t'&&tp!=PY_TUPLE1) p2error_("expected start tuple in Numeric Array");
  }

  // Get a string describes the type of the array: it is
  // a simple string with a typecode
  OCString desc;
  P2LoadString(&desc, lc);
  char array_tag = desc[0];

  // Start getting the length of the data array .. in bytes:
  // by getting the length first, we can preallocate the
  // Array<T>
  size_t byte_len = loadStringLength_(lc);

  switch (array_tag) {
  case '1': LOADNUMERICHELP(int_1, 's'); break;
  case 'b': LOADNUMERICHELP(int_u1,'S'); break;
  case 's': LOADNUMERICHELP(int_2, 'i'); break;
  case 'w': LOADNUMERICHELP(int_u2,'I'); break;
  case 'i': LOADNUMERICHELP(int_4, 'l'); break;
  case 'u': LOADNUMERICHELP(int_u4,'L'); break;
 
  // Take care with these: longs differ on machines
  //case 'l': LOADNUMERICHELP(int_8, 'x'); break;
  //case 'x': LOADNUMERICHELP(int_u8,'X'); break;
  case 'l': {
    v.tag='n'; v.subtype='x'; size_t len=byte_len/sizeof(long); 
    Array<int_8>*ap=(Array<int_8>*)(&v.u.n); 
    new (ap) Array<int_8>(len); 
    ap->expandTo(len);
    int_8* out_data = (int_8*)ap->data();
    long* in_data = (long*)lc.mem;
    for (size_t ii=0; ii<len; ii++) {
      out_data[ii] = in_data[ii];
    }
    break;
  }


    // case 'b': LOADNUMERICHELP(bool,  'b'); break;
  case 'f': LOADNUMERICHELP(real_4,'f'); break;
  case 'd': LOADNUMERICHELP(real_8,'d'); break;
  case 'F': LOADNUMERICHELP(complex_8, 'F'); break;
  case 'D': LOADNUMERICHELP(complex_16,'D'); break;
  default: p2error_("Can't have anything other than POD data in a Numeric array");
  }

  // .. with length known, normally we'd finish "loading the string",
  //  but don't need to  ... we already memcpy the data from
  // the buffer!
  ///  -> NO NEEED! finishLoadingString_(v, dc);
  lc.mem += byte_len;  // Advance to end of string
  handleAPut_(&v, lc); // BINPUT, the handle

  char bool_of = *(lc.mem)++;
  if (bool_of!=PY_NEWTRUE && bool_of!=PY_NEWFALSE) {
    if (bool_of=='K') { // small int for small int for boolean
      bool_of = *(lc.mem)++;
    } else {
      p2error_("expected bool at end of Numeric array");
    }
  }

  lc.mem = P2EXPECT_("tR", lc.mem);// TUPLE, PY_REDUCE
  handleAPut_(&v, lc); // BINPUT, the handle
}


// When there MIGHT be a single value in an Arr or Tab, we have to try
// to load it.  It that load fails, then we throw this exception to
// restart.  This is a bit of a Hack.
struct BadRetry : public runtime_error {
  BadRetry () : runtime_error("retry") { }
}; // BadRetry


// Routine to get where Key is, based on where value is. Kind of a
// hack to get the Key inside the KeyVal struct thingee.
inline Val* GetKeyFromValue (Val& value)
{
  // HACK to get pointer to key!!!!
  Val* ptr = (Val*)&value;
  ptr -= 1;  // key always before value in struct, I Hope!!!!!
  return ptr;
}
 
inline void loadSingleKeyValue_ (Val& t, LoadContext_& lc,
				 Val** keyp, Val** valp)
{
  // Load key: Note that memo for the key will be WRONG, since it
  // currently thinks the memo belongs on the stack!!
  Val key;
  lc.handle = -999;
  bool saw_bin_get = P2LoadValue(key, lc);
  int key_handle = lc.handle;

  // RARE CASE: The next table is the same memoized, which just forced
  // a proxy-ization, which means t is invalid.  TODO: We still can't
  // handle Proxys within Proxys.  This is why we pas the VAL in and
  // not the tab: If the tab changes inside the Val to a Proxy, the
  // subscript (below) will still work.
  Val& value = t[key] = None;
  *valp = &value;
  *keyp = GetKeyFromValue(value);

  // Once the *key* is IN the table, it won't move.  We have to
  // readjust the memoization of the key since it exists now and is
  // in a stable place, whereas before it was on the stack
  if (!saw_bin_get && key_handle >= 0) {
    lc.memoize_table[key_handle] = *keyp;
    lc.is_string[key_handle]     = false;  // Key is ALWAYS a Val in here!
  }

  // Load value
  lc.handle = -444;
  P2LoadValue(value, lc);
}


inline void P2LoadTab (Val& v, LoadContext_& lc)
{
  // Initialize into so always start with empy table
  v = Tab();

  if (*(lc.mem)++ != PY_EMPTY_DICT) p2error_("not the start of a dict");
  handleAPut_(&v, lc); // in case PUT

  // Empty dictionary or single element
  Val *keyp=0, *valp=0;
  if (*(lc.mem) != '(') {
    char* saved_mem = lc.mem;
    bool bad_retry = false;
    try {
      // THIS IS A HACK to "retry":  we really probably want a stack
      loadSingleKeyValue_(v, lc, &keyp, &valp);
    } catch (const BadRetry& br) {
      bad_retry = true;
    }
    if (bad_retry || *lc.mem!=PY_SETITEM) {
      if (keyp) *keyp=None;
      if (valp) *valp=None; // Makes sure no 'accidental' circular references
      lc.mem = saved_mem;
      v = Tab();
    } else {
      lc.mem++;
    }
    return;
  }

  // Once we've seen the '(', only a SETITEM or SETITEMS ends it
  lc.mem++;
  while (1) {
    char tag = *(lc.mem);
    if (tag==PY_SETITEM || tag==PY_SETITEMS) break;
    loadSingleKeyValue_(v, lc, &keyp, &valp);
  }

  (void)*(lc.mem)++;  // Reinstall memory into struct past SETITEM(s)
}

// ) or EMPTY_TUPLE: 0 element tuple
// item then \x85 : 1 element tuple
// item1, item 2 then \x86 : 2 element tuple
// item1,2,3 then \x87 : 3 element tuple
// PY_MARK item1,..n then TUPLE: n element tuple
void P2LoadTup (Val& v, LoadContext_& lc, bool /*saw_memo*/=false)
{
  char start = (*(lc.mem)++);

  // Can't handle TUPLE1, TUPLE2, TUPLE3 with current architecture
  if (start!=PY_MARK && start!=PY_EMPTY_TUPLE) p2error_("not the start of a tuple");

  // This starts the "full" tuple: PY_MARK item1 item2 ... itemn TUPLE
  Tup& u = v = Tup();
  u.impl().resize(12); 
  if (start== PY_MARK) {
    Array<Val>& a = u.impl();
    while (1) {
      if (*(lc.mem) == PY_TUPLE) break;
      a.append(None);
      P2LoadValue(a(a.length()-1), lc);
    }
    lc.mem++;
  }
  handleAPut_(&v, lc);
}

void P2LoadOTab (Val& v, LoadContext_& lc, bool saw_memo=false)
{
  OTab& o = v = OTab();

  // If the preamble was memoized, then we saw the preamble already as a get
  if (!saw_memo) {
    lc.mem = P2EXPECT_(OrderedDictPreamble, lc.mem);
    lc.handle = P2_ORDEREDDICT_HANDLE;
    handleAPut_(0, lc);
  }
  // There cases to make sure we handle correctly:
  // Empty:  OrderedDict()
  // 1 element: OrderedDict([(1,2)])
  // n elements:  OrderedDict([(1,2),(3,4)])
  if (*(lc.mem)++ != PY_EMPTY_LIST) p2error_("Start of OrderedDict, no list?");
  handleAPut_(&v, lc);

  char c = *(lc.mem);
  if (c!=PY_TUPLE1) { // Empty OTab when see a TUPLE: so jump to common PostAmble
    // 1 element or n elements resepctively
    if (c==PY_EMPTY_LIST || PY_MARK==c) { 
      char final_append_mark = (c==PY_EMPTY_LIST) ? PY_APPEND : PY_APPENDS;
      char cn =                (c==PY_EMPTY_LIST) ? *(lc.mem)++ : *(++lc.mem)++; 
      for (; cn!=final_append_mark; cn=*(lc.mem)++) {
	if (cn!=PY_EMPTY_LIST) p2error_("Expected PY_EMPTY_LIST during OTab");
	Val empty;
	handleAPut_(&empty, lc); // Don't expect memoization of this
	
	// Key, value (2 only!)
	if (*(lc.mem)++ !=PY_MARK) p2error_("Expected PY_MARK: n elem OTab case");
	Val key;
	P2LoadValue(key, lc);
	Val value;
	P2LoadValue(value, lc);
	if (*(lc.mem)++ !=PY_APPENDS) p2error_("Expected PY_APPENDS: n elem OTab"); 
	o.swapInto(key, value);
      }  
    } else {    // ???
      p2error_("Expected a PY_MARK or PY_EMPTY_LIST on reducing an OrderedDict");
    }
  }
  // Postamble always the same: \x85, 'R"
  if (*(lc.mem)++ != PY_TUPLE1)  p2error_("Expected TUPLE at Finish of OTab"); 
  if (*(lc.mem)++ != PY_REDUCE) p2error_("Expected PY_REDUCE at Finish of OTab");
  handleAPut_(&v, lc);
}

// RARE CASE: The next Arr is the same memoized, which just forced a
// proxy-ization, which means a is invalid.  This is why we pass the
// VAL in and not the Arr: If the Arr inside the Val to a Proxy, the
// append and subscript (below) will still work.  It's slower, but it
// is correct.
inline void loadSingleArr_ (Val& v, LoadContext_& lc,
			    Val** itemp)
{
  Arr& a = v;     // If Proxy, gets the value out as Arr correctly too ...
  a.append(None);
  Val& item = a[a.length()-1];
  *itemp = &item;

  lc.handle = -999;
  P2LoadValue(item, lc);
}

inline void P2LoadArr (Val& v, LoadContext_& lc)
{
  // Initialize into so always start with empy table
  size_t capacity = 256; // 2; //1024; // TODO: Too big?
  if (v.tag == 'l') capacity = v;
  v = Arr(capacity);
  char* saved_mem = lc.mem;

  if (*(lc.mem)++ != PY_EMPTY_LIST) p2error_("not the start of a list");
  handleAPut_(&v, lc);

  // Complication: if no (, it's either (a) an empty list
  // or (b) list with a single element.  
  // Empty List if no ( to start a tuple!
  Val* itemp = 0;
  if (*lc.mem != '(') {  // Either empty or list with single element
    char* saved_mem = lc.mem;
    bool bad_retry = false;
    try {
      loadSingleArr_(v, lc, &itemp);
    } catch (const BadRetry& br) {
      bad_retry = true;
    }
    if (bad_retry || *lc.mem!=PY_APPEND) { // Yup, single list
      if (itemp) *itemp=None; // Makes sure no 'accidental' circular references
      lc.mem = saved_mem;
      v = Arr();
    } else {
      lc.mem++;
    }
    return;
  } 

  // Once we've seen the '(', only a PY_APPEND or PY_APPENDS end it
  lc.mem++;
  while (1) {
    char tag = *(lc.mem);
    if (tag==PY_APPEND || tag==PY_APPENDS) break;
    
    // List will change size, invalidating all memoizes
    Arr& a = v;
    if (a.length()==a.capacity()) { 
      // HACK!!!  Ugh ... is there a better way?
      v = int_4(2*a.length());
      lc.mem = saved_mem;
      P2LoadArr(v, lc);
      return;
    }
    loadSingleArr_(v, lc, &itemp);
  }
  
  lc.mem++;// Reinstall memory into struct past PY_APPEND(s

}

inline void P2LoadComplex (Val& v, LoadContext_& lc, bool saw_mem=false)
{
  v.tag = 'D';  // Risky; assumes Val is None
  cx_union_t<real_8>* vp = &v.u.D;
  complex_16* cp = (complex_16*)vp;
  loadc16_(*cp, lc, saw_mem);
  handleAPut_(&v, lc);
}

inline bool P2LoadValue (Val& v, LoadContext_& lc)
{
  char tag = *(lc.mem)++;
  switch(tag) {

  case PY_NEWTRUE:  v = bool(1); break;
  case PY_NEWFALSE: v = bool(0); break;
  case PY_NONE :    v = None;    break;

  case PY_BINGET : // Lookup via small integer
  case PY_LONG_BINGET : { // Lookup via integer
    lc.mem--;
    void* vp = handleAGet_(v, lc, (OCString*)0);
    if      (vp==&ComplexPreambleAsString){ P2LoadComplex(v, lc, true); break;}
    else if (vp==&ArrayPreambleAsString)  { P2LoadArray(v, lc, true);   break;}
    else if (vp==&NumericPreambleAsString){ P2LoadNumeric(v, lc, true); break;}
    return true; // Indicate we set ov from a cached copy
  }

  case PY_BININT1:{int_u1 iu1; LOAD_FROM_LITTLE_ENDIAN1(lc.mem,iu1); int_4 i4=iu1; lc.mem+=1; v=i4; break;}
  case PY_BININT2:{int_u2 iu2; LOAD_FROM_LITTLE_ENDIAN2(lc.mem,iu2); int_4 i4=iu2; lc.mem+=2; v=i4; break;}
  case PY_BININT: {v.tag='l';LOAD_FROM_LITTLE_ENDIAN4(lc.mem, v.u.l); lc.mem+=4;break;}
    // ...but the floats always BIG ENDIAN ... ????
  case PY_BINFLOAT: {v.tag = 'd';LOAD_FROM_BIG_ENDIAN8(lc.mem, v.u.d); lc.mem+=8;break;}

  case PY_LONG1 : {
    int_8 i8;
    int_u1 len=*(lc.mem)++; 
    if (len>9) p2error_("Can't support longs over 8/9 bytes");
    char buff[9] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    memcpy(buff, lc.mem, len);
    LOAD_FROM_LITTLE_ENDIAN8(buff, i8); 
    lc.mem+=len;
    if (len==9) { 
      v = int_u8(i8);
    } else { 
      v = i8;
    }
    break;
  }

  case PY_SHORT_BINSTRING :
  case PY_BINSTRING : { lc.mem--; P2LoadString((OCString*)&v.u.a, lc); v.tag='a'; break; }

  case PY_EMPTY_DICT: lc.mem--; P2LoadTab(v, lc); break;
  case PY_EMPTY_LIST: lc.mem--; P2LoadArr(v, lc); break;

  case PY_GLOBAL: { // This can be lots of things, almost certainlly just complex
    char start = *(lc.mem)--;
    if (start=='_')        { P2LoadComplex(v, lc); break;}
    else if (start=='a')   { P2LoadArray(v, lc);   break;}
    else if (start=='N')   { P2LoadNumeric(v, lc); break;}
    else if (start=='c')   { P2LoadOTab(v, lc); break;}
    else p2error_("Unknown global");
  }

    // These two cases are if an older version of Python (2.2) 
    // is writing out integers and Longs as strings
  case 'L': { 
    if (*lc.mem=='-') {
      v = loadLONG_(lc); break;
    } else {
      v = loadULONG_(lc); break;
    }
  }
  case 'I': v = loadINT_(lc); break;
    // Careful with LONG and INT as typedefs
    
  case '(': case ')': /* PY_MARK, EMPTY_TUPLE */ { lc.mem--; P2LoadTup(v, lc); break; }

    // When we are "trying" to see if we have a single
    // entry table or array
  case PY_APPEND: case PY_APPENDS: case PY_SETITEM: case PY_SETITEMS:
  case PY_STOP: throw BadRetry(); break;

  default:  p2error_("Unknown tag in python pickling protocol 2");
  }

  return false;
}

PTOOLS_END_NAMESPACE

