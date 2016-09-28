
#include "m2ser.h"
#include "m2convertrep.h"
#include "occonvert.h"

PTOOLS_BEGIN_NAMESPACE

// From m2opalvaluea.h

enum OpalValueA_Type_e {

  // NOTE TO MAINTAINER: Any changes to the following enumeration MUST
  // be accompanied by changes to the static const char opal_type_map[]
  // declared in Evaluator::evalFunction (m2evaluator.cc)!

  OpalValueA_UNDEFINED = 0,
  OpalValueA_STRING,
  OpalValueA_TEXT,
  OpalValueA_BOOL,
  OpalValueA_NUMBER,
  OpalValueA_VECTOR,
  OpalValueA_MULTIVECTOR,
  OpalValueA_EVENTDATA,
  OpalValueA_TIMEPACKET,
  OpalValueA_TABLE,
  OpalValueA_HEADER,
  OpalValueA_LINK,
  OpalValueA_USERDEFINED,
  OpalValueA_TYPE_E
};


// NEW CODE START HERE!!

// Code to actual dump Vals as OpalValues


// An OMemStream helps us to output the data without having to know
// exactly how many bytes we are going to have to have to serialize.
// It avoid the problem of "always" having to see if we are over the
// array bounds by doing a "reserve" when we know how much space we'll
// be socking it to.  (We still check the array bounds often, just not
// every single put).  Essentially this is a dynamically resizing
// buffer
#define DEFAULT_OMEM_LEN 1024
class OMemStream { 
  public:

  OMemStream (int expected, MachineRep_e rep) :
    data_(new char[expected]),
    len_(0),
    capac_(expected),
    rep_(rep)
  { }
  ~OMemStream() { delete [] data_; }
  
  // Reserve enough space so we can add a few new bytes
  // and return a pointer to where the space would start
  char* reserve (int bytes_of_space) 
  {
    if (bytes_of_space+len_>capac_) {
      // Alocate new space
      int new_capac = max(bytes_of_space+len_, 2*capac_);
      char* new_mem = new char[new_capac];
      memcpy(new_mem, data_, len_);
      // Install
      capac_ = new_capac;
      delete [] data_;
      data_ = new_mem;
    }
    // Aseertion, enough space
    char* current = data_+len_;
    len_ += bytes_of_space;
    return current;
  } 
  
  // Return where memory starts:  this is for adoption!
  char* start () 
  { 
    reserve(1);  // Add extra space for \0 JUST TO MAKE SURE THERE
    char* adopting = data_; 
    adopting[len_-1] = '\0';
    data_ = 0; 
    return adopting; 
  }
  int length () const { return len_; }
  MachineRep_e rep () const { return rep_; }

  protected:
  char* data_;  // buffer
  int len_;     // current length, always <= capac_
  int capac_;   // maximum number of bytes in buffer
  MachineRep_e rep_;  // the rep of the OUTGOING data (this tells the machine to convert the convert internal rep to THIS)
}; // OMemStream 


// Forward
inline void OpalDump (const Val& v, OMemStream& oms);

MachineRep_e NativeEndian ()
{
  static int_u4 tester= 1;
  static char* c = (char*)&tester;
  MachineRep_e native =(*c==1)? MachineRep_EEEI: MachineRep_IEEE;
  return native;
}

template <class T>
char* EndianDump (char* mem, const T* data_ptr, MachineRep_e outgoing_rep)
{
  memcpy(mem, data_ptr, sizeof(*data_ptr));
  MachineRep_e native = NativeEndian();
  ConvertBufferRepInPlace(native, outgoing_rep, mem, 
			  NumericTypeLookup((T*)0), 1);
  return mem+sizeof(*data_ptr);
}

// String
inline void OpalDumpCString (const char* data, int_u4 len, OMemStream& oms,
			     bool dump_without_opal_prefix=false)
{
  char* mem = oms.reserve((dump_without_opal_prefix?0:1)+4+len);

  // The prefix is NOT on tables, but everywhere else
  if (!dump_without_opal_prefix) {
    *mem++ = OpalValueA_STRING; 
  }
  // RTS: memcpy(mem, &len, sizeof(len));
  // RTS: mem+=sizeof(len);
  mem = EndianDump(mem, &len, oms.rep());
  memcpy(mem, data, len);
}

inline void OpalDumpString (const string& s, OMemStream& oms,
			    bool dump_without_opal_prefix=false)
{
  OpalDumpCString(s.data(), s.length(), oms, dump_without_opal_prefix);
}

inline void OpalDump (const string& s, OMemStream& oms)
{
  OpalDumpString(s, oms);
}

#if !defined(OC_USE_OC_STRING)
inline void OpalDump (const OCString& s, OMemStream& oms)
{
  OpalDumpCString(s.data(), s.length(), oms);
}
#endif

// Problem: "OpalLinks" are strings and currently dump as just a
// string.  Do we want to make them full proxies or just leave as is?
// For the moment, when we see a Proxy, we just dump it as is without
// trying.  TODO:  Have OpalLinks become Proxy???
#define OPALARRDUMPPROXY(T) { Array<T>& t=p; OpalDump(t,oms); }
inline void OpalDump (const Proxy& p, OMemStream& oms)
{
  switch (p.tag) {
  case 't': { Tab& t=p; OpalDump(t,oms); } break;
  case 'n': {
    switch (p.subtype) {
    case 's': OPALARRDUMPPROXY(int_1);  break;
    case 'S': OPALARRDUMPPROXY(int_u1); break;
    case 'i': OPALARRDUMPPROXY(int_2);  break;
    case 'I': OPALARRDUMPPROXY(int_u2); break;
    case 'l': OPALARRDUMPPROXY(int_4);  break;
    case 'L': OPALARRDUMPPROXY(int_u4); break;
    case 'x': OPALARRDUMPPROXY(int_8);  break;
    case 'X': OPALARRDUMPPROXY(int_u8); break;
    // case 'b': OPALARRDUMPPROXY(b, BOOL??, oms); break;// bool NOT M2k Number
    case 'f': OPALARRDUMPPROXY(real_4); break;
    case 'd': OPALARRDUMPPROXY(real_8); break;
    case 'F': OPALARRDUMPPROXY(complex_8); break;
    case 'D': OPALARRDUMPPROXY(complex_16); break; 
    case 'a': OPALARRDUMPPROXY(string); break; 
    case 't': OPALARRDUMPPROXY(Tab); break; 
    case 'n': throw logic_error("Don't support Arrays of Arrays"); break; 
    case 'Z': OPALARRDUMPPROXY(Val); break; 
    default: throw logic_error("Unknown tag");
    }
  }
  }
}

// Same code for Tab and OTab
template <class T>
void OpalDumpTable_ (const T& t, OMemStream& oms, 
		     bool dump_without_opal_prefix=false)
{
  int_u4 len = t.entries();
  char* mem = oms.reserve((dump_without_opal_prefix?0:1)+4);

  // The prefix is not on a first time dump of a straight table
  if (!dump_without_opal_prefix) {
    *mem++ = OpalValueA_TABLE;
  }
  // RTS: memcpy(mem, &len, sizeof(len));
  EndianDump(mem, &len, oms.rep());
  for (It ii(t); ii(); ) {
    string key = ii.key();     // Always get string, even though may be Val
    OpalDumpString(key, oms, true);
    
    const Val& v = ii.value();
    OpalDump(v, oms);
  }
}

// Table
void OpalDumpTab (const Tab& t, OMemStream& oms, 
		  bool dump_without_opal_prefix=false)
{ OpalDumpTable_(t, oms, dump_without_opal_prefix); }

void OpalDumpOTab (const OTab& t, OMemStream& oms, 
		  bool dump_without_opal_prefix=false)
{ OpalDumpTable_(t, oms, dump_without_opal_prefix); }




inline void OpalDump (const Tab& t, OMemStream& oms)
{
  OpalDumpTab(t, oms);
}

inline void OpalDump (const OTab& t, OMemStream& oms)
{
  OpalDumpOTab(t, oms);
}


// Arrays of Val, String, or Tab (NOT VECTORS!!!)
template <class T>
inline void OpalDumpArr (const Array<T>& a, OMemStream& oms,
			 bool dump_without_opal_prefix=false)
{
  int_u4 len = a.entries();
  char* mem = oms.reserve((dump_without_opal_prefix?0:1)+4);

  // The prefix is not on a first time dump of a straight table
  if (!dump_without_opal_prefix) {
    *mem++ = OpalValueA_TABLE;
  }
  // RTS: memcpy(mem, &len, sizeof(len));
  mem = EndianDump(mem, &len, oms.rep());

  const T* data = a.data();
  for (int ii=0; ii<int(len); ii++) {
    string key = Stringize(ii);
    OpalDumpString(key, oms, true);
    OpalDump(data[ii], oms);
  }
}


inline void OpalDump (const Arr& a, OMemStream& oms)
{
  OpalDumpArr(a, oms);
}

inline void OpalDump (const Tup& a, OMemStream& oms)
{
  OpalDumpArr(a.impl(), oms);
}

inline void OpalDump (const int_n& a, OMemStream& oms)
{
  string s = a.stringize();
  OpalDump(s, oms);
}

inline void OpalDump (const int_un& a, OMemStream& oms)
{
  string s = a.stringize();
  OpalDump(s, oms);
}


// helper function to dump a single number
template <class T>
inline void OpalNumDump_ (T* n, Numeric_e number_tag, OMemStream& oms)
{ 
  char* mem = oms.reserve(1+1+sizeof(T));

  *mem++ = OpalValueA_NUMBER; 
  *mem++ = number_tag; 
  // RTS: memcpy(mem, n, sizeof(T)); 
  // TODO: It looks like only CONTROL info in an endian, data still appears all little
  EndianDump(mem, n, oms.rep());
}

// helper function to dump a vector of numbers
template <class T>
inline void OpalVecDump_ (Arr& a, T*, Numeric_e number_tag, OMemStream& oms)
{
  int_u4 len=a.length(); 
 
  char* mem = oms.reserve(1+1+sizeof(len)+sizeof(T)*len);

  // opalvalue tag then numeric tag
  *mem++ = OpalValueA_VECTOR; 
  *mem++ = number_tag; 
  // 4 bytes of lenngth
  // RTS: memcpy(mem, &len, sizeof(len)); 
  // RTS: mem   += sizeof(len);
  mem = EndianDump(mem, &len, oms.rep());

  // .. the actual data itself... TODO: Endian?
  memcpy(mem, a.data(),sizeof(T)*len); 
}

inline void OpalDump (const Val& v, OMemStream& oms)
{
  if (IsProxy(v)) { 
    Proxy*tp=(Proxy*)&v.u.P; OpalDump(*tp, oms); 
    return;
  }

  char tag = v.tag;
  switch (tag) {
    // The "Numbers"
  case 's': OpalNumDump_(&v.u.s, BYTE, oms); break;
  case 'S': OpalNumDump_(&v.u.S, UBYTE, oms); break;
  case 'i': OpalNumDump_(&v.u.i, INTEGER, oms); break;
  case 'I': OpalNumDump_(&v.u.I, UINTEGER, oms); break;
  case 'l': OpalNumDump_(&v.u.l, LONG, oms); break;
  case 'L': OpalNumDump_(&v.u.L, ULONG, oms); break;
  case 'x': OpalNumDump_(&v.u.x, XLONG, oms); break;
  case 'X': OpalNumDump_(&v.u.X, UXLONG, oms); break;
  // case 'b': OpalNumDump_(&v.u.b, BOOL??, oms); break; // bool NOT M2k Number
  case 'f': OpalNumDump_(&v.u.f, M2_FLOAT, oms); break;
  case 'd': OpalNumDump_(&v.u.d, M2_DOUBLE, oms); break;
  case 'F': OpalNumDump_(&v.u.F, M2_CX_FLOAT, oms); break;
  case 'D': OpalNumDump_(&v.u.D, M2_CX_DOUBLE, oms); break;

    // bool
  case 'b': {
    char* mem = oms.reserve(1+1);
    *mem++ = OpalValueA_BOOL; 
    *mem++ = v.u.b; 
    break;
  }

    // None: represented by a TEXT
  case 'Z': { 
    char* mem = oms.reserve(1+4);
    *mem++ = OpalValueA_TEXT;
    int_u4 zero(0);
    // RTS: memcpy(mem,&zero,sizeof(zero));
    mem = EndianDump(mem, &zero, oms.rep());
    break;
  }

    // Okay, already done Proxy checks
  case 'a': { OCString*sp=(OCString*)&v.u.a; OpalDump(*sp, oms); break; }
  case 't': { Tab*tp=(Tab*)&v.u.t; OpalDump(*tp, oms); break; } 
  case 'o': {OTab*tp=(OTab*)&v.u.o; OpalDump(*tp, oms); break; } 
  case 'u': {Tup*tp=(Tup*)&v.u.u; OpalDump(*tp, oms); break; } 
  case 'q': {int_n*tp=(int_n*)&v.u.q; OpalDump(*tp, oms); break; } 
  case 'Q': {int_un*tp=(int_un*)&v.u.Q; OpalDump(*tp, oms); break; } 
  case 'n': {     // The arrays (Opaltables and Vectors)
    Arr* ap = (Arr*)&v.u.n;
    switch (v.subtype) {
    case 's': OpalVecDump_(*ap, &v.u.s, BYTE, oms); break;
    case 'S': OpalVecDump_(*ap, &v.u.S, UBYTE, oms); break;
    case 'i': OpalVecDump_(*ap, &v.u.i, INTEGER, oms); break;
    case 'I': OpalVecDump_(*ap, &v.u.I, UINTEGER, oms); break;
    case 'l': OpalVecDump_(*ap, &v.u.l, LONG, oms); break;
    case 'L': OpalVecDump_(*ap, &v.u.L, ULONG, oms); break;
    case 'x': OpalVecDump_(*ap, &v.u.x, XLONG, oms); break;
    case 'X': OpalVecDump_(*ap, &v.u.X, UXLONG, oms); break;
    case 'b': OpalVecDump_(*ap, &v.u.b, BIT, oms); break; // bool is NOT a M2k Number .. BIT?  
    case 'f': OpalVecDump_(*ap, &v.u.f, M2_FLOAT, oms); break;
    case 'd': OpalVecDump_(*ap, &v.u.d, M2_DOUBLE, oms); break;
    case 'F': OpalVecDump_(*ap, &v.u.F, M2_CX_FLOAT, oms); break;
    case 'D': OpalVecDump_(*ap, &v.u.D, M2_CX_DOUBLE, oms); break;
      
    case 'a': { Array<string>*sp=(Array<string>*)&v.u.n; OpalDump(*sp, oms); break; }
    case 't': { Array<Tab>*tp=(Array<Tab>*)&v.u.n; OpalDump(*tp, oms); break; }
    case 'n': throw runtime_error("We don't support Arrays of Arrays!"); break;
    case 'Z' : { Arr*ap=(Arr*)&v.u.n; OpalDump(*ap, oms); break; }
    default: throw runtime_error("Unknown subtype in array");
    }
    break;
  }
  default: throw runtime_error("Unknown type in array");
  }
}


int OpalDumpVal (const Val& v, char*& mem, MachineRep_e rep)
{
  OMemStream oms(DEFAULT_OMEM_LEN, rep);
  OpalDump(v, oms);
  mem = oms.start();
  return oms.length();
}


int OpalDumpTab (const Tab& t, char*& mem, MachineRep_e rep)
{
  OMemStream oms(DEFAULT_OMEM_LEN, rep);
  OpalDumpTab(t, oms, true);
  mem = oms.start();
  return oms.length();
}

int OpalDumpOTab (const OTab& t, char*& mem, MachineRep_e rep)
{
  OMemStream oms(DEFAULT_OMEM_LEN, rep);
  OpalDumpOTab(t, oms, true);
  mem = oms.start();
  return oms.length();
}


int OpalDumpArr (const Arr& a, char*& mem, MachineRep_e rep)
{
  OMemStream oms(DEFAULT_OMEM_LEN, rep);
  OpalDumpArr(a, oms, true);
  mem = oms.start();
  return oms.length();
}

int OpalDumpTup (const Tup& a, char*& mem, MachineRep_e rep)
{
  OMemStream oms(DEFAULT_OMEM_LEN, rep);
  OpalDumpArr(a.impl(), oms, true);
  mem = oms.start();
  return oms.length();
}


PTOOLS_END_NAMESPACE

// Loads

#include "timeconv.h"


PTOOLS_BEGIN_NAMESPACE

template <class T>
char* EndianLoad (T* data_ptr, const char* mem, MachineRep_e incoming_rep)
{
  memcpy(data_ptr, mem, sizeof(*data_ptr));
  MachineRep_e native = NativeEndian();
  ConvertBufferRepInPlace(incoming_rep, native, data_ptr, 
			  NumericTypeLookup(Selector(T)), 1);
  return ((char*)mem)+sizeof(*data_ptr);
}

#define OPALLOADNUMB(N, TAG) { mem=EndianLoad(&(N), mem, endian); v.tag=TAG; }
//#define OPALLOADNUMB(N, TAG) { memcpy(&(N), mem, sizeof(N)); mem+=sizeof(N); v.tag=TAG; }
inline char* OpalLoadNumber (Val& v, char* mem, MachineRep_e endian)
{
  // No need to check for the OpalvalueA_NUMBER ...
  int_u1 numeric_enum = *mem++;
  Numeric_e n = Numeric_e(numeric_enum);
  switch (n) {
  case BYTE:     OPALLOADNUMB(v.u.s, 's'); break;
  case UBYTE:    OPALLOADNUMB(v.u.S, 'S'); break;
  case INTEGER:  OPALLOADNUMB(v.u.i, 'i'); break;
  case UINTEGER: OPALLOADNUMB(v.u.I, 'I'); break;
  case LONG:     OPALLOADNUMB(v.u.l, 'l'); break;
  case ULONG:    OPALLOADNUMB(v.u.L, 'L'); break;
  case XLONG:    OPALLOADNUMB(v.u.x, 'x'); break;
  case UXLONG:   OPALLOADNUMB(v.u.X, 'X'); break;
  case FLOAT:    OPALLOADNUMB(v.u.f, 'f'); break;
  case DOUBLE:   OPALLOADNUMB(v.u.d, 'd'); break;
  case CX_FLOAT: OPALLOADNUMB(v.u.F, 'F'); break;
  case CX_DOUBLE:OPALLOADNUMB(v.u.D, 'D'); break;

    // Time is an int_u8, Duration a real_8: but they need
    // to be strings to be consistent with how all others
    // deserialize
    //case M2_TIME: OPALLOADNUMB(v.u.X, 'X'); break;
    //case M2_DURATION: OPALLOADNUMB(v.u.d, 'd'); break;
  case M2_TIME: {
    int_u8 qns; 
    OPALLOADNUMB(qns, 'X');
    TimeConv tt(qns);
    v = Stringize(tt);
    break;
  }
  case M2_DURATION: {
    real_8 q; 
    OPALLOADNUMB(q, 'd');
    v = Stringize(q);
    break;
  }

  default: throw runtime_error("Cannot handle load Number");
  }
  return mem;
}

inline char* OpalLoadString (Val& v, char* mem, MachineRep_e endian)
{
  // Skip the string check because (a) The OpalTable keys don't use it
  // and (b) the OpalLoad can skip it anyway
  // if (*mem++ != OpalValueA_STRING) { throw runtime_error(); }

  // Assume Val starts out as a None
  v.tag = 'a';

  // Get the length of the string
  int_u4 length;
  // RTS: memcpy(&length, mem, sizeof(length));
  // RTS: mem+=4;
  mem = EndianLoad(&length, mem, endian);

  // Get the string
  OCString* sp = (OCString*)v.u.a;
  new (sp) OCString(mem, length);
  
  mem+=length;
  return mem;
}


#define DEFAULT_CONVERT_TO_ARR true
inline char* OpalLoadTable (Val& v, char* mem, 
			    bool convert_to_arr=DEFAULT_CONVERT_TO_ARR, 
			    MachineRep_e endian=MachineRep_EEEI)
{
  // Make sure starting a table
  //if (*mem++ != OpalValueA_TABLE) {
  //  throw runtime_error("Expected start of an Opaltable");
  // }

  // get the number of key-value entries
  int_u4 entries;
  mem = EndianLoad(&entries, mem, endian);

  // Start going through and getting the key-value pairs
  Tab& t = v = Tab();
  for (int ii=0; ii<int(entries); ii++) {

    Val key;            // Load key
    mem = OpalLoadString(key, mem, endian);
    
    Val& value = t[key]; // Load value
    mem = OpalLoadVal(value, mem, endian);
  }
  if (entries && convert_to_arr) { // Empty dict still dict
    ConvertTabToArr(v, false);  // just top-level (fast, as does it with swaps)
  }
  return mem;
}


#define OPALLOADVEC(T, TAG) { v.tag='n'; v.subtype=TAG; Array<T>*ap=(Array<T>*)&v.u.n; new (ap) Array<T>(length); ap->expandTo(length); memcpy(ap->data(), mem, length*sizeof(T)); mem+=length*sizeof(T); }


inline char* OpalLoadVector (Val& v, char* mem, MachineRep_e endian)
{
  // get the type code
  int_u1 numeric_enum = *mem++;
  Numeric_e n = Numeric_e(numeric_enum);

  // Get the length
  int_u4 length;
  // RTS: memcpy(&length, mem, sizeof(length));
  // RTS: mem+=4;
  mem = EndianLoad(&length, mem, endian);

  // Copy in the raw data
  switch (n) {
  case BYTE:     OPALLOADVEC(int_1,     's'); break;
  case UBYTE:    OPALLOADVEC(int_u1,    'S'); break;
  case INTEGER:  OPALLOADVEC(int_2,     'i'); break;
  case UINTEGER: OPALLOADVEC(int_u2,    'I'); break;
  case LONG:     OPALLOADVEC(int_4,     'l'); break;
  case ULONG:    OPALLOADVEC(int_u4,    'L'); break;
  case XLONG:    OPALLOADVEC(int_8,     'x'); break;
  case UXLONG:   OPALLOADVEC(int_u8,    'X'); break;
  case FLOAT:    OPALLOADVEC(float,     'f'); break;
  case DOUBLE:   OPALLOADVEC(double,    'd'); break;
  case CX_FLOAT: OPALLOADVEC(complex_8, 'F'); break;
  case CX_DOUBLE:OPALLOADVEC(complex_16, 'D'); break;

    // Time is a int_u8, duration a real_8:  In vectors,
    // they are int_u8s and real_8s, by themselves, they
    // are strings
  case M2_TIME:  OPALLOADVEC(int_u8,    'X'); break;
  case M2_DURATION: OPALLOADVEC(double,    'd'); break;

  default: throw runtime_error("Cannot handle load Number");
  }
  return mem;
}

// Helper to load numbers
template <class T>
inline T OpalLoadIt (char*& mem)
{
  T temp;
  memcpy(&temp, mem, sizeof(T));
  mem += sizeof(T);
  return temp;
}

inline char* OpalLoadMultiVector (Val& v, char* mem, MachineRep_e endian)
{
  // Already saw tag
  // int_u4 followed by the vectors
  int_u4 length; // RTS = OpalLoadIt<int_u4>(mem);
  mem = EndianLoad(&length, mem, endian);

  // Get the n arrays
  Arr& a = v = Arr(length);
  for (int ii=0; ii<int(length); ii++) {
    a.append(None);
    mem = OpalLoadVector(a[ii], mem, endian);
  }
  return mem;
}

inline char* OpalLoadEventData (Val& v, char* mem, MachineRep_e endian)
{
  // Already saw tag
  // int_u4 followed by the start time, end time, and vectors
  Tab& t = v = Tab();
  int_u4 length; //  = OpalLoadIt<int_u4>(mem);
  mem = EndianLoad(&length, mem, endian);
  t["startTime"] = OpalLoadIt<real_8>(mem);
  t["endTime"]   = OpalLoadIt<real_8>(mem);

  // Array of Multivectors
  Val& a = t["DATA"] = None;
  int len = length;
  for (int ii=0; ii<len; ii++) {
    mem = OpalLoadMultiVector(a, mem, endian);  // no tag, so go straight to 
  }
  return mem;
}


inline char* OpalLoadITS (Val& v, char* mem, MachineRep_e endian)
{
  // No OpalValue_A tag

  Tab& t = v = Tab();
  bool represents_ptime = bool(OpalLoadIt<int_u1>(mem));
  t["RepresentPTime"] = represents_ptime;
  t["FSO"] = OpalLoadIt<real_8>(mem);
  t["VSO"] = OpalLoadIt<real_8>(mem);

  if (represents_ptime) {
    TimeConv mt = OpalLoadIt<int_u8>(mem);
    t["Time"] = Stringize(mt);
    t["AccumDecimation"] = OpalLoadIt<real_8>(mem);
    bool init_poly = bool(OpalLoadIt<int_u1>(mem));
    if (init_poly) {
      Val& poly = t["pi"] = None;
      mem = OpalLoadVector(poly, mem, endian);
    }
    t["TrackNumber"] = OpalLoadIt<int_u4>(mem);
  } else {
    t["Time"] = Stringize(OpalLoadIt<real_8>(mem));
  }

  return mem;
}

inline char* OpalLoadTimePacket (Val& v, char* mem, MachineRep_e endian)
{
  // Already saw Opal tag

  // Basic structure of turning an TimePacket into Table
  Tab& t = v = Tab("{ "
		   "  'Valid':True,"
		   "  'TimeStamps': {" 
		   "     'Precision': [], "
		   "     'Nominal': [],"
		   "   }"
		   "}");

  // Valid
  t["Valid"] = bool(OpalLoadIt<int_u1>(mem));

  // Precision Time Stamps
  int_u8 plen = OpalLoadIt<int_u8>(mem);
  Arr& prec = t["TimeStamps"]["Precision"] = Arr(plen);
  for (int_u8 ii=0; ii<plen; ii++) {
    prec.append(None);
    mem = OpalLoadITS(prec[ii], mem, endian);
  }

  // Nominal timestamps
  int_u8 nlen = OpalLoadIt<int_u8>(mem);
  Arr& nomi = t["TimeStamps"]["Nominal"] = Arr(nlen);
  for (int_u8 ii=0; ii<nlen; ii++) {
    nomi.append(None);
    mem = OpalLoadITS(nomi[ii], mem, endian);
  }

  return mem;
}

char* OpalLoadVal (Val& v, char* mem, MachineRep_e endian)
{
  char tag = *mem++;
  switch (tag) {
  case OpalValueA_NUMBER: return OpalLoadNumber(v, mem, endian); 
  case OpalValueA_BOOL:   v = bool(*mem++); return mem;
  case OpalValueA_TEXT:   v = None; mem+=4; return mem;
  case OpalValueA_STRING: return OpalLoadString(v, mem, endian);
  case OpalValueA_TABLE:  return OpalLoadTable(v, mem, DEFAULT_CONVERT_TO_ARR, endian);
  case OpalValueA_VECTOR: return OpalLoadVector(v, mem, endian);
    
    // These all have to converted into some Table or Arr as there
    // is no corresponding Val
  case OpalValueA_MULTIVECTOR: return OpalLoadMultiVector(v, mem, endian);
  case OpalValueA_EVENTDATA:   return OpalLoadEventData(v, mem, endian);
  case OpalValueA_TIMEPACKET:  return OpalLoadTimePacket(v, mem, endian);

    // A link is just a string: TODO: Fix this?  M2k didn't really
    // design LINKS very well so they weren't used very much.
  case OpalValueA_LINK:  return OpalLoadString(v, mem, endian);

    // Seems to serialize just like a table
  case OpalValueA_HEADER: return OpalLoadTable(v, mem, DEFAULT_CONVERT_TO_ARR, endian);

    // This is probably the end of the stream ... a \0 at the end of
    // the string (UNDEFINED is a __0___!!!)
  case OpalValueA_UNDEFINED: return mem;

  default: throw runtime_error("Don't know how to handle tag:");
  }
}


char* OpalLoadSomeTable (Val& t, char* mem, 
			 bool convert_tab_to_arr_if_possible,
			 MachineRep_e endian)
{
  return OpalLoadTable(t, mem, convert_tab_to_arr_if_possible, endian);
}

PTOOLS_END_NAMESPACE 
