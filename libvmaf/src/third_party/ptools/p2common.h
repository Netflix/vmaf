#ifndef M2P2COMMON_H_

// Factor out the common code for the serialization code for Python
// Pickling protocol 2.  A lot of code is in here because we want to
// be table to take advantage on inlining to make sure its
// implementation is fast.

#include "cpickle.h"  // Python #defines 
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
# include <ochashtable.h>
PTOOLS_BEGIN_NAMESPACE
#endif

// Macro to dump bits directly into buffer
// TODO: Worry about endian
//#define DUMPVALUE(T,N) { *((T*)(mem)) = *((T*)(&N)); mem+=sizeof(T); } 
//#define PROTO2DUMP(CODE,T,N) { *mem++ = CODE; DUMPVALUE(T,N); }

#define HOST_BYTE_ORDER_LITTLE_ENDIAN



// If you need to swap bytes, use these macros
#define DUMP8_SWAP(N,MEM) { char*b=(char*)&N; char*mmm=(char*)MEM; mmm[0]=b[7]; mmm[1]=b[6]; mmm[2]=b[5]; mmm[3]=b[4]; mmm[4]=b[3]; mmm[5]=b[2]; mmm[6]=b[1]; mmm[7]=b[0]; } 
#define DUMP4_SWAP(N, MEM) { char*b=(char*)&N; char*mmm=(char*)MEM; mmm[0]=b[3]; mmm[1]=b[2]; mmm[2]=b[1]; mmm[3]=b[0]; } 
#define DUMP2_SWAP(N, MEM) { char*b=(char*)&N; char*mmm=(char*)MEM; mmm[0]=b[1]; mmm[1]=b[0]; }

#define LOAD8_SWAP(MEM, N) { char*b=(char*)&N; char*mmm=(char*)MEM; b[7]=mmm[0]; b[6]=mmm[1]; b[5]=mmm[2]; b[4]=mmm[3]; b[3]=mmm[4]; b[2]=mmm[5]; b[1]=mmm[6]; b[0]=mmm[7]; } 
#define LOAD4_SWAP(MEM, N) { char*b=(char*)&N; char*mmm=(char*)MEM; b[3]=mmm[0]; b[2]=mmm[1]; b[1]=mmm[2]; b[0]=mmm[3]; } 
#define LOAD2_SWAP(MEM, N) { char*b=(char*)&N; char*mmm=(char*)MEM;b[1]=mmm[0]; b[0]=mmm[1]; }

// If you need just MEMory dump, use these
#if defined(OSF1_) 
// Alphas can have unaligned memory accesses
#define DUMP8_FORWARD(N, MEM) { char*b=(char*)&N; char* mmm=(char*)MEM; mmm[0]=b[0]; mmm[1]=b[1]; mmm[2]=b[2]; mmm[3]=b[3]; mmm[4]=b[4]; mmm[5]=b[5]; mmm[6]=b[6]; mmm[7]=b[7]; } 
#define DUMP4_FORWARD(N, MEM) { char*b=(char*)&N; char* mmm=(char*)MEM; mmm[0]=b[0]; mmm[1]=b[1]; mmm[2]=b[2]; mmm[3]=b[3]; }  
#define DUMP2_FORWARD(N, MEM) { char*b=(char*)&N; char* mmm=(char*)MEM; mmm[0]=b[0]; mmm[1]=b[1]; }  


#else 
// This tends to be faster, as you do bulk memory xfers
#define DUMP8_FORWARD(N, MEM) { *((int_u8*)(MEM)) = *((int_u8*)(&N));}
#define DUMP4_FORWARD(N, MEM) { *((int_u4*)(MEM)) = *((int_u4*)(&N));}
#define DUMP2_FORWARD(N, MEM) { *((int_u2*)(MEM)) = *((int_u2*)(&N));}

#endif

#define DUMP1(N,MEM) { *MEM = *((char*)&N); }

// This is a compile-time entity for speed: Although a DUMP from
// little endian is the same as a LOAD from little endian, to
// emphasize we have both set of macros.

#if defined(HOST_BYTE_ORDER_LITTLE_ENDIAN)

# define DUMP_AS_BIG_ENDIAN8(N,MEM) DUMP8_SWAP(N,MEM)
# define DUMP_AS_BIG_ENDIAN4(N,MEM) DUMP4_SWAP(N,MEM)
# define DUMP_AS_BIG_ENDIAN2(N,MEM) DUMP2_SWAP(N,MEM)
# define DUMP_AS_BIG_ENDIAN1(N,MEM) DUMP1(N,MEM)

# define DUMP_AS_LITTLE_ENDIAN8(N,MEM) DUMP8_FORWARD(N,MEM)
# define DUMP_AS_LITTLE_ENDIAN4(N,MEM) DUMP4_FORWARD(N,MEM)
# define DUMP_AS_LITTLE_ENDIAN2(N,MEM) DUMP2_FORWARD(N,MEM)
# define DUMP_AS_LITTLE_ENDIAN1(N,MEM) DUMP1(N,MEM)

# define LOAD_FROM_BIG_ENDIAN8(MEM,N) LOAD8_SWAP(MEM, N)
# define LOAD_FROM_BIG_ENDIAN4(MEM,N) LOAD4_SWAP(MEM, N)
# define LOAD_FROM_BIG_ENDIAN2(MEM,N) LOAD2_SWAP(MEM, N)
# define LOAD_FROM_BIG_ENDIAN1(MEM,N) DUMP1((*(MEM)), (&(N)))

# define LOAD_FROM_LITTLE_ENDIAN8(MEM,N) DUMP8_FORWARD((*(MEM)), (&(N)))
# define LOAD_FROM_LITTLE_ENDIAN4(MEM,N) DUMP4_FORWARD((*(MEM)), (&(N)))
# define LOAD_FROM_LITTLE_ENDIAN2(MEM,N) DUMP2_FORWARD((*(MEM)), (&(N)))
# define LOAD_FROM_LITTLE_ENDIAN1(MEM,N) DUMP1((*(MEM)), (&(N)))


#elif (HOST_BYTE_ORDER_BIG_ENDIAN) 

# define DUMP_AS_BIG_ENDIAN8(N,MEM) DUMP8_FORWARD(N,MEM)
# define DUMP_AS_BIG_ENDIAN4(N,MEM) DUMP4_FORWARD(N,MEM)
# define DUMP_AS_BIG_ENDIAN2(N,MEM) DUMP2_FORWARD(N,MEM)
# define DUMP_AS_BIG_ENDIAN1(N,MEM) DUMP1(N,MEM)

# define DUMP_AS_LITTLE_ENDIAN8(N,MEM) DUMP8_SWAP(N,MEM)
# define DUMP_AS_LITTLE_ENDIAN4(N,MEM) DUMP4_SWAP(N,MEM)
# define DUMP_AS_LITTLE_ENDIAN2(N,MEM) DUMP2_SWAP(N,MEM)
# define DUMP_AS_LITTLE_ENDIAN1(N,MEM) DUMP1(N,MEM)

# define LOAD_FROM_BIG_ENDIAN8(MEM,N) DUMP8_FORWARD((*(MEM)), (&(N)))
# define LOAD_FROM_BIG_ENDIAN4(MEM,N) DUMP4_FORWARD((*(MEM)), (&(N)))
# define LOAD_FROM_BIG_ENDIAN2(MEM,N) DUMP2_FORWARD((*(MEM)), (&(N)))
# define LOAD_FROM_BIG_ENDIAN1(MEM,N) DUMP1((*(MEM)), (&(N)))

# define LOAD_FROM_LITTLE_ENDIAN8(MEM,N) DUMP8_SWAP((*(MEM)), (&(N)))
# define LOAD_FROM_LITTLE_ENDIAN4(MEM,N) DUMP4_SWAP((*(MEM)), (&(N)))
# define LOAD_FROM_LITTLE_ENDIAN2(MEM,N) DUMP2_SWAP((*(MEM)), (&(N)))
# define LOAD_FROM_LITTLE_ENDIAN1(MEM,N) DUMP1((*(MEM)), (&(N)))


#else 

# error 667 // Need to define the endianness of your platform

#endif



// These premables are important so we can mark when we've seen
// certain strings.  To avoid excessive string compares,
// we wll be memoize where these strings live.
static char ComplexPreamble[] = "c__builtin__\ncomplex\n";
static string ComplexPreambleAsString(ComplexPreamble, 
				      sizeof(ComplexPreamble)-1);

static char ArrayPreamble[] = "carray\narray\n";
static string ArrayPreambleAsString(ArrayPreamble, 
				    sizeof(ArrayPreamble)-1);

static char NumericPreamble[] = "cNumeric\narray_constructor\n";
static string NumericPreambleAsString(NumericPreamble, 
				      sizeof(NumericPreamble)-1);

static char OrderedDictPreamble[] = "ccollections\nOrderedDict\n";
static string OrderedDictPreambleAsString(OrderedDictPreamble, 
					  sizeof(OrderedDictPreamble)-1);





// The DumpContext allows use to keep a context of relevant variables
// as we dump without using any globals
struct DumpContext_ {

  // The memory we are dumping to.
  char* mem;

  // Special cases for preambles for "special"s
  bool complex_preamble_dumped;
  int  complex_handle;
  bool array_preamble_dumped;
  int  array_handle;
  bool numeric_preamble_dumped;
  int  numeric_handle;
  bool ordereddict_preamble_dumped;
  int  ordereddict_handle;

  bool NumPyReconstruct_dumped;
  int  NumPyReconstruct_handle;
  bool NumPyNDArray_dumped;
  int  NumPyNDArray_handle;
  bool NumPyDtype_dumped;
  int  NumPyDtype_handle;

  // Associate pointer handles (the key) with small, incremental
  // integer handles (the value).  As we serialize Proxies, we want to
  // see if have already serialized it.  If we have, we will pull out
  // the int_handle associated with the pointer from the Proxy.
#if defined(OC_USE_OC)
  AVLHashT<void*, int, 8> handles;
#endif
  // Only use the handles (currently) when using PicklingTools or OC. Not M2k

  // Where the current counting is for handles:  This is also
  // the length of the handles_array
  int_u4 current_handle;

  // Current options which affect serialization
  ArrayDisposition_e disposition;
  PicklingIssues_e pickling_issues;

  bool compat_; // true = perform conversion from OTabTupBigInt to TabArrStr;

  // Create a handle context
  DumpContext_ (char* m, ArrayDisposition_e dis, PicklingIssues_e issues) : 
    mem(m), 
    complex_preamble_dumped(false), 
    complex_handle(-1),
    array_preamble_dumped(false), 
    array_handle(-1),
    numeric_preamble_dumped(false) ,
    numeric_handle(-1),
    ordereddict_preamble_dumped(false) ,
    ordereddict_handle(-1),
    NumPyReconstruct_dumped(false),
    NumPyReconstruct_handle(-1),
    NumPyNDArray_dumped(false),
    NumPyNDArray_handle(-1),
    NumPyDtype_dumped(false),
      NumPyDtype_handle(-1),
    current_handle(1),
    disposition(dis),
    pickling_issues(issues),
    compat_(issues==CONVERT_OTAB_TUP_ARR__TO__TAB_ARR_STR)
  { }
}; // DumpContext_


// ///////////////////////////////////////////// PreambleDumper

// Forward
inline void P2DumpCodeAndInt_ (int_u4 i4, char short_code, char long_code,
			       DumpContext_& dc);

#if defined(THE_IDEA_OF_THE_CODE)
// ... it's the same code over and over for the different handles,
// so we construct a macro 
void PreambleDumper(DumpContext_& dc) 
{
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
}

size_t BytesPremableDumper(DumpContext_& dc)
{
  if (dc.numeric_preamble_dumped) { 
    bytes += 2;
  } else {
    dc.numeric_handle = dc.current_handle++;
    dc.numeric_preamble_dumped = true;
    bytes += sizeof(NumericPreamble)+2;
  }
}
#endif


#define BYTESPREAMBLEDUMPERDEFINE(NAME) size_t BytesPreamble##NAME(DumpContext_& dc) { size_t bytes = 0;  \
  if (dc.NAME##_dumped) { bytes += 2;} else { \
    dc.NAME##_handle = dc.current_handle++; dc.NAME##_dumped = true; \
    bytes += sizeof(NAME##Preamble)+2; } return bytes; }

#define PREAMBLEDUMPERDEFINE(NAME) inline void PreambleDumper##NAME(DumpContext_& dc) { if (dc.NAME##_dumped) { \
     P2DumpCodeAndInt_(dc.NAME##_handle, PY_BINGET, PY_LONG_BINGET, dc); } \
  else { dc.NAME##_handle = dc.current_handle++; dc.NAME##_dumped = true; \
    size_t len=sizeof(NAME##Preamble)-1; \
    memcpy(dc.mem, NAME##Preamble, len); dc.mem += len; \
    P2DumpCodeAndInt_(dc.NAME##_handle, PY_BINPUT, PY_LONG_BINPUT, dc); } }



// Define the dumper for the the NumPy reconstruct 
static char NumPyReconstructPreamble[]="cnumpy.core.multiarray\n_reconstruct\n";
static string NumPyReconstructAsString(NumPyReconstructPreamble, 
				       sizeof(NumPyReconstructPreamble)-1);
BYTESPREAMBLEDUMPERDEFINE(NumPyReconstruct)
PREAMBLEDUMPERDEFINE(NumPyReconstruct)


// Define the GLOBAL dumper for the NumPyNDArray
static char NumPyNDArrayPreamble[] = "cnumpy\nndarray\n";
static string NumPyNDArrayAsString(NumPyNDArrayPreamble, 
				   sizeof(NumPyNDArrayPreamble)-1);
BYTESPREAMBLEDUMPERDEFINE(NumPyNDArray)
PREAMBLEDUMPERDEFINE(NumPyNDArray)


// Define the GLOBAL dumper for dtype
static char NumPyDtypePreamble[] = "cnumpy\ndtype\n";
static string NumPyDtypeAsString(NumPyDtypePreamble, 
				 sizeof(NumPyDtypePreamble)-1);

BYTESPREAMBLEDUMPERDEFINE(NumPyDtype)
PREAMBLEDUMPERDEFINE(NumPyDtype)



inline void dumpString (const string& s, DumpContext_& dc);
#if defined(OC_USE_OC) && !defined(OC_USE_OC_STRING)
inline void dumpString (const OCString& s, DumpContext_& dc);
#endif

// Helper function to dump an integer as compressed integer
inline void dump4ByteInteger (int_4 i4, DumpContext_& dc) 
{
  // TODO:  Should we be handling (when pickling_issues)
  // the 'I' case?  Probably not, as we don't know if the
  // Python it's coming is 8 or 4 bytes

  char* mem = dc.mem;
  // So, the integers dump in LITTLE ENDIAN form always ...

  char buff[4];
  // Bring the integer into a buffer as little endian, then 
  // can worry about compressing number into smaller bytes
  DUMP_AS_LITTLE_ENDIAN4(i4, buff);
  if (buff[3]==0 && buff[2]==0) {
    if (buff[1]==0) {
      *mem++ = PY_BININT1; DUMP1(buff[0], mem); mem+=1;
    } else {
      *mem++ = PY_BININT2; DUMP2_FORWARD(buff[0], mem); mem+=2;
    }
  } else {
    *mem++ = PY_BININT; DUMP4_FORWARD(buff[0], mem); mem+=4;
  }

  dc.mem = mem;
}
inline size_t BytesToDump4ByteInteger () { return 5; }

// Helper function to dump an integer as compressed integer
inline void dump8ByteInteger (int_8 i8, DumpContext_& dc)
{
  char* mem = dc.mem;

  // So, the integers dump in LITTLE ENDIAN form always ...

  // Bring the integer into a buffer as little endian, them 
  // can worry about compressing into fewer bytes
  char buff[8];
  DUMP_AS_LITTLE_ENDIAN8(i8, buff);
  if ((buff[7]==0) && (buff[6]==0) && (buff[5]==0) && (buff[4]==0) && 
      ((buff[3]&128)==0)) {  // > max int for an int_4
    dump4ByteInteger(i8, dc);
    return;  // The dump4byteinteger sets memory for you!!  Don't reset below!
  } else if (dc.pickling_issues==AS_PYTHON_2_2) {
    string i8_as_string = Stringize(i8);
    *mem++ = 'L'; // older protocol ...
    const size_t len = i8_as_string.length();
    const char* data = i8_as_string.data();
    for (size_t ii=0; ii<len; ii++) *mem++=*data++;
    *mem++ = 'L';
    *mem++ = '\n';
  } else {
    // Dump as a Python LONG, which can be any number of bytes,
    // in this case it is 8 
    *mem++ = PY_LONG1; 
    *mem++ = sizeof(int_u8); DUMP8_FORWARD(buff[0], mem); mem+=8;
  }

  dc.mem = mem;
}

inline size_t BytesToDump8ByteInteger (DumpContext_& dc) 
{ return dc.pickling_issues==AS_PYTHON_2_2 ? 22 : 10; }

// Helper function to dump an integer as compressed integer
inline void dump8ByteUnsignedInteger (int_u8 i8, DumpContext_& dc)
{
  char* mem = dc.mem;

  // So, the integers dump in LITTLE ENDIAN form always ...

  // Bring the integer into a buffer as little endian, them 
  // can worry about compressing into fewer bytes
  char buff[8];
  DUMP_AS_LITTLE_ENDIAN8(i8, buff);
  if ((buff[7]==0) && (buff[6]==0) && (buff[5]==0) && (buff[4]==0)) {
    dump4ByteInteger(i8, dc);
    return;  // Dump does right thing, just need to leave before dc.mem = mem
  } else if (dc.pickling_issues==AS_PYTHON_2_2) {
    string i8_as_string = Stringize(i8);
    *mem++ = 'L'; // older protocol ...
    const size_t len = i8_as_string.length();
    const char* data = i8_as_string.data();
    for (size_t ii=0; ii<len; ii++) *mem++=*data++;
    *mem++ = 'L';
    *mem++ = '\n';
  } else {
    // Dump as a Python LONG, which can be any number of bytes,
    // in this case it is 8 or 9
    if (int_u1(buff[0])>127) { // Negative has to be 9
      *mem++ = PY_LONG1; 
      *mem++ = 9; DUMP8_FORWARD(buff[0], mem); mem+=8; *mem++='\0';
    } else {
      *mem++ = PY_LONG1; 
      *mem++ = sizeof(int_u8); DUMP8_FORWARD(buff[0], mem); mem+=8;
    }
  }

  dc.mem = mem;
}
inline size_t BytesToDump8ByteUnsignedInteger (DumpContext_& dc) 
{  return dc.pickling_issues==AS_PYTHON_2_2 ? 22 : 11; }

// Helper function to dump a float
inline void dump8ByteFloat (real_8 r8, DumpContext_& dc)
{
  char* mem = dc.mem;

  // So, the floats dump in BIG ENDIAN form always ... 
  *mem++=PY_BINFLOAT; 
  DUMP_AS_BIG_ENDIAN8(r8, mem); 
  mem+=8; 
  
  dc.mem = mem;
}
inline size_t BytesToDump8ByteFloat () { return 9; }

// Helper function to dump a 1-byte integer or 4-byte integer (for
// puts handles)
inline void P2DumpCodeAndInt_ (int_u4 i4, char short_code, char long_code,
			       DumpContext_& dc)
{
  // Depending on the number of handles, dump either 1 byte integer
  // for the BINPUT (q) or LONG_BINPUT (r)

  char* mem = dc.mem;
  // So, the integers dump in LITTLE ENDIAN form always ...
  char buff[4];
  // Bring the integer into a buffer as little endian, then 
  // can worry about compressing number into smaller bytes
  DUMP_AS_LITTLE_ENDIAN4(i4, buff);
  if (i4<256) {
    *mem++ = short_code; DUMP1(buff[0], mem); mem+=1;
  } else {
    *mem++ = long_code; DUMP4_FORWARD(buff[0], mem); mem+=4;
  }
  dc.mem = mem;
}


// Helper function to dump a float
inline void dump16ByteComplex (const complex_16& c16, DumpContext_& dc)
{
  // The preamble 
  if (dc.complex_preamble_dumped) {
    P2DumpCodeAndInt_(dc.complex_handle, PY_BINGET, PY_LONG_BINGET, dc);
  } else {
    // Get new memoize number
    dc.complex_handle = dc.current_handle++;
    dc.complex_preamble_dumped = true;

    // And copy in the original (plus a memo)
    memcpy(dc.mem, ComplexPreamble, sizeof(ComplexPreamble)-1);
    dc.mem += sizeof(ComplexPreamble)-1;
    P2DumpCodeAndInt_(dc.complex_handle, PY_BINPUT, PY_LONG_BINPUT, dc);
  }

  if (dc.pickling_issues==AS_PYTHON_2_2) *(dc.mem)++ = '(';
  dump8ByteFloat(c16.re, dc);
  dump8ByteFloat(c16.im, dc);
  *(dc.mem)++ = (dc.pickling_issues==AS_PYTHON_2_2) ? 't' : PY_TUPLE2;
  *(dc.mem)++ = PY_REDUCE;
}

inline size_t BytesToDump16ByteComplex (DumpContext_& dc) 
{
  size_t bytes = BytesToDump8ByteFloat()*2 + 2 + 1; // floats*2 + TUPLE2 + REDUCE
  if (!dc.complex_preamble_dumped) { 
    bytes += 2+sizeof(ComplexPreamble); // Preamble itself + q\01
  } else {
    dc.complex_preamble_dumped = true;
    bytes += 2;   // BINGET + handle
  }
  return bytes;
}

inline void dumpBool (bool b, DumpContext_& dc)
{
  char* mem = dc.mem;
  if (dc.pickling_issues==AS_PYTHON_2_2) {
    *mem++ = PY_BININT1;
    *mem++ = b ? '\x01' : '\x00';
  } else {
    *mem++ = b ? PY_NEWTRUE : PY_NEWFALSE;
  }
  dc.mem = mem;
}
inline size_t BytesToDumpBool () { return 1+1; }


inline void dumpCString (const char* cstr, size_t len, DumpContext_& dc)
{
  // Dump header for string
  P2DumpCodeAndInt_(len, PY_SHORT_BINSTRING, PY_BINSTRING, dc);
  
  // Put the string in memory
  memcpy(dc.mem, cstr, len);
  dc.mem+= len;
}

inline size_t BytesToDumpCString (const char* /*cstr*/, size_t bytes)
{
  if (bytes>255) {
    bytes += 5;
  } else {
    bytes += 2;
  }
  return bytes;
}

inline void dumpString (const string& s, DumpContext_& dc)
{
  const size_t len = s.length();
  return dumpCString(s.data(), len, dc);
}

inline size_t BytesToDumpString (const string& s)
{
  const size_t len = s.length();
  return BytesToDumpCString(s.data(), len);
}

#if defined(OC_USE_OC) && !defined(OC_USE_OC_STRING) 
inline void dumpString (const OCString& s, DumpContext_& dc)
{
  const size_t len = s.length();
  return dumpCString(s.data(), len, dc);
}

inline size_t BytesToDumpString (const OCString& s)
{
  const size_t len = s.length();
  return BytesToDumpCString(s.data(), len);
}
#endif


template <class OT>
char* topleveldump_ (const OT& v, char* mem, ArrayDisposition_e dis,
		     PicklingIssues_e issues = ABOVE_PYTHON_2_2)
{
  if (issues!=AS_PYTHON_2_2) {
    *mem++ = PY_PROTO;     // Preamble:  PROTOCOL!!!
    *mem++ = '\x02';    // Pickling protocol #2
  }

  DumpContext_ dc(mem, dis, issues);
  P2DumpValue(v, dc);
  
  *(dc.mem)++ = PY_STOP;
  return dc.mem;
}

template <class OT>
size_t TopLevelBytesToDump (const OT& v, ArrayDisposition_e dis=AS_LIST,
			    PicklingIssues_e issues = ABOVE_PYTHON_2_2)
{
  DumpContext_ dc(0, dis, issues);
  return 3 + BytesToDumpVal(v, dc);
}



template <class OT>
char* P2DumpSingleVal (const OT& v, char* mem, ArrayDisposition_e dis,
		       PicklingIssues_e issues = ABOVE_PYTHON_2_2)
{
  DumpContext_ dc(mem, dis, issues);
  P2DumpValue(v, dc);
  return dc.mem;
}

template <class OT>
size_t P2BytesToDumpSingleVal (const OT& v, ArrayDisposition_e dis=AS_LIST,
			       PicklingIssues_e issues = ABOVE_PYTHON_2_2)
{
  DumpContext_ dc(0, dis, issues);
  return BytesToDumpVal(v, dc);
}








// macro to help when looking for longish strings on input
#define P2EXPECT_(MESG, MEM) (memcmp(MESG,MEM,sizeof(MESG)-1)==0 ? MEM+sizeof(MESG)-1 : ((char*)(p2error_(MESG))) )


// As loads happen, they change the environment
struct LoadContext_ {

  // When we memoize things, they come out in order.  This allows us
  // to look up what it referred to (int_handle) and get back the ptr
  // (to some Val or string).
  Array<void*> memoize_table;  // This is either an Val* or a string*
  Array<bool>  is_string;      // Val* or string* 
  int handle;                  // current handle: negatives indicate no handle

  // Where we are currently in memory in the deserialization process
  char* mem;

  // If we encounter a standard preamble, special case code for them
  int complex_preamble_handle;
  int array_preamble_handle;
  int numeric_preamble_handle;
  int ordereddict_preamble_handle;

  LoadContext_ (Array<void*>& t, Array<bool>& is, char* m) : 
    memoize_table(t), 
    is_string(is),
    handle(-1), 
    mem(m),
    complex_preamble_handle(-1), 
    array_preamble_handle(-1), 
    numeric_preamble_handle(-1),
    ordereddict_preamble_handle(-1) { }
    
}; // LoadContext_


// When we have a string that starts 
#define P2_COMPLEX_HANDLE -55
#define P2_ARRAY_HANDLE   -66
#define P2_NUMERIC_HANDLE -77
#define P2_ORDEREDDICT_HANDLE -88


inline void handleAPut_ (void* ovp, LoadContext_& lc, bool is_string=false)
{
  char* mem = lc.mem;
  int handle = -1;

  // If we are putting, which means we want to "remember this" in case
  // it is referred to later (i.e., aliased later). 
  if (*mem==PY_BINPUT) {
    mem+=1;
    handle = unsigned(int_u1(*mem++)); // Make sure always positive
  } else if (*mem==PY_LONG_BINPUT) {
    mem+=1;
    LOAD_FROM_LITTLE_ENDIAN4(mem, handle);
    mem+=4;
  } else { // Get out, not put, no handle to put in
    lc.handle = -1;
    return;
  }

  // memoize this, if it needed to be remembered
  if (handle>=0) {
    if (handle>=int(lc.memoize_table.length())) {
      lc.memoize_table.append(0);
      lc.is_string.append(false);
    }
    lc.memoize_table[handle] = ovp; // Install
    lc.is_string[handle]     = is_string;

    // Special case: complex, Numeric, arrays, ordereddict preamble
    switch (lc.handle) {
    case P2_COMPLEX_HANDLE:
      lc.memoize_table[handle]   = &ComplexPreambleAsString; 
      lc.is_string[handle]       = true;
      lc.complex_preamble_handle = handle;
      break;
    case P2_ARRAY_HANDLE:
      lc.memoize_table[handle]   = &ArrayPreambleAsString;
      lc.is_string[handle]       = true;
      lc.array_preamble_handle   = handle;
      break;
    case P2_NUMERIC_HANDLE:
      lc.memoize_table[handle]   = &NumericPreambleAsString;
      lc.is_string[handle]       = true;
      lc.numeric_preamble_handle = handle;
      break;
    case P2_ORDEREDDICT_HANDLE:
      lc.memoize_table[handle]   = &OrderedDictPreambleAsString;
      lc.is_string[handle]       = true;
      lc.ordereddict_preamble_handle   = handle;
      break;
    }
  }

  lc.handle = handle;
  lc.mem = mem;
}


template <class OT, class STR>
inline void* handleAGet_ (OT& ov, LoadContext_& lc, STR*)
{
  char* mem = lc.mem;
  void* vp = NULL;
  bool string_was_memoized = false;
  char tag = *mem++;
  if (tag==PY_BINGET) {             // Lookup via small integer
    lc.handle = unsigned(int_u1(*mem++)); 
    vp = lc.memoize_table[lc.handle];
    string_was_memoized = lc.is_string[lc.handle];
  } else if (tag==PY_LONG_BINGET) { // Lookup via integer
    LOAD_FROM_LITTLE_ENDIAN4(mem, lc.handle); 
    mem+=4;  
    vp = lc.memoize_table[lc.handle];
    string_was_memoized = lc.is_string[lc.handle];
  } else {
    p2error_("mishandle of a get");
  }

  // String to memoize
  if (string_was_memoized) {
    ov = *((STR*)vp);  // uncast and get the string pointer
  } else {
    OT* otp = (OT*)vp; // uncast and get the OT (Val or OpalValue) pointer
    HandleGetAssignment(ov, *otp);
  }

  lc.mem = mem;
  return vp;
}


// Sometimes we want to know the length of the string before it comes
// down so we can do some optimizations.  
inline int_u4 loadStringLength_ (LoadContext_& lc)
{
  int_u4 len = 0;
  char string_type = *(lc.mem)++;
  if (string_type==PY_SHORT_BINSTRING) {
    len = unsigned(int_u1(*(lc.mem)++));
  } else if (string_type==PY_BINSTRING) {
    int_u4 len4; LOAD_FROM_LITTLE_ENDIAN4(lc.mem, len4); lc.mem+= 4;
    len = len4;
  } else {
    p2error_("Unknown string type");
  }
  return len;
}

template <class STR>
inline void finishLoadingString_ (STR* sp, size_t len, LoadContext_& lc)
{
  new (sp) STR(lc.mem, len);
  lc.mem += len;
  handleAPut_(sp, lc, true);        // in case PUT
}

template <class STR>
inline void P2LoadString (STR* sp, LoadContext_& lc)
{
  size_t len = loadStringLength_(lc);
  finishLoadingString_(sp, len, lc);
}

template <class STR>
inline void finishLoadingString_ (STR& s, size_t len, LoadContext_& lc)
{
  s=STR(lc.mem, len);
  lc.mem += len;
  handleAPut_(&s, lc, true);        // in case PUT
}

template<class STR>
inline void P2LoadString (STR& s, LoadContext_& lc)
{
  size_t len = loadStringLength_(lc);
  finishLoadingString_(s, len, lc);
}


inline void loadc16_ (complex_16& c16, LoadContext_& lc, 
		      bool saw_memoized_cx_preamble=false) 
{
  // If the preamble was memoized, then we saw the preamble already as a get
  if (!saw_memoized_cx_preamble) {
    lc.mem = P2EXPECT_(ComplexPreamble, lc.mem);
    lc.handle = P2_COMPLEX_HANDLE;
    handleAPut_(0, lc, true);
  }

  // get the raw data
  if (*(lc.mem)=='(') { // Pickling issues with 2.2 needs
    lc.mem++;
  }
  if (*(lc.mem)++!='G') p2error_("expected G for the real part of a complex");
  LOAD_FROM_BIG_ENDIAN8(lc.mem, c16.re); lc.mem+=8;
  if (*(lc.mem)++!='G') p2error_("expected G for the real part of a complex");
  LOAD_FROM_BIG_ENDIAN8(lc.mem, c16.im); lc.mem+=8;

  char tag = *(lc.mem)++;
  if (tag!=PY_TUPLE2 && tag!='t') p2error_("Expected tuple in loading complex");
  if (*(lc.mem)++ != PY_REDUCE)   p2error_("Expected reduce in loading complex");
}


// Saw a stringized int, which means it was from a machine where
// a long was 8 bytes
template <class T>
inline char* loadSomeInt_ (T& res, char* mem)
{ 
  T result = 0;
  T neg = 1;
  if (*mem=='-') { neg = T(-1); mem++; }

  char c;
  int_8 val;
  while (isdigit(c=*mem++)) { // This passes the last non digit (\n or L)
    val = c-'0';
    result = result*10 + val;
  }

  res = neg*result;
  return mem;
}

// These routines handle the rare case when we have been given data
// from an older version of Python who gives us integers as strings
inline int_8 loadINT_ (LoadContext_& lc)
{
  // if (*mem++!='I') p2error_("Expected I for an int_8");
  int_u8 res;
  lc.mem = loadSomeInt_(res, lc.mem);
  // got past \n 
  return res;
}

inline int_8 loadLONG_ (LoadContext_& lc)
{
  // if (*mem++!='L') p2error_("Expected L for an int_8");
  int_8 res;
  lc.mem = loadSomeInt_(res, lc.mem);
  // got past L, now past \n
  if (*(lc.mem)++!='\n') p2error_("Expected \\n to end an int_8");
  return res;
}

inline int_u8 loadULONG_ (LoadContext_& lc)
{
  // if (*mem++!='L') p2error_("Expected L for an int_8");
  int_u8 res;
  lc.mem = loadSomeInt_(res, lc.mem);
  // got past L, now past \n
  if (*(lc.mem)++!='\n') p2error_("Expected \\n to end an int_u8");
  return res;
}


template <class OT>
inline char* toplevelload_ (OT& ov, char* mem)
{
  ov=OT();  // start off right!
  
  // The memoize table allows us to share representations as we unpack
  Array<void*> memoize_table;
  OT empty; // needs start?
  memoize_table.append(&empty);
  Array<bool> is_string_table;
  is_string_table.append(false);
 
  if (*mem == '\x80') { // standard serialization, expect a preamble
    mem = P2EXPECT_("\x80\x02", mem); // PROTO,'\x02';
  } else {
    // Python 2.2 has no preamble for protocol 2
  }

  LoadContext_ lc(memoize_table, is_string_table, mem);
  P2LoadValue(ov, lc);
  if (*(lc.mem)++ != '.') p2error_("expected . at end");
  
  return lc.mem;
}


template <class OT>
char* P2LoadSingleVal (OT& v, char* mem)
{
  v=OT();  // start off right!
  
  // The memoize table allows us to share representations as we unpack
  Array<void*> memoize_table;
  OT empty; // needs start?
  memoize_table.append(&empty);
  Array<bool> is_string_table;
  is_string_table.append(false);

  LoadContext_ lc(memoize_table, is_string_table, mem);
  P2LoadValue(v, lc);

  return lc.mem;
}


// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_END_NAMESPACE
#endif


#define M2P2COMMON_H_
#endif
