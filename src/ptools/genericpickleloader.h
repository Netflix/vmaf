#ifndef GENERICPICKLELOADER_H_

// Complete reimplementation of the Protocol Loading 2.  The codebase
// has a smaller memory footprint, it's just about as fast (faster in
// some cases), it's is easier to read and maintain, and simpler.

#include "cpickle.h"
#include "m2pythontools.h"

OC_BEGIN_NAMESPACE

// A Stack
template <class T>
class Stack {
 public:
  
  // Construct with initial capacity: will resize of course.
  Stack (int initial_len = 256) : impl_(initial_len) { }
    
  // Push something on top of the stack: return a reference to where
  // it is.
  T& push (const T& t)
  {
    int len = impl_.length();
    impl_.append(t);
    return impl_(len);
  }

  // Peek into the stack: 0 is the top, -1 is just below the top, etc.
  // Returns a reference to the element in question.
  T& peek (int where=0) 
  {
    int len = impl_.length();
    return impl_[len+where-1];
  }

  // Pop the top element of the stack.  Returns thing popped.
  T pop () { return impl_.removeLast(); }

  // Pop the top number of elements given.
  void pop (int how_many)
  {
    int len = impl_.length();
    impl_.removeRange(len-how_many, how_many);
  }
  
  // Clean up
  void clear () { impl_.clear(); }

  // The number of elements on the stack
  int length() const { return impl_.length(); }

  // Lets you dig inside the implementation if you have to.
  T& operator[] (int ii) { return impl_[ii]; }

  // Print hook
  ostream& print (ostream& os) { return os << impl_ << endl; }

 protected:
  Array<T> impl_; // A stack is implemented using an array
      
}; // Stack


// Example of how user-defined classes would probably serialize: the
// user-defined classes tend to be built with BUILD instead of REDUCE,
// which passes slightly different arguments to the tuple.  As an
// example, magine a simple Python class:
//
//   class Scott(object) :
//     def __init__(self, n) : self.data = int(n)
//
// A dumps(a,2) (where a= Scott(100)) looks like
// '\x80\x02c__main__\nScott\nq\x01)\x81q\x02}q\x03U\x04dataq\x04Kdsb.'/
//
// type_object: a tuple (string_name, args). i.e., ('__main__\nScott\n', ()) 
// input: a dictionary                       i.e., {'data', 100 }
template <class OBJ>
inline void BuildScottFactoryFunction (const OBJ& type_object, 
				       const OBJ& input,
				       OBJ& /* environment */,
				       OBJ& output_result)
{
  cout << "SCOTT:type_object:" << type_object << endl;
  cout << "SCOTT:input:" << input << endl;

  string name = EXTRACT_STRING(TUP_SUB(type_object,0)); 
  EXTRACT_TUP(args, TUP_SUB(type_object,1));
  cout << "name:" << name << " args:" << args << endl;

  if (name!="__main__\nScott\n") throw runtime_error("Not right name");
  OBJ& result = GENERIC_GET(input, "data");
  output_result = result;
}


// Example: ReduceFactoryFunction for OrderedDict: take a list of
// 2-element Arrs
template <class OBJ>
inline void ReduceOTabFactory (const OBJ& /* name */,
			       const OBJ& tuple, 
	 		       OBJ& env, 
			       OBJ& result)
{  
  // Constructor is a 1-tuple which should have a list of 2-element arrs
  if (tuple.length()!=1) {
    throw runtime_error("Malformed OrderedDict constructor");
  }
  EXTRACT_TUP(u, tuple);
  EXTRACT_LIST(a, TUP_SUB(u,0));
  int len = int(a.length());

  // This code only makes sense for Val
#if defined(OC_USE_OC)    
  if (!env.contains("compatibility")) {
    // Direct conversion to OTab
    OTab& o = result = OTab();
    for (int ii=0; ii<len; ii++) {
      Arr& pair = a(ii);
      o.swapInto(pair(0), pair(1));
    }
  } else 
#endif
  {
    // Convert to a table instead, as we need backwards compatibility
    result = MAKE_DICT();
    EXTRACT_DICT(t, result);
    for (int ii=0; ii<len; ii++) {
      EXTRACT_LIST(p, a(ii));      
      DICT_SWAP_INTO(t, p(0), p(1));
    }
  }
}

// Example: ReduceFactoryFunction for complex: 2-tuple of real_8s
template <class OBJ>
inline void ReduceComplexFactory (const OBJ& /* name */,
				  const OBJ& tuple, 
				  OBJ&,
				  OBJ& result)
{
  // we assume we got the name right
  real_8 rl = TUP_SUB(tuple,0);
  real_8 im = TUP_SUB(tuple,1);
  result = MAKE_COMPLEX(rl, im);
}



// Helper function to convert the string into a data of Array: In case
// we do not support Numeric Arrays, they can be converted to Arrs.
template <class T, class OBJ>
inline void NumericArrayFactoryHelper_ (bool keep_numeric, 
					int elements, T* data, 
					OBJ& result)
{
#if defined(NOTNOW) // defined(OC_USE_OC)
  if (!keep_numeric) {
    Arr& a = result = new Arr(elements);
    a.fill(None);
    OBJ* val_data = a.data();
    for (int ii=0; ii<elements; ii++) {
      val_data[ii] = data[ii];
    }   
  } else {
    Array<T>& a = result = new Array<T>(elements);
    a.expandTo(elements);
    T* val_data = a.data();
    memcpy(val_data, data, elements*sizeof(T));
  }
#else
  if (!keep_numeric) {
    result = MAKE_LIST1(elements);
    EXTRACT_LIST(a,result);
    for (int ii=0; ii<elements; ii++) {
      a.append(MAKE_OBJ_FROM_NUMBER(data[ii]));
    }   
  } else {
    result = MAKE_VECTOR1(T, elements);
    EXTRACT_VECTOR(T, a, result);
    VECTOR_EXPAND(T, a, elements);
    T* val_data = VECTOR_RAW_PTR(T, a);
    memcpy(val_data, data, elements*sizeof(T));
  }
#endif
}

inline size_t ByteLengthNumericTag (char tag)
{
  size_t bytes = 0;
  switch (tag) {
  case '1': bytes=1; break;
  case 'b': bytes=1; break;
  case 's': bytes=2; break;
  case 'w': bytes=2; break;
  case 'i': bytes=4; break;
  case 'u': bytes=4; break;
  case 'l': bytes=8; break;
    //case 'l': bytes='X'; break;
    //case 'b': bytes='b'; break;
  case 'f': bytes=4; break;
  case 'd': bytes=8; break;
  case 'F': bytes=8; break;
  case 'D': bytes=16; break;
  default: throw runtime_error("No corresponding Numeric type for Val type");
  }
  return bytes;
}

// Example: ReduceFactoryFunction for Numeric
#define OC_NUM_FACT(T) { NumericArrayFactoryHelper_(conv,elements,(T*)data,result); break; }
template <class OBJ>
inline void ReduceNumericArrayFactory (const OBJ& /* name */,
				       const OBJ& tuple, 
				       OBJ& env,
				       OBJ& result)
{
  // we assume we got the name right


  EXTRACT_TUP(u, tuple);   //Tup& u = tuple;

  // First element of tuple is a tuple: 
  //           the number of elements in the Numeric Array
  EXTRACT_TUP(ele, TUP_SUB(u, 0)); // Tup& ele = u(0);
  int elements = 1;
  if (ele.length() != 0) {
    OBJ& e=TUP_SUB(ele, 0);
    elements = EXTRACT_INT(e); // This a tuple for some dumb reason
  }

  // Second element of tuple is a string
  //            typecode of tuple (prolly a single char)
  string typecode = EXTRACT_STRING( TUP_SUB(u,1) ); //  string typecode = u(1);
  char type_char = typecode[0]; 

  // Third element is a string: 
  //            the binary data of the array (memcpyed in)
  OBJ& bin_data = TUP_SUB(u,2);
#if defined(OC_USE_OC)  // Optimize for OC, avoid copy
  OCString* ocsp = (OCString*)&bin_data.u.a;
  const char* data = ocsp->data();
#else 
  string str_data = EXTRACT_STRING(bin_data);
  string* ocsp = &str_data;
  const char* data = str_data.data();
#endif

  // Fourth element is a bool
  //            save space (not really used)


  bool conv = DICT_CONTAINS(env, "supportsNumeric") && 
    EXTRACT_BOOL(DICT_GET(env,"supportsNumeric"))==true;


  // If the number of element is NOT equal to what we think the size
  // is, this is probably a 32-bit vs. 64-bit discrepancy: on 32-bit
  // machines, Numeric 'l' is 4 bytes, 8 bytes otherwise.  
  int computed_elements = 
    (ocsp->length()/(ByteLengthNumericTag(type_char)));
  if (elements != computed_elements) {
    if (type_char=='l') { // almost certainally 32-bit vs. 64-bit issue
      // We can catch it here, but the Python on the other end may not
      // get it ... be careful!
      type_char = 'i'; // 4-byte longs
    } else {
      // Huh .. probably just an error serializing .. we will truncate
      // and continue with a warning?
      elements = computed_elements;
      cerr << "Miscomputed number of elements? .. continuing ..." << endl;
    }
  }

  switch (type_char) {
  case '1': OC_NUM_FACT(int_1); 
  case 'b': OC_NUM_FACT(int_u1);
  case 's': OC_NUM_FACT(int_2);
  case 'w': OC_NUM_FACT(int_u2);
  case 'i': OC_NUM_FACT(int_4);
  case 'u': OC_NUM_FACT(int_u4);
  case 'l': OC_NUM_FACT(int_8);
    //case 'b': OC_NUM_FACT(bool);
  case 'f': OC_NUM_FACT(real_4);
  case 'd': OC_NUM_FACT(real_8);
  case 'F': OC_NUM_FACT(complex_8);
  case 'D': OC_NUM_FACT(complex_16);
  default: throw runtime_error("Unknown typecode");
  }

}

#if defined(HARDCODE) && defined(OC_USE_OC)
// Example: ReduceFactoryFunction for Array module
# define OC_ARRAY_FACT(T) { result = Array<T>(); Array<T>& a=result; \
if (data.tag=='a'){ a.expandTo(length/sizeof(T)); OCString*ocp=(OCString*)&data.u.a;memcpy(a.data(),ocp->data(), ocp->length());} \
else { a.expandTo(length); Arr& ss=data; for (int ii=0; ii<length; ii++) { a[ii] = ss[ii]; } } }

#endif

// Python 2.6 reduces strings
template <class T, class OBJ>
inline void ReduceArrayHelper26_ (const OBJ& input, OBJ& result)
{
  string arr_data = EXTRACT_STRING(input);
  int elements = arr_data.length()/sizeof(T);

  result = MAKE_VECTOR1(T, elements);
  EXTRACT_VECTOR(T, a, result);
  VECTOR_EXPAND(T,a,elements);
  T* d = VECTOR_RAW_PTR(T, a);
  memcpy(d, arr_data.data(), arr_data.length());
}

// Python 2.7 uses list
template <class T, class OBJ>
inline void ReduceArrayHelper27_ (const OBJ& input, OBJ& result)
{
  EXTRACT_LIST(lst, input);
  int elements = LIST_LENGTH(lst);

  result = MAKE_VECTOR1(T, elements);
  EXTRACT_VECTOR(T, a, result);
  VECTOR_EXPAND(T,a,elements);
  T* d = VECTOR_RAW_PTR(T, a);
  for (int ii=0; ii<elements; ii++) {
    d[ii] = EXTRACT_T(T, LIST_SUB(lst,ii));
  }
}

// Choose between
template <class T, class OBJ>
inline void ReduceArrayHelper_ (OBJ& input, OBJ& result)
{
  bool is_str = IS_STRING(input);
  bool is_lst = IS_LIST(input);
  if (is_str) {
    ReduceArrayHelper26_<T>(input, result);
  } else if (is_lst) {
    ReduceArrayHelper27_<T>(input, result);
  } else {
    throw runtime_error("Array expects either list or string");
  }
}
  

# define OC_ARRAY_FACT(T) ReduceArrayHelper_<T>(input, result);
// Argh!
// 2.6 dumps different than 2.7 which dumps different than 2.4
// -2,7 expects a Python list of each of the elements, individually labelled
// -2.6 expects a string which you can bitblit
// -2.4 doesn't even work
template <class OBJ>
inline void ReduceArrayFactory (const OBJ& /* name */,
				const OBJ& tuple, 
				OBJ& env,
				OBJ& result)
{
  // First element of tuple:
  //     string of typecode
  string typecode = EXTRACT_STRING(TUP_SUB(tuple,0));
  char type_char = typecode[0];
  // Second element of tuple:
  //     list (2.7) or string (2.6)
  OBJ&  input =     TUP_SUB(tuple,1);
  switch (type_char) {
  case 'c': OC_ARRAY_FACT(int_1);  break;
  case 'b': OC_ARRAY_FACT(int_1);  break;
  case 'B': OC_ARRAY_FACT(int_u1); break;
  case 'h': OC_ARRAY_FACT(int_2);  break;
  case 'H': OC_ARRAY_FACT(int_u2); break;
  case 'i': OC_ARRAY_FACT(int_4);  break;// 2 or 4 bytes?
  case 'I': OC_ARRAY_FACT(int_u4); break;// 2 or 4 bytes?
  case 'l': OC_ARRAY_FACT(int_8);  break;// 4 or 8 bytes?
  case 'L': OC_ARRAY_FACT(int_8);  break;// 4 or 8 bytes?
  case 'f': OC_ARRAY_FACT(real_4); break;
  case 'd': OC_ARRAY_FACT(real_8); break;
  default: throw runtime_error("Unsupported array type");
  }
}



#define NUMPYARRAYCREATE(T) { result=Array<T>(shape); Array<T>&a=result; a.expandTo(shape); outdata=a.data(); memcpy(outdata, raw_data, raw_data_bytes); } 
void dispatchCreateArray_ (int shape, const string& type_desc,
			   const char* raw_data, int raw_data_bytes, 
			   const string& endian,
			   Val& result)
{
  // Parse type description field, usallu something like 'u4', 'i8', 
  // where the first letter is the type, the next letter is the length of
  // the type.
  if (type_desc.length() <2 ) {
    throw runtime_error("Illegal type string on a NumPy array:"+ type_desc);
  }
  char type_tag = type_desc[0];
  int_u4 type_len = StringToInt<int_u4>(&type_desc[1], type_desc.length()-1);

  // Create appropriate array
  bool is_cx = false;
  void* outdata;  // Filled in by MACRO with pointer to data
  switch (type_tag) {
  case 'i': 
    switch(type_len) {
    case 1: NUMPYARRAYCREATE(int_1); break;
    case 2: NUMPYARRAYCREATE(int_2); break;
    case 4: NUMPYARRAYCREATE(int_4); break;
    case 8: NUMPYARRAYCREATE(int_8); break;
    default: throw runtime_error("Unknown type string on NumPy:"+ type_desc);
    }
    break; // integers 
  case 'u':
    switch(type_len) {
    case 1: NUMPYARRAYCREATE(int_u1); break;
    case 2: NUMPYARRAYCREATE(int_u2); break;
    case 4: NUMPYARRAYCREATE(int_u4); break;
    case 8: NUMPYARRAYCREATE(int_u8); break;
    default: throw runtime_error("Unknown type string on NumPy:"+ type_desc);
    } 
    break; // unsigned integers
  case 'f': 
    switch(type_len) {
    case 4: NUMPYARRAYCREATE(real_4); break;
    case 8: NUMPYARRAYCREATE(real_8); break;
    default: throw runtime_error("Unknown type string on NumPy:"+ type_desc);
    }
    break; // floats
  case 'c':
    is_cx = true;
    switch(type_len) {
    case 8: NUMPYARRAYCREATE(complex_8); break;
    case 16: NUMPYARRAYCREATE(complex_16); break;
    default: throw runtime_error("Unknown type string on NumPy:"+ type_desc);
    } 
    break; // complexes
  case 'b': 
    switch(type_len) {
    case 1: NUMPYARRAYCREATE(bool); break;
    default: throw runtime_error("Unknown type string on NumPy:"+ type_desc);
    }
    break; // bool
  case 'S': 
    break; // string, ie., numpy.character
  default: throw runtime_error("Unknown type string on NumPy:"+ type_desc);
  }
  
  // Make sure endian-ness correct
  bool machine_little_endian = IsLittleEndian();
  bool data_little_endian = (endian == "<");
  if (machine_little_endian != data_little_endian) {
    InPlaceReEndianize((char*)outdata, shape, type_len, is_cx);
  }
}

//  numpy.array([15,16,17], numpy.uint32) pickles as
// '\x80\x02cnumpy.core.multiarray\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x03\x85cnumpy\ndtype\nq\x04U\x02u4K\x00K\x01\x87Rq\x05(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x0c\x0f\x00\x00\x00\x10\x00\x00\x00\x11\x00\x00\x00tb.'
//
// This translates to (after some BUILDs and REDUCES):
// type_object = ('numpy.core.multiarray\n_reconstruct\n', ('numpy\nndarray\n', (0), 'b'))
// tuple = (1, (3), (('numpy\ndtype\n', ('u4', 0, 1)), (3, '<', None, None, None, -1, -1, 0)), False, '\x0f\x00\x00\x00\x10\x00\x00\x00\x11\x00\x00\x00')

inline void BuildNumPyArray_ (const Val& type_object,
			      const Val& tuple, 
			      Val& result)
{
  // Is subtype? TODO
  // Val& is_subtype = tuple(0);  // should always be 1

  // Handle shape
  Val& described_shape = tuple(1);
  int shape = 1;
  if (described_shape.tag == 'u') { // flatten!
    int len = described_shape.length(); 
    if (len!=1) { 
      cerr << "Warning: saw " << len << "dimensional array:flattening" << endl; 
    }
    for (int ii=0; ii<len; ii++) {
      int dim = described_shape[ii];
      shape *= dim; 
    }
  } else {
    shape = described_shape;  // take the execption if I get something weird
  }
  // shape is valid single-dimensional quantity at this point

  // Handle raw data
  Val& vdata = tuple(4);
  OCString* ocp = (OCString*) &vdata.u.s;
  const char* raw_data = ocp->data();
  const int raw_data_bytes = ocp->length();

  // Handle type
  Tup& type = tuple(2);
  string type_desc = type(0)(1)(0);

  // Handle endianness
  const string endian = tuple(2)(1)(1);

  // Create the Array result and plop into result
  dispatchCreateArray_(shape, type_desc, raw_data, raw_data_bytes, endian,
		       result);

}


// Function on the way to building a NumPy array:
// The purpose of this is really to put the "type object"
// of the "numpy\nndarray\n" on the stack so that BuildNumPyNDArray
// gets the right object to start with.
//
// This can get called with REDUCE and/or BUILD.
// When called as part of a BUILD, you are simply building the
// type object of the class in question: this is indicated because the
// first argument is a string (the name of class).
//
// When called as part of REDUCE, the first arg is the type-object,
// which is a tuple.  
inline void BuildNumPyReconstruct (const Val& type_object,
				   const Val& tuple, 
				   Val& env,
				   Val& result)
{
  // REDUCE uses string name to build type-object
  if (type_object.tag=='a') { 
    result = Tup(type_object, tuple); // ... actually building type object
  } 

  // BUILD uses type-object (usually represented as a tuple) to build final
  // array
  else if (type_object.tag=='u') { 
    // Here's where all the real work gets done
    BuildNumPyArray_(type_object, tuple, result);
  }                              // by a tuple

}

// Works with REDUCE and BUILD
inline void BuildNumPyDtype (const Val& type_object,
			     const Val& tuple, 
			     Val& env,
			     Val& result)
{
  // REDUCE uses string name to build a "type" on the stack,
  // so we build a fairly typical representation of a type object:
  // a tuple with the string name of the object, and default params
  if (type_object.tag=='a') { 
    result = Tup(type_object, tuple); // ... actually building type object
  } 

  // BUILD uses type-object (a tuple) to actually build a value representing
  // the type.  In this case, we preserve all info in another tuple: For
  // DTYPES this is important because the "defaults" (captured in the 
  // type object) are important
  else if (type_object.tag=='u') { 
    result = Tup(type_object, tuple);
  }                              // by a tuple

}



// The PickleLoaderImpl is an loader which will load any Protocol, just
// like cPickle.loads(): 0 and 2 have been tested, the same
// infrstaructure should support 1 and 3 later.
template <typename OBJ>
class PickleLoaderImpl {

  // When registering things with the factory, they take in some tuple
  // and return a OBJ: REDUCE tend to be more for built-in complicated
  // types like Numeric, array and complex.  BUILD tends to more for
  // user-defined types.
  typedef void (*FactoryFunction)(const OBJ& name, 
				  const OBJ& input_tuple, 
				  OBJ& environment,  
				  OBJ& output_result);

 public:

  // Construct a Pickleloader with the given input.
  PickleLoaderImpl (const char* buffer, int len) :
    marks_(512),
    env_(MAKE_DICT()),
    input_(const_cast<char*>(buffer)),
    len_(len),
    where_(0),
    noteProtocol_(0)
  {
    registry_["collections\nOrderedDict\n"]  = ReduceOTabFactory;
    registry_["Numeric\narray_constructor\n"]= ReduceNumericArrayFactory;
    registry_["array\narray\n"]              = ReduceArrayFactory;
    registry_["__builtin__\ncomplex\n"]      = ReduceComplexFactory;

    registry_["numpy.core.multiarray\n_reconstruct\n"]= BuildNumPyReconstruct;
    //registry_["numpy\nndarray\n"] = BuildNumPyNDArray;
    registry_["numpy\ndtype\n"] =  BuildNumPyDtype;

  }

  // Reset the input so we can reuse this loader
  void reset (const char* buffer, int len)
  {
    noteProtocol_ = 0;
    input_ = const_cast<char*>(buffer);
    len_   = len;
    where_ = 0;
    values_.clear();
    marks_.clear();
    memos_.clear();
  }

  // Load and return the top value
  void loads (OBJ& return_value) { decode_(return_value); }

  // Registering your own Python classes so they load correctly.
  // 
  // Most built-ins (Numeric, Array, complex) use the REDUCE method:
  // this means the FactoryFunction input_tuple will have:
  //   input_name:  string name of class  (i.e., '__builtin__\ncomplex\n')
  //   input_tuple: tuple with the arguments (i.e., (1,2) )
  // See the array, complex, Numeric examples above.  THESE ARE ALREADY
  // BUILT-IN.
  // 
  // User-defined classes tend to use the BUILD method which
  // has a different tuple, it tends to look like:
  //   input_name : a "type_object" tuple: string name and args
  //   input_tuple: dictionary (the state)
  // See the Scott example above. IN THE SCOTT EXAMPLE, YOU MUST
  // "registerFactory" MANUALLY FOR THE EXAMPLE TO WORK.
  void registerFactory (const string& name, FactoryFunction& ff)
  { registry_[name] = ff; }

  
  // Expose the environment: It's just a tab right now, where
  // the following keys are understood:
  //   compatibility: turns OTab->Tab, Tup->Arr (for older systems)
  //   allowNumeric:  when seen, allows Numeric Arrays into POD Arrays
  OBJ& env () { return env_; }

 protected:

  // //// Data Members

  // The Value stack: nothing ever goes "straight into" a data
  // structure: it has to sit on the value stack then be reduced into
  // some data structure.
  Stack<OBJ> values_;

  // Every time a memo is made (and they are always made in order),
  // lookup its associated value.
  Array<OBJ> memos_;

  // Mark stack: every a mark is made, indicate where it is on the
  // value stack ... many things pop back to the last mark.
  Stack<int> marks_;

  // A registry, so when we REDUCE or BUILD, we can "factory" create
  // the right thing.  This is mostly for Numeric, complex, and array
  // but also for user-defined types (although the input is different).
  //AVLHashT<string, FactoryFunction, 8> registry_;
  HashTableT<string, FactoryFunction, 8> registry_;

  // The environment within the registry when it's called.  In order
  // words, relevant flags (Numeric array supported? Array supported?
  // etc.).  Functions are also allowed to change environment if they
  // need to.
  OBJ env_;

  // The input, stored as some chars.  TODO: We may optimize this for
  // file I/O or for buffers.
  char* input_;
  int len_;
  int where_;
 
  // Note the protocol being used ... Not really used right now
  int noteProtocol_;

  // ///// Methods

  // Keep pulling stuff off of input and final thing on top of stack
  // becomes result
  inline void decode_ (OBJ& result);

  // Get the next character from input: may throw exception if at end
  // of buffer.
  inline 
  int getChar_ () { return (where_>=len_) ? -1 : int_u1(input_[where_++]); }

  // Advance the input up "n" characters, but keep a pointer back
  // to the original place where we started: this allows us to get
  // the next few characters easily.
  inline char* advanceInput_ (int n)
  { 
    char* start = input_ + where_;
    where_ += n;
    if (where_>len_) throw out_of_range("...past input for PickleLoader");
    return start;
  }

  // From the current input, keep going until you get a \n: pass
  // the newline
  char* getUpToNewLine_ (int& length_of_input)
  {
    char* start = input_ + where_;
    char* end = input_ + len_;
    char* result = start;
    while (start != end && *start++ != '\n') ;
    length_of_input = start - result;
    where_ += length_of_input;

    if (where_>len_) throw out_of_range("...past input for PickleLoader");
    return result;
  }

  // Getting a string is tricky: it can contain embedded ' and " as 
  // well as \t\n sequences as well as \x00 sequences.  We have to
  // start with the starting punctutation (' or "), ignore all
  // escaped starting punctuation until the end. 
  // Input: something like 'abc\'123'\n
  // Output: input advanced past \n, start points to start quote
  //         len includes quotes and len (so actual string data is 3 less)
  char* getString_ (int& length_of_input)
  {
    char* input = input_ + where_;
    const int input_length = len_;
    length_of_input = -1;
    // Get start quote
    char start_quote = input[0];     
    if (start_quote!='\'' && start_quote!='\"') {
      throw runtime_error("Expecting a quote on the start of a P0 string");
    }
    // Iterate through escapes
    static char       code[] = "n\\rt'\"";
    for (int ii=1; ii<input_length; ) {
      char c = input[ii];
      if (c=='\\') { // non-printable,so was esc
	if (ii+1>=input_length) throw runtime_error("End of Buffer reached");
	char* where = strchr(code, input[ii+1]);
	if (where) {
	  ii+=2;
	  continue;
	} else if (input[ii+1]=='x') {
	  if (ii+3>=input_length) throw runtime_error("End of Buffer reached");
	  ii+=4;
	  continue;
	} else {
	  throw runtime_error("Malformed string for P0" +
		       string(input, input_length)+" ... Error happened at:");
	  // IntToString(ii));
	}
      } else if (c==start_quote) {
	// Error checking to made sure we are at end
	if (++ii>=input_length) throw runtime_error("End of Buffer reached");
	if (input[ii]!='\n') throw runtime_error("No newline at end o string");
	length_of_input = ii+1;
	where_ += length_of_input;
	break;
      } else { // printable, so take as is
	++ii;
      }
    }
    if (length_of_input==-1) {
      throw runtime_error("End of buffer reached without seeing end quote");
    }
    return input;
  }

  
  // Get a 4 byte in from input: a length so int_u4
  inline int_u4 get4ByteInt_ ();

  inline void memoInsert_ (int_u4 memo_number, OBJ& peeked_obj);

  // Routines to handle each input token
  inline void hMARK();
  inline void hFLOAT();
  inline void hBINFLOAT();
  inline void hINT();
  inline void hBININT();
  inline void hBININT1();
  inline void hLONG();
  inline void hBININT2();
  inline void hNONE();
  inline void hREDUCE();
  inline void hSTRING();
  inline void hBINSTRING();
  inline void hSHORT_BINSTRING();
  inline void hAPPEND();
  inline void hBUILD();
  inline void hGLOBAL();
  inline void hDICT();
  inline void hEMPTY_DICT();
  inline void hAPPENDS();
  inline void hGET();
  inline void hBINGET();
  inline void hLONG_BINGET();
  inline void hLIST();
  inline void hEMPTY_LIST();
  inline void hOBJECT();
  inline void hPUT();
  inline void hBINPUT();
  inline void hLONG_BINPUT();
  inline void hSETITEM();
  inline void hTUPLE();
  inline void hEMPTY_TUPLE();
  inline void hSETITEMS();
  inline void hPROTO();
  inline void hNEWOBJ();
  inline void hTUPLE1();
  inline void hTUPLE2();
  inline void hTUPLE3();
  inline void hNEWTRUE();
  inline void hNEWFALSE();
  inline void hLONG1();
  inline void hLONG4();

  inline void NOT_IMPLEMENTED (char c) { string ss; ss = c; throw runtime_error("Don't know how to handle "+ss); }

}; // PickleLoaderImpl


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::decode_ (OBJ& result)
{
  result = None;
  bool done = false;
  while (!done) {
    int in = this->getChar_();
    if (in==-1) break;  // End of input
    char token = in;
    //cerr << "TOKEN:" << token << endl;
    //cerr << "Values:"; values_.print(cerr) << endl;
    //cerr << "marks:";  marks_.print(cerr) << endl;
    switch (token) {
    case PY_MARK:             this->hMARK(); break;            // '('
    case PY_STOP:             done = true; break;        // '.'
    case PY_POP:           NOT_IMPLEMENTED('0'); break; // '0'
    case PY_POP_MARK:      NOT_IMPLEMENTED('1'); break; // '1'
    case PY_DUP:           NOT_IMPLEMENTED('2'); break; // '2'
    case PY_FLOAT:            this->hFLOAT(); break;          // 'F'
    case PY_BINFLOAT:         this->hBINFLOAT(); break;       // 'G'
    case PY_INT:              this->hINT(); break;            // 'I'
    case PY_BININT:           this->hBININT(); break;         // 'J' -2**31 to 2**31-1
    case PY_BININT1:          this->hBININT1(); break;        // 'K' 0..255
    case PY_LONG:             this->hLONG();    break;         // 'L'
    case PY_BININT2:          this->hBININT2(); break;        // 'M' 0..65535
    case PY_NONE:             this->hNONE();    break;        // 'N'
    case PY_PERSID:        NOT_IMPLEMENTED('P'); break; // 'P'
    case PY_BINPERSID:     NOT_IMPLEMENTED('Q'); break; // 'Q'
    case PY_REDUCE:           this->hREDUCE(); break;          // 'R'
    case PY_STRING:           this->hSTRING(); break;     // 'S'
    case PY_BINSTRING:        this->hBINSTRING(); break;       // 'T'
    case PY_SHORT_BINSTRING:  this->hSHORT_BINSTRING(); break; // 'U'  
    case PY_UNICODE:       NOT_IMPLEMENTED('V'); // 'V'
    case PY_BINUNICODE:    NOT_IMPLEMENTED('X'); // 'X'
    case PY_APPEND:           this->hAPPEND(); break;          // 'a'
    case PY_BUILD:            this->hBUILD(); break;           // 'b'
    case PY_GLOBAL:           this->hGLOBAL(); break;          // 'c'
    case PY_DICT:             this->hDICT();  break;           // 'd'
    case PY_EMPTY_DICT:       this->hEMPTY_DICT(); break;     // '}'
    case PY_APPENDS:          this->hAPPENDS(); break;        // 'e'
    case PY_GET:              this->hGET();     break;        // 'g'
    case PY_BINGET:           this->hBINGET(); break;         // 'h'
    case PY_INST:         NOT_IMPLEMENTED('i'); break;  // 'i'
    case PY_LONG_BINGET:      this->hLONG_BINGET(); break;    // 'j'
    case PY_LIST:             this->hLIST(); break;           // 'l'
    case PY_EMPTY_LIST:       this->hEMPTY_LIST(); break;     // ']'
    case PY_OBJ:              this->hOBJECT(); break;         // 'o'
    case PY_PUT:              this->hPUT();   break;         // 'p'
    case PY_BINPUT:           this->hBINPUT(); break;        // 'q'
    case PY_LONG_BINPUT:      this->hLONG_BINPUT(); break;   // 'r'
    case PY_SETITEM:          this->hSETITEM(); break;       // 's'
    case PY_TUPLE:            this->hTUPLE(); break;         // 't'
    case PY_EMPTY_TUPLE:      this->hEMPTY_TUPLE(); break;   // ')' 
    case PY_SETITEMS:         this->hSETITEMS(); break;      // 'u'
      /* Protocol 2. */
    case PY_PROTO:            this->hPROTO(); break; // '\x80' /* identify pickle protocol */
    case PY_NEWOBJ:           this->hNEWOBJ(); break; // '\x81' /* build object by applying cls.__new__ to argtuple */
    case PY_EXT1:    NOT_IMPLEMENTED('\x82'); break; // '\x82' /* push object from extension registry; 1-byte index */
    case PY_EXT2:    NOT_IMPLEMENTED('\x83'); break; // '\x83' /* ditto, but 2-byte index */
    case PY_EXT4:    NOT_IMPLEMENTED('\x84'); break; // '\x84' /* ditto, but 4-byte index */
    case PY_TUPLE1: this->hTUPLE1(); break; // '\x85' /* build 1-tuple from stack top */
    case PY_TUPLE2: this->hTUPLE2(); break ; // '\x86' /* build 2-tuple from two topmost stack items */
    case PY_TUPLE3: this->hTUPLE3(); break; // '\x87' /* build 3-tuple from three topmost stack items */
    case PY_NEWTRUE:  this->hNEWTRUE(); break; // '\x88' /* push True */
    case PY_NEWFALSE: this->hNEWFALSE(); break; // '\x89' /* push False */
    case PY_LONG1:    this->hLONG1(); break; // '\x8a' /* push long from < 256 bytes */
    case PY_LONG4:    this->hLONG4(); break; // '\x8b' /* push really big long */
    default: throw runtime_error("Unknown token");
    }
  }
  // The final result is whatever is on top of the stack!
  OBJ& top = values_.peek(0);
  result.swap(top);
}

// Ugh, put this back into Arr
template <class T, class OBJ>
void SwapIntoAppend (Array<T>& a, OBJ& v)
{
  int len = a.length();
  a.append(CHEAP_VALUE);
  v.swap(a(len));
}



template <class OBJ>
inline int_u4 PickleLoaderImpl<OBJ>::get4ByteInt_ ()
{ 
  int_u4 len4; 
  char* current = input_+where_;
  LOAD_FROM(int_u4, current, &len4); 
  where_ += 4;
  return len4;
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hSHORT_BINSTRING ()
{
  OBJ& s = values_.push(CHEAP_VALUE);
  int len = getChar_();                 // int_u1 of length of string
  char* start = advanceInput_(len);     // advance next few, but keep where was
  
  INSERT_STRING(s, start, len);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hSTRING ()
{
  OBJ& s = values_.push(CHEAP_VALUE);

  int full_len;
  char* start = getString_(full_len); // start is something like S    '12345'\n


  int len = full_len-3;   // len includes the two quotes AND \n
  char* data = new char[len];
  int final_len = CopyPrintableBufferToVector(start+1, len, data, len);
  s = string(data, final_len); 
  delete [] data;
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBINSTRING ()
{
  OBJ& s = values_.push(CHEAP_VALUE);
  int_u4 len = get4ByteInt_();
  char* start_char = advanceInput_(len);

  // TODO: A little sketchy, but it saves a copy
  INSERT_STRING(s, start_char, len);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hMARK ()
{
  // Simulate a "mark" on Python stack with a meta value
  marks_.push(values_.length());
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hGLOBAL ()
{
  char* start = input_+where_;
  char* end   = start+len_;
  // Pass by 2 \n
  while (start<end && *start++ != '\n') ;
  while (start<end && *start++ != '\n') ;
  int len_of_global_string = start - (input_+where_);
  where_ += len_of_global_string;
  values_.push(string(start-len_of_global_string, len_of_global_string));
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hREDUCE ()
{
  // There are two things on the stack: a "global function name" and a
  // tuple of args to apply to the constructor
  OBJ& name  = values_.peek(-1);
  OBJ& tuple = values_.peek(0);
  
  // Here's the thing: we lookup the "constructor" from the registry
  // and apply the arguments to that
  string global_name = name;
  if (registry_.contains(global_name)) {
    // There is an entry: use it to create the appropriate data
    // structure from the Factory Function.
    FactoryFunction& ff = registry_[global_name];
    OBJ result;
    ff(name, tuple, env_, result); // create result
    
    name.swap(result); // put result in place of name
  } else {
    // By default, return the tuple if there is no entity registering
    // so use the tuple (name, tuple) as the final result
    OBJ not_there = MAKE_TUP2(CHEAP_VALUE, CHEAP_VALUE);
    EXTRACT_TUP(u, not_there);
    u(0).swap(name);
    u(1).swap(tuple);
    name.swap(not_there);
  }

  // There are two things on the stack: we only want one
  values_.pop(1);  
}

// DICT, LIST and TUPLE all use the MARK
// EMPTYDICT, EMPTYLIST, EMPTY_TUPLE do not

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hDICT ()  
{
  // Find where top of the stack was at mark time
  int last_mark = marks_.pop();

  // How many items on stack?  The difference between the current
  // length, and where the last mark was
  int items_to_set = values_.length() - last_mark;

  // No actual dict value on stack at this point: it replaces the
  // first thing on the stack
  OBJ v = MAKE_DICT(); 
  EXTRACT_DICT(d, v);

  // For efficiency, swap the values in
  for (int ii=last_mark; ii<last_mark+items_to_set; ii++) {
    DICT_SWAP_INTO(d, values_[ii], values_[ii+1]);
  }

  // Once values in Tab, take em off stack ... all except last,
  // which becomes the Tab itself
  if (items_to_set==2) {
    values_.peek(0).swap(v);
  } else if (items_to_set-1>0) {
    values_.pop(items_to_set-1);
    values_.peek(0).swap(v);
  } else {
    values_.push(v); // If nothing on the stack, push an empty Tab
  }
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hLIST ()  
{ 
  // Find where top of the stack was at mark time
  int last_mark = marks_.pop();

  // How many items on stack?  The difference between the current
  // length, and where the last mark was
  int items_to_append = values_.length() - last_mark;

  // No actual tuple value on stack: it replaces the first thing on
  // the stack
  OBJ v = MAKE_LIST1(items_to_append); 
  EXTRACT_LIST(a, v);

  // For efficiency, swap the values in
  for (int ii=0; ii<items_to_append; ii++) {
    LIST_SWAP_INTO_APPEND(a, values_[last_mark+ii]);
  }

  // Once values in arr, take em off stack ... all except last,
  // which becomes the arr itself
  if (items_to_append==1) {
    values_.peek(0).swap(v);
  } else if (items_to_append-1>0) {
    values_.pop(items_to_append-1);
    values_.peek(0).swap(v);
  } else {
    values_.push(v); // If nothing on the stack, push an empty arr
  }
}

// EMPY doesn't look at the mark, just plops on the stack.
template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hEMPTY_DICT ()  { values_.push(MAKE_DICT()); }
template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hEMPTY_LIST ()  { values_.push(MAKE_LIST()); }
template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hEMPTY_TUPLE () { values_.push(MAKE_TUP0()); }


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hFLOAT () 
{
  int len;
  char* start = getUpToNewLine_(len);
  real_8 rl = atof(start); // TODO: probably need \0
  values_.push(rl);
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBINFLOAT () 
{
  char* start = advanceInput_(8);
  real_8 rl;
  LOAD_DBL(start, &rl);
  values_.push(rl);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::memoInsert_ (int_u4 memo_number, 
						OBJ& peeked_obj)
{
  while (int(memos_.length()) < int(memo_number)) {
    memos_.append(NONE_VALUE);
  }
  memos_.append(peeked_obj);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hPUT ()
{
  // Take top of stack and "memoize" it with this value!
  int len;
  char* start = getUpToNewLine_(len);
  int_u4 memo_number = StringToInt<int_u4>(start, len);
  
  memoInsert_(memo_number, values_.peek());
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBINPUT ()
{
  // Take top of stack and "memoize" it with this value!
  char memo_number_char = getChar_();
  int_u1 memo = int_u1(memo_number_char);
  int_u4 memo_number = memo;
  
  memoInsert_(memo_number, values_.peek());
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hLONG_BINPUT ()
{
  // Take top of stack and "memoize" it with this value!
  int_u4 memo_number = get4ByteInt_();

  memoInsert_(memo_number, values_.peek());
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hGET ()
{
    // Take top of stack and "memoize" it with this value!
  int len;
  char* start = getUpToNewLine_(len);
  int_u4 memo_number = StringToInt<int_u4>(start, len);

  values_.push(memos_(memo_number));
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBINGET ()
{
    // Take top of stack and "memoize" it with this value!
  char memo_number_char = getChar_();
  int_u1 memo = int_u1(memo_number_char);
  int_u4 memo_number = memo;
  
  values_.push(memos_(memo_number));
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hLONG_BINGET ()
{  
  // Take top of stack and "memoize" it with this value!
  int_u4 memo_number = get4ByteInt_();
  
  values_.push(memos_(memo_number));
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hAPPEND ()
{
  // Two items on stack: must be list and some value
  OBJ& oa = values_.peek(-1); 
  EXTRACT_LIST(a, oa);
  Val& v = values_.peek(0);

  // Swap value in
  LIST_SWAP_INTO_APPEND(a, v);

  // Leave just array on stack
  values_.pop(1);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hSETITEM ()
{
  // Three items on stack: must be Tab, key, value
  EXTRACT_DICT(t, values_.peek(-2));
  OBJ& key = values_.peek(-1);
  OBJ& value = values_.peek(0);
  
  // insert key/value
  DICT_SWAP_INTO(t, key, value);

  // Leave just Tab on stack
  values_.pop(2);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hTUPLE1 ()
{
  OBJ& e0 = values_.peek(0);
  OBJ t = MAKE_TUP1(CHEAP_VALUE);
  EXTRACT_TUP(u, t);
  TUP_SUB(u, 0).swap(e0);
  t.swap(e0); // no pop: single element swap with value
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hTUPLE2 ()
{
  OBJ& e0 = values_.peek(-1);
  OBJ& e1 = values_.peek(0);
  OBJ t = MAKE_TUP2(CHEAP_VALUE, CHEAP_VALUE);
  EXTRACT_TUP(u, t);
  TUP_SUB(u,0).swap(e0);
  TUP_SUB(u,1).swap(e1);
  t.swap(e0);

  // Leave just tuple on stack
  values_.pop(1); 
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hTUPLE3 ()
{
  OBJ& e0 = values_.peek(-2);
  OBJ& e1 = values_.peek(-1);
  OBJ& e2 = values_.peek( 0);
  OBJ t = MAKE_TUP3(CHEAP_VALUE, CHEAP_VALUE, CHEAP_VALUE);
  EXTRACT_TUP(u, t);
  TUP_SUB(u, 0).swap(e0);
  TUP_SUB(u, 1).swap(e1);
  TUP_SUB(u, 2).swap(e2);
  t.swap(e0);

  // Leave just tuple on stack
  values_.pop(2);
}

 
template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hAPPENDS ()
{
  // Find where top of the stack was at mark time
  int last_mark = marks_.pop();

  // How many items on values stack to append?  The difference between
  // the current length, and where the last mark was
  int items_to_append = values_.length() - last_mark;

  // For efficiency, swap the values in
  EXTRACT_LIST(a, values_[last_mark-1]);
  for (int ii=0; ii<items_to_append; ii++) {
    LIST_SWAP_INTO_APPEND(a, values_[last_mark+ii]);
  }

  // Once all the values are swapped into the array, pop 'em!
  values_.pop(items_to_append);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hSETITEMS ()
{
  // Find where top of the stack was at mark time
  int last_mark = marks_.pop();

  // How many items on key-values items on stack?  The difference
  // between the current length, and where the last mark was
  int items_to_insert = values_.length() - last_mark;

  // For efficiency, swap the values in
  EXTRACT_DICT(t, values_[last_mark-1]); 
  for (int ii=0; ii<items_to_insert; ii+=2) {
    DICT_SWAP_INTO(t, values_[last_mark+ii], values_[last_mark+ii+1] );
  }

  // Once all the values are swapped into the array, pop 'em! Leaves
  // just dict on stack.
  values_.pop(items_to_insert);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hPROTO ()
{
  char proto = getChar_();
  int_u1 protocol = proto;
  noteProtocol_ = protocol;
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hNEWOBJ ()
{
  // This is very similar to reduce: one difference is that "strictly
  // speaking", new are supposed to create the new object via:
  //      obj = C.__new__(C, *args)

  //hREDUCE();

  // Don't do anything: leave the string, args-tuple on the stack
  // for build to hit
  OBJ& name = values_.peek(-1);
  OBJ& args = values_.peek(0);
  OBJ type_object = MAKE_TUP2(CHEAP_VALUE, CHEAP_VALUE);
  EXTRACT_TUP(u, type_object);
  TUP_SUB(u,0).swap(name);
  TUP_SUB(u,1).swap(args);
  
  type_object.swap(name);
  
  values_.pop(1); // drop args
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hTUPLE ()
{
  // Find where top of the stack was at mark time
  int last_mark = marks_.pop();

  // How many items on key-values items on stack?  The difference
  // between the current length, and where the last mark was
  int items_to_append = values_.length() - last_mark;

  // No actual tuple value on stack: it replaces the first thing on
  // the stack
  OBJ v = MAKE_TUPN(items_to_append);
  EXTRACT_TUP(u, v);

  // For efficiency, swap the values in
  EXTRACT_TUP_AS_LIST(a, u);
  for (int ii=0; ii<items_to_append; ii++) {
    LIST_SWAP_INTO_APPEND(a,values_[last_mark+ii]);
  }

  // Once values in tuple, take em off stack ... all except last,
  // which becomes the tuple itself
  if (items_to_append==1) {
    values_.peek(0).swap(v);
  } else if (items_to_append-1>0) {
    values_.pop(items_to_append-1);
    values_.peek(0).swap(v);
  } else {
    values_.push(v); // If nothing on the stack, push an empty tuple
  }

}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hNEWTRUE ()  { values_.push(TRUE_VALUE); }
template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hNEWFALSE () { values_.push(FALSE_VALUE); }


template <class BI, class I>
size_t sizeofHelper (const BigInt<BI,I>& i) { return sizeof(I); }

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hLONG1 () 
{
  OBJ& v = values_.push(int_n(0));
  int_n* in = (int_n*)&v.u.q;

  char number_bytes_char = getChar_();
  int_u1 number_bytes_i1 = int_u1(number_bytes_char);
  int_u4 number_bytes = number_bytes_i1;

  
  char* start = advanceInput_(number_bytes);
  MakeBigIntFromBinary(start, number_bytes,
		       *in);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hLONG4 () 
{ 
  OBJ& v = values_.push(int_n(0));
  int_n* in = (int_n*)&v.u.q;

  char* start = advanceInput_(4);
  int_u4 number_bytes;
  LOAD_FROM(int_u4, start, &number_bytes);
  
  start = advanceInput_(number_bytes);
  MakeBigIntFromBinary(start, number_bytes,
		       *in);
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hNONE ()     { values_.push(NONE_VALUE); }


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBININT () 
{   // 'J' -2**31 to 2**31-1
  char* start = advanceInput_(4);
  int_4 ii;
  LOAD_FROM(int_4, start, &ii);
  values_.push(ii);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBININT1 () 
{   // 'K' 0..255
  char* start = advanceInput_(1);
  int_u1 ii;
  LOAD_FROM(int_u1, start, &ii);
  values_.push(ii);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBININT2 () 
{   // 'M' 0..65535
  char* start = advanceInput_(2);
  int_u2 ii; 
  LOAD_FROM(int_u2, start, &ii);
  values_.push(ii);
}

template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hINT ()
{
  // Backwards compatible mode: on 64-bit systems, seems to be how they
  // handle 64-bit ints as ASCI!!
  OBJ& s = values_.push(CHEAP_VALUE);

  int len;
  char* start = getUpToNewLine_(len);
  s = StringToInt<int_8>(start, len); // start is something like I    12345\n
  // TODO: optimize this?
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hLONG ()
{
  // Backwards compatible mode: on 64-bit systems, seems to be how they
  // handle 64-bit ints as ASCI!!
  OBJ& s = values_.push(None);

  int len;
  char* start = getUpToNewLine_(len);
  MAKE_BIGINT_FROM_STRING(s, start, len); 
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hOBJECT ()
{
  return hNEWOBJ();
  //  // Put "something" on the stack to repr an Object
  //values_.push(None);
}


template <class OBJ>
inline void PickleLoaderImpl<OBJ>::hBUILD ()
{
  // Build is interesting: it seems to expect the Type Object on the
  // stack as well as the dictionary
  OBJ& type_object = values_.peek(-1);
  if (type_object.tag != 'u') {
    throw runtime_error("Cannot depickle custom type");
  }
  //cerr << "typeobj:" << type_object << endl;
 
  string name = type_object(0);
  //Val&   tupl = type_object(1);
  //cerr << "name:" <<  name << endl;

  OBJ& dictionary  = values_.peek(0);
  //cerr << "dict:" << dictionary << endl;

  // Look up the name in the registry: use our information
  // to the BUILD the appropriate object if it is registered
  if (registry_.contains(name)) {
    FactoryFunction& ff = registry_[name];
    Val result;
    ff(type_object, dictionary, env_, result);
    result.swap(type_object);
  } else {
    // Create a tuple since nothing is registered
    // ( type_object, dictionary )
    OBJ not_there = MAKE_TUP2(CHEAP_VALUE, CHEAP_VALUE);
    EXTRACT_TUP(u, not_there);
    TUP_SUB(u, 0).swap(type_object);
    TUP_SUB(u, 1).swap(dictionary);
    not_there.swap(type_object);
  }
  values_.pop(1);
}



OC_END_NAMESPACE

#define GENERICPICKLELOADER_H_
#endif // GENERICPICKLELOADER_H_
