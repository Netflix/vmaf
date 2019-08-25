#ifndef PICKLELOADER_H_

// Complete reimplementation of the Protocol Loading 2.  The codebase
// has a smaller memory footprint, it's just about as fast (faster in
// some cases), it's is easier to read and maintain, and simpler.

#include "ocval.h"
#include "cpickle.h"
#include "ocavltreet.h"
#include "m2pythontools.h"
#include "ocnumerictools.h"

OC_BEGIN_NAMESPACE


// A Stack
template <class T>
class Stack {
 public:
  
  // Construct with initial capacity: will resize of course.
  Stack (size_t initial_len = 256) : impl_(initial_len) { }
    
  // Push something on top of the stack: return a reference to where
  // it is.
  T& push (const T& t)
  {
    size_t len = impl_.length();
    impl_.append(t);
    return impl_(len);
  }

  // Peek into the stack: 0 is the top, -1 is just below the top, etc.
  // Returns a reference to the element in question.
  T& peek (int_8 where=0) 
  {
    int_8 len = impl_.length();
    return impl_[len+where-1];
  }

  // Pop the top element of the stack.  Returns thing popped.
  T pop () { return impl_.removeLast(); }

  // Pop the top number of elements given.
  void pop (int_8 how_many)
  {
    int_8 len = impl_.length();
    impl_.removeRange(len-how_many, how_many);
  }
  
  // Clean up
  void clear () { impl_.clear(); }

  // The number of elements on the stack
  int_8 length() const { return impl_.length(); }

  // Lets you dig inside the implementation if you have to.
  T& operator[] (int_8 ii) { return impl_[ii]; }

  // Print hook
  ostream& print (ostream& os) { return os << impl_ << endl; }

 protected:
  Array<T> impl_; // A stack is implemented using an array
      
}; // Stack


// Translate all instance of NumPy "classes" (Tuples with name, data)
// to Array<POD>.  This is post-processing: we have to do things this
// because of the way BUILD works: it changes a value INPLACE, and we
// don't have typeless proxies.
inline void TranslateForNumPyClassesToArray (Val& inplace_change)
{
  // Only even care if it is a proxy and has __type__ field
  if (inplace_change.tag=='u' && inplace_change.isproxy) {
    Tup& t = inplace_change;
    if (t.length()>=2) {
      Val& first = t(0);
      if (first.tag=='a' && first=="numpy.core.multiarray\n_reconstruct\n") {
	Val& data = t(1);
	inplace_change = data;
	return;	
      }
    }
  } 

  // Any composite type
  if (inplace_change.tag=='o' || inplace_change.tag=='t' || 
      inplace_change.tag=='u' ||
      (inplace_change.tag=='n' && inplace_change.subtype=='Z')) {
    for (It ii(inplace_change); ii(); ) {
      Val& value = ii.value();
      TranslateForNumPyClassesToArray(value);  
    }
  }
}


// When registering things with the factory, they take in some tuple
// and return a Val: REDUCE tend to be more for built-in complicated
// types like Numeric, array and complex.  BUILD tends to more for
// user-defined types.
typedef void (*FactoryFunction)(const Val& name, 
				const Val& input_tuple, 
				Val& environment,  
				Val& output_result);


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
inline void BuildScottFactoryFunction (const Val& type_object, 
				       const Val& input,
				       Val& /* environment */,
				       Val& output_result)
{
  cout << "SCOTT:type_object:" << type_object << endl;
  cout << "SCOTT:input:" << input << endl;

  string name = type_object(0);
  Tup&   args = type_object(1);
  cout << "name:" << name << " args:" << args << endl;

  if (name!="__main__\nScott\n") throw runtime_error("Not right name");
  Val& result = input("data");
  output_result = result;
}


// Example: ReduceFactoryFunction for OrderedDict: take a list of
// 2-element Arrs
inline void ReduceOTabFactory (const Val& /* name */,
			       const Val& tuple, 
	 		       Val& env, 
			       Val& result)
{  
  // Constructor is a 1-tuple which should have a list of 2-element arrs
  if (tuple.length()!=1) {
    throw runtime_error("Malformed OrderedDict constructor");
  }
  Tup& u=tuple;
  Arr& a = u(0);
  size_t len = a.length();
  if (!env.contains("compatibility")) {
    // Direct conversion to OTab
    OTab& o = result = OTab();
    for (size_t ii=0; ii<len; ii++) {
      Arr& pair = a(ii);
      o.swapInto(pair(0), pair(1));
    }
  } else {
    // Convert to a TAB instead, as we need backwards compatibility
    Tab& t = result = Tab();
    for (size_t ii=0; ii<len; ii++) {
      Arr& pair = a(ii);
      t.swapInto(pair(0), pair(1));
    }
  }
}

// Example: ReduceFactoryFunction for complex: 2-tuple of real_8s
inline void ReduceComplexFactory (const Val& /* name */,
				  const Val& tuple, 
				  Val&,
				  Val& result)
{
  // we assume we got the name right
  real_8 rl = tuple(0);
  real_8 im = tuple(1);
  result = complex_16(rl, im);
}


// Helper function to convert the string into a data of Array: In case
// we do not support Numeric Arrays, they can be converted to Arrs.
template <class T>
inline void NumericArrayFactoryHelper_ (bool keep_numeric, 
					size_t elements, T* data, 
					Val& result)
{
  if (!keep_numeric) {
    Arr& a = result = new Arr(elements);
    a.fill(None);
    Val* val_data = a.data();
    for (size_t ii=0; ii<elements; ii++) {
      val_data[ii] = data[ii];
    }   
  } else {
    Array<T>& a = result = new Array<T>(elements);
    a.expandTo(elements);
    T* val_data = a.data();
    memcpy(val_data, data, elements*sizeof(T));
  }
}

// Example: ReduceFactoryFunction for Numeric
#define OC_NUM_FACT(T) { NumericArrayFactoryHelper_(conv,elements,(T*)data,result); break; }
inline void ReduceNumericArrayFactory (const Val& /* name */,
				       const Val& tuple, 
				       Val& env,
				       Val& result)
{
  // we assume we got the name right

  Tup& u = tuple;
  // First element of tuple is a tuple: 
  //           the number of elements in the Numeric Array
  // Second element of tuple is a string
  //            typecode of tuple (prolly a single char)
  // Third element is a string: 
  //            the binary data of the array (memcpyed in)
  // Fourth element is a bool
  //            save space (not really used)
  Tup& ele = u(0);
  size_t elements = (ele.length()==0) ? 1 : size_t(ele(0)); // This a tuple for some dumb reason
  string typecode = u(1);
  Val& bin_data = u(2);
  OCString* ocsp = (OCString*)&bin_data.u.a;
  const char* data = ocsp->data();
  bool conv = env.contains("supportsNumeric") && 
    bool(env("supportsNumeric"))==true;

  // Macro sets the result with the proper type of array
  char type_char = typecode[0];

  // If the number of element is NOT equal to what we think the size
  // is, this is probably a 32-bit vs. 64-bit discrepancy: on 32-bit
  // machines, Numeric 'l' is 4 bytes, 8 bytes otherwise.  
  size_t computed_elements = 
    (ocsp->length()/(ByteLength(NumericTagToOC(type_char))));
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

// Example: ReduceFactoryFunction for Array module
#define OC_ARRAY_FACT(T) { result = Array<T>(); Array<T>& a=result; \
if (data.tag=='a'){ a.expandTo(length/sizeof(T)); OCString*ocp=(OCString*)&data.u.a;memcpy(a.data(),ocp->data(), ocp->length());} \
else { a.expandTo(length); Arr& ss=data; for (size_t ii=0; ii<length; ii++) { a[ii] = ss[ii]; } } }

// Argh!
// 2.6 dumps different than 2.7 which dumps different than 2.4
// -2,7 expects a Python list of each of the elements, individually labelled
// -2.6 expects a string which you can bitblit
// -2.4 doesn't even work
inline void ReduceArrayFactory (const Val& /* name */,
				const Val& tuple, 
				Val& /*env*/,
				Val& result)
{
  // First element of tuple:
  //     string of typecode
  string typecode = tuple(0);
  char type_char = typecode[0];
  // Second element of tuple:
  //     list (2.7) or string (2.6)
  Val&   data =     tuple(1);
  size_t length = data.length();
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


//  numpy.array([15,16,17], numpy.uint32) pickles as
// '\x80\x02cnumpy.core.multiarray\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x03\x85cnumpy\ndtype\nq\x04U\x02u4K\x00K\x01\x87Rq\x05(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x0c\x0f\x00\x00\x00\x10\x00\x00\x00\x11\x00\x00\x00tb.'


#define NUMPYARRAYCREATE(T) { Val temp=new Array<T>(shape); Array<T>&a=temp; a.expandTo(shape); outdata=a.data(); memcpy(outdata, raw_data, raw_data_bytes); result.swap(temp); }
inline void dispatchCreateNumpyArray_ (size_t shape, const string& type_desc,
				       const char* raw_data, size_t raw_data_bytes,
				       const string& endian,
				       Val& result)
{
  // Parse type description field, usallu something like 'u4', 'i8',
  // where the first letter is the type, the next letter is the length of
  // the type.
  if (type_desc.length() <1 ) {
    throw runtime_error("Illegal type string on a NumPy array:"+ type_desc);
  }
  char type_tag = type_desc[0];
  int_u4 type_len = 1;
  if (type_desc.length()>=2) { 
    type_len = StringToInt<int_u4>(&type_desc[1], type_desc.length()-1);
  }
  
  // Create appropriate array
  bool is_cx = false;
  void* outdata=0;  // Filled in by MACRO with pointer to data
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


// helper for BUILD Numpy Array:
// Any array we are given we flatten out to single dimensional (for now)
inline size_t FlattenShape (const Val& tuple)
{
  // Handle shape
  Val& described_shape = tuple(1);
  size_t shape = 1;
  if (described_shape.tag == 'u') { // flatten!
    size_t len = described_shape.length();
    if (len!=1) {
      cerr << "Warning: saw " << len << "dimensional array:flattening" << endl;
    }
    for (size_t ii=0; ii<len; ii++) {
      size_t dim = described_shape[ii];

      shape *= dim;
    }
  } else {
    shape = described_shape;  // take the execption if I get something weird
  }
  // shape is valid single-dimensional quantity at this point
  return shape;
}

// helper for BUILD Numpy Array
inline void GetTypeAndEndian (const Val& tuple,
			      string& type_desc, string& endian)
{
  Tup& type = tuple(2);
  type_desc = string(type(0)(1)(0));

  // Handle endianness
  endian = string(tuple(2)(1)(1));
}

// Inplace change the Val (which should be some bogus empty Array)
// into the dream Array, based on the values in Tuple
inline void BUILDNumPyArray_ (Val& instance, 
                              const Val& tuple)
{
  // We know instance is an Array of some type:
  // replace with the information of the args

  size_t shape = FlattenShape(tuple);
  
  // Handle raw data
  Val& vdata = tuple(4);
  OCString* ocp = (OCString*) &vdata.u.s;
  const char* raw_data = ocp->data();
  const size_t raw_data_bytes = ocp->length();

  // Handle type and endian
  string type_desc = tuple(2)(1)(0);
  string endian    = tuple(2)(2)(1);

  // Create the Array result and plop into result
  dispatchCreateNumpyArray_(shape, type_desc, raw_data, raw_data_bytes, endian,
			    instance);


}

// REDUCE: type_object is the string name
//         tuple is (simple) Array init:    usually ("string name", (0), "b")
// -> returns a Proxy for a Tup which contains the string name in 0 and 
//    the Array itself in 1
inline void ReduceNumPyCoreMultiarray (const Val& type_object,
				       const Val& tuple,
				       Val& /* env */,
				       Val& result)
{
  // REDUCE uses string name to build type-object
  if (type_object.tag=='a') {
    size_t shape = FlattenShape(tuple);
    string type_desc = (tuple.length()>=3) ? tuple(2) : Val("b");
    const char* raw_data = 0;
    size_t raw_data_bytes = 0;
    string endian="|";

    Val instance;
    dispatchCreateNumpyArray_(shape, type_desc, raw_data, raw_data_bytes, endian,
			      instance);
    result = new Tup(type_object, None);
    result[1].swap(instance);
  } 

  // BUILD uses 
  else if (type_object.tag=='u') {
    Val& instance = type_object(1);
    const Val& data     = tuple;
    BUILDNumPyArray_(instance, data);
    result = type_object;
  }

}


// The DTYPE: can also be shared with GETS/PUTS, so better
// if it's represented as a Proxy table which can 
// be updated with a BUILD correctly.
inline void ReduceNumPyDtype (const Val& type_object,
			      const Val& tuple,
			      Val& /* env */,
			      Val& result)
{
  // REDUCE uses string name to build a "type" on the stack,
  // so we build a fairly typical representation of a type object:
  // a tuple with the string name of the object, and default params
  if (type_object.tag=='a') {
    Val res = new Tup(type_object, None, None);

    // has to be pointer so can be shared when
    // BUILD action happens
    res[1] = tuple;
    result.swap(res); // actually building type object
  }
  
  // BUILD uses type-object (a table) to actually build a value representing
  // the type, the "instance".  In this case, we preserve all info in 
  // another tuple: For
  // DTYPES this is important because the "defaults" (captured in the
  // type object) are important
  else if (type_object.tag=='u' && type_object.isproxy) {
    Tup& res = type_object;  // The proxy containing type info
    res[2] = tuple;
    result = type_object;
  }                              // by a tuple

  // Error
  else {
    throw runtime_error("Bad args to BUILD/REDUCE NumPyDtype");
  }
}



// The PickleLoader is an loader which will load any Protocol, just
// like cPickle.loads(): 0 and 2 have been tested, the same
// infrstaructure should support 1 and 3 later.
class PickleLoader {

 public:

  // Construct a Pickleloader with the given input.
  PickleLoader (const char* buffer, size_t len) :
    env_(Tab()),
    input_(const_cast<char*>(buffer)),
    len_(len),
    where_(0),
    noteProtocol_(0)
  {
    registry_["collections\nOrderedDict\n"]  = ReduceOTabFactory;
    registry_["Numeric\narray_constructor\n"]= ReduceNumericArrayFactory;
    registry_["array\narray\n"]              = ReduceArrayFactory;
    registry_["__builtin__\ncomplex\n"]      = ReduceComplexFactory;

    registry_["numpy.core.multiarray\n_reconstruct\n"]    = ReduceNumPyCoreMultiarray;
    registry_["numpy\ndtype\n"]              = ReduceNumPyDtype;
  }

  // Reset the input so we can reuse this loader
  void reset (const char* buffer, size_t len)
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
  void loads (Val& return_value) 
  { decode_(return_value); 
    TranslateForNumPyClassesToArray(return_value);
  }

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
  Val& env () { return env_; }

 protected:

  // //// Data Members

  // The Value stack: nothing ever goes "straight into" a data
  // structure: it has to sit on the value stack then be reduced into
  // some data structure.
  Stack<Val> values_;

  // Every time a memo is made, lookup its associated value.  The
  // AVLTreeT values do not move once they are in the tree.  These
  // are put in order: 1..n, with a "swapInto" 
  AVLTreeT<size_t, Val, 16> memos_;

  // Mark stack: every a mark is made, indicate where it is on the
  // value stack ... many things pop back to the last mark.
  Stack<size_t> marks_;

  // A registry, so when we REDUCE or BUILD we can "factory" create
  // the right thing.  This is mostly for Numeric, complex, and array
  // but also for user-defined types (although the input is different).
  AVLHashT<string, FactoryFunction, 8> registry_;


  // The environment within the registry when it's called.  In order
  // words, relevant flags (Numeric array supported? Array supported?
  // etc.).  Functions are also allowed to change environment if they
  // need to.
  Val env_;

  // The input, stored as some chars.  TODO: We may optimize this for
  // file I/O or for buffers.
  char* input_;
  size_t len_;
  size_t where_;
 
  // Note the protocol being used ... Not really used right now
  int noteProtocol_;

  // ///// Methods

  // Keep pulling stuff off of input and final thing on top of stack
  // becomes result
  inline void decode_ (Val& result);

  // Get the next character from input: may throw exception if at end
  // of buffer.
  inline 
  int getChar_ () { return (where_>=len_) ? -1 : int_u1(input_[where_++]); }

  // Advance the input up "n" characters, but keep a pointer back
  // to the original place where we started: this allows us to get
  // the next few characters easily.
  inline char* advanceInput_ (size_t n)
  { 
    char* start = input_ + where_;
    where_ += n;
    if (where_>len_) throw out_of_range("...past input for PickleLoader");
    return start;
  }

  // From the current input, keep going until you get a \n: pass
  // the newline
  char* getUpToNewLine_ (size_t& length_of_input)
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
  char* getString_ (size_t& length_of_input)
  {
    char* input = input_ + where_;
    const size_t input_length = len_;
    length_of_input = size_t(-1);
    // Get start quote
    char start_quote = input[0];     
    if (start_quote!='\'' && start_quote!='\"') {
      throw runtime_error("Expecting a quote on the start of a P0 string");
    }
    // Iterate through escapes
    static char       code[] = "n\\rt'\"";
    for (size_t ii=1; ii<input_length; ) {
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
    if (length_of_input==size_t(-1)) {
      throw runtime_error("End of buffer reached without seeing end quote");
    }
    return input;
  }

  
  // Get a 4 byte in from input: a length so int_u4
  inline int_u4 get4ByteInt_ ();

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

  inline void pushMemo_(size_t memo_number);
}; // PickleLoader



inline void PickleLoader::decode_ (Val& result)
{
  result = None;
  bool done = false;
  while (!done) {
    int in = getChar_();
    if (in==-1) break;  // End of input
    char token = in;
    //cerr << "TOKEN:" << token << endl;
    //cerr << "Values:"; values_.print(cerr) << endl;
    //cerr << "marks:";  marks_.print(cerr) << endl;
    switch (token) {
    case PY_MARK:             hMARK(); break;            // '('
    case PY_STOP:             done = true; break;        // '.'
    case PY_POP:           NOT_IMPLEMENTED('0'); break; // '0'
    case PY_POP_MARK:      NOT_IMPLEMENTED('1'); break; // '1'
    case PY_DUP:           NOT_IMPLEMENTED('2'); break; // '2'
    case PY_FLOAT:            hFLOAT(); break;          // 'F'
    case PY_BINFLOAT:         hBINFLOAT(); break;       // 'G'
    case PY_INT:              hINT(); break;            // 'I'
    case PY_BININT:           hBININT(); break;         // 'J' -2**31 to 2**31-1
    case PY_BININT1:          hBININT1(); break;        // 'K' 0..255
    case PY_LONG:             hLONG();    break;         // 'L'
    case PY_BININT2:          hBININT2(); break;        // 'M' 0..65535
    case PY_NONE:             hNONE();    break;        // 'N'
    case PY_PERSID:        NOT_IMPLEMENTED('P'); break; // 'P'
    case PY_BINPERSID:     NOT_IMPLEMENTED('Q'); break; // 'Q'
    case PY_REDUCE:           hREDUCE(); break;          // 'R'
    case PY_STRING:           hSTRING(); break;     // 'S'
    case PY_BINSTRING:        hBINSTRING(); break;       // 'T'
    case PY_SHORT_BINSTRING:  hSHORT_BINSTRING(); break; // 'U'  
    case PY_UNICODE:       NOT_IMPLEMENTED('V'); // 'V'
    case PY_BINUNICODE:    NOT_IMPLEMENTED('X'); // 'X'
    case PY_APPEND:           hAPPEND(); break;          // 'a'
    case PY_BUILD:            hBUILD(); break;           // 'b'
    case PY_GLOBAL:           hGLOBAL(); break;          // 'c'
    case PY_DICT:             hDICT();  break;           // 'd'
    case PY_EMPTY_DICT:       hEMPTY_DICT(); break;     // '}'
    case PY_APPENDS:          hAPPENDS(); break;        // 'e'
    case PY_GET:              hGET();     break;        // 'g'
    case PY_BINGET:           hBINGET(); break;         // 'h'
    case PY_INST:         NOT_IMPLEMENTED('i'); break;  // 'i'
    case PY_LONG_BINGET:      hLONG_BINGET(); break;    // 'j'
    case PY_LIST:             hLIST(); break;           // 'l'
    case PY_EMPTY_LIST:       hEMPTY_LIST(); break;     // ']'
    case PY_OBJ:              hOBJECT(); break;         // 'o'
    case PY_PUT:              hPUT();   break;         // 'p'
    case PY_BINPUT:           hBINPUT(); break;        // 'q'
    case PY_LONG_BINPUT:      hLONG_BINPUT(); break;   // 'r'
    case PY_SETITEM:          hSETITEM(); break;       // 's'
    case PY_TUPLE:            hTUPLE(); break;         // 't'
    case PY_EMPTY_TUPLE:      hEMPTY_TUPLE(); break;   // ')' 
    case PY_SETITEMS:         hSETITEMS(); break;      // 'u'
      /* Protocol 2. */
    case PY_PROTO:            hPROTO(); break; // '\x80' /* identify pickle protocol */
    case PY_NEWOBJ:           hNEWOBJ(); break; // '\x81' /* build object by applying cls.__new__ to argtuple */
    case PY_EXT1:    NOT_IMPLEMENTED('\x82'); break; // '\x82' /* push object from extension registry; 1-byte index */
    case PY_EXT2:    NOT_IMPLEMENTED('\x83'); break; // '\x83' /* ditto, but 2-byte index */
    case PY_EXT4:    NOT_IMPLEMENTED('\x84'); break; // '\x84' /* ditto, but 4-byte index */
    case PY_TUPLE1: hTUPLE1(); break; // '\x85' /* build 1-tuple from stack top */
    case PY_TUPLE2: hTUPLE2(); break ; // '\x86' /* build 2-tuple from two topmost stack items */
    case PY_TUPLE3: hTUPLE3(); break; // '\x87' /* build 3-tuple from three topmost stack items */
    case PY_NEWTRUE:  hNEWTRUE(); break; // '\x88' /* push True */
    case PY_NEWFALSE: hNEWFALSE(); break; // '\x89' /* push False */
    case PY_LONG1:    hLONG1(); break; // '\x8a' /* push long from < 256 bytes */
    case PY_LONG4:    hLONG4(); break; // '\x8b' /* push really big long */
    default: throw runtime_error("Unknown token");
    }
  }
  // The final result is whatever is on top of the stack!
  Val& top = values_.peek(0);
  result.swap(top);
}

// Ugh, put this back into Arr
template <class T>
void SwapIntoAppend (Array<T>& a, Val& v)
{
  size_t len = a.length();
  a.append(None);
  v.swap(a(len));
}

inline int_u4 PickleLoader::get4ByteInt_ ()
{ 
  int_u4 len4; 
  char* current = input_+where_;
  LOAD_FROM(int_u4, current, &len4); 
  //LOAD_FROM_4(current, &len4);
  where_ += 4;
  return len4;
}

inline void PickleLoader::hSHORT_BINSTRING ()
{
  Val& s = values_.push(None);
  int len = getChar_();                 // int_u1 of length of string
  char* start = advanceInput_(len);     // advance next few, but keep where was
  
  // TODO: A little sketchy, but it saves a copy
  new (&s.u.a) OCString(start, len);
  s.tag = 'a';
  //v = string(start, len);
}

inline void PickleLoader::hSTRING ()
{
  Val& s = values_.push(None);

  size_t full_len;
  char* start = getString_(full_len); // start is something like S    '12345'\n


  size_t len = full_len-3;   // len includes the two quotes AND \n
  char* data = new char[len];
  size_t final_len = CopyPrintableBufferToVector(start+1, len, data, len);
  s = string(data, final_len); 
  delete [] data;
}

inline void PickleLoader::hBINSTRING ()
{
  Val& s = values_.push(None);
  int_u4 len = get4ByteInt_();
  char* start_char = advanceInput_(len);

  // TODO: A little sketchy, but it saves a copy
  new (&s.u.a) OCString(start_char, len);
  s.tag = 'a';
  // s = string(start_char, len);
}

inline void PickleLoader::hMARK ()
{
  // Simulate a "mark" on Python stack with a meta value
  marks_.push(values_.length());
}

inline void PickleLoader::hGLOBAL ()
{
  char* start = input_+where_;
  char* end   = start+len_;
  // Pass by 2 \n
  while (start<end && *start++ != '\n') ;
  while (start<end && *start++ != '\n') ;
  size_t len_of_global_string = start - (input_+where_);
  where_ += len_of_global_string;
  values_.push(string(start-len_of_global_string, len_of_global_string));
}


inline void PickleLoader::hREDUCE ()
{
  // There are two things on the stack: a "global function name" and a
  // tuple of args to apply to the constructor
  Val& name  = values_.peek(-1);
  Val& tuple = values_.peek(0);
  
  // Here's the thing: we lookup the "constructor" from the registry
  // and apply the arguments to that
  string global_name = name;
  if (registry_.contains(global_name)) {
    // There is an entry: use it to create the appropriate data
    // structure from the Factory Function.
    FactoryFunction& ff = registry_[global_name];
    Val result;
    ff(name, tuple, env_, result); // create result
    
    name.swap(result); // put result in place of name
  } else {
    // By default, return the tuple if there is no entity registering
    // so use the tuple (name, tuple) as the final result
    Val not_there = new Tup(None, None);
    Tup& u = not_there;
    u(0).swap(name);
    u(1).swap(tuple);
    name.swap(not_there);
  }

  // There are two things on the stack: we only want one
  values_.pop(1);  
}

// DICT, LIST and TUPLE all use the MARK
// EMPTYDICT, EMPTYLIST, EMPTY_TUPLE do not

inline void PickleLoader::hDICT ()  
{
  // Find where top of the stack was at mark time
  size_t last_mark = marks_.pop();

  // How many items on stack?  The difference between the current
  // length, and where the last mark was
  size_t items_to_set = values_.length() - last_mark;

  // No actual dict value on stack at this point: it replaces the
  // first thing on the stack
  Val v = new Tab();
  Tab& d = v;

  // For efficiency, swap the values in
  for (size_t ii=last_mark; ii<last_mark+items_to_set; ii++) {
    d.swapInto(values_[ii], values_[ii+1]);
  }

  // Once values in Tab, take em off stack ... all except last,
  // which becomes the Tab itself
  if (items_to_set==2) {
    values_.peek(0).swap(v);
  } else if (items_to_set>1) {
    values_.pop(items_to_set-1);
    values_.peek(0).swap(v);
  } else {
    values_.push(v); // If nothing on the stack, push an empty Tab
  }
}


inline void PickleLoader::hLIST ()  
{ 
  // Find where top of the stack was at mark time
  size_t last_mark = marks_.pop();

  // How many items on stack?  The difference between the current
  // length, and where the last mark was
  size_t items_to_append = values_.length() - last_mark;

  // No actual tuple value on stack: it replaces the first thing on
  // the stack
  Val v = new Arr();
  Arr& a = v;

  // For efficiency, swap the values in
  a.resize(items_to_append);
  for (size_t ii=0; ii<items_to_append; ii++) {
    SwapIntoAppend(a,values_[last_mark+ii]);
  }

  // Once values in arr, take em off stack ... all except last,
  // which becomes the arr itself
  if (items_to_append==1) {
    values_.peek(0).swap(v);
  } else if (items_to_append>1) {
    values_.pop(items_to_append-1);
    values_.peek(0).swap(v);
  } else {
    values_.push(v); // If nothing on the stack, push an empty arr
  }
}

// EMPY doesn't look at the mark, just plops on the stack.
inline void PickleLoader::hEMPTY_DICT ()  { values_.push(new Tab()); }
inline void PickleLoader::hEMPTY_LIST ()  { values_.push(new Arr()); }
inline void PickleLoader::hEMPTY_TUPLE () { values_.push(new Tup()); }


inline void PickleLoader::hFLOAT () 
{
  size_t len;
  char* start = getUpToNewLine_(len);
  real_8 rl = atof(start); // TODO: probably need \0
  values_.push(rl);
}


inline void PickleLoader::hBINFLOAT () 
{
  char* start = advanceInput_(8);
  real_8 rl;
  LOAD_DBL(start, &rl);
  values_.push(rl);
}

inline void PickleLoader::hPUT ()
{
  // Take top of stack and "memoize" it with this value!
  size_t len;
  char* start = getUpToNewLine_(len);
  size_t memo_number = StringToInt<size_t>(start, len);
  
  memos_[memo_number] = values_.peek();
}

inline void PickleLoader::hBINPUT ()
{
  // Take top of stack and "memoize" it with this value!
  char memo_number_char = getChar_();
  int_u1 memo = int_u1(memo_number_char);
  size_t memo_number = memo;
  
  memos_[memo_number] = values_.peek();
}

inline void PickleLoader::hLONG_BINPUT ()
{
  // Take top of stack and "memoize" it with this value!
  int_u4 memo_number = get4ByteInt_();
  
  memos_[memo_number] = values_.peek();
}

inline void PickleLoader::pushMemo_ (size_t memo_number)
{
  Val& v = memos_(memo_number);
  values_.push(v);
}


inline void PickleLoader::hGET ()
{
    // Take top of stack and "memoize" it with this value!
  size_t len;
  char* start = getUpToNewLine_(len);
  size_t memo_number = StringToInt<size_t>(start, len);
  
  pushMemo_(memo_number);
}

inline void PickleLoader::hBINGET ()
{
    // Take top of stack and "memoize" it with this value!
  char memo_number_char = getChar_();
  int_u1 memo = int_u1(memo_number_char);
  size_t memo_number = memo;
  
  pushMemo_(memo_number);
}

inline void PickleLoader::hLONG_BINGET ()
{  
  // Take top of stack and "memoize" it with this value!
  int_u4 memo_number = get4ByteInt_();
  
  pushMemo_(memo_number);
}


inline void PickleLoader::hAPPEND ()
{
  // Two items on stack: must be list and some value
  Arr& a = values_.peek(-1);
  Val& v = values_.peek(0);

  // Swap value in
  SwapIntoAppend(a,v);

  // Leave just array on stack
  values_.pop(1);
}

inline void PickleLoader::hSETITEM ()
{
  // Three items on stack: must be Tab, key, value
  Tab& t = values_.peek(-2);
  Val& key = values_.peek(-1);
  Val& value = values_.peek(0);
  
  // insert key/value
  t.swapInto(key, value);

  // Leave just Tab on stack
  values_.pop(2);
}

inline void PickleLoader::hTUPLE1 ()
{
  Val& e0 = values_.peek(0);
  Val t = new Tup(None);
  t[0].swap(e0);
  t.swap(e0); // no pop: single element swap with value
}

inline void PickleLoader::hTUPLE2 ()
{
  Val& e0 = values_.peek(-1);
  Val& e1 = values_.peek(0);
  Val t = new Tup(None, None);
  t[0].swap(e0);
  t[1].swap(e1);
  t.swap(e0);

  // Leave just tuple on stack
  values_.pop(1); 
}

inline void PickleLoader::hTUPLE3 ()
{
  Val& e0 = values_.peek(-2);
  Val& e1 = values_.peek(-1);
  Val& e2 = values_.peek( 0);
  Val t = new Tup(None, None, None);
  t[0].swap(e0);
  t[1].swap(e1);
  t[2].swap(e2);
  t.swap(e0);

  // Leave just tuple on stack
  values_.pop(2);
}

 
inline void PickleLoader::hAPPENDS ()
{
  // Find where top of the stack was at mark time
  size_t last_mark = marks_.pop();

  // How many items on values stack to append?  The difference between
  // the current length, and where the last mark was
  size_t items_to_append = values_.length() - last_mark;

  // For efficiency, swap the values in
  Arr& a = values_[last_mark-1];
  for (size_t ii=0; ii<items_to_append; ii++) {
    SwapIntoAppend(a, values_[last_mark+ii]);
  }

  // Once all the values are swapped into the array, pop 'em!
  values_.pop(items_to_append);
}

inline void PickleLoader::hSETITEMS ()
{
  // Find where top of the stack was at mark time
  size_t last_mark = marks_.pop();

  // How many items on key-values items on stack?  The difference
  // between the current length, and where the last mark was
  size_t items_to_insert = values_.length() - last_mark;

  // For efficiency, swap the values in
  Tab& t = values_[last_mark-1];
  for (size_t ii=0; ii<items_to_insert; ii+=2) {
    t.swapInto(values_[last_mark+ii], values_[last_mark+ii+1] );
  }

  // Once all the values are swapped into the array, pop 'em! Leaves
  // just dict on stack.
  values_.pop(items_to_insert);
}

inline void PickleLoader::hPROTO ()
{
  char proto = getChar_();
  int_u1 protocol = proto;
  noteProtocol_ = protocol;
}


inline void PickleLoader::hNEWOBJ ()
{
  // This is very similar to reduce: one difference is that "strictly
  // speaking", new are supposed to create the new object via:
  //      obj = C.__new__(C, *args)

  //hREDUCE();

  // Don't do anything: leave the string, args-tuple on the stack
  // for build to hit
  Val& name = values_.peek(-1);
  Val& args = values_.peek(0);
  Val type_object = new Tup(None, None);
  Tup& u = type_object;
  u(0).swap(name);
  u(1).swap(args);
  
  type_object.swap(name);
  
  values_.pop(1); // drop args
}


inline void PickleLoader::hTUPLE ()
{
  // Find where top of the stack was at mark time
  size_t last_mark = marks_.pop();

  // How many items on key-values items on stack?  The difference
  // between the current length, and where the last mark was
  size_t items_to_append = values_.length() - last_mark;

  // No actual tuple value on stack: it replaces the first thing on
  // the stack
  Val v = new Tup();
  Tup& u = v;

  // For efficiency, swap the values in
  Array<Val>& a = u.impl();
  a.resize(items_to_append);
  for (size_t ii=0; ii<items_to_append; ii++) {
    SwapIntoAppend(a,values_[last_mark+ii]);
  }

  // Once values in tuple, take em off stack ... all except last,
  // which becomes the tuple itself
  if (items_to_append==1) {
    values_.peek(0).swap(v);
  } else if (items_to_append>1) {
    values_.pop(items_to_append-1);
    values_.peek(0).swap(v);
  } else {
    values_.push(v); // If nothing on the stack, push an empty tuple
  } 
}


inline void PickleLoader::hNEWTRUE ()  { values_.push(true); }
inline void PickleLoader::hNEWFALSE () { values_.push(false); }


template <class BI, class I>
size_t sizeofHelper (const BigInt<BI,I>& i) { return sizeof(I); }

inline void PickleLoader::hLONG1 () 
{
  Val& v = values_.push(int_n(0));
  int_n* in = (int_n*)&v.u.q;

  char number_bytes_char = getChar_();
  int_u1 number_bytes_i1 = int_u1(number_bytes_char);
  size_t number_bytes = number_bytes_i1;

  char* start = advanceInput_(number_bytes);
  MakeBigIntFromBinary(start, number_bytes,
		       *in);
}

inline void PickleLoader::hLONG4 () 
{ 
  Val& v = values_.push(int_n(0));
  int_n* in = (int_n*)&v.u.q;

  char* start = advanceInput_(4);
  int_u4 number_bytes;
  LOAD_FROM(int_u4, start, &number_bytes);
  
  start = advanceInput_(number_bytes);
  MakeBigIntFromBinary(start, number_bytes,
		       *in);
}


inline void PickleLoader::hNONE ()     { values_.push(None); }


inline void PickleLoader::hBININT () 
{   // 'J' -2**31 to 2**31-1
  char* start = advanceInput_(4);
  int_4 ii;
  LOAD_FROM(int_4, start, &ii);
  values_.push(ii);
}

inline void PickleLoader::hBININT1 () 
{   // 'K' 0..255
  char* start = advanceInput_(1);
  int_u1 ii;
  LOAD_FROM(int_u1, start, &ii);
  values_.push(ii);
}

inline void PickleLoader::hBININT2 () 
{   // 'M' 0..65535
  char* start = advanceInput_(2);
  int_u2 ii; 
  LOAD_FROM(int_u2, start, &ii);
  values_.push(ii);
}

inline void PickleLoader::hINT ()
{
  // Backwards compatible mode: on 64-bit systems, seems to be how they
  // handle 64-bit ints as ASCI!!
  Val& s = values_.push(None);

  size_t len;
  char* start = getUpToNewLine_(len);
  s = StringToInt<int_8>(start, len); // start is something like I    12345\n
  // TODO: optimize this?
}


inline void PickleLoader::hLONG ()
{
  // Backwards compatible mode: on 64-bit systems, seems to be how they
  // handle 64-bit ints as ASCI!!
  Val& s = values_.push(None);

  size_t len;
  char* start = getUpToNewLine_(len);
  s = StringToBigInt(start, len);
}


inline void PickleLoader::hOBJECT ()
{
  return hNEWOBJ();
  //  // Put "something" on the stack to repr an Object
  //values_.push(None);
}


inline void PickleLoader::hBUILD ()
{
  // Build is interesting: it seems to expect the Type Object on the
  // stack as well as the dictionary
  Val& type_object = values_.peek(-1);

  // Otherwise check
  if (type_object.tag != 'u') {
    throw runtime_error("Cannot depickle custom type");
  }
  string name = type_object(0);
  //Val&   tupl = type_object(1);

  Val& dictionary  = values_.peek(0);

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
    Val not_there = new Tup(None, None);
    Tup& u = not_there;
    u(0).swap(type_object);
    u(1).swap(dictionary);
    not_there.swap(type_object);
  }
  values_.pop(1);
}






OC_END_NAMESPACE

#define PICKLELOADER_H_
#endif // PICKLELOADER_H_
