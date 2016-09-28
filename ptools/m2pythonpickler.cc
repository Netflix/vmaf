
#include "m2pythonpickler.h"

// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_BEGIN_NAMESPACE
#endif

// Directly from Python Module.  According to the Python docs: once an
// opcode is supported, its meaning will not be changed.  Even though
// serialization may change in Python, it will always be backwards
// compatible so this doesn't have to follow that.
#include "cpickle.h"

#if !defined(OC_SERIALIZE_COMPAT)
#  define OC_SERIALIZE_COMPAT true
#endif
// ///// PythonPicklerA

template <class OBJ>
PythonDepicklerA<OBJ>::PythonDepicklerA (bool with_numeric_package,
					 bool use_proxies) :
  withNumericPackage_(with_numeric_package),
  wantToProxyize_(use_proxies),
  compat_(OC_SERIALIZE_COMPAT)
{ }

template <class OBJ> PythonDepicklerA<OBJ>::~PythonDepicklerA () { }

template <class OBJ> OBJ PythonDepicklerA<OBJ>::load ()
{
  int cc;
  while ((cc=getChar_()) != EOF) {
    
    if (cc==PY_STOP) break;  // STOP Marker
    
    switch (cc) {
    case PY_NONE:      ploadNONE_();   break;
    case PY_GLOBAL:    ploadGLOBAL_(); break;
    case PY_INT:       ploadINT_();    break;
    case PY_LONG:      ploadLONG_();   break;
    case PY_FLOAT:     ploadFLOAT_();  break;
    case PY_STRING:    ploadSTRING_(); break;
    case PY_MARK:      ploadMARK_();   break;
    case PY_DICT:      pDICT_();       break;
    case PY_PUT:       pPUT_();        break;
    case PY_GET:       pGET_();        break;
    case PY_SETITEM:   pSETITEM_();    break;
    case PY_APPEND:    pAPPEND_();     break;
    case PY_EMPTY_DICT: pEMPTY_DICT_();break;
    case PY_LIST:      ploadLIST_();   break;
    case PY_TUPLE:     ploadTUPLE_();  break;
    case PY_REDUCE:    ploadREDUCE_(); break;
    default: 
      MakeException("Unknown Type"+cc); 
      break;
    }
  }
  return pop_();
}

template <class OBJ> void PythonDepicklerA<OBJ>::ploadNONE_ ()
{ push_(MakeNone()); }

template <class OBJ> void PythonDepicklerA<OBJ>::ploadGLOBAL_ ()
{
  int len;
  string ascii_class = getUntilNewLine_(len);
  string ascii_constructor = getUntilNewLine_(len);
  char s[2]; s[0] = '\n'; s[1] = '\0';
  string ss = ascii_class+string(s)+ascii_constructor;
  push_(MakeString(ss.c_str()));
}

template <class OBJ> void PythonDepicklerA<OBJ>::ploadINT_ ()
{
  int len;
  char* ascii_int = getUntilNewLine_(len);
  // Python sends bools as I00 and I01
  if (ascii_int[0]!=0 && ascii_int[0]=='0' && ascii_int[1]!=0 && ascii_int[2]==0 && (ascii_int[1]=='0' || ascii_int[1]=='1')) {
    int bb = atoi(ascii_int);
    push_(MakeBool(bb));
  } else {
    push_(MakeInt4(ascii_int));
  }
}

template <class OBJ> void PythonDepicklerA<OBJ>::ploadLONG_ ()
{
  int len;
  char* long_int = getUntilNewLine_(len);
  
  // There is a final 'L' which M2k doesn't like, turn it into a '\0'
  for (char* find_last_L = long_int; ;find_last_L++) {
    if (*find_last_L=='\0') {
      MakeException("Malformed Python Long: no ending L");
    }
    if (*find_last_L=='L') {
      *find_last_L = '\0';  // end early and
      break;
    }
  }
  
  // Now, search for a '-' so we know if it an int_u8 or int_8
  while (isspace(*long_int)) long_int++;
  if (*long_int=='-') {
    push_(MakeInt8(long_int));
  } else {
    push_(MakeIntu8(long_int));
  }
}

template <class OBJ> void PythonDepicklerA<OBJ>::ploadFLOAT_ ()
{
  int len;
  char* ascii_float = getUntilNewLine_(len);
  push_(MakeDouble(ascii_float));
}


template <class OBJ> void PythonDepicklerA<OBJ>::ploadSTRING_ () 
{
  int ascii_string_len;
  char* ascii_string = getUntilNewLine_(ascii_string_len);
  char start_quote = ascii_string[0];

  // Expecting  ' ... some data ... '\0, with two quotes and a\0
  // (where the \0 used to be the newline)
  if ( (ascii_string[ascii_string_len] == '\0') && 
       (ascii_string[ascii_string_len-1] ==start_quote) ) {
    // Everything okay, this is a valid string: Note that the
    // under P0, the only time a \n will appear is at the end:
    // inside of strings, it CONVERTS \n to two character \ n sequences.
    
    // Copy from buffer to itself, overwriting what was in there:
    // The string length is actually -2: 1 for  start quote, 1 for end quote
    char* beyond_start_quote = ascii_string+1;
    int binary_bytes = CopyPrintableBufferToVector(beyond_start_quote,
						   ascii_string_len-2, 
						   beyond_start_quote, 
						   ascii_string_len-2);
    push_(MakeString(beyond_start_quote, binary_bytes)); 
    return;
  }
  
  // Oops, malformed input
  MakeException("String on input is malformed");
}


template <class OBJ> void PythonDepicklerA<OBJ>::ploadMARK_ ()
{ // Simulate a "mark" on the Python stack with a meta value
  mark_.append(pmstack_.length());
}

template <class OBJ> void PythonDepicklerA<OBJ>::pPUT_ ()
{
  // We want to associate this number with what's on top of the stack.
  int len;
  char* ascii_num = getUntilNewLine_(len); 
  int_4 memo_num = atoi(ascii_num); 

  OBJ& top = pmstack_.memoPut(memo_num);

  // This will turn (if it makes sense) a by-value object into a Proxy.
  if (wantToProxyize_) { 
    // This breaks compatibility, as it forces anything that is "put"
    // to become a proxy.  Do we want plain by-value copies?  Or do we
    // want true proxies.  At least it's an option.
    if (IsList(top) || IsTable(top) || IsVector(top) || 
	IsOrderedDict(top) || IsTuple(top)) {
      Proxyize(top);
    } 
  }
}

template <class OBJ> void PythonDepicklerA<OBJ>::pGET_ ()
{
  int len;
  char* ascii_num = getUntilNewLine_(len);
  int_4 addr = atoi(ascii_num);
  pmstack_.memoGet(addr);
}

template <class OBJ> void PythonDepicklerA<OBJ>::ploadLIST_ () 
{ mark_.removeLast(); push_(MakeList()); }

template <class OBJ> void PythonDepicklerA<OBJ>::ploadTUPLE_ ()
{
  // Create a tuple from all arguments back to last mark on the stack
  OBJ tuple = MakeTuple(compat_); // choose whether Arr or true Tup

  int tuple_start = mark_[mark_.length()-1];
  int tuple_length = pmstack_.length() - tuple_start;
  int stack_len = pmstack_.length();
  for (int ii=tuple_start; ii<stack_len; ii++) {
    TupleAppend(tuple, pmstack_(ii).object);
  }
  
  // Pop everything off
  for (int ii=0; ii<tuple_length; ii++) 
    pop_();
  mark_.removeLast(); // including the mark

  push_(tuple);
}

template <class OBJ>
void PythonDepicklerA<OBJ>::reduceArrays_ (const OBJ& tuple)
{
  if (!withNumericPackage_) {
    MakeWarning("Saw a Numeric array to Depickle even though the "
		"mode we're in doesn't support them: continuing ...");
  }
  OBJ single_tuple_of_elements      = TupleGet(tuple, 0); // int 
  OBJ py_typecode   = TupleGet(tuple, 1); // string python typecode
  OBJ string_data   = TupleGet(tuple, 2); 
  
  // Turn the raw bytes inside the string into appropriately aligned
  string typecode = GetString(py_typecode);
  string s        = GetString(string_data);
  int elements = 1;     // Shapeless Numeric array by default has 1 element
  int tuple_len = TupleLength(single_tuple_of_elements);
  if (tuple_len!=0) {
    elements    = GetInt4(TupleGet(single_tuple_of_elements,0));
  }
  
  OBJ res = MakeVector(typecode, elements, s.data(), s.length());
  push_(res);
}

template <class OBJ>
void PythonDepicklerA<OBJ>::reduceOrderedDict_ (const OBJ& tuple)
{
  // Tuple contains list of lists: o{'a':1, 'b':2} is [['a',1],['b',2]]
  OBJ top_list = ListGet(tuple, 0);

  OBJ res = MakeOrderedDict(compatibility());
  for (int ii=0; ii<ListElements(top_list); ii++) {
    OBJ pair = ListGet(top_list, ii);
    OrderedDictSet(res, ListGet(pair, 0), ListGet(pair,1));
  }
  push_(res);
}

template <class OBJ>
void PythonDepicklerA<OBJ>::reduceComplex_ (const OBJ& tuple)
{
  OBJ real_part = TupleGet(tuple, 0);
  OBJ imag_part = TupleGet(tuple, 1);
  OBJ full = MakeComplex(real_part, imag_part);
  push_(full);
}

template <class OBJ> void PythonDepicklerA<OBJ>::ploadREDUCE_ ()
{
  // I believe the Reduce "forces" a call with the args on the stack
  // from the global.  At this point, this is only used for Numeric
  // array constructor.  
  OBJ tuple   = pop_();
  string name = GetString(pop_());

  // Although Python uses Reduce much more generally, we currently
  // ONLY use it for reducing Numeric arrays and complex numbers!
  // This may change.  
  if (name=="Numeric\narray_constructor") {
    reduceArrays_(tuple);
    return;
  } else if (name=="__builtin__\ncomplex") {
    reduceComplex_(tuple);
    return;
  } else if (name=="collections\nOrderedDict") {
    reduceOrderedDict_(tuple);
  } else {
    MakeException("Unknown name/constructor:'"+name+
		  "' in REDUCE in PythonDepickler");
  }
}

template <class OBJ> void PythonDepicklerA<OBJ>::pAPPEND_ () 
{ 
  OBJ& list_ref   = top_(-1);
  OBJ  value      = pop_();
  ListAppend(list_ref, value);
}

template <class OBJ> void PythonDepicklerA<OBJ>::pSETITEM_ ()
{
  OBJ& value      = top_();
  OBJ& key        = top_(-1);
  OBJ& dict_ref   = top_(-2);
  
  TableSet(dict_ref, key, value);
  
  pop_(); // Drop key
  pop_(); // Drop value 
}

template <class OBJ> 
void PythonDepicklerA<OBJ>::pEMPTY_DICT_ () { push_(MakeTable()); }
template <class OBJ> 
void PythonDepicklerA<OBJ>::pDICT_ () 
{ mark_.removeLast(); push_(MakeTable()); }



template <class OBJ> 
PythonDepickler<OBJ>::PythonDepickler (const string& name, 
				  bool with_numeric_package) :
  PythonDepicklerA<OBJ>(with_numeric_package),
  fp_(0),
  buffer_(PMC_BUFF_EXPAND)
{
  buffer_.expandTo(PMC_BUFF_EXPAND);
  fp_ = fopen(name.c_str(), "r");
  if (fp_==0) MakeException("MalformedFile:"+name);
}

template <class OBJ> PythonDepickler<OBJ>::~PythonDepickler () 
{ 
  fclose(fp_); 
}

template <class OBJ> int PythonDepickler<OBJ>::getChar_ () 
{ return fgetc(fp_); }

template <class OBJ> char* PythonDepickler<OBJ>::getUntilNewLine_ (int& len)
//{ return fgets((char*)buffer_.data(), buffer_.length(), fp_); }
{
  buffer_.expandTo(0);
  for (;;) {
    int c = fgetc(fp_);
    if (c==EOF || c=='\n') { // all done
      buffer_.append('\0');
      len = buffer_.length() - 1;
      return buffer_.data();
    } else {
      char cc = c;
      buffer_.append(cc);
    }
  }
}

template <class OBJ> char* PythonDepickler<OBJ>::getString_ (int& len)
{
  int newline_len;
  char* a = getUntilNewLine_(newline_len); 
  char start_quote = *a;
  for (char *cp = a+1; *cp; ++cp) {
    if (*cp=='\\' && *(cp+1) !='\0') {
      ++cp; // skip this char ... and by contining, skip next
      continue;
    }
    if (*cp==start_quote) {
      *cp = '\0'; 
      len = cp-a-1;
      return a+1;
    }
  }
  return a;
}

// ///// PythonBufferDepickler

// The buffer is NOT adopted, but referenced.  Note: The buffer MAY
// BE modified while using it!  If you want to keep the buffer, make
// sure you copy it out first.
template <class OBJ>
PythonBufferDepickler<OBJ>::PythonBufferDepickler (int_u4 buff_len, 
						   char* buff,
						   bool with_numeric_package) :
  PythonDepicklerA<OBJ>(with_numeric_package),
  buffLen_(buff_len),
  buff_(buff),
  start_(0)
{ }


template <class OBJ> PythonBufferDepickler<OBJ>::~PythonBufferDepickler () { }

template <class OBJ> int PythonBufferDepickler<OBJ>::getChar_ () 
{ return (start_>=buffLen_) ? EOF: buff_[start_++];}

template <class OBJ> 
char* PythonBufferDepickler<OBJ>::getUntilNewLine_ (int& len)
{
  int buf_len = buffLen_;
  for (int ii=start_; ii<buf_len; ii++) {
    if (buff_[ii]=='\n') {
      buff_[ii] = '\0';
      len = ii-start_;
      char* retval = &buff_[start_];
      start_ = ii+1;
      return retval;
    }
  }

  // TODO:  Should we throw an exception or something else?
  MakeException("Reached end of buffer without finding newline");
  return 0;
}

template <class OBJ> char* PythonBufferDepickler<OBJ>::getString_ (int& len) 
{ 
  // NOTE:: This is a DESTRUCTIVE routine that actually changes
  // the buffer in memory!!! (just the string portion)

  // Expecting something like "some string \n in memory with \' quotes"\n
  
  // Should be starting with a ' or " character.  In the middle may
  // be escaped \' or \" characters or newlines which we need to
  // process correctly.  
  
  // After the final quote, there should be a
  // new line.  
  
  int ii = start_; int buf_len = buffLen_;
  for (char start_quote=buff_[ii++]; ii<buf_len; ++ii) {
    char current = buff_[ii];
    if (current==start_quote) { 
      // Think we saw end of string: look for \n
      if (ii+1>=buf_len || buff_[ii+1]!='\n') 
	MakeException("Unexpected EOF in string");
      buff_[ii++] = '\0'; // Saw the newline, put \0 at quote to end
      
      // Valid string.  Update the start to be beyond \n
      char* retval = &buff_[start_+1]; // One past start_quote
      len = ii-start_-1;
      start_ = ii+1;
      return retval;
    }
    if (current=='\\') ii++; // Move past escape
  }
  MakeException("Unexpected EOF in string");
  return 0;
}


// ///// PythonPicklerA


template <class OBJ>
PythonPicklerA<OBJ>::PythonPicklerA (bool with_numeric_package) :
  arrDisp_(with_numeric_package ? AS_NUMERIC : AS_LIST),
  currentIntHandle_(1),
  warn_(false),
  compat_(OC_SERIALIZE_COMPAT)
{ } 

template <class OBJ>
PythonPicklerA<OBJ>::PythonPicklerA (ArrayDisposition_e arr_disp) :
  arrDisp_(arr_disp),
  currentIntHandle_(1),
  warn_(false),
  compat_(OC_SERIALIZE_COMPAT)
{ } 

// Your derived destructor MUST CALL putChar_(PY_STOP) so that it puts
// the final stop mark in!
template <class OBJ> PythonPicklerA<OBJ>::~PythonPicklerA () { } 

template <class OBJ> void PythonPicklerA<OBJ>::dumpVector (const OBJ& v,
							   int int_handle)
{
  if (arrDisp_==AS_NUMERIC) {
    dumpNumericArray_(v, int_handle); 
  } else if (arrDisp_==AS_NUMPY) {
    dumpNumPyArray_(v, int_handle);
  } else {
    dumpVectorAsList_(v, int_handle);
  }
}


template <class OBJ>
void PythonPicklerA<OBJ>::dumpNumPyArray_ (const OBJ& v, int int_handle)
{
  putChar_(PY_GLOBAL); 
  putStr_("numpy.core.multiarray\n_reconstruct\n"); 
  putChar_(PY_MARK); 
  {
    putStr_("cnumpy\nndarray\n");
    putStr_("(I0\ntS'b'\ntR(I1\n(");
    OBJ elements = PutInt4(VectorElements(v));    
    dump(elements);
    putStr_("tcnumpy\ndtype\n");
    string typecode = OBJToNumPyCode(v);
    putStr_("(S'"+typecode+"'\n");
    putStr_("I0\nI1\ntR(I3\n");
    string endian = IsLittleEndian() ? "<" : ">";
    putStr_("S'"+endian+"'\nNNNI-1\nI-1\nI0\ntbI00\n");
    
    // Get the vector as a python printable string
    string vector_as_string = BuildNumPyVector(v, endian);
    dumpString(vector_as_string, false);
  }
  
  putChar_(PY_TUPLE);
  putChar_(PY_BUILD);
  if (int_handle!=-1) placeHandle_(int_handle, PY_PUT);
}



template <class OBJ> void PythonPicklerA<OBJ>::dumpString (const string& str,
						       bool translate_escapes)
{
  if (translate_escapes) {
    // Make sure the string doesn't have any weird escape sequences
    Array<char> v(str.length());
    PrintBufferToString(str.data(), str.length(), v);
    
    putChar_(PY_STRING); 
    putChar_('\''); 
    putStr_(v.data()); 
    putStr_("'\n"); 
  } else {
    putChar_(PY_STRING); 
    putChar_('\''); 
    putStr_(str.c_str()); 
    putStr_("'\n"); 
  }
}

template <class OBJ> void PythonPicklerA<OBJ>::dumpTable (const OBJ& t,
							  int int_handle)
{
  putChar_(PY_MARK); putChar_(PY_DICT);
  if (int_handle!=-1) placeHandle_(int_handle, PY_PUT);
  for (TableIterator ii(t); ii(); ) {
    dump(ii.key());   
    dump(ii.value()); 
    putChar_(PY_SETITEM);
  }
}

template <class OBJ> void PythonPicklerA<OBJ>::dumpOrderedDict (const OBJ& t,
								int int_handle)
{
  // In compatibility mode, this has to dump as a Table
  if (compatibility()) { 
    OBJ copy;
    Tablify(t, copy);
    dumpTable(copy, int_handle); 
    return;
  }

  // Otherwise, we have to have tuples to place this correctly: that's
  // why tuples and OrderedDicts come in at the same time
  putChar_(PY_GLOBAL);
  putStr_("collections\nOrderedDict\n");
  // TODO: int new_handle = currentIntHandle_++ ... watch when incremented!
 
  putChar_(PY_MARK); // Outer TUPLE (arg to OrderedDict)
  {
    // The outer level contains another list 
    putChar_(PY_MARK);
    putChar_(PY_LIST);    
    {
      // Each entry of the ordered dict is a list
      for (TableIterator ii(t); ii(); ) {
	putChar_(PY_MARK);
	putChar_(PY_LIST);
	{
	  dump(ii.key());
	  putChar_(PY_APPEND);
	  dump(ii.value());
	  putChar_(PY_APPEND);
	}
	putChar_(PY_APPEND); // outer list (which just contains inner list)
      }
    }
  }
  putChar_(PY_TUPLE);
  putChar_(PY_REDUCE);
}


template <class OBJ> void PythonPicklerA<OBJ>::dumpTuple (const OBJ& t,
							  int int_handle)
{
  // In compatibility mode, this has to dump as a List
  if (compatibility()) { 
    dumpList(t, int_handle); 
    return;
  }
  // Otherwise, dump a true tuple
  putChar_(PY_MARK);
  for (int ii=0; ii<ListElements(t); ii++) {
    dump(ListGet(t,ii));
  }
  putChar_(PY_TUPLE);
  if (int_handle!=-1) placeHandle_(int_handle, PY_PUT);
}


template <class OBJ> void PythonPicklerA<OBJ>::dumpProxy (const OBJ& t)
{
  // See if this has already been dumped.  If so, just mark the int_handle
  // of the previous dumpee.  Otherwise, dump from scratch and mark it.
  void* ptr_handle = GetProxyHandle(t);
  int int_handle = 0xDEADC0DE;
  if (handles_.contains(ptr_handle)) {
    // Already dumped, just dump a GET
    int_handle = handles_[ptr_handle];
    placeHandle_(int_handle, PY_GET);
  } else {
    // New!  Remember this new guy
    int_handle = currentIntHandle_++;
    handles_[ptr_handle] = int_handle;

    // First time, dump whole thing (plus PUT marker)
    if (IsProxyTable(t)) {
      dumpTable(t, int_handle);
    } else if (IsProxyList(t)) {
      dumpList(t, int_handle);
    } else if (IsProxyVector(t)) {
      dumpVector(t, int_handle);
    } else if (IsProxyOrderedDict(t)) {
      dumpOrderedDict(t, int_handle);
    } else if (IsProxyTuple(t)) {
      dumpTuple(t, int_handle);
    } else {
      MakeException("Unknown proxy type");
    }
  }

}


template <class OBJ> void PythonPicklerA<OBJ>::dumpList (const OBJ& l,
							 int int_handle)
{
  putChar_(PY_MARK); putChar_(PY_LIST);
  if (int_handle!=-1) placeHandle_(int_handle, PY_PUT);
  const int list_elements = ListElements(l);
  for (int ii=0; ii<list_elements; ii++) {
    dump(ListGet(l, ii));
    putChar_(PY_APPEND);
  }
}

template <class OBJ> void PythonPicklerA<OBJ>::dumpNumber (const OBJ& n)
{
  string nstr, nstr2; // arg2 only used for complexes
  char choose = ChooseNumber(n, nstr, nstr2);
  switch (choose) {
  case 'i': { 
    putChar_(PY_INT); putStr_(nstr); putChar_('\n'); 
    break;
  }
  case 'q': {
    if (compatibility()) {
      dumpString(nstr);
      break;
    } 
    // else fall through
  }
  case 'l': {
    putChar_(PY_LONG); putStr_(nstr);  putStr_("L\n"); break;
  }
  case 'd': {
    putChar_(PY_FLOAT); putStr_(nstr); putChar_('\n'); break;
  }
  case 'D' : {
    putChar_(PY_GLOBAL); putStr_("__builtin__\ncomplex\n"); putChar_(PY_MARK);
    putChar_(PY_FLOAT);  putStr_(nstr);  putChar_('\n');
    putChar_(PY_FLOAT);  putStr_(nstr2); putChar_('\n');
    putChar_(PY_TUPLE);  putChar_(PY_REDUCE);
    break;
  }
  case 'a' : {
    // Dumped as string
    dumpString(nstr);
    break;
  }
  default:
    MakeException("dumpNumber:do not know how to serialize numeric type:"+nstr);
  }
}


template <class OBJ>
void PythonPicklerA<OBJ>::placeHandle_ (int int_handle, char code)
{
  putChar_(code); // PY_PUT or PY_GET usually
  string nstr, x;
  (void)ChooseNumber(PutInt4(int_handle), nstr, x);
  putStr_(nstr);
  putChar_('\n');
}


template <class OBJ>  
void PythonPicklerA<OBJ>::dump (const OBJ& val)
{
  if (IsProxy(val))       dumpProxy(val); // Needs to be first isproxy is independent of tag
  else if      (IsNone(val))   putChar_(PY_NONE); 
  else if (IsBool(val))   dumpBool(val); 
  else if (IsNumber(val)) dumpNumber(val);
  else if (IsString(val)) dumpString(GetString(val));
  else if (IsVector(val)) dumpVector(val);
  else if (IsTable(val))  dumpTable(val);
  else if (IsList(val))   dumpList(val);
  else if (IsTuple(val))  dumpTuple(val);
  else if (IsOrderedDict(val))   dumpOrderedDict(val);

  else UnknownType(*this, val);
}

template <class OBJ> void PythonPicklerA<OBJ>::dumpBool (const OBJ& v)
{
  putChar_(PY_INT); putChar_('0'); 
  if (IsTrue(v)) putChar_('1'); else putChar_('0');
  putChar_('\n');
}

template <class OBJ>
void PythonPicklerA<OBJ>::dumpVectorAsList_ (const OBJ& v,
					     int int_handle)
{
  putChar_(PY_MARK); putChar_(PY_LIST);
  if (int_handle!=-1) placeHandle_(int_handle, PY_PUT);
  const int elements = VectorElements(v);
  for (int ii=0; ii<elements; ii++) {
    dump(VectorGet(v,ii));
    putChar_(PY_APPEND);
  }
}

template <class OBJ>
void PythonPicklerA<OBJ>::dumpNumericArray_ (const OBJ& v, int int_handle)
{
  putChar_(PY_GLOBAL); 
  putStr_("Numeric\narray_constructor\n"); 
  {
    putChar_(PY_MARK); 
    {
      putChar_(PY_MARK);
      OBJ elements = PutInt4(VectorElements(v));
      dump(elements);
      putChar_(PY_TUPLE);
    }

    string typecode = BestFitForVector(v); // Get python typecode for vector
    dumpString(typecode);

    // Get the vector as a python printable string
    string vector_as_string = BuildVector(v, typecode); 
    dumpString(vector_as_string, false);
    
    OBJ dims = PutInt4(1); // TODO:  I think this is dimensions ...
    dump(dims);

    putChar_(PY_TUPLE);
  }
  putChar_(PY_REDUCE);
  if (int_handle!=-1) placeHandle_(int_handle, PY_PUT);
}


// ///// PythonPickler  
template <class OBJ>
PythonPickler<OBJ>::PythonPickler (const string& name, 
				   bool with_numeric_package) :
  PythonPicklerA<OBJ>(with_numeric_package),
  fp_(fopen(name.c_str(), "w"))
{ 
  if (fp_==0) 
    MakeException("FileError opening:"+name);
}

template <class OBJ>
PythonPickler<OBJ>::PythonPickler (const string& name, 
				   ArrayDisposition_e arr_disp) :
  PythonPicklerA<OBJ>(arr_disp),
  fp_(fopen(name.c_str(), "w"))
{ 
  if (fp_==0) 
    MakeException("FileError opening:"+name);
}


template <class OBJ> PythonPickler<OBJ>::~PythonPickler () 
{ 
  putChar_(PY_STOP); 
  fclose(fp_); 
}

template <class OBJ> void PythonPickler<OBJ>::putChar_ (char c) 
{ fputc(c, fp_); }

template <class OBJ> void PythonPickler<OBJ>::putStr_ (const char* s) 
{ fputs(s, fp_); }

template <class OBJ> void PythonPickler<OBJ>::putStr_ (const string& s) 
{ fputs(s.c_str(), fp_); }


// ///// PythonBufferPickler  
template <class OBJ>
PythonBufferPickler<OBJ>::PythonBufferPickler (Array<char>& buffer, 
					       bool with_numeric_package) :
  PythonPicklerA<OBJ>(with_numeric_package),
  buffer_(buffer)
{ }

template <class OBJ>
PythonBufferPickler<OBJ>::PythonBufferPickler (Array<char>& buffer, 
					       ArrayDisposition_e arr_dis):
  PythonPicklerA<OBJ>(arr_dis),
  buffer_(buffer)
{ }
  
template <class OBJ> PythonBufferPickler<OBJ>::~PythonBufferPickler () 
{ putChar_(PY_STOP); }

template <class OBJ> void PythonBufferPickler<OBJ>::putChar_ (char c) 
{ buffer_.append(c); }

template <class OBJ> void PythonBufferPickler<OBJ>::putStr_ (const char *s) 
{ while (*s) buffer_.append(*s++); }

template <class OBJ> void PythonBufferPickler<OBJ>::putStr_ (const string& s) 
{ putStr_(s.c_str()); }


// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_END_NAMESPACE
#endif

