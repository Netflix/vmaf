#ifndef OPALUTILS_H_

// Tools for converting from OpalTables to Python Dictionaries
// (or from OpenContainers nomenclature, Opaltables to Tabs)

// A class to help read "human-readable" OpalTables and turn
// them into Vals (Tabs, Arrs, numbers, Numeric arrays).  Format 
// is straight-forward:
//
//   OpalReader v(" { a="nice", table=I:1 }");
//   Tab t;
//   v.expectTab(t);
//
// From Files (streams):
//   StreamOpalReader sv(istream& is);
//   Tab t;
//   sv.expectTab(t);

// Includes 
#include "ocval.h"
#include "ocstringtools.h"
#include "opalprint.h"
#include "ocreader.h"

PTOOLS_BEGIN_NAMESPACE

// Abstract base class: All the code for parsing the letters one by
// one is here.  The code for actually getting the letters (from a
// string, stream, etc.) defers to the reader class.
class OpalReaderA { 

 public: 

  OpalReaderA (ReaderA* adopted_reader) :
    reader_(adopted_reader) 
  { }

  virtual ~OpalReaderA () { delete reader_; }

  bool EOFComing () { return reader_->EOFComing(); }
 
  // Expect a number: it may have a tag
  void expectNumber (Val& n) 
  {
    char tag = expectTag_('\0');
    expectNumber(n, tag);
  }

  // Expect a number, force a tag: '\0' means no tag
  void expectNumber (Val& n, char tag)
  {
    if (tag == '[') { 
      expectTime_(n);  // Turn a time into a string
      return;
    } else if (tag==']') {
      expectNumber(n); // Turn a duration into a plain number, double
      real_8 r = n;
      n = r;
      return;
    }
    expectPlainNumber(n);
    // The tag overrides whatever decision was made for the literal
    switch (tag) {
    case 's': { int_1 v = n; n = v; break; }
    case 'i': { int_2 v = n; n = v; break; } 
    case 'l': { int_4 v = n; n = v; break; } 
    case 'x': { int_8 v = n; n = v; break; }
    case 'S': { int_u1 v = n; n = v; break; }
    case 'I': { int_u2 v = n; n = v; break; }
    case 'L': { int_u4 v = n; n = v; break; }
    case 'X': { int_u8 v = n; n = v; break; }
    case 'f': { real_4 v = n; n = v; break; }
    case 'd': { real_8 v = n; n = v; break; }
    case 'F': { complex_8 v = n; n = v; break; }
    case 'D': { complex_16 v = n; n = v; break; }
    case 'b': { bool v = n; n = v; break; }
    case '\0' : break; // There was no tag
      default: 
	char tagme[2]; tagme[0] = tag; tagme[1]='\0';
	syntaxError_("Unknown Numeric Type tag:"+string(tagme));
    }
  }



  // Expect any number (including complex)
  void expectPlainNumber (Val& n)
  {
    consumeWS_();
    int c=peekChar_();
    if (c=='(') {
      expectComplex_(n);
      return;
    }

    // Get the integer part, if any, of the number
    Str integer_part = getSignedDigits_('.');

    // Get the fractional part, if any
    Str fractional_part;
    c = peekChar_();
    if (c=='.') {     
      c = getChar_(); // consume the '.'
      fractional_part = "."+getKey_(OPAL_DIGIT);
      if (fractional_part.length()==1) { 
	const int i_len = integer_part.length();
	if (i_len==0 || (i_len>0 && !isdigit(integer_part[i_len-1]))) {
	  syntaxError_("Expecting some digits after a decimal point");
	}
      }
      c = peekChar_();
    }

    // Get the exponent part, if any
    Str exponent_part;
    if (c=='e' || c=='E') {
      c = getChar_();  // consume the 'e'
      exponent_part = "e"+getSignedDigits_(' ');
      if (exponent_part.length()==1) // only an e
	syntaxError_("Expected '+', '-' or digits after an exponent");
      c = peekChar_();
    }

    // At this point, we are (mostly) finished with the number, and we
    // have to build the proper type of number.
    if (fractional_part.length()>0 || exponent_part.length()>0) {
      // If we have either a fractional part or an exponential part,
      // then we have a floating point number
      Str stringized_number = integer_part+fractional_part+exponent_part;
      Val inside = stringized_number; 
      real_8 num = inside; // Convert out of Val to change from string 
      n = num;
      return;
    }
    
    // Well, no fractional part or exponential.  There had better be
    // some digits!
    if (integer_part.length()==0 || 
	(integer_part.length()>0 && 
	 !isdigit(integer_part[integer_part.length()-1])))
      syntaxError_("Expected some digits for a number");
	
    c=peekChar_();
    Val v = integer_part;
    if (c=='l' || c=='L') { // Okay, it's a long
      getChar_();  // consume long
      if (integer_part[0]=='-') {
	int_8 long_int = v;
	n = long_int;
	return;
      } else {
	int_u8 long_int = v;
	n = long_int;
	return;
      } 
    } else { // plain integer
      int_4 plain_int = v;
      n = plain_int;

      // In case it didn't convert well
      if (plain_int==0) {
	bool all_zeroes = true;
	for (size_t ii=0; ii<integer_part.length(); ii++) {
	  if (isdigit(integer_part[ii]) && integer_part[ii]!='0') {
	    all_zeroes = false;
	  }
	}
	if (all_zeroes) return;  // All good 
	// Otherwise, reconsider how to parse this large number
	int_8 big_plain = v;
	n = big_plain;
	
	string bps = n;
	string string_v = v;
	//cout << bps << " " << string_v << " " << (bps==string_v) << endl;
	if ((bps ==string_v) && big_plain!=0) return;
	// if (big_plain!=0) return;  // Parsed okay ...
	// Otherwise, try a double
	real_8 big_d = v;
	n = big_d;
	return;
      }
      return;
    }
  }

  // Read a string from the current place in the input
  void expectStr (Str& s)
  { 
    consumeWS_();

    char quote_mark = peekNWSChar_();
    if (quote_mark!='"') {
      syntaxError_("A string needs to start with \"");
    }

    expect_(quote_mark); // Start quote

    // Read string, keeping all escapes, and let DeImage handle escapes
    Array<char> a(80);
    for (int c=getChar_(); c!=quote_mark; c=getChar_()) {
      if (c==EOF) syntaxError_("Unexpected EOF inside of string");
      a.append(c);
      if (c=='\\') { // escape sequence
	int next = getChar_(); // Avoid '
	if (next==EOF) syntaxError_("Unexpected EOF inside of string");
	a.append(next);
      } 
    }    
    string temp = string(a.data(), a.length());
    string ss = DeImage(temp, false); // Do escapes 
    s = Str(ss.data(), ss.length());
  }

  // Expect Table on the input.  Returns true if this Tab can be
  // converted into an Arr (because all keys are integers from 0 .. n-1).
  // if we find a Tab that could be an Array AND convert_to_array is true,
  // we do so.
  bool expectTab (Tab& table, bool convert_tab_to_array=true)
  {
    // Allow us to tag how we are supposed to be reading this list
    enum HandleInput_e { VALUE_COMMA_LIST, KEY_EQUALS_VALUE };

    //cerr << "DEBUG: Starting a TAB" << endl;
    expect_('{');
    bool can_be_array = true;  // If all keys goes 0..n-1, can convert to Arr
   
    // Special case, empty Table
    int peek = peekNWSChar_();
    if (peek!='}') {
      
      for (int_u4 ii=0;;ii++) { // Continue getting key value pairs
	peek = peekNWSChar_();
	if (peek==EOF) {
	  syntaxError_("Expected a key and saw EOF");
	}
	consumeWS_();

	// What's next?  A sequence of 
	// characters without quoting is "probably" a key for KEY=VALUE table:
	// but it could also be type-tags:something (which implies V1, V2 list)
	// a string could be either VAL1, VAL2, ... or KEY=VALUE
	// anything else has to be a VAL1, VAL2, .. 
	HandleInput_e handle_input = VALUE_COMMA_LIST;
	string key;
	if (isalnum(peek) || peek=='_') {
	  key = getKey_(OPAL_ALPHA|OPAL_DIGIT|OPAL_UNDER);  
	  peek = peekNWSChar_();
	  if (peek=='=') {  // = so table
	    // Falls through below: SAW KEY
	    handle_input = KEY_EQUALS_VALUE;
	  } else { // if (peek==':' || peek==',' || peek=='}') {         // type-tag or , , so list
	    // Pushback so we can handle this as an arbitrary value in LIST
	    for (int jj=int(key.length())-1; jj>=0; jj--) {
	      pushback_(key[jj]);
	    }
	    handle_input = VALUE_COMMA_LIST;
	  } 
	  //else {                 // don't know, so syntax error
	  //syntaxError_("With "+key+", expected = (for key=value)," 
	  //	 ": (for tag) or , (for list) or } (for end of list)");
	  //}
	} 

	// So NOT some characters, so must be a VALUE, VALUE , ... list
	if (handle_input == VALUE_COMMA_LIST) {
	  Val value;
	  expectAnything(value);
	  peek = peekNWSChar_();
	  if (peek==',') {
	    table.append(value);
	    expect_(",");
	    continue;           // done with this!
	  } else if (peek==EOF) {
	    syntaxError_("Premature EOF: expected , or } for part of list");
	  } else if (peek=='}') {
	    table.append(value);
	    break; // All done!
	  } else if (peek=='=') {
	    // Okay: we should only allow what OpalTables allow: a string
	    if (value.tag != 'a') {
	      syntaxError_("Illegal key (only strings allowed):"+string(value)+ 
			   " before =");
	    } else {
	      key = string(value);
	      handle_input = KEY_EQUALS_VALUE;
	      // Falls through
	    }
	  } else {
	    syntaxError_("Expected separator = or , (or end of list })");
	  }
	}

	// If we make it here, then we have seen a KEY for a KEY = VALUE
	// expectation
	if (handle_input != KEY_EQUALS_VALUE)  {
	  syntaxError_("Unknown input??");
	}
	string string_ii = Stringize(ii);
	if (key!=string_ii) {
	  can_be_array = false; 
	} else {
	  ;
	}

	expect_('=');   
	peek = peekNWSChar_();
	if (peek=='@') {
	  expect_("@");  // TODO:  Real links?
	  //cerr << "Got at" << endl;
	}	
	
	Val value; expectAnything(value, convert_tab_to_array); 
	//cerr << "value:" << value << endl; 
	
	table[key] = value;
	
	char peek = peekNWSChar_();
	if (peek==',') {
	  expect_(',');         // Another k-v pair, grab comma and move on
	  peek = peekNWSChar_();
	  if (peek=='}') break;  // CAN have extra , at the end of the table
	  else continue;
	} else if (peek=='}') { // End of table
	  break;
	} else {
	  syntaxError_("Expecting a '}' or ',' for table");
	}
      }
    }
    expect_('}');
    return can_be_array;
  }

  // Expect an Arr
  void expectArr (Arr& a, bool convert_tab_to_array=true)
  {
    expect_('{');
    
    // Special case, empty list
    char peek = peekNWSChar_();
    if (peek!='}') {
      
      // Continue getting each item one at a time
      while (1) {    
	a.append(Val());
	expectAnything(a[a.length()-1], convert_tab_to_array); 
	
	char peek = peekNWSChar_();
	if (peek==',') {
	  expect_(',');
	  continue;
	} else if (peek=='}') {
	  break;
	} 
      }
    }
    expect_('}');
  }

  // Read in a Numeric Array
  void expectNumericArray (Val& a)
  {
    consumeWS_();
    char tag = expectTag_('Z');
    expectNumericArray(a, tag);
  }


  void expectNumericArray (Val& a, char tag)
  {
    expect_("<");
    char peek = peekNWSChar_();
    if (peek!='>') {
      dispatchNumericArray_(a, tag);
    } else { // Special case, empty list
      a = Array<real_8>(); // By default, real_8 array
    }
    expect_(">");
  }


  void expectMultiVector (Val& mv)
  {
    expect_("MV(");
    mv = Arr();
    for (int peek=peekNWSChar_(); peek!=')'; peek=peekNWSChar_()) {
      // Expecting a numeric array
      mv.append(Val());
      Val& last = mv[mv.length()-1];
      expectNumericArray(last);
      // Expecting , or (
      peek = peekNWSChar_();
      if (peek==',') {
	expect_(',');
      }
    }
    expect_(")"); 
  }

  // Convert a Tab, will all integer keys, to an Arr
  Arr convertTabToArr (const Tab& table)
  {
    Arr a(table.entries());
    int tab_ent = table.entries();
    for (int ii=0; ii<tab_ent; ii++) {
      if (table.contains(ii)) {
	a.append(table(ii));
      } else { 
	a.append(table(Stringize(ii)));
      }
    }
    return a;
  }

  // We could be starting ANYTHING ... we have to dispatch to the
  // proper thing.  If we find Tab, we convert it to an Arr if it
  // makes sense.
  void expectAnything (Val& v, bool convert_tab_to_array=true)
  {
    char c = peekNWSChar_();
    switch (c) {
    case '{' : { 
      v = Tab(); 
      Tab& table = v; 
      bool can_be_array = expectTab(table, convert_tab_to_array);
      if (table.entries()!=0 && convert_tab_to_array && can_be_array) {
	v = convertTabToArr(table);
      }
      break; 
    }
    case '[' : { v = Arr(); Arr& arr=v;   expectArr(arr, convert_tab_to_array); break; }
    case '\'':
    case '"' : {            Str s = "";   expectStr(s); v=s;     break; }
    case '<' : {                          expectNumericArray(v); break; }
    case 'M' : {                          expectMultiVector(v);  break; }
    default: { 
      // This is either a number D:1 or a Vector D:<1,2,3>
      // We have to take a look at the tag
      char tag = expectTag_('\0'); 
      int ii = peekNWSChar_();
      if (ii==EOF) { 
	syntaxError_("Expected something at EOF"); 
      } else if (ii=='<') { // If we see a '<', expect a vector
	v = Val();                 expectNumericArray(v, tag);       break; 
      } else {              // Otherwise, try to read as a number
	v = Val();                 expectNumber(v, tag);       break; 
      }
    }
    }
  }


 protected:

  // Helper Methods
  void syntaxError_ (const string& s) 
  {
    reader_->syntaxError(s);
  }

  void expect_ (char expected)
  {
    // Single character token
    int get      = getNWSChar_();
    if (get!=expected) {
      string get_string, expected_string;
      if (get==EOF) get_string="EOF"; else get_string=get;
      expected_string = expected;
      syntaxError_("Expected:'"+expected_string+"', but saw '"+get_string+"' "
		   "on input");
    }
  }

  void expect_ (const string& s)
  {
    int slen = s.length();
    for (int ii=0; ii<slen; ii++) 
      expect_(s[ii]);
  }



  // Expect a numeric tag to indicate the type of number, if none is
  // specified. Essentially looking for 'tag:' where tag is a single
  // letter (or double letter)..
  char expectTag_ (char default_tag='d') 
  {
    static Tab opal_to_octag("{"
			     "  'B' :'s',"
			     "  'I' :'i',"
			     "  'L' :'l',"
			     "  'X' :'x',"
			     "  'F' :'f',"
			     "  'D' :'d',"
			     "  'CF':'F',"
			     "  'CD':'D',"
			     "  'UB' :'S',"
			     "  'UI' :'I',"
			     "  'UL' :'L',"
			     "  'UX' :'X',"
			     "  'CF' :'F',"
			     "  'CD' :'D',"
			     "  'T'  :'[',"
			     "  'DUR':']',"
                             "  'BIT':'b'"
			     "}");

    // Key the next letters upto the (hopefully) ':'
    string original_tag = getKey_(OPAL_ALPHA);
    string tag = original_tag;
    for (size_t ii=0; ii<tag.length(); ii++) tag[ii]=toupper(tag[ii]);

    if (tag=="") {                     // Probably no tag, just return default
      return default_tag;
    } else if (opal_to_octag.contains(tag)) { // Key there:  return that tag
      expect_(':');
      string octag = opal_to_octag[tag];
      return octag[0];
    } else {                                 // Key not there:  Syntax error
      syntaxError_("Unknown Numeric Tag:'"+original_tag+"'"); 
    }
    return default_tag;
  }

  void expectTime_ (Val& t)
  {
    // TODO: Make this more robust
    Str tt;
    while (1) {
      tt += getKey_(OPAL_DIGIT);
      int peek = peekChar_();
      if (peek==EOF) break;
      if (peek==':' || peek=='.') {
	tt += peek;
	getChar_();
      } else {
	break;
      } 
    }
    t = tt;
    if (tt=="") syntaxError_("Malformed time tag:'"+tt+"'");
  }

  // Expect a complex number:  assumes it will have form (#,#)
  void expectComplex_ (Val& n)
  {
    complex_16 result;
    (void)expectTag_('D');  // Default to complex_16

    expect_('(');
    Val real_part, imag_part;
    expectNumber(real_part);
    consumeWS_();
    expect_(',');
    expectNumber(imag_part);
    consumeWS_();
    expect_(')');

    result.re = real_part;
    result.im = imag_part;
    n = result;
  }


  // From current point in input, get the signed sequence of 
  // digits, alpnums, or _
  Str getSignedDigits_ (char next_marker)
  {
    // Get the sign of the number, if any
    char c=peekChar_();
    Str sign;
    if (c=='+'||c=='-') {
      sign = c;
      getChar_();    // consume the sign
      c=peekChar_(); // .. and see what's next
    }

    // Assertion: passed sign, now should see number or .
    if (!isdigit(c)&&c!=next_marker) {
      char s[2]; s[0] = next_marker; s[1] = '\0';
      syntaxError_("Expected numeric digit or '"+string(s)+"' for number");
    }
    
    Str digits = sign+getKey_(OPAL_DIGIT);
    return digits;
  }
  
  template <class T>
  void readNumericArray_ (Val& arr, Val first_number=None)
  { 
    arr = Array<T>();  // initialize the array
    Array<T>& a = arr; // .. then a reference to it

    // In case first number was used as an indicator
    if (first_number != None) {
      T first = first_number;
      arr.append(first);
    }
    
    // Continue getting each item one at a time
    while (1) {
      char peek = peekNWSChar_();
      if (peek==',') {
	expect_(',');
	continue;
      } else if (peek=='>') {
	break;
      } 

      Val n; 
      expectNumber(n);
      T v = n;
      a.append(v);
    }
  }

  // Helper routine:  Grab everything BUT the '<' and '>' markers.
  void dispatchNumericArray_(Val& a, char tag, Val first_number=None)
  {
    switch (tag) {
    case 's':  readNumericArray_<int_1>(a, first_number); break;
    case 'i':  readNumericArray_<int_2>(a, first_number); break;
    case 'l':  readNumericArray_<int_4>(a, first_number); break;
    case 'x':  readNumericArray_<int_8>(a, first_number); break;
    case 'S': readNumericArray_<int_u1>(a, first_number); break;
    case 'I': readNumericArray_<int_u2>(a, first_number); break;
    case 'L': readNumericArray_<int_u4>(a, first_number); break;
    case 'X': readNumericArray_<int_u8>(a, first_number); break;
    case 'f':  readNumericArray_<real_4>(a, first_number); break;
    case 'd':  readNumericArray_<real_8>(a, first_number); break;
    case 'F': readNumericArray_<complex_8>(a, first_number); break;
    case 'D': readNumericArray_<complex_16>(a, first_number); break;
    case 'b':  readNumericArray_<bool>(a, first_number); break;
    case 'Z': {
      // Special: read the first number and have that be the indicator
      // as to the type of the Numeric Array
      Val nn;
      expectNumber(nn);
      dispatchNumericArray_(a, nn.tag, nn);
      break;
    }        
    default: 
      throw logic_error("Unknown Numeric Type tag:"+tag);
    }
  } 


  // From current point of input, consume all until
  // next non-alnum
  enum OPAL_CHECK { OPAL_ALPHA = 0x1, OPAL_DIGIT = 0x2, OPAL_UNDER = 0x4 };
  Str getKey_ (unsigned int mask)
  {
    Str key;
    while (1) {
      int c = peekChar_();
      if (c==EOF) 
	break;

      bool tester = 0;
      if (mask & OPAL_ALPHA) tester = tester || isalpha(c);  
      if (mask & OPAL_DIGIT) tester = tester || isdigit(c);   
      if (mask & OPAL_UNDER) tester = tester || (c=='_');   
      if (tester) {
	key+=char(c);
	getChar_();
      }
      else 
	break;
    }
    return key;
  }

  // Dispatch for input
  int getNWSChar_ ()  { return reader_->getNWSChar_(); }
  int peekNWSChar_ () { return reader_->peekNWSChar_(); }
  int getChar_ ()     { return reader_->getChar_(); }
  int peekChar_ ()    { return reader_->peekChar_(); }
  int consumeWS_ ()   { return reader_->consumeWS_(); }
  void pushback_ (int put) { reader_->pushback_(put); }

  // Defer IO to another class.  All sorts of discussion on why
  // didn't we inherit, etc.  Look at the Design Patterns book.
  ReaderA* reader_; 

}; // OpalReaderA


// Helper class to handle reading the comments from an Opal String
class OpalStringReader_ : public StringReader {

 public :

  OpalStringReader_ (const char* s, int len=-1) : StringReader(s, len) { }
  OpalStringReader_ (Array<char>& a) : StringReader(a) { } 

 protected:
  // Return the index of the next Non-White Space character.
  // A comment starts with a // and ends with a \n, and counts
  // as white space.  **Returns an index possibly outside of buffer**
  virtual int indexOfNextNWSChar_ () 
  {
    const int len=length_;
    int cur = current_;
    int maybe_start_comment = -2;
    if (cur==len) return cur;
    // Look for WS or comments that start with //
    bool comment_mode = false;
    for (; cur<len; cur++) {
      if (comment_mode) {
	if (data_[cur]=='\n') { comment_mode = false; }
	continue;
      } else {
	if (isspace(data_[cur])) continue;
	else if (data_[cur]=='/') {
	  if (maybe_start_comment+1 == cur) {
	    comment_mode = true;
	  } else {
	    maybe_start_comment = cur;
	  }
	  continue;
	} else {
	  break;
	}
      }
    }
    return cur;
  }

}; // OpalStringReader

// The OpalReader reads Vals from strings that are formatted with
// OpalTables.  
class OpalReader : public OpalReaderA {

 public:

  OpalReader (Array<char>& a) :
    OpalReaderA(new OpalStringReader_(a))
  { }

    OpalReader (const char* s, int len=-1) :
    OpalReaderA(new OpalStringReader_(s, len))
    { }
  
}; // OpalReader


// Helper function for reading ASCII OpalStreams:  it handles
// the // comments
class OpalStreamReader_ : public StreamReader {

 public:

  OpalStreamReader_ (istream& is) : StreamReader(is) { }

 protected:
  // This routines buffers data up until the next Non-White space
  // character, ands returns what the next ws char is _WITHOUT GETTING
  // IT_.  It returns "peek_ahead" to indicate how many characters
  // into the stream you need to be to get it.
  virtual int peekIntoNextNWSChar_ (int& peek_ahead)
  {
    peek_ahead = 0;  // This marks how many characters into the stream
                     // we need to consume
    bool saw_one_slash_already = false;
    bool start_comment = false;
    while (1) {
      // Peek at a character either from the cache, or grab a new char
      // of the stream and cache it "peeking" at it.
      int c;
      if (peek_ahead >= cached_.length()) {
	c = is_.get();
	cached_.put(c);
      } else {
	c = cached_.peek(peek_ahead);
      }

      // Look at each character individually
      if (c==EOF) {
	// We never consume the EOF once we've seen it
	return c;
      } else if (start_comment) {
	peek_ahead++;
	start_comment = (c!='\n');
	continue;
      } else if (c=='/') { 
	if (saw_one_slash_already) { // two in a row, starting comment
	  start_comment = true;
	  peek_ahead++;
	  continue;
	} else { // only one in a row, this is next non white space char
	  saw_one_slash_already = true;
	  peek_ahead++;
	  continue;
	}

      } else if (isspace(c)) { // white and comments
	peek_ahead++;
	continue;
      } else {
	return c;
      }
    }
  }  
}; // StreamReader


// Read Opal values from a stream
class StreamOpalReader : public OpalReaderA {

 public:

  // Open the given file, and attempt to read OpalValues out of it
  StreamOpalReader (istream& is) :
    OpalReaderA(new OpalStreamReader_(is))
  { }

}; // StreamOpalReader


// ///// Globals 

// Open and read the given text file, assuming it contains a text
// OpalTable.  Read it into a Tab.  If there are any problems opening
// the file, a runtime_error will be thrown, otherwise a logic_error
// will be thrown if there are problems parsing the file. 

// Because OpalTables don't distinguish between Arrays and Tables (at
// least not when they are in text form), the only way is to notice
// that a Tab has keys 0..n-1 and guess that it is probably an Array
// and convert the Tab to an Arr.  You can turn this behaviour on and
// off (recusrively) with the convert_tab_to_arr option.  This routine
// ALWAYS return a Tab, but how it converts (recursively) the Tabs/Arrs
// underneath is what the convert_tab_to_arr controls.
inline void ReadTabFromOpalFile (const string& filename, Tab& t,
				 bool convert_tab_to_arr=false)
{
  ifstream ifs(filename.c_str());
  if (ifs.good()) {
    StreamOpalReader sv(ifs);
    sv.expectTab(t, convert_tab_to_arr); 
  } else {
    throw runtime_error("Trouble reading file:"+filename);
  }
}

// Open and read the given text file, assuming it contains a text
// OpalTable.  Read it into a Val.  If there are any problems opening
// the file, a runtime_error will be thrown, otherwise a logic_error
// will be thrown if there are problems parsing the file. 
// Expect some kind of Val on input (usually an Arr or Tab).

// Because OpalTables don't distinguish between Arrays and Tables (at
// least not when they are in text form), the only way is to notice
// that a Tab has keys 0..n-1 and guess that it is probably an Array
// and convert the Tab to an Arr.  You can turn this behaviour on and
// off (recusrively) with the convert_tab_to_arr option.  This
// defaults to TRUE because you want a Val that is either a Tab or an
// Arr (and this options applies at this level AND recursively).
inline void ReadValFromOpalFile (const string& filename, Val& v,
				 bool convert_tab_to_arr=true)
{
  ifstream ifs(filename.c_str());
  if (ifs.good()) {
    StreamOpalReader sv(ifs);
    sv.expectAnything(v, convert_tab_to_arr);
  } else {
    throw runtime_error("Trouble reading file:"+filename);
  }
}

// Open and write the given Tab as a txt OpalFile.  If there are any
// problems opening the file, a runtime_error will be thrown
inline void WriteTabToOpalFile (const Tab& t, const string& filename)
{
  ofstream ofs(filename.c_str());
  if (ofs.good()) {
    prettyPrintOpal(t, ofs);
  } else {
    throw runtime_error("Trouble writing file:"+filename);
  }
}

// Open and write the given Val as a txt OpalFile.  If there are any
// problems opening the file, a runtime_error will be thrown
inline void WriteValToOpalFile (const Val& v, const string& filename)
{
  ofstream ofs(filename.c_str());
  if (ofs.good()) {
    prettyPrintOpal(v, ofs);
  } else {
    throw runtime_error("Trouble writing file:"+filename);
  }
}

PTOOLS_END_NAMESPACE


#define OPALUTILS_H_
#endif // OPALUTILS_H_


