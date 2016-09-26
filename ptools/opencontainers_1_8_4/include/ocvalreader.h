#ifndef VALREADER_H_

// NOTE: Due to weird include dependencies, this file includes nothing
// itself.  If you wish to use it, make sure you include ocval.h
// first.

// A class to help read "human-readable" strings of Vals (Tabs, Arrs,
// numbers, Numeric arrays).  Format is straight-forward:
//
//   ValReader v(" { 'a':'nice', 'table':1 }");
//   Tab t;
//   v.expectTab(t);
//
// From Files (streams):
//   StreamValReader sv(istream& is);
//   Tab t;
//   sv.expectTab(t);
//
// The constructors of Tab and Arr have been augmented so as to allow
// easier construction:
//   
//   Tab t = "{ 'a':1, 'b':2, 'c':3, 'list':[1,2,3] }";
//   Arr a = "[ 'a', 2, 3.3, (1+2j), { 'empty':'ish' } ]";

#include "ocstringtools.h"
#include "ocreader.h"
#include "ocnumerictools.h"
#include "ocnumpytools.h"
#include <limits> // for Nan and Inf and -inf

OC_BEGIN_NAMESPACE

// Abstract base class: All the code for parsing the letters one by
// one is here.  The code for actually getting the letters (from a
// string, stream, etc.) defers to the derived class.

// Make it so we can just return quickly without a throw
#define VAL_SYNTAXERROR(MESG) { if (!throwing_) return false; else { syntaxError_(MESG); }}

#define EXPECT_CHAR(EXP) { int get=getNWSChar_(); if (get!=EXP) { if (!throwing_) return false; \
else { string get_string, expected_string; if (get==EOF) get_string="EOF"; else get_string=get;expected_string=EXP; \
syntaxError_("Expected:'"+expected_string+"', but saw '"+get_string+"' on input"); } } }


class ValReaderA { 

  // Indicate which special value we have
  enum ValReaderEnum_e { VR_NOT_SPECIAL, VR_NAN, VR_INF, VR_NEGINF }; 

 public: 

  ValReaderA (ReaderA* adopted_reader, bool throwing=true) :
    reader_(adopted_reader), throwing_(throwing) { }

  virtual ~ValReaderA () { delete reader_; }

  // Look ahead and see that that next thing coming is an EOF
  bool EOFComing () { return reader_->EOFComing(); }

  // Expect any number (including complex)
  bool expectNumber (Val& n)
  {
    consumeWS_();
    int c=peekChar_();
    if (c=='(') {
      return expectComplex_(n);
    }
    return expectSimpleNumber(n);
  }

  // Expect any "non-complex" number, i.e., any positive or negative
  // int float or real: just not any complex numbers with ()!
  bool expectSimpleNumber (Val& n)
  {
    consumeWS_();
    
    // Get the integer part, if any, of the number
    ValReaderEnum_e special = VR_NOT_SPECIAL;
    Str integer_part;
    bool result = getSignedDigits_('.', integer_part, &special);
    if (!result) return false;
    
    switch (special) { // will print as nan, -inf and inf
    case VR_INF: { 
      real_8 inf = numeric_limits<double>::infinity();
      n = inf;
      return true;
    } 
    case VR_NEGINF: { 
      real_8 ninf = -numeric_limits<double>::infinity();
      n = ninf;
      return true;
    } 
    case VR_NAN: { 
      real_8 nan = numeric_limits<double>::quiet_NaN();
      n = nan;
      return true;
    }
    case VR_NOT_SPECIAL: default: break; // really, just fall through
    }
    
    // Get the fractional part, if any
    Str fractional_part;
    int c = peekChar_();
    if (c=='.') {     
      c = getChar_(); // consume the '.'
      fractional_part = "."+getDigits_();
      if (fractional_part.length()==1) { 
	const int i_len = integer_part.length();
	if (i_len==0 || (i_len>0 && !isdigit(integer_part[i_len-1]))) {
	  VAL_SYNTAXERROR("Expecting some digits after a decimal point, but saw '"+saw_(peekChar_())+"'");
	}
      }
      c = peekChar_();
    }
    
    // Get the exponent part, if any
    Str exponent_part;
    if (c=='e' || c=='E') {
      c = getChar_();  // consume the 'e'
      if (!getSignedDigits_(' ', exponent_part)) return false;
      exponent_part = "e"+exponent_part;
      
      if (exponent_part.length()==1) { // only an e
	VAL_SYNTAXERROR("Expected '+', '-' or digits after an exponent, but saw '"+saw_(peekChar_())+"'");
      }
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
      return true;
    }
    
    // Well, no fractional part or exponential.  There had better be
    // some digits!
    if (integer_part.length()==0 || 
	(integer_part.length()>0 && 
	 !isdigit(integer_part[integer_part.length()-1])))
      VAL_SYNTAXERROR("Expected some digits for a number");
    
    c=peekChar_();
    Val v = integer_part;
    if (c=='l' || c=='L') { // Okay, it's a long
      getChar_();  // consume long
      if (integer_part[0]=='-') {
	int_n long_int = v;
	n = long_int;
	return true;
      } else {
	int_un long_int = v;
	n = long_int;
	return true;
      } 
    } else { // plain integer
      convertInt_(v); // assumes some int, with tstring inside v
      n = v;
      return true;
    }
    return true;
  }
  
  // Read a string from the current place in the input
  virtual bool expectStr (Str& s)
  { 
    consumeWS_();

    char quote_mark = peekNWSChar_();
    if (quote_mark!='\'' && quote_mark!='"') {
      VAL_SYNTAXERROR("A string needs to start with ' or \"");
    }
    
    EXPECT_CHAR(quote_mark); // Start quote

    // Read string, keeping all escapes, and let DeImage handle escapes
    Array<char> a(80);
    for (int c=getChar_(); c!=quote_mark; c=getChar_()) {
      if (c==EOF) VAL_SYNTAXERROR("Unexpected EOF inside of string");
      a.append(c);
      if (c=='\\') { // escape sequence
	int next = getChar_(); // Avoid '
	if (next==EOF) VAL_SYNTAXERROR("Unexpected EOF inside of string");
	a.append(next);
      } 
    }    
    string temp = string(a.data(), a.length());
    string ss = DeImage(temp, false); // Do escapes 
    s = Str(ss.data(), ss.length());
    return true;  
  }


  // Expect Table on the input
  bool expectTab (Tab& table) { return expectSomeTable_(table, "{"); }

  // Expect an Arr
  bool expectArr (Arr& a) { return expectSomeCommaList_(a, '[', ']'); }


  // Read in a Numeric Array
  bool expectNumericArray (Val& a)
  {
    consumeWS_();
    if (!expect_("array([")) return false; 
    a = Array<int_4>(); // By default, int array

    // Special case, empty list
    char peek = peekNWSChar_();
    if (peek!=']') {
      
      Val n;
      if (!expectNumber(n)) return false;
	
      if (n.tag=='l') { if (!readNumericArray_<int_4>(a, n)) return false; }
      else if (n.tag=='D') { if (!readNumericArray_<complex_16>(a, n)) return false; }
      else if (n.tag=='d') { if (!readNumericArray_<real_8>(a, n)) return false; }
      else VAL_SYNTAXERROR("Only support Numeric arrays of cx16, real_8, and int_4");
    }
    EXPECT_CHAR(']');
    peek=peekNWSChar_();
    if (peek==')') {
      EXPECT_CHAR(')');
      return true;
    }
    // Otherwise, expecting a type tag
    EXPECT_CHAR(',');

    // What kind of type tag?  numpy uses 'dtype=', Numeric uses string
    int next = peekNWSChar_();
    if (next==EOF) {
      VAL_SYNTAXERROR("Saw EOF after seeing '('");
    } else if (next=='d') {   // NumPy
      if (!expect_("dtype=")) return false;
      // Got dtype ... want everything upto ')'
      string typestring;
      while (1) {
	int ch = getChar_();
	if (ch==EOF) {
	  VAL_SYNTAXERROR("Saw EOF before seeing ')'");	
	} else if (isalnum(ch)) {
	  typestring += char(ch);
	} else {
	  pushback_(ch);
	  break;
	}
      }
      EXPECT_CHAR(')');
      // Check and make sure is a valid type
      int oc_tag = NumPyStringToOC(typestring.c_str());
      if (oc_tag==-1) {
	VAL_SYNTAXERROR("Unknown typecode in array");
      } else {
	ConvertArray(a, oc_tag);
      }
      return true;
      
    } else {   // Numeric
      Str typetag;
      if (!expectStr(typetag)) return false;
      EXPECT_CHAR(')');

      // Now convert if it makes sense.
      if (typetag.length()!=1) VAL_SYNTAXERROR("Expected single char type tag");
      // Convert the array to the given type
      char numeric_tag = typetag[0];
      char oc_tag = NumericTagToOC(numeric_tag);
      ConvertArray(a, oc_tag);
      return true;
    }
    return true;
  }


  // Expect one of two syntaxes:
  // o{ 'a':1, 'b':2 }
  //    or
  // OrderedDict([('a',1),('b',2)])
  bool expectOTab (OTab& o) 
  {
    char c = peekNWSChar_();
    // Short syntax
    if (c=='o') { // o{ } syntax
      return expectSomeTable_(o, "o{");
    } 
    // Long syntax
    if (!expect_("OrderedDict(")) return false;
    Arr kvpairs;
    // We let the Arr and Tup parse themselves correctly, but since we
    // are using swapKeyValueInto, this just plops them into the table
    // pretty quickly, without excessive extra deep copying.
    if (!expectArr(kvpairs)) return false;
    const int len = int(kvpairs.length());
    for (int ii=0; ii<len; ii++) {
      Tup& t = kvpairs[ii];
      o.swapInto(t[0], t[1]);
    } 
    EXPECT_CHAR(')');
    return true;
  }

  // Know a Tup is coming: starts with '(' and ends with ')'
  bool expectTup (Tup& u) { return expectSomeCommaList_(u.impl(), '(', ')'); } 

  // In general, a '(' starts either a complex number such as (0+1j)
  // or a tuple such as (1,2.2,'three'): This parses and places the
  // appropriate thing in the return value.
  bool expectTupOrComplex (Val& v)
  {
    EXPECT_CHAR('(');
    
    // Handle empty tuple
    int peek = peekNWSChar_();
    if (peek == EOF) VAL_SYNTAXERROR("Saw EOF after seeing '('");
    if (peek==')') {
      v = Tup();
      EXPECT_CHAR(')');
      return true;
    }

    // Anything other than empty, has a first component
    Val first;
    if (!expectAnything(first)) return false;
    
    // Well, plain jane number: if we see a + or -, it must be a complex
    peek = peekNWSChar_();
    if (OC_IS_NUMERIC(first) && !OC_IS_CX(first) && 
	(peek=='+' || peek=='-')) {
      // make sure we keep the "-" or "+" so we get the right sign!
      Val second;
      if (!expectSimpleNumber(second)) return false;
      if (!expect_("j)")) return false;
      if (first.tag=='f' || second.tag=='f') {
	v = complex_8(first, second);
      } else {
	v = complex_16(first, second);
      }
      return true;
    }

    // If we don't see a plain number, it has to be a Tuple
    Tup& u = v = Tup(None); // just put something is first pos
    first.swap(u[0]);
    Array<Val>& a = u.impl();
    return continueParsingCommaList_(a, '(', ')');
  }

  

  // We could be starting ANYTHING ... we have to dispatch to the
  // proper thing
  virtual bool expectAnything (Val& v)
  {
    char c = peekNWSChar_();
    switch (c) {
    case '{' : { v = Tab(); Tab& table = v; return expectTab(table); }
    case '[' : { v = Arr(); Arr& arr = v;   return expectArr(arr);   }
    case '(' : {                            return expectTupOrComplex(v); }
    case '\'':
    case '"' : { Str s; if (!expectStr(s)) return false; v=s; return true; }
    case 'a' : {                            return expectNumericArray(v); }
    case 'N' : { v = Val(); return expect_("None");                        }
    case 'T' : { v = true;  return expect_("True");                        }
    case 'F' : { v = false; return expect_("False");                       }
    case 'o' : // o{ and OrderedDict start OTab
    case 'O' : { v = OTab(); OTab& ot = v;  return expectOTab(ot);         } 
    default:   { v = Val();                 return expectNumber(v);        }
    }
    
  }


 protected:


  // Helper Methods for syntax error: multiple evals would
  // construct way too many strings when all we pass is ptrs
  void syntaxError_ (const char* s) 
  {
    reader_->syntaxError(s);
  }

  void syntaxError_ (const string& s) 
  {
    reader_->syntaxError(s);
  }


  string saw_ (int cc)
  {
    if (cc==EOF) return "EOF";
    char s[2] = { 0 }; 
    s[0] = cc;
    return string(s);
  }


  /*  bool expect_ (char expected)
  {
    // Single character token
    int get      = getNWSChar_();
    if (get!=expected) {
      if (!throwing_) return false;


    }
    }*/

  bool expect_ (const string& s)
  {
    int s_len = s.length();
    const char* data = s.data();
    for (int ii=0; ii<s_len; ii++) 
      EXPECT_CHAR(data[ii])
    return true;
  }

  bool expect_ (const char* s)
  {
    while (*s) {
      EXPECT_CHAR(*s);
      s++;
    }
    return true;
  }

  // Expect a complex number:  assumes it will have form (#+#j)
  bool expectComplex_ (Val& n)
  {
    complex_16 result;
    EXPECT_CHAR('(');
    Val real_part, imag_part;
    if (!expectNumber(real_part)) return false;
    char c=peekChar_();
    if (c=='+' || c=='-') {
      if (!expectNumber(imag_part)) return false;
    }
    if (!expect_("j)")) return false;

    result.re = real_part;
    result.im = imag_part;
    n = result;
    return true;
  }
  
  // From current point of input, consume all digits until
  // next non-digit.  Leaves non-digit on input
  Str getDigits_ ()
  {
    Str digits;
    while (1) {
      int c = peekChar_();
      if (c==EOF) 
	break;
      if (isdigit(c)) {
	digits+=char(c);
	getChar_();
      }
      else 
	break;
    }
    return digits;
  }

  // From current point in input, get the signed sequence of 
  // digits
  bool getSignedDigits_ (char next_marker, Str& digits,
			 ValReaderEnum_e* inf_nan_okay=0)
  {
    // Get the sign of the number, if any
    char c=peekChar_();
    char the_sign='\0'; // Sentry

    // Add ability to recognize nan, inf, and -inf
    if (inf_nan_okay) {
      if (c=='n') { // expecting nan
	if (expect_("nan")) {
	  *inf_nan_okay = VR_NAN;
	  digits = "nan";
	  return true;
	} else {
	  return false;
	}
      } else if (c=='i') {
	if (expect_("inf")) {
	  *inf_nan_okay = VR_INF;
	  digits = "inf";
	  return true;
	} else {
	  return false;
	}
      }  
    }
    
    char saw_sign = ' ';
    if (c=='+'||c=='-') {
      saw_sign = c;
      the_sign = c;
      getChar_();    // consume the sign
      c=peekChar_(); // .. and see what's next
    }
    
    if (inf_nan_okay) {    
      if (saw_sign!=' ' && c=='i') { // must be -inf
	if (!expect_("inf")) return false;
	if (saw_sign=='-') {
	  *inf_nan_okay = VR_NEGINF;
	  digits = "-inf";
	  return true;
	} else { // if (saw_sign=='+') {
	  *inf_nan_okay = VR_INF;
	  digits = "+inf";
	  return true;
	}
      }
    }
    
    // Assertion: passed sign, now should see number or .
    if (!isdigit(c)&&c!=next_marker) {
      if (!throwing_) return false;
      syntaxError_("Expected numeric digit or '"+saw_(next_marker)+"' for number, but saw '"+saw_(c)+"'");
    }
    
    if (the_sign) {
      string g; g = the_sign;
      digits = g+getDigits_();
    }
    else digits = getDigits_();
    //Str digits = sign+getDigits_();
    //return digits;
    return true;
  }
  
  template <class T>
  bool readNumericArray_ (Val& arr, const Val& first_value)
  { 
    arr = Array<T>();  // initialize the array
    Array<T>& a = arr; // .. then a reference to it
    T v = first_value;
    a.append(v);
    
    // Continue getting each item one at a time
    while (1) {
      char peek = peekNWSChar_();
      if (peek==',') {
	EXPECT_CHAR(',');
	continue;
      } else if (peek==']') {
	break;
      } 

      Val n; 
      if (!expectNumber(n)) return false;
      T v = n;
      a.append(v);
    }
    return true;
  }

  // the string of +- and digits: convert to some int
  void convertInt_ (Val& v) 
  {
    // Strip into sign and digit part 
    char sign_part = '+';
    string digit_part = v;
    if (!isdigit(digit_part[0])) {
      sign_part = digit_part[0];
      digit_part = digit_part.substr(1);
    }
    const int len = digit_part.length();

    // Try to turn into int_4 in general, otherwise slowly move from
    // int_8 to real_8
    if (sign_part=='-') { // negative number
      static const string smallest_int_4 = "2147483648";
      static const string smallest_int_8 = "9223372036854775808";
      if (len<int(smallest_int_4.length()) ||
	  (len==int(smallest_int_4.length()) && digit_part<=smallest_int_4)) { 
	int_4 i4_val = v;
	v = i4_val;
      } else if (len<int(smallest_int_8.length()) ||
		(len==int(smallest_int_8.length()) && digit_part<=smallest_int_8)) {
	int_8 i8_val = v;
	v = i8_val;
      } else {
	int_n r8_val = v;
	v = r8_val;
      }

    } else { // positive number
      static const string biggest_int_4 = "2147483647";
      static const string biggest_int_8 = "9223372036854775807";
      static const string biggest_int_u8 = "18446744073709551615";
      if (len<int(biggest_int_4.length()) || 
	  (len==int(biggest_int_4.length()) && digit_part<=biggest_int_4)) {
	int_4 i4_val = v;
	v = i4_val;
      } else if (len<int(biggest_int_8.length()) ||
		 (len==int(biggest_int_8.length()) && digit_part<=biggest_int_8)) {
	int_8 i8_val = v;
	v = i8_val;
      } else if (len<int(biggest_int_u8.length()) ||
		(len==int(biggest_int_u8.length()) && digit_part<=biggest_int_u8)) {
	int_u8 iu8_val = v;
	v = iu8_val;
      } else {
	int_un r8_val = v;
	v = r8_val;
      }
    }
    
  }

  // Expect Table on the input
  template <class TABLE>
  bool expectSomeTable_ (TABLE& table, const char* start_table)
  {
    if (!expect_(start_table)) return false;
    // Special case, empty Table
    char peek = peekNWSChar_();
    if (peek!='}') {
      
      for (;;) { // Continue getting key value pairs
	{ 
	  Val key;   if (!expectAnything(key)) return false;
	  EXPECT_CHAR(':');
	  Val value; if (!expectAnything(value)) return false;
	  
	  table.swapInto(key, value); // Assumes we are done with key-value
	}
	
	int peek = peekNWSChar_();
	if (peek==',') {
	  EXPECT_CHAR(',');         // Another k-v pair, grab comma and move on
	  peek = peekNWSChar_();
          if (peek=='}') break;  // CAN have extra , at the end of the table
	  continue;
	} else if (peek=='}') { // End of table
	  break;
	} else {
	  if (!throwing_) return false;
	  syntaxError_("Expecting a '}' or ',' for table, but saw '"+saw_(peek)+"'");
	}
      }
    }
    EXPECT_CHAR('}');
    return true;
  }

  template <class OLIST> 
  bool continueParsingCommaList_ (OLIST& a, char start, char end)
  {
    // Continue getting each item one at a time
    while (1) {         
      int peek = peekNWSChar_();
      if (peek==',') {
	EXPECT_CHAR(',');
	peek = peekNWSChar_();
	if (peek==end) break;  // CAN have extra , at the end of the table
	
	a.append(Val());
	if (!expectAnything(a[a.length()-1])) return false;
	
	continue;
      } else if (peek==end) {
	break;
      } else {
	if (!throwing_) return false;
	char c[4] = { start, end, '\0' };
	syntaxError_("Problems parsing around '"+saw_(peek)+" for "+string(c));
      }
    }
    EXPECT_CHAR(end);
    return true;
  }

  template <class OLIST> 
  bool expectSomeCommaList_ (OLIST& a, char start, char end)
  {
    EXPECT_CHAR(start);
    
    // Special case, empty list
    char peek = peekNWSChar_();
    if (peek==end) {
      EXPECT_CHAR(end);
    } else {
      // Continue getting each item one at a time, after gotten first
      a.append(Val());
      if (!expectAnything(a[a.length()-1])) return false; 
      if (!continueParsingCommaList_(a, start, end)) return false;
    }
    return true;
  }
  

  // Dispatch for input
  int getNWSChar_ ()  { return reader_->getNWSChar_(); }
  int peekNWSChar_ () { return reader_->peekNWSChar_(); }
  int getChar_ ()     { return reader_->getChar_(); }
  int peekChar_ ()    { return reader_->peekChar_(); }
  int consumeWS_ ()   { return reader_->consumeWS_(); }
  void pushback_ (int pushback_char) { reader_->pushback_(pushback_char); }

  // Defer IO to another class.  All sorts of discussion on why
  // didn't we inherit, etc.  Look at the Design Patterns book.
  ReaderA* reader_; 
  bool throwing_; 

}; // ValReaderA


// The ValReader reads Vals from strings.  The most common usage is to
// use this to read literal constants for the constructors of Tab and Arr.
// Options: 
// make_copy: Allows a ValReader to share the original input 
// supports_context: allows ValReader to turn on/off context in an error mesg
class ValReader : public ValReaderA {

 public:

  ValReader (Array<char>& a, 
	     bool make_copy=false, bool supports_context=true,
	     bool throwing_exceptions=true) :
    ValReaderA(new StringReader(a, make_copy, supports_context), 
	       throwing_exceptions)
  { }

  ValReader (const char* s, int len=-1, 
	     bool make_copy=false, bool supports_context=true,
	     bool throwing_exceptions=true) :
    ValReaderA(new StringReader(s, len, make_copy, supports_context), 
	       throwing_exceptions)
  { }

  ValReader (const string& s, 
	     bool make_copy=false, bool supports_context=true, 
	     bool throwing_exceptions=true) :
    ValReaderA(new StringReader(s, make_copy, supports_context),
	       throwing_exceptions)
  { }

    
 protected:

}; // ValReader


// Read a val from a stream
class StreamValReader : public ValReaderA {

 public:

  // Open the given file, and attempt to read Vals out of it
  StreamValReader (istream& is) :
    ValReaderA(new StreamReader(is))
  { }

}; // StreamValReader


// Read the given Tab from a TEXT file: if there are any problems,
// throw a runtime_error indicating we had trouble reading the file,
// or a logic_error if the input is malformed.
inline void ReadTabFromFile (const string& filename, Tab& t)
{
  ifstream ifs(filename.c_str());
  if (ifs.good()) {
    StreamValReader sv(ifs);
    sv.expectTab(t);
  } else {
    throw runtime_error("Trouble reading file:"+filename);
  }
}


// Write the given Tab to a TEXT file: if there are problems, throw a
// runtime_error indicating we had trouble writing out the file.
inline void WriteTabToFile (const Tab& t, const string& filename)
{
  ofstream ofs(filename.c_str());
  if (ofs.good()) {
    t.prettyPrint(ofs);
  } else {
    throw runtime_error("Trouble writing the file:"+filename);
  }
}

// Read the given Val from a TEXT file: if there are any problems,
// throw a runtime_error indicating we had trouble reading the file,
// or a logic_error if the input is malformed.
inline void ReadValFromFile (const string& filename, Val& v)
{
  ifstream ifs(filename.c_str());
  if (ifs.good()) {
    StreamValReader sv(ifs);
    sv.expectAnything(v);
  } else {
    throw runtime_error("Trouble reading file:"+filename);
  }
}

// Read the given Val from a TEXT file: if there are any problems,
// throw a runtime_error indicating we had trouble reading the file,
// or a logic_error if the input is malformed.
inline void ReadValFromStream (istream& is, Val& v)
{
  if (is.good()) {
    StreamValReader sv(is);
    sv.expectAnything(v);
  } else {
    throw runtime_error("Trouble reading from stream");
  }
}

inline void ReadValFromFile (const char* filename, Val& v)
{ ReadValFromFile(string(filename), v); }


// Write the given Val to a TEXT file: if there are problems, throw a
// runtime_error indicating we had trouble writing out the file.

inline void WriteValToFile (const Val& v, const string& filename)
{
  ofstream ofs(filename.c_str());
  if (ofs.good()) {
    v.prettyPrint(ofs);
  } else {
    throw runtime_error("Trouble writing the file:"+filename);
  }
}

inline void WriteValToFile (const Val& v, const char* filename)
{ WriteValToFile(v, string(filename)); }


// Write the given Val to a TEXT file: if there are problems, throw a
// runtime_error indicating we had trouble writing out the file.
inline void WriteValToStream (const Val& v, ostream& os)
{
  if (os.good()) {
    v.prettyPrint(os);
  } else {
    throw runtime_error("Trouble writing to stream");
  }
}


// Evaluate the given string (Val literal) and return the Val underneath
inline Val Eval (const string& code)
{
  Val v;
  ValReader c(code.data(), code.length());
  c.expectAnything(v);
  return v;
}

inline Val Eval (const char* code, int len=-1)
{
  Val v;
  ValReader c(code, len);
  c.expectAnything(v);
  return v;
}


OC_END_NAMESPACE


#define VALREADER_H_
#endif // VALREADER_H_


