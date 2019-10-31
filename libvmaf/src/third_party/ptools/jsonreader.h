#ifndef JSONREADER_H_
#define JSONREADER_H_ 

// The JSON Reader is very similar to the ValReader: minor differences
// in Python dictionaries and JSON. 
// Python: JSON:
//   True true
//   False false
//   None  null
//  ' or "  ""
// escape sequences in strings (esp. \u one)

// To avoid losing information, we understand the following two conventions:
// { 're': 1.0, 'im': 1.2 } 
//        is the complex_16(1.0, 1.2)
// { 'array': [1,2,3...,n], 'type':'typecode' } 
//        is a POD array: array<POD>(1,2,3,...,n)

#include "ocval.h"
#include "ocvalreader.h"

OC_BEGIN_NAMESPACE

// /////////////////////////////////////////// JSONReaderA

class JSONReaderA : public ValReaderA {

 public:

  JSONReaderA (ReaderA* adopted_reader, bool throwing=true) :
    ValReaderA(adopted_reader, throwing)
  { }

  virtual ~JSONReaderA () { }


  // JSON has ever so slightly differences from Python Dictionaries, 
  // so the expectAnything is slightly different
  virtual bool expectAnything (Val& v) 
  {
    char c = peekNWSChar_();
    switch (c) {
    case '{' : 
      { 
	v = Tab(); 
	Tab& table = v; 
	if (!expectTab(table)) { 
	  return false; 
	}
	checkSpecial_(v);  // A Table may have a special meaning
	return true;
      }
    case '[' : { v = Arr(); Arr& arr = v;   return expectArr(arr);   }
    case '"' : { Str s; if (!expectStr(s))  return false; v=s; return true; }
    case 'n' : { v = Val(); return expect_("null");                        }
    case 't' : { v = true;  return expect_("true");                        }
    case 'f' : { v = false; return expect_("false");                       }
    default:   { v = Val();                 return expectNumber(v);        }
    }
  }


  // JSON has different escape sequences for strings and only supports "
  virtual bool expectStr (Str& s) 
  {
    consumeWS_();
    
    char quote_mark = peekNWSChar_();
    if (quote_mark!='"') {
      VAL_SYNTAXERROR("A string needs to start \"");
    }

    EXPECT_CHAR(quote_mark); // Start quote  

    // Read string, keeping all escapes, and let DeImage handle escapes 
    Array<char> a(80);
    for (int c=getChar_(); c!=quote_mark; c=getChar_()) {
      if (c==EOF) VAL_SYNTAXERROR("Unexpected EOF inside of string");
      if (c != '\\') {
	a.append(c);
      } else {  // escape sequence
        int next = getChar_(); // Avoid '
        if (next==EOF) VAL_SYNTAXERROR("Unexpected EOF inside of string");
	if (next=='u') {  
	  // Next 4 characters are hexdigits
	  static const char hexdigits[] = "0123456789abcdef";
	  int store[5] = { 0 };
	  for (int ii=0; ii<4; ii++) {
	    int toget = getChar_();
	    if (toget==EOF) VAL_SYNTAXERROR("Unexpected EOF inside of string");
	    toget = tolower(toget);
	    const char* where = strchr(hexdigits, toget);
	    if (where==NULL) {
	      VAL_SYNTAXERROR("Expected hex characters after \\u");
	    }
	    int diff = where - hexdigits; // 0-15
	    store[ii] = diff;
	  }
	  // Assertion: Have 4 characters 0-15: convert to 2 chars
	  a.append(store[0]*16 + store[1]);
	  a.append(store[2]*16 + store[3]);
	} else {
	  a.append(handleJSONEscapes_(next));
	}
      }
    }
    s = Str(a.data(), a.length());
    return true;
  }

 protected:
  
  // Handle the JSON escape characters
  static char handleJSONEscapes_ (char escaped)
  {
    switch (escaped) {
    case '"' : return '"';
    case '\\': return '\\';
    case '/' : return '/';
    case 'b' : return '\b';
    case 'f' : return '\f';
    case 'n' : return '\n';
    case 'r' : return '\r';
    case 't' : return '\t';
    case 'u': 
    default : return escaped;
    }
  }

  // Conventions for turning JSON into other datatypes that are useful:
  // Turn { 're':xx, 'im' } into complex_16
  //      { 'array':[1,2,3..], 'typecode':'xxx' } into Array<POD>(1,2,3)
#define CHECKSPECIAL_(T) { Val temp = Array<T>(len); Array<T>&aa=temp; T e; for (int i=0;i<len;i++) { e=a[i]; aa.append(e); } temp.swap(in); break;} 
#define CHECKSPECIALCX_(T) { Val temp = Array<T>(len/2); Array<T>&aa=temp; T e; for (int i=0;i<len;i+=2) { e.re=a[i]; e.im=a[i+1]; aa.append(e); } temp.swap(in); break; } 
  void checkSpecial_ (Val& in)
  {
    Tab& t = in;
    // Special have 2 entries
    if (t.entries()==2) {
      // complex
      if (t.contains("re") && t.contains("im")) {
	real_8 re = t("re");
	real_8 im = t("im");
	in = complex_16(re, im);
	return;
      }
      // POD arrays
      else if (t.contains("array") && t.contains("type")) {
	Val& va = t("array");
	if (va.tag=='n' && va.subtype=='Z') {
	  Arr& a = va;
	  const int len = a.entries();
	  string s = t("type");
	  if (s.length()==1) {
	    switch (s[0]) {
	    case 's': CHECKSPECIAL_(int_1);
	    case 'S': CHECKSPECIAL_(int_u1);
	    case 'i': CHECKSPECIAL_(int_2);
	    case 'I': CHECKSPECIAL_(int_u2);
	    case 'l': CHECKSPECIAL_(int_4);
	    case 'L': CHECKSPECIAL_(int_u4);
	    case 'x': CHECKSPECIAL_(int_8);
	    case 'X': CHECKSPECIAL_(int_u8);
	    case 'f': CHECKSPECIAL_(real_4);
	    case 'd': CHECKSPECIAL_(real_8);
	    case 'F': CHECKSPECIALCX_(complex_8);
	    case 'D': CHECKSPECIALCX_(complex_16);
	    default: throw runtime_error("Unknown convention for converting data");
	    }
	  }
	}	
      }
    }
  }
  
}; // JSONReaderA





// The JSONReader reads Vals from strings.  
// Options: 
// make_copy: Allows a ValReader to share the original input 
// supports_context: allows ValReader to turn on/off context in an error mesg
class JSONReader : public JSONReaderA {

 public:

  JSONReader (Array<char>& a, 
	     bool make_copy=false, bool supports_context=true,
	     bool throwing_exceptions=true) :
    JSONReaderA(new StringReader(a, make_copy, supports_context), 
		throwing_exceptions)
  { }

  JSONReader (const char* s, int len=-1, 
	     bool make_copy=false, bool supports_context=true,
	     bool throwing_exceptions=true) :
    JSONReaderA(new StringReader(s, len, make_copy, supports_context), 
	       throwing_exceptions)
  { }

  JSONReader (const string& s, 
	     bool make_copy=false, bool supports_context=true, 
	     bool throwing_exceptions=true) :
    JSONReaderA(new StringReader(s, make_copy, supports_context),
	       throwing_exceptions)
  { }

    
 protected:

}; // JSONReader


// Read a val from a JSON stream
class StreamJSONReader : public JSONReaderA {

 public:

  // Open the given file, and attempt to read Vals out of it
  StreamJSONReader (istream& is) :
    JSONReaderA(new StreamReader(is))
  { }

}; // StreamJSONReader



// Read the given Val from a TEXT file: if there are any problems,
// throw a runtime_error indicating we had trouble reading the file,
// or a logic_error if the input is malformed.
inline void ReadValFromJSONFile (const string& filename, Val& v)
{
  ifstream ifs(filename.c_str());
  if (ifs.good()) {
    StreamJSONReader sv(ifs);
    sv.expectAnything(v);
  } else {
    throw runtime_error("Trouble reading file:"+filename);
  }
}

// Read the given Val from a TEXT file: if there are any problems,
// throw a runtime_error indicating we had trouble reading the file,
// or a logic_error if the input is malformed.
inline void ReadValFromJSONStream (istream& is, Val& v)
{
  if (is.good()) {
    StreamJSONReader sv(is);
    sv.expectAnything(v);
  } else {
    throw runtime_error("Trouble reading from stream");
  }
}

inline void ReadValFromJSONFile (const char* filename, Val& v)
{ ReadValFromJSONFile(string(filename), v); }


// Write the given Val to a TEXT JSON file: if there are problems, throw a
// runtime_error indicating we had trouble writing out the file.

inline void WriteValToJSONFile (const Val& v, const string& filename)
{
  ofstream ofs(filename.c_str());
  if (ofs.good()) {
    JSONPrint(v, ofs, 0, true);
  } else {
    throw runtime_error("Trouble writing the file:"+filename);
  }
}

inline void WriteValToJSONFile (const Val& v, const char* filename)
{ WriteValToJSONFile(v, string(filename)); }


// Write the given Val to a TEXT file: if there are problems, throw a
// runtime_error indicating we had trouble writing out the file.
inline void WriteValToJSONStream (const Val& v, ostream& os)
{
  if (os.good()) {
    JSONPrint(v, os, 0, true);
  } else {
    throw runtime_error("Trouble writing to stream");
  }
}


// Evaluate the given string (Val literal) and return the Val underneath
inline Val EvalJSON (const string& code)
{
  Val v;
  JSONReader c(code.data(), code.length());
  c.expectAnything(v);
  return v;
}

inline Val EvalJSON (const char* code, int len=-1)
{
  Val v;
  JSONReader c(code, len);
  c.expectAnything(v);
  return v;
}




OC_END_NAMESPACE

#endif // JSONREADER_H_
