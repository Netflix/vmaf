#ifndef OPALPRINT_H_

#include "ocval.h"
#include "ocstringtools.h"
#include <ctype.h>


PTOOLS_BEGIN_NAMESPACE

// Forward references for helper functions (because of the
// mutually recursive nature of the functions)

inline ostream& prettyPrintOpalHelper_ (const Tab& t, 
					ostream& os, int indent, bool pretty,
					int indent_additive, bool as_m2k_array);
inline ostream& prettyPrintOpalHelper_ (const Arr& t, 
					ostream& os, int indent, bool pretty,
					int indent_additive, bool as_m2k_array);
inline ostream& prettyPrintVector_ (const Val& v, ostream& os);


 
// Steal image and unimage from m2k 
inline string M2Image (const string& s)
{
  // The final result
  string result = "";
  string ender = "";
 
  // Start off with a quote
  result += '"';
   
  // Go through, looking for special sequences and output
  for (size_t i=0; i<s.length(); i++) {
 
    // cerr << "Char is:" << s[i] << ":" << int(s[i]) << endl;
    // See if a special escape character
    switch (s[i]) {
    case '\n': ender = "\\n";   break;
    case '\t': ender = "\\t";   break;
    case '\v': ender = "\\v";   break;
    case '\b': ender = "\\b";   break;
    case '\r': ender = "\\r";   break;
    case '\f': ender = "\\f";   break;
    case '\a': ender = "\\a";   break;
    case '\\': ender = "\\\\";  break;
// Taken out:  This was originally in here for compatibility with
// C++ escpaes and trigraphs:  No one cares about trigraphs or compatibility
// so we nuke this special case.
//    case '\?': ender = "\\?";   break;
    case '\'': ender = "\\'";   break;
    case '"':  ender = "\\\"";  break;
    case '\0':  ender = "\\x00";   break;
    default:
       
      // See if an outputtable character
      if (isprint(s[i]))
        ender = s[i];
      else {
        // Build a hexidecimal escape sequence. The output is always
        // going to be \x followed by 2 hex digits.
        ender = "\\x  ";
 
        const string hex = "0123456789ABCDEF";
 
        // Build up the hex constant (reverse)
        int_u1 val = int_u1(s[i]);
        ender[2] = hex[val/16];
        ender[3] = hex[val%16];
      }
    } // switch
    
    // Add that character to the end
    result += ender;
     
  }
   
  // End with another quote
  result += '"';
   
  return result;
}
 
 
 
 
inline string M2Unimage (const string& s)
{
  // The final result
  string result;
 
  // Go through, looking for special sequences and output
  // start at 1 to skip 1 "; stop early to skip last "
  int start = 0;
  int stop = s.length();
  if (s.length() != 0 && s[0] == '"' && s[s.length()-1] == '"') {
    start = 1;
    stop = s.length()-1;
  }
 
  for (int i = start; i < stop; i++) {
 
    // See if a special escape character
    if (s[i] == '\\') {
      i++;
      switch (s[i]) {
      case 'n':
        result += '\n';
        break;
      case 't':
        result += '\t';
        break;
      case 'v':
        result += '\v';
        break;
      case 'b':
        result += '\b';
        break;
      case 'r':
        result += '\r';
        break;
      case 'f':
        result += '\f';
        break;
      case 'a':
        result += '\a';
        break;
      case '\\':
        result += '\\';
        break;
      case '?':
        result += '\?';
        break;
      case '\'':
        result += '\\';
        break;
      case '"':
        result += '\"';
        break;
      case '0':
        result += '\0';
        break;
      case 'x': {
        char t;
        t = (s[i] - '0') * 16;
        i++;
        t = t + s[i] - '0';
        result += t;
        break;
      }
      default:
        result += s[i];
        break;
      } // switch
    } else {
      result = result + s[i];
    }
  }
 
  return result;
}
 


inline string keyoutput_ (const string& s)
{
  int slen=s.length();
  if (slen==0 || !(isalpha(s[0])||(s[0]=='_')))
    return M2Image(s);
  
  // When outputtting a key, make sure it can be read back in
  bool contains_bad_chars = false;
  for (int i=0; i<slen; i++)
    if (!(isalnum(s[i])||(s[i]=='_'))) {
      contains_bad_chars = true;
      break;
    }
  
  if (contains_bad_chars)
    return M2Image(s);
  else
    return s;
}


inline string EncodeOpalTypeTag (char val_tag) 
{
  switch(val_tag) {
  case 's': return "B"; break;
  case 'S': return "UB";  break;
  case 'i': return "I"; break;
  case 'I': return "UI"; break;
  case 'l': return "L"; break;
  case 'L': return "UL"; break;
  case 'x': return "X"; break;
  case 'X': return "XL"; break;
  case 'f': return "F"; break;
  case 'd': return "D"; break;
  case 'F': return "CF"; break;
  case 'D': return "CD"; break;
  case 'b': return "BIT"; break;
  default: return "???";
  }
}

// Assumes tag is a numeric type, or simple type
inline void simplePrettyPrintOpal_ (const Val& value, ostream& os,
				    bool with_tags = true)
{
  // Do we output the tag?
  if (with_tags && value.tag != 'Z') {
    os << EncodeOpalTypeTag(value.tag) << ":";
  }

  switch (value.tag) {
  case 'Z': { os << "\"None\""; break; /* Best thing for None? */ }
  case 'b': { 
    bool* bp = (bool*)&value.u.b; 
    int_4 i_bool = *bp;
    os << i_bool; break;
  }
  case 'F': {
    complex_8 r = value;
    os << "(" << Val(r.re) << "," << Val(r.im) << ")";
    break;
  } 
  case 'D': {
    complex_16 r = value;
    os << "(" << Val(r.re) << "," << Val(r.im) << ")";
    break;
  } 
  default: { // Everything else Numeric
    os << value; break;
  }
  }
}

inline ostream& prettyPrintOpalHelper_ (const Arr& a, 
					ostream& os, int indent, bool pretty,
					int indent_additive, bool as_m2k_array)
{
  // Base case, empty table
  if (a.entries()==0) {
    if (pretty) {
      return os << "{ }";
    } else {
      return os << "{}";
    }
  }
 
  // Recursive case
  os << "{";
  if (pretty) os << endl;
 
  // Iterate through, printing out each element
  for (size_t ii=0; ii<a.entries(); ii++) {
    const int_4 key  = ii;
    const Val& value = a[ii];
     
    if (pretty) indentOut_(os, indent+indent_additive);
    if (!as_m2k_array) os << "\"" << key << "\"" << "=";
     
    // For most values, use default output method
    switch (value.tag) {
    case 'a': {
      OCString* ap = (OCString*)&value.u.a;
      os << M2Image(string(ap->data(), ap->length()));
      break;
    }
    case 't': {
      const Tab& tr = value;
      prettyPrintOpalHelper_(tr, os, pretty ? indent+indent_additive : 0, 
			     pretty, indent_additive, as_m2k_array);
      break;
    }
    case 'n': {
      if (value.subtype=='Z') {
        const Arr& ar = value;
        prettyPrintOpalHelper_(ar, os, pretty ? indent+indent_additive : 0, 
			       pretty, indent_additive, as_m2k_array);
        break;
      } else {
	prettyPrintVector_(value, os);
	break;
      }
    }
    default : simplePrettyPrintOpal_(value, os); break;
    }
    
    if (a.entries()>1 && ii!=a.entries()-1) os << ","; // commas on all butlast
    if (pretty) os << endl;
  }
  
  if (pretty) indentOut_(os, indent);
  return os << "}";
}
 
inline void prettyPrintOpal (const Arr& a, ostream& os, 
			     int indent = 0, int starting_indent=4, 
			     bool as_m2k_array=false) 
{
  indentOut_(os, indent);
  prettyPrintOpalHelper_(a, os, indent, true, starting_indent, as_m2k_array) << endl;
}
  

template <class T>
inline void prettyPrintVectorHelper_ (char tag, Array<T>& a, ostream& os)
{
  os << EncodeOpalTypeTag(tag) << ":<";
  int alen = a.length();
  for (int ii=0; ii<alen-1; ii++) {
    simplePrettyPrintOpal_(a[ii], os, false);
    os << ", ";
  }
  if (alen>0) simplePrettyPrintOpal_(a[alen-1], os, false);
  os << ">";
}

#define PPHELP1(A) \
{ Array<A>& ar=v; prettyPrintVectorHelper_(v.subtype, ar, os); break; }

inline ostream& prettyPrintVector_ (const Val& v, ostream& os)
{
  if (v.tag != 'n') {
    throw runtime_error("Tag for prettyPrintArray needs to be 'n'");
  }
  switch(v.subtype) {
  case 's': PPHELP1(int_1);
  case 'S': PPHELP1(int_u1);
  case 'i': PPHELP1(int_2);
  case 'I': PPHELP1(int_u2);
  case 'l': PPHELP1(int_4);
  case 'L': PPHELP1(int_u4);
  case 'x': PPHELP1(int_8);
  case 'X': PPHELP1(int_u8);
  case 'f': PPHELP1(real_4);
  case 'd': PPHELP1(real_8);
  case 'F': PPHELP1(complex_8);
  case 'D': PPHELP1(complex_16);
  case 'b': PPHELP1(bool);
  default: throw runtime_error("Unknown type of vector");
  }
  return os;
}



inline ostream& prettyPrintOpalHelper_ (const Tab& t, 
					ostream& os, int indent, bool pretty,
					int indent_additive, bool as_m2k_array)
{
  // Base case, empty table
  if (t.entries()==0) {
    if (pretty) {
      return os << "{ }";
    } else {
      return os << "{}";
    }
  }

  // Recursive case
  os << "{";
  if (pretty) os << endl;
 
  // Iterate through, printing out each element
  Sit sii(t);
  for (int ii=0; sii(); ii++) {
    const Val& key = sii.key();
    const Val& value = sii.value();
     
    if (pretty) indentOut_(os, indent+indent_additive);
    os << keyoutput_(key) << "=";
     
    // For most values, use default output method
    switch (value.tag) {
    case 'a': {
      OCString* ap = (OCString*)&value.u.a;
      os << M2Image(string(ap->data(), ap->length()));
      break;
    }
    case 't': {
      const Tab& t = value;
      prettyPrintOpalHelper_(t, os, pretty ? indent+indent_additive : 0, 
			     pretty, indent_additive, as_m2k_array);
      break;
    }
    case 'n': {
      if (value.subtype=='Z') {  // Like a Python List
        Arr& ar = value;
        prettyPrintOpalHelper_(ar, os, pretty ? indent+indent_additive : 0, 
			       pretty, indent_additive, as_m2k_array);
        break;
      } else {                  // A Vector in M2k parlance
	prettyPrintVector_(value, os);
	break;
      }
    }
    default : simplePrettyPrintOpal_(value, os); break;
    }
    
    if (t.entries()>1 && ii!=int(t.entries())-1) os << ","; // commas on all butlast
    if (pretty) os << endl;
  }
  
  if (pretty) indentOut_(os, indent);
  return os << "}";
}

inline void prettyPrintOpal (const Tab& t, ostream& os, 
			     int indent=0, int indent_additive=4, 
			     bool pretty=true, bool as_m2k_array=false)
{
  indentOut_(os, indent);
  prettyPrintOpalHelper_(t, os, indent, pretty, indent_additive, as_m2k_array) << endl;
}


inline void prettyPrintOpal (const Val& v, ostream& os, 
			     int indent=0, int additive_indent=4, 
			     bool pretty=true, bool as_m2k_array=false) 
{
  // Tabs
  if (v.tag=='t') {
    Tab& t = v;
    prettyPrintOpal(t, os, indent, additive_indent, pretty, as_m2k_array);
  } 

  // Arrs or Array<numeric_type>
  else if (v.tag=='n') { 
    if (v.subtype=='Z') {
      Arr& ar = v;
      prettyPrintOpal(ar, os, indent, additive_indent, pretty, as_m2k_array);
    } else {
      prettyPrintVector_(v, os);
    }
  } 

  // Simple numeric types
  else {
    simplePrettyPrintOpal_(v, os);
  }
}

PTOOLS_END_NAMESPACE

#define OPALPRINT_H_ 
#endif // OPALPRINT_H_

