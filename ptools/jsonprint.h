#ifndef JSONPRINT_H_
#define JSONPRINT_H_

// Print out a Val like JSON
// Conversions: array<POD> are turned into just arrays
// { 'array': [], 'type':'typcode' }  (where the typecode is the Val typecode)
// complex numbers are turned into { 're':1, 'im':0 }

#include "ocval.h"

OC_BEGIN_NAMESPACE

// Take a string and give it the escapes that JSON understands: it returns
// a string that a JSON parser can be understood.
inline string JSONImage (const char* s, int len=-1) 
{
  static const char escapes []  = "\"\\/\b\f\n\r\t";
  static const char printable[] = "\"\\/bfnrt"; 
  if (len==-1) len = strlen(s);
  Array<char> a(len + (len>>2));
  a.append('"');
  for (int ii=0; ii<len; ii++) {
    char c = s[ii];
    const char* where = strchr(escapes, s[ii]);
    if (!where) { // Normal 
      a.append(c);
    } else {      // character to escape
      a.append('\\');
      int location = where - escapes;
      a.append(printable[location]);
    }
  }
  a.append('"');
  return string(a.data(), a.length());
}

inline string JSONImage (const string& s) 
{ return JSONImage(s.data(), s.length()); }

// Forwards
inline ostream& JSONListPrintHelper_ (Arr& a, ostream& os, int indent,
				      bool pretty, int indent_additive);
inline ostream& JSONPrimitivePrintHelper_ (const Val& v, ostream& os);


inline ostream& JSONTablePrintHelper_ (const Val& t, ostream& os, int indent, 
				       bool pretty, int indent_additive) 
{
  // Base case, empty table
  if (t.entries()==0) return os << "{ }"; 

  // Recursive case
  os << "{";
  if (pretty) os << endl;

  // Iterate through, printing out each element
  It it(t);
  for (int ii=0; it(); ii++) {
    const Val& key = it.key();
    const Val& value = it.value();
    
    if (pretty) indentOut_(os, indent+indent_additive);
    os << JSONImage(key) << ":";
    
    // For most values, use default output method
    switch (value.tag) {
    case 'a': { 
      OCString* ap = (OCString*)&value.u.a;
      os << JSONImage(ap->c_str());
      break; 
    }
    case 't': case 'o' : {
      JSONTablePrintHelper_(value, os, pretty ? indent+indent_additive : 0, 
			    pretty, indent_additive);
      break; 
    }
    case 'u': { 
      Tup& u = value;
      Array<Val>& a = u.impl();
      Arr& aa = (Arr&)a;
      JSONListPrintHelper_(aa, os, pretty ? indent+indent_additive : 0, 
			   pretty, indent_additive);
      break; 
    }
    case 'n': { 
      if (value.subtype=='Z') {
	Arr& arr = value;
	JSONListPrintHelper_(arr, os, pretty ? indent+indent_additive : 0, 
			     pretty, indent_additive);
	break;
      } // else fall thru for other array types
    }
    
    default: 
      JSONPrimitivePrintHelper_(value, os);    break;
    }

    if (t.entries()>1 && ii!=int(t.entries())-1) os << ","; // commas on all but last
    if (pretty) os << endl;
  }

  if (pretty) indentOut_(os, indent);
  return os << "}";
}

template <class POD> 
ostream& JSONPODListPrintHelper_ (const POD* a, int len, ostream& os, 
				  int indent, bool pretty, 
				  int indent_additive, bool not_cx) 
{
  // Complex numbers handled
  char tag = TagFor((POD*)a);
  if (!not_cx) {
    tag = (tag=='f') ? 'F' : 'D';
  }
  string stag(1, tag);

  // Base case, empty
  string extra_space = pretty ? " " : "";
  if (len==0) return os << "{"+extra_space+"\"array\":[],"+
		extra_space+"\"type\":\"" + stag + "\"}";

  // Recursive case
  os << "{";
  if (pretty) { 
    os << endl;
    indentOut_(os, indent);
  }
  os << "\"type\":\""+stag+"\",";
  if (pretty) { 
    os << endl;
    indentOut_(os, indent);
  }
  os << "\"array\":[";

  // Iterate through, printing out each element
  for (int ii=0; ii<len; ++ii) {
    JSONPrimitivePrintHelper_(a[ii], os);
    if (len>1 && ii!=len-1) os << ","; // commas on all but last
  }
  os << "]";
  if (pretty) { 
    os << endl;
    indentOut_(os, indent);
  }
  return os << "}";
}


#define JSONPRINTER_(T,f) { Array<T>& a = v; return JSONPODListPrintHelper_(a.data(),a.length(),os,indent,pretty,indent_additive,f); }
#define JSONPRINTER_CX(T,T2,f) { Array<T>& a = v; return JSONPODListPrintHelper_((const T2*)a.data(),a.length()*2,os,indent,pretty,indent_additive,f); }
inline ostream& JSONListPrintDispatcher_ (const Val& v, ostream& os, 
					  int indent, bool pretty, 
					  int indent_additive) 
{
  if (v.tag=='u') {
    Tup& u = v;
    Array<Val>& a = u.impl();
    Arr& aa = (Arr&)a;
    return JSONListPrintHelper_(aa, os, indent, pretty, indent_additive);
  } else if (v.tag=='n' && v.subtype=='Z') {
    Arr& a = v;
    return JSONListPrintHelper_(a, os, indent, pretty, indent_additive);
  } else if (v.tag=='n') {
    switch (v.subtype) {
    case 's': JSONPRINTER_(int_1, true); break;
    case 'S': JSONPRINTER_(int_u1, true); break;
    case 'i': JSONPRINTER_(int_2, true); break;
    case 'I': JSONPRINTER_(int_u2, true); break;
    case 'l': JSONPRINTER_(int_4, true); break;
    case 'L': JSONPRINTER_(int_u4, true); break;
    case 'x': JSONPRINTER_(int_8, true); break;
    case 'X': JSONPRINTER_(int_u8, true); break;
    case 'b': JSONPRINTER_(bool, true); break;
    case 'f': JSONPRINTER_(real_4, true); break;
    case 'd': JSONPRINTER_(real_8, true); break;
    case 'F': JSONPRINTER_CX(complex_8, real_4, false); break;
    case 'D': JSONPRINTER_CX(complex_16, real_8, false); break;
    default : throw runtime_error("Unknown POD for JSON print");
    }; 
  } else {
    throw runtime_error("not a list for JSONPrinting purposes");
  }
}


inline ostream& JSONListPrintHelper_ (Arr& a, ostream& os, int indent,
				      bool pretty, int indent_additive) 
{
  // Base case, empty
  if (a.entries()==0) return os << "[ ]";

  // Recursive case
  os << "[";
  if (pretty) os << endl;

  // Iterate through, printing out each element
  int ent = a.entries();
  for (int ii=0; ii<ent; ++ii) {
    const Val& value = a[ii];

    if (pretty) indentOut_(os, indent+indent_additive);

    // For most values, use default output method
    switch (value.tag) {
    case 'a': {
      OCString* ap = (OCString*)&value.u.a;
      os << JSONImage(ap->c_str());
      break;
    }
    case 't': case 'o': {
      JSONTablePrintHelper_(value, os, pretty ? indent+indent_additive : 0,
			    pretty, indent_additive);
      break;
    }
    case 'u': case 'n' : {
      Arr& aa = value;
      JSONListPrintHelper_(aa, os, pretty ? indent+indent_additive : 0,
			   pretty, indent_additive);
      break;
    }

    // else fall thru for other types
    default: JSONPrimitivePrintHelper_(value, os); break;
    }

    if (a.entries()>1 && ii!=int(a.entries())-1) os << ","; // commas on all but last
    if (pretty) os << endl;
  }

  if (pretty) indentOut_(os, indent);
  return os << "]";
}


// Print only primitive types as JSON would
ostream& JSONPrimitivePrintHelper_ (const Val& v, ostream& os)
{
  switch (v.tag) {
  case 'Z' : { os << "null"; return os; }
  case 'b' : { bool b=v; os << (b ? "true" : "false"); return os; }
  case 'a' : { 
    OCString* ocp=(OCString*)&v.u.a; 
    os << JSONImage(ocp->c_str()); 
    return os;
  }
    // True basic, primitive POD types
  case's': case 'S': case'i': case'I': case'l': case'L': case'x': case'X': 
  case'f': case 'd':
    return os << v;
    // Complex types: as an array?
  case 'F': { 
    complex_8 c=v; 
    Val re=c.re; 
    Val im=c.im;
    return os << "{ \"re\":" << re << ", \"im\":" << im << "}";
  }
  case 'D': { 
    complex_16 c=v; 
    Val re=c.re; 
    Val im=c.im;
    return os << "{ \"re\":" << re << ", \"im\":" << im << "}";
  } 
  case 'n': 
    return JSONListPrintDispatcher_(v, os, 0, false, 0);

  default: throw runtime_error("Not primitive type for JSON prim print");
  }
}


// Print out a Val as JSON
inline ostream& JSONPrint (const Val& v, ostream& os, 
			   int starting_indent=0, bool pretty=true,
			   int indent_additive=4)
{
  indentOut_(os, starting_indent);
  switch (v.tag) {
  case 's': case 'S': case 'i': case 'I': case 'l': case 'L': 
  case 'x': case 'X': case 'f': case 'F': case 'd': case 'D':
  case 'b': case 'Z': case 'a':
    return JSONPrimitivePrintHelper_(v, os); 
  case 't': case 'o': {
    JSONTablePrintHelper_(v, os, starting_indent,
			  //pretty ? indent+indent_additive : 0, 
			  pretty, indent_additive);
    return os << endl;
    }
  case 'u': case 'n': {
      JSONListPrintHelper_(v, os, starting_indent, 
			   pretty, indent_additive);
      return os << endl;
    }

  default: throw runtime_error("Unknown tag for JSONPrint");
  }
}

OC_END_NAMESPACE

#endif // JSONPRINT_H_
