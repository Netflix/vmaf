#ifndef XMLDUMPER_H_

#include "opencontainers.h"
#include "arraydisposition.h"

// This class convert dictionaries to XML.  This is usually
// an easy mapping, because key-value mappings go easily to XML
// (the other way is more problematic).  A potential issue
// usually is how to handle attributes.... See the options below!

// Examples:
//
//  { 'book' = {
//        'chapter' = [ 'text chap 1', 'text chap 2']
//        '__attrs__' = { 'attr1':"1", 'attr2':"2" }
//  }
//  --------dumps as --------------
//  <book attr1="1" attr2="2">
//     <chapter>text chap 1</chapter>
//     <chapter>text chap 2</chapter>
//  </book>
// 

// With UNFOLDING on  (attributes are marked with _)
//
//  { 'book' = {
//        'chapter' = [ 'text chap 1', 'text chap 2']
//        '_attr1':"1", 
//        '_attr2':"2" 
//     }
//  }
//  ---------dumps as --------------
//  <book attr1="1" attr2="2">
//      <chapter date="1999">text chap 1</chapter>
//      <chapter data="2000">text chap 2</chapter>
//  </book>


// //////////////////////////////////// Options for dictionaries -> XML

// If XML attributes are being folded up, then you may
// want to prepend a special character to distinguish attributes
// from nested tags: an underscore is the usual default.  If
// you don't want a prepend char, use XML_DUMP_NO_PREPEND option
#if !defined(XML_PREPEND_CHAR)  // In both xmlreader.h and xmldumper.h
#  define XML_PREPEND_CHAR '_'
#endif

// When dumping, by DEFAULT the keys that start with _ become
// attributes (this is called "unfolding").  You may want to keep
// those keys as tags.  Consider:
//
//   { 'top': { '_a':'1', '_b': 2 }} 
//
// DEFAULT behavior, this becomes:
//   <top a="1" b="2"></top>    // This moves the _names to attributes
//
// But, you may want all _ keys to stay as tags: that's the purpose of this opt
//   <top> <_a>1</_a> <_b>2</b> </top>
#define XML_DUMP_PREPEND_KEYS_AS_TAGS 0x100

// Any value that is simple (i.e., contains no nested
// content) will be placed in the attributes bin:
// For examples:
//   { 'top': { 'x':'1', 'y': 2 }} ->  <top x="1" y="2"></top> 
#define XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES 0x200

// By default, everything dumps as strings (without quotes), but those things
// that are strings lose their "stringedness", which means
// they can't be "evaled" on the way back in.  This option makes 
// Vals that are strings dump with quotes.
#define XML_DUMP_STRINGS_AS_STRINGS 0x400

// Like XML_DUMP_STRINGS_AS_STRINGS, but this one ONLY
// dumps strings with quotes if it thinks Eval will return
// something else.  For example in { 's': '123' } : '123' is 
// a STRING, not a number.  When evalled with an XMLLoader
// with XML_EVAL_CONTENT flag, that will become a number.
#define XML_DUMP_STRINGS_BEST_GUESS 0x800

// Show nesting when you dump: like "prettyPrint": basically, it shows
// nesting
#define XML_DUMP_PRETTY 0x1000

// Arrays of POD (plain old data: ints, real, complex, etc) can
// dump as huge lists:  By default they just dump with one tag
// and then a list of numbers.  If you set this option, they dump
// as a true XML list (<data>1.0/<data><data>2.0</data> ...)
// which is very expensive, but is easier to use with other
// tools (spreadsheets that support lists, etc.).
#define XML_DUMP_POD_LIST_AS_XML_LIST 0x2000

// When dumping an empty tag, what do you want it to be?
// I.e., what is <empty></empty>  
// Normally (DEFAULT) this is an empty dictionary 'empty': {}
// If you want that to be empty content, as in an empty string,
// set this option: 'empty': ""
// NOTE: You don't need this option if you are using 
// XML_DUMP_STRINGS_AS_STRINGS or XML_DUMP_STRINGS_BEST_GUESS
#define XML_DUMP_PREFER_EMPTY_STRINGS 0x4000

// When dumping dictionaries in order, a dict BY DEFAULT prints
// out the keys in sorted/alphabetic order and BY DEFAULT an OrderedDict
// prints out in the OrderedDict order.  The "unnatural" order
// for a dict is to print out in "random" order (but probably slightly
// faster).  The "unnatural" order for an OrderedDict is sorted
// (because normally we use an OrderedDict because we WANTS its
// notion of order)
#define XML_DUMP_UNNATURAL_ORDER 0x8000

// Even though illegal XML, allow element names starting with Digits:
// when it does see a starting digit, it turns it into an _digit
// so that it is still legal XML
#if !defined(XML_TAGS_ACCEPTS_DIGITS)
#   define XML_TAGS_ACCEPTS_DIGITS 0x80
#endif 

// Allows digits as starting XML tags, even though illegal XML
#define XML_DIGITS_AS_TAGS 0x80000

// When dumping XML, the default is to NOT have the XML header 
// <?xml version="1.0">:  Specifying this option will always make that
// the header always precedes all content
#if !defined(XML_STRICT_HDR)
#  define XML_STRICT_HDR 0x10000
#endif


PTOOLS_BEGIN_NAMESPACE

// ///// Classes

// An instance of this will help dump a Val as XML
class XMLDumper {
 public:
  
  // On error, do you want to throw exception, silently continue or warn
  // on stderr?  Usually errors happens when there are multiple attributes
  // that conflict.
  enum XMLDumpErrorMode_e { SILENT_ON_ERROR, CERR_ON_ERROR, THROW_ON_ERROR };

  // Create am XML dumper.  Note that options are | together
  // XMLDumper xd(cout, XML_DUMP_PRETTY | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES);
  XMLDumper (ostream& os, int options=0, ArrayDisposition_e arr_dis=AS_LIST, 
	     int indent_increment=4,
	     char prepend_char=XML_PREPEND_CHAR, 
	     XMLDumpErrorMode_e mode = CERR_ON_ERROR) :
    os_(&os),
    fp_(NULL),
    options_(options),
    arrayDisp_(arr_dis),
    indentIncrement_(indent_increment),
    prependChar_(prepend_char), 
    mode_(mode),
    specialCharToEscapeSeq_(),
    DICTTag_("dict__")
  { 
    specialCharToEscapeSeq_['&'] = "&amp;";
    specialCharToEscapeSeq_['<'] = "&lt;";
    specialCharToEscapeSeq_['>'] = "&gt;";
    specialCharToEscapeSeq_['\"'] = "&quot;";
    specialCharToEscapeSeq_['\''] = "&apos;";
    LISTAttrs_["type__"] = "list";
  }

  // Create am XML dumper.  Note that options are | together
  // XMLDumper xd(cout, XML_DUMP_PRETTY | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES);
  XMLDumper (FILE* fp, int options=0, ArrayDisposition_e arr_dis=AS_LIST, 
	     int indent_increment=4,
	     char prepend_char=XML_PREPEND_CHAR, 
	     XMLDumpErrorMode_e mode = CERR_ON_ERROR) :
    os_(NULL),
    fp_(fp),
    options_(options),
    arrayDisp_(arr_dis),
    indentIncrement_(indent_increment),
    prependChar_(prepend_char), 
    mode_(mode),
    specialCharToEscapeSeq_(),
    DICTTag_("dict__")
  { 
    specialCharToEscapeSeq_['&'] = "&amp;";
    specialCharToEscapeSeq_['<'] = "&lt;";
    specialCharToEscapeSeq_['>'] = "&gt;";
    specialCharToEscapeSeq_['\"'] = "&quot;";
    specialCharToEscapeSeq_['\''] = "&apos;";
    LISTAttrs_["type__"] = "list";
  }

  // Wrap whether we do C style IO or C++ streams
  inline void out (char c) 
  { 
    if (fp_) {
      fputc(c, fp_); 
    } else {
      (*os_) << c;
    }
  }

  inline void out (const char* cstr) 
  { 
    if (fp_) {
      fputs(cstr, fp_); 
    } else {
      (*os_) << cstr;
    }
  }

  inline void out (const string& str) 
  { 
    if (fp_) {
      fputs(str.c_str(), fp_); 
    } else {
      (*os_) << str;
    }
  }

  // Dump without a top-level container (i.e., no containing top-level tag)
  inline void XMLDumpValue (const Val& value, int indent=0)
  { XMLDumpKeyValue(NULLKey_, value, indent); } // handles header too

  // Dump with the given top-level key as the top-level tag.
  inline void XMLDumpKeyValue (const string& key, const Val& value,int indent=0)
  {
    XMLHeader_();

    // Top level lists suck: you can't "really" have a
    // list at the top level of an XML document, only
    // a table that contains a list!
    if (value.tag=='n' && value.subtype=='Z') {
      Arr& a = value;
      Proxy p(&a, false);  // DO NOT adopt, just sharing reference
      Val top = Tab();
      top["list__"] = p;
      XMLDumpKeyValue_(key, top, indent); 
    } else {
      XMLDumpKeyValue_(key, value, indent); 
    }
  }

  // Dump *WITHOUT REGARD* to top-level container and/or XML header:
  // this allows you to compose XML streams if you need to: it just
  // dumps XML into the stream.
  inline void dump (const string& key, const Val& value, int indent=0)
  { XMLDumpKeyValue_(key, value, indent); }
  inline void dump (const Val& value, int indent=0)
  { XMLDumpKeyValue_(NULLKey_, value, indent); }

  // If the table is malformed (usually attributes conflicting), throw
  // a runtime error if strict.  By default, outputs to cerr
  void mode (XMLDumpErrorMode_e mode) { mode_ = mode; }

 protected:

  ostream* os_;             // Stream outputting to
  FILE *fp_;                // OR ... CStyle output

  int options_;             // OR ed options
  ArrayDisposition_e arrayDisp_;// The array disposition: how to deal with POD arr
  int indentIncrement_;     // How much to up the indent at each nesting level
  char prependChar_;        // '\0' means NO prepend char
  XMLDumpErrorMode_e mode_; // How to handle errors: silent, cerr, or throw
  HashTableT<char, string, 8> 
    specialCharToEscapeSeq_;  // Handle XML escape sequences
  string NULLKey_;          // Empty key
  Tab    EMPTYAttrs_;       // Empty Attrs when dumping a primitive
  Tab    LISTAttrs_;        // { "type__" = 'list' } 
  string DICTTag_;          // "dict__"
 
  // Handle the XML Header, if we want to dump it
  void XMLHeader_ ()
  {
    if (options_ & XML_STRICT_HDR) {
      out("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    }
  }

  // Dump, but allow passage of attributes
  inline void XMLDumpKeyValue_ (const string& key, const Val& value,
				int indent=0, Tab* attrs_ptr=0)
  {
    switch (value.tag) {
    case 't':
      if (options_ & XML_DUMP_UNNATURAL_ORDER) { // may want speed 
	XMLDumpTable_<Tab, It>(key, value, indent, attrs_ptr); 
      } else { // Natural order is sorted
	XMLDumpTable_<Tab, Sit>(key, value, indent, attrs_ptr); 
      }
      break;
    case 'o': 
      if (options_ & XML_DUMP_UNNATURAL_ORDER) { // may still want sorted
	XMLDumpTable_<OTab, Sit>(key, value, indent, attrs_ptr); 
      } else { // Natural order is the OrderedDict order
	XMLDumpTable_<OTab, It>(key, value, indent, attrs_ptr); 
      }
      break;
    case 'u': XMLDumpList_<Tup>(key, value, indent); 
      break;
    case 'n': XMLDumpPODListChoose_(key, value, indent); 
      break;
    default:
      XMLDumpPrimitive_(key, value, indent, attrs_ptr);
      break;
    }
  }


  // Dump just the name and make sure it's well-formed tag name
  // (or attribute name): TODO: This may be too restrictive
  inline void XMLDumpName_ (const string& tag)
  {
    if (tag.length()==0) {
      handleProblem_("tag cannot be empty (0 length)");
    }
    const char* t = tag.c_str();
    if (isdigit(t[0])) {
      if (options_ & XML_DIGITS_AS_TAGS) {
	;  // just outputs digit
      } else if (options_ & XML_TAGS_ACCEPTS_DIGITS) {
	out('_');  // Add an extra underscore at the front so legal XML
      } else {
	handleProblem_("tag must start with alphabetic or _, not "+string(1,t[0]));
      }
    } else if (!(isalpha(t[0]) || t[0]=='_')) {
      handleProblem_("tag must start with alphabetic or _, not "+string(1,t[0]));
    }
    for (size_t ii=1; ii<tag.length(); ii++) {
      if (!(isalnum(t[ii]) || t[ii]=='_' || t[ii]==':')) {  // TODO: Finish better namespace support: this allows is to read in some, but not well enough
 	string c;
	c = t[ii];
	handleProblem_("tag must contain alphanumeric or _, not "+c);
      } 
    }
    out(tag); // All good
  }

  // Dump content: this means handling escape characters
  inline string XMLContentFilter_ (const string& content)
  {
    string result; // RVO

    const char* t = content.data();
    const int len = content.length();
    string esc_seq;
    for (int ii=0; ii<len; ii++) {
      char c = t[ii];
      if (!isprint(c)) {
	int value = ((unsigned char)(c));
	char hex_val[7] = "&#x00;"; // TODO: handle non-UTF encodings
	static char hex_digits[] = "0123456789abcdef";
	if (value<16) {
	  hex_val[3] = hex_digits[(value & 0x0f)];
	  hex_val[4] = ';';
	  result += string(hex_val, 5);
	} else {
	  hex_val[4] = hex_digits[(value & 0x0f)];
	  hex_val[3] = hex_digits[value>>4];
	  result += string(hex_val, 6);
	}
      } else {
	if (specialCharToEscapeSeq_.findValue(t[ii], esc_seq)) {
	  result.append(esc_seq);
	} else {
	  result.push_back(t[ii]);
	}
      }
    }
    return result;
  }


  // Dump the start tag, with attention to formatting and options
  inline void XMLDumpStartTag_ (const string& tag, const Tab& attrs, 
				int indent, bool primitive_dump=false,
				bool none_dump = false)
  {
    //cout << "tag:" << tag << " attrs:" << attrs << " indent:" << indent << " primitive_dump:" << primitive_dump << endl;
    if (&tag==&NULLKey_) return;

    if (options_ & XML_DUMP_PRETTY) {
      string spaces(indent, ' ');
      out(spaces);
    }
    out('<');
    XMLDumpName_(tag);

    // Attributes  key1="something" key2="somethingelse"
    if (attrs.entries() >= 1) out(' ');
    for (Sit ii(attrs); ii(); ) {
      // String as is
      string attr_name = ii.key();
      if (attr_name.length()>0 && attr_name[0]==prependChar_ &&
	  ((options_ & XML_DUMP_PREPEND_KEYS_AS_TAGS)==0) ) {
	attr_name = attr_name.substr(1); // strip _
      }
      string attr_val = ii.value(); 

      //os_ << attr_name << "=" << attr_val;
      XMLDumpName_(attr_name);
      out( "=\""); out(XMLContentFilter_(attr_val)); out('"'); // TODO: handle '

      Sit jj(ii); if (jj()) out(' '); // last one, no extra space
    }
    if (none_dump) out('/');
    out('>');
    if ((options_ & XML_DUMP_PRETTY) && (!primitive_dump || none_dump)) out("\n");
  }

  // Dump the end tag, with attention to output options and formatting
  inline void XMLDumpEndTag_ (const string& tag, int indent,
			      bool primitive_dump=false)
  {
    if (&tag==&NULLKey_) return;

    if ((options_ & XML_DUMP_PRETTY) && !primitive_dump) {
      string spaces(indent, ' ');
      out(spaces);
    }
    out("</"); XMLDumpName_(tag); out(">"); // Note: Already checked that tag is okay! 

    if (options_ & XML_DUMP_PRETTY) out("\n");
  }


  // Does the tag represent a composite object: any container is
  // a composite: Tab, Arr, Tup, OTab 
  // primitive data: string, complex, int, float, etc.
  inline static bool IsComposite_ (const Val& v)
  { return v.tag=='u' || v.tag=='t' || v.tag=='n' || v.tag=='o'; }


  // Find all key-values that could be XML attributes 
  template <class TAB>
  Tab FindAttributes_ (const TAB& t)
  {
    // Collect all things that could be attributes
    Tab attrs;   // RVO
    if (t.contains("__attrs__")) {  // We want to discover automatically:
      // Attributes all in special key '__attrs__'
      attrs = t("__attrs__");
    } 

    // Attributes may also have to be found
    for (Sit ii(t); ii(); ) {
      const string& key   = ii.key();
      const Val&    value = ii.value();
      if (key=="__attrs__") continue; // already processed
      if (key=="__content__") continue; // special

      // Special character says they *MAY* become attributes
      if (key.length() > 0 && key[0] == prependChar_) {
	if (attrs.contains(key)) {
	  handleProblem_(key+string(" already in ")+Stringize(t));
	}
	string key_without_underscore = key.substr(1);
	if (!(options_ & XML_DUMP_PREPEND_KEYS_AS_TAGS)) {
	  attrs[key_without_underscore] = value;
	  continue;
	}
      }
      // Do All simple values become attributes?
      if (options_ & XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES) {
	bool simple = !IsComposite_(value);
	if (attrs.contains(key)) {
	  handleProblem_(key+string(" already in ")+Stringize(t));
	}
	if (simple) {
	  attrs[key] = value;
	}
	continue;
      }
      
    }
    return attrs; 
  }
  

  template <class LIST>
  inline void XMLDumpList_ (const string& list_name, LIST& l, int indent)
  {
    // This strange business is to handle lists with no names:
    // either nested within other lists or at the top-level so it will
    // still form well-formed XML: this is pretty rare, but we should
    // still do something useful.
    const string& tag = (&list_name==&NULLKey_) ? string("list__") : list_name;

    // Empty list
    if (l.length()==0) {  // Force list type__ so will unXMLize as an Arr()
      XMLDumpPrimitive_(tag, None, indent, &LISTAttrs_); 
      return;
    }

    // Non-empty list
    for (size_t ii=0; ii<l.length(); ii++) {
      string* key_to_use = &NULLKey_; // normally NULL RARELY: empty list
      Val* value_ptr = &l[ii];
      
      // This strange bit is because a table is directly inside a 
      // list, which means there IS no tag, which normally would be
      // an extra indent with an empty name: this is specifically because
      // of the dict within a list.  A table inside can also mean
      // the __contents__
      bool table_inside_value = (value_ptr->tag=='t' || value_ptr->tag=='o');
      int indent_inc = indentIncrement_;
      Tab attrs;
      if (table_inside_value) {
	indent_inc = 0;
	attrs = FindAttributes_(*value_ptr);
	// Special rare case: contents in special key
	if (value_ptr->contains("__content__")) {
	  value_ptr = &((*value_ptr)["__content__"]);
	}

	if ((value_ptr->tag=='t' || value_ptr->tag=='o') && value_ptr->length()==0 && l.length()==1) {
	  // This RARE situation:  
	  // { 'top': [ {} ] } -> <top type__="list"> <dict__/> </top>
	  // Empty table inside a list: Ugh: hard to express in XML
	  // without a new tag ... it's basically an anonymous table:
	  // Normally, it's easy to detect a table, but an empty
	  // dict inside a list is a special case
	  indent_inc = indentIncrement_;
	  key_to_use = &DICTTag_;
	} 
	
      } else if (value_ptr->tag == 'n' && value_ptr->subtype != 'Z' &&
		 arrayDisp_ != AS_LIST) {
	//// Array data, well has peculilarities: let it handle it
	XMLDumpPODListChoose_(tag, *value_ptr, indent, ii, 
			      (ii==0 && l.length()==1));
	continue;
      }
      // If list of 1, preserve listness by adding type field
      if (ii==0 && l.length()==1) {
	attrs["type__"]="list";
      }
      bool primitive_type = !IsComposite_(*value_ptr);
      XMLDumpStartTag_(tag, attrs, indent, primitive_type);
      XMLDumpKeyValue_(*key_to_use, *value_ptr, indent+indent_inc);
      XMLDumpEndTag_(tag, indent, primitive_type);
    } 
  }

  // Choose the appropraite template
#define XMLDUMPPOD(PODTYPE) { Array<PODTYPE>& a=v; XMLDumpPODList_(key,a,indent, inside_list_number, add_type); }
  void XMLDumpPODListChoose_ (const string& key, const Val& v, int indent,
			      int inside_list_number=-1, bool add_type=false)
  {
    if (v.tag!='n') throw runtime_error("Internal Error?");
    if (arrayDisp_==AS_LIST && v.subtype!='Z') {
      // Dump it as a list NOT, losing all Array PODiness
      Val result;
      AsList(v, result, true);
      XMLDumpList_<Arr>(key, result, indent); 
      return;
    }
    switch (v.subtype) {
      case 's': XMLDUMPPOD(int_1); break;
      case 'S': XMLDUMPPOD(int_u1); break;
      case 'i': XMLDUMPPOD(int_2); break;
      case 'I': XMLDUMPPOD(int_u2); break;
      case 'l': XMLDUMPPOD(int_4); break;
      case 'L': XMLDUMPPOD(int_u4); break;
      case 'x': XMLDUMPPOD(int_8); break;
      case 'X': XMLDUMPPOD(int_u8); break;
      case 'f': XMLDUMPPOD(real_4); break;
      case 'd': XMLDUMPPOD(real_8); break;
      case 'c': XMLDUMPPOD(cx_t<int_1>); break;
      case 'C': XMLDUMPPOD(cx_t<int_u1>); break;
      case 'e': XMLDUMPPOD(cx_t<int_2>); break;
      case 'E': XMLDUMPPOD(cx_t<int_u2>); break;
      case 'g': XMLDUMPPOD(cx_t<int_4>); break;
      case 'G': XMLDUMPPOD(cx_t<int_u4>); break;
      case 'h': XMLDUMPPOD(cx_t<int_8>); break;
      case 'H': XMLDUMPPOD(cx_t<int_u8>); break;
      case 'F': XMLDUMPPOD(complex_8); break;
      case 'D': XMLDUMPPOD(complex_16); break;
      case 'Z': XMLDumpList_<Arr>(key, v, indent); break;
      default: throw runtime_error("Currently do not handle Arrays of anything other than POD Data or Vals");
    }
  }
      
  // Dump a list of binary data as a tag with one special key:
  // arraytype__ = "<typetag>" which is some typetag (silxfdSILXFD)
  // or, every individual element as a "type__" = <typetag>"
  template <class POD>
  inline void XMLDumpPODList_ (const string& list_name, Array<POD>& l, 
			       int indent, int inside_list_number, 
			       bool add_type)
  {
    string tag = list_name;
    // The attributes for an Array of POD will the Val type tag
    Val attrs = Tab();
    string bytetag = string(1, TagFor((POD*)0));
    if (options_ & XML_DUMP_POD_LIST_AS_XML_LIST) {
      attrs["type__"] = bytetag;
    } else {
      attrs["arraytype__"] = bytetag;
    }

    // There are two ways to dump Array data: either as one tag
    // with a list of numbers, or a tag for for every number.
    // Dumping array data with a tag for every number works better with 
    // other tools (spreasheet etc.), but if you annotate EVERY ELEMENT 
    // of a long list, the XML becomes much bigger and slow.

    // Dump array with a tag for EVERY element
    bool primitive_type = true;
    Val temp;
    const int len = int(l.length());
    string inner_tag = tag;
    if (options_ & XML_DUMP_POD_LIST_AS_XML_LIST) {

      // Rare case when POD array inside list
      if (inside_list_number!=-1) {
	Val inner_attrs = Tab();
	if (inside_list_number==0 && add_type) { inner_attrs["type__"]="list"; }
	XMLDumpStartTag_(tag, inner_attrs, indent, false);
	inner_tag = "list"+Stringize(inside_list_number)+"__";
	indent += indentIncrement_;
      }

      if (len==0) {
	// Empty list
	XMLDumpStartTag_(inner_tag, attrs, indent, primitive_type);
	XMLDumpEndTag_(inner_tag, indent, primitive_type);
      } else {
	// Non-empty list
	for (int ii=0; ii<len; ii++) {
	  XMLDumpStartTag_(inner_tag, attrs, indent, primitive_type);
	  temp = l[ii];  // so prints with full precision of Val for reals, etc.
	  out(string(temp));
	  XMLDumpEndTag_(inner_tag, indent, primitive_type);
	}
      }


      // Rare case when POD array inside list
      if (inside_list_number!=-1) {
	indent -= indentIncrement_;
	XMLDumpEndTag_(tag, indent, false);
      }
    }
    // Dump as a list of numbers with just one tag: the tag, the list of data, 
    // then the end tag      
    else {
      if (inside_list_number==0 && add_type) { attrs["type__"]="list"; }
      XMLDumpStartTag_(tag, attrs, indent, primitive_type);
      for (int ii=0; ii<len; ii++) {
	temp = l[ii];  // so prints with full precision of Val for reals, etc.
	out(string(temp));
	if (ii<len-1) out(',');
      }
      XMLDumpEndTag_(tag, indent, primitive_type);
    }
  }

  template <class TABLE, class ITER>
  inline void XMLDumpTable_ (const string& dict_name, TABLE& t, int indent,
			     Tab* attrs_ptr = 0)
  {
    // Rare case: when __content__ there
    if (t.contains("__content__")) {
      Tab attrs = FindAttributes_(t);
      XMLDumpKeyValue_(dict_name, t["__content__"], indent, &attrs);
      return;
    }

    // Get attributes, Always dump start tag
    Tab attrs = (attrs_ptr==0) ? FindAttributes_(t) : *attrs_ptr;
    XMLDumpStartTag_(dict_name, attrs, indent);
    
    // Normally, just iterate over all keys for nested content
    for (ITER ii(t); ii(); ) {
      const Val& key   = ii.key();
      const Val& value = ii.value();

      // Skip over keys that have already been put in attributes
      string k = key;
      if (attrs.contains(key) || k=="__attrs__" || 
	  (k.length()>0 && k[0]==prependChar_ && attrs.contains(k.substr(1))))
	continue; // Skip in attrs

      XMLDumpKeyValue_(key, value, indent+indentIncrement_);
    }
    // Always dump end tag
    XMLDumpEndTag_(dict_name, indent);
  }
  
  
  inline void XMLDumpPrimitive_ (const string& key, const Val& value,
				 int indent, Tab* attrs_ptr=0)
  {
    Tab& attrs = (attrs_ptr==0) ? EMPTYAttrs_ : *attrs_ptr;

    if (IsComposite_(value)) {
      throw runtime_error("Trying to dump a composite type as a primitive");
    }
    if (value==None) {
      XMLDumpStartTag_(key, attrs, indent, true, true);
      return;
    }

    XMLDumpStartTag_(key, attrs, indent, true);

    // Force all strings into quotes, messy but will always work
    // with XML_EVAL_CONTENT on the way back if you have to convert
    if (options_ & XML_DUMP_STRINGS_AS_STRINGS) {
      if (value.tag=='a') { // make sure pick up quotes
	out( XMLContentFilter_(Stringize(value)));
      } else {
	out( XMLContentFilter_(value)); // Let Val pick approp. repr
      }
    } 

    // Most of the time, we can keep all strings as-is (and avoid
    // those nasty '&apos;' quotes in XML around them): those
    // strings that will become something "real values" when Evalled 
    // need to have quotes so they will come back as strings
    // when using XML_EVAL_CONTENT.  For example: '123' is a string
    // but when evaled, will become a number;  We dump this as 
    // "&apos;123&apos;" to preserve the numberness.
    else if (options_ & XML_DUMP_STRINGS_BEST_GUESS) {
      if (value.tag=='a') {
	string s = value;    // no quotes on string
	if (s.length()==0 || // always dump empty strings with &apos!
	    ((s.length()>0) && 
	     (isdigit(s[0]) || s[0]=='(' || s[0]=='-' || s[0]=='+'))) {
	  // If it starts with a number or a sign or '(' (for (1+2j), 
	  // probably a number and we WANT to stringize
	  out( XMLContentFilter_(Stringize(value))); // puts quotes on str
	} else {
	  out( XMLContentFilter_(value)); // no quotes!
	}
      } else {
	out( XMLContentFilter_(value)); // Let Val pick approp. repr
      }
    } 

    // Default, just plop content: still be careful of <empty></empty>:
    // Should that be a {}, None, [], or ""?  With this option, makes it
    // empty string (you don't need this option if you are using
    // XML_DUMP_STRINGS_BEST_GUESS or XML_DUMP_STRINGS_AS_STRINGS
    else {
      string s = value;
      if (options_ & XML_DUMP_PREFER_EMPTY_STRINGS) {
	if (s.length()==0) s = "''";  // Makes <empty></empty> into empty string
      }
      out (XMLContentFilter_(s)); 
    }
    XMLDumpEndTag_(key, indent, true);
  }


  // Centralize error handling
  inline void handleProblem_ (const string& text)
  {
    if (mode_==SILENT_ON_ERROR) return;
    if (mode_==THROW_ON_ERROR) throw runtime_error(text.c_str());
    cerr << text << endl;
  }

}; // XMLDumper

// /// Globals

// Write a Val as XML to a stream: throw a runtime-error if anything 
// bad goes down.  These are the best options for invertible transforms:
// AS_LIST might be better for dealing with Python, but you are much less 
// likely to lose information.
inline void WriteValToXMLStream (const Val& v, ostream& ofs,
				 const Val& top_level_key = "top",
				 int options = XML_DUMP_PRETTY | XML_STRICT_HDR | XML_DUMP_STRINGS_BEST_GUESS, 
				 ArrayDisposition_e arr_disp = AS_NUMERIC,
				 char prepend_char=XML_PREPEND_CHAR)
{
  const int indent = 2;
  XMLDumper xd(ofs, options, arr_disp, indent, prepend_char, 
	       XMLDumper::THROW_ON_ERROR);
  if (ofs.good()) {
    if (top_level_key==None) {
      xd.XMLDumpValue(v);
    } else {
      xd.XMLDumpKeyValue(string(top_level_key), v);
    }
  } else {
    throw runtime_error("Trouble writing the stream");
  }
}


// Write Val to a file:  throw a runtime-error if anything 
// bad goes down.  These are the best options for invertible transforms:
// AS_LIST might be better for dealing with Python, but you are much less 
// likely to lose information.
inline void WriteValToXMLFile (const Val& v, const string& filename,
			       const Val& top_level_key = "top", 
			       int options = XML_DUMP_PRETTY | XML_STRICT_HDR | XML_DUMP_STRINGS_BEST_GUESS, // best options for invertible transforms
			       ArrayDisposition_e arr_disp = AS_NUMERIC,
			       char prepend_char=XML_PREPEND_CHAR)
{
  ofstream ofs(filename.c_str());
  if (ofs.good()) {
    WriteValToXMLStream(v, ofs, top_level_key, options, arr_disp, prepend_char);
  } else {
    throw runtime_error("Trouble writing the file:"+filename);
  }
}

inline void WriteValToXMLFile (const Val& v, const char* filename,
			       const Val& top_level_key = "top",
			       int options = XML_DUMP_PRETTY | XML_STRICT_HDR | XML_DUMP_STRINGS_BEST_GUESS, // best options for invertible transforms
			       ArrayDisposition_e arr_disp = AS_NUMERIC,
			       char prepend_char=XML_PREPEND_CHAR)
{ 
  WriteValToXMLFile(v, string(filename), top_level_key, 
		    options, arr_disp, prepend_char);
}


// Write Val to a string and return said string:  
// throw a runtime-error if anything 
// bad goes down.  These are the best options for invertible transforms:
// AS_LIST might be better for dealing with Python, but you are much less 
// likely to lose information.
inline void WriteValToXMLString (const Val& v, string& xml_string,
				 const Val& top_level_key = "top", 
				 int options = XML_DUMP_PRETTY | XML_STRICT_HDR | XML_DUMP_STRINGS_BEST_GUESS, // best options for invertible transforms
				 ArrayDisposition_e arr_disp = AS_NUMERIC,
				 char prepend_char=XML_PREPEND_CHAR)
{
  ostringstream ofs;
  if (ofs.good()) {
    WriteValToXMLStream(v, ofs, top_level_key, options, arr_disp, prepend_char);
    xml_string = ofs.str();
    
  } else {
    throw runtime_error("Trouble writing this string:"+xml_string);
  }
}



// Write Val to a FILE*:  throw a runtime-error if anything 
// bad goes down.  These are the best options for invertible transforms:
// AS_LIST might be better for dealing with Python, but you are much less 
// likely to lose information.
inline void WriteValToXMLFILEPointer (const Val& v, FILE* fp, 
				      const Val& top_level_key = "top", 
				      int options = XML_DUMP_PRETTY | XML_STRICT_HDR | XML_DUMP_STRINGS_BEST_GUESS, // best options for invertible transforms
				      ArrayDisposition_e arr_disp = AS_NUMERIC,
				      char prepend_char=XML_PREPEND_CHAR)
{
  const int indent = 2;
  XMLDumper xd(fp, options, arr_disp, indent, prepend_char, 
	       XMLDumper::THROW_ON_ERROR);
  if (!ferror(fp)) {
    if (top_level_key==None) {
      xd.XMLDumpValue(v);
    } else {
      xd.XMLDumpKeyValue(string(top_level_key), v);
    }
  } else {
    throw runtime_error("Trouble writing the FILE*");
  }
}



// Convert the given Python dictionary to XML and return the XML
//    (a text string).  This uses the most common options that tend to
//    make the conversions fully invertible.
inline string ConvertToXML (const Val& given_dict) 
{
  ostringstream stream_thing;
  WriteValToXMLStream(given_dict, stream_thing, "top");
  return stream_thing.str();
}
    

PTOOLS_END_NAMESPACE


#define XMLDUMPER_H_
#endif // XMLDUMPER_H_
