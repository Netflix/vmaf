#ifndef XMLLOADER_H_

#include "opencontainers.h"
#include "ocreader.h"
#include "arraydisposition.h"

#include "ocval.h"
#include "ocvalreader.h"

// The reader will take and translate the XML into appropriate
// Val data structures.  This is appropriate for Key-Value documents,
// where the tags and attributes translate readily to key-value pairs
// of dictionaries.

// Examples:

//   <book attr1="1" attr2="2">
//      <chapter>text chap 1</chapter>
//      <chapter>text chap 2</chapter>
//   </book>
// 
// becomes
//  { 'book' = {
//        '__attrs__' = { 'attr1':"1", 'attr2':"2" }
//        'chapter' = [ 'text chap 1', 'text chap 2']
//  }


///////////////////////// OPTIONS for XML -> dictionaries

// ATTRS (attributes on XML nodes) by default becomes
// separate dictionaries in the table with a 
// "__attrs__" key.  If you choose to unfold, the attributes
// become keys at the same level, with an underscore.
// (thus "unfolding" the attributes to an outer level).
// 
// For example:
//   <book attr1="1" attr2="2>contents</book>
// WITHOUT unfolding  (This is the DEFAULT)
//   { 'book' : "contents",
//     '__attrs__' : {'attr1'="1", "attr2"="2"}
//   }
// WITH unfolding:  (Turning XML_LOAD_UNFOLD_ATTRS on)
//   { 'book' : "contents",
//     '_attr1':"1", 
//     '_attr2':"2", 
//   }
#define XML_LOAD_UNFOLD_ATTRS 0x01


// When unfolding, choose to either use the XML_PREPEND character '_'
// or no prepend at all.  This only applies if XML_LOAD_UNFOLD_ATTRS is on.
//   <book attr1="1" attr2="2>contents</book>
// becomes 
//  { 'book': "content", 
//    'attr1':'1',
//    'attr2':'2'
//  }
// Of course, the problem is you can't differentiate TAGS and ATTRIBUTES 
// with this option 
#define XML_LOAD_NO_PREPEND_CHAR 0x02

// If XML attributes are being folded up, then you may
// want to prepend a special character to distinguish attributes
// from nested tags: an underscore is the usual default.  If
// you don't want a prepend char, use XML_LOAD_NO_PREPEND_CHAR option
#if !defined(XML_PREPEND_CHAR)  // Loading/Dumping share this option
#  define XML_PREPEND_CHAR '_'
#endif 

// Or, you may choose to simply drop all attributes:
// <book a="1">text<book>
//   becomes
// { 'book':'1' } // Drop ALL attributes
#define XML_LOAD_DROP_ALL_ATTRS 0x04

// By default, we use Dictionaries (as we trying to model
// key-value dictionaries).  Can also use ordered dictionaries
// if you really truly care about the order of the keys from 
// the XML
#define XML_LOAD_USE_OTABS 0x08

// Sometimes, for key-value translation, somethings don't make sense.
// Normally:
//   <top a="1" b="2">content</top>
// .. this will issue a warning that attributes a and b will be dropped
// becuase this doesn't translate "well" into a key-value substructure.
//   { 'top':'content' }
//
// If you really want the attributes, you can try to keep the content by setting
// the value below (and this will supress the warning)
// 
//  { 'top': { '__attrs__':{'a':1, 'b':2}, '__content__':'content' } }
// 
// It's probably better to rethink your key-value structure, but this
// will allow you to move forward and not lose the attributes
#define XML_LOAD_TRY_TO_KEEP_ATTRIBUTES_WHEN_NOT_TABLES 0x10

// Drop the top-level key: the XML spec requires a "containing"
// top-level key.  For example: <top><l>1</l><l>2</l></top>
// becomes { 'top':[1,2] }  (and you need the top-level key to get a 
// list) when all you really want is the list:  [1,2].  This simply
// drops the "envelope" that contains the real data.
#define XML_LOAD_DROP_TOP_LEVEL 0x20

// Converting from XML to Tables results in almost everything 
// being strings:  this option allows us to "try" to guess
// what the real type is by doing an Eval on each member:
// Consider: <top> <a>1</a> <b>1.1</b> <c>'string' </top>
// WITHOUT this option (the default) -> {'top': { 'a':'1','b':'1.1','c':'str'}}
// WITH this option                  -> {'top': { 'a':1, 'b':1.1, 'c':'str' } }
// If the content cannot be evaluated, then content simply says 'as-is'.
// Consider combining this with the XML_DUMP_STRINGS_BEST_GUESS
// if you go back and forth between Tables and XML a lot. 
//  NOTE:  This uses the ValReader, which only parses literals, so they
//         is no chance that an eval will run malicious code (the Eval of
//         the PicklingTools is very very similar to ast.literal_eval of 
//         Python 2.6 and above).
#define XML_LOAD_EVAL_CONTENT 0x40

// Even though illegal XML, allow element names starting with Digits:
// when it does see a starting digit, it turns it into an _digit
// so that it is still legal XML
#if !defined(XML_TAGS_ACCEPTS_DIGITS)
#   define XML_TAGS_ACCEPTS_DIGITS 0x80
#endif 

// Allows digits as starting XML tags, even though illegal XML
#define XML_DIGITS_AS_TAGS 0x80000

// When loading XML, do we require the strict XML header?
// I.e., <?xml version="1.0"?>
// By default, we do not.  If we set this option, we get an error
// thrown if we see XML without a header
#if !defined(XML_STRICT_HDR) 
#   define XML_STRICT_HDR 0x10000
#endif


PTOOLS_BEGIN_NAMESPACE


// ///// Classes

// Abstract base class: All the code for parsing the letters one by
// one is here.  The code for actually getting the letters (from a
// string, stream, etc.) defers and uses the same framework as the
// OCValReader and the OpalReader (so that we can handle
// context for syntax errors).

class XMLLoaderA {

 public:

  // *The ReaderA* handles IO (from file, stream or string) and is _adopted_.
  // *The options are or'ed in:  XML_LOAD_CONTENT | XML_STRICT_HDR: 
  //  see all the options above (this controls most XML loading features).  
  // *The array_disposition tells what to do with Numeric arrays: AS_LIST
  //  turns them into lists, whereas both AS_NUMERIC and AS_PYTHON_ARRAY
  //  turn them into true POD arrays (there's no difference in this
  //  context: it makes more sense in Python when there are multiple POD
  //  array choices).
  // *The prepend_char is what to look for if "folding" attributes (see above).
  // *When "problems" loading, do we output an error to cerr or not.
  XMLLoaderA (ReaderA* reader, 
	      int options,
	      ArrayDisposition_e array_disposition=AS_LIST,
	      char prepend_char=XML_PREPEND_CHAR, 
	      bool suppress_warnings_when_not_key_value_xml=false) :
    reader_(reader),
    options_(options),
    arrayDisp_(array_disposition), 
    escapeSeqToSpecialChar_(),
    prependChar_(1, prepend_char),
    suppressWarning_(suppress_warnings_when_not_key_value_xml),
    emptyString_("")
  { 
    // Low over constructor
    static const char* predeclared_entities[] = { 
        "&amp;", "&",
	"&lt;", "<",
	"&gt;", ">",
	"&apos;", "\'",
	"&quot;", "\"", 
	0
    };

    // Fill table of mappings of escape sequences to single characters
    for (int ii=0; predeclared_entities[ii]!=0; ii+=2) {
      escapeSeqToSpecialChar_[string(predeclared_entities[ii])] = 
	predeclared_entities[ii+1][0];
    }
  }

  virtual ~XMLLoaderA () { delete reader_; }

  // Look for EOF
  bool EOFComing () { return reader_->EOFComing(); }

  // Reads some XML and fills in result appropriately.  TODO: Still need
  // to handle namespaces.
  void expectXML (Val& result)
  {
    handleXMLDeclaration_();
    Arr name_attrs_content;
    expectElement_(name_attrs_content, false);
    result = tableType_();
    fillInOutput_(name_attrs_content, result);
    postProcessListsOflist__(result);
    dropTop_(result);
  }


 protected:


  // ///// Data Members
  ReaderA* reader_;                        // Defer I/O so better syntax errors
  int options_;                            // | ed options
  ArrayDisposition_e arrayDisp_;           // What to do with POD arrays
  HashTable<char> escapeSeqToSpecialChar_; // XML escape sequences
  string prependChar_;                     // When unfolding, prepend char
  bool suppressWarning_;                   // The warnings can be obnoxious
  string emptyString_;                     // thread safe empty

  // ///// Helper methods
    
  void syntaxError_ (const string& s)
  { reader_->syntaxError(s); }

  void XMLNotKeyValueWarning_ (const string& name, const Val& output)
  {
    if (suppressWarning_) { 
      return;
    }
    const char* mesg = 
      "Warning: the given input XML has content interspersed\n"
      "with keys and values:  the last content (and nesting will override\n"
      "content) seen will become the value, but this kind of XML is not good\n"
      "for just key-value pairs. Some info may be lost." ;
    cerr << mesg << endl;
    cerr << "name:" << name << endl;
    cerr << "output:" << output << endl;
    cerr << "... Continuing with best effort ..." << endl;
  }

  void XMLAttributesWithPrimitiveContentWarning_ (const string& tag,
						  const Val& value,
						  const Val& output)
  {
    if (suppressWarning_) { 
      return;
    }
    const char* mesg = 
      "Warning: the given input XML has attributes interspersed with\n"
      "primitive content:  To preserve the primitiveness of the data,\n"
      "the attributes will been dropped from the data. Please revisit\n"
      "your data format to avoid this situation so the data is more\n"
      "key-value centric.";
    cerr << mesg << endl;
    cerr << " tag:" << tag << endl;
    cerr << " value:" << value << endl;
    cerr << " output:" << output <<endl;
    cerr << " ... Continuing with best effort ..." << endl;
  }

  void arrayPODWarning_() 
  { 
    if (suppressWarning_) { 
      return;
    }
    const char* mesg = 
      "Trying to build an ArrayPOD list, but mismatched type__ attributes:\n" 
      "(or other attributes?) continuing with first definition";
    cerr << mesg << endl;
    cerr << " ... Continuing with best effort ..." << endl;
  }

  // Drop the top level key, if it makes sense
  void dropTop_ (Val& result)
  {
    // Drop the top
    if (options_ & XML_LOAD_DROP_TOP_LEVEL) {
      if ((result.tag=='o' || result.tag=='t') && result.length()==1) {
	// Use the swap trick to drop in contant time
	Val temp;
	It ii(result); ii();
	temp.swap(ii.value());
	result.swap(temp);
      } else {
	if (suppressWarning_) { 
	  return;
	}
	const char* mesg = 
	  "Trying to drop the top-level key, but there is no single-top "
	  "level key that makes sense to drop, so not doing it";
	  cerr << mesg << endl;
	cerr << " ... Continuing with best effort ..." << endl;
      }
    }
  }

  // Choose whether we use OTabs or Tabs for dictionaries
  inline Val tableType_ (Val* /* attrsp */ =0) 
  { return (options_ & XML_LOAD_USE_OTABS) ? Val(OTab()) : Val(Tab()); }

  // May or may not have an XML declaration: 
  // <?xml version="1.0" encoding="UTF-8"?> or
  // <?xml version="1.0"?>
  // Really, all we support is version 1.0 and UTF-8.  And we don't
  // have to have the header.
  void handleXMLDeclaration_()
  {
    int cc = peekNWSChar_();
    if (cc!='<') syntaxError_("No top level for XML? Content without tags");
    getNWSChar_();
    cc = peekChar_();
    if (cc==EOF) syntaxError_("Premature EOF");

    // Look for XML Declaration
    if (cc=='?') {
      expectString_("Error during XML declaration", "?xml");
      Tab xml_decl_attrs;
      getAttributeTable_(xml_decl_attrs);
      if (xml_decl_attrs.contains("version") && 
	  xml_decl_attrs("version")!="1.0") {
	syntaxError_("Can't handle any XML other than version 1.0");
      }
      if (xml_decl_attrs.contains("encoding") && 
	  xml_decl_attrs("encoding")!="UTF-8") {
	syntaxError_("Can't handle any XML encodings other than UTF-8");
      }
      expectString_("Error during XML declaration", "?>");
      consumeWS_();
    }
    // Nope, just saw a < which starts some tag
    else {
      if (options_ & XML_STRICT_HDR) {
	syntaxError_("No XML header (i.e., <?xml ... ?> on XML");
      }
      pushback_('<');
    }

    // Might be a comment or some DTDs right after
    while (1) {
      if (peekStream_("<!")) {
	if (peekStream_("<!--")) {
	  consumeComment_();
	  consumeWS_();
	} else {
	  consumeDTD_();
	  consumeWS_();
	} 
      } else {
	break;
      }
    }
	
  }

  
  // Post-processing: if the table has one entry labelled "listx__",
  // then this was a special way for XML->Tab to occur:
  // Basically:
  // { 'top:' [0,1,2] },
  //   becomes
  //<top>
  // <list__>0</list__>
  // <list__>1</list__>
  // <list__>2</list__>
  //</top>
  // We see this is as:
  // { 'top': { 'list__': [0,1,2] } }
  // .... We want to convert this BACK the right way, so we have to 
  // find all <listx__ keys> and post-process.
  void postProcessListsOflist__ (Val& child)
  {
    // Base case for POD arrays
    if (child.tag=='n' && child.subtype!='Z') {
      if (arrayDisp_==AS_LIST) {
	// Turn array into list
	Val result;
	AsList(child, result);
	result.swap(child);
      }
    }
    // Base case for strings
    else if (child.tag=='a') {
      // Eval content instead of just strings
      if (options_ & XML_LOAD_EVAL_CONTENT) {
	try {
	  
	  // Try to make Eval as fast as possible: turn off the
	  // context checking and use the string inside so 
	  // we don't copy: also don't throw, just return quickly.
	  OCString* sp = (OCString*)&child.u.a;
	  ValReader vr(sp->data(), sp->length(), false, false, false);
	  Val temp;
	  if (!vr.expectAnything(temp)) 
	    return; // If exception would be thrown, returns false:just get out.
	  if (!vr.EOFComing()) {
	    return;    // Extra characters after parse, so not reversible
	  }
	  //  Val temp = Eval(child); // Eval too expensive!!! Use above!!!!
	  child.swap(temp);
	  return;

	  // So, wasn't reversible:  something prevents from being completely
	  // invertible. For many real-valued entries, this is simply
	  // a precision (one too many or too few digits) or a format
	  // thing (1e6 vs. 100000.0).  The ValReader can handle this
	  // by JUST expecting a real or complex, and we can check if there
	  // is "more stuff" on input after a full complex or real evaluation
	  /*
	  char t = temp.tag;
	  if (t=='f' || t=='d' || t=='F' || t=='D' || t=='q' || t=='Q' || t=='Z' || ) {
	    // temp is a real-valued datapoint: real values are
	    // problematic because of precision issues.  
	    child.swap(temp);
	    return;
	  }
	  // only allow eval things that are reversible
	  string stringized_temp = Stringize(temp);
	  if (stringized_temp==child) { // Only eval "reversible" operations
	    child.swap(temp);
	    return;
	  }
	  */
    
	} catch (...) {
	  // Just ignore, and leave original as is
	}
      }
      return;
    }
    // Recursive: table
    else if (child.tag=='t' || child.tag=='o') {
      // Recursively descend
      for (It ii(child); ii(); ) {
	Val& value = ii.value();
	postProcessListsOflist__(value);
      }

      // However, A lone list__ in a table is the target audience
      //  or      A lone dict__ in a table is the target audience
      if (child.length()==1) {
	Val* vp=0;
	{
	  It ii(child); ii();
	  
	  const string key = ii.key();
	  const int keylen = key.length();
	  
	  if (key.find("list")==0 && 
	      key[keylen-2]=='_' && key[keylen-1]=='_' && 
	      ii.value().tag=='n') {
	    vp = &ii.value();
	  }
	  if (key.find("dict")==0 && 
	      key[keylen-2]=='_' && key[keylen-1]=='_' && 
	      (ii.value().tag=='t' || ii.value().tag=='o')) {
	    vp = &ii.value();
	  }
	}
	// Make sure calls to child are cleaned up
	if (vp!=0) { 
	  // child.swap(*vp) leaks because contains
	  Val node;
	  node.swap(child);
	  child.swap(*vp);
	}
      }

    }
    // Recursive: list
    else { // child.tag=='n' subtype='Z'
      Arr& a = child;
      const int len = a.length();
      for (int ii=0; ii<len; ii++) {
	postProcessListsOflist__(a[ii]);
      }
    }
  }

  // helper routine for creating a POD array
  template <class T>
    void createPODListHelper_ (const T& entry, Val& fill_me_in, bool appending)
  {
    fill_me_in = Array<T>();
    if (appending) {
      fill_me_in.append(entry);
    }
  }
  
  // Used when we first see an attribute <type__>
#define CREATEPODLIST(T) { createPODListHelper_(T(value), output,appending); }
#define CREATECXPODLIST(T) { createPODListHelper_(T(value.operator complex_16()), output, appending); }
  void createPODArray_ (Val& value, Val& output, char tag, bool appending=true)
  {
    switch (tag) {
    case 's': CREATEPODLIST(int_1); break;
    case 'S': CREATEPODLIST(int_u1); break;
    case 'i': CREATEPODLIST(int_2); break;
    case 'I': CREATEPODLIST(int_u2); break;
    case 'l': CREATEPODLIST(int_4); break;
    case 'L': CREATEPODLIST(int_u4); break;
    case 'x': CREATEPODLIST(int_8); break;
    case 'X': CREATEPODLIST(int_u8); break;
    case 'f': CREATEPODLIST(real_4); break;
    case 'd': CREATEPODLIST(real_8); break;
    case 'c': CREATEPODLIST(cx_t<int_1>); break;
    case 'C': CREATEPODLIST(cx_t<int_u1>); break;
    case 'e': CREATEPODLIST(cx_t<int_2>); break;
    case 'E': CREATEPODLIST(cx_t<int_u2>); break;
    case 'g': CREATEPODLIST(cx_t<int_4>); break;
    case 'G': CREATEPODLIST(cx_t<int_u4>); break;
    case 'h': CREATEPODLIST(cx_t<int_8>); break;
    case 'H': CREATEPODLIST(cx_t<int_u8>); break;
    case 'F': CREATECXPODLIST(complex_8); break;
    case 'D': CREATECXPODLIST(complex_16); break;
    default: throw runtime_error("Don't support lists of anything but POD"); 
    }
  }

  // Specialization for Val code ... slightly more efficient
  void addValToList_ (Val& content, Val& output)
  {
    // Assertion: Have a list now
    Arr& l = output;   
    l.append(None);
    Val& where_to_plop_content = l[l.length()-1];
    
    // Recursively fills in: by default it creates Tabs, so we need
    // to fix up into list.  TODO: Warn about non-empty attributes for lists
    Val temp=tableType_();
    fillInOutput_(content, temp);
    Val& v = temp[content[0]];
    v.swap(where_to_plop_content);
  }


#define HELPGETOUT(T) { Array<T>& out=input_array; result=OUT_T(out[index]); }
#define HELPGETCXOUT(T) { Array<T>& out=input_array; Val temp(out[index]); OUT_T t2=temp; result=t2; }
  template <class OUT_T>
  void get_ (OUT_T& result, Val& input_array, int index) 
  {
    switch (input_array.subtype) {
    case 's': HELPGETOUT(int_1); break;
    case 'S': HELPGETOUT(int_u1); break;
    case 'i': HELPGETOUT(int_2); break;
    case 'I': HELPGETOUT(int_u2); break;
    case 'l': HELPGETOUT(int_4); break;
    case 'L': HELPGETOUT(int_u4); break;
    case 'x': HELPGETOUT(int_8); break;
    case 'X': HELPGETOUT(int_u8); break;
    case 'f': HELPGETOUT(real_4); break;
    case 'd': HELPGETOUT(real_8); break;
    case 'c': HELPGETCXOUT(cx_t<int_1>); break;
    case 'C': HELPGETCXOUT(cx_t<int_u1>); break;
    case 'e': HELPGETCXOUT(cx_t<int_2>); break;
    case 'E': HELPGETCXOUT(cx_t<int_u2>); break;
    case 'g': HELPGETCXOUT(cx_t<int_4>); break;
    case 'G': HELPGETCXOUT(cx_t<int_u4>); break;
    case 'h': HELPGETCXOUT(cx_t<int_8>); break;
    case 'H': HELPGETCXOUT(cx_t<int_u8>); break;
    case 'F': HELPGETCXOUT(complex_8); break;
    case 'D': HELPGETCXOUT(complex_16); break;
    default: throw runtime_error("Type tag not supported");
    }
  }
    
  // helper for fillingNestedList: After we know output is a 
  // list, we fill it in correctly.  This is really only called 
  // when we are doing the long form of Array<POD> types:
  // <l type__='f'>1</l><l type__='f'>/l>
  template <class T>
  void addToList_ (T, Val& content, Val& output) 
  {
    // Assertion: Have a list now
    Array<T>& l = output;   
    l.append(T(0));
    T& where_to_plop_content = l[l.length()-1];
    
    // Recursively fills in: by default it creates Tabs, so we need
    // to fix up into list.  TODO: Warn about non-empty attributes for lists
    Val temp=tableType_();
    fillInOutput_(content, temp);
    Val& nested = temp[content[0]];
    T t;
    if (nested.tag=='n') {
      if (TagFor((T*)0) != nested.subtype) {
	arrayPODWarning_();
      }
      get_<T>(t, nested, 0);  // Had type__ tag too
    } else {
      T t2 = nested;
      t = t2;     // forgot type tag, but okay, still can process
    }
    swap(t, where_to_plop_content);
  }



  // The nested_name is already in the dictionary of output: by XML
  // rules, this needs to become a list.  Then either 
  // is already a list or we need to turn it into one
#define ADDTOLIST(T) { addToList_(T(0), content, output); }
  void fillingNestedList_ (Val& content, Val& output)
  
{    
    // Standard case: Not a list: either content or a dictionary: 
    // needs to be listized! (because we have repeated keys!).
    if (output.tag!='n' ||
	(output.tag=='n' && output.subtype!='Z' && !content[1].contains("type__"))) // special:  the longer array pod sequence .. don't convert!
      {

      // Assertion: "output" is either content or dictionary
      Val* val_ptr = &output; // content by default
      
      // Assertion: val_ptr points to original thing we need to swap into a list
      Val list_holder = Arr();
      Arr& new_list = list_holder;
      new_list.append(None);  // place for original value 
      new_list[0].swap(*val_ptr);
      
      // Swap out old dict for a new list
      output.swap(list_holder);
    }

    // Assertion: "output" is a list ... either POD or Arr:
    // Add in the content
    switch (output.subtype) {
    case 's': ADDTOLIST(int_1); break;
    case 'S': ADDTOLIST(int_u1); break;
    case 'i': ADDTOLIST(int_2); break;
    case 'I': ADDTOLIST(int_u2); break;
    case 'l': ADDTOLIST(int_4); break;
    case 'L': ADDTOLIST(int_u4); break;
    case 'x': ADDTOLIST(int_8); break;
    case 'X': ADDTOLIST(int_u8); break;
    case 'f': ADDTOLIST(real_4); break;
    case 'd': ADDTOLIST(real_8); break;
    case 'c': ADDTOLIST(cx_t<int_1>); break;
    case 'C': ADDTOLIST(cx_t<int_u1>); break;
    case 'e': ADDTOLIST(cx_t<int_2>); break;
    case 'E': ADDTOLIST(cx_t<int_u2>); break;
    case 'g': ADDTOLIST(cx_t<int_4>); break;
    case 'G': ADDTOLIST(cx_t<int_u4>); break;
    case 'h': ADDTOLIST(cx_t<int_8>); break;
    case 'H': ADDTOLIST(cx_t<int_u8>); break;
    case 'F': ADDTOLIST(complex_8); break;
    case 'D': ADDTOLIST(complex_16); break;
    case 'Z': addValToList_(content, output); break;
    default: throw runtime_error("Type tag not supported");
    }

  }


  // Add in the given key-value pair to the output table.  Because the
  // output may not be a table, we handle this consistently by converting
  // (if the user chooses) that to a table.
  void addAttrs_ (const string& tag, Val& value, Val& output)
  {
    // Drop all attributes
    if (options_ & XML_LOAD_DROP_ALL_ATTRS) {
      return;
    }

    // More complex: if there is some "non-insertable" content,
    // we either (a) ignore the attrs or (b) turn the content
    // into a table and THEN insert it
    if (!(output.tag=='o' || output.tag=='t')) {
      if (options_ & XML_LOAD_TRY_TO_KEEP_ATTRIBUTES_WHEN_NOT_TABLES) {
	// Otherwise, we have to turn content into a table
	// that contains the content
	Val content = tableType_();
	content["__content__"].swap(output);
	output.swap(content);
      } else {
	XMLAttributesWithPrimitiveContentWarning_ (tag, value, output);
	return;
      }
    }

    // Assertion: Standard table already there in output, 
    // so easy to just plop this in
    output[tag].swap(value); 
  }
  
  // Taking the options as presented, figure out how to deal with the
  // attributes.  Do they go in as __attrs__?  Each tag as _ members?
  // Dropped entirely?
  void addInAttrs_ (Val& attrs, Val& output)
  {
    if (attrs.length()==0) return;
    if (attrs.contains("type__")) {
      if (attrs.length()!=1) {
	arrayPODWarning_(); // TODO: Better warning?
      }
      return;  
    }

    // Unfold attributes into table
    if (options_ & XML_LOAD_UNFOLD_ATTRS) { 
      // Iterate through all attributes, and add each one to the table
      for (It ii(attrs); ii(); ) {
	const Val& orig_key = ii.key();
	Val& value = ii.value();
	
	// Choose what prepend char is
	string key = string(orig_key);
	if (!(options_ & XML_LOAD_NO_PREPEND_CHAR)) {
	  key = prependChar_ + key;
	}
	addAttrs_(key, value, output);
      }
    }  
    // The DEFAULT is as __attrs__: only do this if we haven't
    // set the unfold
    else {
      addAttrs_("__attrs__", attrs, output);
    }
  }

  // ... helper function for when processing arraytype__ 
  template <class T>
  void handlePODArraysHelper_ (T, Val& output, Arr& name_attrs_content)
  {
    // Set the output, and check a few things
    output = Array<T>();
    if (name_attrs_content.length()<=2) { return; } // No content
    Val& content = name_attrs_content[2];
    string string_content;
    if (content.tag == 'a') {
      string_content = string(content);
    } else if (content.tag=='n' && content.length()>0 && content[0].tag=='a') {
      string_content = string(content[0]);
    } else if (content.tag=='n' && content.length()==0) {
      string_content = "";
    } else {
      cout << content << endl;
      throw runtime_error("Expecting solely string content for array of POD");
    }

    // Assertion: We have a string of stuff, hopefully , separated numbers
    const size_t content_length = size_t(string_content.length());
    for (size_t pos=0; pos < content_length;) {

      // skip front white space
      while (pos<content_length && isspace(string_content[pos]))
	pos++; 

      // Get content out as Val, and have it reuse parsing mech
      size_t where = string_content.find(',', pos);
      if (where==string::npos) where = content_length;
      Val data = 0;
      try {
	data = Eval(string_content.substr(pos, where-pos));
      } catch (...) { 
	// pass; // Ignore weirdness parsing numbers and move on
      }
      T temp = data;
      output.append(temp);

      // Advance
      pos = where+1;
    }
  }

  // If the attsr contain arraytype__, then the content is a POD
  // array:  Turn this into the proper type!
#define LOADPODARRAY(T) handlePODArraysHelper_(T(0),output,name_attrs_content); 
  void handlePODArrays_ (Arr& name_attrs_content, // not const so can swap
			 Val& output)             
  {
    const string name = name_attrs_content[0];
    Val& attrs        = name_attrs_content[1]; // should be a table
    //cout << "name:" << name << " attrs:" << attrs << endl;

    string val_tag = attrs("arraytype__");
    switch (val_tag[0]) {
      case 's': LOADPODARRAY(int_1); break;
      case 'S': LOADPODARRAY(int_u1); break;
      case 'i': LOADPODARRAY(int_2); break;
      case 'I': LOADPODARRAY(int_u2); break;
      case 'l': LOADPODARRAY(int_4); break;
      case 'L': LOADPODARRAY(int_u4); break;
      case 'x': LOADPODARRAY(int_8); break;
      case 'X': LOADPODARRAY(int_u8); break;
      case 'f': LOADPODARRAY(real_4); break;
      case 'd': LOADPODARRAY(real_8); break;
      case 'c': LOADPODARRAY(cx_t<int_1>); break;
      case 'C': LOADPODARRAY(cx_t<int_u1>); break;
      case 'e': LOADPODARRAY(cx_t<int_2>); break;
      case 'E': LOADPODARRAY(cx_t<int_u2>); break;
      case 'g': LOADPODARRAY(cx_t<int_4>); break;
      case 'G': LOADPODARRAY(cx_t<int_u4>); break;
      case 'h': LOADPODARRAY(cx_t<int_8>); break;
      case 'H': LOADPODARRAY(cx_t<int_u8>); break;
      case 'F': LOADPODARRAY(complex_8); break;
      case 'D': LOADPODARRAY(complex_16); break;
    default: throw runtime_error("arraytype__ attribute is not recognized");
    }
    
  }

  // We have seen attrs("type__") == "list", so we have to be careful
  // parsing this thing
  void handleAttributeTypeOfList (Val& look)
  {
    // Returned with something that SHOULD be a list: most of the
    // time this means turning the nested thing into a list
    //cout << "Got type list ... here's what I got:" << look << endl;
    if (look.tag=='n' && look.subtype=='Z') {
      // already a list, don't have to do anything
    }
    else if ((look.tag=='t' || look.tag=='o') && look.length()==0) {
      look = Arr();
    } else {
      Val a=Arr();
      a.append(None);
      a[0].swap(look);
      look.swap(a);
    }
    return;
  }

  // helper; check to see if a "type__" tag is a legal one
  void type__fillIn_ (const Val& type_tag, Val& content, bool appending,
		      Val& result)
  {
    // Tag so we will create a  BRAND NEW minted Array POD!
    char tag = 'Z';
    string stag = type_tag;
    if (stag.length()>0) tag = stag[0];
    
    if (stag=="list") {
      result = Arr();
      if (appending) 
	result.append(content);
    } else if (stag.length()!=1 || !strchr("silxfdSILXceghCEGHFDb",stag[0])) {
      cerr << "Unknown tag '" << stag << "' .. ignoring " << endl;
      result = content;  // TODO: Swap?
    } else { // legal tag
      result = None;
      createPODArray_(content, result, tag, appending);
    }
  }

  // MAIN ENTRY:
  // Centralize all the output options for the given XML here.
  // Once we have parsed all the nessary parsed XML, we want to turn
  // it into a much simpler key-value thingee.  Output is assumed to
  // be the "dictionary" to fill in: it shoudl be set to an empty dict
  // by the caller, and it will be filled in by name_attrs_content
  void fillInOutput_ (Arr& name_attrs_content, // not const so can swap
		      Val& output)          
  {
    // Parsed work here: need to turn into better vals
    const string name = name_attrs_content[0];
    Val& attrs        = name_attrs_content[1]; // should be a table
    //cerr << "name:" << name << " attrs:" << attrs << endl;
    //if (name_attrs_content.length()>2) { // Content may be empty because of />
    //  cerr << "content:" << name_attrs_content[2] << endl;
    //}

    // Default of no content
    output[name] = tableType_();
    Val& look = output(name); 

    // Arrays of POD content have some special keys to distinguish them:
    // array_type as an attribute
    if (attrs.contains("arraytype__")) {
      handlePODArrays_(name_attrs_content, look);
      if (attrs.contains("type__") && attrs["type__"]=="list") {
	handleAttributeTypeOfList(look);
      }
      return; // TODO: revisit?  To handle listn__ ?
    }

    // Special case: type__ tag, empty content
    if (attrs.contains("type__") && 
	(name_attrs_content.length()==2 || 
	 (name_attrs_content.length()==3 && name_attrs_content[2].length())==0)) {
      // <tag type__="something"></tag> or
      // <tag type__="something"/> 
      // NOTE: This needs to become a PODArray of empty
      Val dud = 666;
      type__fillIn_(attrs("type__"), dud, false, // don't append anything
		    output[name]);
    }


    // Figure out which things will be lists, which will be attrs,
    // which will be nested tags.  
    if (name_attrs_content.length()>2) { // Content may be empty because of />
      Arr& contents   = name_attrs_content[2];  
      for (size_t ii=0; ii<contents.length(); ii++) {
	Val& content = contents[ii];
	
	// Nested content: either needs to become a list or a dict
	if (content.tag == 'n') { 
	  const string nested_name = content[0];
	  
	  // If name is already there: then this is from an XML list with
	  // repeated entries, so the entry becomes a list
	  if (look.tag!='a' && look.contains(nested_name)) {
	    fillingNestedList_(content, look(nested_name));
	  }
	  
	  // Name not there, so we need to insert a new table
	  else {
	    // Already filled in for something: careful, because
	    // we may have content littering the key-value portions
	    if (look.tag == 'a') { 
	      XMLNotKeyValueWarning_(nested_name, look); // May destroy content  
	    }
	    // Force name to be a table ... may destroy content
	    if (look.tag!='o' && look.tag!='t') {
	      output[name] = tableType_();  
	    }
	    fillInOutput_(content, output[name]);
	  }
	  
	} 
	// Plain primitive content
	else if (content.tag == 'a') { 
	  if (attrs.contains("type__")) { // special key means Array<POD>, List
	    // Tag so we will create a  BRAND NEW minted Array POD!
	    type__fillIn_(attrs("type__"), content, true, // fill in with con
			  output[name]);
	  } else {
	    if (((look.tag=='n' && look.length()==0)) ||
		((look.tag =='o' || look.tag=='t') && look.length()>0)) {
	      XMLNotKeyValueWarning_(name, content); 
	    }
	    output[name] = content;  // TODO: Swap?
	  }

	} else {
	  syntaxError_("Internal Error?  Malformed XML");
	}
      }
    }

    // POST_PROCESSING

    // Adding a tag of "type__"="list" means we want to make sure
    // some single element lists are tagged as lists, not tables
    if (attrs.contains("type__") && attrs["type__"]=="list") {
      handleAttributeTypeOfList(look);
    }

    // We want to do it AFTER we have processed all tags, as we want to
    // always *PREFER* tags so that if there might be an unfold of attributes,
    // we don't step on a real tag.
    addInAttrs_(attrs, look);
  }


  // Expect a string of particular chars
  void expectString_ (const char* error_message_prefix, 
		      const char* string_to_expect)
    
  {
    int len = strlen(string_to_expect);
    for (int ii=0; ii<len; ii++) {
      int xc = getChar_();
      string char_found;  char_found = xc;
      string char_look;   char_look  = string_to_expect[ii];
      if (xc==EOF) { 
	syntaxError_(string(error_message_prefix)+
		     ":Premature EOF while looking for '"+char_look+"'");
      }
      else if (char_look != char_found) {
	syntaxError_(string(error_message_prefix)+":Was looking for '"+
		     char_look+"' but found '"+char_found+"'");
      }
    }
  }

  // Macro for expect_: one single character getting
#define EXPECTONECHAR(MSG, SUFFIX, EXPECT) { int get=getNWSChar_(); \
if (EXPECT!=get) { string t; t=get; string get_string = (get==EOF) ? "EOF":t; \
  string one_of_set; one_of_set=EXPECT; syntaxError_("Expected one of:'"+\
  one_of_set+"', but saw '"+get_string+"' on input during "+string(MSG)+string(SUFFIX)); } }
   
  // Expect some characters in a set: if not, throw error with message
  char expect_ (const char* message, const string& suffix, 
		const string& one_of_set)
  {
    // Single character token
    int get      = getNWSChar_();
    int expected = -1;
    for (size_t ii=0; ii<one_of_set.length(); ii++) 
      if (get==one_of_set[ii]) 
	expected = one_of_set[ii];

    if (get!=expected) {
      string get_string;
      if (get==EOF) get_string="EOF"; else get_string=get;
      syntaxError_("Expected one of:'"+one_of_set+
		   "', but saw '"+get_string+"' "
		   "on input during "+string(message)+suffix);
    }
    return get;
  }

  // Look for the ending character, grabbing everything
  // in between.  Really, these are for XML escapes
  string expectUntil_ (char end_char) 
  {
    Array<char> ret(8);
    while (1) {
      int ii = getChar_();
      if (ii==EOF) {
	char cc[2] = { 0, 0 }; cc[0] = end_char;
	syntaxError_("Unexpected EOF before "+string(cc)+" encountered");
      }

      char c = ii;
      ret.append(c);

      if (c==end_char) {
	string name = string(ret.data(), ret.length());
	if (name.length()>1 && name[1] == '#') { // Numeric char. references
	  if (name.length()>2 && (tolower(name[2])) == 'x') { // hex number
	    if (name.length()<=4) { // Not really legal ... &#x;
	      syntaxError_("Expected some digits for hex escape sequence");
	    }
	    if (name.length()>19) {
	      syntaxError_("Too many digits in hex escape sequence");
	    }
	    // Every digit in between must be a hex digit
	    static char hexdigits[] = "0123456789abcdef";
	    int_u8 hexme=0;
	    for (size_t ii=3; ii<name.length()-1; ii++) {
	      char dig = tolower(name[ii]);
	      char* result = strchr(hexdigits, dig);
	      if (!result) {
		syntaxError_("Expected hex digits only in escape sequence");
	      }
	      int value = result - hexdigits;
	      hexme = hexme*16 + value;
	    }
	    // if (hexme==0) syntaxError("Can't have \x0 is input"); // ridiculopus restriction
	    // All done accumulating hex digits
	    // Since we only do UTF-8 for now, truncate
	    return string(1, char(hexme));
	  } else { // decimal number
	    syntaxError_("Missing x for escape sequence");
	  }
	}
	char special_char = '*'; // just something to shut up compiler
	if (escapeSeqToSpecialChar_.findValue(name, special_char)) {
	  return string(1,special_char);
	} else {
	  syntaxError_("Unknown XML escape sequence:"+name);
	}

      }
      
    }
  }

  // Simply get the name, everything up to whitespace
  string getElementName_ ()
  {
    Array<char> name; // Array appends better than string
    
    // Makes sure starts with 'a..ZA..Z_/'
    int ii = getChar_();
    if (ii==EOF) syntaxError_("Unexpected EOF inside element name");
    char c = ii;
    if (isdigit(c)) {
      if (options_ & XML_DIGITS_AS_TAGS) {
	;  // just allow it
      } else if (!(options_ & XML_TAGS_ACCEPTS_DIGITS)) {
	syntaxError_("element names can't start with '"+string(1, c)+"'");
      } else {
	name.append('_');  // Add extra underscore to keep legal xml
      }
    } else if (!(isalpha(c) || ii=='_' || ii=='/')) { 
      syntaxError_("element names can't start with '"+string(1, c)+"'");
    }
    name.append(c);

    // .. now, make sure rest of name contains _, A..Za..Z, numbers
    while (1) {
      int ii = peekChar_();
      if (ii==EOF) break;
      char c = ii;
      if (isalnum(c) || c=='_' || c==':') {
	(void)getChar_();
	name.append(c);
      } else {
	break;
      }
    }
    return string(name.data(), name.length());
  }

  // Get the attribute="value" names.  Expect "value" to be surrounded
  // by either " or ' quotes.
  void getKeyValuePair_ (string& key, string& value)
  {
    // Simple name
    key = getElementName_();
    consumeWS_();
    EXPECTONECHAR("looking at key:", key, '=');
    consumeWS_();
    char which_quote = expect_("looking at key:", key, "\"'"); 
    
    // Follow quotes until see new one.  TODO:  look for escapes?
    Array<char> value_a;
    while (1) {
      int ii = getChar_();
      if (ii==EOF) { 
	syntaxError_("Unexpected EOF parsing key:"+key);
      } else if (ii==which_quote) {
	value = string(value_a.data(), value_a.length());
	return;
      } else if (ii=='&') { // start XML escape sequence 
	pushback_('&'); // looks for full &amp;, so need to pushback &
	string esc = expectUntil_(';');
	for (size_t ii=0; ii<esc.length(); ii++) { // Most likely single char
	  value_a.append(esc[ii]);
	}
      } else {
	value_a.append(char(ii));
      }
    }
  }



  // Assumption: We are just after the ELEMENT_NAME in "<ELEMENT_NAME
  // att1=1, ... >" and we are looking to build a table of attributes.
  // TODO: Should this be a list?
  void getAttributeTable_ (Tab& attribute_table)
  {
    // The attribute list may be empty
    int ii = peekNWSChar_();
    char c = ii;
    if ('>'==c || '/'==c || '?'==c || EOF==ii) return;

    // Expecting something there
    while (1) {
      string key, value;
      consumeWS_();
      getKeyValuePair_(key, value);
      attribute_table[key] = value;

      consumeWS_();
      ii = peekChar_();
      c = ii;
      if ('>'==c || '/'==c || '?'==c || EOF==ii) return;
    }
  }


  bool isXMLSpecial_ (char c)
  {
    switch (c) {
    case '<' : 
      return true;
    default : 
      return false;
    }
  }


  // Expect a tag: starts with a < ends with a >.  If there is an
  // attribute list, will be the second element.  
  bool expectTag_ (Arr& a)
  {
    bool is_empty_tag = false;

    // Expecting '<' to start
    EXPECTONECHAR("looking for start of tag", emptyString_, '<');
    
    // Assumption: Saw and got opening '<'.  Get the name
    string element_name = getElementName_();
    a.append(element_name);
    
    // Assumption: Got the < and NAME.  Now, Get the list of attributes
    // before the end of the '>'
    a.append(Tab()); // attribute table ALWAYS a tab
    Tab& attribute_table = a[a.length()-1]; 
    getAttributeTable_(attribute_table);
    
    // Assumption: End of list, consume the ">" or "/>"
    int ii = peekNWSChar_();
    if (EOF==ii) {
      syntaxError_("Unexpected EOF inside tag '"+element_name+"'");
    } 
    if (ii=='/') {
      EXPECTONECHAR("empty content tag", emptyString_, '/');
      is_empty_tag = true;
    }
    EXPECTONECHAR("looking for end of tag", element_name, '>');

    // End of list, make sure its well formed
    if (is_empty_tag && element_name.length()>0 && element_name[0]=='/') {
      syntaxError_(
	   "Can't have a tag start with </ and end with"
           "/> for element:"+element_name);
    }

    return is_empty_tag;
  }

  // Expect a string of base content.  Collect it until you reach a
  // special character which ends the content ('<' for example).
  void expectBaseContent_ (string& content)
  {
    Array<char> ret;
    while (1) {
      int c = peekChar_();
      if (c==EOF) { 
	return;
      } else if ('&'==c) {
	string entity = expectUntil_(';'); // Handles escapes for us
	for (size_t ii=0; ii<entity.length(); ii++) {
	  ret.append(entity[ii]);
	}
      } else if (!isXMLSpecial_(c)) {
	c = getChar_();
	ret.append(char(c));
      } else {
	// We have a '<':  is it a comment or start of a tag?
	if (peekStream_("<!--")) {
	  consumeComment_();
	  continue;
	}
	content += string(ret.data(), ret.length());
	return;
      }
    }
  }


  // [ 'book',                         // name 
  //   {'attr1': "1", 'attr2'="2"},    // table of attributes
  //   ["content"]                     // actual content
  // ]                  
  // 
  // If content is nested:
  // <book attr1="1" attr2="2">
  //     <chapter>text chap 1</chapter>
  //     <chapter>text chap 2</chapter>
  // </book>
  //
  // becomes
  //
  // [ 'book',  
  //   {'attr1'="1" 'attr2'="2"},
  //   [ ... ]
  // ] 
  // where [ ... ] is 
  // [ ['chapter', {}, ["text chap 1"]], ['chapter', {}, ["text chap 2"]] ]
  //
  // We are starting with a beginning <tag> and we will return the table
  // up to the end </tag>.  In other words, the next character we expect
  // to see is a '<'.  This return the tag for the element, and fills
  // in the element with some composite container (based on the options).
  void expectElement_ (Arr& element, bool already_consumed_begin_tag=false)
  {
    // Get '<' NAME optional_attribute_list '>', put NAME, attr_list in
    if (!already_consumed_begin_tag) {
      Arr tag_and_attrs;
      bool is_empty_tag = expectTag_(element); 
      if (is_empty_tag) return;
    }
    const string tag_name = element[0]; // Name always at front element

    // Assumption, consumed < NAME atr_list '>' of ELEMENT.
    // Everything that follow is content until we hit the </theendtag>
    
    // The content is a list of possibly nested ELEMENTs
    element.append(Arr()); 
    Arr& content = element[element.length()-1];

    while (1) {
      
      string whitespace = consumeWSWithReturn_();

      // We immediately see a <, is nested tag or end tag?
      int ci = peekChar_();
      if (ci == EOF) syntaxError_("Premature EOF?");
      char c = ci;
      if ('<' == c) { // Immediately saw <

	// May be comment!
	if (peekStream_("<!--")) {
	  consumeComment_();
	  continue;
	}
	
	// Get next tag 
	Arr new_tag; 
	bool is_empty_tag = expectTag_(new_tag);
	string new_tag_name = new_tag[0]; // name always at front of list
	
	// Check for / to see if end tag
	if (new_tag_name.length() > 0 && new_tag_name[0]=='/') {
	  if (new_tag_name.substr(1)==tag_name) { // saw end tag
	    return; // all good!
	  } else {
	    syntaxError_(
	       "Was looking for an end tag of '"+tag_name+
	       "' and saw an end tag of '"+new_tag_name+"'");
	  }
	}

	// This is a nested XML start tag
	else { 
	  content.append(Arr());
	  Arr& end_of = content[content.length()-1];
	  end_of.swap(new_tag);
	  //content.append(new_tag);
	  Arr& nested_element = content[content.length()-1];
	  if (!is_empty_tag) { 
	    expectElement_(nested_element, true); // already consumed tag!
	  }
	}
      }  
      
      // No <, so it must be some content which we collect
      else {
	string base_content= whitespace;
	expectBaseContent_(base_content);
	content.append(base_content);
      }
    }
    
  }

  // If we see the given string as the next characters on the
  // input, return true.  Otherwise, false.  Note, we leave the
  // stream exactly as it is either way.
  bool peekStream_ (const string& given)
  {
    // Holding area for stream as we peek it
    Array<int> hold(given.length());
    hold.fill(0);

    // Continue peeking and holding each char you see
    bool peeked_stream_ok = true;
    const int len = int(given.length());
    int ii;
    for (ii=0; ii<len; ii++) {
      int ci = getChar_();
      hold[ii] = ci;
      if (ci == EOF || ci != given[ii]) {
	peeked_stream_ok = false;
	break;
      }
    }
    if (peeked_stream_ok) ii--; // All the way through .. okay!

    // Restore the stream to its former glory
    for (int jj=ii; jj>=0; jj--) {
      pushback_(hold[jj]);
    }
    return peeked_stream_ok;
  }


  // Assumes next four characters are <!--: a comment is coming.  
  // When done, stream reads immediately after comment ending -->
  void consumeComment_ ()
  {
    expectString_("Expecting <!-- to start comment?", "<!--");
    while (1) {
      int ci = getChar_();
      if (ci==EOF) syntaxError_("Found EOF inside of comment");
      if (ci!='-') continue;

      // Saw a - ... looking for ->
      ci = getChar_();
      if (ci==EOF) syntaxError_("Found EOF inside of comment");
      if (ci!='-') continue;

      // Saw second - ... looking for >
      while (1) {
	ci = getChar_();
	if (ci==EOF) syntaxError_("Found EOF inside of comment");
	if (ci=='-') continue;   // Saw lots of --
	if (ci=='>') return;     // All done! Consumed a comment
	break; // no --, start over
      }
    }

  }

  // Currently don't handle DTDs; just throw them away
  void consumeDTD_ ()
  {
    expectString_("Expecting <! to start a DTD", "<!");
    while (1) {

      // You can have comments and NESTED DTDs in these things, ugh
      if (peekStream_("<!")) {
	if (peekStream_("<!--")) {
	  consumeComment_();
	} else {
	  consumeDTD_();
	}
      }

      // End of is just >
      int ci = getChar_();
      if (ci==EOF) syntaxError_("Found EOF inside of <!");
      if (ci=='>') return;
    }
  }

  // Plain whitespace, no comments
  string consumeWSWithReturn_ () 
  {
    string retval;
    while (1) {
      int cc = peekChar_();
      if (cc==EOF) break;
      if (isspace(cc)) {
	retval.append(1, char(cc));
	(void)getChar_();
	continue;
      } else {
	break;
      }
    }
    return retval;
  }

  // A derived class implements these methods to read characters from
  // some input source.
  int getNWSChar_ ()    { return reader_->getNWSChar_(); }
  int peekNWSChar_ ()   { return reader_->peekNWSChar_(); }
  int getChar_ ()       { return reader_->getChar_(); }
  int peekChar_ ()      { return reader_->peekChar_(); } 
  int consumeWS_ ()     { return reader_->consumeWS_(); }
  void pushback_ (int pushback_chr) { return reader_->pushback_(pushback_chr);}

}; // XMLLoaderA



// Helper class to handle reading strings from an XML string
class XMLStringReader_ : public StringReader {
 public:
  
  XMLStringReader_ (const char* s, int /*len*/=-1) : StringReader(s, -1) { }
  XMLStringReader_ (Array<char>& a) : StringReader(a) { }
  XMLStringReader_ (const Array<char>& a) : StringReader((Array<char>&)a) { }

 protected:
  // Return the index of the next Non-White Space character.
  // The default string reader handles # comments, which is NOT
  // what we want.  In fact, comments in XML are really only in
  // one syntactic place, so actually expect them explicitly when
  // reading them, otherwise, we don't expect them at all.
    // Return the index of the next Non-White Space character.
  int indexOfNextNWSChar_ () 
  {
    const int len=length_;
    int cur = current_;
    if (cur==len) return cur;
    for (; cur<len && isspace(data_[cur]); cur++) {
      ;
    }
    return cur;
  }

}; // XMLStringReader_


// The XMLLoader reads XML from strings
class XMLLoader : public XMLLoaderA {

 public:

  XMLLoader (const Array<char>& a, int options,
	     ArrayDisposition_e arr_disp = AS_LIST,
	     char prepend_char=XML_PREPEND_CHAR, 
	     bool suppress_warnings_when_not_key_value_xml=false) : 
    XMLLoaderA(new XMLStringReader_(a), options, arr_disp, prepend_char, 
	       suppress_warnings_when_not_key_value_xml)
  { }

  XMLLoader (const char* s, int options,
	     ArrayDisposition_e arr_disp = AS_LIST,
	     char prepend_char=XML_PREPEND_CHAR, 
	     bool suppress_warnings_when_not_key_value_xml=false) : 
    XMLLoaderA(new XMLStringReader_(s), options, arr_disp, prepend_char, 
	       suppress_warnings_when_not_key_value_xml)
  { }

}; // XMLLoader


// Helper class for reading XML ASCII streams
class XMLStreamReader_ : public StreamReader {

 public:

  XMLStreamReader_ (istream& is) : StreamReader(is) { }

 protected:
  // This routines buffers data up until the next Non-White space
  // character, ands returns what the next ws char is _WITHOUT GETTING
  // IT_.  It returns "peek_ahead" to indicate how many characters
  // into the stream you need to be to get it.
  int peekIntoNextNWSChar_ (int& peek_ahead)
  {
    peek_ahead = 0;  // This marks how many characters into the stream
                     // we need to consume
    while (1) {
      // Peek at a character either from the cache, or grab a new char
      // of the stream and cache it "peeking" at it.
      int c = '*';
      if (peek_ahead >= cached_.length()) {
	c = is_.get();
	cached_.put(c);
      } else {
	c = cached_.peek(peek_ahead);
      }

      //cout << "peeklIntoNextNWSChar_: c = " << char(c) << " or " << c << endl;

      // Look at each character individually
      if (c==EOF) {
	// We never consume the EOF once we've seen it
	return c;
      } else if (isspace(c)) { // whitespace but NOT comments!
	peek_ahead++;
	continue;
      } else {
	return c;
      }
    }
  }
}; // XMLStreamLoader_  


// Read an XML table from a stream
class StreamXMLLoader : public XMLLoaderA {

 public:

  // Open the given file, and attempt to read Vals out of it
  StreamXMLLoader (istream& is, 
		   int options,
		   ArrayDisposition_e arr_disp = AS_LIST,
		   char prepend_char=XML_PREPEND_CHAR, 
		   bool suppress_warnings_when_not_key_value_xml=false) : 
    XMLLoaderA(new XMLStreamReader_(is), options, arr_disp, prepend_char, 
	       suppress_warnings_when_not_key_value_xml)
  { }

}; // StreamXMLLoader

// Helper class for reading XML ASCII streams
class XMLFILEPointerReader_ : public FILEPointerReader {

 public:

  XMLFILEPointerReader_ (FILE* fp) : FILEPointerReader(fp) { }

 protected:
  // This routines buffers data up until the next Non-White space
  // character, ands returns what the next ws char is _WITHOUT GETTING
  // IT_.  It returns "peek_ahead" to indicate how many characters
  // into the stream you need to be to get it.
  int peekIntoNextNWSChar_ (int& peek_ahead)
  {
    peek_ahead = 0;  // This marks how many characters into the stream
                     // we need to consume
    while (1) {
      // Peek at a character either from the cache, or grab a new char
      // of the stream and cache it "peeking" at it.
      int c = '*';
      if (peek_ahead >= cached_.length()) {
        c = fgetc(fp_);
        cached_.put(c);
      } else {
        c = cached_.peek(peek_ahead);
      }

      //cout << "peeklIntoNextNWSChar_: c = " << char(c) << " or " << c << endl;

      // Look at each character individually
      if (c==EOF) {
        // We never consume the EOF once we've seen it
        return c;
      } else if (isspace(c)) { // whitespace but NOT comments!
        peek_ahead++;
        continue;
      } else {
        return c;
      }
    }
  }
}; // XMLFILEPointerLoader_


// Read an XML table from a FILE Pointer
class FILEPointerXMLLoader : public XMLLoaderA {

 public:

  // Open the given file, and attempt to read Vals out of it
  FILEPointerXMLLoader (FILE* fp,
			int options,
			ArrayDisposition_e arr_disp = AS_LIST,
			char prepend_char=XML_PREPEND_CHAR,
			bool suppress_warnings_when_not_key_value_xml=false) :
    XMLLoaderA(new XMLFILEPointerReader_(fp), options, arr_disp, prepend_char,
               suppress_warnings_when_not_key_value_xml)
  { }

}; // FILEPointerXMLLoader


// Read XML from a FILE* and turn it into a dictionary.
// The options below represent the 'best choice' for invertibility:
// although AS_NUMERIC is less compatible, you are not going to lose
// any information.
inline void ReadValFromXMLFILE (FILE *fp, Val& v,
				int options = XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, // best option for invertibility
				ArrayDisposition_e arr_disp = AS_NUMERIC,
				char prepend_char=XML_PREPEND_CHAR)
{
  if (!ferror(fp)) {
    FILEPointerXMLLoader sv(fp, options, arr_disp, prepend_char, false);
    sv.expectXML(v);
  } else {
    throw runtime_error("Trouble reading FILE*");
  }
}


// Read XML from a stream and turn it into a dictionary.
// The options below represent the 'best choice' for invertibility:
// although AS_NUMERIC is less compatible, you are not going to lose
// any information.
inline void ReadValFromXMLStream (istream& ifs, Val& v,
				  int options = XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, // best option for invertibility 
				  ArrayDisposition_e arr_disp = AS_NUMERIC,
				  char prepend_char=XML_PREPEND_CHAR)
{
  if (ifs.good()) {
    StreamXMLLoader sv(ifs, options, arr_disp, prepend_char, false);
    sv.expectXML(v);
  } else {
    throw runtime_error("Trouble reading stream");
  }
}

// Read XML from a file and turn it into a dictionary: most invertible options
inline void ReadValFromXMLFile (const string& filename, Val& v,
				int options = XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, // best option for invertibility 
				ArrayDisposition_e arr_disp = AS_NUMERIC,
				char prepend_char=XML_PREPEND_CHAR)
{
  ifstream ifs(filename.c_str());
  if (ifs.good()) {
    ReadValFromXMLStream(ifs, v, options, arr_disp, prepend_char);
  } else {
    throw runtime_error("Trouble reading file:"+filename);
  }
}

inline void ReadValFromXMLFile (const char* filename, Val& v,
				int options = XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, // best option for invertibility 
				ArrayDisposition_e arr_disp = AS_NUMERIC,
				char prepend_char=XML_PREPEND_CHAR)
{ ReadValFromXMLFile(string(filename), v, options, arr_disp, prepend_char); }


// Read XML from a string and turn it into a dictionary: most invertible options
inline void ReadValFromXMLString (const char* xml_string, Val& v,
				  int options = XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, // best option for invertibility 
				  ArrayDisposition_e arr_disp = AS_NUMERIC,
				  char prepend_char=XML_PREPEND_CHAR)
{
  XMLLoader sv(xml_string, options, arr_disp, prepend_char, false);
  sv.expectXML(v);
}


inline void ReadValFromXMLString (const string& xml_string, Val& v,
				  int options = XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, // best option for invertibility 
				  ArrayDisposition_e arr_disp = AS_NUMERIC,
				  char prepend_char=XML_PREPEND_CHAR)
{ ReadValFromXMLString(xml_string.c_str(), v, options, arr_disp, prepend_char);}


// Convert the given XML string (a text string) to a Python dictionary.
// This uses the most common options that tend to makes the 
// conversions fully invertible.
inline Val ConvertFromXML (const string& given_xml_string) 
{
  Val return_value; // RVO
  istringstream stream_thing(given_xml_string, istringstream::in);
  ReadValFromXMLStream(stream_thing, return_value);
  return return_value;
}


PTOOLS_END_NAMESPACE




#define XMLLOADER_H_
#endif // XMLLOADER_H_
