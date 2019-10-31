
#include "xmldumper.h"
#include "ocval.h"
#include "ocvalreader.h"

// Simple test with no attrs
void grindNoAttrs (XMLDumper& xd)
{
  const char* strs[] = { 
    "None", "1", "1.0", "1L", "(1.0-1.0j)", // Primitive types first
    "{ }", "{ 'html': 1 }",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':['chap1', 'chap2']}}",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':[{'p1':{'a':1, 'b':2, 'text':'chapter1'}}, 'chap2']}}",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':[{'p1':{'a':1, 'b':2, 'text':'chapter1'}}, {'text':[1,2,3]}]}}",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':[{'p1':{'a':1, 'b':2, 'text':'chapter1'}}, [1,2,3]]}}",
    "[1,2,3]",
    "[{'a':1}]",
    "{ 'empty': None, 'alsoempty': {} }",
    "[ [0,1,2], [2,3,4] ]",
    0
  };

  for (const char** s=strs; *s!=0; s++) {
    Val v = Eval(*s);
    
    cout << "As normal:";
    v.prettyPrint(cout);  // This alphabetizes so we have expected output
    cout << endl;
    
    cout << "As XML:" << endl;
    xd.XMLDumpKeyValue("top",v);
    cout << endl;
  }
}

// Quick test using attrs of XML
void grindAttrs (XMLDumper& xd)
{
  const char* strs[] = { 
    "{ 'book': 1, '__attrs__': { 'date':100, 'time':'May 16th, 2010' } }",
    "{ 'simple': 100, 'complex':{ 'a':1 }, '_plug':'in', '__attrs__': { 'date':100, 'time':'May 16th, 2010' } }",
    0
  };

  for (const char** s=strs; *s!=0; s++) {
    Val v = Eval(*s);
    
    cout << "As normal:";
    v.prettyPrint(cout);  // This alphabetizes so we have expected output
    cout << endl;
    
    cout << "As XML:" << endl;
    xd.XMLDumpKeyValue("top",v);
    cout << endl;
  }
}

// Quick test using attrs of XML
void grindESC (XMLDumper& xd)
{
  const char* strs[] = { 
    "{ '123': 'illegal because starts with number' }",
    "{ 'a\"': 'illegal because of quote in name' }",
    "{ 'a<': 'illegal because has <' }",
    "{ 'a>': 'illegal because has >' }",
    "{ 'a&': 'illegal because has &' }",
    "{ 'a\\'': 'illegal because has \\'' }",
    "{ '': 'empty'}",
    "{ 'ok': 'special chars: < > & \" \\' '}",
    "{ 'ok': ['special chars: < > & \" \\' ', '\x11']}", // Note that special unprintable characters gets passed as is ONLY weirdness is escape chars
    "{ '__attrs__': { 'a':1, 'b':2 }, '__content__': 'something' }", // handle special __content__
    "{ 'book': [ {'__attrs__':{'da':1}, '__content__':'junk' } ] }",
    "{ 'ns:book': { 'ns:a1':5 } }", /// ya, we need better namespace support
    0
  };

  for (const char** s=strs; *s!=0; s++) {
    Val v = Eval(*s);
    
    cout << "As normal:";
    v.prettyPrint(cout);  // This alphabetizes so we have expected output
    cout << endl;

    cout << "As XML:" << endl;
    try {
      xd.XMLDumpKeyValue("root",v);
    } catch (const runtime_error& e) {
      cout << "Saw Error:" << e.what() << endl;
      cout << " ... continuing..." << endl;
    }
    cout << endl;
  }
}

template <class T>
Val helpMeCore (T*, XMLDumper& xd)
{
  Val v = Array<T>(10);
  Array<T>& a = v;
  a.fill(10);
  for (int_4 ii=0; ii<10; ii++) {
    Val r = real_8(real_8(ii)*123.456789012345678-500.0); 
    Val th = T(0);
    if (th.tag=='x') { r = int_4(r); }
    if (th.tag=='X') { r = int_u4(r); }
    a[ii] = r;
  }
  
  cout << "As normal:";
  v.prettyPrint(cout);  // This alphabetizes so we have expected output
  cout << endl;

  cout << "As XML:" << endl;
  xd.XMLDumpKeyValue("top",v);
  cout << endl;
  return v;
}

template <class T>
Val helpMe (T*, XMLDumper& xd)
{
  Val cc = helpMeCore((T*)0, xd);
  Val v = Tab();
  v["root"] = cc;

  //cout << "As normal:" << v << endl;
  cout << "As normal:";
  v.prettyPrint(cout);  // This alphabetizes so we have expected output
  cout << endl;

  cout << "As XML:" << endl;
  xd.XMLDumpKeyValue("up",v);
  cout << endl;

  return cc;
}


// Plain Old Data (POD)
void grindPOD (XMLDumper& xd)
{
  helpMe((int_1*)0, xd);
  helpMe((int_u1*)0, xd);
  helpMe((int_2*)0, xd);
  helpMe((int_u2*)0, xd);
  Val v1 = helpMe((int_4*)0, xd);
  helpMe((int_u4*)0, xd);
  helpMe((int_8*)0, xd);
  // helpMe((int_u8*)0, xd);
  helpMe((real_4*)0, xd);
  Val v2 = helpMe((real_8*)0, xd);
  helpMe((complex_8*)0, xd);
  helpMe((complex_16*)0, xd);
  
  Tab t;
  t["int_4"] = v1;
  t["real_4"] = v2;
  Val vv = Tab();
  vv["data"] = t;
  cout << "As normal:";
  vv.prettyPrint(cout);
  cout << endl;
  cout << "As XML:" << endl;
  xd.XMLDumpKeyValue("top",vv);
  cout << endl;

  // make sure we can put in lists
  Arr a;
  a.append(v1);
  a.append(v2);
  Val v = Tab();
  v["data"] = a;  
  cout << "As normal:";
  v.prettyPrint(cout);
  cout << endl;
  cout << "As XML:" << endl;
  xd.XMLDumpKeyValue("top",v);
  cout << endl;

  string s = "{\n"
             "   'top':{\n"
             "        'int_4':[\n"
             "             array([1,2,3], 'i'),\n"
             "             array([1.0,2.0,3.0], 'd')\n"
             "        ]\n"
             "    }\n"
             "}\n";
  Val vvv = Eval(s);
  //cout << "As normal:" << vvv << endl;
  cout << "As normal:";
  vvv.prettyPrint(cout);  // This alphabetizes so we have expected output
  cout << endl;

  vvv.prettyPrint(cout);
  cout << "As XML:" << endl;
  xd.XMLDumpValue(vvv);
  cout << endl;

}

Val listify (const Val& v)
{
  Val result;
  if (v.tag=='o' || v.tag=='t') {
    result = ((v.tag == 'o') ? Val(OTab()) : Val(Tab()));
    for (It ii(v); ii();) {
      const Val& key = ii.key();
      Val& value = ii.value();
      result[key] = listify(value);
    }
  }
  else if (v.tag=='u' || (v.tag=='n' && v.subtype=='Z')) {
    result = (v.tag == 'u') ? Val(Tup()) : Val(Arr());
    for (size_t ii=0; ii<v.length(); ii++) {
      result.append(listify(v(ii)));
    }
  }
  else if (v.tag=='n' && v.subtype!='Z') {
    AsList(v, result, true);
  } 
  else {
    result = v;
  }
  return result;
}


#include "xmlloader.h"
void req (XMLDumper& xd, ArrayDisposition_e arr_disp)
{
  const char* strs[] = { 

    "[]",
    "[1]",
    "[1,2]",
    "[ [] ] ",
    "[ [1] ] ",
    "[ [1,2] ] ",
    "[ {} ] ",
    "[ {'a':1} ] ",
    "[ {'a':1, 'b':2} ] ",
    "[ {'a':1, 'b':2}, {} ] ",
    "[ {'a':1, 'b':2}, {'a':1, 'b':2} ] ",



    "{ 'top': { 'duh' : [ { 'ds': 1, 'tcai': { 'stuff': 0 }}, {'ds': 2, 'tcai': { 'stuff': 666 }} ] } }",

    "[ { 'ds': 1, 'tcai': { 'stuff': 0 }}, {'ds': 2, 'tcai': { 'stuff': 666 } } ]",

    "[ { 'ds': [ [ { 'bd':1, 'bo':2 }, { 'bd':3, 'bo':4 } ] ], 'tcai': { 'stuff': 0 }}, {'ds': [ [ { 'bd':111, 'bo':2222 }, { 'bd':333, 'bo':444 } ] ], 'tcai': { 'stuff': 0 } } ]",

    "{ 'empty': [] }",
    "{ 'empty': [1] }",
    "{ 'empty': [1,2] }",
    "{ 'empty': [1,2, {}] }",
    "{ 'empty': [1,2, {'a':1}] }",
    "{ 'empty': [1,2, {'a':1, 'b':2}] }",
    "{ 'empty': [{}] }",
    "{ 'empty': [{}, 1] }",
    "{ 'empty': [{}, 1, 2] }",

    "{ 'empty': [{'a':1}] }",
    "{ 'empty': [{'a':1}, 1] }",
    "{ 'empty': [{'a':1}, 1, 2] }",


    "{ 'empty': [{'a':1, 'b':2}] }",
    "{ 'empty': {} }",
    "{ 'empty': [ 1 ] }",

    "{ 'empty': [ [  ] ] }",
    "{ 'empty': [ [1 ] ] }",

    "{ 'empty': [ [{}] ] }",
    "{ 'empty': [ [{'a':1}] ] }",
    "{ 'empty': [ [{'a':1, 'b':2}] ] }",

    "{ 'empty': [ [[]] ] }",
    "{ 'empty': [ [[{}]] ] }",
    "{ 'empty': [ [[{'a':1}]] ] }",
    "{ 'empty': [ [[{'a':1, 'b':2}]] ] }",

    "{ 'empty': [ [{},1] ] }",
    "{ 'empty': [ [{'a':1},1] ] }",
    "{ 'empty': [ [{'a':1, 'b':2},1] ] }",

    "{ 'empty': [ [{},1,2] ] }",
    "{ 'empty': [ [{'a':1},1,2] ] }",
    "{ 'empty': [ [{'a':1, 'b':2},1,2] ] }",

    "{ 'empty': [ [1,2] ] }",
    "{ 'empty': [ [1,2, {}] ] }",
    "{ 'empty': [ [1,2, {'a':1}] ] }",
    "{ 'empty': [ [1,2, {'a':1, 'b':2}] ] }",

    "{ 'empty': [ [{'a':1},2, {'a':1, 'b':2}] ] }",
    "{ 'empty': [ [{'a':1},2, {'a':1, 'b':2}] ] }",

    " [ [ { 'ds': [ [ { 'ts':1, 'td':2 }, { 'ts':4, 'td':5 } ] ], 'tcai': {'hey':1} } ] ]",

    "[ { 'ds': [ [ {'filename':'something', 'timedelta':1, 'empty':[], 'carl':[1] }, {'filename':'else', 'timedelta':2}] ], 'ot': {'a':1, 'b':2 } } ]",

	       "{ 'a':1, 'SOMETHING': "
"[ { 'ds': [ [ {'filename':'something', 'timedelta':1}, {'filename':'else', 'timedelta':2}], [ {'filename':'something', 'timedelta':1}, {'filename':'else', 'timedelta':2}]], 'ot': {'a':1, 'b':2 } } ]"
    " } ",
    "{ 'carl':1, 'empty':{} }",
    "{ 'carl':1, 'empty':[] }",


    "{ 'empty': array([], 'i') }",
    "{ 'empty': [array([], 'i')]}",
    "{ 'empty': [array([1], 'i')] }",
    "{ 'empty': [array([1,2,3], 'i')] }",
       
    "{ 'empty': [array([], 'i'),1]}",
    "{ 'empty': [array([1], 'i'),1] }",
    "{ 'empty': [array([1,2,3], 'i'),1] }",

    "{ 'empty': [array([], 'i'),1,2]}",
    "{ 'empty': [array([1], 'i'),1,2] }",
    "{ 'empty': [array([1,2,3], 'i'),1,2] }",

    
    "{ 'empty': [1, array([], 'i')] }",
    "{ 'empty': [1, array([1], 'i')] }",
    "{ 'empty': [1, array([1,2], 'i')] }",

    "{ 'empty': [1,2] }",
    "{ 'empty': [1,2, array([], 'i')] }",
    "{ 'empty': [1,2, array([1], 'i')] }",
    "{ 'empty': [1,2, array([1,2], 'i')] }",

    "{ 'empty': [1,2, {'a': array([], 'i')}] }",
    "{ 'empty': [1,2, {'a': array([1], 'i')}] }",
    "{ 'empty': [1,2, {'a': array([1,2,3], 'i')}] }",
    "{ 'empty': [1,2, {'a':array([],'i'), 'b':2}] }",
    "{ 'empty': [1,2, {'a':array([1],'i'), 'b':2}] }",
    "{ 'empty': [1,2, {'a':array([1,2,3],'i'), 'b':2}] }",
    "{ 'empty': [{'a':array([], 'i')}] }",
    "{ 'empty': [{'a':array([1], 'i')}] }",
    "{ 'empty': [{'a':array([1,2,3], 'i')}] }",
    "{ 'empty': [{'a':array([], 'i')},1] }",
    "{ 'empty': [{'a':array([1], 'i')},1] }",
    "{ 'empty': [{'a':array([1,2,3], 'i')},1] }",
    "{ 'empty': [{'a':array([], 'i')},1,2] }",
    "{ 'empty': [{'a':array([1], 'i')},1,2] }",
    "{ 'empty': [{'a':array([1,2,3], 'i')},1,2] }",


    "{ 'empty': [ [  ] ] }",
    "{ 'empty': [ [ array([],'i') ] ] }",
    "{ 'empty': [ [ array([1],'i') ] ] }",
    "{ 'empty': [ [ array([1,2,3],'i') ] ] }",

    "{ 'empty': [ [{}] ] }",
    "{ 'empty': [ [{'a':1}] ] }",
    "{ 'empty': [ [{'a':1, 'b':2}] ] }",

    "{ 'empty': [ [[array([],'i')]] ] }",
    "{ 'empty': [ [[array([1],'i')]] ] }",
    "{ 'empty': [ [[array([1,2,3],'i')]] ] }",
    "{ 'empty': [ [[array([1,2,3],'i')]] ] }",

    "{ 'data': array([(1+2j), (3+4j)], 'F') }",

    0
  };

  for (const char** s=strs; *s!=0; s++) {
    Val v = Eval(*s);

    cout << "-----------As normal:" << endl;
    v.prettyPrint(cout);
    cout << endl;
    
    cout << "As XML:" << endl;
    xd.XMLDumpKeyValue("top",v);
    cout << endl;  
    
    ostringstream os;
    XMLDumper ds(os, XML_DUMP_PRETTY| XML_STRICT_HDR, arr_disp);
    ds.XMLDumpKeyValue("top",v);
    
    string sx = os.str();
    istringstream is(sx);
    Val result;
    ReadValFromXMLStream(is, result, 
			 XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, arr_disp);
    result.prettyPrint(cout);
    
    Val other_v;
    if (arr_disp==AS_LIST) {
      other_v = listify(v);
    } else {
      other_v = v;
    }
    if (result!=other_v) { 
      cout << "********SHOULD BE SAME!!" << endl;
      result.prettyPrint(cerr);
      v.prettyPrint(cerr);
      other_v.prettyPrint(cerr);
      exit(1);
    }
    
    // Read dumping POD data slightly differently: dump pod array as data
    {
      Val result_pod;

      ostringstream oss;
      XMLDumper dss(oss, XML_DUMP_PRETTY| XML_STRICT_HDR | XML_DUMP_POD_LIST_AS_XML_LIST, arr_disp);
      dss.XMLDumpKeyValue("top",v);
      
      string ss = oss.str();
      cout << "POD DATA in XML as lists::" << ss << endl;

      istringstream iss(ss);
      ReadValFromXMLStream(iss, result_pod, XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, arr_disp);
      //cout << "POD DATA result::" << result_pod.prettyPrint(cout);
      if (result_pod!=other_v) {
	cout << "********SHOULD BE SAME!!" << result_pod << other_v << v << endl; 
	exit(1);
      }
    }

  }
}


// Simple test with no attrs
void TryOptions (XMLDumper& xd)
{
  const char* strs[] = { 
    "{ 'a':'' }",
    "{ 'a':'123' }",
    "{ 'a':123 }",
    "{ 'a': ' 123'} ",
    "{ 'a': (1+2j) }",
    "{ 'a': '(1+2j)' }",
    0
  };

  for (const char** s=strs; *s!=0; s++) {
    Val v = Eval(*s);
    
    cout << "As normal:";
    v.prettyPrint(cout);  // This alphabetizes so we have expected output
    cout << endl;

    cout << "As XML:" << endl;
    xd.XMLDumpKeyValue("top",v);
    cout << endl;
  }
}

int tests (ArrayDisposition_e arr_disp)
{  
  {
    cout << "**************OPTIONS: (no attrs): all defaults" << endl; 
    XMLDumper xd(cout);
    grindNoAttrs(xd);
  }
  {
    cout << "**************OPTIONS: (no attrs): XML_DUMP_PRETTY" << endl; 
    XMLDumper xd(cout, XML_DUMP_PRETTY);
    grindNoAttrs(xd);
  }
  {
    cout << "**************OPTIONS: (no attrs): XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS" << endl; 
    XMLDumper xd(cout, XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS);
    grindNoAttrs(xd);
  }
  {
    cout << "**************OPTIONS: (no attrs): XML_DUMP_PRETTY and indent of 1" << endl; 
    XMLDumper xd(cout, XML_DUMP_PRETTY, arr_disp, 1);
    grindNoAttrs(xd);
  }
  {
    cout << "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY and indent of 1" << endl; 
    XMLDumper xd(cout, XML_DUMP_PRETTY, arr_disp, 1);
    grindAttrs(xd);
  }
  {
    cout << "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES and indent of 1" << endl; 
    XMLDumper xd(cout, XML_DUMP_PRETTY | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES, arr_disp, 1);
    grindAttrs(xd);
  }
  {
    cout << "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES and indent of 1" << endl; 
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS |
		 XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES, arr_disp, 1);
    grindAttrs(xd);
  }
  {
    cout << "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS and indent of 1" << endl; 
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS, arr_disp, 1);
    grindAttrs(xd);
  }

  {
    cout << "**************OPTIONS (escape chars): XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES and indent of 1 and throw error" << endl; 

    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS |
		 XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES, 
		 arr_disp,
		 1, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    grindESC(xd);
  }

  {
    cout << "**************OPTIONS (strict hdr): XML_DUMP_PRETTY | XML_STRICT_HDR and indent of 1 and throw error" << endl; 

    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_STRICT_HDR,
		 arr_disp,
		 1, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    grindESC(xd);
  }

  {
    cout << "**************OPTIONS (POD data): XML_DUMP_PRETTY" << endl;
    
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY,
		 arr_disp,
		 4, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    grindPOD(xd);
  }

  {
    cout << "**************OPTIONS (POD data): XML_DUMP_PRETTY | XML_DUMP_POD_LIST_AS_XML_LIST" << endl;
    
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_DUMP_POD_LIST_AS_XML_LIST,
		 arr_disp,
		 4, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    grindPOD(xd);
  }

  {
    cout << "**************OPTIONS (request data): XML_DUMP_PRETTY " << endl;
    cout << " ... make sure what goes out comes back in!" << endl;
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY,
		 arr_disp,
		 4, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    req(xd, arr_disp);
  }

  {
    cout << "**************OPTIONS (try some other options): XML_DUMP_PRETT " << endl;
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY,
		 arr_disp,
		 4, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    TryOptions(xd);
  }

  {
    cout << "**************OPTIONS (try some other options): XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS " << endl;
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS,
		 arr_disp,
		 4, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    TryOptions(xd);
  }

  {
    cout << "**************OPTIONS (try some other options): XML_DUMP_PRETTY | XML_DUMP_STRINGS_BEST_GUESS " << endl;
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_DUMP_STRINGS_BEST_GUESS,
		 arr_disp,
		 4, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    TryOptions(xd);
  }

  {
    cout << "**************OPTIONS (try some other options): XML_DUMP_PRETTY | XML_DUMP_PREFER_EMPTY_STRINGS " << endl;
    XMLDumper xd(cout, 
		 XML_DUMP_PRETTY | XML_DUMP_PREFER_EMPTY_STRINGS,
		 arr_disp,
		 4, 
		 XML_PREPEND_CHAR, 
		 XMLDumper::THROW_ON_ERROR);
    TryOptions(xd);
  }
  return 0;
}

int main ()
{
  tests(AS_NUMERIC);  // same as AS_PYTHON_ARRAY for C++
  tests(AS_LIST);
  exit(0);
}
