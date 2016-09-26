
#include "jsonprint.h"
#include <strstream>
#include "jsonreader.h"
// Test out the JSON reader and pretty printer

int main ()
{
  const char* tests[] = { "None", "True", "False", "{ }", "123456789", 
			  "{'true':True, 'false':False, 'none':None, 'a':1, 'b':2.2, 'c':'three', 'd':'123\\'\r\n\t\\\"', 'list':[0, 1.1, 'two'], 'nest':{'a':2000, 'b':300.3333, 'c':(1+2j), 'd':array([1,2,3], 'd'), 'e':array([1,2,3], 'D'), 'f':array([1,2,3],'F'), } }",
			  "{ }", "[]", "[1]","[1.0, 2, 'three']",
			  
		    0
  };
  for (const char** t = &tests[0]; *t!=0; t++) {
    const char* str = *t;
    Val v = Eval(str);

    cout << v << endl;
    v.prettyPrint(cout);

    JSONPrint(v, cout);
    JSONPrint(v, cout);
    JSONPrint(v, cout,4);
    JSONPrint(v, cout,0,false);
    JSONPrint(v, cout,2, true, 2);
    
    ostringstream os;
    JSONPrint(v, os, 2, true, 2);
    string s = os.str();
    cout << s << endl;
    istringstream is(s);
    
    Val vv;
    ReadValFromJSONStream(is, vv);
    vv.prettyPrint(cout);
  }

  {
    istringstream is("{ \"a\":\"abcde\\u1234abc\" }");
    Val vvv;
    ReadValFromJSONStream(is, vvv);
    vvv.prettyPrint(cout);
  }


}
