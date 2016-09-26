
#include "ocport.h" // handle stream stuff for us

#define MIDAS_COMPILER_TEMPLATES
#include "opalutils.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

using namespace std;


// Test all OpalValue type numbers to make sure we can read them
void numberTest (bool use_string=true)
{
  cout << "NUMBER TEST" << use_string << endl;
  const char* input[] = { "", "Expected numeric digit or '.' for number", "",
		    "0", "0",                    "l",
		    " 0", "0",                   "l",
		    " +0", "0",                  "l",
		    " + 0", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n   +\n---^\nExpected numeric digit or '.' for number", "l",
		    " -0", "0",                                         "l",
		    " - 0", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n   -\n---^\nExpected numeric digit or '.' for number", "l",
		    "B:1", "1", "s",
		    "I:1", "1", "i",
		    "L:1", "1", "l",
		    "X:1", "1", "x",
		    "F:1", "1.0", "f",
		    "D:1", "1.0", "d",
		    "CF:1", "(1+0j)", "F",
		    "CD:1", "(1+0j)", "D",
		    "UB:1", "1",      "S",
		    "UI:1", "1",      "I",
		    "UL:1", "1",      "L",
		    "UX:1", "1",      "X",
		    "CB:1", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  CB\n---^\nUnknown Numeric Tag:'CB'", "",
		    "1.", "1.0", "d",
		    "1.0", "1.0", "d",
		    "1e6", "1000000.0", "d",
		    "1e-6", "1e-06", "d",
		    "1000000", "1000000", "l",
		    "1000000000000", "1000000000000", "x", 
		    "1000000000000.0", "1000000000000.0", "d",
		    "D:1000000000000", "1000000000000.0", "d",
		    "123456789123456789123456789", "1.234567891234568e+26", "d", 
		    "-1000000", "-1000000", "l",
		    "-1000000000000", "-1000000000000", "x", 
		    "-1000000000000.0", "-1000000000000.0", "d", 
		    "D:-1000000000000", "-1000000000000.0", "d",
		    "-123456789123456789123456789", "-1.234567891234568e+26", "d", 
		    "DU:5", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  DU\n---^\nUnknown Numeric Tag:'DU'", "",
		    "DUr:5", "5.0", "d",
		    "T:", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  T:\n---^\nMalformed time tag:''", "",
		    "T:1950", "'1950'", "a",
		    "BIT:1", "True", "b",
		    "BIT:0", "False", "b",
		    "BIT:208234768", "True", "b",
		    0 };

  for (const char** start = input; *start!=0; start+=3) {
    const char* in = *start;
    const char* expected = *(start+1);
    char expected_tag = **(start+2);
    
    string ex;
    bool caught_exception = false;
    cout << "Looking at:" << in << ":" << endl;
    Val n;
    try { 
      if (use_string) {
	OpalReader opr(in);
	opr.expectNumber(n);
      } else {
	const string sin(in);
	istringstream is(sin);
	StreamOpalReader osr(is);
	osr.expectNumber(n);
      }
    } catch (const exception& e) {
      caught_exception = true;
      ex = string(e.what());
    }
    // Check error output for exception
    if (caught_exception) {
      cout << ex << endl;
      if (ex!=expected) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    } 
    // Check error output for normal
    else {
      string s;
      char got_tag = n.tag;
      s = got_tag;
      cout << "Got:" << s << ":";
      n.prettyPrint(cout);
      cout << endl;
      if (expected!=Stringize(n)) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
      if (expected_tag != got_tag) {
	cerr << "ERROR!!! Expected tag " << expected_tag << endl;
	exit(1);
      }
    }
    
  }
  
}


// Test all OpalValue type strings to make sure we can read them
void stringTest (bool use_string=true)
{
  cout << "STRING TEST" << use_string << endl;
  const char* input[] = { "", "A string needs to start with \"",
		    " \"", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n   \"\n---^\nUnexpected EOF inside of string",
		    " \"\"", "",
		    "\"1", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  \"1\n---^\nUnexpected EOF inside of string",
		    "\"1\"", "1",
		    "\"12", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  \"12\n----^\nUnexpected EOF inside of string",
		    "\"12\"", "12",
		    "\"\n\"", "\n",
		    "\"\\n\"", "\n",
		    "\"\\\"\"", "\"",
		    0 };

  for (const char** start = input; *start!=0; start+=2) {
    const char* in = *start;
    const char* expected = *(start+1);
    
    string ex;
    bool caught_exception = false;
    cout << "Looking at:" << in << ":" << endl;
    Str s;
    try { 
      if (use_string) {
	OpalReader opr(in);
	opr.expectStr(s);
      } else {
	const string sin(in);
	istringstream is(sin);
	StreamOpalReader osr(is);
	osr.expectStr(s);
      }
    } catch (const exception& e) {
      caught_exception = true;
      ex = string(e.what());
    }
    // Check error output for exception
    if (caught_exception) {
      cout << ex << endl;
      if (ex!=expected) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    } 
    // Check error output for normal
    else {
      cout << "Got:" << s << ":" << endl;
      if (expected!=s) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    }
    
  }
  
}


// Test all OpalValue type strings to make sure we can read them
void vectorTest (bool use_string=true)
{
  cout << "VECTOR TEST" << use_string << endl;
  const char* input[] = { "", "Expected:'<', but saw 'EOF' on input", "",
		    "<1,2,3>", "array([1,2,3], 'i')", "l",
		    // "<1,2,3>", "array([1 2 3])", "i",
		    "D:<1,2,3>", "array([1.0,2.0,3.0], 'd')", "d",
		    "<>", "array([], 'd')", "d",
		    "<", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  <\n--^\nExpected numeric digit or '.' for number", "",
		    "<1", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  <1\n---^\nExpected numeric digit or '.' for number", "",
		    "<1,", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  <1,\n----^\nExpected numeric digit or '.' for number", "",
		    "B:<1>", "array([1], '1')", "s",
		    "I:<1>", "array([1], 's')", "i",
		    "L:<1>", "array([1], 'i')", "l",
		    "X:<1>", "array([1], 'l')", "x",
		    "F:<1>", "array([1.0], 'f')", "f",
		    "D:<1>", "array([1.0], 'd')", "d",
		    "UB:<1>", "array([1], 'b')", "S",
		    "UI:<1>", "array([1], 'w')", "I",
		    "UL:<1>", "array([1], 'u')", "L",
		    "UX:<1>", "array([1], 'l')", "X",
		    "CF:<1>", "array([(1+0j)], 'F')", "F",
		    "CD:<1>", "array([(1+0j)], 'D')", "D",
		    "I:<100, 127, 128, 255, 256>", "array([100,127,128,255,256], 's')", "i",
		    "I:<100, 127, 128, 255, 256,>", "array([100,127,128,255,256], 's')", "i",
		    "<100,127,100000000,200>", "array([100,127,100000000,200], 'i')", "l", 
		    0 };

  for (const char** start = input; *start!=0; start+=3) {
    const char* in = *start;
    const char* expected = *(start+1);
    char expected_tag = **(start+2);
    
    string ex;
    bool caught_exception = false;
    cout << "Looking at:" << in << ":" << endl;
    Val t;
    try { 
      if (use_string) {
	OpalReader opr(in);
	opr.expectNumericArray(t);
      } else {
	const string sin(in);
	istringstream is(sin);
	StreamOpalReader osr(is);
	osr.expectNumericArray(t);
      }
    } catch (const exception& e) {
      caught_exception = true;
      ex = string(e.what());
    }
    // Check error output for exception
    if (caught_exception) {
      cout << ex << endl;
      if (ex!=expected) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    } 
    // Check error output for normal
    else {
      cout << "Got:"<< t.subtype <<":" << t << ":" << endl;
      if (expected!=Stringize(t)) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
      if (t.tag != 'n' || expected_tag != t.subtype) {
	cerr << "ERROR!!! Expected " << t.tag << " " << t.subtype << endl;
	exit(1);
      }
    }
    
  }
  
}


// Test all OpalValue type strings to make sure we can read them
void tabTest (bool use_string=true)
{
  cout << "TAB TEST" << use_string << endl;
  const char* input[] = { "", "Expected:'{', but saw 'EOF' on input",
		    "{}", "{}",
		    " { } ", "{}",
		    "{", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  {\n--^\nExpected a key and saw EOF",
		    "}", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  }\n--^\nExpected:'{', but saw '}' on input",
		    "{ \"a\" = 1 }", "{'a': 1}",
		    "{ \"a\"= { } }", "{'a': {}}",
		    "{ a = 1 }", "{'a': 1}",
		    "{ a = 1,}", "{'a': 1}",
		    "{ a = 1, }", "{'a': 1}",
		    "{ \"a\"= { } ", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  { \"a\"= { }\n-----------^\nExpecting a '}' or ',' for table",
		    "{a1 = 1} ", "{'a1': 1}",
		    "{a_1=100}", "{'a_1': 100}",
		    "{111=100}", "{'111': 100}",
		    "{ a:1}", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  { a\n----^\nUnknown Numeric Tag:'a'",
                    "{PZ=CF:(0.12015177, -0.95675814), a=1}", "{'PZ': (0.1201518-0.9567581j), 'a': 1}",
		    "{ DATA=MV(<12.3, 17>, I:<1,1,1>) }", "{'DATA': [array([12.3,17.0], 'd'), array([1,1,1], 's')]}",

		    " {1,2,3}", "{0: 1, 1: 2, 2: 3}",
		    " {1,2}", "{0: 1, 1: 2}",
		    " {1}", "{0: 1}",
		    " {}", "{}",

		    " {1.0,2,\"three\"}", "{0: 1.0, 1: 2, 2: 'three'}",

		    0 };

  for (const char** start = input; *start!=0; start+=2) {
    const char* in = *start;
    const char* expected = *(start+1);
    
    string ex;
    bool caught_exception = false;
    cout << "Looking at:" << in << ":" << endl;
    Tab t;
    try { 
      if (use_string) {
	OpalReader opr(in);
	opr.expectTab(t);
      } else {
	const string sin(in);
	istringstream is(sin);
	StreamOpalReader osr(is);
	osr.expectTab(t);
      }
    } catch (const exception& e) {
      caught_exception = true;
      ex = string(e.what());
    }
    // Check error output for exception
    if (caught_exception) {
      cout << ex << endl;
      if (ex!=expected) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    } 
    // Check error output for normal
    else {
      cout << "Got:" << t << ":" << endl;
      if (expected!=Stringize(t)) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    }
    
  }
  
}


static const char* hdr = ""
"{"
"    ATTRIBUTE_PACKET={ },"
"    EVENTS={"
"        ASSOCIATIONS={"
"            \"0\"=\"ALL_EVENTS, ALL_TRACKS\""
"        },"
"        KEYWORDS={ },"
"        TRACKS={"
"            \"0\"={"
"                EPOCH=T:1976:04:22::10:09:08.76543210000,"
"                FIELDS={"
"                    \"0\"={"
"                        FORMAT=\"DUR\","
"                        NAME=\"TOA\","
"                        TIME_INTERPRETATION=\"First Sample\","
"                        UNITS=\"sec\""
"                    },"
"                    \"1\"={"
"                        FORMAT=\"L\","
"                        NAME=\"Length\","
"                        TIME_INTERPRETATION=\"None\","
"                        UNITS=\"\""
"                    }"
"                },"
"                FIELD_INTERPRETATION={ },"
"                KEYWORDS={ },"
"                LENGTH=UX:2,"
"                NAME=\"Track 0\""
"            }"
"        }"
"    },"
"    GRANULARITY=UL:10,"
"    KEYWORDS={"
"        FILENAME=\"r_int\","
"        CONTENTS=\"Integer data from a Ramp\","
"        SPECIAL_R_INT=1"
"    },"
"    MACHINE_REP=\"IEEE\","
"    NAME=\"r_int\","
"    TIME={"
"        DELTA=D:1.7,"
"        KEYWORDS={ },"
"        LENGTH=UX:10,"
"        NAME=\"Time\","
"        START=D:19.3,"
"        UNITS=\"s\""
"    },"
"    TRACKS={"
"        \"0\"={"
"            AXES={"
"                \"0\"=@\"TIME\""
"            },"
"            FORMAT=\"I\","
"            KEYWORDS={"
"                TRACK_FILENAME=\"r_int_track_0\""
"            },"
"            NAME=\"Track 0\","
"            UNITS=\"\""
"        }"
"    }"
"}";


void hdrTest (bool use_string=true) 
{
  cout << "HDR test" << use_string << endl;
  Tab t;
  if (use_string) {
    OpalReader opr(hdr);
    opr.expectTab(t);
  } else {
    const string sin(hdr);
    istringstream is(sin);
    StreamOpalReader osr(is);
    osr.expectTab(t);
  }
  t.prettyPrint(cout);
}


void commentTest (bool use_string=true)
{
  cout << "COMMENT TEST" << use_string << endl;
  const char* input[] = {
		    "1// comment starts here\n", "1",
		    " // comment here\n1", "1",
		    "//\n1", "1",
		    "1//\n0", "1", // comment becomes ws
		    "1/1", "1",
		    "111/\n0", "111",
		    "\"abcd//hello\"", "'abcd//hello'",
		    "{ // This is a table\n a = 1 }", "{'a': 1}",
		    "{ a = // what?\n 100 }", "{'a': 100}",
		    "{ a = / what? 100 }", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  { a = / \n---------^\nExpected numeric digit or '.' for number",
		    "{ a = / what?\n 100 }", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  { a = / \n---------^\nExpected numeric digit or '.' for number",

		    " {1,2,3}", "[1, 2, 3]",
		    " {1,2}", "[1, 2]",
		    " {1}", "[1]",
		    " {}", "{}",

		    "{ I:1} ", "[1]",
		    "{ B:1} ", "[1]",
		    "{I:1,B:2,DUR:3}", "[1, 2, 3.0]",

		    " {1.0,2,\"three\"}", "[1.0, 2, 'three']",

		    0 };

  for (const char** start = input; *start!=0; start+=2) {
    const char* in = *start;
    const char* expected = *(start+1);
    
    string ex;
    bool caught_exception = false;
    cout << "Looking at:" << in << ":" << endl;
    Val v;
    try { 
      if (use_string) {
	OpalReader opr(in);
	opr.expectAnything(v);
      } else {
      	const string sin(in);
	istringstream is(sin);
	StreamOpalReader osr(is);
	osr.expectAnything(v);
      }
    } catch (const exception& e) {
      caught_exception = true;
      ex = string(e.what());
    }
    // Check error output for exception
    if (caught_exception) {
      cout << ex << endl;
      if (ex!=expected) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    } 
    // Check error output for normal
    else {
      cout << "Got:" << v << ":" << endl;
      if (expected!=Stringize(v)) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    }
    
  }

}


void MVTest(bool use_string=true)
{
  cout << "MV TEST" << use_string << endl;
  const char* input[] = { "MV(<1>, <2>)", "[array([1], 'i'), array([2], 'i')]",
		    "MV(D:<1>, D:<2>)", "[array([1.0], 'd'), array([2.0], 'd')]",
		    "MV()", "[]",
		    "MV(I:<1,2>, D:<1,2,3,4>)", "[array([1,2], 's'), array([1.0,2.0,3.0,4.0], 'd')]",
		    "MV()", "[]",
		    "MV(<>)", "[array([], 'd')]",
		    "MV(", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  MV(\n----^\nExpected:'<', but saw 'EOF' on input",
		    "MV[", "****Syntax Error on line:1 Last 1 line of input (lines 1-1) shown below****\n  MV[\n----^\nExpected:'(', but saw '[' on input",
		    " MV(   )", "[]",
		    " MV (   )", "[]",
		    0 };

  for (const char** start = input; *start!=0; start+=2) {
    const char* in = *start;
    const char* expected = *(start+1);
    
    string ex;
    bool caught_exception = false;
    cout << "Looking at:" << in << ":" << endl;
    Val v;
    try {
      if (use_string) {
	OpalReader opr(in);
	opr.expectAnything(v);
      } else {
	const string sin(in);
	istringstream is(sin);
	StreamOpalReader osr(is);
	osr.expectAnything(v);
      }
    } catch (const exception& e) {
      caught_exception = true;
      ex = string(e.what());
    }
    // Check error output for exception
    if (caught_exception) {
      cout << ex << endl;
      if (ex!=expected) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    } 
    // Check error output for normal
    else {
      cout << "Got:" << v << ":" << endl;
      if (expected!=Stringize(v)) {
	cerr << "ERROR!!! Expected " << expected << endl;
	exit(1);
      }
    }
  }
}


int main ()
{
  numberTest();
  stringTest();
  vectorTest();
  tabTest();
  hdrTest();
  commentTest();
  MVTest();

  numberTest(false);
  stringTest(false);
  vectorTest(false);
  tabTest(false);
  hdrTest(false);
  commentTest(false);
  MVTest(false);

  cout << "!!!!!!!!!!!!!!!!If you see this, we are OKAY!!!!!!!!!!!!!!" << endl;
  return 0;
}
