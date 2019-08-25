

#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

void CopyStringToArray (const string& s, Array<char>& a)
{
  for (int ii=0; ii<int(s.length()); ii++) {
    // cout << s[ii] << endl;
    a.append(s[ii]);
  }
}

typedef void (*TRIALFUN)(ValReaderA& vr);

void mainloop (const char** input, TRIALFUN trial, bool use_streams)
{
  for (const char** in = &input[0]; *in; in++) {
    cout << "Input is " << string(*in) << endl;
    
    // Build for ValReader
    Array<char> data;
    CopyStringToArray(*in, data);
    ValReader vr(data);
    
    // Build for StreamValReader
    {
      ofstream ofs("/tmp/doo");
      for (int ii=0; ii<int(data.length()); ii++)
	ofs.put(data[ii]);
    }
    ifstream ifs("/tmp/doo");
    StreamValReader svr(ifs);
    
    // Do the "work"
    if (use_streams) {
      trial(svr);
    } else {
      trial(vr);
    }
  }
  
}

// SIngle trial for numbers
void trialNumbers (ValReaderA& vr)
{
  // Try it
  Val v;
  try {
    vr.expectNumber(v);
  } catch (const logic_error& le) {
    cout << "EXCEPTION:" << le.what() << endl;
  }
  cout << "Output is " << v << " type = " << v.tag << endl;
}

// Numbers strings
void TestNumbers (bool use_streams)
{
  const char* input[] = { 
    "",
    "0", " 0", "  0 ", " 123456", "123456", "123456789",
    "+0", " +0", " -0", "-0", " +-0", "+-0",
    "-1", "-1 ", " -1", " - 1", "+1", "+1 ", " +1", " + 1",
    "1e6", " 1e6", " 1 e 6", "1 e 6", "1e 6", "1 e6", 
    "1.", " 1.", " -1.", " +1. ", "-1.", "+1.",
    ".", " . ", " .", // syntax error
    "0.", ".09", " .01", " . 0", " 1 . ", 
    "1.e", " 1. e", " 1.e-", "1.e-1", 
    "1.e+", " 1.e +", "1e+1",
    "(1+1j)", " ( 1 + 1j)", " (1+1j",
    "1l", "1 l", " 4000000000L",

    // Try a bunch of big numbers that don't fit into anormal int_4
    "100000000000",  // should become a int_8
    "-100000000000",  // should become a int_8
    "10000000000000000000000000000",  // should become a int_un
    "-10000000000000000000000000000",  // should become a int_n
    
    "9223372036854775807", // barely fits into an int_8
    "9223372036854775807.0", // the . makes it a real
    "9223372036854775808", // barely DOES n't fit into an int_8
    "-9223372036854775808", // barely fits into an int_8
    "-9223372036854775809", // barely DOESN'T fits into an int_8

    "18446744073709551615", // Barely fits in int_u8
    "18446744073709551616", // Barely DOESN'T fits in int_u8, int_un

    0
  };

  mainloop(input, trialNumbers, use_streams);
}


// Single trial for strings
void trialStrs (ValReaderA& vr)
{
  Str v;
  try {
    vr.expectStr(v);
  } catch (const logic_error& le) {
    cout << "EXCEPTION:" << le.what() << endl;
  }
  
  cout << "Output is "<< v << endl;
}
   
// Main strs
void TestStrs (bool use_streams)
{
  const char* input[] = { "''", " ''", "'' ", " '' ",
		       "'", " '", " ' '",
		       "\"\"",
		       "'a'", " 'a'", " 'a' ",
                       "'\\\"'", "'\\''", " 'some \\'quotes\\''",
		       "'This is a normal string'",
		       "'Hello \\n'",
		       0 };

  mainloop(input, trialStrs, use_streams);
}
 

// Single trial Arrs
void trialArr (ValReaderA& vr)
{
  Arr v;
  try {
    vr.expectArr(v);
  } catch (const logic_error& le) {
    cout << "EXCEPTION:" << le.what() << endl;
  }
  
  cout << "Output is " << v << endl;    
}

// Main Arrs
void TestArrs (bool uses_streams)
{
  const char* input[] = { "[]", "[ 1 ]", "[1]", "[ 1]", "[ 1 ] ", "[ 1,2]",
		       "[1,2]", "[1, 2 ]", "[ 1, 2]", "[1,2,3]", 
		       "[1., (1+2j), 3.0,3.1]", "[1., 2., 3., 3., 4., 5]",
		       " [[]]", " [ [] ]", " [[ ]]", "[1,2,[3,4,5], 6,[7]]",
		       " ]", "[",
		    // Commas at end 
		    "[1,2,3,]",
		    "[1,]",
		    "[,]",

                    // Pretty Printed table across lines
                    " \n"
		    " [                              \n" 
		    "     'a',                   \n"
                    "     {                   \n"
                    "       'c': None          \n"
                    "     },                   \n"
                    "     'd'                  \n"
                    " ]                              \n",

                    // Pretty Printed table across lines with comments
		    " \n"
                    " \n"
		    " [  # This is the Fred Table   \n" 
		    "     'a',               # Fred's name    \n"
                    "     {                   \n"
                    "       'c': None        # Fred's table  \n"
                    "     },                   \n"
                    "     'd'                # fred's #   \n"
                    " ]                              \n",

		       0 };

  mainloop(input, trialArr, uses_streams);
}


// Single trial Numeric Array 
void trialNumericArrays (ValReaderA& vr)
{
  Val v;
  try {
    vr.expectNumericArray(v);
  } catch (const logic_error& le) {
    cout << "EXCEPTION:" << le.what() << endl;
  }
  
  cout << "Output is " << v << endl;
}

// NUmeric arrays
void TestNumericArrays (bool uses_streams)
{
  const char* input[] = { "array([])", "array([1,2,3])", "array([1.1])",
		       "array([(1+2j), (3+5.6j)])",
		       0 };

  mainloop(input, trialNumericArrays, uses_streams);
}



// Single trial for Tabs
void trialTabs (ValReaderA& vr)
{
  Tab t;
  try {
    vr.expectTab(t);
  } catch (const logic_error& le) {
    cout << "EXCEPTION:" << le.what() << endl;
  }
  
  cout << "Output is " << t << endl;
}

void TestTabs (bool uses_streams)
{
  const char* input[] = { "{}", " {}", " { } ", " {", "{", "}",
		    "{ 'a':1 }", " { 'a':'test', 1:'sub'}",
		    "{ 'sub': {}, 'yo':1.2, 2:[1,2,3] }",
		    "", "{,}", " }", "{a:'}", "{ 1:1,2:2,3:3 }",

		    // Commas at end
		    "{ 'a': 1, 'b':2, }",
		    "{ 'a': 1, }",
		    "{ ,}",

                    // Pretty Printed table across lines
                    " \n"
		    " {                              \n" 
		    "     'a' : 1,                   \n"
                    "     'b' : {                    \n"
                    "             'c': None          \n"
                    "           },                   \n"
                    "     'd' : 3.14159265           \n"
                    " }                              \n",

                    // Pretty Printed table across lines with comments
		    " \n"
                    " { # Simple table with comments      \n" 
		    "     'a' : 1,                        \n"
                    "     'b' : {   # Nested              \n"
                    "             'c': None           \n"
                    "           },                    \n"
                    "     'd' : 3.14159265  # real #s \n"
                    " } # End of scope                   \n",
		    0
  };
  mainloop(input, trialTabs, uses_streams);
}

// test reading until EOF
void valReader ()
{
  const char* input[] = { 
    "{}", 
    "{} {}", 
    "{'a':1}\n{'b':2}",
    "{", // ERROR
    "{} {", // ERROR
    "{ 'a' ", // ERROR
    "{ 'a':1 }\n{'b':1}\n   {", // ERROR on line 3
    0 
  };
  for (int ii=0; input[ii]!=0; ii++) {

    const char* in = input[ii];
    cout << "****** Input is: \n'"<< string(in) << "'" << endl;
    istringstream is(in);
    StreamValReader vr(is);

    while (!vr.EOFComing()) {
      try {
	Val v;
	vr.expectAnything(v);
	cout << v << endl;
      } catch (const exception& e) {
	cout << e.what() << endl;
      }
    }
  }
  
}



int main ()
{
   
  Val v; 

   
  v = Eval("-9223372036854775809");
  cout << v << v.tag << endl;

  v = Eval("-9223372036854775808");
  cout << v << v.tag << endl;

  v = Eval("-9223372036854775807");
  cout << v << v.tag << endl;

  v = Eval("-922337203685477580");
  cout << v << v.tag << endl;

  v  = Eval("9223372036854775807");
  cout << v << v.tag << endl;

  v = Eval("9223372036854775808");
  cout << v << v.tag << endl;

  v = Eval("92233720368547758080");
  cout << v << v.tag << endl;

  v = Eval("922337203685477580800");
  cout << v << v.tag << endl;

  v = Eval("18446744073709551615");
  cout << v << v.tag << endl;

  v = Eval("18446744073709551616");
  cout << v << v.tag << endl;

  v = Eval("100L");
  cout << v << v.tag << endl;

  v = Eval("-100L");
  cout << v << v.tag << endl;
  

  bool uses_streams = true;
  TestNumbers(uses_streams);
  TestStrs(uses_streams);
  TestArrs(uses_streams);
  TestTabs(uses_streams);
  TestNumericArrays(uses_streams);

  uses_streams = false;
  TestNumbers(uses_streams);
  TestStrs(uses_streams); 
  TestArrs(uses_streams); 
  TestTabs(uses_streams); 
  TestNumericArrays(uses_streams);

  Tab t = "{ 'a':1, 'b':2.2, 'c':'three', 4:{ 'nest':None }, 5:[1,2.2,'three'], 6:array([1 2 3]) }";

  try {
    WriteTabToFile(t, "/tmp/sometab");
  } catch (const exception& e) {
    cout << e.what() << endl;
  }
  try {
    Tab f;
    ReadTabFromFile("/tmp/sometab", f);
    f.prettyPrint(cout);
    cout << "Same? " << int(f==t) << endl;
  } catch (const exception& e) {
    cout << e.what() << endl;
  } 


  cout << "Try to read from a file that doesn't exist" << endl;
  Tab g;
  try {
    ReadTabFromFile("/tmp/doesnotexists", g);
  } catch (const runtime_error& e) {
    cout << e.what() << endl;
  }


  // See what we can convert to
  Val a = Eval("array([1.123456789123456789], 'f')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1.123456789123456789], 'd')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1.123456789123456789], 'F')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1.123456789123456789], 'F')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], '1')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'b')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 's')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'w')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'i')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'u')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'l')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'f')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'd')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'F')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([1,2,-3], 'D')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  a = Eval("array([(1+1j),(-2-2j),(-3+3j),(+4+4j),(5-5j)], 'D')");
  cout << "tag:" << a.tag << " subtype:" << a.subtype << endl;
  cout << a << endl;

  {
    Val t = Tab("{ 'a': array([1,2,3], 'i') }");
    cout << bool(Eval(Stringize(t)) == t) << endl;  // 1
  }

  // test reading until EOF
  valReader ();

  // Double check that complexes read okay
  a = Eval("(1+2j)");
  cout << "tag:" << a.tag << endl;
  cout << a << endl;

  a = Eval("(-1+2j)");
  cout << "tag:" << a.tag << endl;
  cout << a << endl;

  a = Eval("(-1-2j)");
  cout << "tag:" << a.tag << endl;
  cout << a << endl;

  a = Eval("(1-2j)");
  cout << "tag:" << a.tag << endl;
  cout << a << endl;

  a = Eval("(+1-2j)");
  cout << "tag:" << a.tag << endl;
  cout << a << endl;
}

