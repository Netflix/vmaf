

#include "valprotocol2.h"
#include "opalprint.h"
#include "ocvalreader.h"
#include "valpickleloader.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


Array<char> readFile (const string& name)
{
  ifstream ifs(name.c_str());
  Array<char> aa;
  while (!ifs.eof()) {
    char c;
    c = (char) ifs.get();
      if (ifs.eof()) break;
      aa.append(c);
  }
  return aa;
}

string outChar (char c)
{
  string res = "";
  if (isprint(c)) {
    res += c;
  } else {
    int_u1 cc = c;
    // cout << cc << endl;
    res+= "\\x";
    res+= "0123456789ABCDEF"[(cc>>4)%16];
    res+= "0123456789ABCDEF"[cc%16];
  }
  return res;
}

string BinString (const string& s)
{
  string result;
  for (size_t ii=0; ii<s.length(); ii++) {
    result += outChar(s[ii]);
  }
  return result;
}

void ErrComp (const string& orig, const string& to_be)
{
  cerr << "******ERROR! Expected: " << endl << BinString(to_be) << "       and saw " << endl << BinString(orig) << endl;
  exit(1);
}

void Compare (const string& orig, const string& to_be)
{
  if (orig.length() != to_be.length()) {
    ErrComp(orig, to_be);
  }
  for (size_t ii=0; ii<orig.length(); ii++) {
    if (orig[ii] != to_be[ii]) {
      ErrComp(orig, to_be);
    }
  }
  cout << "Okay:" << BinString(orig) << ":" << endl;
}

void doTrials_ (const char** inputs, 
		ArrayDisposition_e arr_dis, PicklingIssues_e issues)
{
  Val sv;
  string expected;
  
  for (const char** in=inputs; *in!=NULL; in+=2) { 
    ValReader vr(*in);
    sv = None;
    vr.expectAnything(sv);
    cout << "String '" << string(*in) << "' becomes value " << sv.tag << ":" << sv << endl;
    
    // Expected may contain some \0, go past them
    const char* start = *(in+1);
    while (*start++!='.') ;
    int slen = start-*(in+1);
    string expected = string(*(in+1), slen);
    
    char* mmm=new char[P2TopLevelBytesToDumpVal(sv, arr_dis, issues)];
    char* end_mem = P2TopLevelDumpVal(sv, mmm, arr_dis, issues);
    int len = end_mem - mmm;
    string output(mmm, len);
    
    // for (int ii=0; ii<len; ii++) { cout << hex << int(expected[ii]) << " " << int(output[ii]) << " " << expected[ii] << " " << output[ii] << endl; }

    
    if (output!=expected) { 
      cout << "--ERROR: expected:"<<expected<<" but got " << output << endl; 
    } else { 
      cout << "--Dump passed.  " << endl;
    }
    
    // See if we can load it now
    Val loader;
    P2TopLevelLoadVal(loader, mmm);
    if (loader == sv) {
      cout << "--Load passed. Loaded correctly: " << sv << endl;
    } else {
      cout << "--ERROR: loader!= sv:" << loader << " " << sv << endl;
    }
    
    delete [] mmm;
  }
}

void NumberTrials ()
{
  const char *inputs[] = { "1", "K\x01.",  
		     "2147483647", "J\xff\xff\xff\x7f.",
		     "2147483648", "L2147483648L\n.",
		     "5000000000", "L5000000000L\n.",
		     "9223372036854775807", "L9223372036854775807L\n.",
		     // "9223372036854775808UL", "L9223372036854775808L\n.", Should this overflow to float or become int_u8?
		     "-1", "J\xff\xff\xff\xff.",
		     "-2147483647", "J\x01\x00\x00\x80.",
		     "-2147483648", "J\x00\x00\x00\x80.",
		     "-2147483649", "L-2147483649L\n.",
		     "-5000000000", "L-5000000000L\n.",
		     "-9223372036854775807", "L-9223372036854775807L\n.",
		     "-9223372036854775808", "L-9223372036854775808L\n.",
		     "-9223372036854775809", "G\xc3\xe0\x00\x00\x00\x00\x00\x00.", // numbers that overflow the int_8 become a real
		     "array([1 2 3])", "](K\x01K\x02K\003e.",

		     // EXPECT an error on the last one!!! An array when unpickled has lost its arrayness, but that's just the way it works. 
		     0 };
  
  doTrials_(inputs, AS_LIST, AS_PYTHON_2_2);
}

void NewNumericTrials ()
{
  const char *inputs[] = { "array([1 2 3])", "\x80\002cNumeric\narray_constructor\nq\x01(K\x03\x85U\001iU\x0c\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x88tR.",
		     "array([1])", "\x80\002cNumeric\narray_constructor\nq\x01(K\x01\x85U\001iU\x04\x01\x00\x00\x00\x88tR.",
		     "array([])", "\x80\002cNumeric\narray_constructor\nq\x01(K\x00\x85U\001iU\x00\x88tR.",
		     0 };
  
  doTrials_(inputs, AS_NUMERIC, ABOVE_PYTHON_2_2);
}

void OldNumericTrials ()
{
  const char *inputs[] = { "array([1 2 3])", "cNumeric\narray_constructor\nq\x01((K\x03tU\001iU\x0c\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00K\x01tR.",
		     "array([1])", "cNumeric\narray_constructor\nq\x01((K\x01tU\001iU\x04\x01\x00\x00\x00K\x01tR.",
		     "array([])", "cNumeric\narray_constructor\nq\x01((K\x00tU\001iU\x00K\x01tR.",
		     0 };
  
  doTrials_(inputs, AS_NUMERIC, AS_PYTHON_2_2);
}




int main ()
{

  // char memory[1024];

  // 
  Arr values = " ["
    "  [True, '\x88'], "
    "  [False, '\x89']," 
    "  [None, 'N'], "
    "  [{}, '}'], "
    "  [[], ']'], "
    // "  1.0: 'G?\xf0\x00\x00\x00\x00\x00\x00 ', " // DUH ... zeores in str!
    "]";

  // Put a floating point in
  char floaters[] = "G?\xf0\x00\x00\x00\x00\x00\x00"; 
  Arr a = "[ 1.0 ]";
  a.append(string(&floaters[0], sizeof(floaters)-1));
  values.append(a);

  // Put an int_u1 in
  a = Arr();
  a.append(int_u1(1));
  a.append("K\x01");
  values.append(a);

  a = Arr();
  a.append(int_1(-1));
  a.append("J\xff\xff\xff\xff");
  values.append(a);

  a = Arr();
  a.append(int_u1(255));
  a.append("K\xff");
  values.append(a);

  a = Arr();
  a.append(int_2(-1));
  a.append("J\xff\xff\xff\xff");
  values.append(a);

  a = Arr();
  a.append(int_u2(32767));
  a.append("M\xff\x7f");
  values.append(a);

  a = Arr();
  a.append(int(32767));
  a.append("M\xff\x7f");
  values.append(a);

  a = Arr();
  a.append(int(32769));
  a.append("M\x01\x80");
  values.append(a);

  a = Arr();
  a.append(int(32768));
  a.append(string("M\x00\x80", 3));
  values.append(a);

  a = Arr();
  a.append(-1);
  a.append("J\xff\xff\xff\xff");
  values.append(a);

  a = Arr();
  a.append(65536);
  a.append(string("J\x00\x00\x01\x00", 5));
  values.append(a);

  a = Arr();
  a.append(65537);
  a.append(string("J\x01\x00\x01\x00", 5));
  values.append(a);

  a = Arr();
  a.append(65535);
  a.append(string("M\xff\xff", 3));
  values.append(a);

  a = Arr();
  a.append(2147483647);
  a.append(string("J\xff\xff\xff\x7f", 5));
  values.append(a);

  a = Arr();
  a.append(int_8(4294967296LL));
  a.append(string("\x8a\x08\x00\x00\x00\x00\x01\x00\x00\x00", 10));
  values.append(a);

  char ll[] = "c__builtin__\ncomplex\nq\x01G?\xf0\x00\x00\x00\x00\x00\x00G@\x00\x00\x00\x00\x00\x00\x00\x86R";
  a = Arr();
  a.append(complex_16(1,2));
  a.append(string(ll, sizeof(ll)-1));
  values.append(a);

  {
    char ll2[] = "U\x0bsome string";
    a = Arr();
    a.append(string("some string"));
    a.append(string(ll2, sizeof(ll2)-1));
    values.append(a);
  }

  {
    char ll2[] = "T-\x01\x00\x00 ABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJ";
    a = Arr();
    string big_string = " ";
    for (int ii=0; ii<30; ii++) {
      big_string+= string("ABCDEFGHIJ");
    }
    string big_thing = string(ll2, sizeof(ll2)-1);
    a.append(big_string);
    a.append(big_thing);
    values.append(a);
  }
  

  // Watch out:  \x00 hex things in C end at the first NON HEX
  // char ... so \x00a real should be a '\0' followed by 'a', but
  // it's actually hex constant \x00a ... this is different than
  // Python
  { 
    char ll2[] = "}(U\001aK\001U\001bK\002u";
    a = Arr();
    a.append(Tab("{ 'a': 1, 'b': 2 }"));
    a.append(string(ll2, sizeof(ll2)-1));
    values.append(a);
  }

  { 
    char ll2[] = "](K\001K\002U\005threee";
    a = Arr();
    a.append(Arr("[1, 2, 'three']"));
    a.append(string(ll2, sizeof(ll2)-1));
    values.append(a);
  }



  // cout << values << endl;

  for (size_t ii=0; ii<values.length(); ii++) {
    Arr& a = values[ii];
    const Val& key = a[0];
    string val = a[1];

    // Dump key
    char* dump_memory = new char[P2BytesToDumpVal(key)];
    char* mem = P2DumpVal(key, dump_memory);
    size_t mem_length = mem-dump_memory;
    string s_mem = string(dump_memory, mem_length);

    // See what it looks like
    // for (int jj=0; dump_memory+jj != mem; jj++) {
    //  cout << int(dump_memory[jj]) << " " << dump_memory[jj] << endl;;
    //}
    //cout << endl;
    
    // Compare against what SHOULD be
    if (val != s_mem) {
      cout << "...ERROR: For " << key << " expected " << val << " but got " << s_mem << endl;
      throw 1;
    } else {
      cout << "...Good. For " << key << endl;
    }

    {
      string filename = "P2TestFiles/trial"+Stringize(ii)+".pi2";
      ofstream ofs(filename.c_str());
      char* dmem = new char[P2TopLevelBytesToDumpVal(key)];
      char* hhh = P2TopLevelDumpVal(key, dmem);
      string result(dmem, hhh-dmem);
      ofs << result;
      delete [] dmem;
    }
    
    delete [] dump_memory;
  }


  // char chunkers[4096];
  
  for (size_t ii=0; ii<values.length(); ii++) {
    string filename = "P2TestFiles/trial"+Stringize(ii)+".pi2";
    cout << "**" << filename << "** ";
    Array<char> aa = readFile(filename);

    // for (int jj=0; jj<aa.length(); jj++) {
    //  cout << int(aa[jj]) << " " << aa[jj] << endl;
    //}

    Val v;
    P2TopLevelLoadVal(v, aa.data());

    Arr& a = values[ii];
    const Val& key = a[0];
    string val = a[1];

    if (key!=v) {
      cout << "ERROR: Expected: " << key << " and got " << v << endl;
      throw 1;
    } else {
      cout << "Good. " << key << endl;
      string fn = "P2TestFiles/trial"+Stringize(ii)+".txt";
      ofstream ascout(fn.c_str());
      ascout << key;
    }
  }

  for (size_t ii=0; ii<values.length(); ii++) {
    string fn = "P2TestFiles/trial"+Stringize(ii)+".pythonpi2";
    Array<char> aa= readFile(fn);
    
    Val v;
    P2TopLevelLoadVal(v, aa.data());

    Arr& a = values[ii];
    const Val& key = a[0];
    string val = a[1];

    if (key!=v) {
      cout << "ERROR: Expected: " << key << " and got " << v << endl;
      throw 1;
    } else {
      cout << "Good. " << key << endl;
    }
  }

  char* chunk_mem = new char[P2TopLevelBytesToDumpVal(values)];
  //char* mmm = 
    (void)P2TopLevelDumpVal(values, chunk_mem);
  Val toppers;
  P2TopLevelLoadVal(toppers, chunk_mem);
  cout << values << " " << toppers << endl;
  cout << bool(values==toppers) << endl;
  if (values!=toppers) { cerr << "ERROR! values!=toppers!! EXIT!!" << endl; exit(1); }
  
  {
    ofstream ooo("P2TestFiles/final.txt");
    ooo << toppers << endl;
  }
  {
    ofstream ooo("P2TestFiles/final.pi2");
    char* iii = P2TopLevelDumpVal(values, chunk_mem);
    ooo << string(chunk_mem, iii-chunk_mem);
  }

  {
    Array<char> aa = readFile("P2TestFiles/final.pythonpi2");
    Val v;
    (void)P2TopLevelLoadVal(v, aa.data());
    cout << (v==values) << endl;
    if (values!=v) { cerr << "ERROR! values!=v!! EXIT!!" << endl; exit(1); }
  }
  delete [] chunk_mem;
  chunk_mem = 0;


  const char *filenames[] = { "special1.pythonpi2", "special2.pythonpi2", 
			"special3.pythonpi2", "special4.pythonpi2", 
			"special5.pythonpi2", "special6.pythonpi2",
			"special7.pythonpi2", "special8.pythonpi2", 
			0 };
  for (size_t kk=0; filenames[kk]!=0; kk++) {
    Array<char> aa = readFile(string("P2TestFiles/")+filenames[kk]);
    Val v;
    P2TopLevelLoadVal(v, aa.data());
    cout << filenames[kk] << " " << v << endl;
  }


  cout << "ARRAYS" << endl;
  {
    Array<int_u2> a;
    for (size_t ii=1; ii<6; ii++) { 
      a.append(100*ii);
    }

    char* chu = new char[P2TopLevelBytesToDumpVal(a)];
    char* mem = P2TopLevelDumpVal(a, chu);
    size_t len = mem-chu;
    string got(chu, len);
    // for (int ii=0; ii<len; ii++) { cerr << hex << int_u2(chu[ii]) << " " << chu[ii] << endl; }

    char expect[] = "\x80\x02](KdK\xc8M,\x01M\x90\x01M\xf4\01e.";
    string ex(expect, sizeof(expect)-1);
    
    if (ex!=got) { cerr << "ERROE!" << endl; exit(1); }
    cout << "Good.  Dump as List." << endl;
    
    delete [] chu;
    chu = 0;
    chu = new char[P2TopLevelBytesToDumpVal(a, AS_PYTHON_ARRAY)];
    mem = P2TopLevelDumpVal(a, chu, AS_PYTHON_ARRAY);
    len = mem-chu;
    got=string(chu, len);
    // for (int ii=0; ii<len; ii++) { cerr << hex << int_u2(chunkers[ii]) << " " << chunkers[ii] << endl; }

    char expect2[] = "\x80\002carray\narray\nq\001U\001HU\nd\x00\xc8\x00,\x01\x90\x01\xf4\x01\x86R.";
    string ex2(expect2, sizeof(expect2)-1);
    
    if (ex2!=got) { cerr << "ERROE!" << endl; exit(1); }
    cout << "Good.  Dump as Python Array." << endl;


    delete [] chu;
    chu = 0;
    chu = new char[P2TopLevelBytesToDumpVal(a, AS_NUMERIC)];
    mem = P2TopLevelDumpVal(a, chu, AS_NUMERIC);
    len = mem-chu;
    got=string(chu, len);
    //    for (int ii=0; ii<len; ii++) { cerr << hex << int_u2(chunkers[ii]) << " " << chunkers[ii] << endl; }

    char expect3[] = "\x80\002cNumeric\narray_constructor\nq\001(K\x05\x85U\x01wU\nd\x00\xc8\x00,\x01\x90\x01\xf4\x01\x88tR.";
    string ex3(expect3, sizeof(expect3)-1);
    
    //for (int oo=0; oo<ex3.length(); oo++) {
    //   cerr << ex3[oo] << " " << int(ex3[oo]) << " VS. " << got[oo] << " " << int(got[oo]) << endl; }
    if (ex3!=got) { cerr << "ERROR!" << endl; exit(1); }
    cout << "Good.  Dump as Python Numeric Array." << endl;

    delete [] chu;
  }


  // Try to dump some values as Python 2.2 would:
  // NOTE: No preamble and ints are dumped differently
  // and so are Numeric arrays
  NumberTrials();
  
  NewNumericTrials();
  OldNumericTrials();
  
  Val g = Tab("{ 'a': (1+2j) }");
  cout << g << endl;
  complex_16 f = g("a");
  cout << f << endl;





  // New work:  Let's see how the Proxies serialize

  Array<string> expect;
  Arr options;
  Arr to_do;
  Arr ppp;
  Proxy p;
  Val nt;

  // A proxy with 2 empty proxies: possibly easiest case, but requires
  // backtracking
  ppp = Arr();
  p = new Arr();
  ppp.append(p);
  ppp.append(p);
  expect.append("](]q\x01h\001e"); 
  to_do.append(ppp);
  options.append(None); // Nothing special when serializing

 
  // A proxy with 2 empty proxies: possibly easiest case, but requires
  // backtracking
  ppp = Arr();
  p = new Tab();
  ppp.append(p);
  ppp.append(p);
  expect.append("](}q\x01h\001e"); 
  to_do.append(ppp);
  options.append(None); // Nothing special when serializing

  // A proxy with 3 empty proxies: possibly easiest case, but requires
  // backtracking
  ppp = Arr();
  p = new Tab();
  ppp.append(p);
  ppp.append(p);  
  ppp.append(p);
  expect.append("](}q\x01h\001h\001e"); 
  to_do.append(ppp);
  options.append(None); // Nothing special when serializing

  // A single Proxy with no other references (nothing shared when serializing)
  ppp = Arr();
  nt = new Tab();
  ppp.append(nt);
  expect.append("]}q\001a"); 
  to_do.append(ppp);
  options.append(None); // Nothing special when serializing

  // A proxy with something that doesn't require backtracking
  ppp = Arr();
  nt = new Tab("{'a':1, 'b':2}");
  ppp.append(nt);
  ppp.append(nt);
  expect.append("](}q\x01(U\001aK\x01U\001bK\x02uh\001e");
  to_do.append(ppp);
  options.append(None); // Nothing special when serializing


  // A proxy with just 1 key (this may require backtracking)
  ppp = Arr();
  nt = new Tab("{'a':1}");
  ppp.append(nt);
  ppp.append(nt);
  expect.append("](}q\x01U\001aK\001sh\001e"); 
  to_do.append(ppp);
  options.append(None); // Nothing special when serializing

  ppp = Arr();
  Proxy li = new Array<real_8>(2);
  Array<real_8>& lia = li;
  lia.append(1.0);
  lia.append(0.0);
  ppp.append(li);
  ppp.append(li);
  expect.append(string("](]q\x01(G?\xf0\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\000eh\001e", 28));
  to_do.append(ppp);
  options.append(None); // Nothing special when serializing

  // reuse last one AS NUMERIC
  expect.append(string("](cNumeric\narray_constructor\nq\x01(K\x02\x85U\001dU\x10\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x88tRq\x02h\002e", 64));
  to_do.append(ppp);
  options.append(Arr("[0, 1]")); // AS_NUMERIC=0, ABOVE_PYTHON_2_2=1

  // reuse last one AS NUMERIC. but with python 2.2
  expect.append(string("](cNumeric\narray_constructor\nq\x01((K\x02tU\001dU\x10\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00K\x01tRq\x02h\002e.", 66));
  to_do.append(ppp);
  options.append(Arr("[0, 0]")); // AS_NUMERIC=0, AS_PYTHON_2_2=0

  // reuse last one AS ARRAY
  expect.append(string("](carray\narray\nq\x01U\001dU\x10\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x86Rq\x02h\002e", 45));
  to_do.append(ppp);
  options.append(Arr("[2, 1]")); // AS_ARRAY=2, ABOVE_PYTHON_2_2=1

  // reuse last one AS ARRAYm but with Python 2.2 (CAN't!  2.2 no pickle array)
  //expect.append(string("](carray\narray\nq\x01U\001dU\x10\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x86Rq\x02h\002e", 45));
  //to_do.append(ppp);
  //options.append(Arr("[2, 0]")); // AS_ARRAY=2, ABOVE_PYTHON_2_2=1

  // Let's try a few empties to make sure they work
  ppp = Arr();
  Proxy po = new Array<complex_16>(10);
  Array<complex_16>& uuu = po;
  uuu.append(complex_8(1,2));
  ppp.append(po);
  ppp.append(po);
  to_do.append(ppp);
  expect.append(string("](cNumeric\narray_constructor\nq\x01(K\x01\x85U\001DU\x10\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x88tRq\x02h\002e", 64));
  options.append(Arr("[0, 1]")); // AS_NUMERIC=0, ABOVE_PYTHON_2_2=1

  // Let's put more than 256 in, so that we can makes ure the we use
  // BINPUT and LONG_BINPUT correctly.
  ppp = Arr();
  for (size_t ii=0; ii<257; ii++) {
    Proxy lp;
    ppp.append(lp);
    ppp.append(lp);
  }
  to_do.append(ppp);
  // Build expect string
  string longish = "](";
  for (size_t ii=1; ii<=255; ii++) {
    string h = "}qXhX";
    h[2] = h[4] = ii;
    longish.append(h);
  }
  string pr("}r\000\001\000\000j\000\001\000\000}r\001\001\000\000j\001\001\000\000e", 23);
  longish.append(pr);
  expect.append(longish);
  options.append(None);
  {
    cout << "***Trying proxies" << endl;
    for (size_t ii=0; ii<expect.length(); ii++) {
      char* dump_memory;
      char* mem;
      size_t mem_length;

      Val opt = options[ii];
      if (opt==None) {
	dump_memory = new char[P2BytesToDumpVal(to_do[ii])];
	mem = P2DumpVal(to_do[ii], dump_memory);
	mem_length = mem-dump_memory;
      } else {
	ArrayDisposition_e ad = (ArrayDisposition_e) int(opt(0));
	PicklingIssues_e ps = (PicklingIssues_e) int(opt(1));
	dump_memory = new char[P2BytesToDumpVal(to_do[ii], ad, ps)];
	mem = P2DumpVal(to_do[ii], dump_memory, ad, ps);
	mem_length = mem-dump_memory;
      }
      cout << "mem_len = " << mem_length << endl;
      //string s_mem = string(dump_memory, mem_length);

      // See what it looks like
      if (0) {
	for (size_t jj=0; dump_memory+jj != mem; jj++) {
	  char c = dump_memory[jj];
	  cout << outChar(c) << endl; 
	  // cout << int_u4(int_u1(dump_memory[jj])) << " " << dump_memory[jj] << endl;;
	}
	cout << endl;
      }
      
      // Compare against what SHOULD be
      const char* ee = expect[ii].data();
      if (expect[ii].length() != mem_length) { 
	cerr << "Different lengths for " << to_do[ii] << expect[ii].length() << " " << mem_length << endl;
	throw 1;
      }
      for (size_t jj=0; jj<expect[ii].length(); jj++) {
	if (ee[jj] != dump_memory[jj]) {
	  cout << "...ERROR: For " << to_do[ii] << " expected " << outChar(ee[jj]) << " but got " << outChar(dump_memory[jj]) << endl;
	  throw 1;
	} else {
	  // cout << "...OKAY:  expected " << outChar(ee[jj]) << " and got " << outChar(dump_memory[jj]) << endl;
	  //
	}
      }
      cout << "...Good. For " << to_do[ii] << endl;

      Val ooo;
      P2LoadVal(ooo, dump_memory);
      if (ooo!= to_do[ii]) {
	cout << "Not the same???" << ooo << to_do[ii] << endl;
      }

      delete [] dump_memory;
    }
  }

  // make sure even small ints (int_u2) become "large ints" in Python
  // p2 (which we means full native Python ints, not necessarily
  // Python LONGS)
  {
    char* dump_memory, *mem;
    size_t mem_length;
   
    Val hh;
    Val res;

    hh = int_u2(7000);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_u1(127);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_u1(128);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_u1(255);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_u1(256);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_1(127);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_1(128);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_1(-127);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_1(-128);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;

    hh = int_1(-129);
    dump_memory = new char[P2BytesToDumpVal(hh)];
    mem = P2DumpVal(hh, dump_memory);
    mem_length = mem-dump_memory;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << hh.tag << ":" << hh << endl;
    cout << "load:" << res.tag << ":" << res << endl;

    if (mem_length == 11234) {
      cerr << "Get rid of unused var, but still want to show how length is"
	   << " measured." << endl;
    }
    delete [] dump_memory;
  }

  {
    // This is just a load test: I don't think we produce "shapeless"
    // arrays from our side.
    // Numeric.zeros((), 'i')) gets dumped differently by Python

    // >>> cPickle.dumps(Numeric.zeros((1),'i'),2)
    char *z1 = (char*)"\x80\002cNumeric\narray_constructor\nq\x01(K\x01\x85U\x01iU\x04\x00\x00\x00\x00\x88tRq\x02.";

    Val res1;
    P2TopLevelLoadVal(res1, z1);
    cout << "cPickle.dumps(Numeric.zeros((1), 'i') reloads as: " << endl;
    res1.prettyPrint(cout);
    cout << endl;
    cout << "tag:" << res1.tag << " " << res1.subtype << endl;

    // >>> cPickle.dumps(Numeric.zeros((),'i'),2)
    char* z2 = (char*)"\x80\002cNumeric\narray_constructor\nq\x01()U\x01iU\x04\x00\x00\x00\x00\x88tRq\x02.";
    Val res2;
    P2TopLevelLoadVal(res2, z2);
    cout << "cPickle.dumps(Numeric.zeros( (), 'i') reloads as: " << endl;
    res2.prettyPrint(cout);
    cout << endl;
    cout << "tag:" << res2.tag << " " << res2.subtype << endl;
  }

  {
    cout << "new features" << endl;
    Val v = Tup();
    char* dump_memory = new char[P2BytesToDumpVal(v)+1];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), ")");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
  }

  {
    Val v = Tup(1);
    char* dump_memory = new char[P2BytesToDumpVal(v)];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), "K\x01\x85");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    cout << " ... we expect this to be a zero because IT DOES NOT work.  We have to rewrite P2 to handle tuples better (this includes 1,2 and 3 element tuples)." << endl;
    delete [] dump_memory;
  }

  {
    Val v = Tup(1,2);
    char* dump_memory = new char[P2BytesToDumpVal(v)];
    char* mem = P2DumpVal(v, dump_memory);
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), "K\x01K\x02\x86");
    //P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    //cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;
  }

  {
    Val v = Tup(1,2,3);
    char* dump_memory = new char[P2BytesToDumpVal(v)];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), "K\x01K\x02K\x03\x87");
    //Val res;
    //P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    //cout << "load:" << res.tag << ":" << res << endl;
    delete [] dump_memory;
  }

  {
    Val v = Tup(1,2,3,4);
    char* dump_memory = new char[P2BytesToDumpVal(v)];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), "(K\x01K\x02K\x03K\x04t");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
  }

  {
    Val v = OTab();
    char* dump_memory = new char[P2BytesToDumpVal(v)+1];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), "ccollections\nOrderedDict\nq\x01]\x85R");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
  }

  {
    Val v = OTab("o{'a':1}");
    char* dump_memory = new char[P2BytesToDumpVal(v)];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), 
	    "ccollections\nOrderedDict\nq\x01]](U\001aK\001ea\x85R");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
  }

  {
    Val v = OTab("o{'a':1, 'b':2}");
    char* dump_memory = new char[P2BytesToDumpVal(v)];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), 
	    "ccollections\x0AOrderedDict\x0Aq\x01](](U\001aK\001e](U\001bK\002ee\x85R");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
  }

  {
    Val vp = new OTab("o{'a':1, 'b':2}");
    Val v = Arr();
    v.append(vp); v.append(vp);
    char* dump_memory = new char[P2BytesToDumpVal(v)];
    char* mem = P2DumpVal(v, dump_memory);  *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), 
	    "](ccollections\x0AOrderedDict\x0Aq\001](](U\001aK\001e](U\001bK\002ee\x85Rq\002h\002e");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
  }

  {
    Val vp = new OTab();
    Val v = Tup(vp, vp, vp, vp);
    char* dump_memory = new char[P2BytesToDumpVal(v)+1];
    char* mem = P2DumpVal(v, dump_memory); *mem = '.';
    size_t mem_length = mem-dump_memory;
    Compare(string(dump_memory, mem_length), 
	    "(ccollections\x0AOrderedDict\x0Aq\001]\x85Rq\002h\002h\002h\002t");
    Val res;
    P2LoadVal(res, dump_memory);
    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
  }


  // NUMPY
  {
    Val v = Eval("array([1,2,3,4], 'f')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\002f4K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@tb.",
		   141));
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Val v = Eval("array([1,2,3,4], 'd')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02\x66\x38K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U \x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@tb.",
		   157));
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Val v = Eval("array([1,2,3,4], 'F')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02\x63\x38K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U \x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00@@\x00\x00\x00\x00\x00\x00\x80@\x00\x00\x00\x00tb.",
		   157));
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Val v = Eval("array([1,2,3,4], 'D')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x03\x63\x31\x36K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U@\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x00\x00tb.",
		   190));
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }



  {
    Val v = Eval("array([1,2,3,4], 'b')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02u1K\x00K\x01\x87R(K\x03U\x01|NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x04\x01\x02\x03\x04tb.",
		   129));
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Array<int_1> a;
    a.append(1);
    a.append(2);
    a.append(3);
    a.append(4);
    Val v = a; // Eval("array([1,2,3,4], 'b')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02i1K\x00K\x01\x87R(K\x03U\x01|NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x04\x01\x02\x03\x04tb.",
		   129));
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }


  {
    Array<int_u2> a;
    a.append(1); a.append(2); a.append(3); a.append(4);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02u2K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x08\x01\x00\x02\x00\x03\x00\x04\x00tb.", 133)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Array<int_2> a;
    a.append(1); a.append(2); a.append(3); a.append(4);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02i2K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x08\x01\x00\x02\x00\x03\x00\x04\x00tb.", 133)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }


  {
    Array<int_u4> a;
    a.append(1); a.append(2); a.append(3); a.append(4);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02u4K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x10\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00tb.",
		   141)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Array<int_4> a;
    a.append(1); a.append(2); a.append(3); a.append(4);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02i4K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x10\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00tb.",
		   141)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Array<int_4> a;
    a.append(1); a.append(2); a.append(3); a.append(4);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02i4K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x10\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00tb.",
		   141)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }
  {
    Array<int_8> a;
    a.append(1); a.append(2); a.append(3); a.append(4);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02i8K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U \x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00tb.",
		   157)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }

  {
    Array<int_u8> a;
    a.append(1); a.append(2); a.append(3); a.append(4);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\x02u8K\x00K\x01\x87R(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U \x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00tb.",
		   157)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }


  {
    Array<bool> a;
    a.append(1); a.append(1); a.append(0); a.append(0);
    Val v = a;
    //Val v = Eval("array([1,2,3,4], 'i')");
    char* dump_memory = new char[P2BytesToDumpVal(v, AS_NUMPY)+3];
    dump_memory[0] = '\x80';
    dump_memory[1] = '\x02';
    char* mem = P2DumpVal(v, dump_memory+2, AS_NUMPY); 
    *mem = '.';
    size_t mem_length = mem-dump_memory+1;
    Compare(string(dump_memory, mem_length), 
	    string("\x80\002cnumpy.core.multiarray\n_reconstruct\nq\001cnumpy\nndarray\nq\x02K\x00\x85U\001b\x87R(K\x01K\x04\x85\x63numpy\ndtype\nq\x03U\002b1K\x00K\x01\x87R(K\x03U\x01|NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x04\x01\x01\x00\x00tb.",
		   129)
	    );
    
    Val res;
    PickleLoader pl(dump_memory, mem_length);
    pl.env()["supportsNumeric"] = 1;
    //cout << pl.env() << endl;
    pl.loads(res);

    cout << "orig:" << v.tag << ":" << v << endl;
    cout << "load:" << res.tag << ":" << res << endl;
    cout << bool(v==res) << endl;
    delete [] dump_memory;
    
  }


  return 0;
}
