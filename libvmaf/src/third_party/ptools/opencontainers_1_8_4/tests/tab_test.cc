
// The Val/Tab/Str is young and in flux right now, but included for
// your use.  This test needs to be fleshed out a whole lot more, but
// it does test some basic functionality and interface.

#include "ocval.h"
#include "ocserialize.h"
#include "occonvert.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

template <class T>
void convertOut_ (T v)
{
  /* Solaris doesn't seem to like this, 
     but both gcc and cxx do */
  /*
  { 
    const int_1  ci1  = v; cout << char(ci1)  << " ";
    const int_u1 ciu1 = v; cout << char(ciu1) << " ";
    const int_2  ci2  = v; cout << ci2  << " ";
    const int_u2 ciu2 = v; cout << ciu2 << " ";
    const int_4  ci4  = v; cout << ci4  << " ";
    const int_u4 ciu4 = v; cout << ciu4 << " ";
    const int_8  ci8  = v; cout << ci8  << " ";
    const int_u8 ciu8 = v; cout << ciu8 << " ";
    const real_4 cr4  = v; cout << cr4  << " ";
    const real_8 cr8  = v; cout << cr8  << " ";
    const Str    cs   = v; cout << cs   << " ";
    try {
      const Tab t = v;
      cout << t;
    } catch (const logic_error& e) {
      cout << "Error Converting to Tab:" << v << " " << e.what() << endl;
    }
    cout << endl;
  }
  */

  /* Solaris doesn't seem to like this, 
     but both gcc and cxx do */
  /*
  { 
    const int_1&  ci1  = v; cout << char(ci1)  << " ";
    const int_u1& ciu1 = v; cout << char(ciu1) << " ";
    const int_2&  ci2  = v; cout << ci2  << " ";
    const int_u2& ciu2 = v; cout << ciu2 << " ";
    const int_4&  ci4  = v; cout << ci4  << " ";
    const int_u4& ciu4 = v; cout << ciu4 << " ";
    const int_8&  ci8  = v; cout << ci8  << " ";
    const int_u8& ciu8 = v; cout << ciu8 << " ";
    const real_4& cr4  = v; cout << cr4  << " ";
    const real_8& cr8  = v; cout << cr8  << " ";
    const Str&    cs   = v; cout << cs   << " ";
    try {
      const Tab& t = v;
      cout << t;
    } catch (const logic_error& e) {
      cout << "Error Converting to Tab:" << v << " " << e.what() << endl;
    }
    cout << endl;
  }
  */

  // This seems to work on all platforms
  {
    int_1  i1  = v; cout << char(i1)  << " ";
    int_u1 iu1 = v; cout << char(iu1) << " ";
    int_2  i2  = v; cout << i2  << " ";
    int_u2 iu2 = v; cout << iu2 << " ";
    int_4  i4  = v; cout << i4  << " ";
    int_u4 iu4 = v; cout << iu4 << " ";
    int_8  i8  = v; cout << i8  << " ";
    int_u8 iu8 = v; cout << iu8 << " ";
    bool     b = v; cout << Val(b)   << " ";
    real_4 r4  = v; cout << r4  << " ";
    real_8 r8  = v; cout << r8  << " ";
    complex_8 c8  = v; cout << c8  << " ";
    complex_16 c16  = v; cout << c16  << " ";
    Str    s   = v; cout << s   << " ";
    try {
      Tab t = v;
      cout << t << endl;
    } catch (const logic_error& e) {
      cout << "Error Converting to Tab:" << v << " " << e.what() << endl;
    }
  }

  /* Nobody seems to like this, for good reason.
     References to temporaries.
  {
    int_1&  i1  = v; cout << char(i1)  << " ";
    int_u1& iu1 = v; cout << char(iu1) << " ";
    int_2&  i2  = v; cout << i2  << " ";
    int_u2& iu2 = v; cout << iu2 << " ";
    int_4&  i4  = v; cout << i4  << " ";
    int_u4& iu4 = v; cout << iu4 << " ";
    int_8&  i8  = v; cout << i8  << " ";
    int_u8& iu8 = v; cout << iu8 << " ";
    real_4& r4  = v; cout << r4  << " ";
    real_8& r8  = v; cout << r8  << " ";
    Str&    s   = v; cout << s   << " ";
    try {
      Tab& t = v;
      cout << t << endl;
    } catch (const logic_error& e) {
      cout << "Error Converting to Tab:" << v << " " << e.what() << endl;
    }
  }
  */

}

#include <ctype.h>

static void SerializeTest (Val& big)
{
  cout << "Attempt serialization of " << big << endl;
  Val something;
  size_t bytes = BytesToSerialize(big);
  cout << " ... bytes to serialize = " << bytes << endl;
  char* mem = new char[bytes];
  try {
    Serialize(big, mem);
    // cout << "Done ser." << endl;
    //for (int ii=0; ii<bytes; ii++) {
    //  if (isprint(mem[ii])) {
    //  	cerr << char(mem[ii]) << " "; 
    //  } else {
    //    cerr << int(mem[ii]) << " "; 
    //  }
    //}
    
    Deserialize(something, mem);
  } catch (exception& e) {
    cout << "CAUGHT EXCEPTION:" << e.what() << endl;
  }
  cout << "Deserialized into:" << something << endl;
  if (something != big) {
    cerr << "ERROR! Should be the same tables!" << endl;
  }
  delete [] mem;
}


int main()
{
  // Try converting out Val, const Val&, Val&, const Val
  Val v = 123.567890123456;
  convertOut_(v);
  v = 56;
  convertOut_(v);
  v = Tab();
  convertOut_(v);

  Val& vr = v;
  v = 123.567890123456;
  convertOut_(vr);
  v = 56;
  convertOut_(vr);
  v = Tab();
  convertOut_(vr);

  const Val& cvr = v;
  v = 123.567890123456;
  convertOut_(cvr);
  v = 56;
  convertOut_(cvr);
  v = Tab();
  convertOut_(cvr);

  const Val v1 = 123.567890123456;
  convertOut_(v1);
  const Val v2 = 56;
  convertOut_(v2);
  const Val v3 = Tab();
  convertOut_(v3);


  // Basic ops
  Tab t;
  t["a"] = 1;
  t["b"] = 3.141569;
  t["c"] = "hello";
  t["d"] = complex_8(5,6);
  cout << t << endl;

  Val ggg = complex_8(4,5);
  cout << ggg << endl;
  complex_16 cc = ggg;
  cout << cc << endl;
  Str scc = ggg;
  cout << scc << endl;

  Tab sub;
  sub["100"] = 100;
  cout << sub << endl;

  t["subtable"] = sub;
  cout << t << endl;

  cout << t["subtable"]["100"] << endl;
  
  try {
    cout << t["a"]["not there!"] << endl;
  } catch (const logic_error& e) {
    cout << "As expected: Can't subscript!" << endl;
  }
  
  for (It ii(t); ii(); ) {
    cout << ii.key() << " " << ii.value() << endl;
  }

  Arr aa(10);
  for (int zz=0; zz<10; zz++) 
    aa.append(zz);
  aa.append("EOF");
  aa.append(100);
  aa.append(123.456);
  aa.append(complex_16(12,17.17));
  aa.append(Val()); // None
  aa.append(Tab()); // None
  cout << "Arr aa = " << aa << endl;

  // Try looking into an array
  Val av(aa);
  for (int yy=0; yy<int(aa.length()); yy++) {
    cout << av[yy] << " ";
  }
  cout << endl;

  // cascase through array and table
  aa[15]["hello"] = 666;
  cout << aa << endl;
  
  Tab one;
  one[int_4(1)] = 100;
  one[int_2(1)] = 200;
  one[real_8(1)] = 300;
  one["1"] = 400;
  cout << one << endl;

  Tab big;
  big["0"] = 0;
  big["1"] = int_1(1);
  big["2"] = int_u1(2);
  big["3"] = int_2(3);
  big["4"] = int_u2(4);
  big["5"] = int_4(5);
  big["6"] = int_u4(6);
  big["7"] = int_8(7);
  big["8"] = int_u8(8);
  big["9"] = 9.0;
  big["10"] = 10.0f;
  big["11"] = 'a';
  big["12"] = "hello";
  big["13"] = Str();
  big["14"] = Tab();
  big["15"] = Val();
  big["16"] = aa;
  big["17"] = complex_8(3, .1415);
  big["18"] = complex_16(3, .1415926535897);
  big["19"] = (1==0);
  cout << big << endl;
  Tab new_big = big;
  cout << new_big << endl;
 
  cout << "Big Table=" << big << endl;
  for (int kk=0; kk<int(big.entries()); kk++) {
    Str key = Val(kk);
    Val& value = big[key];
    cout << key << " = "<< value.tag <<":"<< value << endl;
  }


  cout << "Testing tables converting to Tab references:" << big["14"] << endl;
  Tab& tref = big["14"];
  tref["666"] = "the beast";
  cout << "Testing tables converting to Tab references:" << tref << endl;

  cout << "Testing arrays converting to refs:" << big["16"] << endl;
  Array<Val>& aref = big["16"];
  aref[2] = "something else";
  cout << big["16"] << endl;

  Val realval = Array<real_8>();
  Array<real_8>& a4 = realval;
  for (int ii=0; ii<8; ii++)
    a4.append(ii);
  cout << realval << endl;


  SerializeTest(realval);
  Val art=big["14"];
  SerializeTest(art);
  Val vg = Arr();
  Arr& av1=vg; av1.append("hello"); av1.append(16.7);
  SerializeTest(vg);
  Val ttt = big;
  SerializeTest(ttt);


  Tab f;
  f["1"] = 1;
  f[1] = 2;
  cout << f << endl;

  Arr a = 
    "[ 1, 1.0, "
    "  1.0, 1, "
    "  1, None, "
    " None, 1, "
    "  1, '1',"
    "  '1', 1,"
    "  1, [1], "
    "  [1], 1, "
    "  1, { '1': 1 }, "
    "  1, 1, "
    "  1, 1L ,"
    " True, 1,"
    " 1, True,"
    " True, 0,"
    " 0, True,"
    " True, 17,"
    " 17, True,"
    " False, 1,"
    " 1, False,"
    " False, 0,"
    " 0, False,"
    " False, 17,"
    " 17, False,"
    " True, '1',"
    " False, '1',"
    " [1,2,3], [1.0, 2.0, 3.0],"
    " [1.0, 2.0, 3.0], [1,2,3] ]"
    ;

  for (int ii=0; ii<int(a.length()); ii+=2) {
    Val& l1 = a[ii];
    Val& l2 = a[ii+1];
    cout << l1 << " <  " << l2 << ":" << int(l1<l2) << endl;
    cout << l1 << " <= " << l2 << ":" << int(l1<=l2) << endl;
    cout << l1 << " >  " << l2 << ":" << int(l1>l2) << endl;
    cout << l1 << " >= " << l2 << ":" << int(l1>=l2) << endl;
    cout << l1 << " == " << l2 << ":" << int(l1==l2) << endl;  
    cout << l1 << " != " << l2 << ":" << int(l1!=l2) << endl;
  }

  
  Val vgg = Tab(" { 1:'a', 2:'b', 3:'c', 'a':'AAA' }");
  Tab& gg = vgg;
  try {
    Val v = gg(1);   // okay
    Val v2 = gg(1.0); // okay
    Val gh = gg(666);// should fail
  } catch (const out_of_range& re) {
    cout << "As expected:" << re.what() << endl;
  }
  try {
    Val v = gg("a");   // okay
    Val gh = gg("bbb");// should fail
  } catch (const out_of_range& re) {
    cout << "As expected:" << re.what() << endl;
  }
  try {
    Val v = vgg(1);   // okay
    Val v2 = vgg(1.0); // okay
    Val gh = vgg(666);// should fail
  } catch (const out_of_range& re) {
    cout << "As expected:" << re.what() << endl;
  }
  try {
    Val v = vgg("a");
    Val v2 = vgg("bbb");
  } catch (const out_of_range& re) {
    cout << "As expected:" << re.what() << endl;
  }


  cout << "Testing output:" << endl;
  Val vf = "123";
  cout << "As a val: v should have the quotes" << endl << vf << endl;
  cout << "As a string, v should NOT have the quotes" << endl << string(vf) << endl;


  swap(vf, vgg);
  cout << vf << " " << vgg << endl;
  swap(vf, vgg);
  cout << vf << " " << vgg << endl;

  Val zz1 = 1;
  Val zz2 = 1.1;
  cout << zz1 << " " << zz2 << endl;
  swap(zz1, zz2);
  cout << zz1 << " " << zz2 << endl;

  // test out append with new exception behavior
  Tab uuu;
  uuu.append(1.1);
  uuu.append(None);
  uuu.append("hello");
  cout << uuu << endl;
  uuu.remove(1);
  try {
    uuu.append("not going in");
  } catch (const runtime_error& re) {
    cout << "EXPECTED:" << re.what() << endl;
  }
  
  Tab hhh;
  hhh.appendStr(1.1);
  hhh.appendStr(None);
  hhh.appendStr("hello");
  cout << hhh << endl;
  hhh.remove("1");
  try {
    hhh.appendStr("not going on");
  } catch (const runtime_error& re) {
    cout << "EXPECTED:" << re.what() << endl;
  } 
  cout << hhh << endl;


  // make sure we can put arrays of strings inside Val
  Array<string> astr;
  astr.append("hello");
  astr.append("there");
  astr.append("everyone");
  Val vstr = astr;
  Array<string>& af = vstr;
  cout << af << endl;
  try {
    Array<real_8>& u = vstr;
    if (u.length()==777777777) exit(1); // dumb check
  } catch (exception& x) {
    cout << "Expected:" << x.what() << endl;
  }

  // make sure we can put arrays of Tabs inside Val
  Array<Tab> tarr;
  tarr.append("{'a':1}");
  tarr.append("{'b':2}");
  tarr.append("{'c':3}");
  Val gggg = tarr;
  Array<Tab>& ta = gggg;
  cout << ta << endl;

  // contains test
  {
    Tab t = "{'there':1, 'b':2}";
    Val v =t;
    cout << t.contains("there") << " " << v.contains("there") << endl;
    cout << t.contains(1) << " " << v.contains(1) << endl;
    cout << t.contains(666) << " " << v.contains(666) << endl;


    Arr a = "[ 1, 2.2, 'there']";
    v = a;
    cout << a.contains("there") << " " << v.contains("there") << endl;
    cout << a.contains(1) << " " << v.contains(1) << endl;
    cout << a.contains(666) << " " << v.contains(666) << endl;
    
    Array<int_u4> a4;
    a4.append(1);
    a4.append(32768);
    v = a4;
    cout << a4.contains(1) << " " << v.contains(1) << endl;
    cout << a4.contains(666) << " " << v.contains(666) << endl;

    string s = "there";
    v = s;
    try {
      v = s;
      cout << v.contains("there"); // SHOULD FAIL
    } catch (const logic_error& le) {
      cout << "GOOD: Expected string to fail:" << le.what() << endl;
    }

    int i = 1;
    v = i;
    try {
      v = i;
      cout << v.contains(1); // SHOULD FAIL
    } catch (const logic_error& le) {
      cout << "GOOD: Expected int to fail:" << le.what() << endl;
    }
  }

  // Allow the nested assignment idiom for Tab
  {
    Tab big = "{ 'nest': { 'a':1 } }";
    cout << "TAB:big before:" << big << endl;
    big = big["nest"];
    cout << "TAB:big after:" << big << endl;
    
  }

  // Allow the nested assignment idiom for Arr
  {
    Arr big = "[ [ 1,2,3], None] ";
    cout << "ARR:big before:" << big << endl;
    big = big[0];
    cout << "ARR:big after:" << big << endl;
    
  }

  // Allow the nested assignment idiom for Val
  {
    Val big = Eval("[ {'nest': [1,2,3]}, None] ");
    cout << "VAL:big before:" << big << endl;
    big = big[0];
    cout << "VAL:big after:" << big << endl;
    big = big["nest"];
    cout << "VAL:big after2:" << big << endl;
  }

  // See if the conversion routines work
  {
    cout << "ConvertArrToTab" << endl;

    Val t = Arr();
    cout << t << endl;
    ConvertArrToTab(t);
    cout << t << endl;

    t = Arr("[1,2.2,'three']");
    cout << t << endl;
    ConvertArrToTab(t);
    cout << t << endl;

    t = Arr("[1,2.2,'three']");
    cout << t << endl;
    ConvertArrToTab(t, false, true); // Stringize keys
    cout << t << endl;

    t = Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << endl;
    ConvertArrToTab(t); 
    cout << t << endl;

    t = Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << endl;
    ConvertArrToTab(t, false); 
    cout << t << endl;

    t = Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << endl;
    ConvertArrToTab(t, true); 
    cout << t << endl;

    t = Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << endl;
    ConvertArrToTab(t, true, false); 
    cout << t << endl;

    t = Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << endl;
    ConvertArrToTab(t, true, true); 
    cout << t << endl;

    t = Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << endl;
    ConvertArrToTab(t, false, true); 
    cout << t << endl;

    t = Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << endl;
    ConvertArrToTab(t, false, false); 
    cout << t << endl;

    t = new Arr("[1,2.2,'three', [0,1,2]]");
    cout << t << IsProxy(t) << endl;
    ConvertArrToTab(t, false, false); 
    cout << t << IsProxy(t) << endl;

    t = Arr("[1,2,3,{'0':[5,6,7], '1':7}]");
    cout << t << IsProxy(t) << endl;
    ConvertArrToTab(t, false, false); 
    cout << t << IsProxy(t) << endl;

    t = Arr("[1,2,3,{'0':[5,6,7], '1':7}]");
    cout << t << IsProxy(t) << endl;
    ConvertArrToTab(t, true); 
    cout << t << IsProxy(t) << endl;

    t = Arr("[1,2,3,{'0':[5,6,7], '1':7}]");
    cout << t << IsProxy(t) << endl;
    ConvertArrToTab(t, true, true); 
    cout << t << IsProxy(t) << endl;
  }
  
  {
    cout << "ConvertTabToArr" << endl;

    Val t = Tab();
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{'a':1, 'b':2}");
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{'0':1}");
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{0:1}");
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{1:None}");
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{-1:None}");
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{4:4, 1:None, 0:7, 3:3, 2:2}");
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{0:{0:'a','1':7}, 1:{1:7}}");
    cout << t << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << endl;

    t = Tab("{0:{0:'a','1':7}, 1:{1:7}}");
    cout << t << endl;
    cout << ConvertTabToArr(t, true) << endl;
    cout << t << endl;

    t = Tab("{0:{0:'a','1':7}, 1:{1:7}}");
    cout << t << endl;
    cout << ConvertTabToArr(t, false) << endl;
    cout << t << endl;

    t = new Tab("{0:{0:'a','1':7}, 1:{1:7}}");
    cout << t << IsProxy(t) << endl;
    cout << ConvertTabToArr(t, false) << endl;
    cout << t << IsProxy(t) << endl;

    t = Tab("{0:{0:'a','1':[{0:'s',1:'ss'}]}, 1:{1:7}}");
    cout << t << IsProxy(t) << endl;
    cout << ConvertTabToArr(t) << endl;
    cout << t << IsProxy(t) << endl;

    t = Tab("{0:{0:'a','1':[{0:'s',1:'ss'}]}, 1:{1:7}}");
    cout << t << IsProxy(t) << endl;
    cout << ConvertTabToArr(t, true) << endl;
    cout << t << IsProxy(t) << endl;

    t = Tab("{0:{0:'a','1':[{0:'s',1:'ss'}]}, 1:{1:7}}");
    cout << t << IsProxy(t) << endl;
    cout << ConvertTabToArr(t, false) << endl;
    cout << t << IsProxy(t) << endl;

    t = Tab("{0.0:1, 1:1}");
    cout << t << IsProxy(t) << endl;
    cout << ConvertTabToArr(t, false) << endl;
    cout << t << IsProxy(t) << endl;    
  }

  t.prettyPrint(cout);
  cout << t.get("a") << endl;
  cout << t.get("NOTTHERE") << endl;
  cout << t.get("a", Tab("{'a':1}")) << endl;
  cout << t.get("NOTTHERE", Tab("{'a':1}")) << endl;
  t.prettyPrint(cout);

  // test to see that you can make a Val from a size_t
  size_t szt = 17;
  Val vszt = szt;
  const Val cvszt = szt;


  // make sure you can compare Tab too!
  cout << "** Make sure you can compare Tab" << endl;
  Tab t1("{'a':1, 'b':2, 'c':3 }");
  Tab t2("{'a':1, 'c':2}");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  t1 = Tab("{'a':1, 'b':2, 'd':3 }");
  t2 = Tab("{'a':1, 'c':3, 'd':3 }");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  t1 = Tab("{'a':1, 'b':2, 'c':3 }");
  t2 = Tab("{'a':1, 'b':3, 'c':3 }");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  {
    long l = 17;
    Val vl = l;
    vl = 2;

    long ll = vl;
    ll = vl;
    cout << long(vl) << " " << ll << endl;
  }
  {
    unsigned long l = 17;
    Val vl = l;
    vl = 2;
    
    unsigned long ll = vl;
    ll = vl;
    cout << (unsigned long)(vl) << " " << ll << endl;
  }
  {
    long long l = 17;
    Val vl = l;
    vl = 2;
    
    long long ll = vl;
    ll = vl;
    cout << (long long)(vl) << " " << ll << endl;
  }
  {
    unsigned long long l = 17;
    Val vl = l;
    vl = 2;
    
    unsigned long long ll = vl;
    ll = vl;
    cout << (unsigned long long)(vl) << " " << ll << endl;
  }
}
