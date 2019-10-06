
#include "ocval.h"
#include "occonvert.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

int main()
{
  Val oo = OTab();

  // Verify tags tests still work
  cout << "cx? o" << OC_IS_CX(oo) << endl; 
  cout << "int? o" << OC_IS_INT(oo) << endl; 
  cout << "uint? o" << OC_IS_UINT(oo) << endl; 
  cout << "is numeric? o" << OC_IS_NUMERIC(oo) << endl; 

  cout << "Empty Otab has tag:" << oo.tag << endl;
  cout << " .. length=" << oo.length() << endl;
  cout << " .. entries=" << oo.entries() << endl;

  oo["one"] = 1;
  oo["two"] = 2.2;
  oo["three"] = "three";

  for (It ii(oo); ii(); ) {
    cout << ii.key() << ":" << ii.value () << endl;
  }

  try {
    oo("not here") = 17;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  cout << "*When keys there, () and [] act same on lookup" << endl;
  cout << oo("one") << " " << oo("two") << " " << oo("three") << endl;
  cout << oo["one"] << " " << oo["two"] << " " << oo["three"] << endl;

  cout << "*Two ways to print: normal and pretty" << endl;
  cout << oo << endl;
  oo.prettyPrint(cout);

  static const char* init[] = { "o{}", 
				"o{'a':1}",
				"o{'a':1,}",
				"o{ 'a':o{'a':1, 'b':2, 'c':3}, 'k':o{ } }" ,
				"o{ 'a':o{'a':1, 'b':2, 'c':3}, 'k':None }" ,
				"OrderedDict([])",
				"OrderedDict([('a',1)])",
				"OrderedDict([('a',1,)])",

			  NULL };
  for (int ii=0; init[ii]!=NULL; ii++) {

    cout << "** Trying literal:" << init[ii] << endl;
    OTab o(init[ii]);
    cout << o << endl;
    o.prettyPrint(cout);

    cout << " .. reversing process" << endl;
    string s = Stringize(o);
    OTab o_fromstr(s);
    cout << "  from stream:" << o_fromstr << endl;
    
    ostringstream os;
    o.prettyPrint(os);
    string ss = os.str();
    OTab o_ss(ss);
    cout << "  from prettyPrint:";
    o_ss.prettyPrint(cout);
    cout << endl;
  }
  
  cout << "**Append" << oo << endl;
  oo.append("hello");
  cout << "** after" << oo << endl;
  oo.remove("one");
  cout << "** after" << oo << endl;
  try {
    oo.append("hello again");
  } catch (exception& e) {
    cout << "Expected Exception:" << e.what() << endl;
  }

  for (It it(oo); it(); ) {
    cout << it.key() << " " << it.value() << endl;
  }

  // make sure you can compare OTab too!
  cout << "** Make sure you can compare OTab" << endl;
  OTab t1("o{'a':1, 'b':2, 'c':3 }");
  OTab t2("o{'a':1, 'c':2}");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  t1 = OTab("o{'a':1, 'b':2, 'd':3 }");
  t2 = OTab("o{'a':1, 'c':3, 'd':3 }");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  t1 = OTab("o{'a':1, 'b':2, 'c':3 }");
  t2 = OTab("o{'a':1, 'b':3, 'c':3 }");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;


  // IN python, you can compare Dicts and OrderedDict,
  // so you can compare Tab and OTab
  cout << "** Make sure you can compare OTab and Tab together like Python" << endl;
  Tab t("{'a':1, 'b':2}");
  OTab ot("o{'a':1, 'b':2}");
  cout << bool(t==ot) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t<ot) << endl;
  cout << bool(t<=ot) << endl;
  cout << bool(t>=ot) << endl;
  cout << bool(t>ot) << endl;


  cout << "** Testing get" << endl;
  OTab ooo("o{'a':1, 'b':2}");
  Val v = ooo.get("a");
  cout << v << endl;
  v = ooo.get("aa");
  cout << v << endl;
  v = ooo.get("a", 17);
  cout << v << endl;
  v = ooo.get("aa", 17);
  cout << v << endl;

  cout << "**Testing append" << endl;
  ooo = OTab();
  for (int ii=0; ii<3; ii++) {
    ooo.append(ii);
  }
  cout << ooo << endl;
  ooo.remove(1);
  cout << ooo << endl;
  try {
    ooo.append(3);
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  cout << "**Try out the conversions:" << endl;
  OTab ott = "o{'a':1, 'b':2, 'c':3.3 }";
  Tab tt;
  ConvertOTabToTab(ott,tt);
  cout << ott << " " << tt << endl;

  ott = OTab();
  tt = "{'a':1, 'b':2, 'c':3.3}";
  ConvertTabToOTab(tt,ott);
  cout << ott << " " << tt << endl;

  ott = "o{'a':1, 'b':2, 'c':3 }";
  tt = Tab();
  SwapInto(ott,tt);
  cout << ott << " " << tt << endl;

  ott = OTab();
  tt =  "{'a':1, 'b':2, 'c':3 }";
  SwapInto(tt, ott);
  cout << ott << " " << tt << endl;

  Val tv = OTab("o{ 0:'a', 1: 'b', 2:'c', 3:'d', 4:(1,2,{0:1,1:2}), 5:o{'a':{0:1,1:3}, 'b':{'0':'yay', '1':'boo'}}}");
  tv.prettyPrint(cout);
  ConvertAllTabsToArrs(tv);
  tv.prettyPrint(cout);

  tv = Tab("{ 0:'a', 1: 'b', 2:'c', 3:{'0':'yay', '1':'boo'}}");
  tv.prettyPrint(cout);
  ConvertAllTabsToArrs(tv);
  tv.prettyPrint(cout);

  Val at = Arr("[1,2,(1,2,[1,2,3],o{'a':[1,2,3]})]");
  ConvertAllArrsToTabs(at);
  at.prettyPrint(cout);

  {
    Val v;
    cout << v << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;

    v = Tab("{'a':1, 'b':2, 'c':3}");
    cout << v << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;

    v = Arr("[1,2,3]");
    cout << v << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;

    v = OTab("o{'a':1, 'b':2, 'c':3}");
    cout << v << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;

    v = Tup(1,2,3);
    cout << v << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;

    v = Tup(1,2,3);
    cout << v << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;

    v = Tup(1,Tup(1,2,3),OTab("o{'a':1, 'b':(1,2,3), 'c':o{ 1:1, 2:2} }"));
    cout << v << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;

    // amke sure proxyness is preserved
    Val k = new OTab("o{'a':1, 'b':2}");
    v = Tup(k,k);
    cout << v << endl;
    cout << IsProxy(v[0]) << IsProxy(v[1]) << is(v[0], v[1]) << endl;
    ConvertAllOTabTupBigIntToTabArrStr(v);
    cout << v << endl;
    cout << IsProxy(v[0]) << IsProxy(v[1]) << is(v[0], v[1]) << endl;
  }

  {
    cout << "*** Trying AsList" << endl;
    Val v = 100;
    Val result;
    AsList(v, result);
    cout << v << " " << result << endl;

    v = Arr("[1,2,3]");
    AsList(v, result);
    cout << v << " " << result << endl;

    v = Tab("{ 'a': [1,2,3], 'b':{'q':'bert'} }");
    AsList(v, result);
    cout << v << " " << result << endl;

    v = OTab("o{ 'a': [1,2,3], 'b':{'q':'bert'} }");
    AsList(v, result);
    cout << v << " " << result << endl;

    v = Array<int_4>(3);
    Array<int_4>& aa = v;
    aa.fill(5);
    aa[2]=666;
    AsList(v, result);
    cout << v << " " << result << endl;
  }
}
