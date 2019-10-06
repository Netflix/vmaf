
#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

void PrintTupAndCreateFromString (const Tup& t)
{
  cout << t << endl;
  t.prettyPrint(cout);

  cout << " .. reversing process" << endl;
  string s = Stringize(t);
  cout << " ... stringized tup:" << s << endl;
  Tup o_fromstr;
  o_fromstr = Eval(s);
  cout << "  from stream:" << o_fromstr << endl;
    
  ostringstream os;
  t.prettyPrint(os);
  string ss = os.str();
  Tup o_ss;
  o_ss = Eval(ss);
  cout << "  from prettyPrint:";
  o_ss.prettyPrint(cout);
  cout << endl;

  for (size_t ii=0; ii<t.entries(); ii++) {
    cout << t[ii] << endl;
    cout << t(ii) << endl;
  }

}

int main()
{
  Val uu = Tup();

  cout << "cx? u" << OC_IS_CX(uu) << endl; 
  cout << "int? u" << OC_IS_INT(uu) << endl; 
  cout << "uint? u" << OC_IS_UINT(uu) << endl; 
  cout << "is numeric? u" << OC_IS_NUMERIC(uu) << endl; 

  Tup u1 = Eval("(1,2,3)"); // Tuple within a Tuple: ((1,2,3))
  cout << u1 << endl;
  Tup u2;
  u2 = Eval("(1,2,3)");    // (1,2,3)
  cout << u2 << endl;
  Tup u3 = u2;             // Tup copy works as expected: (1,2,3)
  cout << u3 << endl;

  Tup t;
  PrintTupAndCreateFromString(t);
  t = Tup(1);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3,4);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3,4,5);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3,4,5,6);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3,4,5,6,7);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3,4,5,6,7,8);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3,4,5,6,7,8,9);
  PrintTupAndCreateFromString(t);
  t = Tup(1,2,3,4,5,6,7,8,9,10);
  PrintTupAndCreateFromString(t);

  try {
    cout << t[t.entries()] << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  //try {
  //  cout << t(t.entries()) << endl;
  //} catch (const exception& e) {
  //  cout << "Expected exception:" << e.what() << endl;
  //}

  // make sure you can compare Tup too!
  cout << "** Make sure you can compare Tup" << endl;
  Tup t1(1,2,3);
  Tup t2(1,2);
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  t1 = Tup("a","b","c");
  t2 = Tup("a","c","d");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  t1 = Tup("a", Tup("b",2), "c");
  t2 = Tup("a", Tup("b",3), "c");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;

  t1 = Tup("a", Tup("b",3), "c");
  t2 = Tup("a", Tup("b",3), "c");
  cout << t1 << " " << t2 << endl;
  cout << bool(t1==t2) << endl;
  cout << bool(t1!=t2) << endl;
  cout << bool(t1<t2) << endl;
  cout << bool(t1<=t2) << endl;
  cout << bool(t1>=t2) << endl;
  cout << bool(t1>t2) << endl;
  

  // make sure the proxy works
  cout << "** Testing proxies" << endl;
  Proxy px = new Tup(1,2,3);
  Tup& tp = px;
  cout << tp[0] << tp[1] << tp[2] << endl;

  Val pp = new Tup(4,5,6);
  Tup& tpp = pp;
  cout << tpp[0] << tpp[1] << tpp[2] << endl;
  
  // Can't use anything other than ints for subscript into Tuple
  try {
    cout << pp["string"] << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }
  try {
    cout << pp[1.1] << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }
  try {
    cout << pp[complex_8(1,2)] << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  try {
    tpp.remove(0);
    cout << pp[complex_8(1,2)] << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  try {
    tpp.append(0);
    cout << pp[complex_8(1,2)] << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  // But you can change a single element already there
  tpp[0] = "change";
  cout << tpp << endl;

  cout << tpp.contains("change")  << endl;
  cout << tpp.contains(5) << endl;
  cout << tpp.contains(0) << endl;
  cout << Tup(1,2,Tup(0,0)).contains(0) << endl;  // not nested search!

  cout << "** Test cascading lookup and changes" << endl;
  pp[0] = Tup(1,2,3);
  cout << pp << endl;
  pp[0][0] = 17;
  cout << pp[0] << endl;     // Tuple
  cout << pp[0][0] << endl;  // 0
  try {
    cout << pp[17] << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }
  try {
    pp[17] = 17;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }
  try {
    cout << pp[0][0][0] << endl; // exception
  } catch (const exception& e) {
    cout << e.what() << endl;
  }

  cout << "**Test operator(), which should throw exceptions if not there" << endl;
  pp(0) = Tup(1,2,3);
  cout << pp << endl;
  pp(0)(0) = 17;
  cout << pp(0) << endl;     // Tuple
  cout << pp(0)(0) << endl;  // 0
  try {
    cout << pp(17) << endl;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }
  try {
    pp(17) = 17;
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  try {
    cout << pp(0)(0)(0) << endl; // exception
  } catch (const exception& e) {
    cout << "Expected exception:" << e.what() << endl;
  }

  cout << "** Should use for loops to iterate through, but can use It" << endl;
  Tup y("a","b","c");
  for (It ii(y); ii(); ) {
    cout << ii.key() << " " << ii.value() << endl;
  }

  Tup yv("d","e","f");
  for (It ii(yv); ii(); ) {
    cout << ii.key() << " " << ii.value() << endl;
  }
}
