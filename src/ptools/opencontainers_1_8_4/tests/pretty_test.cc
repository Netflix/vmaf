
#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

int main()
{
  Arr a;
  a.prettyPrint(cout);
  a.prettyPrint(cout, 2);
  a.prettyPrint(cout, 2, 2);

  a.prettyPrint(cout);
  a.prettyPrint(cout, 1);
  a.prettyPrint(cout, 1, 1);

  a = Arr("[1,'two',3.0]");
  a.prettyPrint(cout);
  a.prettyPrint(cout, 2);
  a.prettyPrint(cout, 2 ,2);

  a.prettyPrint(cout);
  a.prettyPrint(cout, 1);
  a.prettyPrint(cout, 1 ,1);


  Tab t;
  t.prettyPrint(cout);
  t.prettyPrint(cout, 2);
  t.prettyPrint(cout, 2, 2);

  t.prettyPrint(cout);
  t.prettyPrint(cout, 1);
  t.prettyPrint(cout, 1, 1);

  t = Tab("{'a':1, 'b':None, 'c': 3.14159265}");
  t.prettyPrint(cout);
  t.prettyPrint(cout, 2);
  t.prettyPrint(cout, 2, 2);

  t.prettyPrint(cout);
  t.prettyPrint(cout, 1);
  t.prettyPrint(cout, 1, 1);

  t["array"] = a;
  t.prettyPrint(cout);
  t.prettyPrint(cout, 2);
  t.prettyPrint(cout, 2, 2);

  t.prettyPrint(cout);
  t.prettyPrint(cout, 1);
  t.prettyPrint(cout, 1, 1);

  t = Tab("{ 'a': nan, 'b': inf, 'c':-inf }");
  t.prettyPrint(cout);

  real_8 inf = t["b"];
  real_8 ninf = t["c"];
  real_8 nan = t["a"];
  cout << inf << " " << ninf << " " << nan << endl;
  t.prettyPrint(cout);

  try { 
    t = Tab("{ 'a': NaN, 'b': inf, 'c':-inf }");
  } catch (const exception& e) {
    cout << "Expected exception .. NaN is bad repr: has to be all lower case" << endl;
  }
  
  try {
    t = Tab("{ 'c':- inf }");
  } catch (const exception& e) {
    cout << "Expected exception .. no space after -" << endl;
  }

  t = Tab("{ 'a':+inf }");
  t.prettyPrint(cout);
}
