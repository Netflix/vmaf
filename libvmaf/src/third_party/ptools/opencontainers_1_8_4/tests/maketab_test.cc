
#include "ocmaketab.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  Tab t = Tab("{ 'a':1.0, 'b':'two', 'c': None }");
  Arr a = Arr("[1, 'two', 3.1415]");
  Val v;
  

  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v);
  v.prettyPrint(cout, 2, 2);

  v = 3.14159265f; 
  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v);
  v.prettyPrint(cout, 2, 2);

  v = Array<int_4>();
  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v);
  v.prettyPrint(cout, 2, 2);

  v = t;
  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v);
  v.prettyPrint(cout, 2, 2);

  v = a;
  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v);
  v.prettyPrint(cout, 2, 2);

  v = a;
  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v, true);
  v.prettyPrint(cout, 2, 2);
  
  a = Arr("[1,2,[4,5,6]]");
  v = a;
  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v, true, true);
  v.prettyPrint(cout, 2, 2);

  a = Arr("[1,2,[4,5,6]]");
  v = a;
  v.prettyPrint(cout, 2, 2);
  MakeTabInPlace(v, true, false);
  v.prettyPrint(cout, 2, 2);

  Tab tt = MakeTab(a);
  tt.prettyPrint(cout, 2,2);

  Tab ttt = MakeTab(1);
  ttt.prettyPrint(cout, 2,2);

  Tab tttt = MakeTab(Tab("{'a':1, 'b':2}"));
  tttt.prettyPrint(cout, 2,2);

  tttt = MakeTab(Arr("[1,2,3]"));
  tttt.prettyPrint(cout, 2,2);

  tttt = MakeTab(Arr("[1,2,3]"), true);
  tttt.prettyPrint(cout, 2,2);

  tttt = MakeTab(Arr("[1,2,3]"), true, true);
  tttt.prettyPrint(cout, 2,2);

  tttt = MakeTab(Arr("[1,2,3]"), false, true);
  tttt.prettyPrint(cout, 2,2);

}


