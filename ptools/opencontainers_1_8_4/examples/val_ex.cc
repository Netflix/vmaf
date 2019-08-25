
#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  cout << "Constructing some Arrays" << endl;
  Arr a;
  Arr empty_list = "[]";
  if (a!=empty_list) throw logic_error("What?");

  Arr c = "[1,2,3, '100']";
  cout << c << endl;
  c[3] = 100.1;
  cout << c << endl;

  c = "[ {}, {}, (1+2j) ]";
  cout << c << endl;
  
  cout << "Constructing some Tabs" << endl;
  Tab t;
  Tab empty = "{ }";
  if (t!=empty) throw logic_error("What?");

  // Give me a slightly more complex table
  Tab tt = " {'a':1, 'b':2, 'c':3, 'anchovies':'no' }";

  cout << tt << endl;

  cout << tt << " " << tt["a"] << endl;
  tt["a"] = 17;
  cout << tt << " " << tt["a"] << endl;
  // Another gotcha ... 'a' is  small int in C++ but a string in Python
  tt['a'] = "a small int";
  tt["a"] = "a string";
  cout << tt << endl;

  // CAREFUL! This will construct a STRING
  Val vv = " { 'a':1, 'b':2 }";
  // ... THIS will construct a table for you
  Val vv2 = Tab("{ 'a':1, 'b':2 }");

  cout << "This is a string:" << vv << endl 
       << " ... and this is a Tab:" << vv2 << endl;


  tt = "{ 'a':444, 10:100, 'dog':'pet', 'b':2, 'c':3, 'anchovies':'no' }";
  cout << tt << endl;

  cout << "Plain iteration through a table" << endl;
  for (It ii(tt); ii(); )
    cout << ii.key() << " " << ii.value() << endl;
  cout << "Sorted iteration through a table" << endl;
  for (Sit ii(tt); ii(); )
    cout << ii.key() << " " << ii.value() << endl;


  // Pretty Printing
  tt.prettyPrint(cout);
  tt = "{ 'a': { 0: 'zero', 1:'one', 2:'two', 3.141592: { 0:3, 1:.3 } }, 'b': 2 }";
  tt.prettyPrint(cout);

  c.prettyPrint(cout); 
  vv2.prettyPrint(cout);


  // Appending
  Arr app;
  app.append(3.3);
  app.append("hello");
  app.prettyPrint(cout);

  Val aaa = app;
  aaa.prettyPrint(cout);

  Tab t9;
  t9.append("zero");
  t9.append(1.1);
  t9.append(2);
  t9.prettyPrint(cout);

  Array<int> aint;
  Val vaint = aint;
  vaint.append(1);
  vaint.append("200");
  cout << vaint << endl;

  // None constant
  vaint = None;
  cout << vaint << endl;
  t9.append(None);
  t9[None] = None;
  t9.prettyPrint(cout);

  t9 = "{ }";
  t9.prettyPrint(cout);

  Val t10 = Tab();
  t10.append(111);
  t10.append(222);
  cout << t10 << endl;
  t10.appendStr(333);
  cout << t10 << endl;

  Val t11 = Tab();
  t11.appendStr(111);
  t11.appendStr(222);
  cout << t11 << endl;
  t11.append(333);
  cout << t11 << endl;

  Val a11 = Arr();
  a11.append(111);
  a11.append(222);
  cout << a11 << endl;
  a11.appendStr(333);
  cout << a11 << endl;

}
