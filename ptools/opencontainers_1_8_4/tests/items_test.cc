#include "ocport.h"
#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

template <class T>
void output (const T& t)
{
  cout << "Here's the container:" << endl;
  t.prettyPrint(cout);

  Arr keys = t.keys();
  cout << "Here's the keys:" << keys << endl;

  Arr values = t.values();
  cout << "Here's the values:" << values << endl;

  Arr items = t.items();
  cout << "Here's the items:" << items << endl;
}

int main ()
{
  Tab t1;
  output(t1);
  Tab t2("{'a':1, 'b':2.2, 'c':'three'}");
  output(t2);

  OTab t3;
  output(t3);
  OTab t4("o{'a':1, 'b':2.2, 'c':'three'}");
  output(t4);

  Tab t5("{'a':1, 'b':2, 'c':'three', 'nested':{'d':3}}");
  output(t5);
  
  // This is probably what most people will write
  Arr keys  = t5.keys();
  Arr values = t5.values();
  Arr items = t5.items();
  for (size_t ii=0; ii<keys.entries(); ++ii) {
    cout << keys[ii] << endl;
    cout << values[ii] << endl;
    cout << "key:" << items[ii][0] << " value:" << items[ii][1] << endl;
  }
}
