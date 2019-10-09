
#include "ocport.h"

#define MIDAS_COMPILER_TEMPLATES
#include "valpython.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


using namespace std;

int main (int argc, char* argv[])
{
  if (argc != 4) {
    cerr << "Usage: filename-to-read 1(uses Numeric)|0(no Numeric) 1(compat)|0(no compat)" << endl;
    cerr << " ... compatibility mode: 1 means that Tuples and OTabs serialize like Arrs and Tabs (resp.), 0 means that Tuples and OTabs serialize as Python Tuples and Python OrderedDicts (resp.)." << endl;
    return 0;
  }

  string name = argv[1];
  cout << "Reading " << argv[1] << endl;
  bool uses_numeric = (argv[2][0]=='1');
  bool compat = (argv[3][0]=='1');
  cerr << "compat is " << compat << endl;

  Val v;
  {
    PythonDepickler<Val> pp(name, uses_numeric);  
    pp.compatibility(compat);
    v = pp.load();
  }
  cout << v << endl;
  //v.prettyPrint(cout);
  return 1;
}
