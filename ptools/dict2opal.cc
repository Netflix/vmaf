
#include "ocport.h"  // handle stream stuff for us

#define MIDAS_COMPILER_TEMPLATES
#include "opalprint.h"
#include "ocvalreader.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


using namespace std;

int main (int argc, char* argv[])
{
  if (argc != 2) {
    cerr << "Usage: filename-to-read " << endl;
    return 0;
  }

  string name = argv[1];
  cerr << "Reading " << argv[1] << endl;

  try {
    Tab t;
    ReadTabFromFile(name, t);
    prettyPrintOpal(t, cout);
  } catch (const exception& e) {
    cerr << e.what() << endl;
  }
  return 1;
}
