
#include "ocport.h"  // handle stream stuff

#define MIDAS_COMPILER_TEMPLATES
#include "opalutils.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


using namespace std;

int main (int argc, char* argv[])
{
  if (argc != 3) {
    cerr << "Usage: opal2dict filename-to-read convert_tabs_to_arrs_if_possible=0|1" << endl;
    return 0;
  }

  string name = argv[1];
  bool   convert_tab_to_arr = atoi(argv[2]);
  cerr << "Reading " << argv[1] << " " << convert_tab_to_arr << endl;

  try {
    Val v; 
    ReadValFromOpalFile(name, v, convert_tab_to_arr);
    v.prettyPrint(cout);
  } catch (const exception& e) {
    cerr << e.what() << endl;    
  }

  return 1;
}
