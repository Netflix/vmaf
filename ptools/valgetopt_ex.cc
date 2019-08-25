
#include "valgetopt.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// Demo to show how to use the VelGetOpt

int main (int argc, char** argv) 
{
  // -a is just a switch, -b expects a number so, -b=1 or -b 1
  Tab short_args="{ 'a': None, 'b':1 }"; 
  
  // --anon  is just a switch with no value, --bitnumber=1 or --bitnumber 1
  Tab long_args ="{ 'anon': None, 'bitnumber': 1 }"; 

  // Get the options parsed out
  Arr res;
  try {
    // Typical usage is to ignore the program name in argv[0]
    res = ValGetOpt (argc-1, argv+1, short_args, long_args);
  } catch (const ParseError& pe) {
    cerr << "Usage: valgetopt_ex [-a or --anon] [-b=1 or --bitnumber=n] file1 file2 ..." << endl;
    cerr << " Problem with your command line: " << pe.what() << endl;
    exit(1);
  }
  Tab& opts = res[0];  // The options in a table
  cout << "OPTIONS are: " << opts << endl;

  Arr& args = res[1];  // The arguments in order
  cout << "ARGUMENTS are: " << args << endl;

  // Here's an example how you might parse the options and set values
  bool anonymous_user = false;
  int_4 bit_number = -1;
  
  It ii(opts);
  while (ii()) {
    const Val& key   = ii.key();
          Val& value = ii.value();
    if      (key=="--anon" || key=="-a")       anonymous_user = true;
    else if (key=="--bitnumber" || key=="-b")  bit_number = value;
  }
  cout << "**HERE ARE HOW YOUR OPTIONS GOT SET:" << endl;
  cout << "anonymous user? " << anonymous_user << endl;
  cout << "bit_number : " << bit_number << endl;

  // Here's an example of how you might deal with the actual arguments
  cout << "**ARGUMENTS HANDLING" << endl;
  for (size_t ii=0; ii<args.length(); ii++) {
    cout << "Argument " << ii << " handling: " << args[ii] << endl; 
  }

  return 0;
}
