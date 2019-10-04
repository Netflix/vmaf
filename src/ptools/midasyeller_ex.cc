
// This is a simple example of how to build a MidasYeller (a server
// that talks to a list of given MidasListeners with UDP sockets).
// Creating the MidasYeller only sets it up: when you 'addListener' is
// when you actually add Clients.  Since it's UDP, a send to a client
// may or may not get there, but will probably be very fast.


#include "ocport.h"         // handle stream stuff
#include "midasyeller.h"
#include "valgetopt.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif



// See the README in this directory for a better description of options

void usage (char* name, const char* what)
{
  cerr << "usage: " << string(name) << " [--ser=0|1|2|-2|4|5|6|7|8] [--arrdisp=0|1|2] [--convert=1] hostname1 portnumber1 [hostname2 portnumber2] ... " << endl;
  if (what) cerr << " *** Problem on the command line:" << what << endl;
  cerr << "   See the C++/README file for more information on the command line options" << endl;
  exit(1);
}

int main (int argc, char *argv[])
{
  // Parse the command line and get the options and arguments
  Arr command_line;
  try {
    Tab long_opts = "{ 'ser':0, 'arrdisp':1, 'convert':0 }", short_opts;
    command_line = ValGetOpt(argc-1, argv+1, short_opts, long_opts);
  } catch (const ParseError& pe) {
    usage(*argv, pe.what());
  }

  // Get host and port arguments from the command line
  const Arr& arguments = command_line[1];
  if (arguments.length()<2 || arguments.length()%2!=0) {
    usage(*argv, "missing host port pairs");
  }
  
  // Get options from the command line
  Serialization_e serialization = SERIALIZE_P0;   
  ArrayDisposition_e array_disposition = AS_LIST;
  bool convert = false;

  const Tab& options = command_line[0];
  It ii(options); 
  while (ii()) { 
    const Val& key = ii.key();
    const Val& value = ii.value(); 
    if      (key=="--ser")     serialization = Serialization_e(int_4(value));
    else if (key=="--arrdisp") array_disposition = ArrayDisposition_e(int_4(value));
    else if (key=="--convert") convert = bool(value);
  }

  
  // Got options, still getting arguments from command line
  MidasYeller my(1024, serialization, array_disposition);
  if (convert) {
    cout << "... you have selected forced conversion mode, which means all\n"
	 << "    OTab, Tup, BigInt will be converted to Tab, Arr, Str:\n"
	 << "    This is for preserving legacy with pre-existing PTOOLS"
	 << "    installations (i.e., backwards compatibility)."
	 << endl;
    my.compatibilityMode(convert);
  }

  for (size_t x=0; x<arguments.length(); x+=2) {
    string host = arguments[x];
    int port    = arguments[x+1];
    cout << "Adding:" << host << " " << port << endl;
    my.addListener(host, port);
  }

  for (int x=0; x<1e6; x++) {
    string xx = Stringize(x);
    Tab data = "{'hello there':"+xx+", "
      "'This is really  kind of a big table': "+xx+"}";
    my.send(data); 
    cout << "Just sent UDP message " << x << endl;
  }
      
}




