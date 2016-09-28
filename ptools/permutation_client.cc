
// This is a MidasTalker client that gets permutations back from the
// PermutationServer (an example MidasServer).

// This example shows how to handle non-blocking calls as well as
// error handling.

#include "midastalker.h"
#include "valgetopt.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


// Main loop where you send and receive messages to the Midastalker (mt)
void sendloop (MidasTalker& mt, Arr& data)
{
  while (1) {

    // Create an Arr to send that the permutation server will permute
    cout << "Sending request:" << data << endl;
    mt.send(data);

    // See if we can receive, wait 5 seconds
    while (1) {
      Val res = mt.recv(5.0);
      if (res == None) {
	cout << "...retrying to receive after 5 seconds ..." << endl;
	// Maybe try to do some other work
	continue;
      } else {
	// Do something with the result
	if (res=="EOF") return;
	cout << res << endl;
      }
    }
  }
}

// Try to open up the MidasTalker, in case not immediately available
void openup (MidasTalker& mt)
{
  while (1) {

    try {
      mt.open();
      cout << "Opened connection to host:" 
	   << mt.host() << " port:" << mt.port() << endl;
      break;
    } catch (const runtime_error& e) {
      cout << "Problem: " << e.what() << endl;
      cout << "...couldn't open right away, backing off 5 seconds" << endl;
      sleep(5);
    }
  }
}


void usage (char* name, const char* what)
{
  cerr << "usage: " << name << " [--ser=0|1|2|-2|4|5] [--sock=1|2|777] [--arrdisp=0|1|2] [--convert=1] hostname portnumber array_to_permute" << endl;
    cerr << "  An example invocation:" << endl;
    cerr << "    permutation_client --ser=2 localhost 9111 \"[1,2,3]\"" << endl;
  if (what) cerr << " *** Problem on the command line:" << what << endl;
   cerr << "   See the C++/README file for more information on the command line options" << endl;

  exit(1);
}

int main (int argc, char *argv[])
{
  // Parse the command line and get the options and arguments
  Arr command_line;
  try {
    Tab long_opts = "{ 'ser':0, 'sock':2, 'arrdisp':1, 'convert':0 }", short_opts;
    command_line = ValGetOpt(argc-1, argv+1, short_opts, long_opts);
  } catch (const ParseError& pe) {
    usage(*argv, pe.what());
  }

  // Get host and port arguments from the command line
  const Arr& arguments = command_line[1];
  if (arguments.length()!=3) {
    usage(*argv, "missing host,port and/or array to permute");
  }
  string host = arguments[0];
  int    port = int(arguments[1]);
  string the_data = arguments[2];
  Arr    data(the_data); // Evaluate string to turn into Arr
  
  // Get options from the command line
  Serialization_e serialization = SERIALIZE_P0;   
  SocketDuplex_e socket_duplex = DUAL_SOCKET;
  ArrayDisposition_e array_disposition = AS_LIST;
  bool convert = false;

  const Tab& options = command_line[0];
  It ii(options); 
  while (ii()) { 
    const Val& key = ii.key();
    const Val& value = ii.value(); 
    if      (key=="--ser")     serialization = Serialization_e(int_4(value));
    else if (key=="--sock") {
      Tab options = "{ '1':0, '2':1, '777':777 }";
      int_4 enum_val = options(value);
      socket_duplex = SocketDuplex_e(enum_val);
    }
    else if (key=="--convert") convert = bool(value);
    else if (key=="--arrdisp") array_disposition = ArrayDisposition_e(int_4(value));
  }


  // After parsing command line, can create 
  MidasTalker mt(host, port, serialization, socket_duplex, array_disposition);
  if (convert) {
    cout << "... you have selected forced conversion mode, which means all\n"
	 << "    OTab, Tup, BigInt will be converted to Tab, Arr, Str:\n"
	 << "    This is for preserving legacy with pre-existing PTOOLS"
	 << "    installations (i.e., backwards compatibility)."
	 << endl;
    mt.compatibilityMode(convert);
  }

  openup(mt);
  try {
    sendloop(mt, data);
  } catch (const runtime_error& e) {
    cout << "Problem: " << e.what() << endl;
    cout << "Server appears to have gone away?  Quitting" 
	 << endl;
  }
  
}
