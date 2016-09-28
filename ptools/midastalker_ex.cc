
// This is a simple example program showing how to talk to an
// OpalPythonDaemon running on the given host and port.  Running this
// program assumes the OpalPythonDaemon is up and running on the other
// port.

// For a more complex and more robust example, see the midastalker_ex2.cc
// source file.

// Type "make -f Makefile.`uname` midastalker_ex"  to build

#include "midastalker.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main (int argc, char *argv[])
{
  // Parse the command line and get the options and arguments
  if (argc!=3) {
    cerr << "usage: " << argv[0] << " hostname portnumber" << endl;
    exit(1);
  }
  
  // Create a client but don't connect
  string host = argv[1];
  int    port = atoi(argv[2]);

  MidasTalker mt(host, port);

  mt.open(); // connect

  Val res;
  while (res != Arr("['EOF']")) {

    // Build a request to send
    Val t = Tab("{ 'request':'helpme', 6000:[1.23456789, 7000, {} ]}");

    // Send the request to OpalPythonDaemon
    mt.send(t);

    // Wait and get a response from the OpalPythonDaemon: this is a
    // blocking call
    res = mt.recv();

    // "Do something" with the result
    cout << res << endl;
  }

}
