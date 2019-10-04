
// This is an example program showing how to talk to an
// OpalPythonDaemon running on the given host and port.  Running this
// program assumes the OpalPythonDaemon is up and running on the other
// port.

// This example shows how to handle non-blocking calls as well as
// error handling and serialization options from the command line.

// Type "make -f Makefile.`uname` midastalker_ex2"  to build

#include "midastalker.h"
#include "valgetopt.h"
#include "timeconv.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// Main loop where you send and receive messages to the Midastalker (mt)
void sendloop (MidasTalker& mt)
{
  string s(256, '\0');
  for (size_t ii=0; ii<s.length(); ii++) {
    s[ii] = char(ii);
  }

  int loop = 0;
  while (1) {
    loop++; 
    // if (loop==13) exit(1);

    // Create a message (a table) and send it over to OpalPythonDaemon
    Tab t
    ="{ 'a':1, 'hello':[.3, (1+2j), True], 'next':{'cc':17}, "
       "  'na':None, 'aa':array([1.0 2.0 3.0]), 'long':1L }";
    t["proxy1"] = new Tab("{'share':1}");
    t["proxy2"] = t("proxy1");
    t["p1"] = new Arr("[1,2,3]");
    t["p2"] = t("p1");
    t["a1"] = new Array<real_8>(3);
    Array<real_8>& a = t("a1"); a.append(1.0); a.append(2.0);
    t["a2"] = t("a1"); 
    Array<int_u8>& tt = t["arr"] = Array<int_u8>(10);
    tt.append(int_u8(1)<<61);
    tt.append(int_u8(1)<<62);
    tt.append(int_u8(1)<<63);

    t["rawdata"] = s;

    if (loop % 10 == 0) {
      /*
      cerr << "About to send a REALLY BIG Table!" << endl;
      t["LARGEARRAY"] = new Array<int_1>();
      Array<int_1>& a = t("LARGEARRAY");
      a.expandTo(size_t(4294967297ULL));
      int_1* data = a.data();
      const size_t len = a.length();
      memset(data, 0, len);
      */

      /*
      cerr << "About to send a REALLY LARGE string!" << endl;
      Val& ts = t["LARGESTRING"] = "";
      OCString& sp = *((OCString*) &ts.u.a);
      sp.append(4294967297, 'c');
      cerr << sp.length() << endl; 
      */
    }
    cout << "Sending request:" << t.entries() << endl;
    mt.send(t);
    cout << "Request sent." << endl;

    // See if we can receive, wait 5 seconds
    const real_8 timeout = 5.0;
    while (1) {
      Val res = mt.recv(timeout);
      if (res == None) {
	cout << "...retrying to receive after " 
	     << timeout <<" seconds ..." << endl;
	// Maybe try to do some other work
	continue;
      } else {
	// Do something with the result
	cout << "Got result: length = " << res.length() << " typetag=" << res.tag << " data=" << res << endl;

	// M2k puts a "CONTENTS" envelope around the message: this is
	// how we can drop the envelope in O(1) time without copying
	// everything
	if (res.contains("CONTENTS")) {
	  Val& contents = res("CONTENTS");
	  Val top;
	  top.swap(contents);
	  top.swap(res);
	}

	// If data coming back has raw data key like we sent,
	// it should be same
	if (res.contains("raw data")) {
	  if (res("raw data") != s) {
	    cerr << "Raw data not going across okay!" << endl;
	    string kk = res("raw data");
	    cerr << kk.length() << endl; 
	    for (size_t ii=0; ii<kk.length(); ii++) {
	      cerr << "[" << kk[ii] << "," << int(kk[ii]) << "] ";
	    }
	    cerr << endl;
	  } else {
	    cerr << "RAW DATA OKAY!" << endl;
	  }
	}

	if (0) { // Debugging to see if we get back Proxies or not!
	  cerr << res.tag << endl;
	  //cerr << res(0).tag << res(1).tag << endl;
	  Tab& tt = res;
	  cerr << bool(tt==t) << endl;
	  cerr << is(tt("proxy1"), tt("proxy2")) << endl;
	  cerr << is(tt("p1"), tt("p2")) << endl;
	  cerr << is(tt("a1"), tt("a2")) << endl;
	  //exit(1);
	}
	break;
      }
    }
  }
}

// Try to open up the MidasTalker, in case not immediately available
void openup (MidasTalker& mt)
{
  while (1) {

    try {
      mt.open(5.0);
      cout << "Opened connection to host:" 
	   << mt.host() << " port:" << mt.port() << endl;
      break;
    } catch (const timeout& e) {
      cout << "Open TIMEOUT: is your server slow, suspended, or in a weird state? " << e.what() << endl;
      cout << "...couldn't open right away, backing off 5 seconds" << endl;
      sleep(5);
    } catch (const gaierror& e) {
      cout << "Had trouble with name: did you mistype the hostname? " << e.what() << endl;
      cout << "...couldn't open right away, backing off 5 seconds" << endl;
      sleep(5);
    } catch (const herror& e) {
      cout << "Host error? Check your DNS? " << e.what() << endl;
      cout << "...couldn't open right away, backing off 5 seconds" << endl;
      sleep(5);
    } catch (const runtime_error& e) {
      cout << "Problem: " << e.what() << endl;
      cout << "...couldn't open right away, backing off 5 seconds" << endl;
      sleep(5);
    }
  }
}


// See the README in this directory for a better description of options

void usage (char* name, const char* what)
{
  cerr << "usage: " << string(name) << " [--ser=0|1|2|-2|4|5|6|7|8] [--sock=1|2|777] [--arrdisp=0|1|2|4] [--convert=1] hostname portnumber" << endl;
  if (what) cerr << " *** Problem on the command line:" << what << endl;
  cerr << "   See the C++/README file for more information on the command line options" << endl;
  exit(1);
}


void routine (int /*signal*/) { }

int main (int argc, char *argv[])
{
  // Turn on signal handler for USR1 signal so we can test EINTR code
  // working
  struct sigaction oldact;
  struct sigaction newact;
  newact.sa_handler = routine;
  sigemptyset(&newact.sa_mask);
  newact.sa_flags = 0;
  int r = sigaction(SIGUSR1, &newact, &oldact);
  if (r<0) throw runtime_error("sigaction");


  // Parse the command line and get the options and arguments
  Arr command_line;
  try {
    Tab long_opts = "{ 'ser':0, 'sock':2, 'arrdisp':1, 'convert':0 }";
    Tab short_opts;
    command_line = ValGetOpt(argc-1, argv+1, short_opts, long_opts);
  } catch (const ParseError& pe) {
    usage(*argv, pe.what());
  }

  // Get host and port arguments from the command line
  const Arr& arguments = command_line[1];
  if (arguments.length()!=2) usage(*argv, "missing host and/or port");
  string host = arguments[0];
  int    port = int(arguments[1]);
  
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
    else if (key=="--arrdisp") array_disposition = ArrayDisposition_e(int_4(value));
    else if (key=="--convert")    convert   = bool(value);
  }

  // Can finally create a MidasTalker once we have the options
  MidasTalker mt(host, port, serialization, socket_duplex, array_disposition);
  if (convert) {
    cout << "... you have selected forced conversion mode, which means all\n"
	 << "    OTab, Tup, BigInt will be converted to Tab, Arr, Str:\n"
	 << "    This is for preserving legacy with pre-existing PTOOLS"
	 << "    installations (i.e., backwards compatibility)."
	 << endl;
    mt.compatibilityMode(convert);
  }

  while (1) {
    openup(mt);
    try {
      sendloop(mt);
    } catch (const runtime_error& e) {
      cout << "Problem: " << e.what() << endl;
      cout << "Server appears to have gone away? Attempting to reconnect" 
	   << endl;
    }
  }
  
}
