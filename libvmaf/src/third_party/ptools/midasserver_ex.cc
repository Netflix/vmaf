// Simple example showing how to use the MidasServer C++ class

#include "ocport.h"
#include "midasserver.h"
#include "valgetopt.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// MyServer demonstrates how to write your own MidasServer for
// communicating with MidasTalkers

class MyServer : public MidasServer {

  public:

  MyServer (bool have_server_send_mesg_at_connect,
	    string host, int port, Serialization_e ser, 
	    SocketDuplex_e socket_duplex=DUAL_SOCKET, 
	    ArrayDisposition_e dis = AS_LIST) :
    MidasServer(host, port, ser, socket_duplex, dis),
    count_(0),
    haveServerSendMessageAtConnect_(have_server_send_mesg_at_connect)
  { }

  virtual void acceptNewClient_ (int read_fd, const SockAddr_& read_addr,
				 int write_fd, const SockAddr_& write_addr)
  {
    cout << "MYSERVER: Connection (" <<read_fd<<","<< write_fd<<")" << endl  
	 << "          made from:" << read_addr.addr.sa_data << " " << write_addr.addr.sa_data 
	 << endl;

    // For test purposes, send a message back immediately at connect
    if (haveServerSendMessageAtConnect_) {
      cout << "Sending a test message at connect time:  This establishes\n" 
	   << " the serialization for the session. If you wish to let the\n" 
	   << " client set the serialization, don't send this message.\n" 
	   << endl;
      try {
	Val v = Tab("{ 'TEST': 'at connect', 'a':[1,2,3] }");
	sendBlocking_(write_fd, v);
      } catch (exception& e) {
	// Don't want to bring down the server if the send fails
	cerr << "Troubling writing back to client?  Probably disconnected:"
	     << e.what() << " ... continuing and keeping server up." << endl; 
      }
    }
  }

  virtual void readClientData_ (int read_fd, int write_fd, const Val& data)
  {
    cout << "MYSERVER: Client (" << read_fd << "," << write_fd << ")" << endl
	 << "          saw data:" << data << endl;
    // Send the same data back to the client who sent it
    try {
      sendBlocking_(write_fd, data);
    } catch (exception& e) {
      // Don't want to bring down the server if the send fails
      cerr << "Troubling writing back to client?  Probably disconnected:" 
	   << e.what() << " ... continuing and keeping server up." << endl;
    }
    cout << " ... and send the same data back!" << endl;
    
    // Show how to allow shutdown
    int max_count = 10000;
    count_+=1;
    if (count_>max_count) {
      cout << "... saw " << max_count << " messages ... shutting down" << endl;
      shutdown();
    }
  }

  virtual void disconnectClient_ (int read_fd, int write_fd) 
  {
    cout << "MYSERVER: Client (" << read_fd << "," << write_fd << ")" << endl
	 << "          disconnected." << endl;
  }

  protected:

   int count_;  // Dumb example, shutdown after 10 messages

  // If the server sends a message at connect time, that establishes
  // the serialization of the session, which you may or may not want.
  // If you wish the client to establish the serialization of the session,
  // then set this to false.  Otherwise, the host always does.
  bool haveServerSendMessageAtConnect_;

}; // MyServer


// See the README in this directory for a better description of options

void usage (char* name, const char* what)
{
  cerr << "usage: " << string(name) << " [--ser=0|1|2|-2|4|5|6|7|8] [--sock=1|2|777] [--arrdisp=0|1|2] [--convert=1] [--server_send_message_at_connect=0|1] hostname portnumber" << endl;
  if (what) cerr << " *** Problem on the command line:" << what << endl;
  cerr << "   See the C++/README file for more information on the command line options" << endl;
  exit(1);
}

int main (int argc, char *argv[])
{
  // Parse the command line and get the options and arguments
  Arr command_line;
  try {
    Tab long_opts = "{ "
      " 'ser':0,"
      " 'sock':2, "
      " 'arrdisp':1,"
      " 'convert': 0,"
      " 'server_send_message_at_connect':1 "
      "}";
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
  int_4 server_send = 1;
  bool convert = false;

  const Tab& options = command_line[0];
  It ii(options); 
  while (ii()) { 
    const Val& key = ii.key();
    const Val& value = ii.value(); 
    if      (key=="--ser")     serialization = Serialization_e(int_4(value));
    else if (key=="--sock")    {
      Tab options = "{ '1':0, '2':1, '777':777 }";
      int_4 enum_val = options(value);
      socket_duplex = SocketDuplex_e(enum_val);
    } else if (key=="--arrdisp") array_disposition = ArrayDisposition_e(int_4(value));
    else if (key=="--server_send_message_at_connect") server_send=int_4(value);
    else if (key=="--convert")    convert   = bool(value);
  }


  // After parsing command line, we can create a MidasServer
  MyServer ms(server_send,
	      host, port, serialization, socket_duplex, array_disposition);
  if (convert) {
    cout << "... you have selected forced conversion mode, which means all\n"
	 << "    OTab, Tup, BigInt will be converted to Tab, Arr, Str:\n"
	 << "    This is for preserving legacy with pre-existing PTOOLS"
	 << "    installations (i.e., backwards compatibility)."
	 << endl;
    ms.compatibilityMode(convert);
  }
  ms.open();

  ms.waitForMainLoopToFinish();
}
