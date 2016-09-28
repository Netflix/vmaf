
// More complex example demonstrating how to use the MidasServer
// class.  This example shows how to start a thread for each request
// and have them active waiting to handle requests.

//  There is a major difference between the C++ and Python version:
// Python doesn't have truly concurrent threads (more like pre-emptive
// single processor threads), but C++ does.  So, if you need true
// concurrency, you should be using the C++ MidasServer.

#include "ocport.h"
#include "midasserver.h"
#include "valgetopt.h"
#include "ocpermutations.h"


#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


// To write your own socket socket, you inherit from MidasServer,
// and there is a callback event whenever one of three events
// happen:  (1) New client connects 
//          (2) Client has a request/data for the server 
//          (3) Client disconnects
// The MidasServer runs in a thread (yes a *thread*) that wakes up and
// animates the callback whenever one of the above events occurs.

// In this example, whenever a new client connects, we create a new
// thread to handle the processing.  The client then sends an Arr
// (such as [1,2,3] ) and the client responds with n!  responses,
// where each of the responses are all the different permutations of
// the table (n is the length of the table, so there are n! different
// permutations).  In the case of the request being [1,2,3], the
// server would respond 6 times with [1,2,3], [2,1,3], [3,2,1],
// [1,3,2], [3,1,2], [2,3,1] then disconnect the client.


// PermutationsServer demonstrates how to write your own MidasServer for
// communicating with MidasTalkers
class PermutationServer : public MidasServer {

  public:

  // This is a simple MainLoop which serves out the permutations of
  // its data in order to a client until it finishes.  Each client
  // gets its own thread which computes permutations.

  // When we start the Permutation Main Loop, this is the data we pass in
  struct PermutationData {
    PermutationData (PermutationServer* s, int rfd, int wfd, Arr& t) :
      csp(s), 
      original_data(t), 
      read_fd(rfd), 
      write_fd(wfd), 
      forceDone_(false) 
    { }
    PermutationServer* csp;
    Arr original_data;
    int read_fd, write_fd;
    int n;
    Mutex lock;                   // Lock for checking if done early 
    volatile bool forceDone_;  // In case the client goes away early
  }; // PermutationData
  
  static void* PermutationMainLoop (void* data) 
  {
    try {
      // Capture all the relevant data in local variables, and delete
      // this stucture for passing data
      PermutationData* pd = (PermutationData*)data;
      int read_fd  = pd->read_fd; if (read_fd==-1) cerr << "Error" << endl;
      int write_fd = pd->write_fd;
      PermutationServer& server = *(pd->csp);
      Arr original_data = pd->original_data;
      
      // Let's permute the Array<Val> (aka Arr) given to us
      PermutationsT<Val> p(original_data);
      while (p.next()) { // Advance to the next Permutation
	Arr a = p.currentPermutation();    // Get the current
	
	// Check and see if we have a premature end
	{
	  ProtectScope ps(pd->lock);
	  if (pd->forceDone_) break; 
	}
	server.sendBlocking_(write_fd, a); // ... send it out
      }
      if (!pd->forceDone_) server.sendBlocking_(write_fd, "EOF");

    } catch (const exception& e) {
      // Don't want to bring down the server if the sendBlocking_ fails
      cerr << "Troubling writing back to client?  Probably disconnected:" 
	   << e.what() << " ... continuing and keeping server up." << endl;
    }
    return 0;
  }

  // Constructor: Running the PermutationServer on this host, port
  PermutationServer (const string& host, int port, Serialization_e ser,
		     SocketDuplex_e socket_duplex=DUAL_SOCKET, 
		     ArrayDisposition_e dis=AS_LIST) :
    MidasServer(host, port, ser, socket_duplex, dis)
  { }

  virtual void acceptNewClient_ (int read_fd, const SockAddr_& /*read_addr*/,
				 int write_fd, const SockAddr_& /*write_addr*/)
  {
    // Create a new thread to be associated with this client.  All we
    // know is that a socket connected, so we can't actually start 
    // the thread to start processing until we get the first request.
    OCThread* tp = new OCThread("client"+Stringize(read_fd), false);
    clientList_[read_fd] = tp;

    // And show the world that we saw the connect
    cout<<"Permutations:Connection ("<<read_fd<<","<< write_fd<<")"<<endl;
  }

  virtual void readClientData_ (int read_fd, int write_fd, const Val& data)
  {
    // Clients are mapped to their thread by their read_fd
    cout<<"Permutations:Client ("<<read_fd<<","<<write_fd<<")"<<endl;
    
    OCThread* client_thread = clientList_[read_fd];

    // Make sure legal request
    Arr t;
    try {
      t = data;  // Get the array to permute
    } catch (const logic_error& e) {
      cerr << e.what() << endl;
      cerr << "Warning!  Can only send Arrs of data to the Permutation "
	" Server.  Your request of " << data << " will be ignored." << endl;
      return;
    } 
    
    if (client_thread->started()) {
      ; // Don't do anything if we get another request and have started
    } else {
      PermutationData* pd = new PermutationData(this, read_fd, write_fd, t);
      clientData_[read_fd] = pd; 
      client_thread->start(PermutationMainLoop, pd);
    }
  }

  virtual void disconnectClient_ (int read_fd, int write_fd) 
  {
    // Simple threads: we want to make sure resources shut down a
    // little more cleanly, so are more careful and do joins so we
    // know when the thread resources are gone.
    OCThread* client_thread  = clientList_[read_fd];
    clientList_.remove(read_fd);
    PermutationData* client_data = clientData_[read_fd];
    clientData_.remove(read_fd);
    {
      ProtectScope ps(client_data->lock);
      client_data->forceDone_ = true;
    }
    // The delete client_thread does the join
    delete client_thread;
    delete client_data;

    cout << "Permutations: Client ("<<read_fd<<","<<write_fd<<")"<< endl
	 << "          disconnected." << endl;
  }

  protected:
  
  AVLHashT<int, OCThread*, 8> clientList_;
  AVLHashT<int, PermutationData*, 8> clientData_;

}; // PermutationServer


void usage (char* name, const char* what)
{
  cerr << "usage: " << string(name) << " [--ser=0|1|2|-2|4] [--sock=1|2|777] [--arrdisp=0|1|2] [--convert=1] hostname portnumber" << endl;
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
    else if (key=="--convert") convert = bool(value);
    else if (key=="--arrdisp") array_disposition = ArrayDisposition_e(int_4(value));
  }


  // Can finally create a MidasTalker once we have the options
  PermutationServer ms(host, port, 
		       serialization, socket_duplex, array_disposition);
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
