// Simple example showing how to use the MidasHTTPServer C++ class

#include "ocport.h"
#include "simplehttpserver.h"
#include "valgetopt.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// The basic HTTP server is threaded: this means that a new thread
// is created per connection (i.e, per HTTP request).  So, to write
// your own HTTPServer, you have to inherit from HTTPThreadedWorker
// to do your work:  you still inherit from MidasHTTPServer, but
// it becomes very boilerplate code, where you really only supply
// createThreadedServerWorker_

// /////////////////////////////////// Class MyHTTPWorker

// The client handles HTTP requests and just the time of day back.

class MyHTTPWorker : public ThreadedHTTPWorker {
 
 public:

  // Each one gets its own.
  MyHTTPWorker (const string& name, ThreadedServer& server) :
    ThreadedHTTPWorker(name, server)
  { }

  
  // Handle a GET request
  virtual void handleGET_ (const string& uri, const string& http_version, 
			   const OTab& headers)
  {
    // Headers have stock HTTP header information to look for
    headers.prettyPrint(cerr);

    // The uri is the "path" of the GET
    if (uri != "time") {
      httptools_.HTTPBadRequest(); // If malformed, this handles closing and reporting
                         // the right error back to the client
    } else {
      // Well-formed, make a string and put the result in!
      time_t rawtime;
      time(&rawtime);
      string result_time = ctime(&rawtime);
      httptools_.HTTPValidResponse(result_time);
    }
  }


};  // MyHTTPWorker

// /////////////////////////////////// Class MyHTTPServer

// MyHTTPServer demonstrates how to write your own MidasHTTPServer for
// communicating with MidasTalkers: This is pretty boilerplate.

class MyHTTPServer : public SimpleHTTPServer {

 public:

  // Supply the host and port
  MyHTTPServer (const string& host, int port) :
    SimpleHTTPServer(host, port)
  { }

 protected:

  // Each HTTP connection gets its own thread, and this thread
  // works on it.
  virtual ThreadedServerWorker* createThreadedServerWorker_ ()
  { return new MyHTTPWorker("myhttpworkert", *this);  } 

}; // MyServer


// See the README in this directory for a better description of options

void usage (char* name, const char* what)
{
  cerr << "usage: " << string(name) << "hostname portnumber" << endl;
  if (what) cerr << " *** Problem on the command line:" << what << endl;
  exit(1);
}

int main (int argc, char *argv[])
{
  // Get host and port arguments from the command line
  if (argc!=3) usage(*argv, "missing host and/or port");
  string host = argv[1];
  int    port = atoi(argv[2]);
  

  // After parsing command line, we can create a MidasServer
  MyHTTPServer ms(host, port);
  ms.open();
  ms.waitForMainLoopToFinish();
}
