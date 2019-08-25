#ifndef SIMPLEHTTPSERVER_H_
#define SIMPLEHTTPSERVER_H_

// An HTTP Server that implements a subset of the HTTP protocol, with
// one thread per connection.  The threaded server, which we inherit from,
// handles most of the connection set-up and the like,
// but the real work is handled in the ThreadedHTTPWorker, which does
// all the parsing of the HTTP and reading/writing from the socket.

#include "threadedserver.h"
#include "httptools.h"
#include "ocsplit.h"

PTOOLS_BEGIN_NAMESPACE

// /////////////////////////////////////// ThreadedHTTPWorker

// One thread per connection: each thread parse the HTTP data from the
// given file descriptor.
class ThreadedHTTPWorker : public ThreadedServerWorker {
  
 public: 

  // Put into empty, known state and start into main loop.
  ThreadedHTTPWorker (const string& name, ThreadedServer& server): 
    ThreadedServerWorker(name, server),
    httptools_(readfd_, writefd_),
    logging_(true),
    httpVersion_(""),
    serverName_("PTOOLSHTTPServer")
  { }

  // Set-up the worker 
  virtual void initialize (int readfd, int writefd)
  {
    httptools_.cleanseBuffer();
    ThreadedServerWorker::initialize(readfd, writefd); // Call parent
  }


 protected:

  // Tools for parsing, reading, responding and making well-formed
  // requests
  HTTPTools_ httptools_;

  // Do we log to cerr?  TODO: more extensive log
  bool logging_;

  // Use same version as input request? TODO: Reconsider
  string httpVersion_;

  // HTTP request: The headers after the initial line: Eg, Content-Type:text/xml
  OTab headers_;

  // The name of the server
  string serverName_;
  

  virtual void handleErrors_ (const string& text)
  {
    if (logging_) {
      cerr << text << endl;
    }
  }

  // SATISFY PARENT
  virtual void dispatchWork_ ()
  {
    // Errors will propogate as exceptions (usually from 
    // a HTTPthingee()).
    try {
      // Get the preamble, which tells us what we are supposed to do
      string initial_line;
      OTab headers;
      httptools_.getPreamble(initial_line, headers);

      // Parse initial request line: 1st line
      const string& irl = initial_line;
      Array<string> words = Split(irl);
      if (words.length()!=3) {
	httptools_.HTTPBadRequest("Initial request line malformed");
      } 
      const string& op                 = words[0];
      const string& url                = words[1];
      const string& http_version       = words[2];
      httpVersion_ = http_version; // HACK
      
      processHTTP_(op, url, http_version, headers);
      
    } catch (const runtime_error& re) {
      handleErrors_("Runtime Error:"+string(re.what()));
    } catch (const logic_error& le) {
      handleErrors_("Logic Error:"+string(le.what()));
    } catch (const exception& e) {
      handleErrors_("Generic exception:"+string(e.what()));
    } catch (...) {
      handleErrors_("??? Unknown Exception ???");
    }
    
    // At this point, we are all done with the HTTP connection:
    // close 'er up!
    httptools_.close();

    // And add myself back to pool
    server_.addToClientPool(this);

    postDispatchWork_();
  }

  // Hook for users who inherit and need to do some work after 
  virtual void postDispatchWork_ () { }

  // When this is called, the initial line is set (which descibes
  // what's coming) and all the headers are set.
  virtual void processHTTP_ (const string& op, const string& url, 
			     const string& http_version, const OTab& headers)
  {
    if (op=="GET") {
      handleGET_(url, http_version, headers); // user hook
    } else if (op=="POST") {
      handlePOST_(url, http_version, headers); // user hook
    } else {
      httptools_.HTTPNotImplemented(op.c_str());
    }
  }

  // User-hook for dealing with a GET: All initial-line, headers data
  // has been processed, but the rest of the message needs to be processed
  // (i.e,) and the response generated.
  virtual void handleGET_ (const string& url, const string& http_version, 
			   const OTab& headers)
  { httptools_.HTTPNotImplemented(); }

  // User-hook for dealing with a POST: All initial-line, headers data
  // has been processed, but the rest of the message needs to be processed
  // (i.e,) and the response generated.
  virtual void handlePOST_ (const string& url, const string& http_version, 
			    const OTab& headers) 
  { httptools_.HTTPNotImplemented(); }

}; // ThreadedHTTPWorker

// /////////////////////////////////// SimpleHTTPServer

// The actual threaded server that you start/end.

class SimpleHTTPServer : public ThreadedServer {
  
 public:
  
  // Create an HTTP Server on this host and port
  SimpleHTTPServer (const string& host, int port=80) :
    ThreadedServer(host, port)
  { }

 protected:

  // When a new connection comes in, we have to make sure
  // we create the proper type of client for this server.
  virtual ThreadedServerWorker* createThreadedServerWorker_ ()
  { return new ThreadedHTTPWorker("ThreadedHTTPWorker", *this); }


}; // SimpleHTTPServer


PTOOLS_END_NAMESPACE

#endif // SIMPLEHTTPSERVER_H_


