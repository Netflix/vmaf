#ifndef THREADEDSERVER_H_
#define THREADEDSERVER_H_

#include "simpleserver.h"
#include "ocsynchronizedworker.h"

// This type of server extends the notion of a server so that
// it creates a thread per connection: each thread handles its own I/O 
// after being poked. This allows for more concurrency, and one request 
// doesn't hold up the others.  
// Each thread must override a ThreadedServerWorker

PTOOLS_BEGIN_NAMESPACE

class ThreadedServer;

// //////////////////////////////////////////////// Class ThreadedServerWorker

// The threaded server worker is a thin wrapper around SynchronizedWorker.
// It captures most of the co-ordination that needs to happen between
// the ThreadedServer and the threads.  A client should just need
// to inherit from ThreadedServerWorker to do the right thing.
class ThreadedServerWorker : public SynchronizedWorker {
 public:

  ThreadedServerWorker (const string& name, ThreadedServer& server) :
    SynchronizedWorker(name, false, false),
    readfd_(-1),
    writefd_(-1),
    server_(server)
  { }


  // On connection, the worker is cleaned-up and gotten ready to go!
  virtual void initialize (int readfd, int writefd) 
  {
    if (readfd_!=-1 || writefd_!=-1) {
      throw runtime_error("Uninitialized");
    }
    readfd_ = readfd;
    writefd_ = writefd;
    startUp(); // tell worker to okay to start
  }

  // dispatchWork_ still needs to be overridden by subclasses
  
 protected:

  int readfd_, writefd_;
  ThreadedServer& server_; // back pointer back to server who created me

}; // ThreadedServerWorker


// ///////////////////////////////////////////////// Class ThreadedServer

class ThreadedServer : public SimpleServer {
  
 public:
  
  // Create a threaded server on this host and port
  ThreadedServer (const string& host, int port) :
    SimpleServer(host, port)
  {
    // Once the server has seen a request, he gives all respsonsibility
    // of the connection to the thread who grabs it.  In other words,
    // the server hands all responsibility over to the threads.
    serverHandlesDataUpdates_ = false;
  }

  // Allows client to add itself back to the pool
  virtual void addToClientPool (ThreadedServerWorker* adopt_me)
  {
    ProtectScope ps(clientPoolLock_);
    clientPool_.append(adopt_me);
  }

 protected: 

  // ///// Data Members

  // Lock the client pool and client map
  Mutex clientPoolLock_;

  // A list of "waiting for work" clients which we can just reuse:
  // Before we can use them, we have to "wait for them" to make sure
  // they are ready to go.
  Array<ThreadedServerWorker*> clientPool_;

  // For each client, we create a thread: we pool so that
  // we don't keep creating new ones.  TODO: limit the number
  // of connections/pool members in the future?
  virtual ThreadedServerWorker* createThreadedServerWorker_ () = 0;
  //{ return new ThreadedServerWorker("ThreadedServerWorker", *this); }

  // On Connect, spawn a new thread which will handle all the 
  // socket work from then on for that connection.
  virtual void acceptNewClient_ (int read_fd, const SockAddr_& read_addr, 
				 int write_fd, const SockAddr_& write_addr) 
  {
    ThreadedServerWorker* tsw = 0;
    {
      ProtectScope ps(clientPoolLock_); 
      
      try {
	// See if there is a worker waitingL otherwise create new one
	if (clientPool_.length()) {
	  tsw = clientPool_.removeLast();
	  tsw->waitFor();  // Make sure ready to go
	} else {
	  // Want to start the thread manually: auto starting let
	  // to problems if an exception got thrown.
	  tsw = createThreadedServerWorker_();
	  if (!tsw->started()) {
	    tsw->start(SyncWorkerMainLoop, tsw); // Start the thread
	  }
	}
      } catch (exception& e) {
	string mesg = "Problems starting up a thread:" + string(e.what());
	throw runtime_error(mesg.c_str());
      }

    }

    // Make it so it's ready to be poked with data comes across!
    tsw->initialize(read_fd, write_fd); 
  }


  // On disconnect, separate the thread from the pool
  virtual void disconnectClient_ (int read_fd, int write_fd) 
  {
    throw runtime_error("Illegal Mixing of MidasServer and ThreadedMidasSever");
  }


}; // ThreadedServer


PTOOLS_END_NAMESPACE

#endif // THREADEDSERVER_H_


