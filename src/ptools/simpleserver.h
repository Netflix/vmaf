#ifndef SIMPLESERVER_H_
#define SIMPLESERVER_H_

#include "midasserver.h"

PTOOLS_BEGIN_NAMESPACE

// A Simple Server: this server doesn't know anything except when
// sockets connect, disconnect and have data ready: It's up to the
// derived class grab the data off from the socket (via the file
// descriptor) and do stuff with the data.  This is really useful for
// HTTP and other "plain" socket work (i.e., no funky Midas 
// dual/single socket).

class SimpleServer : public MidasServer {

  // Use some of the guts of MidasSocket/MidasServer to do the work:
  // There are enough differences that we don't want to inherit
  // publicly, but we want the mainloop of MidasServer to work for us,
  // so it has to be public inheritance (blech).  Should we refactor
  // MidasServer? No: A minor goal of the PicklingTools is to keep the
  // C++ and Python in sync, and refactoring might screw that up.

 public:

  SimpleServer (const string& host, int port) :
    MidasServer(host, port, SERIALIZE_NONE, NORMAL_SOCKET) 
  { 
  }

  virtual ~SimpleServer () { }

 protected:
    
  // RAW HOOK after data is seen coming down the pipe: the normal 
  // MidasServer handles all the deserializaion and the like: WE DON'T
  // WANT ANY OF THAT!   We need to read raw data and headers ourself.
  virtual void newClientData (int fd) 
  {
    // Derived class fills this in ... 
    throw runtime_error("SimpleServer derived class forgot to override newClientData");
  }

  // THIS SHOULD NEVER BE CALLED:  this was original called by
  // new client data above, but not needed to HTTP
  virtual void readClientData_ (int read_fd, int write_fd, const Val& data)
  { throw runtime_error("Mixing SimpleServer and MidasServer wrongly: With SimpleServer, this should never be called!"); }

  
}; // SimpleServer


PTOOLS_END_NAMESPACE

#endif //  SIMPLESERVER_H_
