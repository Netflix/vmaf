#ifndef MIDASSERVER_H_

// This module allows creation a server which can connect to multiple
// clients.  It implements a server, emulating what an
// OpalPythonDaemon (from M2k) does.  The current incarnation DOES
// support adaptive serialization.  If using "adaptive" serialization
// (which by default all _C++_ Servers and Talkers do), then the
// MidasServer keeps track of what serializations were used for each
// conversation.  Then when the MidasServer talks to a client, it will
// use the same serialization used by the client.  If there is doubt
// about what serialization to use, the parameters of the constructor
// are the defaults. Note that a class inheriting from MidasServer
// DOES NOT need to worry about the serializations being used: it is
// handled for them.
//
// The MidasServer supports both dual-socket and single socket mode,
// BUT must be chosen up front at construction time: there is no
// adaptive measure (yet) on this.  Dual-socket mode exists for
// compatibility with very old sockets libraries, and is the default
// mode for the M2k OpalPythonDaemon, thus it is the default mode for
// the MidasServer.  If you can, you probably want to use single
// socket mode, as frequent connects/disconnects exposes a race
// condition in dual-socket mode.
//
// The typical user will inherit from MidasServer, overriding the
// methods
//
//    acceptNewClient_    # Callback for when new client appears 
//    readClientData_     # Callback for when data appears for a client
//    disconnectClient_   # Callback for when a client disconnects
//
// The callback methods all pass a file descriptor to the user, which
// can be used for a blocking send back to the client.


#include <algorithm>
#include "midassocket.h"
#include "valprotocol2.h"  // for ArrayDisposition
#include "ocsynchronizer.h"
#include "octhread.h"

#include <sys/select.h> // for select
#include <sys/types.h>  // for select


PTOOLS_BEGIN_NAMESPACE

// The MidasServer class: A base class for users to inherit from to
// implement a server.

class MidasServer : protected MidasSocket_ {

  public :
    
    // Initialize the MidasServer with host and port number.  This
    // doesn't open the server for business yet: see the open() method.
    // See the MidasSocket docs for more documentation on the other
    // parameters.
    MidasServer (string host, int port,
		 Serialization_e serialization,
		 bool dual_socket=1,
		 ArrayDisposition_e disposition=AS_LIST,
		 bool adaptive=true) :
     MidasSocket_(serialization, disposition, true, adaptive),
     host_(host),
     port_(port),
     dualSocket_(dual_socket == true ? DUAL_SOCKET : SINGLE_SOCKET),
     s_(-1),
     mainloopStop_(false),
     mainloopDone_(false),
     readsocks_(),
     writesocks_(),
     serverHandlesDataUpdates_(true), 
     clientFDTable_(Tab()),
     mainloopThread_("mainloop")
    { }

    // Allow NORMAL_SOCKET activity
    MidasServer (string host, int port,
		 Serialization_e serialization,
		 SocketDuplex_e dual_socket=DUAL_SOCKET,
		 ArrayDisposition_e disposition=AS_LIST,
		 bool adaptive=true) :
     MidasSocket_(serialization, disposition, true, adaptive),
     host_(host),
     port_(port),
     dualSocket_(dual_socket),
     s_(-1),
     mainloopStop_(false),
     mainloopDone_(false),
     readsocks_(),
     writesocks_(),
     serverHandlesDataUpdates_(true), 
     clientFDTable_(Tab()),
     mainloopThread_("mainloop")
    { }
    
    // This constructor exists for backwards compatibility with pre 093
    // versions:: prefer the constructor above as it is more explicit
    MidasServer (string host, int port,
                 bool send_strings_as_is_without_serialization=0,
                 bool force_dual_socket=1, bool force_numeric_off=1) :
      MidasSocket_((send_strings_as_is_without_serialization) ? SERIALIZE_NONE : SERIALIZE_P0, 
		   (force_numeric_off) ? AS_LIST : AS_NUMERIC),
     host_(host),
     port_(port),
	dualSocket_(force_dual_socket==true ? DUAL_SOCKET : SINGLE_SOCKET),
     s_(-1),
     mainloopStop_(false),
     mainloopDone_(false),
     readsocks_(),
     writesocks_(),
     serverHandlesDataUpdates_(true), 
     clientFDTable_(Tab()),
     mainloopThread_("mainloop")
    { }

   
    // Clean-up, disconnect everyone, and close file descriptors.
    // This can be called over and over to set-up the server.
    void cleanUp () 
    {      
      // Close all clients
      Tab clientKeys = Tab();
      for (It ii(clientFDTable_); ii(); ) {
	clientKeys[clientKeys.entries()] = ii.key();
      }

      for (int ii = 0; ii < int(clientKeys.entries()); ++ii) {
	removeClient(clientKeys[ii]);
      }

      if (clientFDTable_ != Tab()) {
	errout_("Internal Error: Shouldn't be any clients");
      }

      // Close main door
      if (s_ != -1) { 
	closing_(s_);
	s_ = -1;
      }
      readsocks_.clear();
    }

    // Alias for cleanUp
    void close () { cleanUp(); }

    virtual ~MidasServer ()
    { 
      shutdown();
      waitForMainLoopToFinish();
    }
    int port () const { return port_; }
    string host () const { return host_; }
        
    // Create a server thread that will start accepting clients.
    // Timeout of None will block until some activity is detected on
    // the socket.

    // Note that we've changed the default on the open: This is how
    // long the accept watches the the sockets inside the mainloop
    // before it checks for shutdown events: if it's too long, it's
    // hard for the server to recognize external events: too short,
    // and it polls and wastes CPU.  NONE is almost always the wrong
    // value, as it will never wakeup and look for the "shutdown"
    // message: So we make 1.0 be the default: every 1.0 second, the
    // accept wakes up, checks shutdown, then goeas back to watching
    // the sockets.
    void open (const Val& timeout=1.0) // You almost NEVER want this to be None 
    {
      cleanUp(); // Clean-up file descriptors and reinitialize
      
      // Create the socket that we will be listening on
      int r = socket(AF_INET, SOCK_STREAM, 0);
      if (r<0) errout_("socket");
      s_ = r;
      
      // Make sure we can immediately reconnect if server goes down
      int one = 1;
      r = setsockopt(s_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
      if (r<0) errout_("setsockopt");

      bzero(&mainsock_, sizeof(mainsock_));
      mainsock_.sin_family = AF_INET;
      mainsock_.sin_port = htons(port_);
      mainsock_.sin_addr.s_addr = htonl(INADDR_ANY);
      // r=inet_pton(AF_INET, host_.c_str(), &mainsock_.sin_addr);
      //if (r<=0) errout_("inet_pton");
 
      r = bind(s_, (struct sockaddr*)&mainsock_, sizeof(mainsock_));
      if (r<0) errout_("bind");
      
      r = listen(s_, 5);
      if (r<0) errout_("listen");
      
      // How long to wait until check for shutdown
      timeoutInSeconds_ = timeout;
      
      // A list of client file descriptors: Whenever a client
      // connects or disconnects, it is updated.  Structure:
      // { read_file_desc1 : (address1, write_file_desc1, address_write1),
      //   read_file_desc2 : (address2, write_file_desc2, address_write2),
      //   ... }
      // In single socket mode, since the read_file_desc is the same
      // as the write_file_desc, it is simply a duplicate.
      clientFDTable_ = Tab();
      
      // This is a list of currently active read file descriptors.
      // When a new client attaches, it goes to the end of the list.
      // When a client does away, it is deleted.  Note that the
      // "listen" port is always on the read_socks so we can listen
      // for new clients.
      readsocks_.append(s_);
      
      // Create a thread that watches that socket
      mainloopThread_.start(&MidasServer::mainloop, this);
    }

    // Return (nominally) if the connection is open or not
    bool isStarted () const { return mainloopThread_.started(); }
    
    // A client has appeared and connected: perform maintenance and
    // call user hook.  Users override the acceptNewClient_ routine
    // to get the hook.
    virtual void newClientConnect (int fd) 
    {
      SockAddr_ read_addr, write_addr;
      int read_fd = -1, write_fd = -1;

      try {
	// New Client attempting connect 
	if (dualSocket_ == NORMAL_SOCKET) {
	  read_fd  = socketAccept_(fd, NULL, &read_addr);
	  write_fd = read_fd;  write_addr = read_addr;
	} else if (dualSocket_ == DUAL_SOCKET) { 
	  // This is why there is a potential race condition:  We do
	  // two accepts right after each other.  This is why we suggest
	  // you use single socket mode.
	  read_fd  = socketAccept_(fd, "RECVRECVRECVRECV", &read_addr);
	  write_fd = socketAccept_(fd, "SENDSENDSENDSEND", &write_addr);
	} else if (dualSocket_ == SINGLE_SOCKET) { //  Single socket
	  read_fd  = socketAccept_(fd, "SNGLSNGLSNGLSNGL", &read_addr);
	  write_fd = read_fd;  write_addr = read_addr;
	} else {
	  errout_("Unknown socket protocol?  Needs to be DUAL_SOCKET, SINGLE_SOCKET or NORMAL_SOCKET");
	}
	readWriteAssociate_(read_fd, write_fd);

	// User hook
	acceptNewClient_(read_fd, read_addr, write_fd, write_addr);
      } catch (const exception& re) {  // all sockets error runtime_error
	cerr << "Problems with connection??? Aborting connection:" 
	     << re.what() << endl;
	readWriteDisassociate_(read_fd, write_fd);
	if (read_fd!=-1)  closing_(read_fd);
	if (write_fd!=-1) closing_(write_fd);
	return;
      }


      { 
	ProtectScope ps(mutex_);
	Arr a; // (read_addr, write_fd, write_addr);
	a.append(read_addr);
	a.append(write_fd);
	a.append(write_addr);
	clientFDTable_[read_fd] = a;
	if (serverHandlesDataUpdates_) {
	  readsocks_.append(read_fd); // List of sockets to select
	}
      }
    }

    // A client has gotten data: perform maintenance and call user hook.
    // Users override the readClientData_ routine to get the hook.
    virtual void newClientData (int fd) 
    {
      Val data = recvBlocking_(fd);
      int read_fd = fd; int write_fd = fd;
      if (dualSocket_!=0 && dualSocket_ !=NORMAL_SOCKET) {
	ProtectScope ps(mutex_);
	write_fd = clientFDTable_[read_fd][1]; // (_, write_fd, _);
      }
      readClientData_(read_fd, write_fd, data);
    }

    // A client has gone away.  Remove him from internal lists and
    // cleanly close down its connection.
    virtual void removeClient (int fd) 
    {
      Val entry;
      {
	ProtectScope ps(mutex_);
	// Remove from list containing active "read" file descriptors
	readsocks_.remove(fd);
	if (!clientFDTable_.contains(fd)) { return; } // Pre-emptive
	// Remove from list containing list of file descriptors
        entry = clientFDTable_[fd];  // .. keep copy of for close     
	clientFDTable_.remove(fd);
      }

      int read_fd = fd; int write_fd = entry[1];
      if (!mainloopDone_) {
        disconnectClient_(read_fd, write_fd); // User hook
      }

      // Make sure we clear out the conversation 
      readWriteDisassociate_(read_fd, write_fd);

      // Don't close until all done just in case client needs
      if (read_fd==write_fd) {
	closing_(read_fd);
      } else { 
	closing_(read_fd);
	closing_(write_fd);
      }
    }
       
    // Mainloop for looping through and watching the socket.  
    static void* mainloop (void* data) 
    {
      MidasServer* msp = (MidasServer*)data;
      MidasServer& ms = *msp;

      ProtectScope ma(ms.mainloopMutex_);
        
      ms.mainloopDone_ = false;
      while (1) {
	
	volatile bool done = false;
	Array<int> rs, ws;
	{ 
	  ProtectScope ps(ms.mutex_);
	  // Do a FULL COPY so we only have to lock this transaction
	  rs = ms.readsocks_;
	  ws = ms.writesocks_;
	  done = ms.mainloopStop_;
	}

	if (done) break;
        
	// Convert timeout in n.m seconds to (seconds and
	// microseconds) In order to make sure we 'wake up' once on a
	// while and check, the timout should ALMOST NEVER be None:
	// the default has changed in open to 1.0
	struct timeval timeout;
	struct timeval* timeout_p=
	  ms.convertTime_(&timeout, ms.timeoutInSeconds_);

	// Set-up file descriptors "sets" for select
	fd_set rfds;        FD_ZERO(&rfds); // Set up read descriptors 
	int max_of_fds = -1;
	for (size_t ii=0; ii<rs.length(); ii++) { 
	  int fd = rs[ii];
	  FD_SET(fd, &rfds);
	  max_of_fds = max(max_of_fds, rs[ii]);
	}
	fd_set wfds;        FD_ZERO(&wfds); 	// Set up write descriptors
	for (size_t ii=0; ii<ws.length(); ii++) {
	  int fd = ws[ii];
	  FD_SET(fd, &wfds);
	  max_of_fds = max(max_of_fds, ws[ii]);
	}
	max_of_fds+=1;  // select requires keep max of all fds (+1)

	// ... and call select
	int r = ms.mySelect_(max_of_fds, &rfds, &wfds, NULL, timeout_p);
	if (r<0) ms.errout_("select");
	if (r==0) { continue; } // Nothing available or timeout

	// Look through the readables
	for (int ii=0; ii<int(rs.length()); ii++) {

	  int fd = rs[ii];
	  bool is_in_readables=(FD_ISSET(fd, &rfds)); 
	  
	  if (is_in_readables) { 
	    --r;
	    bool is_main_port = (fd == ms.s_);  // Main port
	    if (is_main_port) { 	 
	      ms.newClientConnect(fd); // New client 
	    } else {                             // Client w/Data to be read
	      try {
		ms.newClientData(fd);
	      } catch (const exception& e) {
		// An error, client went away, take off list
		ms.removeClient(fd);
	      }
	    }
	  }
	  
	}

	if (r!=0) {
	  cerr << "Shouldn't be any writables:" << endl;
	}
	
      } 
      
      // Finish main loop
      ms.mainloopDone_ = true;
      return 0;
    }

    // Wait for the mainloop to finish. 
    void waitForMainLoopToFinish () 
    {
      if (mainloopThread_.started()) {
        bool all_done = false;
        do {
          mainloopMutex_.lock();
          all_done = mainloopDone_;
          mainloopMutex_.unlock();
          if (all_done) break;
          sleep(1);
        } while (1);
      }
      cleanUp();
    }

    // Asynchronously tell the server to shutdown.  The user needs to
    // call waitForMainLoopToFinish to make sure it has finished.
    void shutdown () 
    {
      ProtectScope ps(mutex_);
      mainloopStop_ = true;
    }

    using MidasSocket_::compatibilityMode;

  protected:
  
    string host_;           // host running the service. this host ..
    int port_;              // port number (associated with s_ below)
    SocketDuplex_e dualSocket_;      // Flag to indicate if we are single/dual socket or a normal socket
    int s_;                 // File descriptor for main server socket port
    volatile bool mainloopStop_;     // Flag requesting to leave main dispatch
    volatile bool mainloopDone_;     // Flag indicating when we have left main dispatch
    Mutex mutex_;           // Mutex to protect server from other threads
    Mutex mainloopMutex_;   // Make it easy to "waitfor" the mainloop to quit
    Array<int> readsocks_;  // List of current reading sockets
    Array<int> writesocks_; // List of current writing sockets
    bool serverHandlesDataUpdates_;  // When a new connect happens, does the server handle connection activity or a thread
    Tab clientFDTable_;     // Table of clients and info, see open() 
    Val timeoutInSeconds_;  // TImeout

    OCThread mainloopThread_; // Thread for mainloop
    struct sockaddr_in mainsock_; // 

    // Helper code for when a socket connects: it connects and sends
    // out the appropriate preamble (SNGL, RECV or SEND)*4.  In the
    // case of single-socket mode, this is only called once, as the
    // same socket is used for read/write.  In dual-socket, it is
    // called twice: one accept for what will be the read socket, one
    // accept for what will be the write socket.
    int socketAccept_ (int fd, const char* preamble, SockAddr_* s)
    {
      // May be an interrupted system call for accept
      int r;
      int retries = 0;
      const int retry_max = 1000;
      for (retries=0; retries<retry_max;retries++) {
	r = accept(fd, &s->addr, &s->addrlen);
	if (r<0) {
	  if (errno==EINTR) {
	    retries++;
	    continue;
	  } else errout_("accept");
	}
	break;
      } 
      if (retries>=retry_max) errout_("accept: too many retries");
      int connection_fd = r;

      if (preamble!=NULL) {
	writeExact_(connection_fd, (char*)preamble, strlen(preamble));
      }
      return connection_fd;
    }


    // User Callback for when a new client is added.  Each client can
    // be distiguished by its read_fd (a unique number for each
    // client).  The read_addr (and write_addr) gives more information
    // about the connection: See the man page for accept and struct
    // defintion above In dual-socket mode, the read_fd and write_fd
    // are different numbers, in single-socket mode, they are the
    // same.
    virtual void acceptNewClient_ (int read_fd, const SockAddr_& read_addr, 
				   int write_fd, const SockAddr_& write_addr) 
      = 0;

    // User Callback for when client data is posted.  The read_fd
    // identifies the client that sent the data.  The write_fd is the
    // file descriptor to use to send data back to that client.  In
    // dual-socket mode, the read_fd and write_fd are different
    // numbers, in single-socket mode, they are the same.  Use
    // 'self.sendBlocking_(write_fd, result)' to send data back to the
    // client as a response.
    virtual void readClientData_ (int read_fd, int write_fd, const Val& data) 
      = 0;

    // User Callback for when a client disconnects.  The read_fd
    // distiguishes which client is disconnecting.  The read_fd and
    // write_fd will be closed almost immediately after this method
    // returns (the user does not have to close the descriptors).
    virtual void disconnectClient_ (int read_fd, int write_fd) 
      = 0;
  
}; // MidasServer


PTOOLS_END_NAMESPACE


#define MIDASSERVER_H_
#endif // MIDASSERVER_H_
