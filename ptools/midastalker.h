#ifndef MIDASTALKER_H_

// A MidasTalker is a client which talks to a server using sockets.
// This module exists to allow a simplified interface to talk to a
// Midas system, whether talking to an M2k OpalPythonDaemon or the the
// MidasServer in either C++ or Python.

// Most problems when using the MidasTalker are due to socket problems
// (wrong host port or name, server unexpectedly goes away).  These can
// be caught with try/expect blocks.  See the example below.


// string host = "somehost"; int port=9999;
// MidasTalker mt(host, port);
//
// try {
//      mt.open();
//      cout << "Opened connection to host:"
//           << mt.host() << " port:" << mt.port() << endl;
//      break;
// } catch (const runtime_error& e) {
//      cout << "Problem: " << e.what() << endl;
//      cout << "...couldn't open right away" << endl;
// }
// try {
//      Val res = mt.recv(5.0);
//      if (res == None) {
//         cout << "...didn't get a receive after 5 seconds ..." << endl;
//         // Maybe try to do some other work
//    } else {
//      // Do something with the result
//      cout << "Got result" << res << endl;
//    }
// } catch (const runtime_error& e) {
//      cout << "Problem: " << e.what() << endl;
//      cout << "Server appears to have gone away? Attempting to reconnect"
//           << endl;
// }


#include "midassocket.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include <sys/select.h> // for select
#include <sys/types.h>  // for select

#define MIDAS_COMPILER_TEMPLATES
#include "valpython.h"

#include "socketerror.h"   // more specific errors for socket problems

PTOOLS_BEGIN_NAMESPACE

// This class exists as a helper class to talk to the OpalPythonDaemon
// in Midas2k.  See the "midastalker_ex.cc" program for a simple
// example usage and "midastalker_ex2.cc" for a more complicated
// example.

class MidasTalker : protected MidasSocket_ {
 public:


  // The MidasTalker class: a C++ class for talking to a
  // MidasServer or an M2k OpalPythonDaemon.

  // Host is the name of the machine running the server that we wish
  // to connect to, port is the port number on that machine

  // Serialization should be one of:
  //   0: SERIALIZE_P0 (use Python Pickling Protocol0: slow but compatible)
  //   1: SERIALIZE_NONE (send data in strings as-is: good for RAW data),
  //   2: SERIALIZE_P2 (use Python Pickling Protocol2: fast but less compat)
  //  -2: SERIALIZE_P2_OLD  (for old version only, see below for info)
  //   4: SERIALIZE_M2K (for compatibility with M2K)
  //   5: SERIALIZE_OC  (for OpenContainers serialization: fastest, 
  //                          but only C++ and Python C Extension module)
  // We default to SERIALIZE_P0 for backwards compatibility, but
  // strongly urge users to use SERIALIZE_P2 for the speed.

  // dual_socket refers to how you communicate with the server: using one
  // bi-directional socket or two single directional sockets.
  //   0: SINGLE_SOCKET
  //   1: DUAL_SOCKET
  // 777: NORMAL_SOCKET
  // The dual socket mode is a hack around an old TCP/IP stack problem
  // that has been irrelevant for years, but both X-Midas and M2k have to
  // support it.  We recommend using single_socket mode when you can, but
  // default to dual socket mode because that's what X-Midas and Midas 2k
  // default to. Note: A server either supports single or dual socket
  // mode: NOT BOTH.
  //
  // NORMAL_SOCKET can be used to talk to normal sockets that don't have
  // the ridiculous DUAL_SOCKET or SINGLE_SOCKET protocol.

  // array_disposition ONLY applies if you are using SERIALIZE_P0 or
  // SERIALIZE_P2: array_disposition describes how you expect to send/recv
  // binary arrays: as Python arrays (import array) or Numeric arrays
  // (import Numeric).  By default, the MidasTalker sends the data as
  // lists, but this isn't efficient as you lose the binary/contiguous
  // nature.  This option lets the constructor check to see if your
  // Python implementation can support your choice.  Choices are:
  //  0: AS_NUMERIC         (DEPRECATED: recommended if using XMPY< 4.0.0)
  //  1: AS_LIST            (default)
  //  2: AS_PYTHON_ARRAY    (recommended if using Python 2.5 or greater)
  //  3: AS_NUMERIC_WRAPPER (NOT CURRENTLY SUPPORTED)
  //  4: AS_NUMPY           (recommended!!)

  // When using SERIALIZE_P2_OLD: Version 2.2.x of Python serializes P2
  // slightly differently than later versions, and we support this (to a
  // certain extent).  Stay away from this if you possibly can.
  // See the MidasSocket_ documentation for more information.
  
  // Note that both the Client (MidasTalker and MidasServer) need
  // to match on all 3 of these parameters or they WILL NOT TALK!

  // If using "adaptive" serialization (which by default all C++
  // Servers and Talkers do), then we keep track of what
  // serializations happen for each conversation and try to talk to
  // servers in the same serialization they talked to us.  When there
  // is a doubt as to what to talk, then the parameters below decide
  // that.
  MidasTalker (const string& host, int port, 
	       Serialization_e serialization, // can't default for back compat 
	       bool dual_socket=1,
	       ArrayDisposition_e dis=AS_LIST,
	       bool adaptive=true) :
    MidasSocket_(serialization, dis, adaptive),
    host_(host),
    port_(port),
    dualSocket_(dual_socket==true ? DUAL_SOCKET : SINGLE_SOCKET)
  { s_[0] = s_[1] = -1; }

  // All us to be able to set NORMAL socket coms
  MidasTalker (const string& host, int port, 
	       Serialization_e serialization, // can't default for back compat 
	       SocketDuplex_e dual_socket=DUAL_SOCKET,
	       ArrayDisposition_e dis=AS_LIST,
	       bool adaptive=true) :
    MidasSocket_(serialization, dis, adaptive),
    host_(host),
    port_(port),
    dualSocket_(dual_socket)
  { s_[0] = s_[1] = -1; }

  // This constructor exists for backwards compatibility with pre 093
  // versions:: prefer the constructor above as it is more explicit
  MidasTalker (const string& host, int port, 
	       bool send_strings_as_is_without_serialization=false,
	       bool dual_socket=DUAL_SOCKET,
	       bool force_numeric_off=true) :

    MidasSocket_((send_strings_as_is_without_serialization) ? SERIALIZE_NONE : SERIALIZE_P0, 
		 (force_numeric_off) ? AS_LIST : AS_NUMERIC),
    host_(host),
    port_(port),
    dualSocket_(dual_socket == true ? DUAL_SOCKET : SINGLE_SOCKET)
  { s_[0] = s_[1] = -1; }

  // Clean up and close file descriptors
  void cleanUp ()
  { 
    // When we clean up, make sure we clean up the conversation
    // parameters
    readWriteDisassociate_(s_[0], s_[1]);

    // It's important that we immediately assign into the s[0] and
    // s[1] so we can track that resource in case some exceptional
    // activity takes place.
    int fd0 = s_[0]; 
    s_[0] = -1;
    if (fd0!=-1) {
      closing_(fd0);
    }

    int fd1 = s_[1];
    s_[1] = -1;
    if (fd0!=fd1 && fd1!=-1) {
      closing_(fd1);
    }
  }

  // Alias for cleanUp
  void close () { cleanUp(); }

  // Clean-up
  virtual ~MidasTalker () { cleanUp(); }

  int port () const { return port_; }
  string host () const { return host_; }

  // With the host and the port set, open up the socket and connect.
  // During the MidasSocket/talker open negotations, some bytes are
  // sent back and forth:  if things aren't there right away, we allow
  // a timeout so you can recover from a failed open.
  void open (Val timeout=None) 
  {
    cleanUp();
    helpOpenSocket_(AF_INET, SOCK_STREAM, 0,
		    HELP_CONNECT, host_, port_,
		    &s_[0], &servaddr_[1], timeout);

    // Single-socket or dual-socket mode or normal-socket?
    if (dualSocket_ == NORMAL_SOCKET) {
      s_[1] = s_[0];
      // Nothing, just don't do crazy midas socket stuff
    } else if (dualSocket_ == SINGLE_SOCKET) { // single socket, read and write same descriptor
      s_[1] = s_[0];
      string m1 = nextBytes_(s_[0], timeout, 16);
      clientExpectingSingleSocket_(m1);

    } else if (dualSocket_ == DUAL_SOCKET) {  // Must be dual-socket mode
      helpOpenSocket_(AF_INET, SOCK_STREAM, 0,
		      HELP_CONNECT, host_, port_,
		      &s_[1], &servaddr_[1], timeout);

      // Distinguish reader and writer
      string m1 = nextBytes_(s_[0], timeout, 16);
      clientExpectingDualSocket_(m1, "");
      string m2 = nextBytes_(s_[1], timeout, 16);
      if (m1=="SENDSENDSENDSEND" && m2=="RECVRECVRECVRECV") {
	// No need to swap file desc
      } else if (m1=="RECVRECVRECVRECV" && m2=="SENDSENDSENDSEND") {
	// Swap file desc
	int t = s_[0]; s_[0] = s_[1]; s_[1] = t;
      } else {
	clientExpectingDualSocket_(m1, m2);
      }  
    } else {
      errout_("Unknown socket protocol? Check your dual_socket param");
    }
    readWriteAssociate_(s_[0], s_[1]);
  }

  // Return (nominally) if the connection is open or not
  bool isOpen () const { return s_[0]!=-1; }

  // Try to send a value over the socket: If the message can't be sent
  // immediately, wait for timeout (m.n) seconds for the the socket to
  // be available.  If the timeout is None (the default), then this is
  // a blocking call.  If the timeout expires before we can send, then
  // None is returned from the call and nothing is queued, otherwise
  // true is returned.
  Val send (const Val& v, Val timeout_in_seconds = None) 
  { return sendTimed_(s_[1], v, timeout_in_seconds); } 

  // Blocking call to send Val over socket.
  void sendBlocking (const Val& val) { sendBlocking_(s_[1], val); }
     
  // Try to receive the result from the socket: If message is not
  // available, wait for timeout (m.n) seconds for something to be
  // available.  If no timeout is specified (the default), this is a
  // blocking call.  If timeout expires before the data is available,
  // then None is returned from the call, otherwise the val is
  // returned from the call.  A socket error can be thrown if the
  // socket goes away.
  Val recv (Val timeout_in_seconds = None) 
  { return recvTimed_(s_[0], timeout_in_seconds); }

  // Blocking call to recv
  Val recvBlocking () { return recvBlocking_(s_[0]); }

  // This function returns True immediately if the MidasTalker can
  // read immediately without waiting.  If the socket is NOT
  // available, then the socket is watched for timeout seconds: if
  // something becomes available on the socket in that time, this
  // returns True, otherwise the entire timeout passes and this
  // returns False.  Note that this method allows a user to 'poll' a
  // MidasTalker without having to do an actual read.  A timeout of
  // None is forever, a timeout of 0 is no wait. 

  // Slightly unintuitive: If the other end of the socket goes away,
  // THIS STILL RETURNS TRUE!  Only when you do the read can you
  // detect that the socket has gone away (see UNIX socket FAQs to
  // verify this).
  bool dataReady (Val timeout_in_seconds=0) 
  { return dataReady_(s_[0], timeout_in_seconds); }


  using MidasSocket_::compatibilityMode;
   
 protected:

  // ///// Data Members

  string host_;
  int port_;
  int s_[2];        // read and write descripter, same if single-socket mode
                    // s_[0] = read, s_[1] = write, like stdin and stdout
  struct sockaddr_in servaddr_[2];  // socket impl. detail
  SocketDuplex_e dualSocket_; // By default, we support dual-socket mode 


  // Helper routine for opening exactly one socket: Unclear if the
  // servaddr struct has to live beyond the connect, which is why we
  // have potentially two copies (for dual-socket: one for read, one
  // for write).

  // Helper routine: Read exactly the next bytes from the fd and plop
  // them into a string.
  string nextBytes_ (int fd, Val timeout_in_seconds, size_t bytes)
  {
    // Convert timeout in m.n seconds to (seconds and microseconds)
    struct timeval timeo;
    struct timeval* timeout_p = convertTime_(&timeo, timeout_in_seconds);

    //  Set up the file descriptors:  Really only setting the read
    fd_set rfds;         FD_ZERO(&rfds); 
    int readfd = fd;  FD_SET(readfd, &rfds); 
    int max_of_fds = readfd + 1; // select requires keep max of all fds (+1)
    
    // ... and call select
    int r = mySelect_(max_of_fds, &rfds, NULL, NULL, timeout_p);
    if (r<0) errout_("select");
    
    if (r==0) { // Nothing available or timeout
      throw timeout("Timeout in first stages of socket negotiation");
    };

    // Some data, grab it
    Array<char> buff(bytes); 
    buff.expandTo(bytes);
    readExact_(fd, buff.data(), bytes);
    string retval =  string(buff.data(), bytes);
    return retval;
  }

  // Check and see that we are expecting DUAL_SOCKET correctly:
  // throw an exception with a good error message if they don't match,
  // otherwise return ok
  void clientExpectingDualSocket_ (const string& m1, const string& m2)
  {
    string mesg;
    if (m1!="SENDSENDSENDSEND" && m1!="RECVRECVRECVRECV") {
      if (m1=="SNGLSNGLSNGLSNGL") {
	mesg = 
	  "Your client is configured as DUAL_SOCKET, but your server\n"
	  "is configured as SINGLE_SOCKET.  You need to set them both to\n"
	  "match, then restart both sides (if you change just the client,\n"
	  "you may only need to restart the client).\n";
      } else {
	mesg = 
	  "Something is wrong: the client is set-up as DUAL_SOCKET\n"
	  "but we are getting data which suggests your host is set-up as\n"
	  "NORMAL_SOCKET, or maybe even a port that doesn't use the\n"
	  "Midastalker/Server/OpalPythonDaemon protocols!\n"
	  "Recheck your client and server socket type: \n"
	  "   (DUAL_SOCKET, SINGLE_SOCKET, NORMAL_SOCKET)\n"
	  "and make sure both client and server match (also check port #).\n";
      }
      errout_(mesg);
    }
    // The preamble is one of the two expected, what if both are same?
    if (m1==m2) {
      mesg = 
	"Congratulations!  You have found the DUAL_SOCKET race condition!\n"
	"It's an innate problem that occurs fairly rarely.  Your sockets\n"
	"are messed up and you need to reset both sides.  If this error\n"
	"message occurs constantly, you may need to go to SINGLE_SOCKET\n"
	"or NORMAL_SOCKET on both sides (if you interface with old systems,\n"
	"SINGLE_SOCKET is your only choice)\n";
      errout_(mesg);
    }
    // If make it here, connection is okay
  }

  // Check and see that we are expecting SINGLE_SOCKET correctly:
  // throw an exception with a good error message if they don't match,
  // otherwise return ok
  void clientExpectingSingleSocket_ (const string& m1)
  {
    string mesg;
    if (m1=="SENDSENDSENDSEND" || m1=="RECVRECVRECVRECV") {
      mesg = 
	"Your server seems to be in DUAL_SOCKET mode, and your client\n"
	"is in SINGLE_SOCKET mode.  You need to reconfigure both client\n"
	"and server to match and RESTART both sides (you may need to only\n"
	"restart your client side if you set this client to DUAL_SOCKET)\n";
	errout_(mesg);
    } else if (m1!="SNGLSNGLSNGLSNGL") {
      mesg = 
	"Something is wrong: the client is set-up as SINGLE_SOCKET\n"
	"but we are getting data which suggests your host is set-up as\n"
	"NORMAL_SOCKET, or maybe even a port that doesn't use the\n"
	"Midastalker/Server/OpalPythonDaemon protocols!\n"
	"Recheck your client and server socket type: \n"
	"   (DUAL_SOCKET, SINGLE_SOCKET, NORMAL_SOCKET)\n"
	"and make sure both client and server match (also check port #).\n";
      errout_(mesg);
    } else {
      // Okay
    }
  }

}; // MidasTalker



// Given a list of MidasTalkers, this returns a list of all 'ready'
// MidasTalkers.  Take the given timeout, distribute it evenly over
// all MidasTalkers so that the maximum wait (if all are unavailable)
// is the given timeout.  If none are ready, a full timeout seconds
// elapse, and an empty list is returned.  With an immediate return,
// this function returns as soon as ANY socket is ready, without it,
// each talkers may wait the full distributed quanta of time.
inline Array<MidasTalker*> 
DistributedWaitForReadyMidasTalker (Array<MidasTalker*> midastalker_list, 
				    bool immediate_return,
				    Val timeout=None) 
{
  // Choose between distributing the timeout, and having no timeout.
  Val distributed_timeout;
  if (timeout==None) {
    distributed_timeout = None;
  } else {
    distributed_timeout = float(timeout) / float(midastalker_list.length());
  }

  // Gather all MidasTalkers that are available in a list
  Array<MidasTalker*> result;
  for (size_t ii=0; ii<midastalker_list.length(); ii++) {
    MidasTalker* mt = midastalker_list[ii];
    if (mt->dataReady(distributed_timeout)) {
      result.append(mt);
      if (immediate_return)   // Get out right away if there's one available
	break;
    }
  }
     
  return result;
}

// Query a list of MidasTalkers, returning a list of MidasTalkers that
// are 'ready' (where 'ready' means the socket can be read from
// immediately).  If no MidasTalkers are immediately ready, then the
// timeout is distributed over all MidasTalkers until at least one
// MidasTalker is ready.  If immediate_return is True, then this
// returns as soon as ANY MidaTalker is ready, otherwise we must look
// at all MidasTalkers.  
inline Array<MidasTalker*> 
WaitForReadyMidasTalker (Array<MidasTalker*> midastalker_list, 
			 bool immediate_return=false,
			 Val timeout=None)
{
  // First pass, gather up all that are immediately ready, no waiting.
  Array<MidasTalker*> res = 
    DistributedWaitForReadyMidasTalker(midastalker_list, false, 0);
  if (res.length()==0) { 
    // Second pass, no one was ready.  Distribute the wait (if there is one)
    // and wait for someone to be available.
    res = DistributedWaitForReadyMidasTalker(midastalker_list,
                                             immediate_return,
                                             timeout);
  }
  return res;
}
    

PTOOLS_END_NAMESPACE


#define MIDASTALKER_H_
#endif // MIDASTALKER_H_
