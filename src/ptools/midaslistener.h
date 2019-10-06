#ifndef MIDASLISTENER_H_

// A client for a MidasYeller

#include "midassocket.h"
#if defined(OSF1_)
// This may or may not be necessary
typedef int socklen_t;
#endif

PTOOLS_BEGIN_NAMESPACE

//The MidasListener listens to a single MidasYeller.  If we are not
//actively trying to receive data, it may be lost as the MidasYeller
//sends UDP packets.

class MidasListener : protected MidasSocket_ {
  
 public: 
  
    // Initalize the UDP listener with the given host and port.  Since
    // we are listening to UDP packets, we have an upper limit on how
    // big the packet may be.
    MidasListener (const string& host, int port, 
		   int udp_message_length/*=1024*/,
		   Serialization_e serialization/*=SERIALIZE_P0*/,
		   ArrayDisposition_e disposition=AS_LIST) :
      MidasSocket_(serialization, disposition),
      host_(host),
      port_(port),
      udpMessageLength_(udp_message_length),
      fd_(-1),
      senderLength_(sizeof(sender_))
    { }

    // This constructor exists for backwards compatibility with pre 093
    // versions:: prefer the constructor above as it is more explicit
    MidasListener (const string& host, int port, int udp_message_length=1024,
		   bool send_strings_as_is_without_serialization=false,
		   bool force_numeric_off=true) :
      
      MidasSocket_((send_strings_as_is_without_serialization) ? SERIALIZE_NONE : SERIALIZE_P0, 
		   (force_numeric_off) ? AS_LIST : AS_NUMERIC),
      
      host_(host),
      port_(port),
      udpMessageLength_(udp_message_length),
      fd_(-1),
      senderLength_(sizeof(sender_))
    { }



    ~MidasListener () { cleanUp(); }
    
    // Clean up and close file descriptors
    void cleanUp () 
    {
      int fd = fd_;
      fd_ = -1;
      if (fd!=-1) {
	::close(fd);
      }
    }

    // Alias for cleanUp
    void close () { cleanUp(); }

    // Create the socket and bind it to the server,port we want to
    // listen to
    void open ()
    {
      cleanUp();
      helpOpenSocket_(AF_INET, SOCK_DGRAM, IPPROTO_UDP,
		      HELP_BIND, host_, port_,
		      &fd_, (struct sockaddr_in*)&addr_);
    }	

    // Try to receive a UDP packet from the MidasYeller.  If a message
    // is not available, wait for a timeout of (m.n) seconds for
    // something to become available.  If no timeout is specified (the
    // default), this is a blocking call.  If timeout expires before
    // the data is available, then None is returned from the call,
    // otherwise the val is returned from the call.  A socket error
    // can be thrown if the socket goes away."""
    Val recv (Val timeout_in_seconds = None)
    {
      Val retval;

      // Convert timeout in m.n seconds to (seconds and microseconds)
      struct timeval timeout;
      struct timeval* timeout_p = convertTime_(&timeout, timeout_in_seconds);
      
      //  Set up the file descriptors:  Really only setting the read
      fd_set rfds;         FD_ZERO(&rfds);
      int readfd = fd_;    FD_SET(readfd, &rfds);
      int max_of_fds = readfd + 1; // select requires keep max of all fds (+1)
      
      // ... and call select
      int r = mySelect_(max_of_fds, &rfds, NULL, NULL, timeout_p);
      if (r<0) errout_("select");

      if (r==0) { // Nothing available or timeout
	// leave return as None
      } else {    // r is the number of descriptors, and ready to go!
	
	// create buffer
	Array<char> packaged_data(udpMessageLength_);
	packaged_data.expandTo(udpMessageLength_);
	
	// receive the raw data and make sure no errors
	int r = recvfrom(fd_, packaged_data.data(), packaged_data.length(),
			 0, (struct sockaddr*)&sender_, &senderLength_);
	if (r<0) { errout_("recvfrom"); }
	if (r>int(packaged_data.length())) {
	  errout_("too much received!  Overflow buffer");
	}
	// Unpackage so we can return it
	// TODO:  make this adaptive?  Probably not, because the M2k UDP
	// thingees don't allow us to have a header
	unpackageData_(packaged_data, serialization_, arrayDisposition_, 
		       retval);
      }
      return retval;
    }
 
    using MidasSocket_::compatibilityMode;

 protected:

    string host_;             // The host we want to be listening to
    int port_;                // The port on host above we want to be lis. to
    int udpMessageLength_;    // Since UDP packet sizes are limited, 
                                  // enforce some limit
    int fd_;                  // Socket file descriptor
    struct sockaddr_in addr_; // address of 

    struct sockaddr_in sender_;
    socklen_t          senderLength_;  // return addr of sender

    
    
}; // MidasListener


PTOOLS_END_NAMESPACE

#define MIDASLISTENER_H_
#endif // MIDASLISTENER_H_
