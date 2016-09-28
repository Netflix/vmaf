#ifndef MIDASYELLER_H_


// This module exists to provide a simplified interface for
// implementing a UDP Server (MidasYeller) talking to to UDP Clients
// (MidasListener).  It should feel sort of like Midastalker and
// MidasServer interface, but of course uses UDP sockets instead, as
// it may be lossy.

#include "midassocket.h"

PTOOLS_BEGIN_NAMESPACE

// A MidasYeller sends (lossy) UDP packets to a bunch listeners.
class MidasYeller : protected MidasSocket_ {

  public:

    // Initialize the e MidasYeller.  Note that we have to manually
    // add (using addListener below) listeners before you send data."""
    MidasYeller (int udp_message_length_upper_limit /*=1024 */,
		 Serialization_e serialization /*=SERIALIZE_P0*/,
		 ArrayDisposition_e disposition=AS_LIST) :
      MidasSocket_(serialization, disposition),
      hostPortMap_("{ }"),
      upperByteLimit_(udp_message_length_upper_limit) 
    { }

    // This constructor exists for backwards compatibility with pre 093
    // versions:: prefer the constructor above as it is more explicit
    MidasYeller (int udp_message_length_upper_limit=1024,
		 bool send_strings_as_is_without_serialization=false,
		 bool force_numeric_off=true) :
      
      MidasSocket_((send_strings_as_is_without_serialization) ? SERIALIZE_NONE : SERIALIZE_P0, 
		   (force_numeric_off) ? AS_LIST : AS_NUMERIC),
      hostPortMap_("{ }"),
      upperByteLimit_(udp_message_length_upper_limit) 
    { }


    ~MidasYeller () { cleanUp(); }

    // Alias for close
    void cleanUp ()
    {
      for (It ii(hostPortMap_); ii(); ) {
	Arr host_port_tuple = ii.key();
	//string host = host_port_tuple[0];
	//int port    = host_port_tuple[1];

	Arr fd_sockaddr_tuple = ii.value();
	int fd       = fd_sockaddr_tuple[0];
	//SockAddr_ sa = fd_sockaddr_tuple[1];
	if (fd != -1) { ::close(fd); }
      }
      hostPortMap_ = Tab();
    }

    // Clean up and destroy all sockets 
    void close () { cleanUp(); }

    // Add a new client (listener) that should be at the given host/port
    void addListener (const string& host, int port) 
    {
      // Clean-up and make sure not host/port already there first, or
      // if it is, get rid of it.
      Arr host_pair_tuple;
      host_pair_tuple.append(host);
      host_pair_tuple.append(port);      
      removeListener(host_pair_tuple);

      // Open the socket without a bind
      int fd;
      SockAddr_ sa;
      helpOpenSocket_(AF_INET, SOCK_DGRAM, IPPROTO_UDP,
		      HELP_NOTHING, host, port,
		      &fd, (sockaddr_in*)&sa.addr);

      // Copy into our map
      Arr fd_addr_tuple(2);
      fd_addr_tuple.append(fd); 
      fd_addr_tuple.append(sa);

      hostPortMap_[host_pair_tuple] = fd_addr_tuple;
    }

    // Stop yelling at a particular listener
    void removeListener (const Arr& host_pair_tuple) 
    {
      if (hostPortMap_.contains(host_pair_tuple)) {
	int fd = hostPortMap_(host_pair_tuple);
	int rc = ::close(fd);
	if (rc<0) {
	  errout_("close");
	}
      }
    }

    // Send the data to all listeners
    void send (const Val& v) 
    {
      // Iterate through all added Hosts and Ports
      for (It ii(hostPortMap_); ii(); ) {
	// Key:  Host Port
	Arr host_port_tuple = ii.key();
	//string host = host_port_tuple[0];
	//int port    = host_port_tuple[1];

	// Value: socket file descriptor and Unix SockAddr_
	Arr fd_addr_tuple = ii.value();
	int fd       = fd_addr_tuple[0];
	SockAddr_ sa = fd_addr_tuple[1];
	
	if (fd>-1) {
	  // Figure out how to pack data
	  Array<char> serialized_data(1024);
	  // TODO:  make this adaptive?  Probably not, because the M2k UDP
	  // thingees don't allow us to have a header
	  packageData_(v, serialization_, arrayDisposition_, 
		       serialized_data);
	  const int mlen = serialized_data.length();
	  if (mlen > upperByteLimit_) {
	    string mesg = "Message sending is "+Stringize(mlen)+
	      " bytes, but yoru hard-coded limit is "+
	      Stringize(upperByteLimit_);
	    throw runtime_error(mesg);
	  }

	  // In case there is any splitting of the data
	  ssize_t bytes_to_send = serialized_data.length();
	  char *byte_data = serialized_data.data();
	  while (bytes_to_send) {
	    ssize_t r = sendto(fd, byte_data, bytes_to_send,
			       0, &sa.addr, sizeof(sa.addr));
	    if (r<0) {
	      if ((errno==EAGAIN) || (errno==EINTR)) {
		continue; // spurious wakeup?
	      } else {
		errout_("sendto");
	      }
	    } else {
	      bytes_to_send -= r;
	      byte_data += r;
	    }
	  } // end while (bytes_to_send)


	} else {
	  cout << host_port_tuple  
	       << " not currently opened to send data to" << endl;
	}
      }
    }

    using MidasSocket_::compatibilityMode;

  protected:
  
    Tab hostPortMap_;   // (host, port) keys mapping to 
                        // (open socket descriptor, bin. version of host name)
    int upperByteLimit_; // UDP packets are limited, try to enforce some limit


    
}; // MidasYeller

PTOOLS_END_NAMESPACE

#define MIDASYELLER_H_
#endif // MIDASYELLER_H_
