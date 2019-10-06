#ifndef MIDASSOCKET_H_

// The internals of the MidasServer and MidasTalker share a lot of
// code (especially to do primitive reads and primitive writes to the
// socket).  This module encapsulates all that code.

#include "fdtools.h"
#include "ocsynchronizer.h"
#include "chooseser.h"


PTOOLS_BEGIN_NAMESPACE 

// Forward
class StreamDataEncoding;

// Do you use two sockets or 1 socket for full duplex communication?
// Some very old versions (VMWARE) only supported single duplex sockets,
// and so full duplex sockets had to be emulated with 2 sockets.
// The single/dual socket protocols imply extra data sending:
// we also support "normal sockets" (which just connects a single
// full duplex socket, like HTTP uses). (Normal socket implies you
// DO NOT send out a preamble indentifying it:  Single socket
// implies you send out a preamble of "SNGL"*4, Double socket
// implies you send out a preamble of "DUAL"*4)
enum SocketDuplex_e {
  SINGLE_SOCKET = 0,
  DUAL_SOCKET   = 1,
  NORMAL_SOCKET = 777    
};


// A base class with some primitive I/O facilities for MidasServer and
// MidasTalker
class MidasSocket_ : public FDTools_ {
 public:

  // Create MidasSocket_ that either Python Pickles or sends strings
  // as is. This also chooses whether or not to use Numeric or Arrays
  // in Array serialization.  For backwards compat, AS_LIST is 1
  // (being true), AS_NUMERIC is 0 (being false) and the new
  // AS_PYTHON_ARRAY is 2.

  // Adaptive means that the MidasSocket keeps track of the type of
  // serialization used in the messages sent/received over the
  // socket.  If possible, we will talk to a server or client in
  // the serialization they talked to us.  If there is ever a doubt
  // as to what serialization to use, the serialization parameters
  // passed (in the constructor) here are used.  If adapative is
  // sent to false, then ALWAYS use the serialization parameters
  // given here: This is probably most important if you are using
  // SERIALIZE_NONE

  // In UNIX systems, issues with the socket cause a UNIX signal to
  // occur: we almost always want that signal ignored because we can
  // basically detect the socket going away.
  MidasSocket_ (Serialization_e serialization=SERIALIZE_P0,
		ArrayDisposition_e disposition=AS_LIST,
		bool ignore_sigpipe=true,
		bool adaptive=true) :
    FDTools_(),               // No associated fd
    adaptive_(adaptive),
    serialization_(serialization),
    arrayDisposition_(disposition),
    compatibilityMode_(false),
    forceShutdownOnClose_(true)
  {
    if (ignore_sigpipe) installSIGPIPE_ignore();
  }

  // To preserve compatibility with pre-existing servers and talkers,
  // OTab->Tab, Tup->Arr, BigInt->Str.  You really only need this if
  // you plan to send those (OTab, Tup, BigInt) to multiple
  // servers/client where there's a mix of legacy and newer.
  // Otherwise, just don't send those data structure over.  TRUE means
  // convert OTab/Tup/BigInt, FALSE (default) is just try to pass
  // things on as-is (no conversions).
  void compatibilityMode (bool mode) { compatibilityMode_ = mode; }

  // EXPERTS:
  // The most compatible way to ensure a socket dies is to force
  // a shutdown on the socket, but this assumes close won't work
  // for some reason (a bug?  some processes are still holding it open?).
  // For those who want to hope that close does the right thing, 
  // we DON'T force a shutdown when we close by setting this to False.
  // If that isn't good enough, set this to True: 
  // This forces the shutdown.
  bool forceShutdownOnClose () const { return forceShutdownOnClose_; }
  void forceShutdownOnClose (bool v) { forceShutdownOnClose_ = v; }


  virtual ~MidasSocket_ () { }

 protected:

  // ///// Data Members

  bool adaptive_;  // If true: Prefer to just look at headers to see
                   // what kind of serialization to use.  When
                   // questions arise as to what serialization to use,
                   // use defaults below (supplied with constructor)

  Serialization_e serialization_;       // Serialization:M2k, Python, OC, None?
  ArrayDisposition_e arrayDisposition_; // How do arrays of POD serialize?

 
  bool compatibilityMode_; // TRUE if we convert OTab/Tup/BigInt->Tab/Arr/Str
                           // FALSE (default) if we pass as-is. 
  
  bool forceShutdownOnClose_;    // Do we do close only or shutdown/close ?

  string header_;          // Header is PY00 for Non-Numeric, PYN0 for Numeric,
                           //           PYA0 for python Arrays


  Mutex conversationsLock_;  // Lock activity on conversations table 
  Tab conversations_;        // key: file descriptors value: list of 
                             //     (serialization, array_disposition)
  Tab readWriteAssociations_; // Be able to get from read fd to write fd
                              // Use lock for converstaions

  // When we want to close a socket, some platforms don't seem
  // to respond well to "just" a close: it seems like a 
  // shutdown is the only portable way to make sure the socket dies.
  // Some people may spawn processes and prefer to use close only,
  // so we allow 'plain' close is people specify it, otherwise
  // we force a shutdown.  The default is TRUE.
  void closing_ (int fd)
  {
      if (forceShutdownOnClose_)
      {
#if defined(_MSC_VER) || defined(__MINGW32__)
          int flag = SD_BOTH;
#else
          int flag = SHUT_RDWR;
#endif
          shutdown(fd, flag);
      }
    ::close(fd);
  }


  // make it so you can get from read fd to write fd and vice-versa
  void readWriteAssociate_(int read_fd, int write_fd)
  {
    ProtectScope ps(conversationsLock_);
    if (read_fd!=-1)
      readWriteAssociations_[read_fd] = write_fd;
    if (write_fd!=-1)
      readWriteAssociations_[write_fd] = read_fd;
  }

  // When the socket closes, take file desc out of the table and also
  // clear out the conversation record so we start fresh
  void readWriteDisassociate_ (int read_fd, int write_fd)
  {
    ProtectScope ps(conversationsLock_);
    if (readWriteAssociations_.contains(read_fd)) {
      readWriteAssociations_.remove(read_fd);
    }
    if (readWriteAssociations_.contains(write_fd)) {
      readWriteAssociations_.remove(write_fd);
    }
    if (conversations_.contains(read_fd)) {
      conversations_.remove(read_fd);
    }
  }

  // Choose what header to create with given serialization choices
  static string createHeader_ (Serialization_e serialization, 
			       ArrayDisposition_e array_disposition)
  {
    string header;
    switch (array_disposition) {
    case AS_NUMERIC:       header = "PYN0";  break;
    case AS_LIST:          header = "PY00";  break;
    case AS_PYTHON_ARRAY:  header = "PYA0";  break;
    case AS_NUMPY:         header = "PYU0";  break;
    default: throw runtime_error("Unknown array disposition");
    }
    switch (serialization) {
    case SERIALIZE_P0:     header[3] = '0'; break;
    case SERIALIZE_P2:     header[3] = '2'; break;
    case SERIALIZE_P2_OLD: header[3] = '-'; break;
    case SERIALIZE_NONE:   header    = ""; break;
    case SERIALIZE_M2K:    header    = "M2BD"; break; 
    case SERIALIZE_OC:     header    = "OC00"; break;

    default: throw runtime_error("Unknown serialization");
    }
    return header;
  }
 

  // Choose the receive serialization.  If the user has choosen NOT to
  // use adaptive serialization, this forces the values from the
  // constructor to be used as the serialization choice.  Otherwise,
  // the serialization choice is based on the currently passed in
  // header.  The file descriptor is so we can RECORD the
  // serialization choice for the return conversation.
  Serialization_e chooseRecvSerialization_ (int read_fd, const string& hdr,
					    ArrayDisposition_e& array_dis)
  {
    // Force what we want to recv, even if the header lies
    if (!adaptive_) {
      array_dis = arrayDisposition_;
      return serialization_;
    }
    
    // Otherwise, use the header to guide us
    Serialization_e serialization = SERIALIZE_NONE;
    array_dis = AS_LIST; // Really doesn't matter, just not unititalized!

    switch (hdr[0]) {
    case 'O': // hopefully OC
      if (hdr[1]=='C') {
	serialization = SERIALIZE_OC;
      }
      break;

    case 'M': // hopefully M2K
      if (hdr[1]=='2' && hdr[2]=='B' && hdr[3]=='D') {
	serialization = SERIALIZE_M2K;
      }
      break;

    case 'P': // hopefully Python of some flavor
      if (hdr[1]=='Y') {
	// Choose serialization
	switch (hdr[3]) {
	case '0': serialization = SERIALIZE_P0;     break;
	case '2': serialization = SERIALIZE_P2;     break;
	case '-': serialization = SERIALIZE_P2_OLD; break;
	}
	// Choose Array Disposition
	switch (hdr[2]) {
	case 'N': array_dis = AS_NUMERIC;       break;
	case 'A': array_dis = AS_PYTHON_ARRAY;  break;
	case 'U': array_dis = AS_NUMPY;         break;
	case '0': 
	default:  array_dis = AS_LIST;          break;
	}
      }
      break;

    default:  // hopefully raw
      break;     // NONE When you don't know what to do, rather than throw 
    }
   
    // We've recieved a header, log this information so that
    // a send will serialize the way it was received
    Arr options("[0,0]");
    options[0] = int(serialization);
    options[1] = int(array_dis);
    { 
      ProtectScope ps(conversationsLock_);
      conversations_[read_fd] = options;
    }

    return serialization;
  }

  // Choose the kind of serialization to use for a send.  To send, we
  // either have to default to what was in the constructor if we've
  // never seen any information, or use the conversation information
  // we have.
  virtual Serialization_e chooseSendSerialization_ (int write_fd,
						    ArrayDisposition_e& array_dis)
  {
    Serialization_e serialization = serialization_;

    // IF adaptive: Check the table to see if there is already a
    // conversation with the server.
    bool contains = false;
    int read_fd;
    Arr how_server_talked_to_us;
    if (adaptive_) {
      ProtectScope ps(conversationsLock_);
      //if (!readWriteAssociations_.contains(write_fd)) return;// Early return
      read_fd = readWriteAssociations_(write_fd); // ... exception if not there
      contains = conversations_.contains(read_fd);
      if (contains) {
	how_server_talked_to_us = conversations_(read_fd);
      }
    }

    // Adaptive...do we have a header from this so we know what to send? 
    if (adaptive_ && contains) {
      serialization = Serialization_e(int(how_server_talked_to_us[0]));
      array_dis     = ArrayDisposition_e(int(how_server_talked_to_us[1]));
    }

    // Not adaptive, just force the serialization
    else {     
      serialization = serialization_;
      array_dis     = arrayDisposition_;
    }

    return serialization;
  }

  void readExact_ (int fd, char* data, size_t len) 
  { FDTools_::ReadExact(fd,data,len);}


  // Blocking call to get next Val off of socket.
  Val recvBlocking_ (int fd) 
  {
    Val retval;

    // Preamble: number of bytes to read (doesn't include 4 bytes hdr, next).
    // If we see the ESCAPE SEQUNCE, then we know that we have a packet if
    // int_u8 bytes, so another 8 bytes are coming.
    size_t bytes_to_read = 0;
    {
      int_u4 bytes_to_read4 = 0;
      readExact_(fd, (char*)&bytes_to_read4, sizeof(bytes_to_read4));
      bytes_to_read4 = ntohl(bytes_to_read4);
      if (bytes_to_read4==int_u4(0xFFFFFFFF)) {
	// Escape sequence for length coming
	int_u8 bytes_to_read8 = 0;
	readExact_(fd, (char*)&bytes_to_read8, sizeof(bytes_to_read8));
	bytes_to_read = ntohll(bytes_to_read8);
      } else {
	bytes_to_read = bytes_to_read4;
      }
    }

    // Read the header and decide what kind of serialization we have
    char rep[4] = { 0 };
    readExact_(fd, &rep[0], sizeof(rep)); 
    string hdr(rep, sizeof(rep));
    ArrayDisposition_e array_disposition;
    Serialization_e serialization = chooseRecvSerialization_(fd, hdr, 
							    array_disposition);
    // Set-up array to read into
    Array<char> buffer(bytes_to_read+1); // Extra 1 space for zero for M2k
    buffer.expandTo(bytes_to_read);
    buffer.data()[bytes_to_read] = '\0'; // zero for M2k, doesn't hurt others
    
    // Handle reading the different types of headers
    size_t correction = 0;
    MachineRep_e endian = NativeEndian();
    switch (serialization) {
    case SERIALIZE_M2K :   correction = handleReadingM2kHdr_(fd, rep, 
							     endian); break;
    case SERIALIZE_OC:
    case SERIALIZE_P0:
    case SERIALIZE_P2:
    case SERIALIZE_P2_OLD: correction = 0;  break;
    case SERIALIZE_NONE: // In case we read 4 bytes prematurely
      memcpy(buffer.data(), rep, sizeof(rep)); 
      correction = sizeof(rep);
      break;
    default: throw runtime_error("unknown serial:"+Stringize(serialization));
    };
    readExact_(fd, buffer.data()+correction, bytes_to_read-correction);
    unpackageData_(buffer, serialization, array_disposition,
		   retval, endian);
    return retval;
  }

  // Helper function for timed receives
  Val recvTimed_ (int fd, Val timeout_in_seconds = None) 
  {
    // Convert timeout in m.n seconds to (seconds and microseconds)
    struct timeval timeout;
    struct timeval* timeout_p = convertTime_(&timeout, timeout_in_seconds);

    //  Set up the file descriptors:  Really only setting the read
    fd_set rfds;         FD_ZERO(&rfds); 
    int readfd = fd;     FD_SET(readfd, &rfds); 
    int max_of_fds = readfd + 1; // select requires keep max of all fds (+1)
    
    // ... and call select
    int r = mySelect_(max_of_fds, &rfds, NULL, NULL, timeout_p);
    if (r<0) errout_("select");
    
    if (r==0) { // Nothing available or timeout
      return None;
    } else {    // r is the number of descriptors, and ready to go!
      return recvBlocking_(fd);
    }
  }


  // Helper function to pack data (Pickled, raw, M2k, etc.), and
  // return the packed data in the given buffer: Kept for backwards
  // compatibility (but really just calls generic routines).
  void packageData_ (const Val& val, Serialization_e serialization, 
		     ArrayDisposition_e array_disposition,
		     Array<char>& buffer, MachineRep_e endian)
  {
    DumpValToArray(val, buffer, serialization, 
		   array_disposition, compatibilityMode_, endian);
  }
  void packageData_ (const Val& val, Serialization_e serialization, 
		     ArrayDisposition_e array_disposition,
		     Array<char>& buffer)
  {
    DumpValToArray(val, buffer, serialization, 
		   array_disposition, compatibilityMode_, NativeEndian());
  }
  
  // Helper function to unpackage the data for the different types of
  // serialization (Pickling, raw data, etc.). The unpackaging policy
  // is dictated by the arguments.
  void unpackageData_ (Array<char>& buffer, Serialization_e serialization, 
		       ArrayDisposition_e array_disposition,
		       Val& retval, MachineRep_e endian=MachineRep_EEEI)
  
  {
    LoadValFromArray(buffer, retval, serialization, 
		     array_disposition, compatibilityMode_, endian);
  }

  void writeExact_ (int fd, char* data, size_t len) 
  { FDTools_::WriteExact(fd,data,len); }

  // Blocking call to send Val over socket.
  void sendBlocking_ (int fd, const Val& val)
  {
    // Pickle into buffer
    Array<char> buffer(1024);
    ArrayDisposition_e array_disposition;
    Serialization_e serialization = 
      chooseSendSerialization_(fd, array_disposition);
    // Package it up
    MachineRep_e endian=NativeEndian();
    packageData_(val, serialization, array_disposition,
		 buffer, endian);

    // Preamble: number of bytes to write.  If bigger than 4 bytes
    // can hold, we send 4 bytes of ESCAPE_SEQUENCE then the actual
    // 8 bytes.
    {
      int_u8 full_len = buffer.length();
      if (full_len < int_u8(0xFFFFFFFF)) {
	// Okay, len fits in 4 bytes and not esc sequence
	int_u4 bytes_to_write4 = htonl(int_u4(full_len));
	writeExact_(fd, (char*)&bytes_to_write4, sizeof(bytes_to_write4));
      } else {
	int_u4 esc_seq = htonl(0xFFFFFFFF);    
	writeExact_(fd, (char*)&esc_seq, sizeof(esc_seq));
	int_u8 bytes_to_write8 = htonll(full_len);
	writeExact_(fd, (char*)&bytes_to_write8, sizeof(bytes_to_write8));
      }
    }

    // Headers support:  M2k is complex mess, Python is just 4 bytes
    if (serialization == SERIALIZE_M2K) {
      handleWritingM2kHdr_(fd); 
    } else if (serialization != SERIALIZE_NONE) {
      // Header (but only if serialized): 4 bytes of supports Numeric,version #
      string header = createHeader_(serialization, array_disposition); 
      writeExact_(fd, &header[0], 4);
    }

    // Write pickled buffer
    writeExact_(fd, buffer.data(), buffer.length());
  }

  // Helper function for timed-out sends.
  Val sendTimed_ (int fd, const Val& v, Val timeout_in_seconds = None) 
  {
    // Convert timeout in m.n seconds to (seconds and microseconds)
    struct timeval timeout;
    struct timeval* timeout_p = convertTime_(&timeout, timeout_in_seconds);

    //  Set up the file descriptors:  Really only setting the write
    fd_set wfds;          FD_ZERO(&wfds); 
    int writefd = fd;     FD_SET(writefd, &wfds); 
    int max_of_fds = writefd + 1; // select requires keep max of all fds (+1)
    
    // ... and call select
    int r = mySelect_(max_of_fds, NULL, &wfds, NULL, timeout_p);
    if (r<0) errout_("select");
    
    if (r==0) { // Nothing available or timeout, return None
      return None;
    } else {    // r is the number of descriptors, and ready to go!
      sendBlocking_(fd, v);
      return true;
    }
  }


  // Helper code used by M2k for binary serialization: put in .cc so
  // all .m2k issms isolated in .cc file.  

  // Straight from opalmsghdrnet code ... member so it gets readExact
  void readContinueStreamDataEncoding_ (int fd,
					StreamDataEncoding& sde,
					char* rep);
  // This code handles reading the complex M2k preambles.
  int handleReadingM2kHdr_ (int fd, char* rep,
			    MachineRep_e& endian);
  // This code handles writing the complex M2k preamble
  void handleWritingM2kHdr_ (int fd); 

}; // MidasSocket_


PTOOLS_END_NAMESPACE


#define MIDASSOCKET_H_
#endif // MIDASSOCKET_H_
