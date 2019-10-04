#ifndef FDTOOLS_H_
#define FDTOOLS_H_

extern "C" {
#if defined(_MSC_VER) || defined(__MINGW32__)
    #include <io.h>
    #include <Winsock2.h>
    #include <string.h>
#else
    #include <unistd.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <sys/socket.h>
#endif
#include <signal.h>
#include <sys/types.h>

#if defined(_MSC_VER)
    typedef unsigned __int64    ssize_t;
#endif

};
#include "ocval.h"
#include "socketerror.h"

// Force the struct hostent into the global environment, even with just
// a silly typedef
typedef struct hostent for_sill;

PTOOLS_BEGIN_NAMESPACE

// This is a system structure for managing sockaddr from accepts.  It
// also shows how "non-Val" things can be converted to and from Vals.
// THIS WORKS BECAUSE SOCKADDR can be passed by value.
struct SockAddr_ {

  // Basic data of struct
  struct sockaddr addr;
#if defined(OSF1_) || defined(_MSC_VER) || defined(__MINGW32__)
  int             addrlen;
#else
  socklen_t       addrlen;
#endif

  SockAddr_ () : addrlen(sizeof(addr)) { }

  // Allow users to construct a SockAddr_ from Val, if legal
  SockAddr_ (const Val& sa)
  {
    if (sa.tag=='a') {
      OCString* sp = (OCString*) &sa.u.a;
      OCString& s = *sp;
      memcpy(&addr, &s[0], s.length());
      addrlen = s.length();
    } else {
      throw runtime_error("Don't know how to convert:"+string(sa)+" to SockADdr_");
    }
  }

  // Allow users to construct a Val from a SockAddr_
  operator Val () const 
  { // Just bit-blit into a string
    Val retval;
    string s((char*)&addr, addrlen);
    retval = s;
    return retval;
  }

}; // SockAddr_


// Dumb helper routine for debugging
inline void debugArray_ (Array<char>& buffer, const char* routine)
{
  cerr << "****" << routine << "****" << endl;
  for (size_t ii=0; ii<buffer.length(); ii++) {
    char c = buffer[ii];
    if (isprint(c)) cerr << c;
    else {
      char s[4] = { 0,0,0,0 };
      s[0] = 'x';
      int g = (c >> 4);
      s[1] = g["0123456789ABCDEF"];
      s[2] = (c &0x0F)["0123456789ABCDEF"];
      s[3] = 0;
      string hexer = s;
      cerr << "\\0" << hexer;
    }
  }
}

// Handle 8 byte Network Host Order
inline int_u8 (htonll) (int_u8 bytes)
{
  int_u8 check = 1;
  char *check1 = reinterpret_cast<char*>(&check);
  if (*check1==1) { 
    // This is little-endian machine, have to convert to big
    char *look = reinterpret_cast<char*>(&bytes);
    OC_NAMESPACED::swap(look[0], look[7]);
    OC_NAMESPACED::swap(look[1], look[6]);
    OC_NAMESPACED::swap(look[2], look[5]); 
    OC_NAMESPACED::swap(look[3], look[4]); 
    return bytes;
  } else {
    // Big endian, do nothing
    return bytes;
  }
}

// Handle 8 byte Network Host Order
inline int_u8 (ntohll) (int_u8 bytes)
{
  int_u8 check = 1;
  char *check1 = reinterpret_cast<char*>(&check);
  if (*check1==1) { 
    // This is little-endian machine, have to convert to big
    char *look = reinterpret_cast<char*>(&bytes);
    OC_NAMESPACED::swap(look[0], look[7]);
    OC_NAMESPACED::swap(look[1], look[6]);
    OC_NAMESPACED::swap(look[2], look[5]); 
    OC_NAMESPACED::swap(look[3], look[4]); 
    return bytes;
  } else {
    // Big endian, do nothing
    return bytes;
  }
}


// A class for handling lots of file-descriptor related
// activities, usually for sockets.
class FDTools_ {

  enum { MAX_BUFF = 1024 }; // I/O buffering on socket
  
 public:

  // Initialize the file descriptor.  Set buffer size ff we plan to use 
  // a buffer (this is more for free-form formatted data (like HTTP) 
  // where we can't be sure how far to read until we see some parsing markers).
  FDTools_ (Val timeout, int& read_fd, int& write_fd, int buffer_cap=MAX_BUFF) :
    timeout_(timeout),
    readFD_(read_fd),
    writeFD_(write_fd), 
    buffer_(0), 
    length_(0),
    where_(0),
    unused_(-1)
  {
    if (buffer_cap) buffer_ = new char[buffer_cap];
  }

  // Use fdtools more for the dataReady, select usage
  FDTools_ () :
    timeout_(None),
    readFD_(unused_),
    writeFD_(unused_), 
    buffer_(0), 
    length_(0),
    where_(0),
    unused_(-1)
  { }

  virtual ~FDTools_ ()
  {
    delete [] buffer_;
  }

  // Close down both read and write (frequently the same)
  virtual void close ()
  {
    cleanseBuffer();
    if (readFD_ == writeFD_) {
      closeUp_(readFD_);
      writeFD_ = -1;
    } else {
      closeUp_(readFD_);
      closeUp_(writeFD_);
    }

  }

  // All exceptions thrown by the MidasTalker are runtime_errors
  static void errout (const string& message) 
  {
    string m = strerror(errno);
    throw runtime_error(message+":"+m);
  }


  // Helper blocking read method: Reads ONLY from input 
  // (Doesn't observe the buffer in FDTools)
  static void ReadExact (int fd, char* buff, size_t bytes) 
  {
    // Sometimes reads get broken up... it's very rare (or
    // impossible?)  to get an empty read on a blocking call.  If we
    // get too many of them, then it probably means the socket went
    // away.
    int empty_retries = 1000;
    char* n_buff = buff;
    size_t bytes_to_read  = bytes;
    while (bytes_to_read) {
      // raw read: no timeout checking
      ssize_t r = read(fd, n_buff, bytes_to_read);
      if (r<0) errout("read");   
      // Instead of getting a SIGPIPE, when we ignore the SIGPIPE signal,
      // a number of "empty retries" is equivalent to the SIGPIPE, but then
      // at least we catch it.
      if (r==0 && --empty_retries<0) errout("read: too many empty retries");
      if (size_t(r) > bytes_to_read) errout("read:got more bytes back than asked for?");
      bytes_to_read -= r;
      n_buff        += r;
    }
  }
  void readExact (char* buff, size_t bytes) 
  { FDTools_::ReadExact(readFD_, buff, bytes); }


  // Helper, blocking write method
  static void WriteExact (int fd, const char* buff, size_t bytes) 
  {
    // Sometimes writes get broken up... it's very rare (or
    // impossible?)  to get an empty write on a blocking call.  If we
    // get too many of them, then it probably means the socket went
    // away.
    int empty_retries = 1000;
    const char* n_buff = buff;
    size_t bytes_to_write  = bytes;
    while (bytes_to_write) {
      ssize_t r = write(fd, n_buff, bytes_to_write);
      if (r<0) errout("write");
      if (r==0 && --empty_retries<0) errout("write: too many empty retries");
      if (size_t(r)> bytes_to_write) errout("write: more bytes written than asked?");
      bytes_to_write -= r;
      n_buff         += r;
    }
  }

  void writeExact (const char* buff, size_t bytes) 
  { FDTools_::WriteExact(writeFD_, buff, bytes); }

  // Buffer-centric:
  // Get a single character, buffered up.
  int peekAChar ()
  {
    // Nothing there: need to read something
    if (where_>=length_) {
      int empty_retries=1000;
      while (1) {
        if (empty_retries<0) return EOF;
	checkTimeout_();
	ssize_t r = read(readFD_, &buffer_[0], MAX_BUFF);
	// 0: probably, socket went away, try again a few times just to be sure
	if (r==0) {
	  empty_retries--;
	  continue;
	}
	// -1: Error?
	if (r==-1) {
	  // Just a signal or EAGAIN retry
	  int error_num = errno;
	  if (error_num==EINTR || error_num==EAGAIN) {
	    empty_retries--;
	    continue;
	  }
	  // Really an error
	  errout("Problems reading");
	}
	// >0, something was read
	length_ = r;
	where_  = 0;
	break;
      }
    }
    return buffer_[where_];
  }

  // Buffer-centric:
  // Look at the next character: if not there, may be EOF
  // or may have to just read a little more
  int getAChar ()
  {
    int c = peekAChar();
    if (c!=EOF) where_++;
    return c;
  }

  // Buffer-centric:
  // Read until data is full or EOF, reading from buffer
  // if some is already read:  Returns 0 if everything okay,
  // EOF if we think we saw end of file, and throws runtime_error
  // if there's an error.
  int clientReadExact (char* data, size_t initial_len)
  {
    size_t read_so_far   = 0;
    size_t still_to_read = initial_len;

    // Stuff still in buffer
    if (where_<length_) {
      size_t buffer_len = length_ - where_;
      size_t copy_from_buffer = min(initial_len, buffer_len);
      
      memcpy(data, &buffer_[where_], copy_from_buffer);
      read_so_far += copy_from_buffer;
      still_to_read -= copy_from_buffer;
      where_ += copy_from_buffer;
    }

    // Now that buffer is empty, read from socket
    int empty_retries=1000;
    while (still_to_read) {
      if (empty_retries<0) return EOF;
      checkTimeout_();
      ssize_t r = read(readFD_, &data[read_so_far], still_to_read);
      // 0: probably, socket went away, try again a few times just to be sure
      if (r==0) {
	empty_retries--;
	continue;
      }
      // -1: Error?
      if (r==-1) {
	// Just a signal or EAGAIN retry
	int error_num = errno;
	if (error_num==EINTR || error_num==EAGAIN) {
	  empty_retries--;
	  continue;
	}
	// Really an error
	errout("Problem when reading data");
      }
      // >0, something was read
      read_so_far += r;
      still_to_read -=r ;
    }
    return still_to_read;
  }

  // Buffer-centric:
  // Read the rest of the data from the socket, and plop it
  // in the array: reads from the buffer first, if there was
  // already stuff still there.  May throw runtime_error
  // if there are problems.
  void readUntilFull (Array<char>& data_from_socket, 
		      size_t amt_of_data_to_read)
  {
    if (amt_of_data_to_read<=0) {
      // Just keep reading until we are done: not the best way to
      // proceed, because at EOF we close the socket? TODO: Check
      // that.
      char c;
      while ((c=getAChar()) != EOF) {
	data_from_socket.append(c);
      }
    } else {
      // We can't use readExact_ because we have buffered
      // some of the data.
      size_t len = data_from_socket.length();
      data_from_socket.expandBy(amt_of_data_to_read);
      int err = clientReadExact(&data_from_socket[len], 
				amt_of_data_to_read);
      if (err) {
	throw runtime_error("Malformed request? Not enough data?");
      }
    }
  }


  // Buffer-centric:
  // Helper for getPreamble_: gets a single line and returns its length.
  // Discards the CR or CRLF at the end of the line.
  // May throw runtime_error if there is a problem.
  int getLine (string& inplace_str)
  {
    Array<char> inplace_get;
    while (1) {
      // Assertion: should be able to read something
      int c = getAChar();

      // End of file: strictly, we need CRLF, but we will accept, with a warn
      if (c==EOF) {
	break;
      }

      // End of line: strictly, we need CRLF, but we accept LF as well
      if ((c=='\r' && peekAChar()=='\n')) { 
	getAChar();
	break;
      } else if (c=='\n') {
	break;
      }
      inplace_get.append(c);
    }
    inplace_str = string(inplace_get.data(), inplace_get.length());
    return inplace_str.length();
  } 


  // throw away everything from buffer
  void cleanseBuffer () { length_ = where_ = 0; }

  // Uggy
  int& fd ()  { return readFD_; } 

  // Definitely a UNIX thing: if you see a SIGPIPE signal, ignore it
  void installSIGPIPE_ignore ();

 protected:

  // Timeout for all reads: floating point number
  // None means "block until read succeeds", 0 means no blocking
  // can use floating point seconds
  Val timeout_;

  // Hold a REFERENCE back to the file descriptor that is used
  // for reading and writing: frequently, they are the same.
  int& readFD_;
  int& writeFD_;

  // For reading routines that need to hold a buffer
  char* buffer_;
  size_t length_;   // 0..length_-1 is valid of buffer
  size_t where_;    // where is betwen 0 and length_

  // Complelty unused fd
  int unused_;

  // All exceptions thrown by the MidasTalker are runtime_errors
  void errout_ (const string& message) { FDTools_::errout(message); }

  // Shutdown this fd
  virtual void closeUp_ (int& fd)
  {
    if (fd!=-1) {
#if defined(_MSC_VER) || defined(__MINGW32__)
        int flag = SD_BOTH;
#else
        int flag = SHUT_RDWR;
#endif
      shutdown(fd, flag);
      ::close(fd);
    } 
    fd = -1;
  }

  // Checks to see if there is data to get: if not, throws a
  // timeout exception
  void checkTimeout_ (Val timeout_=None)
  {
    if (!dataReady_(readFD_, timeout_)) {
      throw timeout("timeout of "+string(timeout_)+" on read");
    }
  }


  // This routine turns a m.n seconds into a timestruct with (seconds,
  // microseconds).  Also, if the timeout value is "None", the time
  // val struct us set to NULL meaning that select 
  struct timeval* convertTime_ (struct timeval* tv, Val timeout_in_seconds) 
  {
    // No timeout, set to null: This causes select to block indef.
    if (timeout_in_seconds == None) return NULL;
    
    // Convert timeout in m.n seconds to (seconds and microseconds)
    real_8 full_timeout = timeout_in_seconds;    
    tv->tv_sec  = long(full_timeout); // Get seconds
    real_8 t = (full_timeout-real_8(tv->tv_sec))*1e6;
    tv->tv_usec = long(t);            // Get microseconds
    
    return tv;
  }


  // Helper function: This function returns True immediately if the
  // given file descriptor (for a socket) is available to read without
  // blocking.  If the socket is NOT available, then the socket is
  // watched for timeout seconds: if something becomes available on the
  // socket in that time, this returns True, otherwise the entire
  // timeout passes and this returns False.  Note that this method
  // allows a user to 'poll' a read socket without having to do an
  // actual read.  A timeout of None is forever, a timeout of 0 is no
  // wait.
  bool dataReady_ (int read_fd, Val timeout_in_seconds=0);
  

  // Select can return EINTR which is not truly a timeout,
  // so we need to have some code around select to handle this
  int mySelect_ (int nfds, fd_set* readfds, fd_set* writefds, fd_set* errfds,
		 struct timeval *timeout);
  
  // A truly re-entrant (or at least thread-safe) version of
  // gethostbyname_r
  struct hostent getHostByNameThreadSafe_ (const string& hostname, 
					   Array<char>& host_buff);  
  
  // Help someone open a socket on a named host
  enum HelpOpenSocket_e { HELP_CONNECT, HELP_BIND, HELP_NOTHING };
  void helpOpenSocket_ (int domain, int type, int protocol,
			HelpOpenSocket_e connect_or_bind,
			const string& name, int port,
			int* fd,struct sockaddr_in* addr,
			Val timeout=None);

  // Set socket to blocking or non-blocking: None means put it
  // in blocking mode, otherwise it goes into non-blocking mode
  void setblocking_ (int fd, Val timeout);


  // Helper function to allow connect to timeout quicker (the
  // standard timeout for connect seems to be minutes).  We do
  // this by setting the socket descriptor to NON-BLOCKING for the
  // duration of the connect call, do a select, then return back
  // to BLOCKING.  The current coding makes it so it defers the
  // timeout error to happen in socket init instead: this seems
  // more stable (and allows reconnects easier).  Returns 0 if
  // everything okay.
  int nonBlockingConnect_ (int* fdp, struct sockaddr_in* addr,
			   Val timeout_in_seconds);


}; // FDTools

PTOOLS_END_NAMESPACE

#endif // FDTOOLS_H_
