
#include "fdtools.h"
#include "socketerror.h"
#include <netdb.h>   // for gethostbyname ... this causes conflicts
                     // on tru64 with X-Midas, so it has been moved to a .cc

#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

PTOOLS_BEGIN_NAMESPACE 

// Has to be a global lock so that multiple users serialize on it.
Mutex gethostbynameLock_; // Lock to thread-safe gethostbyname_r


// Definitely a UNIX thing: if you see a SIGPIPE signal, ignore it
void FDTools_::installSIGPIPE_ignore ()
{
  struct sigaction oldact;
  struct sigaction newact;
  newact.sa_handler = SIG_IGN;
  sigemptyset(&newact.sa_mask);
  newact.sa_flags = 0;
  int r = sigaction(SIGPIPE, &newact, &oldact);
  if (r<0) errout_("sigaction");
}


// print time
static inline void printTime (ostream& os, const char* mesg, 
			      const struct timeval& t)
{
  real_8 ts = real_8(t.tv_usec)/1e6 + t.tv_sec;
  os << mesg << ":(" << Val(ts) << ") [" << t.tv_sec << "," << t.tv_usec << "]" << endl;
}

// Some installations of OSF has suseconds_t
#if defined(OSF1_) 
#  if _XOPEN_SOURCE <500 
typedef int suseconds_t;
#endif
#endif 

// Add two times
static inline struct timeval addTimes (const struct timeval& t1, 
				       const struct timeval& t2)
{
  //printTime(cerr, "t1 in addTimes", t1);
  //printTime(cerr, "t2 in addTimes", t2);

  struct timeval ret;
  real_8 t1s = real_8(t1.tv_usec)/1e6 + t1.tv_sec;
  real_8 t2s = real_8(t2.tv_usec)/1e6 + t2.tv_sec;
  real_8 sum = t1s + t2s;
  
  real_8 integral;
  ret.tv_usec  = suseconds_t(1e6*modf(sum, &integral));    
  ret.tv_sec   = time_t(integral);

  return ret;
}

// Substract two times
static inline struct timeval subTimes (const struct timeval& t1, 
				       const struct timeval& t2)
{
  //printTime(cerr, "t1 in subTimes", t1);
  //printTime(cerr, "t2 in subTimes", t2);
  struct timeval ret;
  real_8 t1s = real_8(t1.tv_usec)/1e6 + t1.tv_sec;
  real_8 t2s = real_8(t2.tv_usec)/1e6 + t2.tv_sec;
  real_8 diff = t1s - t2s;

  real_8 integral;
  ret.tv_usec  = suseconds_t(1e6*modf(diff, &integral));    
  ret.tv_sec   = time_t(integral);

  //printTime(cerr, "result in subTimes", ret);

  return ret;
}

// Get the current time
static inline struct timeval now ()
{
  struct timeval current_time; 
  gettimeofday(&current_time, NULL);
  return current_time;
}


bool FDTools_::dataReady_ (int read_fd, Val timeout_in_seconds)
{
  if (read_fd<0) throw runtime_error("dataReady called with closed fd");

  // Convert timeout in m.n seconds to (seconds and microseconds)
  struct timeval timeout;
  struct timeval* timeout_p = convertTime_(&timeout, timeout_in_seconds);

  //  Set up the file descriptors:  Really only setting the read
  fd_set rfds;           FD_ZERO(&rfds); 
  int readfd = read_fd;  FD_SET(readfd, &rfds); 
  int max_of_fds = readfd + 1; // select requires keep max of all fds (+1)
  
  // ... and call select
  int r = mySelect_(max_of_fds, &rfds, NULL, NULL, timeout_p);
  if (r<0) errout_("select");
  
  // r is the number of descriptors, and ready to go!
  return r != 0;
}

// Select can return EINTR which is not truly a timeout,
// so we need to have some code around select to handle this
int FDTools_::mySelect_ (int nfds, 
			 fd_set* readfds, fd_set* writefds, fd_set* errfds,
			 struct timeval *passed_timeout_p)
{
  // Timeouts have to be updated if select returns prematurely
  int r;
  struct timeval time_out, *time_out_p;
  if (passed_timeout_p==NULL) {
    time_out_p = NULL;
    //cerr << "NULL timeout" << endl;
  } else {
    time_out = *passed_timeout_p;
    time_out_p = &time_out;
    //printTime(cerr, "Non null timeout", time_out);
  }
  
  while (1) {
    struct timeval mark_time = now();     // mark time when started select
    //printTime(cerr, "mark time in mySelect", mark_time);
    r = select(nfds, readfds, writefds, errfds, time_out_p);
    if (r==-1 && errno==EINTR) { 
      // Interrupted system call:  recalculate timeout
      //cerr << "EINTR!" << endl;
      if (time_out_p==NULL) continue; // try again when NULL timeout  

      // Need to recompute timeout
      struct timeval current_time = now(); // mark time select finished premat 
      //printTime(cerr, "mySelect EINTR", current_time);
      time_out = subTimes(time_out, subTimes(current_time, mark_time));
      if (time_out.tv_sec<0 || time_out.tv_usec<0) {
	struct timeval zero;
	zero.tv_usec = 0; zero.tv_sec = 0;
	time_out = zero; // To make sure select returns the proper return code 
      }                  // immediately
      continue;
    } else {
      break;
    }
  }

  return r;
}


// A truly re-entrant (or at least thread-safe) version of
// gethostbyname_r
struct hostent FDTools_::getHostByNameThreadSafe_ (const string& hostname,
						   Array<char>& host_buff)
{
  // Get the host by name.  The re-entrant version stores the result
  // in the given buffer ... since it's kinda of unclear how much
  // space the routine needs, we allow it to keep regrowing and
  // allow retries.
  host_buff.expandTo(host_buff.capacity());
  struct hostent h, *hp;
  int r, herr;
  bool first_time = true;
  while (1) {  // continue to retry gethostbyname_r if buffer too small
    
    if (!first_time) { // Slight optimization, only expand if needed
      host_buff.expandTo(host_buff.length()*2); // double size
    }
    first_time = false;
    
    // According to Stevens UNIX Network Programming, p. 305,
    // gethostbyname_r may STILL not be re-entrant because bind may
    // not be re-entrant underneath.  We get back the re-entrant
    // data structure, but force serialization of all calls to
    // gethostbyname_r.
    {
      ProtectScope ps(gethostbynameLock_);
      
#if defined(SOLARIS_)
      
      hp = gethostbyname_r(hostname.c_str(), &h,
			   host_buff.data(), host_buff.length(), &herr);
      if (hp==0 && errno==ERANGE) continue; // resize, buffer too small
      else if (hp!=0) break;
      else throw gaierror("gethostname_r:"+string(hstrerror(herr)));
      
#elif defined(LINUX_)
      
      r = gethostbyname_r(hostname.c_str(), &h,
			  host_buff.data(), host_buff.length(), &hp, &herr);
      if (r==ERANGE || r==EINVAL) { // resize, buffer too small
	// cerr << "resize:" << host_buff.length() << endl;
	continue; 
      } 
      // Hmm ... the man says we return ERANGE if the buffer is too small
      // but code seems to think it returns EINVAL
      else if (r==0 && hp!=0) break;
      else throw gaierror("gethostname_r:"+string(hstrerror(herr)));
      
#elif defined(OSF1_)
      
      // resize until Array is same size as hostent_data struct,
      // which everything in hp points to
      if (sizeof(struct hostent_data)>=host_buff.length()) {
	// cerr << "resize:" << host_buff.length() << endl;
	continue;  
      }
      struct hostent_data* d = (struct hostent_data*)host_buff.data();
      r = gethostbyname_r(hostname.c_str(), &h, d);
      if (r<0) {
	herr = h_errno;
	string errmsg = "Unknown?";
	switch (herr) {
	case HOST_NOT_FOUND: errmsg = "Host not found"; break;
	case NO_DATA:        errmsg = "No data"; break;
	case NO_RECOVERY:    errmsg = "No recovery"; break;
	case TRY_AGAIN:      errmsg = "Try again"; break;
	}
	throw gaierror("gethostname_r:"+errmsg);
      }
      else break;
#else
#  error 666  // gethostbyname_r is very platform specific: specify platform
#endif
    } // ProtectScope to serialize call to gethostbyname_r
    
  } // continue retrying if buffer was too small
  
  return h;
}


void FDTools_::setblocking_ (int fd, Val timeout)
{
  int flags = fcntl(fd, F_GETFL, 0);
  //cerr << "Timeout is " << timeout << endl;
  if (timeout==None) { // Blocking 
    fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
  } else { // anything else, non-blocking
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }
}


int FDTools_::nonBlockingConnect_ (int* fdp, struct sockaddr_in* addr,
				   Val timeout_in_seconds)
{
  // Helper function to allow connect to timeout quicker (the
  // standard timeout for connect seems to be minutes).  We do
  // this by setting the socket descriptor to NON-BLOCKING for the
  // duration of the connect call, do a select, then return back
  // to BLOCKING.  The current coding makes it so it defers the
  // timeout error to happen in socket init instead: this seems
  // more stable (and allows reconnects easier)
  if (timeout_in_seconds!=None) 
    setblocking_(*fdp, 1.0);
  int r = connect(*fdp,(struct sockaddr*)addr,sizeof(struct sockaddr_in));

  if (r && errno != EINPROGRESS) { // expect in progress
    errout_("helpOpenSocket"); // Try to have same error as before
  }
  errno = 0;  // Clears errno

  // r==0, Connect immediately connected,prolly same machine
  if (r!=0) { 
    // Assertion: in progress, do a select wating for write or read

    // Convert timeout in m.n seconds to (seconds and microseconds)
    struct timeval timeo;
    struct timeval* timeout_p = convertTime_(&timeo, timeout_in_seconds);
    
    // Same file descriptor on both read and write, idea from Stevens book
    fd_set rset, wset; 
    FD_ZERO(&rset);
    int sockfd = *fdp;
    FD_SET(sockfd, &rset);
    wset = rset;
    
    r = mySelect_(sockfd+1, &rset, &wset, NULL, timeout_p);
    if (r==0) { // timeout
      ::close(sockfd);
      *fdp = -1;
      throw timeout("open timed out");
    }
    if (r<0) errout_("select");
    // If here, some file descriptor pass in was active
    if (FD_ISSET(sockfd, &rset) || FD_ISSET(sockfd, &wset)) {
      int error;
#if defined(OSF1_)
      int len = sizeof(error);
#else
      socklen_t len = sizeof(error);
#endif
      if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &len)<0) {
	throw runtime_error("Internal socket error 1???");
      }
    } else {
      throw runtime_error("Internal socket error 2???");
    }
  }
  setblocking_(*fdp, None);
  return 0;
}


void FDTools_::helpOpenSocket_ (int domain, int type, int protocol,
				HelpOpenSocket_e connect_or_bind,
				const string& host, int port,
				int* fdp, struct sockaddr_in* addr,
				Val timeout_in_seconds)
{
  Array<char> a(2); // This array needs to stay around for the struct
  struct hostent h = getHostByNameThreadSafe_(host,a);

  // Assertion: At this point, we have the host name.  We connect to
  // the first host we find.
  struct in_addr **pptr = (struct in_addr**)h.h_addr_list;
  for (; *pptr!=NULL; pptr++) {

    int r = socket(domain, type, protocol);
    if (r<0) { 
      *fdp = -1;
      bzero(addr, sizeof(struct sockaddr_in));
      errout_("socket");
    }
    *fdp = r; // commit file desc.

    bzero(addr, sizeof(struct sockaddr_in));
    addr->sin_family = domain;
    addr->sin_port = htons(port);
    memcpy(&addr->sin_addr, *pptr, sizeof(struct in_addr));

    if (connect_or_bind==HELP_CONNECT) {
      r = nonBlockingConnect_(fdp, addr, timeout_in_seconds);
      if (r==0) return; 
    } else if (connect_or_bind==HELP_BIND) {
      r = bind(*fdp,(struct sockaddr*)addr,sizeof(struct sockaddr_in));
      if (r==0) return; // Bound ... all done
    } else {
      return;           // do nothing
    }

    // Clean up in case bind or connect failed
    ::close(*fdp);
    *fdp = -1;
    bzero(addr, sizeof(struct sockaddr_in));
  }
  errout_("helpOpenSocket");

  return;

}

PTOOLS_END_NAMESPACE

