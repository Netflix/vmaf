#ifndef HTTPCONNECTION_H_
#define HTTPCONNECTION_H_

#include "ocval.h" 
#include "httptools.h"
#include "fdtools.h"
#include "midastalker.h"

// Modelled after Python's HTTPConnection and HTTPResponse
//
// ///////////////Here is an example session that uses the GET method:
// #include "httplib.h"
// HTTPConnection conn("www.python.org");
// conn.request("GET", "/index.html");
// HTTPResponse r1 = conn.getresponse();
// cout << r1.status <<  r1.reason << endl;
// // output: 200 OK
// data1 = r1.read()
// conn.request("GET", "/parrot.spam")
// HTTPResponse r2 = conn.getresponse();
// cout << r2.status <<  r2.reason << endl;
// // output: 404 Not Found
// Array<char> data2 = r2.read();
// conn.close();

// Here is an example session that uses the HEAD method. Note that the 
// HEAD method never returns any data.

// #include "httplib.h"
// HTTPConnection conn("www.python.org");
// conn.request("HEAD","/index.html");
// HTTPResponse res = conn.getresponse();
// cout << res.status() << " " << res.reason() << endl;
// // output: 200 OK
// Array<char> data = res.read();
// cout << data.length() << endl;
// output: 0
//>>> data == ''
//True

// Here is an example session that shows how to POST requests:

// #include "httplib.h"
// #include "urllib.h"
// string params = URLEncode(Tab("{'@number': 12524, '@type': 'issue', '@action': 'show'}"));
// headers = Tab('{"Content-type": "application/x-www-form-urlencoded",
//            "Accept": "text/plain"}');
// HTTPConnection conn("bugs.python.org");
// conn.request("POST", "", params, headers);
// HTTPResponse response = conn.getresponse();
// cout << response.status << " " << response.reason << endl;
// // output: 302 Found
// Array<char> data = response.read();
// cerr << data;
//'Redirecting to <a href="http://bugs.python.org/issue12524">http://bugs.python.org/issue12524</a>'
// conn.close();

PTOOLS_BEGIN_NAMESPACE

class HTTPConnection; // Forward

class HTTPResponse {

  friend class HTTPConnection;

 public:

  // Reads and returns the response body, TODO: or up to the next amt bytes.
  Array<char> read (int /*amt*/=-1)
  {
    Array<char> result; // RVO
    HTTPTools_&  httptools = getHTTPTools_();

    // Head ONLY returns the headers
    if (method_ == "HEAD") return result;

    try {
      
      // make sure we can get data
      checkTimeout_();

      // Chunked?
      if (headers_.contains("transfer-encoding") && 
	  headers_("transfer-encoding")=="chunked") {
	OTab footers; // TODO: something with the footers?
	
	if (httptools.getSingleChunkedResponse(result, footers)) {
	  this->close();
	}
      } 

      // Not chunked: So all done at end
      else {
	int len = -1;
	if (headers_.contains("content-length")) {
	  len = headers_("content-length");
	}
	httptools.readUntilFull(result, len);
	this->close();
      }

    } catch (const runtime_error& err) {
      this->close();
    }
    return result;
  }

  // Close the connection
  inline void close ();


  // Get the contents of the header name, or default if there is no 
  // matching header.
  const Val getheader (const string& name, const Val& defaultp=None) const
  { return headers_.get(name, defaultp); }

  // Python: Return a list of (header, value) tuples.  
  // C++:    Return an OTab
  OTab& getheaders() { return headers_; }

  // Returns the fileno of the underlying socket.
  int fileno() { return getHTTPTools_().fd(); } 

  // A mimetools.Message instance containing the response headers.
  Val msg () const { throw runtime_error("not implemented"); }

  // HTTP protocol version used by server. 10 for HTTP/1.0, 11 for HTTP/1.1.
  string version () const { return version_; }

  // Status code returned by server.
  int status () const { return status_; }

  // Reason phrase returned by server.
  string reason () const { return reason_; }

  // is the connection still open?  Can be for chunked responses
  bool isclosed () const; 

 protected:

  HTTPConnection& connection_;
  string method_;
  string version_;
  int    status_;
  string reason_;
  OTab headers_;


  // Create an HTTP response: Can only be created by an HTTPConnection
  HTTPResponse (HTTPConnection& backptr, const string& method,
		const string& version, int status, const string& reason,
		OTab& headers) :
    connection_(backptr),
    method_(method),
    version_(version),
    status_(status),
    reason_(reason),
    headers_(headers)
  { }

  // Helper function: forward declared since HTTPConnection not defined yet
  HTTPTools_& getHTTPTools_();

  // Throws a runtime exception if, after timeout seconds, there
  // is no data on the fd
  void checkTimeout_();

}; // HTTPResponse

// ////////////////////////////////////////////// HTTPConnection Class

class HTTPConnection : protected MidasTalker { 

  friend class HTTPResponse;

 public:

  // Create a connection to an HTTP Connection
  HTTPConnection (const string& host, int port=80, Val timeout=None) :
    MidasTalker(host, port, SERIALIZE_NONE, NORMAL_SOCKET),
    httptools_(s_[0], s_[1]), // bind the http tools to the socket desciptor of MidasTalker
    timeout_(timeout)
  { }
 
  // This will send a request to the server using the HTTP request method 
  // *method* and the selector url. If the body argument is present, it 
  // should be a string of data to send after the headers are finished. 
  // Alternatively, it may be an open file object, in which case the
  // contents of the file is sent; this file object should support fileno() 
  // and read() methods. The header Content-Length is automatically set to 
  // the correct value. The headers argument should be a mapping of extra 
  // HTTP headers to send with the request.
  void request (const string& method, const string& url, 
		const string& name="", const OTab& headers = OTab())
  { request_(method, url, name.data(), name.length(), headers); }

  void request (const string& method, const string& url, 
		const Array<char>& a, const OTab& headers = OTab())
  { request_(method, url, a.data(), a.length(), headers); }


  // Should be called after a request is sent to get the response from the 
  // server. Returns an HTTPResponse instance.
  // Note that you must have read the whole response before you can send a 
  // new request to the server.
  HTTPResponse getresponse () 
  {
    // See if we can get result back
    if (!dataReady(timeout_)) { // dataReady from MidasTalker
      throw timeout("getresponse");
    }

    // Get the response and parse
    string initial_line;
    OTab headers;
    httptools_.getPreamble(initial_line, headers);

    // Parse the intro line
    Array<string> a = Split(initial_line, None,2);
    if (a.length() != 3) FDTools_::errout("Unkempt response:"+initial_line);
    string& version = a[0];
    int status      = StringToInt<int>(a[1].data(), a[1].length());
    string& reason  = a[2];
    
    HTTPResponse resp(*this, method_, version, status, reason, headers);
    return resp;
  }

  void connect () { open(timeout_); }
  void close ()   { MidasTalker::close(); }

 protected:

  string method_;         // Which method was last called
  HTTPTools_ httptools_;  // Tools for managing the socket
  Val timeout_;           // None if block forever, or float seconds otherwise

  // Helper 
  void request_ (const string& method, const string& url, 
		 const char* data, int len, const OTab& headers)
  {
    httptools_.close();
    connect();

    // Make sure 1.1 contains host if not already there
    OTab hdrs = headers;
    if (!(hdrs.contains("Host") || hdrs.contains("host"))) {
      hdrs["host"] = host() + ":" + Stringize(port());
    }

    method_ = method;
    
    try {
      httptools_.sendHTTPRequest(method, url, "HTTP/1.1", hdrs, data, len);
    } catch (const runtime_error& re) {
      // cerr << re.what() << endl;
      this->close();
    }
  }

}; // HTTPConnection


// Because HTTPResponse defined before HTTPConnection, have to define
// methods here

inline HTTPTools_& HTTPResponse::getHTTPTools_ () 
                                   { return connection_.httptools_; }
inline void HTTPResponse::close () { connection_.close(); }
inline bool HTTPResponse::isclosed () const { return !connection_.isOpen(); }
inline void HTTPResponse::checkTimeout_ () 
{ 
  if (!connection_.dataReady(connection_.timeout_)) {
    throw timeout("HTTPResponse:read");
  }
}

PTOOLS_END_NAMESPACE

#endif // HTTPCONNECTION_H_
