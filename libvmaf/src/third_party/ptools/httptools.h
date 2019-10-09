#ifndef HTTPTOOLS_H_
#define HTTPTOOLS_H_

#include "fdtools.h"
#include "ocsplit.h"

// A group of tools for creating HTTP messages and responses.
// These keep track of the open connection: this holds the
// file descriptor of the socket open, witbh some convenience
// routines for accessing the socket.

#define CRLF "\r\n"    // Most HTTP initial lines, headers end with CRLF


PTOOLS_BEGIN_NAMESPACE 

// /////////////////////////////////////// Class HTTPTools_

class HTTPTools_ : public FDTools_ {
 public:
  // Create an HTTP 
  HTTPTools_ (int& read_fd, int& write_fd, Val timeout=9.5) :
    FDTools_(timeout, read_fd, write_fd),  
    httpVersion_("HTTP/1.1"),
    serverName_("PicklingToolsServer")
  { }


  // Get the top body, which describes what comes next:
  //   intial line CRLF
  //   Header CRLF
  //   Header CRLF
  //   ...
  //   Header CRLF
  //   CRLF
  // Basically, we are looking for a CRLF on a line by itself to tell
  // us when we are done.
  void getPreamble (string& initial_line, OTab& headers)
  {
    // Get initial line
    if (getLine(initial_line) == 0) {
      HTTPBadRequest("No inital-line on HTTP request");
    }
    getHeaders(headers);
  }

  // Build the preamble for the response
  virtual void buildHTTPResponsePreamble (const string& http_version, 
					  int error_code, 
					  const string& human_text, 
					  const OTab& headers,
					  Array<char>& response)
  {
    string initial_line = http_version + " " + Stringize(error_code)+ " " +
      human_text + CRLF;
    int len = response.length();
    response.expandBy(initial_line.length());
    memcpy(response.data()+len,initial_line.data(), initial_line.length());
    formHeaders(headers, response, http_version);
  }

  // build and send an http request
  virtual void sendHTTPRequest (const string& method,
				const string& url,
				const string& version,
				const OTab& orig_headers,
				const char* data, int len)
  {
    string u = url.length()==0 ? "/" : url;
    string initial_line = method + " " + u + " " + version + CRLF;
    Array<char> preamble(initial_line.length());
    preamble.expandTo(initial_line.length());
    memcpy(preamble.data(), initial_line.data(), initial_line.length());

    // If we are sending content, make sure we have a length
    OTab headers(orig_headers);
    if (len) {
      if (!(headers.contains("content-length") || headers.contains("Content-Length"))) {
	headers["content-length"] = len;
      }
    }
    formHeaders(headers, preamble, version);

    // Send preamble
    writeExact(preamble.data(), preamble.length());
    // Send data .. if there is some
    if (len) {
      writeExact(data, len);
    }
  }


  // Helper: once the HTTP response and data
  virtual void sendHTTPResponse (const Array<char>& preamble,
				 const string& data = string(""),
				 bool throw_exception=true)
  {
    writeExact(preamble.data(), preamble.length());
    if (data.length()>0) {
      writeExact(data.data(), data.length());
    }
    // Once response is sent, throw an exception to force the
    // connection closed on error conditions.  Have the exception 
    // have all the preamble data for logging purposes
    if (throw_exception) {
      string err = string(preamble.data(), preamble.length());
      throw runtime_error(err.c_str());
    }
  }

  // Valid response, simple.  This can be used with chunked coding
  // by making the raw_response_data empty and using sendChunked
  // routines below.
  virtual void HTTPValidResponse (const string& raw_response_data,
				  const string& type,
				  OTab& headers)
  {
    headers["Server"] = serverName_;
    headers["Content-Type"] = type;
    if (!headers.contains("Transfer-Encoding")) {
      // 1.1 chunked and content-length are mutually exclusive
      headers["Content-Length"] = raw_response_data.length();
    }
    Array<char> preamble;
    buildHTTPResponsePreamble(httpVersion_, 200, "OK", headers,
			      preamble);
    sendHTTPResponse(preamble, raw_response_data, false);
  }

  virtual void HTTPValidResponse (const string& raw_response_data,
				  const string& type = "text/xml")
  {
    OTab headers;
    HTTPValidResponse(raw_response_data, type, headers);
  }


  // Send back a response of problems with the client message
  virtual void HTTPBadRequest (const char* msg = NULL)
  {
    OTab headers;
    headers["Server"] = serverName_;
    headers["Content-Type"] = "text/xml";
    Array<char> preamble;
    buildHTTPResponsePreamble(httpVersion_, 400, 
			       msg==NULL? "Bad Request":msg,
			       headers, preamble);
    sendHTTPResponse(preamble);
  }

  // Send back a response of server error 
  virtual void HTTPInternalServerError (const char* msg = NULL)
  {
    OTab headers;
    headers["Server"] = serverName_;
    headers["Content-Type"] = "text/xml";
    Array<char> preamble;
    buildHTTPResponsePreamble(httpVersion_, 500,
			      msg==NULL? "Internal Server Error":msg, 
			      headers,preamble);
    sendHTTPResponse(preamble);
  }


  virtual void HTTPNotImplemented (const char* msg = NULL)
  {
    OTab headers;
    headers["Server"] = serverName_;
    headers["Content-Type"] = "text/xml";
    Array<char> preamble;
    buildHTTPResponsePreamble(httpVersion_, 501,
			      msg==NULL ? "Not Implemented" : msg, 
			      headers, preamble);
    sendHTTPResponse(preamble);
  }

  virtual void HTTPBadGateway (const char* msg = NULL)
  {
    OTab headers;
    headers["Server"] = serverName_;
    headers["Content-Type"] = "text/xml";
    Array<char> preamble;
    buildHTTPResponsePreamble(httpVersion_, 502,
			      msg==NULL ? "Bad Gateway" : msg, 
			      headers, preamble);
    sendHTTPResponse(preamble);
  }

  virtual void HTTPServiceUnavailable (const char* msg = NULL)
  {
    OTab headers;
    headers["Server"] = serverName_;
    headers["Content-Type"] = "text/xml";
    Array<char> preamble;
    buildHTTPResponsePreamble(httpVersion_, 503,
			      msg==NULL ? "Service Unavailable" : msg, 
			      headers, preamble);
    sendHTTPResponse(preamble);
  }

  // Tools for chunked responses for HTTP 1.1
  
  // For hex lengths in chunked sequences
  static int HexToInt (const char* hex_str, int len=-1)
  {
    int return_value = 0;
    static char hexdigits[] ="0123456789abcdef";
    if (len==-1) len = strlen(hex_str); 
    char* result; 
    for (int ii=0; ii<len; ii++) {
      char c=tolower(hex_str[ii]);
      if ((result=strchr(hexdigits, c))!=0) {
	int value = result - &hexdigits[0];
	return_value = (return_value<<4) + value;
      } else {
	throw runtime_error("not a hex string");
      }
    }
    return return_value;
  }
  static int HexToInt (const string& s)
  { return HexToInt(s.data(), s.length()); }
  
  // Parse a single chunked response: returns true while there is still
  // more data to get and that means data is valid.  Otherwise,
  // it returns false and fills in the footers (meaning we're also over 
  // and are done).  Note that data is appended to the end of the
  // buffer.
  bool getSingleChunkedResponse (Array<char>& single_chunk, OTab& /* footers */)
  {
    // First, expect a line with the size of the chunked data in hex,
    // possibly followed by a semicolon and extra parameters
    string hex_len;
    if (getLine(hex_len) == 0) {
      HTTPBadRequest("saw empty line when looking for hex value");
    }
    Array<string> a = Split(hex_len, ";", 1);
    if (a.length() < 1) {
      HTTPBadRequest("no hex length on line");
    }
    string hex_value = a[0];
    int length = HexToInt(hex_value);
    // Nope, that was the last of the chunked data
    if (length==0) {
      OTab footers;
      getHeaders(footers);
      return true;
    }
    // Get the chunk off the socket, including CRLF after data
    int amt_data_to_read   = length + 2;  // +2 for CRLF
    readUntilFull(single_chunk, amt_data_to_read); 
    single_chunk.expandBy(-2);  // Drop CRLLF
    return false;  // Means there's more to read!
  }

  // Turn value into hex string
  static string Hexify (int_u4 n)
  {
    static char hexdigits[]="0123456789abcdef";
    const int LEN = sizeof(n)*4;
    char a[LEN];
    int ii=LEN-1;
    do {
      a[ii--] = hexdigits[n%16];
      n/=16;
    } while (n);
    return string(&a[ii+1], LEN-ii-1);
  }


  // Protocol for sending chunked responses:
  // HTTPValidResponse() with empty data, and transfer-encoding=chunked
  // sendSingleChunk()
  // ...
  // sendSingleChunk()
  // sendChunkedFooter()     # for final footer


  // Send out one piece of chunked data
  void sendSingleChunk (const char* single_chunk, int length=-1)
  {
    if (length==-1) length=strlen(single_chunk);
    string preamble = Hexify(length) + ";" + CRLF;
    writeExact(preamble.data(), preamble.length());
    writeExact(single_chunk, length);
    writeExact("\r\n", 2);
  }

  // Array version
  void sendSingleChunk (Array<char>& single_chunk)
  { sendSingleChunk(single_chunk.data(), single_chunk.length()); }
  // string version
  void sendSingleChunk (const string& single_chunk) 
  { sendSingleChunk(single_chunk.data(), single_chunk.length()); }

  void sendChunkedFooter (const OTab& footer) 
  {
    // Last data: send 0, footers, and blank line
    Array<char> out(1024);
    out.append('0'); out.append('\r'); out.append('\n');
    formHeaders(footer, out);
    writeExact(out.data(), out.length());
  }


  // Get headers .. or footers from network
  void getHeaders (OTab& headers /* or footers */)
  {
    // Get and form the headers
    string temp;
    for (;;) {
      if (getLine(temp) == 0) {
	// Done with preamble
	return;
      } else {
	// Still some headers to read
	const Array<string> splitting = Split(temp, ":", 1);
	if (splitting.length()<=1) {
	  HTTPBadRequest(string("Malformed header in HTTP request:"+temp).c_str());
	}
	string header_name = Lower(Strip(splitting[0])); // canonical format:lower
	headers[header_name] = Strip(splitting[1]);
      }
    }
  }

  // Form headers or footers to send (with final CRLF): Appends the data
  // into the end of the buffer (so there may be data already in the buffer)
  void formHeaders (const OTab& headers,
		    Array<char>& into_buffer, 
		    const string& /*http_version*/="HTTP/1.0")
  {
    for (It ii(headers); ii(); ) {
      // form line, copy into buffer
      const string   key = ii.key();
      const string value = ii.value();
      string line = key + ": " + value + CRLF;
      
      int len = into_buffer.length();
      into_buffer.expandBy(line.length());
      memcpy((into_buffer.data())+len, &line[0], line.length());
    }

    // Final CRLF
    into_buffer.append('\r');
    into_buffer.append('\n');
  }


  string&   httpVersion () { return httpVersion_; }
  string&   serverName () { return serverName_; }

 protected:

  string httpVersion_;
  string serverName_;

}; // HTTPTools_

PTOOLS_END_NAMESPACE 

#endif // HTTPTOOLS_H_
