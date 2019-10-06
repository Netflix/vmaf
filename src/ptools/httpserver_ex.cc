
// Simple echo HTTP Server:  
// GET, returns just the same path that GET sent, even if not there
// POST returns the same data from the post data

#include "simplehttpserver.h"

// Each client has its own thread to handle the file descriptor

class EchoHTTPWorker : public ThreadedHTTPWorker {
  
  public:
  
  EchoHTTPWorker (const string& name, ThreadedServer& server) :
    ThreadedHTTPWorker(name, server)
  { }


  // User-hook for dealing with a GET: All initial-line, headers data
  // has been processed, but the rest of the message needs to be processed
  // (i.e,) and the response generated.
  virtual void handleGET_ (const string& url, const string& http_version, 
			   const OTab& in_header)
  {
    cerr << "GET " << "url:" << url << " http_version:" << http_version << endl;
    cerr << "headers:" << in_header << endl;

    // Echo the data back TWICE

    OTab out_header;
    string type = in_header.get("type", "HTTP/1.0");

    static int counter = 1;
    if (http_version=="HTTP/1.1") { // Then we support chunked encoding

      // Send out just out header, with no content.  Leave that to chunked
      out_header["Transfer-Encoding"] = "chunked";
      httptools_.HTTPValidResponse("", type, out_header);  // send header

      // now, send data, twice
      OTab footers;
      httptools_.sendSingleChunk(url);
      httptools_.sendSingleChunk(url+Stringize(counter));
      counter+=1;

      // Need to do this to finish request
      httptools_.sendChunkedFooter(footers);

    } else { // unknown, go back to 1.0, just make data twice as big
      string data = url + url;
      httptools_.HTTPValidResponse(data, type, out_header); // Just send back url as echo
    }

  }

  // User-hook for dealing with a POST: All initial-line, headers data
  // has been processed, but the rest of the message needs to be processed
  // (i.e,) and the response generated.
  virtual void handlePOST_ (const string& url, const string& http_version, 
			    const OTab& headers) 
  {
    cerr << "POST " << "url:" << url << " http_version:" << http_version<< endl;
    cerr << "headers:" << headers << endl;

    Array<char> result;

    // Chunked?
    if (headers.contains("transfer-encoding") && 
	headers("transfer-encoding")=="chunked") {
      if (http_version=="HTTP/1.0") {
	httptools_.HTTPBadRequest();
      }
      OTab footers;
      while (httptools_.getSingleChunkedResponse(result, footers)) {
	// Keep appending data into result
	cerr << "chunked data ... length of data in buffer" << result.length();
      }
      cerr << "FINAL chunked data ... length of data in buffer" << 
	result.length();
      cerr << "footers: " << footers << endl;
    } 
    
    // Not chunked
    else {
	int len = -1;    
	if (headers.contains("content-length")) {
	  // Just get in one chunk
	  len = headers("content-length");
	}
	httptools_.readUntilFull(result, len);
      }
    
    // Made it here, got result: 
    httptools_.HTTPValidResponse(string(result.data(), result.length()));
  } 
  

}; // EchoHTTPWorker 


// ///////////////////////////////////// EchoHTTPServer

class EchoHTTPServer : public SimpleHTTPServer {

  public:
  
  EchoHTTPServer (const string& host, int port) :
    SimpleHTTPServer(host, port)
  { }

  protected:

  // When a new connection comes in, we have to make sure
  // we create the proper type of client for this server.
  virtual ThreadedServerWorker* createThreadedServerWorker_ ()
  { return new EchoHTTPWorker("EchoHTTPServer", *this); }


}; // EchoHTTPServer


int main (int argc, char* argv[])
{
  if (argc!=3) {
    cerr << "Usage: " << argv[0] << " host port" << endl;
    exit(1);
  }

  string host = argv[1];
  int port    = atoi(argv[2]);
  EchoHTTPServer server(host, port);
  server.open();
  server.waitForMainLoopToFinish();
}
