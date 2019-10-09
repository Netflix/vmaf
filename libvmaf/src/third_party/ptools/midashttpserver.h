#ifndef MIDASHTTPSERVER_H_
#define MIDASHTTPSERVER_H_

#include "simplehttpserver.h"
#include "midastalker.h"

// Implement a server converting HTTP requests into 
// MidasTalker requests:  it basically serves as an adapter
// between an HTTP Server (which speaks XML) and MidasTalker
// (which speaks Python dictionaries).  


PTOOLS_BEGIN_NAMESPACE

// /////////////////////////////////// Class MidasHTTPWorker

// The client handles HTTP requests and converts them 
// into MidasSpeak

class MidasHTTPWorker : public ThreadedHTTPWorker {
 public:
  MidasHTTPWorker (const string& name, ThreadedServer& server, 
		   bool reopening,
		   const string& host, int port, Serialization_e ser, 
		   SocketDuplex_e socketness,   
		   ArrayDisposition_e arrdis) :
    ThreadedHTTPWorker(name, server),
    reopening_(reopening),
    mt_(host, port, ser, socketness, arrdis)
  { }

 protected:
  
  // ///// Data Members

  // Do we close/reopen the MidasTalker on every request?
  bool reopening_;

  // The client talking to the MidasServer
  MidasTalker mt_; 

  // ///// Methods

  // Convert the XML content to a dict, if you can
  Val convertContentToDict_ (const Array<char>& content)
  {
    Val request;
    int xml_options = 
      XML_STRICT_HDR |     // make sure we have XML header
      XML_LOAD_DROP_TOP_LEVEL | // extra top-level container, nuke it
      XML_LOAD_EVAL_CONTENT     // Try to turn some strings into real values
      ;
    ArrayDisposition_e arr_dis = AS_NUMERIC;
    bool valid_conversion = true;
    try {
      XMLLoader loader(content, xml_options, arr_dis);
      loader.expectXML(request);
    } catch (const runtime_error& re) {
      valid_conversion = false;
    }
    if (!valid_conversion) {
      httptools_.HTTPBadRequest("Can't convert XML to dict: malformed XML");
    }
    return request;
  }

  // Try to open up the MidasTalker, in case not immediately available
  void openup_ (Val time_out=5.0)
  {
    if (mt_.isOpen()) return;
    string err;
    try {
      mt_.open(time_out);
      //cout << "Opened connection to host:" 
      //	   << mt.host() << " port:" << mt.port() << endl;
      return;
    } catch (const timeout& e) {
      err = "Open TIMEOUT: is your server slow, suspended, or in a weird state? "+string(e.what());
    } catch (const gaierror& e) {
      err = "Had trouble with name: did you mistype the hostname? " + string(e.what());
    } catch (const herror& e) {
      err = "Host error? Check your DNS? " + string(e.what());
    } catch (const runtime_error& e) {
      err = "Problem: "+string(e.what());
    }
    httptools_.HTTPBadGateway(err.c_str());
  }


  // Contact the server via a MidasTalker and try to get a response back.
  // If we can't get to the Server, we send back a BadGateway (for the most
  // part) to the HTTP Client.
  Val contactMidasServer_ (const Val& request)
  {
    //cerr << endl << "Trying to open up the MidasTalker on " << mt_.host() << " " << mt_.port() << endl;
    openup_();
    //cerr << "...opened succesfully" << endl;

    // Try to send request off to some MidasServer. If request fails, send back
    // appropriate badnness to HTTP Client.
    bool sending_data = true;
    try {
      mt_.send(request);
    } catch (const runtime_error& re) {
      sending_data = false;
    }
    if (sending_data == false) {
      httptools_.HTTPBadGateway("Trouble sending data via MidasTalker");
    }
    //cerr << "...sent data sucessfully" << endl;

    // Send worked okay, can we get a response?
    Val response;
    try {
      response = mt_.recv(20);
      if (response==None) {
	httptools_.HTTPInternalServerError("Too much time elapsed, request timeout?");
      }
    } catch (const runtime_error& re) {
      httptools_.HTTPBadGateway("Trouble getting data from MidasTalker");
    }
    //cerr << "...got a response from MidasTalker:" << response << endl;
    return response;
  }

  // Take the response from the MidasServer, convert it to
  // XML, and send that response back over the socket.
  void convertToXMLAndRespond_ (const Val& response)
  {
    // Convert to XML for response back
    bool conversion_from_dict_to_xml_valid = true;
    string xml;
    try {
      xml = ConvertToXML(response);
    } catch (const runtime_error& re) {
      conversion_from_dict_to_xml_valid = false;
    }
    //cerr << "...XML response" << xml << endl;
    if (conversion_from_dict_to_xml_valid) {
      httptools_.HTTPValidResponse(xml);
    } else { 
      httptools_.HTTPInternalServerError();
    }
  }

  // Extend dispatchWork_ so we may or may not close the 
  // MidasTalkers every time (by default, probably want 
  // to keep them open so we can just talk directly to the server
  // and not keep re-opening).
  virtual void postDispatchWork_ ()
  { 
    if (reopening_) {
      //cerr << "... closing MidasTalker" << endl;
      mt_.close(); 
    }
  }

  // Handle a POST request
  virtual void handlePOST_ (const string& url, const string& http_version, 
			    const OTab& headers)
  {
    //headers.prettyPrint(cerr);

    // Has to be XML content type or we don't know what to do with it
    if (!headers.contains("content-type")) {
      httptools_.HTTPBadRequest("No Content-Type field");
    }
    if (!((headers("content-type") == "text/html") ||
	  (headers("content-type") == "text/xml") )) {
      string err="Unknown Content-Type:" + Stringize(headers("content-type"));
      httptools_.HTTPBadRequest(err.c_str());
    } 
    // Assertion: content type is good: just read data from socket

    // Read to content-length is specified, or EOF if not specified
    int length = -1;
    if (headers.contains("content-length")) {
      length = headers("content-length");
    }
    Array<char> xml_content(length>0 ? length : 1024);
    httptools_.readUntilFull(xml_content, length);
    // debugArray_(xml_content, "handlePost");

    // At this point, we are done reading from socket: we keep 
    // it open so we can return the response.
   
    // xml->dict, talk to Midas, dict->xml, and send back XML response
    Val dict_request = convertContentToDict_(xml_content);
    Val dict_response = contactMidasServer_(dict_request);
    convertToXMLAndRespond_(dict_response);
  }


}; // MidasThreadedWorker


// /////////////////////////////////////////// MidasHTTPServer

// Simple wrapper for an threaded, http Server that understands
// the midastalker

class MidasHTTPServer : public SimpleHTTPServer {

 public:

  // Create an HTTP server with a MidasTalker talking to a client
  MidasHTTPServer (const string& http_server_name, int http_server_port,

		   const string& midas_server, int midas_port,
		   bool reopen_midas_talker_after_every_http = false,
		   Serialization_e ser=SERIALIZE_P2,
		   SocketDuplex_e dual_socket = DUAL_SOCKET, 
		   ArrayDisposition_e arrdisp = AS_LIST) :
    SimpleHTTPServer(http_server_name, http_server_port),

    reopening_(reopen_midas_talker_after_every_http),
    midasServerName_(midas_server),
    midasPort_(midas_port),
    midasDualSocket_(dual_socket),
    midasSer_(ser),
    midasArrDis_(arrdisp)
  { }

 protected:

  // If true, close the MidasTalker after HTTP request done, then
  // reopen on next HTTP request.
  bool reopening_; 

  // The midas Talker parameters: but each client has to have it's
  // own MidasTalker
  string             midasServerName_;
  int                midasPort_;
  SocketDuplex_e     midasDualSocket_;
  Serialization_e    midasSer_;
  ArrayDisposition_e midasArrDis_;


  // Each HTTP connection gets its own thread, and its own MidasTalker
  // to talk to the server.  
  virtual ThreadedServerWorker* createThreadedServerWorker_ ()
  { return new MidasHTTPWorker("midashttpworker", *this, reopening_, 
			       midasServerName_, midasPort_, midasSer_,
			       midasDualSocket_, midasArrDis_); 
  }


}; // MidasHTTPServer


PTOOLS_END_NAMESPACE

#endif // MIDASHTTPSERVER_H_
