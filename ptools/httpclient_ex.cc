
// Shows an example of how to use an HTTPConnection, HTTPResponse


#include "httplib.h"
#include "urllib.h"

///////////////Here is an example session that uses the GET method:

void exampleGET ()
{
  HTTPConnection conn("www.python.org");
  conn.request("GET", "/index.html");
  HTTPResponse r1 = conn.getresponse();
  cout << r1.status() << " " << r1.reason() << endl;
  // // output: 200 OK
  Array<char> data1 = r1.read();
  cout << data1 << endl;

  conn.request("GET", "/parrot.spam");
  HTTPResponse r2 = conn.getresponse();
  cout << r2.status() <<  r2.reason() << endl;
  // // output: 404 Not Found
  Array<char> data2 = r2.read();
  cout << data2.length() << endl;
  cout << data2 << endl;
  conn.close();
}

void exampleHEAD ()
{
  //Here is an example session that uses the HEAD method. Note that the HEAD method never returns any data.>
  HTTPConnection conn("www.python.org");
  conn.request("HEAD","/index.html");
  HTTPResponse res = conn.getresponse();
  cout <<  res.status() << " " <<  res.reason() << endl;
  // 200 OK
  Array<char> data = res.read();
  cout << data.length() << endl;
  // 0
}

void examplePOST ()
{
  string params = URLEncode(Tab("{'@number': 12524, '@type': 'issue', '@action': 'show'}"));
  cerr << params << endl;
  //string params = "%40action=show&%40number=12524&%40type=issue";
  OTab headers = "o{'Content-type': 'application/x-www-form-urlencoded', "
                 "  'Accept': 'text/plain'}";
  HTTPConnection conn("bugs.python.org");
  conn.request("POST", "", params, headers);
  HTTPResponse response = conn.getresponse();
  cout << response.status() << " " << response.reason() << endl;
  // 302 Found
  Array<char> data = response.read();
  //'Redirecting to <a href="http://bugs.python.org/issue12524">http://bugs.python.org/issue12524</a>'
  cerr << data << endl;
  cerr << " is closed? " << response.isclosed() << endl;
  Array<char> data2 = response.read(); // chunked response
  cerr << "next chunked response" << endl;
  conn.close();
}


void exampleCHUNKEDGET ()
{
  // This will work, but it's better to make a new HTTPConnection every time
  HTTPConnection conn("localhost", 8888);  
  for (int count=0; count<3; count++) {
    // HTTPConnection conn("localhost", 8888);  
    cout << "Sending request" << endl;
    conn.request("GET", "/index.html");
    HTTPResponse r1 = conn.getresponse();
    cout << r1.status() << " " << r1.reason() << endl;
    // // output: 200 OK
    Array<char> data1 = r1.read();
    cout << "data from chunk 1" << data1 << endl;
    
    Array<char> data2 = r1.read();
    cout << "data from chunk 2" << data2 << endl;
    r1.close();
  }

}

// Demonstrate sending a dictionary as a "request" and waiting for a response
void exampleBIGPOST ()
{
  // Imagine a table with lots of keys (that won't fir into the URL
  Tab table_request = 
    "{ 'file1': '/home/midas/file1', 'file2':'/home/midas/file2',"
    "  'up': 1, 'down': 17 }";
  string data = string(Val(table_request));

  // Header
  OTab headers = "o{'Content-type': 'text', "
                 "  'Accept': 'text/plain'}";
  
  HTTPConnection conn("localhost", 8888);
  conn.request("POST",  // Have to use POST because want to send data
	       "/",     // url less useful here(maybe can disambiguate server?)
	       data,
	       headers);
	       
  HTTPResponse resp = conn.getresponse();
  cout << resp.status() << " " << resp.reason() << endl;
  // // output: 200 OK
  Array<char> data1 = resp.read();
  cout << "data from chunk 1" << data1 << endl;
  
  Array<char> data2 = resp.read();
  cout << "data from chunk 2" << data2 << endl;
  resp.close();
}


int main ()
{
  exampleGET();
  exampleHEAD();
  examplePOST();
  exampleCHUNKEDGET();
  exampleBIGPOST ();
}


