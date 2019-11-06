#ifndef SOCKETERROR_H_

// These are more specific errors the MidasTalker/Server, etc. can throw
// when there are problems opening, reading, or writing a socket.
// They all inherit from runtime_error to preserve backwards compatibility:
// all the old code will work, this simply allows you to be more
// specific on which error you deal with.

// Note that these names come directly from the Python Socket module
// to try to preserve readability: the socket module usually throws
// socket.error, where as C++ usually throws runtime_error.

#include "ocport.h"

PTOOLS_BEGIN_NAMESPACE

// GetHostByName errors:  usually indicates a DNS error
class herror : public runtime_error {
 public:
  herror (const string& name) : runtime_error(name) { }
}; // Herrr

// GetAddr* errors: usually indicate an unknown name or service
class gaierror : public runtime_error {
 public:
  gaierror (const string& name) : runtime_error(name) { }
}; // gaierror

// Timeout error:  the open failed with a timeout
class timeout :  public runtime_error {
 public:
  timeout (const string& name) : runtime_error(name) { }
}; // timeout

PTOOLS_END_NAMESPACE

#define SOCKETERROR_H_
#endif // SOCKETERROR_H_
