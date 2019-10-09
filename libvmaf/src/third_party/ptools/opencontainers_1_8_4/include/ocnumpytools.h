#ifndef OCNUMPYTOOLS_H_

// A few helper functions/defines for help for dealing with Numeric
// (Python Numeric)
#include "ocport.h"


OC_BEGIN_NAMESPACE

// Convert from a Val tag to a Python Numeric Tab
inline const char* OCTagToNumPy (char tag, bool supports_cx_int=false)
{
  switch (tag) {
  case 's': return "int8"; break;
  case 'S': return "uint8"; break; 
  case 'i': return "int16"; break; 
  case 'I': return "uint16"; break; 
  case 'l': return "int32"; break; 
  case 'L': return "uint32"; break; 
  case 'x': return "int64"; break; 
  case 'X': return "uint64"; break; 
  case 'b': return "bool"; break; 
  case 'f': return "float32"; break; 
  case 'd': return "float64"; break; 
  case 'F': return "complex64"; break; 
  case 'D': return "complex128"; break; 
  default: 
    if (supports_cx_int) {
      switch (tag) {
      case 'c': return "complexint8"; break;
      case 'C': return "complexuint8"; break;
      case 'e': return "complexint16"; break;
      case 'E': return "complexuint16"; break;
      case 'g': return "complexint32"; break;
      case 'G': return "complexuint32"; break;
      case 'h': return "complexint64"; break;
      case 'H': return "complexuint64"; break;
      default: break;
      }
    }
    throw runtime_error("No corresponding NumPy type for Val type");
  }
  return 0;
}


// Convert from Numeric tag to OC Tag
inline char NumPyStringToOC (const char* tag, bool supports_cx_int=false)
{

  char ret = '*';
  if (tag==NULL || tag[0]=='\0') {
    throw runtime_error("No corresponding OC tag for NumPy tag");
  }
  typedef AVLHashT<string, char, 16> TABLE;
  static TABLE* lookup = 0;
  if (lookup == 0) { // Eh, any thread that reaches here will do a commit
    TABLE& temp = *new TABLE();
    temp["bool"]  = 'b';
    temp["int8"]  = 's';      
    temp["uint8"] = 'S';
    temp["int16"] = 'i';      
    temp["uint16"] ='I';
    temp["int32"] = 'l';     
    temp["uint32"] ='L';
    temp["int64"]  ='x';
    temp["uint64"] ='X';
    temp["float32"]='f';
    temp["float64"]='d';
    temp["complex64"]='F';
    temp["complex128"]='D';
    temp["complexint8"] = 'c';
    temp["complexuint8"] = 'C';
    temp["complexint16"] = 'e';
    temp["complexuint16"] = 'E';
    temp["complexint32"] = 'g';
    temp["complexuint32"] = 'G';
    temp["complexint64"] = 'h';
    temp["complexuint64"] = 'H';
    lookup = &temp;
  }

  /// AVLHashTIterator<string, char, 16> it(*lookup);
  /// while (it()) {
  ///  cout << it.key() << ":" << it.value() << endl;
  /// }
  // If found, return full char, otherwise -1 to inidcate it failed
  string tagger = tag;
  if (lookup->findValue(tagger, ret)) {
    if (supports_cx_int) {
      return int_1(ret);
    } 
    // Doesn't support cx int
    if (tagger.find("complex") != string::npos && 
	tagger.find("int")!=string::npos) {
      return -1;
    } else {
      return int_1(ret);
    }

  } else {
    return -1;
  }
}

OC_END_NAMESPACE


#define OCNUMPYTOOLS_H_
#endif // OCNUMPYTOOLS_H_
