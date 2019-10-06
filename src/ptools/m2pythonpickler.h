#ifndef M2PYTHONPICKLER_H_

#include <stdio.h>
#include <ctype.h>
#include "m2pythontools.h"

// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_BEGIN_NAMESPACE
#endif

// Classes for helping with Pickling and Depickling.  The way Vectors
// changes serialization depending on if the Package on the other side
// supports the Python package "Numeric" which has built-in arrrays.
// If Numeric is not used, Vectors are turned into Python Lists.

#include "m2pmstack.h" 


// ///// PythonDepickerA 
// Class to do all the unpickling, but I/O mechanism is left to
// subclass.
template <class OBJ>
class PythonDepicklerA {

 public:
  
  PythonDepicklerA (bool with_numeric_package, bool use_proxies=true);
  virtual ~PythonDepicklerA ();
  
  // Load a pickled OBJ from the given 
  OBJ load ();

  // Inspector/Mutator for the compatibility mode: by default, Tups
  // and OTab (introduced in PicklingTools 1.2.0) aren't available, so
  // we have to "simulate" them with Arr and Tab (respectively).
  bool compatibility () const { return compat_; }
  void compatibility (bool disallow_tup_and_otab) 
  { compat_=disallow_tup_and_otab; }
  
 protected:
  
  // Get a single character:  return EOF if hits EOF
  virtual int getChar_ () = 0;
  
  // Returns chars on input until we hit a newline, at which point it
  // replaces the newline with a \0 (giving a c-style string).  We
  // consume all the characters upto/including the newline.  (Most
  // pickled entries end with a newline).
  virtual char* getUntilNewLine_ (int& len) = 0;
  
  // Get a string
  virtual char* getString_ (int& len) = 0;
  

  // Pop the stop of the PM Context stack and return it.  When we pop,
  // if this value had been memoized, then we need to copy the "full
  // copy" back to the memoize area.
  OBJ pop_ () { return pmstack_.pop(); }

  // Look at the top of the PM Stack: "peek" into the top (or a few
  // down) from the top of the stack.  0 means the top, -1 means 1
  // from the top, etc.
  OBJ& top_ (int offset=0) { return pmstack_.peek(offset); }

  // Push an item on top of the PM Context stack.
  void push_ (const OBJ& v) { pmstack_.push(v); }
  
  // Dispatches foe the different pickle tokens
  void ploadNONE_ ();
  void ploadGLOBAL_ ();
  void ploadINT_ ();
  void ploadLONG_ ();
  void ploadFLOAT_ ();
  void ploadSTRING_ ();
  void ploadMARK_ ();
  void pPUT_ ();
  void pGET_ ();
  void ploadLIST_ ();
  void ploadTUPLE_ ();
  void ploadREDUCE_ ();
  void pAPPEND_ ();
  void pSETITEM_ ();
  void pEMPTY_DICT_ ();
  void pDICT_ ();

  void reduceComplex_ (const OBJ& tuple);
  void reduceArrays_ (const OBJ& tuple);
  void reduceOrderedDict_ (const OBJ& tuple);

  // ///// Data Members 
  
  // The PM stack, as described by pickletools.  Every item on the
  // stack has "possibly" been memoized, which means when it is
  // popped, we need to copy the value out into the memo stack: that's
  // why we record the memo number along with the value (so we know
  // what its memo is).
  PMStack_<OBJ> pmstack_;

  Array<int> mark_;  // remember where we start marks ... 
  bool withNumericPackage_; // Do we expect the other end to support Numeric?

  // So, previous incarnations ALWAYS returned full-copies.  With the
  // advent of Proxy support in OpenContainers, it is possible to
  // return a table which is a PROXY to a table, not a Tabel itself.
  // We at least give the user the option for both.  By default, it
  // returns the "new" way, with proxies.
  bool wantToProxyize_;

  // Allow PTOOLS to be backward compatible with previous versions
  // already installed: if true, all OTabs and Tups and serialized as
  // Tabs and Arr (resp.).  BY DEFAULT, we probably need to backward
  // compatible: the user must EXPLICITLY request OTab and Tup (we
  // don't want to break any old installs).
  // However, this may vary from install to install, so you can
  // set the "default" as a macro: -DOC_SERIALIZE_COMPAT=true
  // or -DOC_SERIALIZE_COMPAT=false.
  //
  // Of course, the version string from the server helps us...
  bool compat_;
}; // PythonDepicklerA 


// Unpickle data from a file
template <class OBJ>
class PythonDepickler : public PythonDepicklerA<OBJ> {
 public:

  enum { PMC_BUFF_EXPAND = 2048 };
  
  PythonDepickler (const string& name, bool with_numeric_package);
  virtual ~PythonDepickler ();

 protected:

  // Satsify interface
  virtual int getChar_ ();
  virtual char* getUntilNewLine_ (int& len);
  virtual char* getString_ (int& len);

  // ///// Data Members
  FILE* fp_;
  Array<char> buffer_; // Buffer for fgets

}; // PythonDepickler


// Depickle from a inplace memory buffer
template <class OBJ>
class PythonBufferDepickler : public PythonDepicklerA<OBJ> {
 public:

  // The buffer is NOT adopted, but referenced.  Note: The buffer MAY
  // BE modified while using it!  If you want to keep the buffer, make
  // sure you copy it out first.
  PythonBufferDepickler (int_u4 buff_len, char* buff, 
			 bool with_numeric_package);
  virtual ~PythonBufferDepickler ();

 protected:

  // Satsify interface
  virtual int getChar_ ();
  virtual char* getUntilNewLine_ (int& len);
  virtual char* getString_ (int& len);

  // ///// Data Members
  int_u4 buffLen_; // The length of the referenced buffer
  char*  buff_;    // The buffer:  referenced, NOT adopted
  int_u4 start_;   // Start of the buffer
  
}; // PythonBufferDepickler



// Class used for pickling data (protocol 0)
template <class OBJ>
class PythonPicklerA {
public:

  PythonPicklerA (bool with_numeric_package);
  PythonPicklerA (ArrayDisposition_e arr_disp);

  // Your derived destructor MUST CALL putChar_(PY_STOP) so that it
  // puts the final stop mark in!
  virtual ~PythonPicklerA ();

  void dumpBool (const OBJ& val);
  void dumpVector (const OBJ& val, int int_handle=-1);
  void dumpString (const string& val, bool translate_escapes=true);
  void dumpTable (const OBJ& val, int int_handle=-1);
  void dumpOrderedDict (const OBJ& val, int int_handle=-1);
  void dumpTuple (const OBJ& val, int int_handle=-1);
  void dumpProxy (const OBJ& val);
  void dumpList (const OBJ& val, int int_handle=-1);
  void dumpNumber (const OBJ& val);

  // General Serializer
  void dump (const OBJ& val);

  // Mutator/Inspector pair for checking or setting the warning state
  bool warn () const { return warn_; }
  void warn (bool set_to) { warn_ = set_to; }

  // Inspector/Mutator for the compatibility mode: by default, Tups
  // and OTab (introduced in PicklingTools 1.2.0) aren't available, so
  // we have to "simulate" them with Arr and Tab (respectively).
  bool compatibility () const { return compat_; }
  void compatibility (bool disallow_tup_and_otab) 
  { compat_=disallow_tup_and_otab; }


protected:

  // Put the appropriate stuff onto the output
  virtual void putChar_ (char c) = 0;
  virtual void putStr_ (const char* s) = 0;
  virtual void putStr_ (const string& str) = 0;

  // Dump depending on how we interact with Numeric
  void dumpVectorAsList_ (const OBJ& v, int int_handle=-1);
  void dumpNumericArray_ (const OBJ& v, int int_handle=-1);
  void dumpNumPyArray_ (const OBJ& v, int int_handle=-1);

  // Helper function: when doing get or put, need to be able to output number
  void placeHandle_(int int_handle, char code);


  // ///// Data Members

  // Are we considering the Numeric package?
  ArrayDisposition_e arrDisp_;

  // Map between ptr_handles (inside the proxy) and int_handles (the
  // small int that goes into the serialization).
  HashTableT<void*, int, 8> handles_;

  // What would be the "next" handle if we had to get one.
  int currentIntHandle_;

  // Indicate whether or not we are being verbose with warnings
  bool warn_;

  // Allow PTOOLS to be backward compatible with previous versions
  // already installed: if true, all OTabs and Tups and serialized as
  // Tabs and Arr (resp.).  BY DEFAULT, we probably need to backward
  // compatible: the user must EXPLICITLY request OTab and Tup (we
  // don't want to break any old installs).
  // However, this may vary from install to install, so you can
  // set the "default" as a macro: -DOC_SERIALIZE_BACK_COMPAT=true
  // or -DOC_SERIALIZE_BACK_COMPAT=false.
  //
  // Of course, the version string from the server helps us...
  bool compat_;

}; // PythonPickler


// Class used for pickling data to a FILE
template <class OBJ>
class PythonPickler : public PythonPicklerA<OBJ> {
 public:

  PythonPickler (const string& name, bool with_numeric_package);
  PythonPickler (const string& name, ArrayDisposition_e arr_disp);
  virtual ~PythonPickler ();

protected:

  // Put the appropriate stuff onto the output
  virtual void putChar_ (char c);
  virtual void putStr_ (const char* s);
  virtual void putStr_ (const string& s);

  // ///// Data Members
  FILE* fp_;

}; // PythonPickler


// Class used for pickling data to a memory buffer
template <class OBJ>
class PythonBufferPickler : public PythonPicklerA<OBJ> {
 public:
  
  PythonBufferPickler (Array<char>& buffer, bool with_numeric_package);
  PythonBufferPickler (Array<char>& buffer, ArrayDisposition_e arr_disp);
  virtual ~PythonBufferPickler ();

  protected:

  // Put the appropriate stuff onto the output
  virtual void putChar_ (char c);
  virtual void putStr_ (const char *s);
  virtual void putStr_ (const string& s);

  // ///// Data Members
  Array<char>& buffer_;

}; // PythonBufferPickler

// Distinguish between M2k and OpenContainers usage
#if defined(OC_USE_OC) || defined(OC_NEW_STYLE_INCLUDES)
PTOOLS_END_NAMESPACE
#endif

#define M2PYTHONPICKLER_H_

#ifdef MIDAS_COMPILER_TEMPLATES
#  include "m2pythonpickler.cc"
#endif

#endif // M2PYTHONPICKLER_H_
