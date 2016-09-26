#ifndef CHOOSESER_H_

// Convenience routines for reading/writing Vals to/from Arrays and
// files using the different serializations supported by PTOOLS.
// Their usage is very simple.  These routines are used by the
// MidasServer family to do serialization.
//
// ============For example, to write a pickled file that Python can read:
//
// // C++ side: Write a Value
//  Val v = .. something ..
//  DumpValToFile(v, "state.p0", SERIALIZE_P0);
//
// # Python side: read the same value
//  import cPickle
//  result = cPickle.load( file('state.p0') ) # load figures out the protocol
//
//
// ==========Another example: have C++ read a file that Python created
// 
// # Python side: write a file
//  v = ... something ...
//  import cPickle
//  cPickle.dump( v, file('state.p2'), 2) # Use Pickling Protocol 2
// 
// // C++ side: read the same file
//  Val result;
//  LoadValFromFile("state.p2", result, SERIALIZE_P2); 
//  /// .. and we have the same value from Python!
//
//
// ==========Another example: have C++ create an Array 
//  // C++: Creating an array of pickled data I can use
//  Val v = .. something ...
//  Array<char> buff;
//  LoadValFromArray(v, buff, SERIALIZE_OC);  // Faster of the protocols
//  cout << "serialized data:" << buff.data() << " length:" << buff.length();

#define MIDAS_COMPILER_TEMPLATES
#include "valpython.h"

#include "valprotocol2.h"
#include "m2ser.h"
#include "ocserialize.h"
#include "pickleloader.h"
#include "ocvalreader.h"
#include "xmltools.h"
#include "opalutils.h"
#include "serialization.h"

PTOOLS_BEGIN_NAMESPACE

////////////////////////////////// ArrayDisposition_e ////////////////
// Different kinds of POD (Plain Old Data: int_1, int_2, real_4, etc.) arrays: 
// there are essentially 3 different types of POD arrays that might be moving 
// around: 
//
// (1) a = [1,2,3]  
//     Python styles lists (which are inefficient for storing 
//     homogeneous data)
//
// (2) import array; a = array.array('i',[1,2,3])
//     the arrays from the Python module array 
//
// (3) import Numeric: a = Numeric.array([1,2,3], 'i')
//     the Numeric arrays which are built in to XMPY,
//     but most standard Pythons do not have it installed.
//
// In C++, POD arrays are handled as Array<T>, thus (2) & (3)
// are handled with the same:  (1) is handled as the C++ Arr.  
// These distinctions are more important if you are in Python, or talking 
// to a Python system, as you have to specify how a C++ Array
// converts to a Python POD array.
//
// These 4 distinctions are made because Python doesn't deal
// well with POD (plain old data) arrays well:  This option allows
// you to choose what you want when dealing with POD when you
// convert between systems.  Consider:
// (1) Python style lists work, but are horribly inefficient for
//     large arrays of just plain numbers, both from a storage
//     perspective or accessing.  Also, you "lose" the fact 
//     that this is true POD array if you go back to C++.
// (2) Numeric is old, but handles all the different types well,
//     including complex (although Numeric doesn't deal with int_8s!).
//     It is also NOT a default-installed package: you may have to find
//     the proper RPM for this to work.
// (3) Python array from the array module are default but have issues:
//     (a) can't do complex data 
//     (b) may or may not support int_8
//     (c) pickling changes at 2.3.4 and 2.6, so if you are
//         pickling with protocol 2, you may have issues.
// (4) Python array from the numpy module: major drawback is that it's
//     not built-in
//     
// None of these solutions is perfect, but going to NumPy is
// probably your best choice for the future, although Numeric
// may be your choice if you are stuck on an older Python.
/////////////////////////////////////////////////////////////////////



// This is a convenience function for dumping a Val as an of several
// serializations.  Dump the given Val to the given array of chars: it
// serializes via the given serialization: ArrayDisposition only makes
// sense for SERIALIZE_P0/P2 (see above).
//
// Expert Parameter: perform_conversion_of_OTabTupBigInt_to_TabArrStr
// Some legacy systems may not support OTab, Tup or int_un/int_n, in
// which case we have the OPTION to convert those to something the
// legacy systems can understand: Tab, Arr and Str.  
inline void DumpValToArray (const Val& given, Array<char>& dump,
			    Serialization_e ser=SERIALIZE_P0,
			    ArrayDisposition_e arrdisp=AS_LIST,
			    bool perform_conversion_of_OTabTupBigInt_to_TabArrStr = false,
			    MachineRep_e endian=MachineRep_EEEI)
{
  static int once = 0;
  if (arrdisp==AS_PYTHON_ARRAY && once==0) {
    once = 1;
    cerr << "As of PicklingTools 1.2.0, the array disposition\n" 
	 << " AS_PYTHON_ARRAY is deprecated.  You will see this warning\n"
	 << " only once...." << endl;
    cerr << "\n"
"  The whole purpose of adding the ArrayDisposition AS_PYTHON_ARRAY\n"
"  was because, in Python 2.6, it was a binary dump: dumping arrays\n"
"  of POD (Plain Old Data, like real_4, int_8, complex_16) was blindingly\n"
"  fast (as it was basically a memcpy): This was to help Python users who\n"
"  did not necessarily have the Numeric module installed.  As of Python 2.7,\n"
"  however, the Pickling of Arrays has changed: it turns each element into a\n"
"  Python number and INDIVIDUALLY pickles each element(much like the AS_LIST\n"
"  option).  The new Pickleloader DOES DETECT AND WORK with both 2.6\n"
"  and 2.7 pickle streams, but we currently only dump as 2.6: this change\n"
"  in Python 2.7 (and also Python 3.x) defeats the whole purpose of\n"
"  supporting array .. we wanted a binary protocol for dumping large amounts\n"
"  of binary data!  As of this release, we deprecate the AS_PYTHON_ARRAY\n"
"  serialization, but will keep it around for a number of releases. \n" 
	 << endl;
  }
  bool conv = perform_conversion_of_OTabTupBigInt_to_TabArrStr;
  switch (ser) {
  case SERIALIZE_P0: case SERIALIZE_P0_OLDIMPL: {
    if (arrdisp==AS_PYTHON_ARRAY) {
      throw runtime_error("SERIALIZE_P0 doesn't support AS_PYTHON_ARRAY");
    }
    PythonBufferPickler<Val> pd(dump, arrdisp);
    pd.compatibility(conv);  // Makes it so we don't have to convert at top,
                             // only as we go.
    pd.dump(given);
    break;
  }
  case SERIALIZE_P2: case SERIALIZE_P2_OLDIMPL: case SERIALIZE_P2_OLD: {
    PicklingIssues_e issues = 
      ((ser==SERIALIZE_P2_OLD) ? AS_PYTHON_2_2 : ABOVE_PYTHON_2_2);
    if (conv) {
      issues = PicklingIssues_e(int(issues)|int(CONVERT_OTAB_TUP_ARR__TO__TAB_ARR_STR));
    }
    
    size_t bytes = P2TopLevelBytesToDumpVal(given, arrdisp, issues);
    dump.expandTo(bytes); // overestimate
    char* mem = dump.data();
    char* rest = P2TopLevelDumpVal(given, mem, arrdisp, issues);
    size_t len = rest-mem;   // exact
    dump.expandTo(len);
    break;
  }
  case SERIALIZE_M2K: {
    // Because there are no Tup, OTab or BigInt in M2k, those
    // data structures HAVE to be converted to OpaList, OpalDict and Str
    // by default ... no extra option needed for it.

    // Unfortunately, there are about 3-4 ways to serialize data
    // in M2k: the way the OpalPythonDaemon and UDP daemon do it is by
    // forcing it to be a table 
    char *mem;
    size_t bytes = 0;
    if (given.tag=='n' && given.subtype == 'Z') {
      bytes = OpalDumpArr(given, mem, endian); // Tab and Arr closely related
    } else if (given.tag=='t') {
      bytes = OpalDumpTab(given, mem, endian); 
    } else if (given.tag=='o') {
      bytes = OpalDumpOTab(given, mem, endian); 
    } else if (given.tag=='u') {
      bytes = OpalDumpTup(given, mem, endian); 
    } else {
      // Most sockets expect a table, and they can cause seg faults
      // if they don't get an opalTable on input, so force one
      Tab special;
      special["__SPECIAL__"] = given;
      bytes = OpalDumpTab(special, mem, endian);
    }
    dump.adoptMemory(mem, bytes, 2); //adopt as new T[], so delete w/ delete []
    dump.expandTo(bytes);
    break;
  }
  case SERIALIZE_OC: {
    // The OC, if we request conversion, does it in place so its faster
    size_t bytes = BytesToSerialize(given, conv);
    dump.expandTo(bytes); // overestimate
    char* mem = dump.data();
    char* rest = Serialize(given, mem, conv);
    size_t len = rest-mem;
    dump.expandTo(len);   // exact
    break;
  }
    //case SERIALIZE_TEXT: {
  case SERIALIZE_OPALTEXT:
  case SERIALIZE_PYTHONTEXT: {
    Val* vp = const_cast<Val*>(&given);
    try {
      if (conv) {
	vp = new Val(given);
	ConvertAllOTabTupBigIntToTabArrStr(*vp);
      }
      string s;
      if (ser==SERIALIZE_OPALTEXT) {
	ostringstream os;
	prettyPrintOpal(*vp, os, 0, 0, false); // No pretty print!
	s = os.str();
      } else {
	s = Stringize(*vp);
      }
      dump.expandTo(s.length());
      char* mem = dump.data();
      memcpy(mem, s.data(), s.length());
    } catch (...) {
      if (conv) delete vp;
      throw;
    }
    break;
  }

    //case SERIALIZE_PRETTY: {
  case SERIALIZE_OPALPRETTY: 
  case SERIALIZE_PYTHONPRETTY: {
    Val* vp = const_cast<Val*>(&given);
    try {
      if (conv) {
	vp = new Val(given);
	ConvertAllOTabTupBigIntToTabArrStr(*vp);
      }
      ostringstream os;
      if (ser==SERIALIZE_PYTHONPRETTY) {
	vp->prettyPrint(os);
      } else {
	prettyPrintOpal(*vp, os);
      }
      string s = os.str();
      dump.expandTo(s.length());
      char* mem = dump.data();
      memcpy(mem, s.data(), s.length());
    } catch (...) {
      if (conv) delete vp;
      throw;
    }
    break;
  }


  // This is *slightly* different from Stringize: on the other side,
  // this just becomes a string, where text and pretty do an Eval.
  // The convert flags has ABSOLUTELY NO EFFECT on this.
  case SERIALIZE_NONE: {
    Str s = given;  
    dump.expandTo(s.length());
    char* mem = dump.data();
    memcpy(mem, s.data(), s.length());
    break;
  }
  default: throw runtime_error("Unknown serialization");
  }

}

// Load a Val from an array containing serialized data. 
inline void LoadValFromArray (const Array<char>& dump, Val& result,
			      Serialization_e ser=SERIALIZE_P0,
			      ArrayDisposition_e array_disposition=AS_LIST,
			      bool perform_conversion_of_OTabTupBigInt_to_TabArrStr = false,
			      MachineRep_e endian=MachineRep_EEEI)
{
  bool conv = perform_conversion_of_OTabTupBigInt_to_TabArrStr;
  char* mem = const_cast<char*>(dump.data());
  size_t   len = dump.length();
  switch (ser) {
    // The new loader supports both versions: P0, P2 (and P1 to a certain ex).
  case SERIALIZE_P0: case SERIALIZE_P2: {
    PickleLoader pl(mem, len);
    pl.env()["supportsNumeric"] = (array_disposition==AS_NUMERIC);
    //cout << pl.env() << endl;
    pl.loads(result);
    if (conv) ConvertAllOTabTupBigIntToTabArrStr(result);
    break;
  }

  // The older P0 loader is available for backwards compat.
  case SERIALIZE_P0_OLDIMPL: {
    bool supports_numeric = (array_disposition==AS_NUMERIC);
    Array<char> c(dump); // This routine ACTUALLY CHANGES THE INPUT BUFFER
    PythonBufferDepickler<Val> pbd(c.length(), c.data(), supports_numeric);
    result = pbd.load();

    if (conv) ConvertAllOTabTupBigIntToTabArrStr(result);
    break;
  }

  // The older P2 loader is available for backwards compat.
  case SERIALIZE_P2_OLDIMPL: case SERIALIZE_P2_OLD: {
    P2TopLevelLoadVal(result, mem);
    if (conv) ConvertAllOTabTupBigIntToTabArrStr(result);
    break;
  }

  // There are many ways to serialize in M2k, we do what
  // the OpalPythonDaemon does: we expect a table
  case SERIALIZE_M2K: {
    Val junk;
    OpalLoadSomeTable(junk, mem, true, endian);
    if (junk.tag=='t') {
      Tab& t = junk;
      if (t.entries()==1 && t.contains("__SPECIAL__")) {
	Val& value = t("__SPECIAL__");
	result.swap(value);
	return;
      }
    }
    result.swap(junk);
    if (conv) ConvertAllOTabTupBigIntToTabArrStr(result);
    break;
  }

  case SERIALIZE_OC: {
    Deserialize(result, mem, conv);
    break;
  }
    //case SERIALIZE_TEXT: 
    //case SERIALIZE_PRETTY: {
  case SERIALIZE_PYTHONTEXT:
  case SERIALIZE_PYTHONPRETTY: {
    result = Eval(mem, len);
    if (conv) ConvertAllOTabTupBigIntToTabArrStr(result);
    break;
  }

  case SERIALIZE_OPALTEXT:
  case SERIALIZE_OPALPRETTY: {
    OpalReader ora(mem, len);
    ora.expectAnything(result, false);
    break;
  }

  case SERIALIZE_NONE: {
    result = Str(mem, len);
    break;
  }
  default: throw runtime_error("Unknown serialization");
  }
}

// A convenience function for dumping a Val to a file: if you want
// finer control over a dump, use the particular serialization by
// itself.  Dump a val to a file, using one of the serializations
// given: ArrayDisposition only matters for SERIALIZE_P0, SERIALIZE_P2
// (See top of file for full discussion).  Note that SERIALIZE_P0 and
// SERIALIZE_P2 should be readable with Python's cPickle.load(file)
inline void DumpValToFile (const Val& v, const string& filename, 
			   Serialization_e ser=SERIALIZE_P0,
			   ArrayDisposition_e arrdisp=AS_LIST,
			   bool perform_conversion_of_OTabTupBigInt_to_TabArrStr = false,
			   MachineRep_e endian=MachineRep_EEEI)
{
  Array<char> buff;
  DumpValToArray(v, buff, ser, arrdisp,
		 perform_conversion_of_OTabTupBigInt_to_TabArrStr, endian);
  ofstream ofs(filename.c_str(), ios::out|ios::binary|ios::trunc);
  if (ofs.good()) {
    char* mem = buff.data();
    size_t len = buff.length();
    ofs.write(mem, len);
  } else {
    throw runtime_error("Trouble writing the file:"+filename);
  }
}

// Load a Val from a file.
inline void LoadValFromFile (const string& filename, Val& result, 
			     Serialization_e ser=SERIALIZE_P0,
			     ArrayDisposition_e arrdisp=AS_LIST,
			     bool perform_conversion_of_OTabTupBigInt_to_TabArrStr = false,
			     MachineRep_e endian=MachineRep_EEEI)
{
  Array<char> buff;
  ifstream ifs(filename.c_str(),ios::in|ios::binary|ios::ate);
  if (ifs.good()) {
    ifstream::pos_type length_of_file = ifs.tellg();
    buff.expandTo(length_of_file);
    char* mem = buff.data();
    ifs.seekg(0, ios::beg);
    ifs.read(mem, length_of_file);
    ifs.close();
  } else {
    throw runtime_error("Trouble reading the file:"+filename);
  }
  LoadValFromArray(buff, result, ser, arrdisp, 
		   perform_conversion_of_OTabTupBigInt_to_TabArrStr,
		   endian);
}

PTOOLS_END_NAMESPACE

#define CHOOSESER_H_
#endif // CHOOSESER_H_
