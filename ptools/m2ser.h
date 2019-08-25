#ifndef M2SER_H_

// These funcations allows us to serialize and deserialize data as
// Midas 2k would.

// A lot of the code here is pulled directly out of Midas 2k, slightly
// modified to work under the Val model.

#include "ocval.h"
#include "m2convertrep.h"

PTOOLS_BEGIN_NAMESPACE

MachineRep_e NativeEndian ();

// Dump as M2k Binary Serialization would and return a buffer
// that IS ADOPTED!  You must "delete [] buffer" when you are
// done with it!! 
int OpalDumpVal (const Val& v, char*& mem, 
		 MachineRep_e endian=MachineRep_EEEI);

// For OpalDaemon Serialization, it assumes it's a Tab
// and DOES NOT put the preamble on *the single char 9*
int OpalDumpTab (const Tab& t, char*& mem,
		 MachineRep_e endian=MachineRep_EEEI);
int OpalDumpArr (const Arr& t, char*& mem,
		 MachineRep_e endian=MachineRep_EEEI);
int OpalDumpOTab (const OTab& t, char*& mem,
		  MachineRep_e endian=MachineRep_EEEI);
int OpalDumpTup (const Tup& t, char*& mem,
		 MachineRep_e endian=MachineRep_EEEI);


// Load from M2k Binary Serialization
char* OpalLoadVal (Val& v, char* mem, 
		   MachineRep_e stream_endian=MachineRep_EEEI);

// For OpalDaemon Serialization, it assumes it's a Tab
// and DOES NOT put the preamble on *the single char 9*
// Also, this could be either a TAB or an ARR!!
char* OpalLoadSomeTable (Val& v, char* mem, 
			 bool convert_tab_to_arr_if_possible=true,
			 MachineRep_e endian = MachineRep_EEEI);

PTOOLS_END_NAMESPACE


#define M2SER_H_
#endif  // M2SER_H_
