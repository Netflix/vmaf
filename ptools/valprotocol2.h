#ifndef VALPROTOCOL2_H_
#define VALPROTOCOL2_H_

// Tools for pickling using Python Pickling protocol 2

#include "ocval.h"


// ISSUE:
// When serializing Arrays, they may dump as one of three entities:
// (1) A Python List (heterogenoues list where the elements are not contiguous in memory)
// (2) A Python Array (import array)
// (3) A Numeric Array (import Numeric)
// (4) A numpy array (import numpy)
// NOTE: load is smart enough to handle whatever comes across,
// so this option in NOT necessary in the loading, only dumping.
//
// From C++, both (2) & (3) & (4) mean the same thing (Array<T>),
//           where as (1) means to turn those Array<T> into Arr
//
// but from Python (2) & (3) & (4) are very different.
//
// NumPy is a more recent de-facto standard, and is probably your best bet.
// If you are using older XMPY for any entities, AS_NUMERIC is
// probably your best best.  Array (2) has recently changed the way they
// unpickle (between 2.6 and 2.7), so we would suggest avoiding it.
#include "arraydisposition.h"

PTOOLS_BEGIN_NAMESPACE


// ISSUE:
// Two issues here:
// (1) If you wish to talk to legacy PicklingTools systems that
//     DO NOT have OTAB, Tup, or int_un/int_n, then you can
//     make the serialization FORCE the conversion of those things
//     to compatibile entities. 
// (2) Python 2.2.x does serialization a little differently using cPickle.
//     Longs serialize differently with cPickle (they serialize as strings)
//     Numeric arrays serialize slightly differently
//     Dumps versions don't have the P2 Preamble
//
// IF YOU HAVE NO ENTITIES USING Python 2.2, IGNORE THIS AND USE THE DEFAULT.
//
// NOTE: load is smart enough to handle whatever comes across,
// so this option in NOT necessary in the loading, only dumping.
enum PicklingIssues_e { AS_PYTHON_2_2=0, ABOVE_PYTHON_2_2=1, 
                        CONVERT_OTAB_TUP_ARR__TO__TAB_ARR_STR=2};

// If you wish to set CONVERT__OTAB_TUP_ARR__TO__TAB_ARR_STR=2,
// set it WITH A BIT MASK, i.e.,
// P2TopLevelDumpVal(ov, mem, AS_LIST, 
//                  ABOVE_PYTHON_2_2 | CONVERT_OTAB_TUP_ARR__TO__TAB_ARR_STR);


// Load a Val from memory that heas been serialized using Python
// Pickling Protocol 2 (the binary/fast protocol for pickling).  Fills
// in ov with the appropriate value, and returns where in memory the
// serialization started.
char* P2TopLevelDumpVal (const Val& ov, char* mem, 
			 ArrayDisposition_e dis=AS_LIST,
			 PicklingIssues_e issues=ABOVE_PYTHON_2_2);

// Figure out how many bytes you need: It's (probably) faster to
// precompute the number of bytes you need (approx.) than to do an if
// after every memory access.
size_t P2TopLevelBytesToDumpVal (const Val& ov, ArrayDisposition_e dis=AS_LIST,
				 PicklingIssues_e issues=ABOVE_PYTHON_2_2);
size_t P2BytesToDumpVal (const Val& ov, ArrayDisposition_e dis=AS_LIST,
			 PicklingIssues_e issues=ABOVE_PYTHON_2_2);

// Load a Val from memory that heas been serialized using Python
// Pickling Protocol 2 (the binary/fast protocol for pickling).
char* P2TopLevelLoadVal (Val& ov, char* mem);

// Dump/Load, without preambles and postambles
char* P2DumpVal (const Val& v, char* mem, ArrayDisposition_e dis=AS_LIST,
		 PicklingIssues_e issues=ABOVE_PYTHON_2_2);
char* P2LoadVal (Val& v, char* mem);


PTOOLS_END_NAMESPACE

#endif // VALPROTOCOL2_H_
