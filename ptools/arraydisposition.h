#ifndef ARRAYDISPOSITION_H_
#define ARRAYDISPOSITION_H_

////////////////////////////////// ArrayDisposition_e ////////////////
// Different kinds of POD (Plain Old Data: int_1, int_2, real_4, etc.) arrays: 
// there are essentially 4 different types of POD arrays that might be moving 
// around: 
//
// (1) a = [1,2,3]  
//     Python styles lists (which are inefficient for storing 
//     homogeneous data)
//
// (2) import array; a = array.array('i',[1,2,3])
//     the arrays from the Python module array 
//
// (3) DEPRECATED 
//     import Numeric; a = Numeric.array([1,2,3], 'i')
//     the Numeric arrays which are built in to XMPY,
//     but most standard Pythons do not have it installed.
// 
// (4) import numpy; a = numpy.array([1,2,3], dtype=uint32)
//     numpy has replaced Numeric as is the defacto numeric array repr
//     
// (5) As Numeric wrapper (only supported in Python currently)
//
// In C++, POD arrays are handled as Array<T>, thus (2) & (3) & (4)
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
// (4) Python numpy is an external module, but a de-facto standard of late
//     (a) may not be installed on your system
// None of these solutions is perfect, but going to NumPy will
// probably fix most of these issues in the future.
/////////////////////////////////////////////////////////////////////

PTOOLS_BEGIN_NAMESPACE

enum  ArrayDisposition_e { AS_NUMERIC, AS_LIST, AS_PYTHON_ARRAY, 
			   AS_NUMERIC_WRAPPER, AS_NUMPY };

PTOOLS_END_NAMESPACE

#endif // ARRAYDISPOSITION_H_
