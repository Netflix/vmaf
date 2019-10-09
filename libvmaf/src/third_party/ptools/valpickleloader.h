#ifndef VALPICKLELOADER_H_
#define VALPICKLELOADER_H_

// Set of functions and defines needed to make the PickleLoader work
// with Val and PicklingTools

#include "ocval.h"


// Defines: we do this to preserve speed 

#define INSERT_STRING(s,start,len) new (&s.u.a) OCString(start, len); s.tag = 'a';   // TODO: A little sketchy, but it saves a copy

# define NONE_VALUE                  None
# define CHEAP_VALUE                 None
# define MAKE_DICT()                 new Tab()
# define EXTRACT_DICT(DICTNAME, ov)  Tab& DICTNAME = (ov)
# define DICT_CONTAINS(D,VALUE)      (D).contains(VALUE) 
# define DICT_GET(D,VALUE)           D(VALUE)

# define MAKE_LIST()                 new Arr()
# define MAKE_LIST1(EXPECT_LEN)      new Arr(EXPECT_LEN)
# define EXTRACT_LIST(LISTNAME, ov)  Arr& LISTNAME = (ov)
# define LIST_SUB(LIST, I)           LIST(I)
# define LIST_LENGTH(LIST)           LIST.length()

# define MAKE_OBJ_FROM_NUMBER(N)     N
# define MAKE_TUP0()                 new Tup()
# define MAKE_TUP1(A)                new Tup(A)
# define MAKE_TUP2(A,B)              new Tup(A,B)
# define MAKE_TUP3(A,B,C)            new Tup(A,B,C)
# define MAKE_TUPN(EXPECT_LEN)       MakeTupN(EXPECT_LEN)
# define EXTRACT_TUP(TUPNAME, FROM)  Tup& TUPNAME = (FROM)
# define TUP_SUB(TUPLE, INDEX)       TUPLE(INDEX)
# define EXTRACT_TUP_AS_LIST(a, u)   Array<OBJ>& a = u.impl()

# define DICT_SWAP_INTO(T,KEY,VALUE) (T).swapInto((KEY), (VALUE))
# define TRUE_VALUE                  true
# define FALSE_VALUE                 false

# define LIST_SWAP_INTO_APPEND(A, V) SwapIntoAppend(A, V)
# define MAKE_BIGINT_FROM_STRING(S,START,LEN)  S = StringToBigInt(START, LEN)

# define MAKE_COMPLEX(R,I)           complex_16(R,I)

# define MAKE_VECTOR1(T, LENGTH)     new Array<T>(LENGTH)
# define EXTRACT_VECTOR(T, A, FROM)  Array<T>& A = (FROM)
# define VECTOR_EXPAND(T,A,LENGTH)   A.expandTo(LENGTH)
# define VECTOR_RAW_PTR(T, A)        A.data()

# define EXTRACT_STRING(O)           string(O)
# define EXTRACT_INT(O)              int(O)
# define EXTRACT_BOOL(O)             bool(O)
# define EXTRACT_T(T, O)             T(O)

# define IS_STRING(O)                ((O).tag=='a')
# define IS_LIST(O)                  ((((O).tag)=='n') && (((O).subtype)=='Z'))

# define GENERIC_GET(CONT, KEY)      CONT(KEY)
# define GENERIC_LENGTH(D)           (D).length()

PTOOLS_BEGIN_NAMESPACE
 
inline Tup* MakeTupN (int expected) 
{ Tup* n=new Tup(); n->impl().resize(expected); return n;}


// When registering things with the factory, they take in some tuple
// and return a Val: REDUCE tend to be more for built-in complicated
// types like Numeric, array and complex.  BUILD tends to more for
// user-defined types.
typedef void (*FactoryFunction)(const Val& name, 
				const Val& input_tuple, 
				Val& environment,  
				Val& output_result);

PTOOLS_END_NAMESPACE 

// The previous works sets-up the generic PickleLoader
#include "pickleloader.h"

//PTOOLS_BEGIN_NAMESPACE
//
//typedef PickleLoaderImpl<Val> PickleLoader;
//
//PTOOLS_END_NAMESPACE 

#endif  // VALPICKLELOADER_H_
