#ifndef OC_AVLHASH_H_

// Specialized versions of the AVLHash that we use all the time.
// Many times, Keys are strings and 8 buckets is more than enough.

// ///////////////////////////////////////////// Include Files

#include "ocport.h"
#include "ocstring.h"
#include "ocavlhasht.h"


OC_BEGIN_NAMESPACE 

// ///////////////////////////////////////////// Forwards

// Needed so the iterators can properly reference the AVLHashs
template <class K, class V, int_u4 OC_AVLCHUNKSIZE>
class AVLHashT;

template <class V>
class AVLHash;

#define OC_AVLHASHCHUNK 8

// ///////////////////////////////////////////// The AVLHashIterator

template <class V>
class AVLHashIterator : public AVLHashTIterator<string, V, OC_AVLHASHCHUNK> {

  public:

    // ///// Methods

    // Constructor
    AVLHashIterator (AVLHash<V>& hash) :
      AVLHashTIterator<string, V, OC_AVLHASHCHUNK>(hash)
    { }

}; // AVLHashIterator


// ///////////////////////////////////////////// The AVLHash

template <class V>
class AVLHash : public AVLHashT<string, V, OC_AVLHASHCHUNK> {
    // Everything from AVLHashT!
};


OC_END_NAMESPACE 

#define OC_AVLHASH_H_
#endif



