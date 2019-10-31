#ifndef OC_ORDAVLHASH_H_

// Specialized versions of the OrdAVLHashT that we use all the time.
// Many times, Keys are strings and 8 buckets is more than enough.

// ///////////////////////////////////////////// Include Files

#include "ochashtable.h"
#include "ocordavlhasht.h"


OC_BEGIN_NAMESPACE 

// ///////////////////////////////////////////// Forwards

// Needed so the iterators can properly reference the AVLHashs
template <class K, class V, int_u4 OC_AVLCHUNKSIZE>
class OrdAVLHashT;

template <class V>
class OrdAVLHash;

#define OC_AVLHASHCHUNK 8

// ///////////////////////////////////////////// The AVLHashIterator

template <class V>
class OrdAVLHashIterator : public OrdAVLHashTIterator<string, V, OC_AVLHASHCHUNK> {

  public:

    // ///// Methods

    // Constructor
    OrdAVLHashIterator (OrdAVLHash<V>& hash) :
      OrdAVLHashTIterator<string, V, OC_AVLHASHCHUNK>(hash)
    { }

}; // OrdAVLHashIterator


// ///////////////////////////////////////////// The AVLHash

template <class V>
class OrdAVLHash : public OrdAVLHashT<string, V, OC_AVLHASHCHUNK> {
    // Everything from OrdAVLHashT!
};


OC_END_NAMESPACE 

#define OC_ORDAVLHASH_H_
#endif



