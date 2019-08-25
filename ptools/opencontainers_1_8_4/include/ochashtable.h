#ifndef OC_HASHTABLE_H_

// Specialized versions of the HashTableTs that we use all the time.
// Many times, Keys are strings and 8 buckets is more than enough.

// ///////////////////////////////////////////// Include Files

#include "ocport.h"
#include "ocstring.h"

OC_BEGIN_NAMESPACE

// Newer versions of g++ seem to require HashFunctions be defined
// BEFORE the actual include

// HashFunction necessary for handles in lookups
inline int_u4 HashFunction (void* v) { AVLP p = (AVLP)v; return p; }
inline int_u4 HashFunction (const int& val) { return val; }

OC_END_NAMESPACE

#include "ochashtablet.h"


OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// Forwards

// Needed so the iterators can properly reference the tables.
template <class K, class V, int_u4 bucketLists>
class HashTableT;

template <class V>
class HashTable;

#define OC_BUCKETS 8

// ///////////////////////////////////////////// The HashTableIterator

template <class V>
class HashTableIterator : public HashTableTIterator<string, V, OC_BUCKETS> {

  public:

    // ///// Methods

    // Constructor
    HashTableIterator (HashTable<V>& table) :
      HashTableTIterator<string, V, OC_BUCKETS>(table)
    { }

}; // HashTableIterator


// ///////////////////////////////////////////// The HashTable

template <class V>
class HashTable : public HashTableT<string, V, OC_BUCKETS> {
    // Everything from Hashtable!
};

OC_END_NAMESPACE

#define OC_HASHTABLE_H_
#endif



