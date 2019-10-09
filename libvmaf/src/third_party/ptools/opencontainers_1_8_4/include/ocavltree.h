#ifndef OC_AVLTREE_H_

// Specialized versions of the AVLTree that we use all the time.
// Many times, Keys are strings and 8 buckets is more than enough.

// ///////////////////////////////////////////// Include Files

#include "ocport.h"
#include "ocstring.h"
#include "ocavltreet.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// Forwards

// Needed so the iterators can properly reference the AVLTrees
template <class K, class V, int_u4 OC_AVLCHUNKSIZE>
class AVLTreeT;

template <class V>
class AVLTree;

#define OC_AVLCHUNK 8

// ///////////////////////////////////////////// The AVLTreeIterator

template <class V>
class AVLTreeIterator : public AVLTreeTIterator<string, V, OC_AVLCHUNK> {

  public:

    // ///// Methods

    // Constructor
    AVLTreeIterator (AVLTree<V>& tree) :
      AVLTreeTIterator<string, V, OC_AVLCHUNK>(tree)
    { }

}; // AVLTreeIterator


// ///////////////////////////////////////////// The AVLTree

template <class V>
class AVLTree : public AVLTreeT<string, V, OC_AVLCHUNK> {
    // Everything from AVLTreeT!
};


OC_END_NAMESPACE

#define OC_AVLTREE_H_
#endif



