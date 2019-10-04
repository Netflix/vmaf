#ifndef OC_ORDAVLHASHT_H_

// The AVLOrdHashT is a class for managing Key/Value pairs, much like
// HashTableT (in fact, you should go look there first).  It has
// almost exactly the same interface (there are a few extra routines).

// Unlike AVLTreeT, you DO NOT need operator< supported by your keys.

// This is an "extension" of the AVLHashT: The real difference is that
// every insert is also kept track of "in order" so that you can
// traverse the data structure in the order of insertion.

// The OrdAVLHashT has the same interface as the AVLHashT, and is
// a plug-in replacement.

// Implementation Notes: All we really need to do is add a doubly-link
// list to every node in the AVLTree: whenever an insert happens, the
// item simply goes to the end of the list (we maintain a circular
// doubly-linked list with a dummy header so this is trivial).
// Deleting an item from the list is standard, simply delete the node
// out of the linked list.


// ///////////////////////////////////////////// Includes

#include "ocport.h"
#include "ocarray.h"  // only needed to consistency checks

#include "ocstreamingpool.h" // class Allocator;  // Forward
#include "ocsort.h"          // for OCQuickSort

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// The AVLOrdNode_ struct

// Implementation Detail: Management information needed for each node
// and Key-Value pairs
template <class K, class V>
struct AVLOrdNode_ {
    AVLOrdNode_ (AVLOrdNode_* l, AVLOrdNode_* r, AVLOrdNode_* p, 
	      const K& k, int_u4 keyhash, const V& v):
      left_(l), right_(r), parent_(p), 
      balance(0), 
       // on_freelist_ref_count(0), // DO NOT INITIALIZE!!
      hashkey(keyhash), 
      next(0),
       // TODO:  have to figure out how to use allocator on these
      key(k), value(v)

    { }

  
    ~AVLOrdNode_ () { unlink_(); } 

    // Pointers to left, right, and parent.  Rather than store NULL
    // pointers for left and right when they are empty, we set the
    // lower bit of the pointer to indicate its empty and use the rest
    // of the pointer to point to the inorder predecessor and inorder
    // succesor (left and right subtrees respectively).  Why don't we
    // just use an extra flag?  (a) save space (b) When you rotate, no
    // need to check anything extra ... just copy the pointer (with or
    // without the bit set) directly.
    AVLOrdNode_ *left_, *right_, *parent_;

    // Return left node, or 0 if "empty" (threaded)
    AVLOrdNode_<K,V>* left() const
    { AVLP t=(AVLP)left_; return t&1 ? 0 : (AVLOrdNode_<K,V>*)t;}

    // Return right node, or 0 if "empty" (threaded)
    AVLOrdNode_<K,V>* right() const
    { AVLP t=(AVLP)right_; return t&1 ? 0: (AVLOrdNode_<K,V>*)t;}

    // Return parent, or 0 if empty.
    AVLOrdNode_<K,V>* parent() const { return parent_; } 
    
    // Return the plain pointer, threaded.
    AVLOrdNode_<K,V>* threadMe () const
    { AVLP t=(AVLP)this; t |= 1; return (AVLOrdNode_<K,V>*)t; }
    
    // Return the inorder succesor, or 0 if there is none.
    AVLOrdNode_<K,V>* inorderSucc() const
    {
      AVLP t=(AVLP)this->right_;
      AVLOrdNode_<K,V>* r = (AVLOrdNode_<K,V>*)(t & AVL_EMPTY_MASK);
      if (t && !(t & 1)) { // If not threaded, follow left as far as possible
	for (; r->left(); r=r->left())
	  ;
      }
      return r;
    }
    
    // Return the inorder predecessor, or 0 if there is none.
    AVLOrdNode_<K,V>* inorderPred() const
    {
      AVLP t=(AVLP)this->left_;
      AVLOrdNode_<K,V>* l = (AVLOrdNode_<K,V>*)(t & AVL_EMPTY_MASK);
      if (t && !(t & 1)) { // If not threaded, follow right as far as possible
	for (; l->right(); l=l->right())
	  ;
      }
      return l;
    }
    
    
    // right subtree height - left subtree height. AVL property says
    // that this can never other than -1, 0, or 1.  If it is, then we
    // need a rotate.  (- means left heavy, + means right heavy).
    int_1 balance;  
    
    // Flags for memory allocation: if negative, then this node is one
    // of the later nodes in an allocated chunk and represents how
    // many away from the header.  If positive or zero, then this is
    // the header and this is the ref count of how many nodes of this
    // chunk are currently on the freelist.
    int_1 on_freelist_ref_count; 
    
    // The hash key: computed hash value, stored in the table.
    int_u4 hashkey;
    
    // In the rare case of collisions, there is a linked list of these
    // nodes we search through.  When a node is on a linked list (not
    // the header), balance will be a special key
    AVLOrdNode_* next;

    // All nodes are kept in insertion order: This is a circularly-linked
    // list (with the root as a dummy node, so insertion and deletion is 
    // simple)
    AVLOrdNode_ *ordnext, *ordprev;
    
    // Insert this node after the given node
    void link_ (AVLOrdNode_<K,V>* given)
    {
      this->ordnext = given->ordnext;
      this->ordprev = given;
      given->ordnext = this;
      this->ordnext->ordprev = this;
    }

    // Take node off of doubly-linked list with dummy node.
    void unlink_()
    {
      this->ordprev->ordnext = this->ordnext;
      this->ordnext->ordprev = this->ordprev;
    }

    // This node was previously on a list: the data in the node
    // moved from the old_node to the new_node, so we have to relink:
    //  this causes the old node to "link itself"
    static void relink_ (AVLOrdNode_<K,V>* oldnode, AVLOrdNode_<K,V>* newnode)
    {
      // Make links at oldnode valid
      {
	// Take old node off its old lists
	oldnode->unlink_();
	// .. and move back 
	oldnode->ordnext = newnode->ordnext;
	oldnode->ordprev = newnode->ordprev;
	oldnode->ordnext->ordprev = oldnode;
	oldnode->ordprev->ordnext = oldnode;
      }
      // The newnode (which will disappear) needs to be
      // put on a linked-list with itself so its final
      // unlink will work
      {
	newnode->ordprev = newnode->ordnext = newnode;
      }
    }

    // The key and value for this node
    K key;
    V value;

  
    // Similar to a swap, but we want to preserve where the node is in
    // the tree, so we only swap key, value, and insertion order
    // (value related stuff)
    void swapNodeValues (AVLOrdNode_<K,V>* other)
    {
      OC_NAMESPACED::swap(hashkey, other->hashkey); // use default for int
      OC_NAMESPACED::swap(key, other->key);
      OC_NAMESPACED::swap(value, other->value);
    }

    // Some nodes need to do more management after a swap
    void nodeUpdateAfterSwap (AVLOrdNode_<K,V>* other)
    {
      // Can't just swap linked-list linked, duh!
      relink_(this, other);
      
      // When we swap, *THIS* node must have a non-empty bucketlist
      if (next!=0) { throw runtime_error("Internal OrdAVLerror"); }

      // The node moving up may have a bucketlist!
      if (other->next) {
	this->next  = other->next;
	other->next->parent_ = this; // DOUBLY-linked list
      }
    }
    
};  // AVLOrdNode_

// Most AVLHashT's can get by with this
template <class K,class V>
inline AVLOrdNode_<K,V>* 
AVLCreateNode (void* memory_to_construct_into,
	       AVLOrdNode_<K,V>*l, AVLOrdNode_<K,V>*r, AVLOrdNode_<K,V>*p, 
	       const K& k, int_u4 keyhash, const V& v,
	       Allocator*)
{
  return new (memory_to_construct_into) AVLOrdNode_<K,V>(l,r,p,k,keyhash,v); 
}

// ///////////////////////////////////////////// Forwards

template <class K, class V, int_u4 CHUNKSIZE> class OrdAVLHashTIterator;
template <class K, class V, int_u4 CHUNKSIZE> class OrdAVLHashTSortedIterator;


// ///////////////////////////////////////////// The OrdAVLHashT Class

template <class K, class V, int_u4 CHUNKSIZE>
class OrdAVLHashT {

  protected:  
    friend class OrdAVLHashTIterator<K,V,CHUNKSIZE>;
    friend class OrdAVLHashTSortedIterator<K,V,CHUNKSIZE>;
    enum Found_e { FOUND,NOT_FOUND_LEFT,NOT_FOUND_RIGHT,NOT_FOUND_BUCKETS };
    enum { AVLBUCKETLISTFLAG = 127 }; // Indicates bucketlist node
    typedef AVLOrdNode_<K,V> N;

  public:

    // ///// Methods
  
    // Constructor 
    OrdAVLHashT (Allocator* a=0) :
      entries_(0),
      allocator_(a)
    {
      helpConstruct_();
      if (sizeof(root_->right_) != sizeof(AVLP))
	throw 1; // MidasException("Pointers are wrong size for AVL trees");
    }
    
    // Copy Constructor
    OrdAVLHashT (const OrdAVLHashT<K,V,CHUNKSIZE>& rhs, Allocator* a=0) :
      entries_(0),
      allocator_(a)
    { 
      helpConstruct_();
      copyTable_(rhs);
    }
    
    // Assignment
    OrdAVLHashT<K,V,CHUNKSIZE>& operator= (const OrdAVLHashT<K,V,CHUNKSIZE>& rhs)
    {
      // allocator stays the same!!
      clear();
      copyTable_(rhs);
      return *this;
    }
    
    // Destructor
    ~OrdAVLHashT () 
    { 
      clear(); 
      
      // root_ and freelist_ special, see helpConstruct_
      char* temp = (char*)freelist_;

      // Gets both root_ AND freelist_
      if (allocator_) { 
	allocator_->deallocate(temp);
      } else {
	delete [] temp; 
      }
    }
    
    // Clear the table
    void clear ()
    {
      // Iterate and destroy: no need for recursion
      N* dummy = root_;
      N* next_node = 0;
      for (N* curr=dummy->ordnext; curr!=dummy; curr=next_node) {
	next_node = curr->ordnext;
	deleteAVLOrdNode_(curr); // Handles unlinking for us
      }
      entries_ = 0;
      root_->right_ = 0;
    }
        
    // Returns true is the table contains a key which is equal to key.
    // Returns false otherwise.
    bool contains (const K& key) const
    { 
      Found_e found_where; (void)lookup_(key, found_where);
      return (found_where==FOUND);
    }
    
    // The current number of key-value pairs in the table
    size_t entries () const { return entries_; }


    // From Python PEP-372: 
    // How does OrderedDict handle equality testing?  Comparing two
    // ordered dictionaries implies that the test will be
    // order-sensitive so that list (od1.items())==list(od2.items()).
    // When ordered dicts are compared with other Mappings, their
    // order insensitive comparison is used. This allows ordered
    // dictionaries to be substituted anywhere regular dictionaries
    // are used.
    //
    // The OrdAVLHashT respects how Python does ordered dictionaries
    // and does order sensitive comparison.  Thus, two odicts only
    // compare the same if they have the same number of keys in the
    // same order.
    //
    // See if two tables are equal: Two tables are equal if they have
    // the same number of keys and values and all keys and values
    // match.  (NOTE: Yes, this should be a global function, but older
    // Solaris's don't like non-type template params on global
    // functions)
    bool operator== (const OrdAVLHashT<K,V,CHUNKSIZE>& rhs) const
    {
      if (entries() != rhs.entries()) return false;
      OrdAVLHashTIterator<K,V,CHUNKSIZE> it1(*this); 
      OrdAVLHashTIterator<K,V,CHUNKSIZE> it2(rhs); 
      while (it1() && it2()) {
	if (it1.key()!=it2.key()) return false;
	if (it1.value()!=it2.value()) return false;
      }
      return true;
    }
    bool operator!= (const OrdAVLHashT<K,V,CHUNKSIZE>&t1) const
    { return !(*this==t1); }

    // Returns true is the table contains a key which is equal to
    // "target" and puts the matching key into return_key.  Returns
    // false otherwise and leaves retKey untouched.  Equality is
    // measured by key equality.
    bool find (const K& target, K& return_key) const
    {
      Found_e found_where; N* node = lookup_(target, found_where);
      if (found_where==FOUND) {
	return_key = node->key;
	return true;
      } else {
	return false;
      }
    }
    
    // Returns true if the table contains a key which is equal to
    // "key" and puts the associated value into return_val.  Returns
    // false otherwise and leaves return_val untouched.
    bool findValue (const K& key, V& return_val) const
    {
      Found_e found_where; N* node = lookup_(key, found_where);
      if (found_where==FOUND) {
	return_val = node->value;
	return true;
      } else {
	return false;
      }
    }
    
    // Returns true if the dictionary contains a key which is equal to
    // key and puts the matching key into into return_key and the
    // associated value into return_val.  Returns false otherwise and
    // leaves return_key and return_val untouched.
    bool findKeyAndValue (const K& key, K& return_key, V& return_val) const
    {
      Found_e found_where; N* node = lookup_(key, found_where);
      if (found_where==FOUND) {
	return_val = node->value;
	return_key = node->key;
	return true;
      } else {
	return false;
      }
    }
    
    // Inserts the key and value into the table.  If the key is
    // already in there, replace it.  New entries are conceptually
    // added to the end, overwriting a previous value keeps its
    // current position.
    void insertKeyAndValue (const K& key, const V& value)
    {
      int_u4 hashkey = HashFunction(key);
      Found_e found_where; N* node = lookup_(key, hashkey, found_where);
      if (found_where==FOUND) {
	node->value = value; // Assumes the op= for V works
      } else {
	(void)notInTableInsert_(node, key, hashkey, value, found_where);
      }
    }
    
    // Returns true if the table has no items in it, false otherwise.
    bool isEmpty () const { return entries_==0; }
    
    // Returns true and removes the (key/value) pair where the key is
    // equal to the key.  Returns false if there is no such key.
    bool remove (const K& key) 
    {
      Found_e found_where; N* node = lookup_(key, found_where);
      if (found_where==FOUND) {
	deleteItemFromTable_(node);
	return true;
      } else {
	return false;
      }
    }
    
    // Lookup the key "key" and return its associated value as an
    // l-value reference.  If the key is not in the dictionary, then
    // it is added to the dictionary.  In this case, the value
    // associated with the key will be provided by the default
    // constructor ofor objects of types V.  New entries are
    // conceptually added to the end, overwriting a previous value
    // keeps its current position.
    V& operator[] (const K& key)
    {
      int_u4 hashkey = HashFunction(key);
      Found_e found_where; N* node = lookup_(key, hashkey, found_where);
      if (found_where==FOUND) {
	return node->value;
      } else {
	return notInTableInsert_(node, key, hashkey, V(), found_where)->value;
      }
    }

    // Lookup the key "key" and return its associated value as an
    // l-value reference.  If the key is not in the dictionary, then
    // an out_of_range is thrown
    V& operator() (const K& key) const
    {
      int_u4 hashkey = HashFunction(key);
      Found_e found_where; N* node = lookup_(key, hashkey, found_where);
      if (found_where==FOUND) {
	return node->value;
      } else {
	throw out_of_range("Key "+Stringize(key)+" not in table");
      }
    }


    // See if data structure internally consistent.  Very expensive.
    bool consistent ()
    {
      if (root_->right()==0) return true;
      if (!testBST_(root_->right(), root_->right()->hashkey))
	return false;
      if (!testBalance_(root_->right()))
	return false;
      if (!parentage_(root_))
	return false;
      if (!recursiveElements_(root_->right())==entries_)
	return false;
      return threadedOK_();
    }
    
    void prefix (ostream& os) {
      prefixNode_(os, root_->right());
    }
    
    void infix (ostream& os) {
      infixNode_(os, root_->right());
    }

    void swap (OrdAVLHashT<K,V,CHUNKSIZE>& rhs)
    {
      OC_NAMESPACED::swap(root_,     rhs.root_);
      OC_NAMESPACED::swap(freelist_, rhs.freelist_);
      OC_NAMESPACED::swap(entries_,  rhs.entries_);
      OC_NAMESPACED::swap(allocator_, rhs.allocator_);
    }

    // Avoid an extra copy when inserting into a table by giving key
    // and value which are "disposable" (but still have to be left in
    // a valid state): thus after the swapInto, the key and value
    // returned have the "default construction" K()/V(), and the table
    // has the inserted key and value.  This happens via the "swap"
    // which is typically O(1), which is why this is typically faster
    // than the standard insert, but harder to use (obviously, as the
    // K&V are no longer the same after the call!).  This is a useful
    // routine for serializations where you construct the value once
    // as you deserialize, only to realize it needs to be copied into
    // a container (but a full copy is too expensive).  Like contains,
    // this returns true to show the key was already there (so we just
    // swapped out the value) false if the key wasn't there already
    // (so both key and value are swapped)
    bool swapInto (K& key_to_swap, V& value_to_swap)
    {
      int_u4 hashkey = HashFunction(key_to_swap);
      Found_e found_where; N* node = lookup_(key_to_swap, found_where);
      // Found the key already in the table, so we just swap
      // out the value.
      if (found_where == FOUND) {
	OC_NAMESPACED::swap(value_to_swap, node->value);
	return true;
      } 
      // The key isn't there, so we find where it would go, and
      // instead of copying the key, we swap it in!
      else {
	// All of this insertion assumes the hashkey has been
	// correctly computed.  We don't futz with with hashkey, and
	// the key and value only need to be updated after all the
	// inserting is done.
	N* current = notInTableInsert_(node, K(), hashkey, V(), found_where);
	OC_NAMESPACED::swap(key_to_swap, current->key);
        OC_NAMESPACED::swap(value_to_swap, current->value);
	return false;
      }
    }
    
  protected:
    
    // ///// Data Members
    
    // Points to dummy node: the tree is on the left, parent is 0 (as
    // is right).  Dummy node makes insertions and deletions easier
    // because no special cases.
    N* root_;
    
    // Freelist of nodes: Freelist_ is a dummy node in a circular,
    // doubly-linked list of free node.  We only use left and right as
    // next and prev pointers: the nodes on this list are NOT
    // threaded.
    N* freelist_;
    
    // The number of actual entries in the table
    size_t entries_;

    // The allocator used.  This is not quite like STL allocators:
    // we need to be able to choose between different allocators at
    // run-time because some tables may exist in global shared memory,
    // some use new, etc.  
    Allocator* allocator_;
    
    // ///// Methods
    
    // Helper function used by a lot of methods above: Do a lookup and
    // see if the item is already there: if it is, return pointer to
    // it and indicate FOUND.  If not there, return a pointer to its
    // parent (as if it WERE there) and indicate whether the child
    // would be on the left or the right, or in a bucket list.
    N* lookup_ (const K& key, Found_e& found_where) const
    { return lookup_(key, HashFunction(key), found_where); }
    
    N* lookup_ (const K& key, int_u4 hashkey, Found_e& found_where) const
    {
      // It's just a binary search tree: look left is less than, right
      // otherwise.  IF WE DON'T FIND it, then we return the parent
      // (so we don't have to do another lookup).
      N* current = root_->right();
      while (current) {
	if (hashkey==current->hashkey) {
	  // Search for exact match:  keyhash only gave us "bucket"
	  for (N* front=current;front;front=front->next) {
	    if (front->key==key) {
	      found_where = FOUND;
	      return front;
	    }
	  }
	  // Return the front node so we can link it in
	  found_where = NOT_FOUND_BUCKETS;
	  return current;
	} else if (hashkey<current->hashkey) {
	  N* left = current->left();
	  if (left) {
	    current = left;
	  } else {
	    found_where = NOT_FOUND_LEFT;
	    return current;
	  }
	} else {
	  N* right = current->right();
	  if (right) {
	    current = right;
	  } else {
	    found_where = NOT_FOUND_RIGHT;
	    return current;
	  }
	}
      }
      // Return dummy node if nothing in tree
      found_where = NOT_FOUND_RIGHT;
      return (N*)root_;
    }
    
    // Insert an item into the table: The parent is the node above us
    // that adopts us!  The key and value get copied into the new
    // node, and the found_where indicates whether the parent should
    // point us at the left or the right or if in bucket list.
    N* notInTableInsert_ (N* parent, const K& key, int_u4 keyhash, 
			  const V& value, Found_e found_where)
    {
      // Insert new node in the right place: When we do this, we also
      // thread the tree: Note that we use parent->left and
      // parent->right directly to preserve the threaded bit.
      N* current=0;
      N* threaded_dad = parent->threadMe();
      if (found_where==NOT_FOUND_LEFT) {
	current = newAVLOrdNode_(parent->left_,threaded_dad,parent,
				 key,keyhash,value);
	parent->left_ = current; // left not threaded anymore
      } else if (found_where==NOT_FOUND_RIGHT) {
	current = newAVLOrdNode_(threaded_dad,parent->right_,parent,
				 key,keyhash,value);
	parent->right_ = current; // right not threaded anymore
      } else if (found_where==NOT_FOUND_BUCKETS) {
	current = newAVLOrdNode_(0,0,parent,key,keyhash,value);
	current->balance = AVLBUCKETLISTFLAG;
	current->next = parent->next;
	parent->next = current; // NUL ended doubly-linked list
	if (current->next) current->next->parent_ = current;  
      } 
      N* ret_val = current;
      entries_++;
      
      // Insert at the end of the "insertion order" list, only if this
      // is a new node.
      if (found_where!=FOUND) {
	current->link_(root_->ordprev); 
      }

      if (found_where!=NOT_FOUND_BUCKETS) {	
	// Return up through the parents, updating balances as we go.
	// We stop as soon as a balance goes to 0 (or a rotate happens).
	while (current->parent()!=root_) {
	  // Move up to the parent
	  N* kid = current;
	  current = current->parent();
	  // Figure out if we got here by going left or right
	  if (current->left()==kid) {  // ... came left
	    switch (--current->balance) {
	    case -2: rotate_(current, kid); // Need to rotate, then done
	    case  0: return ret_val;;    // All done
	    case -1: continue;  // Balance okay @ -1: update parent's balance
	    }
	  } else { // ... came right
	    switch (++current->balance) {
	    case +2: rotate_(current, kid); // Need to rotate, then done
	    case  0: return ret_val;    // All done
	    case +1: continue;  // Balance okay @ +1: update parent's balance
	    } 
	  }
	}
      }
      return ret_val;
    }
    
    // Delete a node from the tree: we know that the node's left or
    // right subtree is empty, which makes it easy to delete (the
    // parent can adopt the non-empty subtree in place of the node and
    // we can delete the node directly).
    void easyDelete_ (N* current)
    {
      // Both subtrees empty
      if (!current->left() && !current->right()) {
	// Adjust succ. and pred threads:  no need (both must be ABOVE)
	;
	// Adopt proper threaded pointer
	if (current->parent_->left_==current) {
	  current->parent_->left_ = current->left_;
	} else {
	  current->parent_->right_ = current->right_;
	}
      }
      // Left of current is empty, so parent adopts right subtree of current
      else if (!current->left()) {
	// Adjust succ. and pred threads:  When there's no left subtree,
	// No predecessor points to current (because must be ABOVE current),
	// the successor of current may point to us.
	N* succ = current->inorderSucc();
	if (succ && succ->left_==current->threadMe()) { // threaded pointer?
	  succ->left_ = current->left_; // Get current's threaded left 
	}
	// Adopt right subtree.  
	if (current->parent_->left_==current) { 
	  current->parent_->left_ = current->right_; // keep threaded
	} else { 
	  current->parent_->right_ = current->right_; // keep threaded
	}
	if (current->right()) current->right_->parent_ = current->parent();
      } 
      // Right of current is empty, so parent adopts left subtree of current
      else if (!current->right()) {
	// Adjust succ. and pred threads:  When there's no right subtree,
	// No succesor points to current (because must be ABOVE current),
	// the predecessor to current may point to us.
	N* pred = current->inorderPred();
	if (pred && pred->right_==current->threadMe()) { // threaded pointer?
	  pred->right_ = current->right_; // Get current's threaded right
	}
	// Adopt left subtree.  
	if (current->parent_->left_==current) { 
	  current->parent_->left_ = current->left_; // keep threaded
	} else { 
	  current->parent_->right_ = current->left_; // keep threaded
	}
	if (current->left()) current->left_->parent_ = current->parent();
      } 
      entries_--;
      deleteAVLOrdNode_(current); // Give "back" to memory system
    }
    
    
    // Delete current from the tree
    void deleteItemFromTable_ (N* current)
    {
      // Are we a normal node with a non-empty bucketlist?
      N* next = current->next;
      if (current->balance != AVLBUCKETLISTFLAG && next) {
	// ARGH! We have to move up another entry to the top
	current->key = next->key; // No need for keyhash, same bucket
	current->value = next->value; 
	current = next; // Fall through and let below unlink bucket
      } 
      
      // Is current a node on a bucketlist (rather than an AVL Node)?
      if (current->balance==AVLBUCKETLISTFLAG) {
	// In the middle of a list, easy delete, just unlink!
	current->parent_->next = current->next;
	if (current->next) current->next->parent_ = current->parent_;
	deleteAVLOrdNode_(current);
	entries_--;
	return;
      } 
      
      // Otherwise, plain node to delete with no next.
      else {
	deleteNode_(current);
      }
    }
    
    
    // Delete a node only after we know it's a NODE, not a bucket list
    // entry.
    void deleteNode_ (N* current)
    {
      // The problem is, it only makes sense to delete items with no
      // right node [eqv: left] (the other subtree must have either
      // balance 0 or 1 as well because of the AVL nature).  Thus, we
      // find the "inorder succesor" of the current node (which is
      // guranteed to have no left kids) and move that inorder
      // successor to the "current" and delete the current node AS IF
      // it were at the inorder succesors spot.
      if (current->left() && current->right()) {
	N* inorder_succ = current->inorderSucc();
	// Copy inorder_succ over node to REALLY delete, then delete
	// it instead
	current->swapNodeValues(inorder_succ); // swap faster than op=
	current->nodeUpdateAfterSwap(inorder_succ); // extra? 
	current = inorder_succ;
      }
      
      // Assertion: current is now the node to be deleted: it is a
      // node that has either no left kid or it has no right kid.
      N* tbd = current; // To Be Deleted
      
      // The updating of balances is like insertion.  Note: once we
      // get to the critical node (where balance goes to -2 or +2),
      // notice that we rotate on the OTHER side of the tree [Think of
      // a deletion as an insertion on the OTHER side of the tree].
      
      // Return up through the parents, updating balances as we go.
      // We stop as soon as a balance goes to -1,+1.
      while (current->parent()!=root_) {
	// Move up to the parent
	N* kid = current;
	current = current->parent();
	// Figure out if we got here by going left or right
	if (current->left()==kid) {  // ... came left
	  switch (++current->balance) {
	  case +2: rotate_(current,current->right());// Need to rotate
	    current = current->parent();// After rotate, update current
	    if (current->balance==0) continue;
	  case +1: goto breakout;    // All done
	  case  0: continue;  // Balance okay @ 0: update parent's balance
	  }
	} else { // ... came right
	  switch (--current->balance) {
	  case -2: rotate_(current,current->left()); // Need to rotate
	    current = current->parent();// After rotate, update current
	    if (current->balance==0) continue;
	  case -1: goto breakout;    // All done
	  case  0: continue;  // Balance okay @ 0: update parent's balance
	  } 
	}
      }
      
      // Delete the node at the end so no special case in above logic.
      // (no worrying about already deleted nodes while traversing upwards).
    breakout:
      easyDelete_(tbd);    // Unlink node out of tree, and give back memory
    }
    
    
    // This is called when a rotation is needed: This figures out,
    // based on the balance of the parent and kid, what kind of
    // rotation is needed: single (left or right) or double
    // (left-right, or right-left).
    void rotate_ (N* parent, N* kid)  
    {
      // Double-rotate if parent and kid balances are different signs
      int val = parent->balance*kid->balance;
      if (val==-2) {
	doubleRotate_(parent, kid);
      }
      // Single rotate if the same sign:  balances go to zero
      else if (val==2) {
	singleRotate_(parent, kid);
      }
      // Single rotate if balance product is zero (means kid is zero).
      // This means the new nodes get a balance of -1*same_sign as
      // original parent.
      else if (val==0) {
	int sign_of_original_parent = parent->balance < 0 ? -1 : +1;
	singleRotate_(parent, kid);
	parent->balance = sign_of_original_parent;
	kid->balance = -sign_of_original_parent;
      }
      // Huh?
      else {
	throw 1; // MidasException("Internal Consistency error of AVL Tree");
      }
    }
    
    // Single rotate.  Update balances.
    void singleRotate_ (N* parent, N* kid)
    {
      // Draw the picture ... ya, this code sucks Note: this is one of
      // the few places where we mess with left and right directly: we
      // want to preserve the threaded bit when we copy rotations
      // around, but we need to check for "empty" as well
      N* temp;
      if (parent->left()==kid) { // rotate kid over parent left to right
	temp = kid->right();
	if (temp) {
	  temp->parent_ = parent;
	  parent->left_ = temp;
	} else { // Special: right of kid being rotated, watch Threads!
	  parent->left_ = kid->threadMe();
	}
	kid->right_ = parent;
      } else { // rotate kid over parent right to left
	temp = kid->left();
	if (temp) {
	  temp->parent_ = parent;
	  parent->right_ = temp;
	} else { // Special: left of kid being rotated, watch Threads!
	  parent->right_ = kid->threadMe();
	}
	kid->left_ = parent;
      }
      // Reparent the kid
      temp = parent->parent();
      parent->parent_ = kid;
      if (temp->left() == parent) { 
	temp->left_ = kid; 
      } else {
	temp->right_ = kid;
      }
      kid->parent_ = temp;
      // When single rotates happen in AVL trees, both balances go to
      // zero.  Double-rotates are more complicated. See below.
      parent->balance = 0;
      kid->balance = 0;
    }
    
    // Double rotate
    void doubleRotate_ (N* parent, N* kid)
    {
      // Double rotate is two single rotates: all balances go to zero,
      // except one: The grandkid's balance determines this.
      if (parent->left() == kid) {
	int_4 grandkid_balance = kid->right()->balance;
	singleRotate_(kid, kid->right());
	singleRotate_(parent, parent->left());
	
	switch (grandkid_balance) {
	case +1: kid->balance = -1; break;
	case -1: parent->balance = +1; break;
	}
      } else { 
	int_4 grandkid_balance = kid->left()->balance;
	singleRotate_(kid, kid->left());
	singleRotate_(parent, parent->right());
	
	switch (grandkid_balance) {
	case +1: parent->balance = -1; break;
	case -1: kid->balance = +1; break;
	}
      }
    }
    
    
    // Give me a new node.
    N* newAVLOrdNode_ (N* left, N* right, N* parent, 
		       const K& key, int_u4 keyhash, const V& value)
    {
      // Nothing on the freelist_, so we need to allocate a chunk.
      // The freelist_ iteself is a dummy node, and maintains a
      // circular doubly-linked list (so deletion in the middle of 
      // the list is easy)
      if (freelist_->left_==freelist_) {
	// Big Chunk:  AVLCHUNK nodes
	N* chunk = 0;
	if (allocator_) { 
	  chunk = (N*) allocator_->allocate(sizeof(N)*CHUNKSIZE);
	} else {
	  chunk = (N*)new char[sizeof(N)*CHUNKSIZE];
	}
	// TODO: Allocator
	
	// Put all free nodes on circular doubly-linked list
	for (int ii=0; ii<int(CHUNKSIZE); ii++) {
	  N* node = &chunk[ii];
	  node->on_freelist_ref_count = -ii;
	  // link
	  node->left_ = freelist_;
	  node->right_ = freelist_->right_;
	  freelist_->right_->left_ = node;
	  freelist_->right_ = node;
	}
	// First node special: it holds the ref count for all the
	// nodes in this chunk.
	chunk->on_freelist_ref_count = CHUNKSIZE;
      }
      
      // Assertion: something on freelist.  Take off an item.
      N* node = freelist_->left_;
      node->left_->right_ = node->right_;
      node->right_->left_ = node->left_;
      
      // Management of shared ref count.  Find front of chunk and
      // update refcount
      N* front;
      if (node->on_freelist_ref_count >=0) {
	front = node; 
      } else { 
	front = node + node->on_freelist_ref_count;
      }
      front->on_freelist_ref_count--;
      
      // Construct the new node in place and return it.  To allow
      // specializations for constructing a node in shared memory, we
      // encapsulate this in a function which will be specialized for
      // those things that need allocators.
      return AVLCreateNode(node,left,right,parent,key,keyhash,value,allocator_);
      // return new (node) N(left,right,parent,key,keyhash,value,allocator_); 
    }    
    
    // Return node
    void deleteAVLOrdNode_ (N* node)
    {
      // Destruct
      // deleteMyValue_(node->value);
      node->N::~AVLOrdNode_<K,V>();
      
      // All nodes are allocated in chunks: the front node has a ref
      // count so we know when all elements of a chunk are freed.  We
      // keep the free nodes on a doubly-linked list so they are easy
      // to delete.
      N* front;
      if (node->on_freelist_ref_count >=0) {
	front = node; 
      } else { 
	front = node + node->on_freelist_ref_count;
      }
      
      // If all of chunk pieces on freelist, go ahead and delete this
      // chunk, after we unlink all the free pieces.
      if (++front->on_freelist_ref_count == CHUNKSIZE) {
	for (int ii=0; ii<int(CHUNKSIZE); ii++) {
	  N* inner_node = &front[ii];
	  if (inner_node == node) continue;  // Don't unlink thing just freed!
	  inner_node->left_->right_ = inner_node->right_;
	  inner_node->right_->left_ = inner_node->left_;
	}
	// Delete the chunk after all pieces taken off free list
	char* memory = (char*)front;

	if (allocator_) {
	  allocator_->deallocate(memory);
	} else {
	  delete [] memory;
	}
      }
      
      // Just put the node on the freelist
      else {
	node->left_ = freelist_;
	node->right_ = freelist_->right_;
	freelist_->right_->left_ = node;
	freelist_->right_ = node;
      }
      
    }
    
    
    // Copy (without recursion): To preserve insertion order,
    // we have to iterate in order.
    void copyTable_ (const OrdAVLHashT<K,V,CHUNKSIZE>& rhs) 
    {
      N* dummy = rhs.root_;
      for (N* curr=dummy->ordnext; curr!=dummy; curr=curr->ordnext) {
	this->insertKeyAndValue(curr->key, curr->value);
      }
    }
    
    // root_ and freelist_ are dummy nodes allocated different than
    // all the other nodes (to avoid constructing an extra K and V,
    // and also avoids the K needed a default constructor).
    void helpConstruct_ ()
    {
      if (allocator_) {
	freelist_ = (N*) allocator_->allocate(sizeof(N)*2);
      } else {
	freelist_ = (N*) new char[sizeof(N)*2];
      }

      freelist_->left_ = freelist_->right_ = freelist_;
      freelist_->parent_ = 0;
      
      root_ = &freelist_[1]; // Only 1 allocation
      root_->left_  = root_->right_ = root_->parent_ = root_->next = 0;

      // Ordered list is a dummy header (the dummy root) doubly_linked
      // list.
      root_->ordnext = root_->ordprev = root_; 
    }
    
    // Everything below here is just consistency checking routines
    // builtin
    
    // Not efficient, but exhaustive
    bool testBST_ (N* node, int_u4 hashkey)
    {
      if (node==0) return true;
      bool left = testLeft_(node->left(), hashkey);
      bool right = testRight_(node->right(), hashkey);
      if (!left || !right) {
	return false;
      }
      
      if (node->left())
	left = testBST_(node->left(), node->left()->hashkey);
      if (node->right())
	right = testBST_(node->right(), node->right()->hashkey);
      return left && right;      
    }
    
    bool testLeft_(N* node, int_u4 hashkey)
    {
      if (node==0) return true;
      if (node->hashkey<hashkey) {
	return testLeft_(node->left(), hashkey) && testLeft_(node->right(), hashkey);
      }
      else return false;
    }
    
    bool testRight_(N* node, int_u4 hashkey)
    {
      if (node==0) return true;
      if (node->hashkey>hashkey) {
	return testRight_(node->left(), hashkey) &&testRight_(node->right(), hashkey);
      }
      else return false;
    }
    
    void prefixNode_ (ostream& os, N* node)
    {
      if (node==0) return;
      os << node->key << " ";
      prefixNode_(os, node->left());
      prefixNode_(os, node->right());
    }
    
    void infixNode_ (ostream& os, N* node)
    {
      if (node==0) return;
      infixNode_(os, node->left());
      os << node->key << " ";
      infixNode_(os, node->right());
    }
    
    bool testBalance_(N* node)
    {
      if (node==0) return true;
      int l = height_(node->left());
      int r = height_(node->right());
      if (r-l != node->balance) { 
	cerr << "[" << l << "," << r << " " << node->balance << " key: " << node->key << "]" << endl; 
	return false;
      }
      
      return testBalance_(node->left()) && testBalance_(node->right());
    }
    
    int height_ (N* node)
    {
      if (node==0) return 0;
      int l = height_(node->left());
      int r = height_(node->right());
      if (l>r) return l+1; else return r+1;
    }
    
    bool parentage_ (N* node)
    {
      if (node==0) 
	return true;
      if (node->left()) {
	if (node->left()->parent() != node) 
	  return false;
      }
      if (node->right()) {
	if (node->right()->parent() != node) 
	  return false;
      }
      return parentage_(node->left()) && parentage_(node->right());
    }
    
    size_t recursiveElements_ (N* node) 
    {
      if (node==0) return 0;
      size_t l = recursiveElements_(node->left());
      size_t r = recursiveElements_(node->right());
      size_t n = recursiveElements_(node->next);
      return 1 + l + r + n;
    }
    
    bool threadedOK_ () 
    {
      Array< N* > in_rec, in_for, in_back;
      
      // Build recursive inorder traversal
      inorderRecursive_(root_->right(), in_rec);
      
      // Build iterative inorder traversal
      for (N* n = root_->inorderSucc(); n; n=n->inorderSucc())
	in_for.append(n);
      
      // Build reverse iterative inorder traversal:  harder
      N* nr = root_->right(); 
      if (nr) {
	// Get biggest element ... all the way to the right.
	for (;nr->right(); nr=nr->right())
	  ;
	// Continue with pred from there
	for (; nr; nr=nr->inorderPred())
	  if (nr != root_)
	    in_back.append(nr);
      }
      
      // compare
      if (in_rec.length() != in_for.length() || 
	  in_rec.length() != in_back.length()) {
	cerr << "Arrays have different lengths:" << in_rec << " " 
	     << in_for << " " << in_back << endl;
	return false;
      }
      int in_rec_len = in_rec.length();
      for (int ii=0; ii< in_rec_len; ii++) {
	if (in_rec[ii] != in_for[ii]) 
	  return false;
	if (in_rec[ii] != in_back[in_rec_len-ii-1])
	  return false;
      }
      
      return true;
    }
    
    void inorderRecursive_ (N* node,  Array< N* >& a)
    {
      if (node==0) return;
      inorderRecursive_(node->left(), a);
      a.append(node);
      inorderRecursive_(node->right(), a);
    }
    
    // TODO: Keep this?

    // Take over deletion of values...normally, we don't need to do
    // anything.  The whole purpose of this is to allow the adopting
    // hash table to delete values by overriding this value.  It's
    // not useful in this class when V's are values and not pointers.
    
    // virtual void deleteMyValue_ (V&) { }

    
}; // OrdAVLHashT


template <class K, class V, int_u4 CHUNKSIZE> 
inline void swap (OrdAVLHashT<K,V,CHUNKSIZE>& lhs, 
		  OrdAVLHashT<K,V,CHUNKSIZE>& rhs)
{ lhs.swap(rhs); }

// ////////////////////////////////////// The OrdAVLHashTIterator Class

// This iterates in insertion order
template <class K, class V, int_u4 CHUNKSIZE>
class OrdAVLHashTIterator {
    
  public:
    
    // ///// Methods
    
    // Constructor
    OrdAVLHashTIterator (const OrdAVLHashT<K,V,CHUNKSIZE>& tree) :
      tree_(&tree) { reset(); }
    
    // Advance the iterator one position.  Returns true if the
    // new position is valid, false otherwise.
    bool next () 
    {       
      AVLOrdNode_<K,V>* dummy = tree_->root_;
      current_ = current_->ordnext;
      return current_ != dummy;
    }

    // Syntactic sugar for next 
    bool operator++ () { return next();}
    
    // Advance the iterator one position.  Returns true if the
    // new position is valid, false otherwise.    
    bool operator() () { return next(); }
    
    // Returns the key at the iterator's current position.  The
    // results are undefined if the iterator is no longer valid.
    const K& key () const { return current_->key; }
    
    // Resets the iterator to the state it had immediately after
    // construction.
    void reset () { node_ = current_ = (AVLOrdNode_<K,V>*)(tree_->root_); }
    
    // Resets the iterator to iterate over collection c
    void reset (const OrdAVLHashT<K,V,CHUNKSIZE>& c) { tree_=&c; reset(); }
    
    // Returns the value at the iterator's current position.  The
    // results are undefined if the iterator is no longer valid.
    V& value () const { return current_->value; }

    // Backpointer to container
    OrdAVLHashT<K, V, CHUNKSIZE>* container () const 
    { typedef OrdAVLHashT<K, V, CHUNKSIZE> NCHT; return (NCHT*)tree_; }

  protected:
    
    // ///// Data Members
    
    // The current place in the tree
    AVLOrdNode_<K,V>* current_;
    
    // The current AVL Node (different from current_, which could be
    // an AVL Node or a bucketlist entry).
    AVLOrdNode_<K,V>* node_;
    
    // The AVL tree we are looking at.  Not adopted, just reffed.
    const OrdAVLHashT<K, V, CHUNKSIZE>* tree_;

    // Protected so only friends can see: just so don't have to
    // construct everything.
    OrdAVLHashTIterator () : current_(0), node_(0), tree_(0) { }
}; // OrdAVLHashTIterator


#include "ocsort.h"

// Helper class for Sorted Iteration
template <class K, class V> struct OAVLNR {
  OAVLNR (AVLOrdNode_<K, V>* data=0) : node(data) { }
  void swap (OAVLNR<K,V>& rhs) { OC_NAMESPACED::swap(node, rhs.node); }
  bool operator== (OAVLNR<K,V>a2) const { return node->key==a2.node->key; }
  bool operator> (OAVLNR<K,V>a2) const  { return node->key > a2.node->key; }
  bool operator< (OAVLNR<K,V>a2) const  { return node->key < a2.node->key; }
  AVLOrdNode_<K,V>* node;
};  // OAVLNR<K,V>

template<class K, class V>
void swap(OAVLNR<K,V>& lhs, OAVLNR<K,V>& rhs) { lhs.swap(rhs); } 

// An iterator, sorted via keys
template <class K, class V, int_u4 CHUNKSIZE>
class OrdAVLHashTSortedIterator {
  public:

    OrdAVLHashTSortedIterator (const OrdAVLHashT<K,V,CHUNKSIZE>& tree) :
      keys_(tree.entries()),
	current_(~size_t(0)),
      tree_(&tree)
    { reset(); }

    void reset () 
    { 
      // Iterate and insert keys into table
      current_ = ~size_t(0);
      keys_.clear();
      typedef AVLOrdNode_<K,V> N; 
      
      N* dummy = tree_->root_;
      for (N* curr=dummy->ordnext; curr!=dummy; curr=curr->ordnext) {
	keys_.append(OAVLNR<K,V>(curr));
      }

      OAVLNR<K,V>* keys_ptr = keys_.data();
      OCQuickSort(keys_ptr, 0, keys_.length());
    }
    void reset (const OrdAVLHashT<K,V,CHUNKSIZE>& tree) 
    { tree_ = &tree; reset(); }

    bool next () { return ++current_ < keys_.length(); }
    bool operator() () { return next(); }

    const K& key ()   const { return keys_[current_].node->key; }
          V& value () const { return keys_[current_].node->value; }

    OrdAVLHashT<K,V,CHUNKSIZE>& container () const 
    { typedef OrdAVLHashT<K, V, CHUNKSIZE> NCHT; return (NCHT*)tree_; }

  protected:

    // Keys kept by pointer in an array ... obviously this is 
    // messed up by inserts or deletes into the tree.
    Array<OAVLNR< K, V > > keys_;
    size_t current_;

    // The AVL tree we are looking at.  Not adopted, just reffed.
    const OrdAVLHashT<K, V, CHUNKSIZE>* tree_; 

    // Protected so only friends can access. Just want to avoid lots
    // on default construction
    OrdAVLHashTSortedIterator () : keys_(), current_(0), tree_(0) { }

}; // OrdAVLHashTSortedIterator



// /////////////////// Global Functions

// Handle comparison other than equality correctly
template <class K, class V, int_u4 LEN>
inline bool operator< (const OrdAVLHashT<K,V,LEN>& o1, const OrdAVLHashT<K,V,LEN>& o2)
{
  if (o1.entries()<o2.entries()) return true;
  if (o1.entries()>o2.entries()) return false;
  // Same entries, compare keys and values in order to decide To make
  // sure we compare keys in order, we use insertion order.
  OrdAVLHashTIterator<K,V,LEN> ii(o1);
  OrdAVLHashTIterator<K,V,LEN> jj(o2);
  while (ii() && jj()) {
    const K& k1 = ii.key(); const V& v1 = ii.value();
    const K& k2 = jj.key(); const V& v2 = jj.value();
    if (k1<k2) {
      return true;
    } else if (k1==k2) {
      if (v1<v2) {
	return true;
      } else if (v1==v2) {
	continue;
      } else {
	return false;
      }
    } else {
      return false;
    }
  }
  return false;
}

template <class K, class V, int_u4 LEN>
bool operator<= (const OrdAVLHashT<K,V,LEN>& o1, const OrdAVLHashT<K,V,LEN>&o2)
{ return o1<o2 || o1==o2; }

template <class K, class V, int_u4 LEN>
bool operator> (const OrdAVLHashT<K,V,LEN>& o1, const OrdAVLHashT<K,V,LEN>&o2)
{ return o2<o1; }

template <class K, class V, int_u4 LEN>
bool operator>= (const OrdAVLHashT<K,V,LEN>& o1, const OrdAVLHashT<K,V,LEN>&o2)
{ return o1>o2 || o1==o2; }


OC_END_NAMESPACE

#define OC_ORDAVLHASHT_H_
#endif
