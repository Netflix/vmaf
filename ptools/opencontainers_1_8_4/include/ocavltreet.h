#ifndef OCAVLTREET_H_

// The AVLTreeT is a class for managing Key/Value pairs, much like
// HashTableT (in fact, you should go look there first).  It has
// almost exactly the same interface (there are a few extra routines).

// Basically, AVLHashT is a plug-in replacement for HashTableT.

// The only difference, besides the implementation, is that AVLTreeT
// Keys needs to support the < operation.

// In general, the HashTableT is good if you have some "expected" idea
// of how many key-value pairs at compile time.  The AVLTreeT is more
// useful if you really expect a lot of variation in the number of
// key-value pairs you have.

// Another useful piece of knowledge: the iterators for AVLTrees
// traverse in alphabetical order: this itself may be the reason you
// choose AVLTreeT over AVLHashes/HashTableT.

// ///////////////////////////////////////////// Includes

#include "ocport.h"
#include "ocarray.h"  // only needed for consistency checks


OC_BEGIN_NAMESPACE


// ///////////////////////////////////////////// The AVLTreeTNode_ struct

// Implementation Detail: Management information needed for each node
// and Key-Value pairs.  Perhaps this should be nested, but some
// compilers have problems with that.

template <class K, class V>
struct AVLTreeTNode_ {
    AVLTreeTNode_ (AVLTreeTNode_* l, AVLTreeTNode_* r, AVLTreeTNode_* p, 
		  const K& k, const V& v):
      left_(l), right_(r), parent_(p), 
      balance(0), 
      // on_freelist_ref_count(0), // DO NOT INITIALIZE!!
      key(k), value(v) 
    { }

    // Pointers to left, right, and parent.  Rather than store NULL
    // pointers for left and right when they are empty, we set the
    // lower bit of the pointer to indicate its empty and use the rest
    // of the pointer to point to the inorder predecessor and inorder
    // succesor (left and right subtrees respectively).  Why don't we
    // just use an extra flag?  (a) save space (b) When you rotate, no
    // need to check anything extra ... just copy the pointer (with or
    // without the bit set) directly.
    AVLTreeTNode_ *left_, *right_, *parent_;

    // Return the left node, or 0 if "empty" (threaded)
    AVLTreeTNode_<K,V>* left() const
    { AVLP t=(AVLP)left_; return t&1 ? 0 : (AVLTreeTNode_<K,V>*)t; }

    // Return the right node, or 0 if "empty" (threaded)
    AVLTreeTNode_<K,V>* right() const
    { AVLP t=(AVLP)right_; return t&1 ? 0: (AVLTreeTNode_<K,V>*)t; } 

    // Return the parent
    AVLTreeTNode_<K,V>* parent() const { return parent_; } 

    // Return the plain pointer, threaded.
    AVLTreeTNode_<K,V>* threadMe () const
    { AVLP t=(AVLP)this; t |= 1; return (AVLTreeTNode_<K,V>*)t; }

    // Return the inorder succesor, or 0 if there is none.
    AVLTreeTNode_<K,V>* inorderSucc() const
    {
      AVLP t=(AVLP)this->right_;
      AVLTreeTNode_<K,V>* r = (AVLTreeTNode_<K,V>*)(t & AVL_EMPTY_MASK);
      if (t && !(t & 1)) { // If not threaded, follow left as far as possible
	for (; r->left(); r=r->left())
	  ;
      }
      return r;
    }

    // Return the inorder predecessor, or 0 if there is none.
    AVLTreeTNode_<K,V>* inorderPred() const
    {
      AVLP t=(AVLP)this->left_;
      AVLTreeTNode_<K,V>* l = (AVLTreeTNode_<K,V>*)(t & AVL_EMPTY_MASK);
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
    // of the later nodes in the array (not the header).  If positive
    // or zero, then this is the header and this is the ref count of
    // how many nodes of this chunk are currently on the freelist.
    int_1 on_freelist_ref_count; 

    // The key and value for this node
    K key;
    V value;

    // Similar to swap, but we want to preserve where the node
    // is in the tree, so we only swap key, value, and
    // insertion order (value related stuff).
    void swapNodeValues (AVLTreeTNode_<K,V>* other)
    {
      // We use swapper because the proper swap for K and V
      // may not be instatiated yet (because OC tends to be 
      // inline-only library and we want to make sure to use
      // the best swap possible for performance
      OC_NAMESPACED::swap(key, other->key);
      OC_NAMESPACED::swap(value, other->value);
    }

    // Some nodes need to do more management after a swap
    void nodeUpdateAfterSwap (AVLTreeTNode_<K,V>* other) { }

};  // AVLTreeTNode_


// ///////////////////////////////////////////// Forwards

template <class K, class V, int_u4 OC_AVLCHUNKSIZE> class AVLTreeTIterator;


// ///////////////////////////////////////////// The AVLTreeT Class

// The OC_AVLCHUNKSIZE is a tuning parameter for how many nodes to
// allocate a time.  We suggest 8.  It should never be more than 127.

template <class K, class V, int_u4 OC_AVLCHUNKSIZE>
class AVLTreeT {

    friend class AVLTreeTIterator<K,V,OC_AVLCHUNKSIZE>;
    enum Found_e { FOUND, NOT_FOUND_LEFT, NOT_FOUND_RIGHT };
    typedef AVLTreeTNode_<K,V> N;

  public:

    // ///// Methods

    // Constructor 
    AVLTreeT () :
      root_(0),
      freelist_(0),
      entries_(0)
    {
      helpConstruct_();
      if (sizeof(root_->right_) != sizeof(AVLP))
	 throw 1; // MidasException("Pointers are wrong size for AVL trees");
    }

    // Copy Constructor
    AVLTreeT (const AVLTreeT<K,V,OC_AVLCHUNKSIZE>& rhs) :
      root_(0),
      freelist_(0),
      entries_(0)
    { 
      helpConstruct_();
      copyTable_(rhs);
    }

    // Assignment
    AVLTreeT<K,V,OC_AVLCHUNKSIZE>& operator= (const AVLTreeT<K,V,OC_AVLCHUNKSIZE>& rhs)
    {
      clear();
      copyTable_(rhs);
      return *this;
    }

    // Destructor
    ~AVLTreeT () 
    { 
      clear(); 

      // root_ and freelist_ special, see helpConstruct_
      char* temp = (char*)freelist_;
      delete [] temp; // Gets both root_ AND freelist_
    }

    // Clear the table
    void clear ()
    {
      // Iterate and destroy: no need for recursion
      N* temp;
      for (N* n = root_->inorderSucc(); n;) {
	temp = n;
	n = n->inorderSucc();
	deleteAVLNode_(temp);
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

    // See if two AVLTreeTs are equal: They are equal if they have the
    // same number of keys and values and all keys and values match.
    // (NOTE: Yes, this should be a global function, but older
    // Solaris's don't like non-type template params on global functions)
    bool operator== (const AVLTreeT<K,V,OC_AVLCHUNKSIZE>& t1) const 
    {
      // AVLTrees can have different structures, but because they are
      // sorted, we iterate through them the same.
      if (t1.entries() != entries()) return false;
      AVLTreeTIterator<K,V,OC_AVLCHUNKSIZE> it1(t1), it2(*this);
      while (it1() && it2()) {
	if (it1.key()==it2.key() && it1.value()==it2.value()) continue;
	return false;
      }
      return true;
    }
    bool operator!= (const AVLTreeT<K,V,OC_AVLCHUNKSIZE>& t1) const
    { return !(t1==*this); }


    // Returns true is the table contains a key which is equal to
    // "target" and puts the matching key into return_key.  Returns
    // false otherwise and leaves retKey untouched.  Equality is
    // measured by key equality.
    bool find (const K& target, K& return_key) const
    {
      Found_e found_where; (void)lookup_(target, found_where);
      if (found_where==FOUND) {
	return_key = target;
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
	return_key = key;
	return true;
      } else {
	return false;
      }
    }

    // Inserts the key and value into the table.  If the key is
    // already in there, replace it.
    void insertKeyAndValue (const K& key, const V& value)
    {
      Found_e found_where; N* node = lookup_(key, found_where);
      if (found_where==FOUND) {
	node->value = value; // Assumes the op= for V works
      } else {
	(void)notInTableInsert_(node, key, value, found_where);
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
    // constructor ofor objects of types V.
    V& operator[] (const K& key)
    {
      Found_e found_where; N* node = lookup_(key, found_where);
      if (found_where==FOUND) {
	return node->value;
      } else {
	return notInTableInsert_(node, key, V(), found_where)->value;
      }
    }

    // Lookup the key "key" and return its associated value as an
    // l-value reference.  If the key is not in the dictionary, then
    // an out_of_range is thrown
    V& operator() (const K& key) const
    {
      Found_e found_where; N* node = lookup_(key, found_where);
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
      if (!testBST_(root_->right(), root_->right()->key))
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

    void swap (AVLTreeT<K,V,OC_AVLCHUNKSIZE>& rhs)
    {
      OC_NAMESPACED::swap(root_, rhs.root_);
      OC_NAMESPACED::swap(freelist_, rhs.freelist_);
      OC_NAMESPACED::swap(entries_, rhs.entries_);
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
	N* current = notInTableInsert_(node, K(), V(), found_where);
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

    // ///// Methods

    // Helper function used by a lot of methods above: Do a lookup and
    // see if the item is already there: if it is, return pointer to
    // it and indicate FOUND.  If not there, return a pointer to its
    // parent (as if it WERE there) and indicate whether the child
    // would be on the left or the right.
    N* lookup_ (const K& key, Found_e& found_where) const
    {
      // It's just a binary search tree: look left is less than, right
      // otherwise.  IF WE DON'T FIND it, then we return the parent
      // (so we don't have to do another lookup).
      N* prev = root_;
      N* current = prev->right();
      Found_e found = NOT_FOUND_RIGHT;
      while (current) {
	const K& current_key = current->key;
	prev = current;
	if (key<current_key) {
	  current = current->left();
	  found = NOT_FOUND_LEFT;
	  continue;
	} else if (current_key==key) {
	  found = FOUND;
	  break;
	} else {
	  current = current->right();
	  found = NOT_FOUND_RIGHT;
	  continue;
	}
      }
      // Return parent if not found in tree
      found_where = found;
      return prev;
    }

    // Insert an item into the table: The parent is the node above us
    // that adopts us!  The key and value get copied into the new
    // node, and the found_where indicates whether the parent should
    // point us at the left or the right.
    N* notInTableInsert_ (N* parent, const K& key, const V& value,
			  Found_e found_where)
    {
      // Insert new node in the right place: When we do this, we also
      // thread the tree: Note that we use parent->left and
      // parent->right directly to preserve the threaded bit.
      N* current;
      N* threaded_dad = parent->threadMe();
      if (found_where==NOT_FOUND_LEFT) {
	current = newAVLNode_(parent->left_,threaded_dad,parent,key,value);
	parent->left_ = current; // left not threaded anymore
      } else {
	current = newAVLNode_(threaded_dad,parent->right_,parent,key,value);
	parent->right_ = current; // right not threaded anymore
      }
      N* ret_val = current;
      entries_++;

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
      deleteAVLNode_(current); // Give "back" to memory system
    }


    // Delete current from the tree
    void deleteItemFromTable_ (N* current)
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
	//current->key = inorder_succ->key;
	//current->value = inorder_succ->value; // swap is faster than op=
	current->swapNodeValues(inorder_succ); // swap is faster than op=
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
	// throw MidasException("Internal Consistency error of AVL Tree");
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
    

    // Give me a new node: Take a node off the freelist if you can (to
    // avoid going to the heap), otherwise go to the heap and get
    // OC_AVLCHUNKSIZE nodes at one time (instead of just 1).
    N* newAVLNode_ (N* left, N* right, N* parent, const K& key, const V& value) 
    {
      // Nothing on the freelist_, so we need to allocate a chunk.
      // The freelist_ iteself is a dummy node, and maintains a
      // circular doubly-linked list (so deletion in the middle of 
      // the list is easy)
      if (freelist_->left_==freelist_) {
	// Big Chunk:  OC_AVLCHUNK nodes in one allocation
	N* chunk = (N*)new char[sizeof(N)*OC_AVLCHUNKSIZE];
	// Put all free nodes on circular doubly-linked list
	for (int_u4 ii=0; ii<OC_AVLCHUNKSIZE; ii++) {
	  N* node = &chunk[ii];
	  node->on_freelist_ref_count = -(signed(ii));
	  // link
	  node->left_ = freelist_;
	  node->right_ = freelist_->right_;
	  freelist_->right_->left_ = node;
	  freelist_->right_ = node;
	}
	// First node special: it holds the ref count for all the
	// nodes in this chunk.
	chunk->on_freelist_ref_count = OC_AVLCHUNKSIZE;
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
      
      // Construct the new node in place and return it
      return new (node) N(left, right, parent, key, value); 
    }


    // Return node
    void deleteAVLNode_ (N* node)
    {
      // Destruct
      node->N::~N();

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
      if (++front->on_freelist_ref_count == OC_AVLCHUNKSIZE) {
	for (int_u4 ii=0; ii<OC_AVLCHUNKSIZE; ii++) {
	  N* inner_node = &front[ii];
	  if (inner_node == node) continue;  // Don't unlink thing just freed!
	  inner_node->left_->right_ = inner_node->right_;
	  inner_node->right_->left_ = inner_node->left_;
	}
	// Delete the chunk after all pieces taken off free list
	char* memory = (char*)front;
	delete [] memory;
      }

      // Just put the node on the freelist
      else {
	node->left_ = freelist_;
	node->right_ = freelist_->right_;
	freelist_->right_->left_ = node;
	freelist_->right_ = node;
      }
    }

    // Copy (without recursion)
    void copyTable_ (const AVLTreeT<K,V,OC_AVLCHUNKSIZE>& rhs) 
    {
      // Iterate and insert
      for (N* n = rhs.root_->inorderSucc(); n; n=n->inorderSucc())
	insertKeyAndValue(n->key, n->value);
    }

    // root_ and freelist_ are dummy nodes allocated different than
    // all the other nodes (to avoid constructing an extra K and V,
    // and also avoids the K needed a default constructor).
    void helpConstruct_ ()
    {
      freelist_ = (N*) new char[sizeof(N)*2];
      freelist_->left_ = freelist_->right_ = freelist_;
      freelist_->parent_ = 0;
      
      root_ = &freelist_[1]; // Only 1 allocation
      root_->left_  = root_->right_ = root_->parent_ = 0;
    }

    // Everything below here is just consistency checking routines
    // builtin

    // Not efficient, but exhaustive
    bool testBST_ (N* node, const K& key)
    {
      if (node==0) return true;
      bool left = testLeft_(node->left(), key);
      bool right = testRight_(node->right(), key);
      if (!left || !right) {
	return false;
      }

      if (node->left())
	left = testBST_(node->left(), node->left()->key);
      if (node->right())
	right = testBST_(node->right(), node->right()->key);
      return left && right;      
    }

    bool testLeft_(N* node, const K& key)
    {
      if (node==0) return true;
      if (node->key<key) {
	return testLeft_(node->left(), key) && testLeft_(node->right(), key);
      }
      else return false;
    }

    bool testRight_(N* node, const K& key)
    {
      if (node==0) return true;
      if (node->key>key) {
	return testRight_(node->left(), key) &&testRight_(node->right(), key);
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
	// cerr << "[" << l << "," << r << " " << node->balance << " key: " << node->key << "]" << endl; 
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
      return 
	1+recursiveElements_(node->left())+recursiveElements_(node->right());
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
	// cerr << "Arrays have different lengths:" << in_rec << " " 
	//     << in_for << " " << in_back << endl;
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
    
    
}; // AVLTreeT

template <class K, class V, int_u4 OC_AVLCHUNKSIZE>
inline void swap (AVLTreeT<K,V,OC_AVLCHUNKSIZE>& lhs, 
		  AVLTreeT<K,V,OC_AVLCHUNKSIZE>& rhs) 
{ lhs.swap(rhs); }

// ////////////////////////////////////// The AVLTreeTIterator Class

template <class K, class V, int_u4 OC_AVLCHUNKSIZE>
class AVLTreeTIterator {
    
  public:
    
    // ///// Methods
    
    // Constructor
    AVLTreeTIterator (const AVLTreeT<K, V, OC_AVLCHUNKSIZE>& tree) :
      tree_(&tree) { reset(); }
    
    // Advance the iterator one position.  Returns true if the
    // new position is valid, false otherwise.
    bool next () { return (current_=current_->inorderSucc())!=0; }

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
    void reset () { current_ = (AVLTreeTNode_<K,V>*)(tree_->root_); }
    
    // Resets the iterator to iterate over collection c
    void reset (const AVLTreeT<K,V,OC_AVLCHUNKSIZE>& c) { tree_=&c; reset(); }
    
    // Returns the value at the iterator's current position.  The
    // results are undefined if the iterator is no longer valid.
    V& value () const { return current_->value; }

    // Backpointer to container
    AVLTreeT<K, V, OC_AVLCHUNKSIZE>* container () const 
    { typedef AVLTreeT<K, V, OC_AVLCHUNKSIZE> NCHT; return (NCHT*)tree_; }    

  protected:
    
    // ///// Data Members

    // The current place in the tree
    AVLTreeTNode_<K,V>* current_;

    // The AVL tree we are looking at.  Not adopted, just reffed.
    const AVLTreeT<K,V,OC_AVLCHUNKSIZE>* tree_;

}; // AVLTreeTIterator



// /////////////////// Global Functions

// Handle comparison other than equality correctly
template <class K, class V, int_u4 LEN>
inline bool operator< (const AVLTreeT<K,V,LEN>& o1, const AVLTreeT<K,V,LEN>& o2)
{
  if (o1.entries()<o2.entries()) return true;
  if (o1.entries()>o2.entries()) return false;
  // Same entries, compare keys and values in order to decide To make
  // sure we compare keys in order, we use insertion order.
  AVLTreeTIterator<K,V,LEN> ii(o1);
  AVLTreeTIterator<K,V,LEN> jj(o2);
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
bool operator<= (const AVLTreeT<K,V,LEN>& o1, const AVLTreeT<K,V,LEN>&o2)
{ return o1<o2 || o1==o2; }

template <class K, class V, int_u4 LEN>
bool operator> (const AVLTreeT<K,V,LEN>& o1, const AVLTreeT<K,V,LEN>&o2)
{ return o2<o1; }

template <class K, class V, int_u4 LEN>
bool operator>= (const AVLTreeT<K,V,LEN>& o1, const AVLTreeT<K,V,LEN>&o2)
{ return o1>o2 || o1==o2; }


OC_END_NAMESPACE

#define OCAVLTREET_H_
#endif
