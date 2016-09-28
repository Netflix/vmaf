#ifndef OC_HASHTABLET_H_

// A Hash Table is a table of associated Key-Value pairs.  You insert
// a key-value pair into the table and can retrieve (or delete or
// change) the value in the table by using the key at a later time.
// Both the keys and the values are contained in the table by value
// (so the HashTableT makes a copy of the key and the value).  If you
// insert a key into the table that's already there, the new value
// replaces the old value (thus, there cannot be multiple instances of
// a key in the table.  Of course, the same value can be associated
// with multiple keys).

// NOTES: The bucketList parameter is simply a tuning parameter of
// approximately how many elements you expect: [it really is how many
// buckets lists you want] For larger tables, a bigger value is more
// appropriate.  A good default would be something between 8 and 32.
// Note that the number of buckets is a COMPILE TIME constant, so you
// can't change the size once the table is instantiated.
//
// The HashTable is not thread safe.
//
// Note that the hash functions are "built-in" for speed (the rogue
// wave allows some more flexibility here, but perhaps sacrificing
// some speed).  To compute its hash value, the global function
// "HashFunction(const K& key)" is used.
//
// This class has been slightly rewritten to avoid heap allocations
// except when they are necessary (it tends to group allocations so
// that it does one big allocation rather than many smaller ones).
// This should improve performance, but the interface and
// functionality should NOT have changed at all.
//
//
// HashEntryT_ is an implementation class.  The HashTableT is the
// class to use (and the interface to look at).


// For some example uses of the HashTableT and HashTableTIterator,
// take a look at the "hashtablet_test" in the tests directory.


// ///////////////////////////////////////////// Include Files

#include "ocport.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// Forwards

// Needed so the iterators can properly reference the tables.
template <class K, class V, int_u4 bucketLists>
class HashTableT;


// ///////////////////////////////////////////// The HashEntryT_ class

// An implementation class.  A key-value pair.  Note that these will
// always be constructed with placement new
template <class K, class V>
struct HashEntryT_ {

    // Constructor
    HashEntryT_ (const K& key_value, const V& val, HashEntryT_<K,V>* n) :
      key(key_value), value(val), next(n)
    { }

    // Copy constructor
    HashEntryT_ (const HashEntryT_<K,V>& rhs) :
      key(rhs.key), value(rhs.value), next(0)
    { }

    // The key, value pair
    K key;
    V value;

    // The next entry in the bucket chain
    HashEntryT_<K,V>* next;

}; // HashEntryT_


// ///////////////////////////////////////////// Defines

// Until all compilers support default template parameters, have to
// use a define here.
#define OC_HASHTABLE_BULK 4


// ///////////////////////////////////////////// The HashEntryTBulk_ class

// An implementation class meant to allocate HashEntryT_'s in bulk so
// that we only hit "new" rarely (i.e., we are more "scalable").
// Rather than use MultipleAllocatorT (which uses 2 extra arrays to
// track), the HashEntryTBulk_ uses just an extra pointer per "bulk".
template <class K, class V>
struct HashEntryTBulk_ {

    // A group of HashEntryT_s allocated in "bulk".  Note that this
    // struct is NOT allocated with "new" directly because that would
    // cause a default constructor for each K and V in bulk_ to be
    // constructed.  We don't want to levy this requirement (because
    // it wasn't there before), so this struct serves as a place
    // holder for the memory, and we operate on it with constructors
    // or destructors: Just plain ole memory.
    HashEntryT_<K,V> bulk[OC_HASHTABLE_BULK];

    // A pointer to the next bulk.  The list is NULL teminated.
    HashEntryTBulk_* nextbulk;

};  // HashEntryTBulk_


// ///////////////////////////////////////////// Defines

// Until all compiler support default template arguments, we have to
// define the granularity of allocation (how many buckets each hash
// table allocates at a time).
#define OC_HASHGRANULARITY 4

// ////////////////////////////////////// The HashTableTIterator Class

// Usage:
// HashTableTIterator<K,V,8> it;
// while (it()) {
//   cout << "value=" << it.value() << " key=" << it.key() << endl;
// }

template <class K, class V, int_u4 bucketLists>
class HashTableTIterator {
    
  public:
    
    // ///// Methods
    
    // Constructor    
    HashTableTIterator (const HashTableT<K, V, bucketLists>& table) :
      table_(&table) { reset(); }
    
    // Advance the iterator one position.  Returns true if the
    // new position is valid, false otherwise.
    bool next ()
    {
      // Need to move up lists if at the end of current bucket list,
      // or just starting
      if (currentBucket_ == 0 || currentBucket_->next == 0) {
        return advanceList_();
      // This bucket list still has some entries.
      } else {
        currentBucket_=currentBucket_->next;
        return true;
      }  
    }

    // Syntactic sugar for next 
    bool operator++ () { return next(); }
    
    // Advance the iterator one position.  Returns true if the
    // new position is valid, false otherwise.
    bool operator() () { return ++(*this); }  // reuse op++'s implementation

    // Returns the key at the iterator's current position.  The
    // results are undefined if the iterator is no longer valid.    
    const K& key () const { return currentBucket_->key; }
    
    // Resets the iterator to the state it had immediately after
    // construction
    void reset () { currentBucketList_ = -1; currentBucket_ = 0; }
    
    // Resets the iterator to iterate over collection c
    void reset (const HashTableT<K, V, bucketLists>& c)
    { reset(); table_ = &c; }
    
    // Returns the value at the iterator's current position.  The
    // results are undefined if the iterator is no longer valid.
    V& value () const { return currentBucket_->value; }

    // Backpointer to the container
    HashTableT<K, V, bucketLists>* container () const
    { typedef HashTableT<K, V, bucketLists> NCHT; return (NCHT*)table_; }

  protected:
    
    // ///// Data Members
    
    // The current bucket list.  Note that this HAS to signed
    // so that when we first start, the current bucket list is
    // -1 (so that advanceList_ works correctly).
    int_4 currentBucketList_;

    // The current bucket.  Not adopted ... just referenced.
    HashEntryT_<K, V>* currentBucket_;

    // The Hash table we are looking at.  Not adopted, just reffed.
    const HashTableT<K, V, bucketLists>* table_;

    // ///// Methods

    // Advance to the next non-empty list of buckets.  Note that if
    // the iterator is just contructed that currentBucketList_ is -1,
    // so we automatically increment by 1 to move us up a list in here
    // so that we start at bucket list 0.
    bool advanceList_ ()
    {
      // Go until we find a valid entry, if there is one!
      int blists = bucketLists;
      while (++currentBucketList_ < blists) {

        // Get out if we find a non-empty list
        if (table_->table_[currentBucketList_] != 0) {
          // Otherwise, we found a non-empty list.
          currentBucket_ = table_->table_[currentBucketList_];  
          return true;    
        }
      }
      // Over the edge!
      return false;
    }
    
}; // HashTableTIterator



// ///////////////////////////////////////////// The HashTableT Class

template <class K, class V, int_u4 bucketLists>
class HashTableT {

    // Allow the iterator to see inside
    friend class HashTableTIterator<K, V, bucketLists>;
    
  public:

    // ///// Methods

    // Constructor.  
    HashTableT () :
      freelist_(0),
      bulklist_(0),
      entries_(0)
    {
      // Empty hash table to start with
      int blist = bucketLists;
      for (int i=0; i<blist; i++)
	table_[i] = 0;
    }
    
    // Copy Constructor
    HashTableT (const HashTableT<K, V, bucketLists>& rhs) :
      freelist_(0),
      bulklist_(0)
    {
      copyTable_(rhs);
    }

    // Assignment
    HashTableT<K, V, bucketLists>& operator= (const HashTableT<K, V, bucketLists>& rhs)
    {
      clear();
      copyTable_(rhs);
      return *this;
    }

    // Destructor
    ~HashTableT () 
    {
      clear();
    }

    // Clear the table
    void clear ()
    {
      // Go through each bucket list
      int blists = bucketLists;
      for (int i=0; i<blists; i++) {

	// Go through each ChainBlock
	HashEntryT_<K,V>* current  = table_[i];
	HashEntryT_<K,V>* previous = 0;
	for (; current!=0;) {
	  previous = current;
	  current = current->next;
	  destroy_(previous);
	}

	// ..And clean out first bucket after all buckets cleaned
	table_[i] = 0;
      }

      // Go through everything on bulk list and clear it out.  This
      // deallocats all the memory after all the elements have been
      // destructed.
      HashEntryTBulk_<K,V>* bcurrent = bulklist_;
      HashEntryTBulk_<K,V>* bprevious = 0;
      for (; bcurrent!=0;) {
	bprevious = bcurrent;
	bcurrent = bcurrent->nextbulk;

	// Delete it as it can into the world: as a char array.
	delete [] (char*) bprevious;
      }
      freelist_ = 0;
      bulklist_ = 0;

      entries_ = 0;
    }

    // Returns true is the table contains a key which is equal to key.
    // Returns false otherwise.
    bool contains (const K& key) const { return containsHelper_(key)!=0; }

    // The current number of key-value pairs in the table
    int_u4 entries () const { return entries_; }

    // See if two tables are equal: Two tables are equal if they have
    // the same number of keys and values and all keys and values
    // match.  (NOTE: Yes, this should be a global function, but older
    // Solaris's don't like non-type template params on global
    // functions)
    bool operator== (const HashTableT<K,V,bucketLists>& t1) const
    {
      // Since HashTables can be built in different orders (with
      // collisions happening in different orders), have to iterate
      // through one table and see if they match what's in the other
      // table via lookup.
      if (entries() != t1.entries()) return false;
      for (HashTableTIterator<K,V,bucketLists> it(t1); it(); ) {
	HashEntryT_<K, V>* node = containsHelper_(it.key());
	if (node && it.value()==node->value) continue;
	return false; // single failure, not equal
      }
      return true;
    }
    bool operator!= (const HashTableT<K,V,bucketLists>&t1) const
    { return !(*this==t1); }


    // Returns true is the table contains a key which is equal to
    // "target" and puts the matching key into return_key.  Returns
    // false otherwise and leaves retKey untouched.  Equality is
    // measured by key equality.
    bool find (const K& target, K& return_key) const
    {
      if (containsHelper_(target)) {
	return_key = target;
	return true;
      } else 
	return false;
    }


    // Returns true if the table contains a key which is equal to
    // "key" and puts the associated value into return_val.  Returns
    // false otherwise and leaves return_val untouched.
    bool findValue (const K& key, V& return_val) const
    {
      HashEntryT_<K, V>* temp = containsHelper_(key);
      if (temp) {
	return_val = temp->value;
	return true;
      } else 
	return false;
    }

    // Returns true if the dictionary contains a key which is equal to
    // key and puts the matching key into into return_key and the
    // associated value into return_val.  Returns false otherwise and
    // leaves return_key and return_val untouched.
    bool findKeyAndValue (const K& key, K& return_key, V& return_val) const
    {
      HashEntryT_<K,V>* temp = containsHelper_(key);
      if (temp) {
	return_val = temp->value;
	return_key = temp->key;
	return true;
      } else {
	return false;
      }
    }

    // Inserts the key and value into the table.  If the key is
    // already in there, replace it.
    void insertKeyAndValue (const K& key, const V& value) 
    {
      // If already in there, replace the value.
      int_u4 bucket;
      HashEntryT_<K,V>* entry = containsHelper_(key, bucket);
      if (entry) {
	destroy_(entry);
	create_(entry, key, value, entry->next);
      } else {
	(void)notInTableInsert_(bucket, key, value);
      }
    }
	    
    // Returns true if the table has no items in it, false otherwise.
    bool isEmpty () const { return entries_==0; }

    // Returns true and removes the (key/value) pair where the
    // key is equal to the key.  Returns false if there is no
    // such key.
    bool remove (const K& key) 
    {
      const int_u4 bucket = HashFunction(key) % bucketLists;
      
      // Go through each bucket in this ChainBlock list
      HashEntryT_<K,V>* current  = table_[bucket];
      HashEntryT_<K,V>* previous = 0;
      for (; current!=0;) {
	
	// Is this the one to delete?
	if (key==current->key) {
	 
	  // Yes, but we are at the front of the list!  Unlink from
	  // the front of the list.
	  if (previous==0) {
	    table_[bucket] = current->next;
	  
	  // Yes, in the middle of the list.Unlink the thing before it
	  } else {
	    previous->next = current->next;
	  }

	  // Take it away!
	  deleteHashEntryT_(current);

	  // We know there's only one, because the table will only
	  // insert over old ones.
	  entries_--;
	  return true;
	}
	  
	// Move up the list
	previous = current;
	current = current->next;
      }
      
      // Nope, not in the list.
      return false; 
    }

    // Swap in O(1) time
    void swap (HashTableT<K, V, bucketLists>& rhs) 
    {
      OC_NAMESPACED::swap(table_,    rhs.table_); // Array copy?
      OC_NAMESPACED::swap(freelist_, rhs.freelist_);
      OC_NAMESPACED::swap(bulklist_, rhs.bulklist_);
      OC_NAMESPACED::swap(entries_,  rhs.entries_);
    }


    // Lookup the key "key" and return its associated value as an
    // l-value reference.  If the key is not in the dictionary, then
    // it is added to the dictionary.  In this case, the value
    // associated with the key will be provided by the default
    // constructor ofor objects of types V.
    V& operator[] (const K& key)
    {
      int_u4 bucket;
      HashEntryT_<K,V>* entry = containsHelper_(key, bucket);
      if (entry) {
	return entry->value;
      } else {
	return notInTableInsert_(bucket, key, V());
      }
    }

    // Lookup the key "key" and return its associated value as an
    // l-value reference.  If the key is not in the dictionary, then
    // an out_of_range is thrown
    V& operator() (const K& key) const
    {
      int_u4 bucket;
      HashEntryT_<K,V>* entry = containsHelper_(key, bucket);
      if (entry) {
	return entry->value;
      } else {
	throw out_of_range("Key "+Stringize(key)+" not in table");
      }
    }
    

  protected:

    // ///// Data Members
    
    // An array of pointers to the entries.
    HashEntryT_<K,V>* table_[bucketLists];

    // When a node is deallocated, it is put on the freelist
    // for possible reuse by the HashTableT.  This list is
    // NULL terminated.
    HashEntryT_<K,V>* freelist_;
    
    // Since HashEntryTs are allocated in "bulk", we remember
    // all "bulks" on a linked list so we can return the memory
    // back to the system when a "clear" happens.  
    HashEntryTBulk_<K,V>* bulklist_;

    // The current number of entries actually in the table
    size_t entries_;

    // ///// Methods
    
    // Returns a pointer to the entry if in the table, 0 otherwise.
    // Also returns the bucket by reference so you don't have to
    // rehash.
    HashEntryT_<K,V>* containsHelper_ (const K& key, int_u4& bucket) const
    {
      bucket = HashFunction(key) % bucketLists;
      for (HashEntryT_<K,V>* temp=table_[bucket]; temp; temp=temp->next) {
	if (key==temp->key) {
	  return temp;
	}
      }
      return 0;
    }

    // Returns a pointer to the entry if in the table, 0 otherwise.
    HashEntryT_<K,V>* containsHelper_ (const K& key) const
    {
      int_u4 bucket;
      return containsHelper_(key, bucket);
    }
    
    // A helper function which copies from the lhs to the rhs table.
    // This assumes that the lhs resources have already been
    // freed/reclaimed.
    void copyTable_ (const HashTableT<K,V,bucketLists>& rhs)
    {
      // Same # of entries
      entries_ = rhs.entries_;

      // Go through all bucketLists copying each list there
      int blists = bucketLists;
      for (int i=0; i<blists; i++) {
	
	// If no entries in this bucketlist, no need to copy
	if (rhs.table_[i]==0) {
	  table_[i] = 0;
	  continue;
	}
	
	// Otherwise, the list has some elements, and we need to
	// copy them
    
	// Need to remember the previous so we can link it in
	// correctly to the current copy
	HashEntryT_<K,V>* current_copy = 0;
	HashEntryT_<K,V>* previous_copy = 0;
	
	// Iterate through the right hand side
	for (HashEntryT_<K,V>* temp=rhs.table_[i]; temp; temp=temp->next) {
	  
	  // Copy current entry on rhs 
	  current_copy = newHashEntryT_(temp->key, temp->value, 0);
	  
	  // Put the current_copy in the proper place for new lhs list
	  if (previous_copy==0)
	    table_[i] = current_copy;
	  else
	    previous_copy->next = current_copy;
	  
	  // Remember
	  previous_copy = current_copy;
	} 
	
	// Done copying this particular bucket list
      }
      
      // Done iterating through all buckets
    }


    // This value is not in the table.  Insert it in the given bucket
    // list.
    V& notInTableInsert_ (int_u4 bucket, const K& key, const V& value)
    {
      table_[bucket] = newHashEntryT_(key, value, table_[bucket]);
      entries_++;
      // Return a reference to the value we just set
      return table_[bucket]->value;
    }

    // Create an entry in the given memory
    HashEntryT_<K,V>* create_ (void* memory, const K& key, const V& value, 
			       HashEntryT_<K,V>* next)
    {
      return new (memory) HashEntryT_<K,V>(key, value, next);
    }


    // Moral Equivalent of "new HashEntryT_(key,value,next)"
    HashEntryT_<K,V>* newHashEntryT_ (const K& key, const V& value,
				      HashEntryT_<K,V>* next)
    {
      // Nothing on the free list, so create a new bulk, and link in
      // all the new entries.  Note that we don't use new to allocate
      // it directly because we don't want the default constructors
      // called for the Ks and Vs in the bulk array.
      if (!freelist_) {                
	void* memory = new char[sizeof(HashEntryTBulk_<K,V>)];
	HashEntryTBulk_<K,V>* front = // new HashEntryTBulk_<K,V>;
	  (HashEntryTBulk_<K,V>*) memory;
	
	int ii=0;
	for (ii=0; ii<OC_HASHTABLE_BULK-1; ii++) {
	  front->bulk[ii].next = &(front->bulk[ii+1]);
	}
	front->bulk[ii].next = 0;
	
	// Save this bulk entry
	front->nextbulk = bulklist_;
	bulklist_ = front;
	
	// Remember this for the freelist
	freelist_ = &(front->bulk[0]);
      }

      // Grab front item off the list and chop it off!
      HashEntryT_<K,V>* temp = freelist_;
      freelist_ = freelist_->next;
      return create_(temp, key, value, next);
    }


    // Destruct the given entry without freeing up the memory
    // underneath.  This USED to call "deleteMyValue" as needed.
    void destroy_ (HashEntryT_<K,V>* entry)
    {
      // deleteMyValue_(entry->value);
      entry->HashEntryT_<K,V>::~HashEntryT_();
    }

    // Moral equivalent of "delete HashEntryT_"
    void deleteHashEntryT_ (HashEntryT_<K,V>* entry)
    {
      // Destruct 
      destroy_(entry);

      // Put this item on the free list
      entry->next = freelist_;
      freelist_ = entry;
    }

    
}; // HashTableT


template <class K, class V, int_u4 bucketLists>
inline void swap (HashTableT<K, V, bucketLists>& lhs,
		  HashTableT<K, V, bucketLists>& rhs)
{ lhs.swap(rhs); }


OC_END_NAMESPACE

#define OC_HASHTABLET_H_
#endif


