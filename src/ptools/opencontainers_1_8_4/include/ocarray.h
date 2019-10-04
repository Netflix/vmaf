#ifndef OCARRAY_H_

// A completely inlineable Array/ArrayPtr class.  These two classes
// essentially implement a sub set of the the Roguewave
// RWTValOrderedVector/RWTPtrOrderedVector.

// How to use the Array class: It is what you'd expect: A wrapper for
// C-style arrays with some error checking (if requested) with
// built-in management for resizing of the array.  It contains things
// by value.  In order for Array to contain elements of type T, T
// needs to support: (1) a copy constructor (2) an operator= (3) an
// operator== (4) a default constructor.

// How to use the ArrayPtr class: It is what you'd expect: A wrapper
// for C-style arrays where you contain things by pointer.
// In order to use ArrayPtr with your type T, T needs to support
// (1) op== only  

// One thing to note about how both Array/ArrayPtr manages memory:
// they actually holds an array of some CAPACITY (which is always
// greater than or equal to the actual LENGTH) number of elements.
// The elements are NOT constructed, but there is enough memory to
// hold them when they need to be constructed.

// ////////////////////////////////////////////////// Include Files

#include "ocport.h"          // Handles portability issues
#include "ocstreamingpool.h" // Allocator

// Intel compiler workaround (10.0.12) for overly aggressive template
// instantiations for complex_8 and complex_16 comparisons.
#if defined(__ICC)
#  if __ICC==1010 && __INTEL_COMPILER_BUILD_DATE==20080112 // LAME! Can't find 10.0.12 anywhere in INTEL's defines, only if build date???
#  include "occomplex.h" 
#  endif
#endif

#include "ocmove.h"  // Allows optimized resizes

OC_BEGIN_NAMESPACE

///////////////////////////////////////////// The Array Class

template <class T>
class Array {

  public:

    // Why a default of 2?  Well these classes are used everywhere,
    // and frequently they are left completely empty.  A significant
    // amount of memory gets wasted to store nothing.
    enum {ARRAY_DEFAULT_CAPACITY = 2, ARRAY_NPOS = ~0};

  
    // ///// Methods

    // Constructor.  Create an empty ordered vector with capacity
    // capac.  Should the number of items inserted in exceed this
    // value, the vector will be resized autotmatically.  By default,
    // all memory allocation/deallocations will be done with new and
    // delete, but you can specify that the allocations/deallocations
    // will be done with malloc and free if necessary.
    Array (size_t capac = ARRAY_DEFAULT_CAPACITY, int use_new_and_delete = 1) :
      allocator_(reinterpret_cast<Allocator*>(int_uptr(use_new_and_delete))),
      length_(0),
      capac_(capac),
      data_(allocate_(capac))  // dependency ... useMallocAndFree needs to be initialized before allocate_ is called
    { }


    // Array that can use a shared memory allocator
    Array (size_t capac, Allocator* a) :
      allocator_(a!=0 ? a : reinterpret_cast<Allocator*>(int_uptr(1))),
      length_(0),
      capac_(capac),
      data_(allocate_(capac))  // dependency ... allocator needs to be initialized before allocate_ is called
    { }


    // Array that can "refer" to other memory: BE CAREFUL!  This is mostly
    // useful for raw POD data and memory where things are already constructed.
    Array (size_t capac, T* data) :
      allocator_(reinterpret_cast<Allocator*>(int_uptr(3))),
      length_(capac),
      capac_(capac),
      data_(data)  
    { }

    // Copy constructor.  Constructs a new order vector as a copy of
    // c.  The copy constructor of all elements will be called.  The
    // new array will have the same capacity and number of elements as
    // the old array.
    Array (const Array<T>& c, Allocator*a = 0) : 
      allocator_(a!=0 ? a : reinterpret_cast<Allocator*>(int_uptr(1))),
      length_(c.length_),
      capac_(c.capac_),
      data_(allocate_(c.capacity()))
    {
      // Iterate through all legal elements, calling copy constructor.
      size_t ii;
      size_t len = length();
      Allocator *myalloc = allocator();
      for (ii=0; ii<len; ii++) {
	//(void)new (&data_[ii]) T(c[ii]);
	PlaceCopyCons_(&data_[ii], c[ii], myalloc);
      }
    }


    // operator=.  Sets self to a copy of rhs. The new array will have
    // the same capacity and number of elements as the old array.  The
    // copy constructor of all elements will be called.  The new array
    // will have the same capacity and number of elements as the old
    // array.
    Array<T>& operator= (const Array<T>& rhs)
    {
      // Catch the assignment to myself
      if (&rhs!=this) {
     
	// Release lhs resources and then copy rhs resources.
	releaseResources_();
	//(void)new (this) Array<T>(rhs);
	PlaceCopyCons_(this, rhs, allocator());
      }
      return *this;
    }


    // Destructor.  Release resources.
    ~Array () { releaseResources_(); }


    // Returns the ith value in the array.  The first variant can be
    // used as an lvalue, the second cannot.  The index i must be
    // between zero and the number of items in the collection less 1.
    // No bounds checking is performed.
    T& operator() (size_t i)             { return data_[i]; }
    const T& operator() (size_t i) const { return data_[i]; }


    // Returns the ith value in the array.  The first variant can be
    // used as an lvalue, the second cannot.  The index i must be
    // between zero and the number of items in the collection less 1,
    // or an Exception of Type MidasException will be thrown.
    T& operator[] (size_t i) { return at(i); }
    const T& operator[] (size_t i) const { return at(i); }

    // Adopt memory allocated via new [] T(): This releases the old
    // memory and sets the new memory, and capacity to cap: the length
    // is set to 0.
    void adoptMemory (T* adopt_me, size_t cap, int use_new_and_delete=1)
    {
      releaseResources_();
      allocator_ = reinterpret_cast<Allocator*>(int_uptr(use_new_and_delete));
      length_ = 0;
      capac_ = cap;
      data_ = adopt_me;
    }
    
    // Return the allocator being used, 0 if none
    Allocator* allocator () const
    {
      int_uptr memory_policy = int_uptr(allocator_);
      if (memory_policy<=3) {
	return 0;
      } else {
	return allocator_;
      }
    }

    // Appends the value a to the end of the array.  The collection
    // will automatically be resized if this causes the number of
    // items in the collection to exceed the capacity.
    void append (const T& a) 
    {
      if (capacity()==length_)
	resize(2*capacity());
      //(void)new (&data_[length_]) T(a);
      PlaceCopyCons_(&data_[length_], a, allocator());
      length_++;
    }


    // Returns the ith value in the array.  The first variant can be
    // used as an lvalue, the second cannot.  The index i must be
    // between zero and the number of items in the collection less 1,
    // or an Exception of Type MidasException will be thrown.
    T& at (size_t i) 
    { if (i>=length()) arrayError_(i); return data_[i]; }
    const T& at (size_t i) const 
    { if (i>=length()) arrayError_(i); return data_[i]; }


    // Removes all items from the collection.  The clear() method
    // calls the destructor on each item.  If the Array is holding a
    // built-in type or a simple struct that need not be destructed,
    // clearFast() clears the items without calling the destructor on
    // each one.  Caveat emptor when using clearFast.
    void clear () 
    {
      size_t len = length();
      for (size_t ii=0; ii<len; ii++)
	(&data_[ii])->~T();
      length_ = 0;
    }
    void clearFast () { length_=0; }


    // Returns true if the collection contains an item equal to a.  A
    // linear search is done.  Equality is measured by T's operator==.
    bool contains (const T& a) const 
    {
      size_t len = length();
      for (size_t ii=0; ii<len; ii++)
	if (a==data_[ii])
	  return true;
      return false;
    }

    
    // Returns a pointer to the raw data of the array. Should be used
    // with care.
    T* data () { return data_; }
    const T* data () const { return data_; }  


    // Returns the number of items currently in the collection.
    size_t entries () const { return length_; }

    // Performs a linear search, returning the index of the first item
    // that is equal to a.  Returns ARRAY_NPOS if there is no such
    // item.  Equality is measured by T's operator==.
    size_t index (const T& a) const
    {
      size_t len = length();
      for (size_t ii=0; ii<len; ii++) {
	if (a==data_[ii]) {
	  return ii;
	}
      }
      return ARRAY_NPOS;
    }

    // Perform append of the given item n times.  If n is not given,
    // then the array is filled to capacity()-length() (in other
    // words, the rest of the capacity is filled up).  This routine is
    // frequently called after the constructor to fill the array with
    // a known value
    void fill (const T& item, size_t n)
    {
      for (size_t ii=0; ii<n; ii++) {
	this->append(item);
      }
    }
    void fill (const T& item) { this->fill(item, capacity()-length()); }

    // Performs a linear search and return trues if the array contains
    // an object that is equal to the object target and puts a copy of
    // the matching object into ret.  Returns false otherwise and does
    // not touch ret.  Equality is determined by T's operator==.
    bool find (const T& target, T& ret) const
    {
      size_t ii=index(target);
      if (ii==size_t(ARRAY_NPOS))
	return false;
      else {
	ret = data_[ii];
	return true;
      }
    }


    // Appends the value of a to the end of the array.  The collection
    // will automatically be resized if this causes the number of
    // items in the collection to exceed the capacity.
    void insert (const T& a) { append(a); }

    // Inserts the value a into the array at index i.  The item
    // previously at position i is moved to i+1, etc.  The collection
    // will automatically be resized if this causes the number of
    // items in the collection to exceed the capacity.  The index i
    // must be between 0 and the number of items in the array or an
    // exception of type out_of_range will be thrown.
    void insertAt (size_t i, const T& a)
    {
      if (i>length())
	arrayError_(i);
      if (length_==capacity())
	resize(2*capacity());
      
      // At this point, you know that the point to insert is valid and
      // that there is enough space to move everything over.
      
      // Default construct into the last space so it actually has been
      // constructed.  (right now, it's only guaranteed to have
      // memory, it has NOT been constructed).
      //(void)new (&data_[length()]) T();
      PlaceCopyCons_(&data_[length()], T(), allocator());

      for (size_t jj=length(); jj>i; jj--)
	data_[jj] = data_[jj-1];
      data_[i] = a;
      length_++;
    }

    // Returns true if there are no items in the collection,
    // false otherwise
    bool isEmpty () const { return length_==0; }

    // Returns the number of items currently in the collection.
    size_t length () const { return length_; }

    // Prepends the value a to the beginning of the array.  The
    // collection will automatically be resized if this causes the
    // number of items in the collection to exceed the capacity.
    void prepend (const T& a) { insertAt(0, a); }


    // Performs a linear search, removing the first object which is
    // equal to the object and returns true.  Returns false if there
    // is no such object.  Equality is measure by operator== for T.
    bool remove (const T& val)
    {
      size_t len = length();
      for (size_t ii=0; ii<len; ii++)
	if (val==data_[ii]) {
	  (void)removeAt(ii);
	  return true;
	}
      return false;
    }

    // Removes all items which are equal to val, returning the number
    // removed.
    size_t removeAll (const T& val) 
    {
      size_t count = 0;
      if (length()) {
	size_t ii=length()-1;   // Removing from end first faster than front
	while (1) {
	  if (val==data_[ii]) {
	    removeAt(ii);
	    count++;
	  }
	  if (ii==0) break;
	  ii--;
	}
      }
      return count;
    }


    // Removes and returns the object at index i.  The array has to
    // shift down (by one) all the items above it to perform the
    // removal. An exception of type out_of_range will be thrown if the
    // list is empty.
    T removeAt (size_t i) 
    {
      if (i>=length())
	arrayError_(i); // Note:  This always throws an exception

      // Otherwise, things are okay
      T ret_val = data_[i];
    
      // Move em over
      size_t jj=i; size_t len_minus_1=length()-1;
      for (; jj<len_minus_1; jj++)
	data_[jj] = data_[jj+1];

      (&data_[jj])->~T(); // Destruct the last
      length_--;          // Shrink
      
      return ret_val;  // All done, return ret_val
    }
    

    // Removes, starting at position i, the next run_length objects.
    // Note that we have to move all items above down by run_length.
    // This operation is essentially run_length removeAt operations,
    // except that this is more efficient.  If, during the process, we
    // run off the end of the array, an exception of type out_of_range
    // will be thrown.
    void removeRange (size_t i, size_t run_length)
    {
      // Three checks to avoid accidental -1s
      size_t len = length();
      if (i>=len || i+run_length>len || run_length>len) 
	arrayError_(i); // Note:  This always throws an exception

      // Otherwise, things are okay: Move em over (
      size_t jj=i;
      for (; jj+run_length<len; jj++) {
	data_[jj] = data_[jj+run_length];
      }
      
      // Destruct last bit of objects
      for (size_t kk=0; kk<run_length; kk++) {
	(&data_[jj+kk])->~T(); // Destruct the last run_length
      }
      
      // And shrink
      length_-= run_length;         
    }


    // Removes and returns the first object in the collection.  An
    // exception of type out_of_range will be thrown if the array is
    // empty.
    T removeFirst () { return removeAt(0); }

    // Removes and returns the last object in the collection.  An
    // exception of type out_of_range will be thrown if the array is
    // empty.
    T removeLast () { return removeAt(length_==0 ? 0 : length_-1); }

    // Changes the capacity of the collection to n.  Note that the
    // number of objects in the collection does not change, just the
    // capacity.    
    void resize (size_t new_capacity)
    {
      // Zero capac becomes 1
      if (!new_capacity)
	new_capacity = 1;

      // The new capacity has to be at least as big as the length.  If
      // not, no need to do any work.
      if (new_capacity<=length()) {
	size_t bit = getBit();
	capac_ = length();
	setBit(bit);
	return;
      }
      
      T* new_data = allocate_(new_capacity);

      // Copy the real elements first
      size_t ii=0;
      size_t len = length();
      bool run_destructors = true;
      Allocator* a = allocator();
      if (a) {
	for (; ii<len; ii++) {
	  PlaceCopyCons_(&new_data[ii], data_[ii], a);
	}
	run_destructors = true;
      } else {
	run_destructors = MoveArray(&data_[ii], &new_data[ii], len);
      }

      // Clean up old resources and install new resources.
      releaseResources_(run_destructors);
      data_ = new_data;

      // Set new capacity, but preserve the upper bit
      size_t bit = getBit();
      capac_ = new_capacity;
      setBit(bit);
    }

    // Swap with another Array: This is a O(1) operation since we only
    // swap pointers (no copying of full arrays).
    void swap (Array<T>& rhs) 
    {
      OC_NAMESPACED::swap(allocator_, rhs.allocator_);
      OC_NAMESPACED::swap(length_, rhs.length_);
      OC_NAMESPACED::swap(capac_,  rhs.capac_);
      OC_NAMESPACED::swap(data_,   rhs.data_);
    }

    // Inspect the current capacity (useful if you decide you want to
    // resize or not)
    size_t capacity () const
    { return capac_ & ~(size_t(1) << ((sizeof(size_t)<<3)-1)); } // Upper bit is useful
    
    // Expand the array to the given size.  Caveat Emptor when using
    // this method, as it will NOT call the destructors when setting
    // the length.  This is really most useful when using POD (data
    // with out constructors/destructors).
    void expandTo (size_t l) 
    {
      if (l>capacity()) {
	resize(l);
      }
      length_ = l;
    }

    // Expand BY 
    void expandBy (int_8 by)
    {
      int_8 len = length_;
      int_8 upto = len + by;
      size_t supto = upto;
      expandTo(supto);
    }

    // Users are allowed 1-bit of information to save/restore.  This
    // bit propagates to other copies.
    size_t getBit () const { return capac_ >> ((sizeof(size_t)<<3)-1); }
    void setBit (size_t bit)
    {
      if (bit) {
	capac_ |= (bit << ((sizeof(size_t)<<3)-1));
      } else {
	capac_ &= ~(size_t(1) << ((sizeof(size_t)<<3)-1));
      }
    }
    
  protected:

    // ///// Data Members

    // The allocator if we want to allocate things in shared memory.
    // Also indicates whether you want to use malloc/free or op new/op
    // delete for managing memory.
    //
    // 0 means malloc/free
    // 1 means operator new/operator delete
    // 2 means new T[]
    // 3 means "we don't adopt: just referring" (deallocate does nothing, 
    //                                           allocate errs out)
    //               
    // Anything else, it's an allocator.
    // We have to be careful and ONLY pass legal allocators to
    // other routines!!
    Allocator* allocator_;

    // The number of items currently in the array.
    size_t length_;

    // The number of spaces allocated in the array.  This is
    // necessarily larger than length.  NOTE: We reserve the upper bit
    // to allow a user to set/get a bit.
    size_t capac_;

    // The pointer to the actual data
    T* data_;

    // ///// Methods

    // Centralize the throw out_of_range 
    void arrayError_ (size_t i) const
    { 
      throw out_of_range("array: attempting to access element:"+
			 Stringize(i)+" in array of length:"+
			 Stringize(length()));
      // throw out_of_range("array");
    }

    // Allocate the memory resources.  This does not call the
    // constructors
    T* allocate_ (size_t amt) const 
    {
      if (amt==0 && capacity()==0) return 0;

      // If allocator is a small int (0,1,2), special case
      int_uptr memory_policy = int_uptr(allocator_);
      if (memory_policy==0) {
	return (T*) malloc(sizeof(T)*amt);
      } else if (memory_policy==1) {
	return (T*) operator new(sizeof(T)*amt);
      } else if (memory_policy==2) {
	return (T*) operator new[](sizeof(T)*amt);
      } else if (memory_policy==3) {
	throw runtime_error("Array Policy 3 only refers to the data inside: it cn't allocate more data!");
      } else {
	// Real allocator .. use it!
	return (T*) allocator_->allocate(sizeof(T)*amt);
      }
    }

    // Deleting the resources.  This involves calling the destructor
    // on each legal element.
    void releaseResources_ (bool run_destructors=true)
    {
      if (length()==0 && data_==0) return;

      // Force destructor call on this memory
      size_t len = length();
      if (run_destructors) {
	for (size_t ii=0; ii<len; ii++)
	  (&data_[ii])->~T();
      }

      // Delete the memory
      int_uptr memory_policy = int_uptr(allocator_);
      if (memory_policy==0) {
	free(data_);
      } else if (memory_policy==1) {
	operator delete(data_);
      } else if (memory_policy==2) {
	operator delete[](data_);
      } else if (memory_policy==3) {
	// Do nothing: Array is a thin wrapper around some memory
      } else {
	allocator_->deallocate((char*)data_);
      }
    }

   
}; // Array

// Special functions for dispatching

///////////////////////////////////////////////////// Global Functions

// Chose the proper inplace construction
// Default, most classes don't use an allocator for placement new construct
template <class T>
inline void PlaceCopyCons_ (void* location, const T& copy, Allocator*)
{
  new (location) T(copy);  // default doesn't support allocator 
} 

// If using an array, they support an allocator!
template <class T>
inline void PlaceCopyCons_ (void* location, const Array<T>& copy, Allocator* a)
{ 
  new (location) Array<T>(copy, a); 
}


// ///////////////////////////////////////////// The ArrayPtr Class

// The ArrayPtr: essentially a drop-in replacement for the Roguewave
// RWTPtrOrderedVector class.  Most of the implementation comes from
// the Array class, but we are templatized on T* rather than T.
template <class T>
class ArrayPtr {

  public:

    // Why a default of 2?  Well these classes are used everywhere,
    // and frequently they are left completely empty.  A significant
    // amount of memory gets wasted to store nothing.
    enum {ARRAY_DEFAULT_CAPACITY = 2, ARRAY_NPOS = ~0};

    // ///// Methods

    // Constructor.  Constructs an empty ordered vector with capacity
    // capac.  Should the number of items exceed this value, the
    // vector will ber resized automatically. By default, all memory
    // allocation/deallocations will be done with new and delete, but
    // you can specify that the allocations/deallocations will be done
    // with malloc and free if necessary.
    ArrayPtr (size_t capac=ARRAY_DEFAULT_CAPACITY, int use_new_and_delete = 1):
      a_(capac, use_new_and_delete)
    { }

    // Make sure has an allocator 
    ArrayPtr (size_t capac, Allocator* alloc) :
      a_(capac, alloc)
    { }

    // Copy constructor.  Constructs self to be a shallow copy of
    // c.  Afterwards, pointers will be shared between the two
    // collections.
    ArrayPtr (const ArrayPtr<T>& c) :
      a_(c.a_)
    { }
    
    // Destructor.  Removes all pointers but does not delete what they
    // point to.
    ~ArrayPtr () { }


    // Operator=.  Constructs self to be a shallow copy of c.
    // Afterwards, pointers will be shared between the two
    // collections.
    ArrayPtr<T>& operator= (const ArrayPtr<T>& c) { a_=c.a_; return *this; }


    // Returns a pointer to the ith value of the array.  The first
    // variant can be used as an l-value, the second cannot.  The
    // index i must be between zero and the number of items in the
    // collection less one.  No bounds checking is performed.
    T*& operator() (size_t i) { return a_(i); }
    T* const& operator() (size_t i) const { return a_(i); }


    // Returns a pointer to the ith value of the array.  The first
    // variant can be used as an l-value, the second cannot.  The
    // index i must be between zero and the number of items in the
    // collection less one, or an exception of type out_of_range will be
    // thrown.
    T*& operator[] (size_t i) { return a_[i]; }
    T* const& operator[] (size_t i) const { return a_[i]; }

    // Appends the item pointed to by a to the end of the array.  The
    // collection will automatically be resized if this causes the
    // number of items in the collection to exceed the capacity.
    void append (T* a) { a_.append((T*)a); }


    // Returns a pointer to the ith value of the array.  The first
    // variant can be used as an l-value, the second cannot.  The
    // index i must be between zero and the number of items in the
    // collection less one, or an exception of type out_of_range will be
    // thrown.
    T*& at (size_t i) { return a_[i]; }
    T* const& at (size_t i) const { return a_[i]; }


    // Removes all items from the collection without deleting them.
    void clear () { a_.clearFast(); }
    void clearFast () { a_.clearFast(); }
    
    // Removes and deletes all items from the collection.
    void clearAndDestroy () 
    {
      // Delete old stuff
      size_t len = length();
      for (size_t ii=0; ii<len; ii++)
	delete a_(ii);
      clear();
    }

    // Returns TRUE if the collection contains an item equal to the
    // object pointed to by a, false otherwise.  A linear search is
    // done.  Equality is measured by T's operator==.
    bool contains (const T* a) const { return a_.contains((T*)a); }
    
    // Returns a pointer to the raw memory of the array ptr.  The
    // contents should not be changed.  Should be used with care.
    T** data () const { return a_.data(); }

    // Returns the number of items currently in the collection.
    size_t entries () const { return length(); }

    // Returns a pointer to the first object encountered which is
    // equal to the object pointed to by target, or nil is no such
    // object can be found.  Equality is measured by the class-defined
    // equality operator for type T.
    T* find (const T* target) const 
    {
      size_t ind = index(target);
      if (ind==size_t(ARRAY_NPOS))
	return 0;
      else
	return a_(ind);
    }

    // Performs a linear search, returning the index of the first
    // object that is equal to the object pointed to by target, or
    // ARRAY_NPOS if there is no object.  Equality is measured by the
    // class-defined equality operator for type T.
    size_t index (const T* target) const 
    {
      size_t len = length();
      for (size_t ii=0; ii<len; ii++) 
	if (*target==*a_(ii))
	  return ii;
      return size_t(ARRAY_NPOS);
    }

    // Adds the object pointed to by a to the end of the array.  The
    // collection will be resized automatically if this causes the
    // number of items to exceed the capacity.
    void insert (T* a) { a_.insert(a); }

    // Inserts the value a into the array at index i.  The item
    // previously at position i is moved to i+1, etc.  The collection
    // will automatically be resized if this causes the number of
    // items in the collection to exceed the capacity.  The index i
    // must be between 0 and the number of items in the collection or
    // an exception of type out_of_range will be thrown.
    void insertAt (size_t i, T* a) { a_.insertAt(i,a); }

    // Returns true if there are no items in the collection,
    // false otherwise.
    bool isEmpty () const { return length()==0; }

    // Returns the number of items currently in the collection.
    size_t length () const { return a_.length(); }

    // Add the item pointed to by a to the beginning of the
    // collection.  The collection will be resized automatically if
    // this causes the number of items to exceed the capacity.
    void prepend (T* a) { a_.prepend(a); }

    // Performs a linear search, removing the first object which is
    // equal to the object pointed to by a and returns a pointer to
    // it, or NIL if no such object could be found.  Equality is
    // measured by T's operator==.
    T* remove (const T* a)
    {
      size_t len = length();
      for (size_t ii=0; ii<len; ii++) 
	if (*a==*a_(ii))
	  return removeAt(ii);
      return 0;
    }

    // Performs a linear search, removing all objects which are equal
    // to the object pointed to by a.  Returns the number of objects
    // removed.  Equality is measured by T's operator==.    
    size_t removeAll (const T* a) { return a_.removeAll((T*)a); }

    // Removes the object at index i and returns a pointer to it.  An
    // exception of type out_of_range will be thrown if it is not a
    // valid index.  Valid indices are from zero to the number of
    // items in the array less one.  Note, all the other indices will
    // then slide down.
    T* removeAt (size_t i) { return a_.removeAt(i); }

    // Removes run_length objects starting at index i. An exception of
    // type BadIndex will be thrown if the range is not valid.  Note
    // that all the other indices slide down.
    void removeRange (size_t i, size_t run_length) { return a_.removeRange(i, run_length); }

    // Removes the first item in the collection and returns a pointer
    // to it.  An exception of type out_of_range will be thrown if the
    // list is empty.
    T* removeFirst () { return removeAt(0); }

    // Removes the last item in the collection and returns a pointer
    // to it.  An exception of type out_of_range will be thrown if the
    // list is empty.
    T* removeLast () { return removeAt(length()-1); }

    // Changes the capacity of the collection to N.  Note that the
    // number of objects in the collection does not change, just the
    // capacity.
    void resize (size_t N) { a_.resize(N); }

    // Swap with another Array: This is a O(1) operation since we only
    // swap pointers (no copying of full arrays).
    void swap (ArrayPtr<T>& rhs) { a_.swap(rhs.a_); }

  protected:

    // ///// Data Members

    // The implementation: Holds an array of pointers.  Why don't we
    // inherit from Array directly? Because the interfaces between
    // Array and ArrayPtr are just different enough.  (In particular,
    // the const T&, when templatized on T* becomes T* const which
    // causes problems in the interface.  Also you have to worry about
    // how "==" is defined ... by derefencing or straight?)
    Array<T*> a_;

    
}; // ArrayPtr



// ///////////////////////////////////////////// Global Operators

template <class T>
inline bool operator== (const Array<T>& a1, const Array<T>& a2)
{
  if (a1.length()!=a2.length()) return false;
  size_t a1_len = a1.length();
  for (size_t ii=0; ii<a1_len; ii++)
    if (a1[ii]!=a2[ii]) return false;
  return true;
}

template <class T>
inline bool operator!= (const Array<T>& a1, const Array<T>& a2)
{ return !(a1==a2); }

template <class T>
inline bool operator== (const ArrayPtr<T>& a1, const ArrayPtr<T>& a2)
{
  if (a1.length()!=a2.length()) return false;
  size_t a1_len = a1.length();
  for (size_t ii=0; ii<a1_len; ii++)
    if (a1[ii]!=a2[ii]) return false;
  return true;
}

template <class T>
inline bool operator!= (const ArrayPtr<T>& a1, const ArrayPtr<T>& a2)
{ return !(a1==a2); }

template <class T>
inline bool operator< (const Array<T>& a1, const Array<T>& a2)
{
  const bool smaller = a1.length() < a2.length();
  const size_t m = smaller ? a1.length() : a2.length();
  for (size_t ii=0; ii<m; ii++)
    if (a1[ii]<a2[ii]) return true;
  return smaller;
}

template <class T>
inline bool operator<= (const Array<T>& a1, const Array<T>& a2) 
{ return a1<a2 || a1==a2;}

template <class T>
inline bool operator> (const Array<T>& a1, const Array<T>& a2)
{ return a2<a1; }

template <class T>
inline bool operator>= (const Array<T>& a1, const Array<T>& a2)
{ return a1>a2 || a1==a2; }

template <class T>
inline ostream& operator<< (ostream& os, const Array<T>& a)
{
  size_t len = a.length();
  for (size_t ii=0; ii<len; ii++)
    os << a(ii) << " ";
  return os;
}

template <class T>
inline ostream& operator<< (ostream& os, const ArrayPtr<T>& a)
{
  size_t len = a.length();
  for (size_t ii=0; ii<len; ii++)
    os << *a(ii) << " ";
  return os;
}

// Alias for append.
template<class T>
inline Array<T>& operator<< (Array<T>& arr, const T& item)
{
  arr.append(item);

  return arr;
}

template <class T>
inline void swap (Array<T>& lhs, Array<T>& rhs)
{
  lhs.swap(rhs);
}


// Aliases for append.  Since only pointers can be added for arrays,
// the operator will work with pointers or references, but in both
// cases, quietly store only the pointer.
template<class T>
inline ArrayPtr<T>& operator<< (ArrayPtr<T>& arr, T& item)
{
  arr.append(&item);

  return arr;
}

template<class T>
inline ArrayPtr<T>& operator<< (ArrayPtr<T>& arr, T* item)
{
  arr.append(item);

  return arr;
}

// Array concatenation operator.  
template<class T>
inline Array<T>& operator+= (Array<T>& lhs, const Array<T>& rhs)
{
  size_t rhs_len = rhs.length();
  for (size_t i = 0; i < rhs_len; ++i) {
    lhs.append(rhs[i]);
  }
  return lhs;
}

template<class T>
inline ArrayPtr<T>& operator+= (ArrayPtr<T>& lhs, const ArrayPtr<T>& rhs)
{
  size_t rhs_len = rhs.length();
  for (size_t i = 0; i < rhs_len; ++i) {
    lhs.append(rhs[i]);
  }
  return lhs;
}

template <class T>
inline void swap (ArrayPtr<T>& lhs, ArrayPtr<T>& rhs)
{
  lhs.swap(rhs);
}

OC_END_NAMESPACE

#define OCARRAY_H_
#endif // OCARRAY_H_



