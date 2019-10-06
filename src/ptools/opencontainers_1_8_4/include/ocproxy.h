#ifndef OC_PROXY_H_
#define OC_PROXY_H_

// The Proxy exists as a val for a type of "link" or "pointer": a way
// to share data without having to do a full copy.  There are three
// current ways to really use this:
// (1) In a system with no threads
// (2) In a system with threads sharing Proxies
// (3) In a system with threads and/or process sharing in a shared-memory pool

// In more detail, here's how you'd use the proxies:
// (1) Some table is shared, but no threads share it.  Thus, we don't
//     have to worry about locking access to it.  The proxy adopts the
//     new Tab to it.
//
//       // CREATION
//       Proxy p = new Tab("{'a':1}");
//                or
//       Proxy p(new Tab("{'a':1}"), true, false);
//
//       // USE
//       Tab& t = p;  // Let me read this and change it, changing what others
//                    // see as well.
//       t["b"] = "add";
//
//     This proxy adopts the memory so when the last proxy goes away,
//     the table will be destructed and the memory will go away.  
// 
//     Because this idiom (below) is so common,
//
//       Val v = new Tab("{'a':1}"); 
//        // ABOVE IS EQUIVALENT TO
//       Proxy p = new Tab("{'a':1}"); 
//       Val v = p; 
//     
// 
// (2) Some table is shared in one process among multiple threads.
//     Thus all threads can access it, but they may need to lock it.
//     The last thread to use it will delete it from the standard free store.
//     The proxy adopts the new Tab given to it.
//
//       // CREATION
//       Proxy p = Locked(new Tab("{'a':1}"));  
//   
//       // USE
//       Tab& t = p;  // I __KNOW__ no one else has the lock .. rare ...
//            or
//       { // Lock is held so no one else may use or modify this
//         TransactionLock tl(p);
//         Tab& t = p;
//         t["b"]  = "add";
//       }
//
//     The "Locked" routine takes memory from the free store
//     like we'd expect, but it allows annontation of the Tab
//     so that it knows is has to be shared, with a Lock. 
// 
// (3) Some table is shared among many processes (and threads)---we want 
//     the table to be returned to its shared memory pool when we are done.
//     We simply copy the table given to us into Shared Memory.
//
//       // CREATION
//       Proxy  p = Shared(SHM, Tab("{'a': 1}"));
//
//       // USE
//       Tab& t = p;  // I __KNOW__ no one else has the lock .. rare ...
//            or
//       { // Lock is held so no one else in any process (or this one) may use 
//         // or modify this
//         TransactionLock tl(p);
//         Tab& t = p;
//         t["b"]  = "add";
//       }
//     The "SHM" is an allocator that allocates memory from some
//     shared memory pool among multiple processes.  When the last proxy
//     goes away, the Tab is deallocated from OUT of the pool.
// 
// (4) Some table is shared but we DO NOT want anyone to adopt it!
//     In other words, when the ref count goes to 0, don't try to
//     deallocate.  This is less useful, but occasionally necessary
//     if you don't control over the allocation of some Table.  The
//     memory may be from a shared pool or not, (just use as above).
//     These are also called "weak references".
//
//       // CREATION
//       void* memory = ... from some source ...
//       Proxy  p = new (memory) Tab("{'a': 1}"); // No locking
//                    or
//       Proxy p(new Tab("{'a':1}", Shm), false, enable_locking);
//
//       // USE
//       Tab& t = p;
//       // ... someone else deallocates at the proper time ...
//       // t.~Tab(); // call destructor when you want to clean up
//       // ... dellocate(memory); ...


#include "ocsynchronizer.h"
#include "ocstreamingpool.h"
#include "ocarray.h"
#include "occomplex.h"


#if defined(OC_USE_OC_STRING)
# include "ocstring.h"

  OC_BEGIN_NAMESPACE
  typedef OCString Str; // choose to use Str because of speed
  OC_END_NAMESPACE

# define VALSTR 32      // should be sizeof(Str)
#else
# include "ocstring.h"    // To pick up "HashFunction"
# include <string>

  OC_BEGIN_NAMESPACE
  typedef string Str;     // choose STL string for generality
  OC_END_NAMESPACE

# define VALSTR sizeof(Str)
#endif 

OC_BEGIN_NAMESPACE

// Forwards
class TransactionLock; 
class OCSerializer;
class OCDeserializer;
class Tab;            
class OTab;            
class Tup;            
class Arr;            
class Val;

// The most basic implementation of a proxy is a simple reference
// counting scheme that is NOT thread or process safe
template <class T>
class RefCount_  {
 public:
  RefCount_ (T* data, bool adopt=true, Allocator* a=NULL) :
    refCount_(1), 
    adopted_(adopt), 
    sharedAcrossProcesses_(false),
    allocator_(a), 
    data_(data) { }
  
  OC_INLINE ~RefCount_ ();
  void* inc ()  { refCount_++; return this; }
  void  dec ()  { if (--refCount_==0) { RCdeleteThis_(); } }
  T* data ()    { return data_; }

  //protected:
  // Disallow copying!  Only used in context of the proxy which controls
  // how it gets copied and deleted (that's why we DON'T write delete!)
  // NO IMPLEMENTATION ON PURPOSE
  RefCount_ (const RefCount_& rc);        
  RefCount_& operator= (const RefCount_& lhs); 
  const RefCount_& operator= (const RefCount_& lh) const;

  // Since the memory may come from a pool, can't delete this
  // directly (this basically decomposing delete this into
  // its pieces: call destructor then delete memory)
  OC_INLINE void RCdeleteThis_ ();

  volatile int        refCount_;
  bool       adopted_;
  bool       sharedAcrossProcesses_; // also means a ProtectedRefCount!
  Allocator* allocator_; // allocator_ is just referenced, not adopted
  T*         data_;      // data_ only adopted if the adopted_ flag is set
}; // RefCount_


// Adds protection for copying the class from threads
template <class T>
class ProtectedRefCount_ : public RefCount_<T> {
  friend class TransactionLock;

 public:
  ProtectedRefCount_ (T* data, bool adopted=true, Allocator* a=0,
		      bool shared_across_processes=false) : 
    RefCount_<T>(data, adopted, a),
    cv_(shared_across_processes)
  {
    this->sharedAcrossProcesses_ = true;
  }

  void* inc ()   // NOT virtual on purpose!
  {
    {
      this->cv_.lock();
      this->refCount_ += 1;
      //this->cv_.signal();
      this->cv_.unlock();
    }
    return this;
  }

  void dec ()  // NOT virtual on purpose!
  {
    bool done = false;
    {
      this->cv_.lock();
      if (--(this->refCount_)==0) {
	done = true;
      }
      // this->cv_.signal();
      this->cv_.unlock();
    }
    // Normally a bad idea to delete this, but this is a standard way
    // to handle resource deallocation and make sure this resource 
    // is cleaned up when there are no further references to it
    if (done) {
      PRCdeleteThis_();
    }
  }


 protected: 
  // Disallow copying!  Only used in context of the proxy which controls
  // how it gets copied and deleted (that's why we DON'T write delete!)
  // NO IMPLEMENTATION ON PURPOSE
  ProtectedRefCount_ (const ProtectedRefCount_& rc);  
  ProtectedRefCount_& operator= (const ProtectedRefCount_& lhs); 
  const ProtectedRefCount_& operator= (const ProtectedRefCount_& lhs) const ; 

  // Since the memory may come from a pool, can't delete this
  // directly (this basically decomposing delete this into
  // its pieces: call destructor then delete memory)
  OC_INLINE void PRCdeleteThis_ ();

  // Lock:  We only need mutual exclusion for ++ and -- on ref count, 
  // no signalling, but the TransactionLock needs the full condvar.
  CondVar cv_;
}; // ProtectedRefCount_



// Helper routine to error out the same way if no conversion available
inline void NoConversion_(char /*tag*/, const string& obj, const char* to_thing)
{
  string mesg = "No conversion from:"+obj+" to "+to_thing+".";
  throw logic_error(mesg.c_str());
}


// The Proxy exists as a val for a type of "link" or "pointer": a way
// to share data without having to do a full copy.  We currently only
// support proxies for Tabs and Arrs and Array<POD> (where POD is some
// Plain-Old-Data type like int_4, real_8, etc).  We DO NOT supports
// proxies for strings (because strings are supposed to be like
// strings in Python: immutable)

// Proxies keep a reference count of the Tab (or whatever) they
// contain: to get to the tab contained, simply ask for a Tab&: this
// gives you a borrowed reference (this term is taken from Python)
// where the Tab& is valid as long as the proxy itself is valid.  You
// can also ask for a Tab and you will get a complete copy.
//
//  {
//    Proxy p = new Tab("{ }");
//    Tab& borrowed_reference = p;
//    Tab complete_copy = p;          // TODO: make sure locked as copied
//  }
//  // borrowed_reference would be invalid after proxy goes away

class Proxy  {

 public:

  // Like a Val, we keep the type of the value we are proxy for
  // so we can coerce in the proper way
  char tag; char subtype; char adopt; char lock;

  // Default proxy creates an empty adopted table with no locks
  // and not shared.
  OC_INLINE Proxy ();

  // Create a proxy of a Tab.  
  OC_INLINE Proxy (Tab* table, bool adopt_table=true, 
		   bool enable_transaction_lock=false,
		   bool shared_across_processes=false);

  // Create a proxy of a OTab.  
  OC_INLINE Proxy (OTab* table, bool adopt_table=true, 
		   bool enable_transaction_lock=false,
		   bool shared_across_processes=false);

  // Create a proxy of a Tup.  
  OC_INLINE Proxy (Tup* table, bool adopt_table=true, 
		   bool enable_transaction_lock=false,
		   bool shared_across_processes=false);

  // Create a proxy of a Arr.  
  OC_INLINE Proxy (Arr* arr, bool adopt_table=true, 
		   bool enable_transaction_lock=false,
		   bool shared_across_processes=false);

  // Create a proxy of an Array (or Arr)
  template <class T>
  OC_INLINE Proxy (Array<T>* ar, bool adopt_arry=true,
		   bool enable_transaction_lock=false,
		   bool shared_across_processes=false);

  // Allows someone to get a borrowed reference:  a reference to the
  // table that is valid as long as the Proxy is still valid.
  // Proxy p = new Tab();
  // Tab& borrow_ref = p;
  operator Tab& () const 
  {
    if (tag!='t') {
      NoConversion_(tag, "Proxy", "Tab&");
    }
    RefCount_<Tab>* pp = (RefCount_<Tab>*)handle_;
    return *pp->data();
  }

  // Allows someone to get a borrowed reference:  a reference to the
  // table that is valid as long as the Proxy is still valid.
  // Proxy p = new OTab();
  // OTab& borrow_ref = p;
  operator OTab& () const 
  {
    if (tag!='o') {
      NoConversion_(tag, "Proxy", "OTab&");
    }
    RefCount_<OTab>* pp = (RefCount_<OTab>*)handle_;
    return *pp->data();
  }

  // Allows someone to get a borrowed reference:  a reference to the
  // table that is valid as long as the Proxy is still valid.
  // Proxy p = new Tup();
  // Tab& borrow_ref = p;
  operator Tup& () const 
  {
    if (tag!='u') {
      NoConversion_(tag, "Proxy", "Tup&");
    }
    RefCount_<Tup>* pp = (RefCount_<Tup>*)handle_;
    return *pp->data();
  }
  
  // Allows someone to get a borrowed reference to an Array: a reference
  // to the Array is still valid as long as the Proxy is still Valid.
  // Usage:  Proxy p = new Array<real_8>(100);
  //         Array<real_8>& a = p;    // borrowed ref
  //         Array<real_8> copy = p;  // FULL COPY

#if defined(TEMPLATE_OUTCONVERSION_SUPPORTED)
  // Some compilers don't support outconverters with templates well,
  // so we have to do it manually
  template <class T> 
  OC_INLINE operator Array<T>& () const;
#else
#define OC_PROXY2ARR_OUTCONV(T) OC_INLINE operator Array<T>& () const;
    OC_PROXY2ARR_OUTCONV(int_1)
    OC_PROXY2ARR_OUTCONV(int_u1)
    OC_PROXY2ARR_OUTCONV(int_2)
    OC_PROXY2ARR_OUTCONV(int_u2)
    OC_PROXY2ARR_OUTCONV(int_4)
    OC_PROXY2ARR_OUTCONV(int_u4)
    OC_PROXY2ARR_OUTCONV(int_8)
    OC_PROXY2ARR_OUTCONV(int_u8)
    OC_PROXY2ARR_OUTCONV(ALLOW_SIZE_T)
    OC_PROXY2ARR_OUTCONV(ALLOW_LONG)
    OC_PROXY2ARR_OUTCONV(ALLOW_UNSIGNED_LONG)
    OC_PROXY2ARR_OUTCONV(ALLOW_LONG_LONG)
    OC_PROXY2ARR_OUTCONV(ALLOW_UNSIGNED_LONG_LONG)
    OC_PROXY2ARR_OUTCONV(bool)
    OC_PROXY2ARR_OUTCONV(real_4)
    OC_PROXY2ARR_OUTCONV(real_8)
    OC_PROXY2ARR_OUTCONV(cx_t<int_1>)
    OC_PROXY2ARR_OUTCONV(cx_t<int_u1>)
    OC_PROXY2ARR_OUTCONV(cx_t<int_2>)
    OC_PROXY2ARR_OUTCONV(cx_t<int_u2>)
    OC_PROXY2ARR_OUTCONV(cx_t<int_4>)
    OC_PROXY2ARR_OUTCONV(cx_t<int_u4>)
    OC_PROXY2ARR_OUTCONV(cx_t<int_8>)
    OC_PROXY2ARR_OUTCONV(cx_t<int_u8>)
    OC_PROXY2ARR_OUTCONV(complex_8)
    OC_PROXY2ARR_OUTCONV(complex_16)
    OC_PROXY2ARR_OUTCONV(Str)  
    OC_PROXY2ARR_OUTCONV(Tab)
    OC_PROXY2ARR_OUTCONV(OTab)
    OC_PROXY2ARR_OUTCONV(Tup)
    OC_PROXY2ARR_OUTCONV(Val)
#endif

  // Allows someone to get a borrowed reference to an Array: a reference
  // to the Array is still valid as long as the Proxy is still Valid.
  // Usage:  Proxy p = new Array<real_8>(100);
  //         Array<real_8>& a = p;    // borrowed ref
  //         Array<real_8> copy = p;  // FULL COPY
  operator Arr& () const
  {
    if (tag!='n' || subtype!='Z') {
      NoConversion_(tag, "Proxy", "Arr&");
    }
    RefCount_<Arr>* pp = (RefCount_<Arr>*)handle_;
    return *pp->data();
  }

  // One less proxy
  ~Proxy () { decrement_(); }

  // Copy constructor
  Proxy (const Proxy& p) :
    tag(p.tag),
    subtype(p.subtype),
    adopt(p.adopt),
    lock(p.lock),
    handle_(p.increment_()) 
  { }

  // operator=
  Proxy& operator= (const Proxy& p)
  { 
    void* new_handle = p.increment_();
    decrement_();

    // These have to be done AFTER the decrement because
    // the decrement depends on the tag currently in place
    tag     = p.tag;
    subtype = p.subtype;
    adopt   = p.adopt;
    lock    = p.lock;

    handle_ = new_handle;
    return *this;
  }

  // Get the allocator of the Proxy
  OC_INLINE Allocator* allocator ();

  // Just clone the top-level type, with its parameters: shared,
  // locked, and allocator.  DO NOT ACTUALLY COPY ANY DATA!  Just
  // gives an empty type, but everything filled in correctly.
  OC_INLINE Proxy cloneType (); 

  // Shared across processes?
  OC_INLINE bool sharedAcrossProcesses () 
  {
    RefCount_<void>* prc=(RefCount_<void>*)handle_; 
    return prc->sharedAcrossProcesses_; 
  }


  // protected:

  friend class TransactionLock;
  friend class OCSerializer; 
  friend class OCDeserializer;
  friend bool is (const Proxy& p1, const Proxy& p2);

  OC_INLINE void* increment_ () const;  // Increment ref count 
  OC_INLINE void  decrement_ () const;  // Decrement ref count

  // Gives a pointer to the void* data in the RefCount
  inline void* data_ () const 
  { 
    RefCount_<void>* prc=(RefCount_<void>*)handle_; 
    return prc->data_;
  }
    
  // A pointer to the implementation.  Note, originally we did the
  // whole Design Pattern Bridge, but since this proxy may sit in
  // shared memory, neither the implementation nor this class may have
  // a VTBL. VTBL placement is completely out of our control across
  // processes.  Thus, no classes in shared memory can contain virtual
  // functions!
  void* handle_;

}; // Proxy


// ///////////////////// Globals

// Output a proxy as it's underlying type (for text)
OC_INLINE ostream& operator<< (ostream& os, const Proxy& p);

// Returns true if both Proxies point to the same thing, false otherwise
inline bool is (const Proxy& l, const Proxy& r) { return l.handle_==r.handle_;}

// Returns true if both Proxies points to Vals that are equal.  This is
// essentially what Python does.  [Note that we don't set up how
// Proxy's and Vals are compared: This is handled in ocval to avoid
// questions as to which overloaded operator to call] 
OC_INLINE bool operator== (const Proxy& p1, const Proxy& p2);
inline bool operator!= (const Proxy& p1, const Proxy& p2) { return !(p1==p2); }

OC_INLINE bool operator< (const Proxy& p1, const Proxy& p2);
inline bool operator<=  (const Proxy& p1, const Proxy& p2) { return p1<p2 || p1==p2; }
inline bool operator>  (const Proxy& p1, const Proxy& p2) { return !(p1<=p2); }
inline bool operator>=  (const Proxy& p1, const Proxy& p2) { return !(p1<p2); }


// Helper function: create a new Proxy, copying all the features
// of the given proxy and give me a new T!
template <class T>
OC_INLINE Proxy ProxyCopy (Proxy& p, T*);
template <class T>
OC_INLINE Proxy ProxyCopy (Proxy& p, Array<T>*);

// Give me a proxy for the table where we adopt the memory (assuming it
// comes from standard free store) and add a lock so we can do protected
// Transactions on it.
// Usage:   Proxy pl = Locked(new Tab("'something':'whatever'}"));
OC_INLINE Proxy Locked (Tab* adopted_table);
OC_INLINE Proxy Locked (OTab* adopted_table);
OC_INLINE Proxy Locked (Tup* adopted_table);
OC_INLINE Proxy Locked (Arr* adopted_arr);
template <class T>
OC_INLINE Proxy Locked (Array<T>* adopted_array);

// Give me a Proxy for the table.  The given table is immediately copied 
// to an adopted piece of SharedMemory.  We then add a lock so we can do 
// protected transactions on it.  
// Usage:  Proxy ps = Shared(shm, Tab("{ 'a':1 }"));
OC_INLINE Proxy Shared (StreamingPool* shm, const Tab& table_to_copy);
OC_INLINE Proxy Shared (StreamingPool* shm, const OTab& table_to_copy);
OC_INLINE Proxy Shared (StreamingPool* shm, const Tup& table_to_copy);
OC_INLINE Proxy Shared (StreamingPool* shm, const Arr& arr_to_copy);
template <class T>
OC_INLINE Proxy Shared (StreamingPool* shm, const Array<T>& array_to_copy);



// This class exists solely to allow protected transactions on a Proxy.
// For example:
//
//  Proxy p = some_table["something shared"];
//  {
//    TransactionLock tl(p); // LOCK OPERATIONS SO NO OTHER PROXIES CAN MANIPULATE
//    Tab& t = p;
//    t[0] = 1;
//    t[1] = 2; /// etc
//  }                        // Lock released
//
// TODO: Reader-Writer Transactions so can have multiple readers, 1 writer
class TransactionLock {
 public:
  TransactionLock (const Proxy& p, real_8 timeout_in_seconds=-1) : p_(p)
  {
    if (p_.lock) {
      ProtectedRefCount_<void>* prc=(ProtectedRefCount_<void>*)p_.handle_;
      CondVar& cv = prc->cv_;
      bool timed_out = false;
      {
	cv.lock();
	while (cv()) {
	  if (timeout_in_seconds<0) {
	    cv.wait();
	  } else {
	    timed_out = cv.timedwait_sec(timeout_in_seconds);
	    if (timed_out) break; // Defer get out work until cleaned up
	  }
	}
	cv() = true; // we got it!
	cv.unlock();
	if (timed_out) {
	  throw runtime_error("Transaction Lock timed out");
	}
      }
    }
  }

  ~TransactionLock ()
  {
    if (p_.lock) {
      ProtectedRefCount_<void>* prc=(ProtectedRefCount_<void>*)p_.handle_;
      CondVar& cv = prc->cv_;
      {
	cv.lock();
	cv() = false;  // no one there anymore!
	cv.broadcast();  // indicate we are leaving
	cv.unlock();
      }
    }
  }
  
 protected:
  Proxy p_; 

}; // TransactionLock

// The implementation: can be put into a .o if you don't want
// everything inlined.
// NOTE:  Because of twisted dependencies (Val on Proxy on Val), 
// in an inlineable environment, we put all code at the end of
// ocval.cc
//#if !defined(OC_FACTOR_INTO_H_AND_CC)
//# include "ocproxy.cc"
//#endif 

OC_END_NAMESPACE

#endif // OC_PROXY_H_

