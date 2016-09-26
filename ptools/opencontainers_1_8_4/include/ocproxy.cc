
#if defined(OC_FACTOR_INTO_H_AND_CC)
# include "ocproxy.h"
# include "ocval.h"

OC_BEGIN_NAMESPACE

// Explicit template instantiations

#define OC_INST_PROXYCONSARR(T) \
 template Proxy::Proxy (Array<T>*, bool, bool, bool);
OC_INST_PROXYCONSARR(int_1)
OC_INST_PROXYCONSARR(int_u1)
OC_INST_PROXYCONSARR(int_2)
OC_INST_PROXYCONSARR(int_u2)
OC_INST_PROXYCONSARR(int_4)
OC_INST_PROXYCONSARR(int_u4)
OC_INST_PROXYCONSARR(int_8)
OC_INST_PROXYCONSARR(int_u8)
OC_INST_PROXYCONSARR(ALLOW_SIZE_T)
OC_INST_PROXYCONSARR(ALLOW_LONG)
OC_INST_PROXYCONSARR(ALLOW_UNSIGNED_LONG)
OC_INST_PROXYCONSARR(ALLOW_LONG_LONG)
OC_INST_PROXYCONSARR(ALLOW_UNSIGNED_LONG_LONG)
OC_INST_PROXYCONSARR(real_4)
OC_INST_PROXYCONSARR(real_8)
OC_INST_PROXYCONSARR(cx_t<int_1>)
OC_INST_PROXYCONSARR(cx_t<int_u1>)
OC_INST_PROXYCONSARR(cx_t<int_2>)
OC_INST_PROXYCONSARR(cx_t<int_u2>)
OC_INST_PROXYCONSARR(cx_t<int_4>)
OC_INST_PROXYCONSARR(cx_t<int_u4>)
OC_INST_PROXYCONSARR(cx_t<int_8>)
OC_INST_PROXYCONSARR(cx_t<int_u8>)
OC_INST_PROXYCONSARR(complex_8)
OC_INST_PROXYCONSARR(complex_16)
OC_INST_PROXYCONSARR(bool)
OC_INST_PROXYCONSARR(Val)
OC_INST_PROXYCONSARR(Tab)
OC_INST_PROXYCONSARR(OTab)
OC_INST_PROXYCONSARR(Tup)
OC_INST_PROXYCONSARR(Str)

#if defined(TEMPLATE_OUTCONVERSION_SUPPORTED)
#define OC_INST_ARRAYOUTCONV(T) \
 template Proxy::operator Array<T>& () const; 
OC_INST_ARRAYOUTCONV(int_1)
OC_INST_ARRAYOUTCONV(int_u1)
OC_INST_ARRAYOUTCONV(int_2)
OC_INST_ARRAYOUTCONV(int_u2)
OC_INST_ARRAYOUTCONV(int_4)
OC_INST_ARRAYOUTCONV(int_u4)
OC_INST_ARRAYOUTCONV(int_8)
OC_INST_ARRAYOUTCONV(int_u8)
OC_INST_ARRAYOUTCONV(ALLOW_SIZE_T)
OC_INST_ARRAYOUTCONV(ALLOW_LONG)
OC_INST_ARRAYOUTCONV(ALLOW_UNSIGNED_LONG)
OC_INST_ARRAYOUTCONV(ALLOW_LONG_LONG)
OC_INST_ARRAYOUTCONV(ALLOW_UNSIGNED_LONG_LONG)
OC_INST_ARRAYOUTCONV(real_4)
OC_INST_ARRAYOUTCONV(real_8)
OC_INST_ARRAYOUTCONV(cx_t<int_1>)
OC_INST_ARRAYOUTCONV(cx_t<int_u1>)
OC_INST_ARRAYOUTCONV(cx_t<int_2>)
OC_INST_ARRAYOUTCONV(cx_t<int_u2>)
OC_INST_ARRAYOUTCONV(cx_t<int_4>)
OC_INST_ARRAYOUTCONV(cx_t<int_u4>)
OC_INST_ARRAYOUTCONV(cx_t<int_8>)
OC_INST_ARRAYOUTCONV(cx_t<int_u8>)
OC_INST_ARRAYOUTCONV(complex_8)
OC_INST_ARRAYOUTCONV(complex_16)
OC_INST_ARRAYOUTCONV(bool)
OC_INST_ARRAYOUTCONV(Val)
OC_INST_ARRAYOUTCONV(Tab)
OC_INST_ARRAYOUTCONV(OTab)
OC_INST_ARRAYOUTCONV(Tup)
OC_INST_ARRAYOUTCONV(Str)
#endif

#define OC_INST_LOCKED(T) \
 template Proxy Locked<T> (Array<T>*);
OC_INST_LOCKED(int_1)
OC_INST_LOCKED(int_u1)
OC_INST_LOCKED(int_2)
OC_INST_LOCKED(int_u2)
OC_INST_LOCKED(int_4)
OC_INST_LOCKED(int_u4)
OC_INST_LOCKED(int_8)
OC_INST_LOCKED(int_u8)
OC_INST_LOCKED(ALLOW_SIZE_T)
OC_INST_LOCKED(ALLOW_LONG)
OC_INST_LOCKED(ALLOW_UNSIGNED_LONG)
OC_INST_LOCKED(ALLOW_LONG_LONG)
OC_INST_LOCKED(ALLOW_UNSIGNED_LONG_LONG)
OC_INST_LOCKED(real_4)
OC_INST_LOCKED(real_8)
OC_INST_LOCKED(cx_t<int_1>)
OC_INST_LOCKED(cx_t<int_u1>)
OC_INST_LOCKED(cx_t<int_2>)
OC_INST_LOCKED(cx_t<int_u2>)
OC_INST_LOCKED(cx_t<int_4>)
OC_INST_LOCKED(cx_t<int_u4>)
OC_INST_LOCKED(cx_t<int_8>)
OC_INST_LOCKED(cx_t<int_u8>)
OC_INST_LOCKED(complex_8)
OC_INST_LOCKED(complex_16)
OC_INST_LOCKED(bool)
OC_INST_LOCKED(Val)
OC_INST_LOCKED(Tab)
OC_INST_LOCKED(OTab)
OC_INST_LOCKED(Tup)
OC_INST_LOCKED(Str)

#define OC_INST_SHARED(T) \
 template Proxy Shared<T> (StreamingPool* shm, const Array<T>& arr_to_copy);
OC_INST_SHARED(int_1)
OC_INST_SHARED(int_u1)
OC_INST_SHARED(int_2)
OC_INST_SHARED(int_u2)
OC_INST_SHARED(int_4)
OC_INST_SHARED(int_u4)
OC_INST_SHARED(int_8)
OC_INST_SHARED(int_u8)
OC_INST_SHARED(ALLOW_SIZE_T)
OC_INST_SHARED(ALLOW_LONG)
OC_INST_SHARED(ALLOW_UNSIGNED_LONG)
OC_INST_SHARED(ALLOW_LONG_LONG)
OC_INST_SHARED(ALLOW_UNSIGNED_LONG_LONG)
OC_INST_SHARED(real_4)
OC_INST_SHARED(real_8)
OC_INST_SHARED(cx_t<int_1>)
OC_INST_SHARED(cx_t<int_u1>)
OC_INST_SHARED(cx_t<int_2>)
OC_INST_SHARED(cx_t<int_u2>)
OC_INST_SHARED(cx_t<int_4>)
OC_INST_SHARED(cx_t<int_u4>)
OC_INST_SHARED(cx_t<int_8>)
OC_INST_SHARED(cx_t<int_u8>)
OC_INST_SHARED(complex_8)
OC_INST_SHARED(complex_16)
OC_INST_SHARED(bool)
OC_INST_SHARED(Val)
OC_INST_SHARED(Tab)
OC_INST_SHARED(OTab)
OC_INST_SHARED(Tup)
OC_INST_SHARED(Str)

OC_END_NAMESPACE

#endif


OC_BEGIN_NAMESPACE

template <class T>
OC_INLINE RefCount_<T>::~RefCount_<T> ()
{
  if (adopted_) {
    T* data = this->data_;
    Allocator* a=allocator_;
    if (a) {
      data->~T();
      a->deallocate((char*)data); // can't depend on allocator being that after destruction, so had to copy into local var
    } else {
      delete data;
    }
  }
}

template <class T>
OC_INLINE void RefCount_<T>::RCdeleteThis_ ()
{
  // Get allocator used so we know how to delete
  Allocator* alloc = allocator_;
  char* mem = (char*) this;

  // First call destructor, which should release everything pointed to
  this->~RefCount_<T>();

  // Now, clean up the memory for the actual ref count struct itself
  if (alloc) {
    alloc->deallocate(mem);
  } else {
    ::operator delete(mem);
  }
}


template <class T>
OC_INLINE void ProtectedRefCount_<T>::PRCdeleteThis_ ()
{
  // Get allocator used so we know how to delete
  Allocator* alloc = this->allocator_;
  char* mem = (char*) this;

  // First call destructor, which should release everything pointed to
  this->~ProtectedRefCount_<T>();

  // Now, clean up the memory for the actual ref count struct itself
  if (alloc) {
    alloc->deallocate(mem);
  } else {
    ::operator delete(mem);
  }
}


template <class T>
inline void* helpConstruct_ (T* thing, bool adopt_thing, 
			     bool enable_transaction_lock,
			     bool shared_across_processes,
			     Allocator* a)
{
  void* memory = 0;

  // Ref count with locking
  if (enable_transaction_lock) {
    if (a) {
      memory  = a->allocate(sizeof(ProtectedRefCount_<T>));
    } else {
      memory  = ::operator new(sizeof(ProtectedRefCount_<T>));
    }  
    return new (memory) ProtectedRefCount_<T>(thing,adopt_thing,a, 
					      shared_across_processes);
  }
  
  // Ref Count with no locking
  else {
    if (a) {
      memory  = a->allocate(sizeof(RefCount_<T>));
    } else {
      memory  = ::operator new(sizeof(RefCount_<T>));
    }  
    return new (memory) RefCount_<T>(thing,adopt_thing,a);
  }
}

template <class T>
inline void* helpConstruct_ (T* thing, bool adopt_thing, 
			     bool enable_transaction_lock,
			     bool shared_across_processes)
{
  Allocator* a = thing->allocator();
  return helpConstruct_(thing, adopt_thing, enable_transaction_lock, 
			shared_across_processes, a);
}

// Implementation: a lot of gross co-ersions occur.  Mostly, this is
// to avoid VTBLS because VTBLS cannot be shared across processes

OC_INLINE Proxy::Proxy () :
  tag('t'),
  subtype('Z'),
  adopt(true),
  lock(false),
  handle_(0)
{ 
  handle_ = helpConstruct_(new Tab(), adopt, lock, false);
}


OC_INLINE Proxy::Proxy (Tab* table, bool adopt_table, 
			bool enable_transaction_lock,
			bool shared_across_processes) :
  tag('t'),
  subtype('Z'),
  adopt(adopt_table),
  lock(enable_transaction_lock),
  handle_(0)
{
  handle_ = helpConstruct_(table, adopt, lock, shared_across_processes);
}

OC_INLINE Proxy::Proxy (OTab* table, bool adopt_table, 
			bool enable_transaction_lock,
			bool shared_across_processes) :
  tag('o'),
  subtype('Z'),
  adopt(adopt_table),
  lock(enable_transaction_lock),
  handle_(0)
{
  handle_ = helpConstruct_(table, adopt, lock, shared_across_processes);
}

OC_INLINE Proxy::Proxy (Tup* table, bool adopt_table, 
			bool enable_transaction_lock,
			bool shared_across_processes) :
  tag('u'),
  subtype('Z'),
  adopt(adopt_table),
  lock(enable_transaction_lock),
  handle_(0)
{
  handle_ = helpConstruct_(table, adopt, lock, shared_across_processes);
}

OC_INLINE Proxy::Proxy (Arr* arr, bool adopt_table, 
			bool enable_transaction_lock,
			bool shared_across_processes) :
  tag('n'),
  subtype('Z'),
  adopt(adopt_table),
  lock(enable_transaction_lock),
  handle_(0)
{
  handle_ = helpConstruct_(arr, adopt, lock, shared_across_processes);
}


template <class T>
OC_INLINE Proxy::Proxy (Array<T>* ar, bool adopt_array,
			bool enable_transaction_lock,
			bool shared_across_processes) :
  adopt(adopt_array),
  lock(enable_transaction_lock),
  handle_(0)
{
  // Set tag (array) and figure out its subtype
  tag = 'n';
  subtype = TagFor((T*)0);

  // Set it's handle
  handle_ = helpConstruct_(ar, adopt_array, enable_transaction_lock,
			   shared_across_processes);
}



// Helper function for Proxy
template <class T>
OC_INLINE void Decrement (T*, void* handle, bool locking)
{
  if (locking) {
    ProtectedRefCount_<T>* h=(ProtectedRefCount_<T>*) handle; 
    h->dec();
  } else {
    RefCount_<T>* h=(RefCount_<T>*) handle; 
    h->dec();
  }
}
#define DECREMENT(T) { T* i=0; Decrement(i, handle_, lock); }
#define DECREMENTARR(T) { Array<T>* i=0; Decrement(i, handle_, lock); }
 

OC_INLINE void Proxy::decrement_ () const
{ 
  switch (tag) {

    // TODO!!!!!
    //case 'a':  DECREMENT(OCString);   break; 
  case 't':  DECREMENT(Tab);   break; 
  case 'o':  DECREMENT(OTab);   break; 
  case 'u':  DECREMENT(Tup);   break; 
  case 'n':  
    switch(subtype) {
    case 's': DECREMENTARR(int_1);  break;
    case 'S': DECREMENTARR(int_u1); break;
    case 'i': DECREMENTARR(int_2);  break;
    case 'I': DECREMENTARR(int_u2); break;
    case 'l': DECREMENTARR(int_4);  break;
    case 'L': DECREMENTARR(int_u4); break;
    case 'x': DECREMENTARR(int_8);  break;
    case 'X': DECREMENTARR(int_u8); break;
    case 'b': DECREMENTARR(bool);   break;
    case 'f': DECREMENTARR(real_4); break;
    case 'd': DECREMENTARR(real_8); break;
    case 'c': DECREMENTARR(cx_t<int_1>); break;
    case 'C': DECREMENTARR(cx_t<int_u1>); break;
    case 'e': DECREMENTARR(cx_t<int_2>); break;
    case 'E': DECREMENTARR(cx_t<int_u2>); break;
    case 'g': DECREMENTARR(cx_t<int_4>); break;
    case 'G': DECREMENTARR(cx_t<int_u4>); break;
    case 'h': DECREMENTARR(cx_t<int_8>); break;
    case 'H': DECREMENTARR(cx_t<int_u8>); break;
    case 'F': DECREMENTARR(complex_8); break;
    case 'D': DECREMENTARR(complex_16); break;
      //case 'a': DECREMENTARR(Str);    break;
      //case 't': DECREMENTARR(Tab);    break;
      //case 'o': DECREMENTARR(OTab);    break;
      //case 'u': DECREMENTARR(Tup);    break;
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': DECREMENT(Arr);    break;
    default: throw logic_error("decrement");
    }
    break;
  default: throw logic_error("decrement");
  }
}

// Helper function for Proxy
template <class T>
OC_INLINE void* Increment (T*, void* handle, bool locking)
{
  if (locking) {
    ProtectedRefCount_<T>* h=(ProtectedRefCount_<T>*) handle; 
    return h->inc();
  } else {
    RefCount_<T>* h=(RefCount_<T>*) handle; 
    return h->inc();
  }
}
#define INCREMENT(T) { T* i=0; return Increment(i, handle_, lock); }
#define INCREMENTARR(T) { Array<T>* i=0; return Increment(i, handle_, lock); }

OC_INLINE void* Proxy::increment_ () const
{ 
  switch (tag) {
    //case 'a':  INCREMENT(OCString);   break; 
  case 't':  INCREMENT(Tab);   break; 
  case 'o':  INCREMENT(OTab);  break; 
  case 'u':  INCREMENT(Tup);   break; 
    case 'n':  
    switch(subtype) {
    case 's': INCREMENTARR(int_1);  break;
    case 'S': INCREMENTARR(int_u1); break;
    case 'i': INCREMENTARR(int_2);  break;
    case 'I': INCREMENTARR(int_u2); break;
    case 'l': INCREMENTARR(int_4);  break;
    case 'L': INCREMENTARR(int_u4); break;
    case 'x': INCREMENTARR(int_8);  break;
    case 'X': INCREMENTARR(int_u8); break;
    case 'b': INCREMENTARR(bool);   break;
    case 'f': INCREMENTARR(real_4); break;
    case 'd': INCREMENTARR(real_8); break;
    case 'c': INCREMENTARR(cx_t<int_1>); break;
    case 'C': INCREMENTARR(cx_t<int_u1>); break;
    case 'e': INCREMENTARR(cx_t<int_2>); break;
    case 'E': INCREMENTARR(cx_t<int_u2>); break;
    case 'g': INCREMENTARR(cx_t<int_4>); break;
    case 'G': INCREMENTARR(cx_t<int_u4>); break;
    case 'h': INCREMENTARR(cx_t<int_8>); break;
    case 'H': INCREMENTARR(cx_t<int_u8>); break;
    case 'F': INCREMENTARR(complex_8); break;
    case 'D': INCREMENTARR(complex_16); break;
      //case 'a': INCREMENTARR(Str);    break;
      //case 't': INCREMENTARR(Tab);    break;
      //case 'o': INCREMENTARR(OTab);    break;
      //case 'u': INCREMENTARR(Tup);    break;
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': INCREMENT(Arr);    break;    
    default: throw logic_error("increment");   
    }
    break;
  default: throw logic_error("increment");
  }
}

#if defined(TEMPLATE_OUTCONVERSION_SUPPORTED)
template <class T> 
OC_INLINE Proxy::operator Array<T>& () const 
{
  if (tag != 'n' || subtype != TagFor((T*)0)) {
    NoConversion_(tag, "Proxy", "Array<>");
  }
  RefCount_<Array<T> >* pp = (RefCount_<Array<T> >*)handle_;
  return *pp->data();
}
#else 
  // Same code as above, but squished to fit in a Macro
#define OC_PROXY2ARR_CODE(T) OC_INLINE Proxy::operator Array<T>& () const {\
if (tag!='n' || subtype != TagFor((T*)0)) { \
  NoConversion_(tag, "Proxy", "Array<>"); } \
 RefCount_<Array<T> >*pp=(RefCount_<Array<T> >*)handle_;return *pp->data(); }


    OC_PROXY2ARR_CODE(int_1)
    OC_PROXY2ARR_CODE(int_u1)
    OC_PROXY2ARR_CODE(int_2)
    OC_PROXY2ARR_CODE(int_u2)
    OC_PROXY2ARR_CODE(int_4)
    OC_PROXY2ARR_CODE(int_u4)
    OC_PROXY2ARR_CODE(int_8)
    OC_PROXY2ARR_CODE(int_u8)
    OC_PROXY2ARR_CODE(ALLOW_SIZE_T)
    OC_PROXY2ARR_CODE(ALLOW_LONG)
    OC_PROXY2ARR_CODE(ALLOW_UNSIGNED_LONG)
    OC_PROXY2ARR_CODE(ALLOW_LONG_LONG)
    OC_PROXY2ARR_CODE(ALLOW_UNSIGNED_LONG_LONG)
    OC_PROXY2ARR_CODE(bool)
    OC_PROXY2ARR_CODE(real_4)
    OC_PROXY2ARR_CODE(real_8)
    OC_PROXY2ARR_CODE(cx_t<int_1>)
    OC_PROXY2ARR_CODE(cx_t<int_u1>)
    OC_PROXY2ARR_CODE(cx_t<int_2>)
    OC_PROXY2ARR_CODE(cx_t<int_u2>)
    OC_PROXY2ARR_CODE(cx_t<int_4>)
    OC_PROXY2ARR_CODE(cx_t<int_u4>)
    OC_PROXY2ARR_CODE(cx_t<int_8>)
    OC_PROXY2ARR_CODE(cx_t<int_u8>)
    OC_PROXY2ARR_CODE(complex_8)
    OC_PROXY2ARR_CODE(complex_16)
    OC_PROXY2ARR_CODE(Str)  
    OC_PROXY2ARR_CODE(Tab)
    OC_PROXY2ARR_CODE(OTab)
    OC_PROXY2ARR_CODE(Tup)
    OC_PROXY2ARR_CODE(Val)
#endif

#define PROXYOS(T) { T& t = p; os << t; }
#define PROXYOSARR(T) { Array<T>& a = p; PrintArray(os, a); }
OC_INLINE ostream& operator<< (ostream& os, const Proxy& p)
{
  // TODO:  A Proxy will either print as a string or special
  // table for certain cases?
  switch (p.tag) {

    //case 'a':  PROXYOS(OCString);   break; 
  case 't':  PROXYOS(Tab);   break; 
  case 'o':  PROXYOS(OTab);   break; 
  case 'u':  PROXYOS(Tup);   break; 
  case 'n':  
    switch(p.subtype) {
    case 's': PROXYOSARR(int_1);  break;
    case 'S': PROXYOSARR(int_u1); break;
    case 'i': PROXYOSARR(int_2);  break;
    case 'I': PROXYOSARR(int_u2); break;
    case 'l': PROXYOSARR(int_4);  break;
    case 'L': PROXYOSARR(int_u4); break;
    case 'x': PROXYOSARR(int_8);  break;
    case 'X': PROXYOSARR(int_u8); break;
    case 'b': PROXYOSARR(bool);   break;
    case 'f': PROXYOSARR(real_4); break;
    case 'd': PROXYOSARR(real_8); break;
    case 'c': PROXYOSARR(cx_t<int_1>); break;
    case 'C': PROXYOSARR(cx_t<int_u1>); break;
    case 'e': PROXYOSARR(cx_t<int_2>); break;
    case 'E': PROXYOSARR(cx_t<int_u2>); break;
    case 'g': PROXYOSARR(cx_t<int_4>); break;
    case 'G': PROXYOSARR(cx_t<int_u4>); break;
    case 'h': PROXYOSARR(cx_t<int_8>); break;
    case 'H': PROXYOSARR(cx_t<int_u8>); break;
    case 'F': PROXYOSARR(complex_8); break;
    case 'D': PROXYOSARR(complex_16); break;
      //case 'a': PROXYOSARR(Str);    break;
      //case 't': PROXYOSARR(Tab);    break;
      //case 'o': PROXYOSARR(OTab);    break;
      //case 'u': PROXYOSARR(Tup);    break;
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': PROXYOS(Arr);    break;
    default: throw logic_error("decrement");   
    }
    break;
   
  default: throw logic_error("decrement");
  }
  return os;
}

#define PROXYEQ(T) { T& t1 = p1; T& t2 = p2; return t1==t2; }
#define PROXYAREQ(T) { Array<T>&a1 = p1; Array<T>& a2 = p2; return a1==a2; }
OC_INLINE bool operator== (const Proxy& p1, const Proxy& p2) 
{
  if (is(p1, p2)) return true;
  // Same handle ... equal!! Different tags, not equal!
  if (p1.tag == p2.tag) {   
    if (p1.tag=='t') {
      PROXYEQ(Tab);
    } else if (p1.tag=='o') {
      PROXYEQ(OTab);
    } else if (p1.tag=='u') {
      PROXYEQ(Tup);
    } else if (p1.tag == 'n') {
      // To be totally ordered, two arrays of different types will be 
      // compared as strings
      if (p1.subtype != p2.subtype) return Stringize(p1)==Stringize(p2);
      // Otherwise, same subtype
      switch (p1.subtype) {
      case 's': PROXYAREQ(int_1);  break;
      case 'S': PROXYAREQ(int_u1); break;
      case 'i': PROXYAREQ(int_2);  break;
      case 'I': PROXYAREQ(int_u2); break;
      case 'l': PROXYAREQ(int_4);  break;
      case 'L': PROXYAREQ(int_u4); break;
      case 'x': PROXYAREQ(int_8);  break;
      case 'X': PROXYAREQ(int_u8); break;
      case 'b': PROXYAREQ(bool);   break;
      case 'f': PROXYAREQ(real_4); break;
      case 'd': PROXYAREQ(real_8); break;
      case 'c': PROXYAREQ(cx_t<int_1>); break;
      case 'C': PROXYAREQ(cx_t<int_u1>); break;
      case 'e': PROXYAREQ(cx_t<int_2>); break;
      case 'E': PROXYAREQ(cx_t<int_u2>); break;
      case 'g': PROXYAREQ(cx_t<int_4>); break;
      case 'G': PROXYAREQ(cx_t<int_u4>); break;
      case 'h': PROXYAREQ(cx_t<int_8>); break;
      case 'H': PROXYAREQ(cx_t<int_u8>); break;
      case 'F': PROXYAREQ(complex_8); break;
      case 'D': PROXYAREQ(complex_16); break;
      case 'a': 
      case 't': 
      case 'o':
      case 'u':
      case 'n': throw logic_error("Proxies can only have arrays of POD");break;
      case 'Z': PROXYEQ(Arr);    break;
      default: throw logic_error("Unknown type for Proxy???");
      }
    } else {
      throw logic_error("can't have proxies other than Tab, OTab, Tup, Arr, Array<T>");
    }
  } 
  return false;
}


#define PROXYLT(T) { T& t1=p1; T& t2=p2; return t1<t2; } 
#define PROXYARLT(T) { Array<T>& t1=p1; Array<T>& t2=p2; return t1<t2; } 
OC_INLINE bool operator< (const Proxy& p1, const Proxy& p2) 
{
  if (is(p1, p2)) return false;   // Same handle: me<me is false!

  if (p1.tag == p2.tag) {         // Same tag, can probably compare directly
    if (p1.tag=='t') {
      PROXYLT(Tab);
    } else if (p1.tag=='o') {
      PROXYLT(OTab);
    } else if (p1.tag=='u') {
      PROXYLT(Tup);
    } else if (p1.tag == 'n') {
      // To be totally ordered, two arrays of different types will be 
      // compared as strings
      if (p1.subtype != p2.subtype) return Stringize(p1)<Stringize(p2);
      // Otherwise, same subtype
      switch (p1.subtype) {
      case 's': PROXYARLT(int_1);  break;
      case 'S': PROXYARLT(int_u1); break;
      case 'i': PROXYARLT(int_2);  break;
      case 'I': PROXYARLT(int_u2); break;
      case 'l': PROXYARLT(int_4);  break;
      case 'L': PROXYARLT(int_u4); break;
      case 'x': PROXYARLT(int_8);  break;
      case 'X': PROXYARLT(int_u8); break;
      case 'b': PROXYARLT(bool);   break;
      case 'f': PROXYARLT(real_4); break;
      case 'd': PROXYARLT(real_8); break;
      case 'c': PROXYARLT(cx_t<int_1>); break;
      case 'C': PROXYARLT(cx_t<int_u1>); break;
      case 'e': PROXYARLT(cx_t<int_2>); break;
      case 'E': PROXYARLT(cx_t<int_u2>); break;
      case 'g': PROXYARLT(cx_t<int_4>); break;
      case 'G': PROXYARLT(cx_t<int_u4>); break;
      case 'h': PROXYARLT(cx_t<int_8>); break;
      case 'H': PROXYARLT(cx_t<int_u8>); break;
      case 'F': PROXYARLT(complex_8); break;
      case 'D': PROXYARLT(complex_16); break;
      case 'a': 
      case 't': 
      case 'o': 
      case 'u': 
      case 'n': throw logic_error("Proxies can only have arrays of POD");break;
      case 'Z': PROXYLT(Arr);    break;
      default: throw logic_error("Unknown type for Proxy in op<???");
      }
    } else {
      throw logic_error("can't have proxies other than Tab, OTab, Tup, Arr, Array<T>");
    }
  } 
  // Totally different types for Proxies: compare as strings
  return Stringize(p1) < Stringize(p2);
}

// Sigh... all allocators should ALWAYS be first bytes
#define PROXYALLOCATOR(T) { T*tp=(T*)this->data_();return tp->allocator(); }
#define PROXYALLOCATORARR(T) { Array<T>*tp=(Array<T>*)this->data_();return tp->allocator(); }
OC_INLINE Allocator* Proxy::allocator ()
{ 
  switch (tag) {
    //case 'a':  PROXYALLOCATOR(OCString);   break; 
  case 't':  PROXYALLOCATOR(Tab);   break; 
  case 'o':  PROXYALLOCATOR(OTab);  break; 
  case 'u':  PROXYALLOCATOR(Tup);   break; 
    case 'n':  
    switch(subtype) {
    case 's': PROXYALLOCATORARR(int_1);  break;
    case 'S': PROXYALLOCATORARR(int_u1); break;
    case 'i': PROXYALLOCATORARR(int_2);  break;
    case 'I': PROXYALLOCATORARR(int_u2); break;
    case 'l': PROXYALLOCATORARR(int_4);  break;
    case 'L': PROXYALLOCATORARR(int_u4); break;
    case 'x': PROXYALLOCATORARR(int_8);  break;
    case 'X': PROXYALLOCATORARR(int_u8); break;
    case 'b': PROXYALLOCATORARR(bool);   break;
    case 'f': PROXYALLOCATORARR(real_4); break;
    case 'd': PROXYALLOCATORARR(real_8); break;
    case 'c': PROXYALLOCATORARR(cx_t<int_1>); break;
    case 'C': PROXYALLOCATORARR(cx_t<int_u1>); break;
    case 'e': PROXYALLOCATORARR(cx_t<int_2>); break;
    case 'E': PROXYALLOCATORARR(cx_t<int_u2>); break;
    case 'g': PROXYALLOCATORARR(cx_t<int_4>); break;
    case 'G': PROXYALLOCATORARR(cx_t<int_u4>); break;
    case 'h': PROXYALLOCATORARR(cx_t<int_8>); break;
    case 'H': PROXYALLOCATORARR(cx_t<int_u8>); break;
    case 'F': PROXYALLOCATORARR(complex_8); break;
    case 'D': PROXYALLOCATORARR(complex_16); break;
      //case 'a': PROXYALLOCATORARR(Str);    break;
      //case 't': PROXYALLOCATORARR(Tab);    break;
      //case 'o': PROXYALLOCATORARR(OTab);    break;
      //case 'u': PROXYALLOCATORARR(Tup);    break;
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': PROXYALLOCATOR(Arr);    break;    
    default: throw logic_error("increment");   
    }
    break;
  default: throw logic_error("increment");
  }
}

template <class T>
OC_INLINE Proxy ProxyCopy (Proxy& p, T*)
{
  Allocator* ap = p.allocator();
  void* mem;
  if (ap) {
    mem = ap->allocate(sizeof(T));
  } else {
    mem = ::operator new(sizeof(T));    
  }
  T* new_thing = new (mem) T(ap);
  Proxy new_p(new_thing, p.adopt, p.lock, p.sharedAcrossProcesses());
  return new_p;
}

template <class T>
OC_INLINE Proxy ProxyCopy (Proxy& p, Array<T>*)
{
  Allocator* ap = p.allocator();
  void* mem;
  if (ap) {
    mem = ap->allocate(sizeof(Array<T>));
  } else {
    mem = ::operator new(sizeof(Array<T>));    
  }
  Array<T>* new_thing = new (mem) Array<T>(4, ap);
  Proxy new_p(new_thing, p.adopt, p.lock, p.sharedAcrossProcesses());
  return new_p;
}

#define OC_INST_PROXYCOPYARR(T) \
 template Proxy ProxyCopy (Proxy& p, Array<T>*);
//template OC_INLINE Proxy ProxyCopy (Proxy& p, Array<T>*);
OC_INST_PROXYCOPYARR(int_1)
OC_INST_PROXYCOPYARR(int_u1)
OC_INST_PROXYCOPYARR(int_2)
OC_INST_PROXYCOPYARR(int_u2)
OC_INST_PROXYCOPYARR(int_4)
OC_INST_PROXYCOPYARR(int_u4)
OC_INST_PROXYCOPYARR(int_8)
OC_INST_PROXYCOPYARR(int_u8)
OC_INST_PROXYCOPYARR(ALLOW_SIZE_T)
OC_INST_PROXYCOPYARR(ALLOW_LONG)
OC_INST_PROXYCOPYARR(ALLOW_UNSIGNED_LONG)
OC_INST_PROXYCOPYARR(ALLOW_LONG_LONG)
OC_INST_PROXYCOPYARR(ALLOW_UNSIGNED_LONG_LONG)
OC_INST_PROXYCOPYARR(real_4)
OC_INST_PROXYCOPYARR(real_8)
OC_INST_PROXYCOPYARR(cx_t<int_1>)
OC_INST_PROXYCOPYARR(cx_t<int_u1>)
OC_INST_PROXYCOPYARR(cx_t<int_2>)
OC_INST_PROXYCOPYARR(cx_t<int_u2>)
OC_INST_PROXYCOPYARR(cx_t<int_4>)
OC_INST_PROXYCOPYARR(cx_t<int_u4>)
OC_INST_PROXYCOPYARR(cx_t<int_8>)
OC_INST_PROXYCOPYARR(cx_t<int_u8>)
OC_INST_PROXYCOPYARR(complex_8)
OC_INST_PROXYCOPYARR(complex_16)
OC_INST_PROXYCOPYARR(bool)
OC_INST_PROXYCOPYARR(Val)

#define OC_INST_PROXYCOPY(T) \
 template Proxy ProxyCopy (Proxy& p, T*);
// template OC_INLINE Proxy ProxyCopy (Proxy& p, T*);
OC_INST_PROXYCOPY(Tab)
OC_INST_PROXYCOPY(OTab)
OC_INST_PROXYCOPY(Tup)
OC_INST_PROXYCOPY(Arr)  // This will work because Arr can take just 1 args (the allocator): the only reason there is a difference is because of the constructor of Array vs. the other

// Clone the way the proxy works: it's allocator, lock, etc.
#define PROXYCLONE(T) return ProxyCopy(*this, (T*)0); 
#define PROXYCLONEARR(T) return ProxyCopy(*this, (Array<T>*)0); 
OC_INLINE Proxy Proxy::cloneType ()
{ 
  switch (tag) {
    //case 'a':  PROXYCLONE(OCString);   break; 
  case 't':  PROXYCLONE(Tab);   break; 
  case 'o':  PROXYCLONE(OTab);  break; 
  case 'u':  PROXYCLONE(Tup);   break; 
    case 'n':  
    switch(subtype) {
    case 's': PROXYCLONEARR(int_1);  break;
    case 'S': PROXYCLONEARR(int_u1); break;
    case 'i': PROXYCLONEARR(int_2);  break;
    case 'I': PROXYCLONEARR(int_u2); break;
    case 'l': PROXYCLONEARR(int_4);  break;
    case 'L': PROXYCLONEARR(int_u4); break;
    case 'x': PROXYCLONEARR(int_8);  break;
    case 'X': PROXYCLONEARR(int_u8); break;
    case 'b': PROXYCLONEARR(bool);   break;
    case 'f': PROXYCLONEARR(real_4); break;
    case 'd': PROXYCLONEARR(real_8); break;
    case 'c': PROXYCLONEARR(cx_t<int_1>); break;
    case 'C': PROXYCLONEARR(cx_t<int_u1>); break;
    case 'e': PROXYCLONEARR(cx_t<int_2>); break;
    case 'E': PROXYCLONEARR(cx_t<int_u2>); break;
    case 'g': PROXYCLONEARR(cx_t<int_4>); break;
    case 'G': PROXYCLONEARR(cx_t<int_u4>); break;
    case 'h': PROXYCLONEARR(cx_t<int_8>); break;
    case 'H': PROXYCLONEARR(cx_t<int_u8>); break;
    case 'F': PROXYCLONEARR(complex_8); break;
    case 'D': PROXYCLONEARR(complex_16); break;
      //case 'a': PROXYCLONEARR(Str);    break;
      //case 't': PROXYCLONEARR(Tab);    break;
      //case 'o': PROXYCLONEARR(OTab);    break;
      //case 'u': PROXYCLONEARR(Tup);    break;
    case 'n': throw logic_error("Arrays of Arrays not currently supported");
    case 'Z': PROXYCLONE(Arr);    break;    
    default: throw logic_error("clone");   
    }
    break;
  default: throw logic_error("clone");
  }
}


OC_INLINE Proxy Locked (Tab* adopted_table)
{
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = false;
  return Proxy(adopted_table, adopting, requires_lock,shared_across_processes);
}

OC_INLINE Proxy Locked (OTab* adopted_table)
{
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = false;
  return Proxy(adopted_table, adopting, requires_lock,shared_across_processes);
}

OC_INLINE Proxy Locked (Tup* adopted_table)
{
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = false;
  return Proxy(adopted_table, adopting, requires_lock,shared_across_processes);
}

OC_INLINE Proxy Locked (Arr* adopted_arr)
{
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = false;
  return Proxy(adopted_arr, adopting, requires_lock,shared_across_processes);
}

template <class T>
OC_INLINE Proxy Locked (Array<T>* adopted_array)
{
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = false;
  return Proxy(adopted_array, adopting, requires_lock,shared_across_processes);
}

// TODO: A lot of replicated code ... these should probably be templates

OC_INLINE Proxy Shared (StreamingPool* shm, const Tab& table_to_copy)
{
  // Give me memory for the top-level of Tab
  char* shared = shm->allocate(sizeof(Tab));

  // Create a Tab, top-level in shared memory, and pass in the
  // allocator we have to use
  new (shared) Tab(table_to_copy, shm);
  
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = true;
  return Proxy((Tab*)shared, adopting, requires_lock, shared_across_processes);
}

OC_INLINE Proxy Shared (StreamingPool* shm, const OTab& table_to_copy)
{
  // Give me memory for the top-level of OTab
  char* shared = shm->allocate(sizeof(OTab));

  // Create a OTab, top-level in shared memory, and pass in the
  // allocator we have to use
  new (shared) OTab(table_to_copy, shm);
  
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = true;
  return Proxy((OTab*)shared, adopting, requires_lock, shared_across_processes);
}

OC_INLINE Proxy Shared (StreamingPool* shm, const Tup& table_to_copy)
{
  // Give me memory for the top-level of Tup
  char* shared = shm->allocate(sizeof(Tup));

  // Create a Tup, top-level in shared memory, and pass in the
  // allocator we have to use
  new (shared) Tup(table_to_copy, shm);
  
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = true;
  return Proxy((Tup*)shared, adopting, requires_lock, shared_across_processes);
}


OC_INLINE Proxy Shared (StreamingPool* shm, const Arr& arr_to_copy)
{
  // Give me memory for the top-level of Arr
  char* shared = shm->allocate(sizeof(Arr));

  // Create a Arr, top-level in shared memory, and pass in the
  // allocator we have to use
  new (shared) Arr(arr_to_copy, shm);
  
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = true;
  return Proxy((Arr*)shared, adopting, requires_lock, shared_across_processes);
}


template <class T>
OC_INLINE Proxy Shared (StreamingPool* shm, const Array<T>& arr_to_copy)
{
  // Give me memory for the top-level of Tab
  char* shared = shm->allocate(sizeof(Array<T>));

  // Create a Tab, top-level in shared memory, and pass in the
  // allocator we have to use
  new (shared) Array<T>(arr_to_copy, shm);
  
  bool adopting      = true;
  bool requires_lock = true;
  bool shared_across_processes = true;
  return Proxy((Array<T>*)shared, adopting, requires_lock, shared_across_processes);
}



OC_END_NAMESPACE
