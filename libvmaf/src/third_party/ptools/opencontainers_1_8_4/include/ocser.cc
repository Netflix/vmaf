
#if defined(OC_FACTOR_INTO_H_AND_CC)
# include "ocser.h"
#endif

#define TRYING_MEMCPY

#if defined(TRYING_MEMCPY)
#define INT4OCCOPY(M,I) {char*a=(char*)&I;memcpy(M,a,4);mem+=4;}
#define INT8OCCOPY(M,I) {char*a=(char*)&I;memcpy(M,a,8);mem+=8;}
#else 
#define INT4OCCOPY(M,I) {char*a=(char*)&I;M[0]=a[0];M[1]=a[1];M[2]=a[2];M[3]=a[3];mem+=4;}
#define INT8OCCOPY(M,I) {char*a=(char*)&I;M[0]=a[0];M[1]=a[1];M[2]=a[2];M[3]=a[3];M[4]=a[4];M[5]=a[5];M[6]=a[6];M[7]=a[7];mem+=8;}
#endif 

OC_BEGIN_NAMESPACE


OC_INLINE void OCSerializer::put (const Str& s)
{
  size_t full_len = s.length();
  char *mem;
  if (full_len <= 0xFFFFFFFF) {
    int_u4 len = full_len;
    mem = reserve_(1+sizeof(len)+len);
    *mem++ = 'a';
    INT4OCCOPY(mem, len);
  } else {
    mem = reserve_(1+sizeof(full_len)+full_len);
    *mem++ = 'A';
    INT8OCCOPY(mem, full_len);
  }
  
  memcpy(mem, s.data(), full_len);
}


#if !defined(OC_USE_OC_STRING)
OC_INLINE void OCSerializer::put (const OCString& s)
{
  size_t full_len = s.length();
  char *mem;
  if (full_len <= 0xFFFFFFFF) {
    int_u4 len = full_len;
    mem = reserve_(1+sizeof(len)+len);
    *mem++ = 'a';
    INT4OCCOPY(mem, len);
  } else {
    mem = reserve_(1+sizeof(full_len)+full_len);
    *mem++ = 'A';
    INT8OCCOPY(mem, full_len);
  }
  memcpy(mem, s.data(), full_len);
}
#endif


OC_INLINE void OCSerializer::put (const Arr& a) 
{
  size_t full_len = a.length();
  char* mem;

  if (full_len <= 0xFFFFFFFF) {
    int_u4 len = full_len;
    mem = reserve_(1+1+sizeof(len));  // 'n' 'Z' + length
    *mem++ = 'n';
    *mem++ = 'Z';
    INT4OCCOPY(mem, len);
  } else {
    mem = reserve_(1+1+sizeof(full_len));  // 'n' 'Z' + length
    *mem++ = 'N';
    *mem++ = 'Z';
    INT8OCCOPY(mem, full_len);
  }
  
  const Val* data = a.data();
  for (size_t ii=0; ii<full_len; ii++) {
    put(data[ii]);
  }
}


template <class T>
OC_INLINE void OCSerializer::put (const Array<T>& a)
{
  size_t full_len = a.length();
  size_t bytes = sizeof(T)*full_len;
  char *mem;
  if (full_len <= 0xFFFFFFFF) {
    int_u4 len = full_len;
    size_t total = 1+1+sizeof(len)+bytes;  // 'n' tag + length,dat
    mem = reserve_(total); 
    char tag = TagFor((T*)0); 
    *mem++ = 'n';
    *mem++ = tag;
    INT4OCCOPY(mem, len);
  } else {
    size_t total = 1+1+sizeof(full_len)+bytes;  // 'n' tag + length,dat
    mem = reserve_(total); 
    char tag = TagFor((T*)0); 
    *mem++ = 'N';
    *mem++ = tag;
    INT8OCCOPY(mem, full_len);
  }

  memcpy(mem, a.data(), bytes);
}


  
OC_INLINE void OCSerializer::put (const Tab& t) 
{
  size_t full_len = t.entries();
  char *mem;
  if (full_len <= 0xFFFFFFFF) {
    int_u4 len = full_len;
    mem = reserve_(1+sizeof(len));  // 't' + length
    *mem++ = 't';
    INT4OCCOPY(mem, len);
  } else {
    mem = reserve_(1+sizeof(full_len));  // 'T' + length
    *mem++ = 'T';
    INT8OCCOPY(mem, full_len);
  }
  
  It ii(t);
  while (ii()) {
    put(ii.key());
    put(ii.value());
  }
}



#define VALPRARRPUT(T) { Array<T>*ar=((Array<T>*)(handle->data_)); put(*ar); }
OC_INLINE void OCSerializer::put (const Proxy& p)
{
  // Check to see if already been serialized and get the marker,
  // otherwise we'll just be appending new marker
  RefCount_<void*>* handle = (RefCount_<void*>*)p.handle_;
  bool already_serialized = lookup_.contains(handle);
  int_4 marker = (already_serialized) ? lookup_(handle) : lookup_.entries();

  // Don't serialize nearly as much if already a Proxy
  if (already_serialized) {
    char* mem = reserve_(1+4); // P + marker
    *mem++ = 'P';
    INT4OCCOPY(mem, marker);
    return;
  }

  // Otherwise, have to a lot more work!
  // Reserve space and copy in the Proxy marker, and proxy flags
  char* mem = reserve_(1+4+3); // P + marker + adopt flag, lock flag, alloc ind
  *mem++ = 'P';
  INT4OCCOPY(mem, marker);
  *mem++ = p.adopt;
  *mem++ = p.lock;
  *mem++ = ' ';  // TODO: Allocator indicator

  // Copy in as normal
  if (p.tag=='t') {
    Tab& t = p;
    put(t);
  } else if (p.tag=='n') {
    switch (p.subtype) {
    case 's': VALPRARRPUT(int_1);  break;
    case 'S': VALPRARRPUT(int_u1); break;
    case 'i': VALPRARRPUT(int_2);  break;
    case 'I': VALPRARRPUT(int_u2); break;
    case 'l': VALPRARRPUT(int_4);  break;
    case 'L': VALPRARRPUT(int_u4); break;
    case 'x': VALPRARRPUT(int_8);  break;
    case 'X': VALPRARRPUT(int_u8); break;
    case 'b': VALPRARRPUT(bool);   break;
    case 'f': VALPRARRPUT(real_4); break;
    case 'd': VALPRARRPUT(real_8); break;
    case 'c': VALPRARRPUT(cx_t<int_1>); break;
    case 'C': VALPRARRPUT(cx_t<int_u1>); break;
    case 'e': VALPRARRPUT(cx_t<int_2>); break;
    case 'E': VALPRARRPUT(cx_t<int_u2>); break;
    case 'g': VALPRARRPUT(cx_t<int_4>); break;
    case 'G': VALPRARRPUT(cx_t<int_u4>); break;
    case 'h': VALPRARRPUT(cx_t<int_8>); break;
    case 'H': VALPRARRPUT(cx_t<int_u8>); break;
    case 'F': VALPRARRPUT(complex_8); break;
    case 'D': VALPRARRPUT(complex_16); break;
    case 'a': 
    case 't': throw logic_error("Can't have arrays of tables or strings");
    case 'n': throw logic_error("Can't have arrays of arrays");
    case 'Z': { Arr*ar=((Arr*)(handle->data_)); put(*ar); }
    }
  } else {
    throw logic_error("Can't have Proxy for other than None, Tab, Arr, Array<T>");
  }
  // If we make it here, we have serialized and added this back in
  lookup_[handle] = marker;
}



#if !defined(TRYING_MEMCPY)

#define VAL1OCCOPY(T,IN) { char*mem=reserve_(1+sizeof(T));*mem++=tag; *mem++=IN;}
#define VAL2OCCOPY(T,IN) { char*mem=reserve_(1+sizeof(T));char*r=(char*)&IN;*mem++=tag; mem[0]=r[0]; mem[1]=r[1]; mem+=sizeof(T);}
#define VAL4OCCOPY(T,IN) { char*mem=reserve_(1+sizeof(T));char*r=(char*)&IN;*mem++=tag; mem[0]=r[0]; mem[1]=r[1]; mem[2]=r[2]; mem[3]=r[3]; mem+=sizeof(T);}

#else 

#define VAL1OCCOPY(T,IN) { char*mem=reserve_(1+sizeof(T));*mem++=tag; *mem++=IN;}
#define VAL2OCCOPY(T,IN) { char*mem=reserve_(3);char*r=(char*)&IN;*mem++=tag; memcpy(mem,r,2); mem+=2;}
#define VAL4OCCOPY(T,IN) { char*mem=reserve_(5);char*r=(char*)&IN;*mem++=tag; memcpy(mem,r,4); mem+=4;}

#endif


#define VALGENOCCOPY(T,IN) { char*mem=reserve_(1+sizeof(T));char*r=(char*)&IN;*mem++=tag; memcpy(mem,r,sizeof(T));mem+=sizeof(T);}

#define VALARRPUT(T) { Array<T>*ap=(Array<T>*)&v.u.n; put(*ap); }

OC_INLINE void OCSerializer::put (const Val& v)
{
  if (IsProxy(v)) { Proxy*pp=(Proxy*)&v.u.P; put(*pp); return; } 
  char tag = v.tag;
  switch(v.tag) {
  case 's': VAL1OCCOPY(int_1,  v.u.s); break;
  case 'S': VAL1OCCOPY(int_u1, v.u.S); break;
  case 'i': VAL2OCCOPY(int_2,  v.u.i); break;
  case 'I': VAL2OCCOPY(int_u2, v.u.I); break;
  case 'l': VAL4OCCOPY(int_4,  v.u.l); break;
  case 'L': VAL4OCCOPY(int_u4, v.u.L); break;
  case 'x': VALGENOCCOPY(int_8,  v.u.x); break;
  case 'X': VALGENOCCOPY(int_u8, v.u.X); break;
  case 'b': VAL1OCCOPY(bool,   v.u.b); break;
  case 'f': VAL4OCCOPY(real_4, v.u.f); break;
  case 'd': VALGENOCCOPY(real_8, v.u.d); break;
  case 'c': VALGENOCCOPY(cx_t<int_1>,  v.u.c); break;
  case 'C': VALGENOCCOPY(cx_t<int_u1>, v.u.C); break;
  case 'e': VALGENOCCOPY(cx_t<int_2>,  v.u.e); break;
  case 'E': VALGENOCCOPY(cx_t<int_u2>, v.u.E); break;
  case 'g': VALGENOCCOPY(cx_t<int_4>,  v.u.g); break;
  case 'G': VALGENOCCOPY(cx_t<int_u4>, v.u.G); break;
  case 'h': VALGENOCCOPY(cx_t<int_8>,  v.u.h); break;
  case 'H': VALGENOCCOPY(cx_t<int_u8>, v.u.H); break;
  case 'F': VALGENOCCOPY(complex_8, v.u.F); break;
  case 'D': VALGENOCCOPY(complex_16, v.u.D); break;
  case 'a': { OCString*sp=(OCString*)&v.u.a; put(*sp); break;}
  case 't': { Tab*tp=(Tab*)&v.u.t; put(*tp); break;}
  case 'n': { 
    switch (v.subtype) {
    case 's': VALARRPUT(int_1);  break;
    case 'S': VALARRPUT(int_u1); break;
    case 'i': VALARRPUT(int_2);  break;
    case 'I': VALARRPUT(int_u2); break;
    case 'l': VALARRPUT(int_4);  break;
    case 'L': VALARRPUT(int_u4); break;
    case 'x': VALARRPUT(int_8);  break;
    case 'X': VALARRPUT(int_u8); break;
    case 'b': VALARRPUT(bool);   break;
    case 'f': VALARRPUT(real_4); break;
    case 'd': VALARRPUT(real_8); break;
    case 'c': VALARRPUT(cx_t<int_1>); break;
    case 'C': VALARRPUT(cx_t<int_u1>); break;
    case 'e': VALARRPUT(cx_t<int_2>); break;
    case 'E': VALARRPUT(cx_t<int_u2>); break;
    case 'g': VALARRPUT(cx_t<int_4>); break;
    case 'G': VALARRPUT(cx_t<int_u4>); break;
    case 'h': VALARRPUT(cx_t<int_8>); break;
    case 'H': VALARRPUT(cx_t<int_u8>); break;
    case 'F': VALARRPUT(complex_8); break;
    case 'D': VALARRPUT(complex_16); break;
    case 'a': 
    case 't': throw logic_error("Can't have arrays of tables or strings");
    case 'n': throw logic_error("Can't have arrays of arrays");
    case 'Z': {Arr*ap=(Arr*)&v.u.n; put(*ap); break;}
    }
    break;
  }
  case 'Z': *reserve_(1) = 'Z'; break; // Just a tag
  default: unknownType_("OCSerializer", v.tag);
  }
}


// OCDeserializer


#define VAL1DEOCCOPY(T,W) W = *mem++;
#define VAL2DEOCCOPY(T,W) {char*p=(char*)&W;p[0]=*mem++;p[1]=*mem++;}  
#define VAL4DEOCCOPY(T,W) {char*p=(char*)&W;p[0]=mem[0];p[1]=mem[1];p[2]=mem[2]; p[3]=mem[3];mem+=4;}  
#define VAL8DEOCCOPY(T,W) {char*p=(char*)&W;p[0]=mem[0];p[1]=mem[1];p[2]=mem[2]; p[3]=mem[3];p[4]=mem[4];p[5]=mem[5];p[6]=mem[6];p[7]=mem[7];mem+=8;}  
#define VAL16DEOCCOPY(T,W) {memcpy(&W,mem,16); mem+=16;}  


OC_INLINE void OCDeserializer::load (OCString* sp, char tag)
{
  char* mem = mem_;
  //if (*mem++!='a') throw runtime_error("load from Tab: not starting a Tab");
  if (tag=='a') {
    int_4 len;
    VAL4DEOCCOPY(int_4, len);
    new (sp) OCString(mem, len);
    mem_ = mem + len;
  } else {
    size_t len;
    VAL8DEOCCOPY(size_t, len);
    new (sp) OCString(mem, len);
    mem_ = mem + len;
  }
}


OC_INLINE void OCDeserializer::load (Tab* tp, char tag)
{
  char* mem = mem_;
  //if (*mem++!='t') throw runtime_error("load from Tab: not starting a Tab");
  size_t final_len;
  if (tag=='t') {
    int_4 len;
    VAL4DEOCCOPY(int_4, len);
    mem_ = mem;
    final_len = len;
  } else {
    VAL8DEOCCOPY(size_t, final_len);
    mem_ = mem;
  }

  Tab& t = *(new ((void*)tp) Tab());
  for (size_t ii=0; ii<final_len; ii++) {
    Val key;
    load(key);
    Val& value = t[key] = None;
    load(value);
  }
}


OC_INLINE void OCDeserializer::load (Arr* ap, char tag)
{
  char* mem = mem_;
  //if (*mem++!='n' || *mem++!='Z') throw runtime_error("load:not starting Arr");
  size_t final_len;
  if (tag=='n') {
    int_4 len;
    VAL4DEOCCOPY(int_4, len);
    mem_ = mem;
    final_len = len;
  } else {
    VAL8DEOCCOPY(size_t, final_len);
    mem_ = mem;
  }

  Arr& a = *(new ((void*)ap) Arr(final_len));
  for (size_t ii=0; ii<final_len; ii++) {
    a.append(Val());
    Val& value = a[ii];
    load(value);
  }
}


template <class T>
OC_INLINE void OCDeserializer::load (Array<T>* ap, char tag)
{
  char* mem = mem_;
  //Val empty = T();
  //if (*mem++!='n' || *mem++!=empty.tag) throw runtime_error("load:not starting Array<T> properly");

  size_t final_len;
  if (tag=='n') {
    int_4 len;
    VAL4DEOCCOPY(int_4, len);
    final_len = len;
  } else {
    VAL4DEOCCOPY(size_t, final_len);
  }

  Array<T>& a = *(new (ap) Array<T>(final_len));
  a.expandTo(final_len);
  memcpy(a.data(), mem, sizeof(T)*final_len);

  mem_ = mem + sizeof(T)*final_len;
}



#define VALARRDEOCCOPY(T) { Array<T>*ap=(Array<T>*)&v.u.n; load(ap,v.tag); break; }
OC_INLINE void OCDeserializer::load (Val& v)
{
  char*& mem = mem_;
  v.tag = *mem++;

  switch(v.tag) {
  case 's': VAL1DEOCCOPY(int_1,  v.u.s); break;
  case 'S': VAL1DEOCCOPY(int_u1, v.u.S); break;
  case 'i': VAL2DEOCCOPY(int_2,  v.u.i); break;
  case 'I': VAL2DEOCCOPY(int_u2, v.u.I); break;
  case 'l': VAL4DEOCCOPY(int_4,  v.u.l); break;
  case 'L': VAL4DEOCCOPY(int_u4, v.u.L); break;
  case 'x': VAL8DEOCCOPY(int_8,  v.u.x); break;
  case 'X': VAL8DEOCCOPY(int_u8, v.u.X); break;
  case 'b': VAL1DEOCCOPY(bool,   v.u.b); break;
  case 'f': VAL4DEOCCOPY(real_4, v.u.f); break;
  case 'd': VAL8DEOCCOPY(real_8, v.u.d); break;
  case 'c': VAL2DEOCCOPY(cx_t<int_1>,  v.u.c); break;
  case 'C': VAL2DEOCCOPY(cx_t<int_u1>, v.u.C); break;
  case 'e': VAL4DEOCCOPY(cx_t<int_2>,  v.u.e); break;
  case 'E': VAL4DEOCCOPY(cx_t<int_u2>, v.u.E); break;
  case 'g': VAL8DEOCCOPY(cx_t<int_4>,  v.u.g); break;
  case 'G': VAL8DEOCCOPY(cx_t<int_u4>, v.u.G); break;
  case 'h': VAL16DEOCCOPY(cx_t<int_8>, v.u.h); break;
  case 'H': VAL16DEOCCOPY(cx_t<int_u8>,v.u.H); break;
  case 'F': VAL8DEOCCOPY(complex_8, v.u.F); break;
  case 'D': VAL16DEOCCOPY(complex_16, v.u.D); break;
  case 'a': 
  case 'A': { OCString* sp = (OCString*)&v.u.a; load(sp, v.tag); break; }
  case 'P': { 
    Proxy*pp = (Proxy*)&v.u.P; 
    load(pp); 
    v.isproxy=true; 
    v.tag=pp->tag;
    v.subtype=pp->subtype; 
    break; 
  }
  case 't': 
  case 'T': { Tab* tp = (Tab*)&v.u.t; load(tp, v.tag); break; }
  case 'n': 
  case 'N':  { 
    v.subtype = *mem++;
    switch (v.subtype) {
    case 's': VALARRDEOCCOPY(int_1);  break;
    case 'S': VALARRDEOCCOPY(int_u1); break;
    case 'i': VALARRDEOCCOPY(int_2);  break;
    case 'I': VALARRDEOCCOPY(int_u2); break;
    case 'l': VALARRDEOCCOPY(int_4);  break;
    case 'L': VALARRDEOCCOPY(int_u4); break;
    case 'x': VALARRDEOCCOPY(int_8);  break;
    case 'X': VALARRDEOCCOPY(int_u8); break;
    case 'b': VALARRDEOCCOPY(bool);   break;
    case 'f': VALARRDEOCCOPY(real_4); break;
    case 'd': VALARRDEOCCOPY(real_8); break;
    case 'c': VALARRDEOCCOPY(cx_t<int_1>); break;
    case 'C': VALARRDEOCCOPY(cx_t<int_u1>); break;
    case 'e': VALARRDEOCCOPY(cx_t<int_2>); break;
    case 'E': VALARRDEOCCOPY(cx_t<int_u2>); break;
    case 'g': VALARRDEOCCOPY(cx_t<int_4>); break;
    case 'G': VALARRDEOCCOPY(cx_t<int_u4>); break;
    case 'h': VALARRDEOCCOPY(cx_t<int_8>); break;
    case 'H': VALARRDEOCCOPY(cx_t<int_u8>); break;
    case 'F': VALARRDEOCCOPY(complex_8); break;
    case 'D': VALARRDEOCCOPY(complex_16); break;
    case 'a': 
    case 'P': 
    case 't': throw logic_error("Can't have arrays of tables or strings");
    case 'n': throw logic_error("Can't have arrays of arrays");
    case 'Z': {Arr*ap = (Arr*)&v.u.n; load(ap); break;}
    }
    break;
  }
  case 'Z': break; // Just a tag
  default: unknownType_("load Val", v.tag);
  }
}
 

#define VALOCDESER(T) { T* ap=(T*)::operator new(sizeof(T));load(ap);new ((void*)pp) Proxy(ap, adopt, lock, shm);}
#define VALOCARRDESER(T) { Array<T>*ap=(Array<T>*)::operator new(sizeof(Array<T>));load(ap, tag);new ((void*)pp) Proxy(ap, adopt, lock, shm);}

OC_INLINE void OCDeserializer::load (Proxy* pp)
{
  char*& mem = mem_;
  //if (*mem++!='P') throw runtime_error("load:not starting Proxy");
  int_4 marker;
  VAL4DEOCCOPY(int_4, marker);

  bool already_seen = lookup_.contains(marker);
  if (already_seen) {
    // There, just attach
    Proxy seen_proxy = lookup_(marker);
    pp = new (pp) Proxy(seen_proxy);
    return;
  }

  // Not there, completely deserialize
  bool adopt, lock, shm;
  VAL1DEOCCOPY(bool, adopt);
  VAL1DEOCCOPY(bool, lock);
  VAL1DEOCCOPY(bool, shm);
  
  char tag = *mem++;
  if (tag=='t' || tag=='T') {
    VALOCDESER(Tab); // TODO: Allocator
  } else if (tag=='n' || tag=='N') {
    char subtype = *mem++;
    switch (subtype) {
    case 's': VALOCARRDESER(int_1);  break;
    case 'S': VALOCARRDESER(int_u1); break;
    case 'i': VALOCARRDESER(int_2);  break;
    case 'I': VALOCARRDESER(int_u2); break;
    case 'l': VALOCARRDESER(int_4);  break;
    case 'L': VALOCARRDESER(int_u4); break;
    case 'x': VALOCARRDESER(int_8);  break;
    case 'X': VALOCARRDESER(int_u8); break;
    case 'b': VALOCARRDESER(bool);   break;
    case 'f': VALOCARRDESER(real_4); break;
    case 'd': VALOCARRDESER(real_8); break;
    case 'c': VALOCARRDESER(cx_t<int_1>); break;
    case 'C': VALOCARRDESER(cx_t<int_u1>); break;
    case 'e': VALOCARRDESER(cx_t<int_2>); break;
    case 'E': VALOCARRDESER(cx_t<int_u2>); break;
    case 'g': VALOCARRDESER(cx_t<int_4>); break;
    case 'G': VALOCARRDESER(cx_t<int_u4>); break;
    case 'h': VALOCARRDESER(cx_t<int_8>); break;
    case 'H': VALOCARRDESER(cx_t<int_u8>); break;
    case 'F': VALOCARRDESER(complex_8); break;
    case 'D': VALOCARRDESER(complex_16); break;
    case 'a': 
    case 'A':
    case 'P': 
    case 't': 
    case 'T': throw logic_error("Can't have arrays of tables or strings or Proxies");
    case 'N':
    case 'n': throw logic_error("Can't have arrays of arrays");
    case 'Z': VALOCDESER(Arr); break;
    }
  } else {
    throw logic_error("Can't have Proxy for other than Tab, Arr, Array<T>");
  }
  // Add marker back in
  lookup_[marker] = *pp;      
}


OC_END_NAMESPACE



