#ifndef OC_STRING_IMPL_H_

// This is an implementation of a "fair" amount of the STL string
// class.  Not all compiler implementations support STL, so we need an
// implementation for those platforms, while still trying to adhere to
// the standard.  The class supports embedded '\0's.

// This implementation is not threadsafe (by design ... this class is
// used too much and is too low-level to do synchronization here: it
// is too expensive).

// This class was originally called "EasyString" because it is "easy"
// on the heap: If possible, all strings are copied to a local cache
// inside the string itself.  Although a simple idea, it has HUGE
// scaling ramifications on a multiple CPU machine (i.e., the string
// class allows scalability because the heap does not become a
// bottleneck).

// This implementation can be completely inlineable (it does depend on
// standard C/C++ libraries, though).  Some platforms don't support
// bool as a type, so some functions that "naturally" return bool
// return an int instead (mostly the comparison routines).  

// ///////////////////////////////////////////// Include Files

extern "C" { 
#include <string.h>           // c-style string access
#include <ctype.h>
}
#include "ochashfunction.h"   

// Forwards so we don't have to necessarily use allocators
OC_BEGIN_NAMESPACE
class StreamingPool;
typedef StreamingPool Allocator;
extern char* Allocate (Allocator*, size_t len);
extern void DeAllocate (Allocator*, char*);
OC_END_NAMESPACE

#define OC_ALLOCATE(A,LEN) Allocate(A,LEN)
#define OC_DEALLOCATE(A,LEN) DeAllocate(A,LEN)

// ///////////////////////////////////////////// Defines

// How big the internal buffer is for strings.  If a string can fit in
// this buffer, there is no need to go to the heap to get memory.
// This is very important ...the less we go to the heap, the less we
// bang up against mutexes/algorithms for memory management.

// NOTE: With this implementation, OC_STRING_LENGTH is restricted to a max 128
// This leaves a strlen of OC_STRING_LENGTH - 2 stored in the class itself as
// opposed to allocated on the heap.  Once this string goes to the heap, the
// string length can be INTMAX in length.
// NOTE: OC_STRING_LENGTH can not be less than 16 bytes for an int size of
// 4 bytes.  If int size is 8 bytes on any platform, then this should be
// at least 24.
#define OC_STRING_BUFFER_LENGTH 32
// For small strings, we store the char data in the buffer itself.  We
// use the last character to indicate the length.  All strings are null
// terminated, so the maximum length of a string that can be stored in
// the buffer is OC_STRING_BUFFER_LENGTH - 2.  When the string is longer
// than that, we store a 127 (the largest value for a signed char) in
// the last byte (the length cell), and we use the buffer to store a
// pointer to the real string data and a length.  This is done with
// the union declared in the class.
#define OC_MAX_INTERNAL_LENGTH (OC_STRING_BUFFER_LENGTH - 2)
#define OC_OFFSET_OF_LENGTH_CELL (OC_STRING_BUFFER_LENGTH - 1)
#define OC_USING_EXTERNAL_DATA 127

OC_BEGIN_NAMESPACE 

// ///////////////////////////////////////////// The OCString Class

class OCString {
    
  public:
    
    // ///// Typedefs

    typedef size_t size_type;

    // ///// Constants
    
    // npos is a "sentinel" value used to indicate the end of string
    // in many operations on strings.
    
#if defined(__SUNPRO_CC) && (0x500 > __SUNPRO_CC)
    enum { npos = -1 };
#else
    static const size_type npos = size_type(-1);
#endif    

    // ///// Methods
    
    // Constructor.  Construct an empty string.
    OCString ()
    {
      build_(0, 0);
    }
    
    
    // Constructor.  Create an internal representation for C-style
    // strings.
    OCString (const char* str) 
    { 
      size_type len= strlen(str);
      build_(str, len);
    }
    
    // Constructor.  Construct from a c-style string only the first
    // length characters.
    OCString (const char* str, size_type len, Allocator* a=0) 
    {
      build_(str, len, a);
    }

    // Destructor.  Release the string resources. 
    ~OCString ()
    {
      release_();
    }
   
    // Copy constructor.     
    OCString (const OCString& str, Allocator* a=0) 
    {
      build_(str.data(), str.length(), a);
    }

    // Operator=.  Copy the string.    
    OCString& operator= (const OCString& rhs)
    {
      // Assign to self is a no-op
      if (this!=&rhs) {
	release_(); 	                   // Release lhs resources
        build_(rhs.data(), rhs.length());  // Copy
      }
      return *this;
    }
    
    
    // Operator+=.  Concatenate rhs to this string.
    void operator+= (const OCString& rhs) 
    { 
      if (rhs.length()==0)
	return;
      else if (length()==0)
	(*this) = rhs;
      else {

	// Note: This code works even if lhs==rhs
	size_type len = length() + rhs.length();
	
	// Figure out if this can fit in the internal buffer.
	if (len < OC_MAX_INTERNAL_LENGTH) {
	  memcpy(ncdata_() + length(), rhs.data(), rhs.length());
	  ncdata_()[len]= '\0';
	  lengthCellValue_(char(len));
	}
  
	// Has to go on heap, so we have to reallocate no matter what.
	else {
	  char* temp = (u_.s_.a_) ? OC_ALLOCATE(u_.s_.a_, len+1) : new char[len+1];
	  memcpy(temp, data(), length());
	  memcpy(temp+length(), rhs.data(), rhs.length());
	  temp[len]= '\0';
	  
	  // Deallocate lhs if we have to
	  release_();

	  u_.s_.ptr_= temp;
	  u_.s_.len_= len;
	  lengthCellValue_(OC_USING_EXTERNAL_DATA);
	}
      }
    }
    
    // Create a string from a single character
    OCString& operator= (char c) 
    {
      release_();
      u_.data_[0]= c;
      u_.data_[1]= '\0';
      lengthCellValue_(1);
      return *this;
    }
    
    // Appends num_chars copies of the char c to the string
    inline OCString& append (size_type num_chars, char c);
    
    // Inserts a copy of the of the null-terminated char* s into the
    // string at byte position pos.
    inline OCString& insert (size_type pos, const char* s);
    
    // Inserts a copy of the of the string str into the string at byte
    // position pos.
    inline OCString& insert (size_type pos, const OCString& str); 
    
    // Returns true if the string is zero lengthed.
    inline int empty () const;
    
    // Removes num_chars bytes or to the end of the string (whichever
    // comes first) starting at the byte position pos, which must be
    // no greater than length.    
    inline OCString& erase (size_type pos = 0, size_type num_chars = size_type(npos));
    
    // The length of the string
    inline OCString::size_type length () const 
    {
      if (usingExternalBuffer_()) {
	return u_.s_.len_;
      } else {
	return size_type(lengthCellValue_());
      }
    }
    
    // Show (but not for adoption, just use) the c style string
    // underneath.  To quote Stroustrup (3E, p. 589) "The data()
    // function writes the characters of the string into an array and
    // returns a pointer to that array.  The array is owned by the
    // string, and the user should not try to delete it.  The user
    // also cannot rely on its value after a subsequent call on a
    // non-const function on the string.  Th c_str() function is like
    // data, except that it adds a 0 (zero) at the end as a
    // C-String-style terminator."
    inline const char* c_str () const {
      return data();
    }

    const char *data () const {
      if (usingExternalBuffer_()) {
	return u_.s_.ptr_;
      } else {
	return u_.data_;
      }
    }

    
    
    // Return a substring, starting from index i, run for length n.
    // Note that lengths too large simply return the to the end of the
    // string, but indices too large cause an out of range exception
    // to be thrown.  Note that you can index one position beyond the
    // end of the string (length of the string) so you can specify the
    // empty string. Example: string("Xanadu").substr(3,2) is "ad".
    // Indexing starts numbering from 0 (as expected in C/C++) and
    // includes the index specified.
    inline OCString substr (size_type i = 0, size_type n = npos) const;
    
    
    // Finding characters and strings within a string.  The parameter
    // 'i' is always where to start within this string.  If nothing is
    // found, npos is returned.   
    inline size_type find (char ch, size_type i = 0) const;
    inline size_type find (const OCString& str, size_type i = 0) const;
    inline size_type find (const char* cp, size_type i = 0) const;

    inline size_type rfind (char ch, size_type i = npos) const;
    inline size_type rfind (const OCString& str, size_type i = npos) const;
    inline size_type rfind (const char* cp, size_type i = npos) const;

    // Concatenation.  Note that we don't allow cascading +=.
    // Note that + is implemented in terms of += (in the global section).
    void operator+= (const char* c2)
    {
      OCString s2(c2);
      (*this)+=s2;
    }
    void operator+= (char c)
    {
      OCString s2(&c, 1);
      (*this)+= s2;
    }

    // Single element mutation/inspection.
    char& operator[] (size_type i) { return ncdata_()[i]; }
    const char& operator[] (size_type i) const { return data()[i]; }

    // Single element mutation/inspection with error-checking: If the
    // index is out range (>= length), then an out_of_range is thrown.
    inline char& at (size_type index);
    inline const char& at (size_type index) const;

    // Swap in O(1) time
    void swap (OCString& rhs)
    { OC_NAMESPACED::swap(this->u_, rhs.u_); }
    
    // Return the allocator used:  Note, if 
    Allocator* allocator () const
    { 
      if (usingExternalBuffer_()) {
	return u_.s_.a_; 
      } else { 
	return 0; 
      }
    }
  private:
    
    // ///// Data Members

    union ignored1_ {
      struct ignored2_ {
	char *ptr_;
	size_t len_;
	Allocator* a_;  
      } s_;
      char data_[OC_STRING_BUFFER_LENGTH];
    } u_;

    // This function is used by the autotest to verify that 
    // OC_STRING_BUFFERLENGTH is not too small for how we are using it.
    friend class OCStringTest;
    int testPassed_ () {
      // We need to have at least 1 byte free in addition to the ptr_
      // and len_ field for out indicator to tell us whether or not
      // we're usng the internal buffer or a heap allocated one.
      if (sizeof(ignored1_::ignored2_) > OC_STRING_BUFFER_LENGTH - 1) {
	return 0;
      }
      return 1;
    }

    void build_ (const char *str, size_type len, Allocator* a=0) {
      if (len > OC_MAX_INTERNAL_LENGTH - 2) {
	u_.s_.a_ = a;
	u_.s_.ptr_= (a) ? OC_ALLOCATE(a, len+1) : new char[len + 1];
	memcpy(u_.s_.ptr_, str, len);
	u_.s_.ptr_[len]= '\0';
	u_.s_.len_= len;
	lengthCellValue_(OC_USING_EXTERNAL_DATA);
      } else {
	if (str) {
	  memcpy(u_.data_, str, len);
	}
	u_.data_[len]= '\0';
	lengthCellValue_(char(len));
      }
    }

    // In g++ 4.4.X, this may cause a very annoying warning:
    //    warning: dereferencing pointer 'sp' does break strict-aliasing rules
    // The best workaround is to set the compiler line option: i.e.,
    //   g++ -fno-strict-aliasing (other options) something.cc -o something
    char lengthCellValue_ () const {
      return u_.data_[OC_OFFSET_OF_LENGTH_CELL];
    }

    void lengthCellValue_ (char c) {
      u_.data_[OC_OFFSET_OF_LENGTH_CELL] = c;
    }

    int usingExternalBuffer_ () const {
      return (lengthCellValue_() > OC_MAX_INTERNAL_LENGTH) ? 1 : 0;
    }

    void release_ () {
      if (usingExternalBuffer_()) {
	char* mem = u_.s_.ptr_;
	if (u_.s_.a_) { 
	  OC_DEALLOCATE(u_.s_.a_,mem);
	} else {
	  delete [] mem;
	}
      }
    }

    char * ncdata_ () {
      if (usingExternalBuffer_()) {
	return u_.s_.ptr_;
      } else {
	return u_.data_;
      }
    }
    
    // ///// Methods

    // Helper function for rfind on strings, char arrays
    size_type rfind_ (const char* tmpl, size_type ipos, size_type tlen) const;


}; // OCString

// ///////////////////////////////////////////// Inlining Stuff

#if defined(OC_USE_OC_EXCEPTIONS)
inline void OCThrowRangeEx (const char* routine, unsigned int,unsigned);
#else
#include <stdexcept>
inline void OCThrowRangeEx (const char* routine, unsigned int,unsigned){ throw std::out_of_range(routine); }
#endif


// ///////////////////////////////////////////// Global Methods

// If either is a char*, DO NOT go to the trouble to try and construct
// the OCString from the char *, just use a plain ole C-style c
// strcmp.  This should avoid unnecessary constructions.
inline int operator== (const OCString& lhs, const OCString& rhs)
{
  // Length check
  int len = int(lhs.length());
  if (len!=int(rhs.length())) return false;
  if (len==0) return true; // Both empty

  // Cache
  const char* l = lhs.data();
  const char* r = rhs.data();

  // Check and see if we can do "memory aligned" instructions:
  AVLP lp = (AVLP)l;
  AVLP rp = (AVLP)r;
  if ((lp | rp) & 0x3 ) {
    // Nope, one of bottom two bits set on at least one of the two
    return memcmp(l,r,len)==0; // SIGH: use memcmp
  }

  // Assertion: pointers are aligned, can do compares ACROSS 4 bytes
  int_u4 *li = (int_u4*)l;
  int_u4 *ri = (int_u4*)r;
  int greater_len = len >> 2;
  int lesser_len  = len & (0x3);
  // Major pass, int bytes at a time
  for (int ii=0; ii<greater_len; ii++) {
    if (li[ii] != ri[ii]) return false;
  }
  // last little bit
  l += greater_len*sizeof(int_u4);
  r += greater_len*sizeof(int_u4);
  switch (lesser_len) {
  case 3: if (*l++ != *r++) return false;
  case 2: if (*l++ != *r++) return false;
  case 1: if (*l++ != *r++) return false;
  default: break;
  }
 
  return true;
}

inline int operator== (const OCString& s1, const char* s2)
{ return strcmp(s1.c_str(), s2)==0; }

inline int operator== (const char* s1, const OCString& s2)
{ return strcmp(s1, s2.c_str())==0; }

inline int operator!= (const OCString& s1, const OCString& s2)
{ return !(s1==s2); }

inline int operator!= (const OCString& s1, const char* s2)
{ return strcmp(s1.c_str(), s2)!=0; }

inline int operator!= (const char* s1, const OCString& s2)
{ return strcmp(s1, s2.c_str())!=0; }

inline OCString operator+ (const OCString& s1, const OCString& s2)
{ OCString s(s1); s += s2; return s; }

inline OCString operator+ (const char* c1, const OCString& s2)
{ OCString s1(c1); return s1+s2; }

inline OCString operator+ (const OCString& s1, const char* c2)
{ OCString s2(c2); return s1+s2; }

inline OCString operator+ (const OCString& s1, char c)
{ return s1+OCString(&c, 1); }

inline OCString operator+ (char c, const OCString& s1)
{ return OCString(&c, 1)+s1; }

inline int operator< (const OCString& s1, const OCString& s2)
{
  int len = (s1.length()<s2.length()) ? s1.length() : s2.length();
  int result = memcmp(s1.data(), s2.data(), len);
  return (result!=0) ? (result<0) : (s1.length() < s2.length());
}

inline int operator< (const char* c1, const OCString& s2)
{ return strcmp(c1, s2.c_str()) < 0; }

inline int operator< (const OCString& s1, const char* c2)
{ return strcmp(s1.c_str(), c2) < 0; }


inline int operator<= (const OCString& s1, const OCString& s2)
{ return (s1<s2) || (s1==s2); }

inline int operator<= (const char* c1, const OCString& s2)
{ return strcmp(c1, s2.c_str()) <= 0; }

inline int operator<= (const OCString& s1, const char* c2)
{ return strcmp(s1.c_str(), c2) <= 0; }


inline int operator> (const OCString& s1, const OCString& s2)
{ return !(s1<=s2); }

inline int operator> (const char* c1, const OCString& s2)
{ return strcmp(c1, s2.c_str()) > 0; }

inline int operator> (const OCString& s1, const char* c2)
{ return strcmp(s1.c_str(), c2) > 0; }


inline int operator>= (const OCString& s1, const OCString& s2)
{ return !(s1<s2); }

inline int operator>= (const char* c1, const OCString& s2)
{ return strcmp(c1, s2.c_str()) >= 0; }

inline int operator>= (const OCString& s1, const char* c2)
{ return strcmp(s1.c_str(), c2) >= 0; }


inline char& OCString::at (size_type i)
{ 
  if (i>=length()) // size_type always guaramteed to be unsigned
    OCThrowRangeEx("string::at", i, length());
  return ncdata_()[i]; 
}

inline const char& OCString::at (size_type i) const
{ 
  if (i>=length()) 
    OCThrowRangeEx("string::at const", i, length());
  return data()[i]; 
}

inline int OCString::empty () const
{ 
  return (length() == 0);
}


inline OCString& OCString::append (size_type num_chars, char c) 
{
  // TODO? This could be optimized, but wait for a profiler to tell you so.
  if (num_chars == 0)
    return *this;

  char* buf = new char[num_chars + 1]; // This is a temp, so no allocator

  // place num number of c into buf
  memset(buf, c, num_chars);
  buf[num_chars] = '\0';

  // append new string formed into str
  (*this) += OCString(buf, num_chars);
  delete [] buf;

  return *this;
}
  

inline OCString& OCString::insert (size_type pos, const char* s)
{
  size_type len = length();

  // can only insert between position zero to the length of the string
  if (pos>len) //  || pos<0) 
    OCThrowRangeEx("string::insert", pos, len);

  (*this) = substr(0, pos) + s + substr(pos);

  return *this;
}
  


inline OCString& OCString::insert (size_type pos, const OCString& str) 
{
  size_type len = length();

  // can only insert between position zero to the length of the string
  if (pos>len) //  || pos<0) 
    OCThrowRangeEx("string::insert", pos, len);

  (*this) = substr(0, pos) + str + substr(pos);

  return *this;
}

inline OCString& OCString::erase (size_type pos, size_type num_chars) 
{
  size_type len = length();

  if (len == 0)
    return *this;

  // can only remove chars between 0 and len 
  if (pos>len) //  || pos<0) 
    OCThrowRangeEx("string::erase", pos, len);

  if (num_chars == 0)
    return *this;

  if ((len - pos)  > num_chars) 
    (*this) = substr(0, pos) + substr(pos+num_chars);  
  else
    (*this) = substr(0, pos);

  return *this;
}

inline OCString OCString::substr (OCString::size_type i, 
				  OCString::size_type n) const
{
  // Figure if too far out.
  if (i > length() ) //  || i<0)
    OCThrowRangeEx("string::substr", i, length());
  
  // Figure out the actual length: This makes so that even if you
  // specify a "length that's too large", it'll do the right thing,
  // which is to give you all the chars in the rest of the OCString
  OCString::size_type len = (length()-i < n) ? (length()-i) : n;
  return OCString(data()+i, len);
}



OCString::size_type OCString::find (char ch, OCString::size_type i) const
{
  // Be sure 'i' is not out of range.
  if (i > length())
    return npos;

  const char* cp = c_str() + i;
  const char* where = strchr(cp, ch);
  if (!where)
    return npos;
  else
    return (where - cp) + i;
}



inline OCString::size_type OCString::find (const OCString& str, 
					   OCString::size_type i) const
{ return find(str.c_str(), i); }



inline OCString::size_type OCString::find (const char* cp, 
					   OCString::size_type i) const
{
  // Be sure 'i' is not out of range.
  if (i > length())
    return npos;

  const char* c_str_plus_offset = c_str() + i;
  const char* where = strstr(c_str_plus_offset, cp);
  if (!where)
    return npos;
  else
    return (where - c_str_plus_offset) + i;
}



inline OCString::size_type OCString::rfind (char ch, 
					    OCString::size_type i) const
{
  const char* cp_start = c_str();

  if (NULL == cp_start)
    return npos;


  OCString::size_type len = length();
  
  if (i >= len) {
    i = len;
  } else {
    i++;
  }

  while (i--) {
    if (ch == cp_start[i]) {
      return i;
    }
  }

  return npos;
}



inline OCString::size_type OCString::rfind_ (const char* tmpl,
						 OCString::size_type ipos,
						 OCString::size_type tlen) const
{
  OCString::size_type len = length();
  const char* c_str;
  
  // The maximum value for ipos is the position plen chars before the
  // end of the string.  Remember that str may be "".
  
  if (tlen > len) {
    return npos;
  }
  if ((len - tlen) < ipos) {
    ipos = len - tlen;
  }

  // TODO:  Should this be an int_8 or bigger?  Recode?
  int si = (int) ipos;
  c_str = data();
  while ((0 <= si) && (0 != memcmp(c_str + si, tmpl, tlen))) {
    --si;
  }
  if (0 > si) {
    return npos;
  }
  return si;
}



inline OCString::size_type OCString::rfind (const OCString& str,
						OCString::size_type i) const
{
  return rfind_(str.c_str(), i, str.length());
}



inline OCString::size_type OCString::rfind (const char* cp,
						OCString::size_type i) const
{
  return rfind_(cp, i, ((0 == cp) ? 0 : strlen(cp)));
}


// ///////////////////////////////////////////// Globals

// Output OCStrings
inline ostream& operator<< (ostream& os, const OCString& s)
{
  return os << s.c_str();
}


// Input strings
inline istream& operator>> (istream& is, OCString& s)
{
  // In lieu of conforming to the standard, this should match the
  // behavior of the Rogue Wave RWCString class, which invokes its
  // readToken method: "Whitespace is skipped before saving
  // characters. Characters are then read from the input stream s,
  // replacing previous contents of self, until trailing whitespace or
  // an EOF is encountered. The whitespace is left on the input
  // stream. Null characters are treated the same as other characters.
  // Whitespace is identified by the standard C library function
  // isspace()."
  // 
  // NB: By default istrstreams seem to have the skipws flag on, so if
  // you read with the formatted method "is >> *bp", you'll never see
  // a whitespace character.
  // 
  // We could maybe do "is >> ws;" first to skip leading whitespace,
  // but I don't trust that.
 

  // TODO:  Automatic buffer expansion
  char buffer[4096];
  char * ebp = buffer + sizeof(buffer);
  char * bp = buffer;

  // Attempt to read a char
  is.get(*bp);

  // As long as we successfully read a char, and it's a space,
  // read another.
  while ((! (is.rdstate() & ios::failbit))
	 && isspace(*bp)) {
    is.get(*bp);
  }

  // Either the stream is empty, or we read a non-space char.
  // As long as the stream has data, and the last thing we
  // read wasn't a space, and the place we're going to put
  // the next char is still within the buffer, try to read
  // another char.
  while ((! (is.rdstate() & ios::failbit)) &&
	 (! isspace(*bp)) &&
	 (++bp < ebp)) {
    is.get(*bp);
  }

  // If we stopped because we read a space char, put it back
  // on the stream.  In all cases, bp points to the first char
  // after what we want to return, so the string length
  // can be computed from bp's position in the buffer.
  if ((! (is.rdstate() & ios::failbit)) &&
      isspace(*bp)) {
    is.putback(*bp);
  }
  
  s = OCString(buffer, bp - buffer);
  return is;
}

// Swap
inline void swap (OCString& lhs, OCString& rhs) { lhs.swap(rhs); }




// ///////////////////////////////////////////// Global Hashing Function

inline unsigned long HashFunction (const OCString& key)
{
  return OCStringHashFunction(key.data(), key.length());
}

#ifndef OC_USE_OC_STRING
inline unsigned long HashFunction (const string& key)
{
  return OCStringHashFunction(key.data(), key.length());
}
#endif

inline unsigned long HashFunction (const char* key)
{
  return OCStringHashFunction(key, strlen(key));
}

OC_END_NAMESPACE 


#define OC_STRING_IMPL_H_
#endif // OC_STRING_IMPL_H_



