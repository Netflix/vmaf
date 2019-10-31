#ifndef BIGINT_H_

// An arbitrarily length signed integer.  No apriori size.  

// Use as typical signed int class:
//  int_n a = -100;
//  a += 17;
//  a = a*a;
//  cout << a << endl;

// Discussion: Do we implement full 2s complement add/subtract for
// easy addition and subtraction thus making multiplication and
// division more difficult, OR do we implement sign-magnitude, which
// makes division and multiplication easy, but addition and
// subtraction a little harder?  Because multiplication and division
// are usually more difficult to get right, we choose implement a
// signed-magnitude implementation and leverage the tested BigUInt
// class which does most of the work.

// TODO: Needs a few more standard bit operations

#include "ocbiguint.h"

OC_BEGIN_NAMESPACE

template <class I, class BI>
  inline real_8 MakeRealFromBigInt (const BigInt<I,BI>& int_thing);
template <class I, class BI>
  inline void MakeBigIntFromReal (real_8 r, 
				  BigInt<I,BI>& int_return);
template<class I, class BI>
inline void MakeBigIntFromBinary (const char* in, size_t len,
				  BigInt<I, BI>& int_return);

template<class I, class BI>
inline string MakeBinaryFromBigInt (const BigInt<I, BI>& ii);

template<class I, class BI>
inline BigUInt<I,BI> MakeBigUIntFromBigInt (const BigInt<I,BI>& s, Allocator *a);

template <class BIGINT>
inline BIGINT StringToBigIntHelper (const char* data, int len=-1, 
				    Allocator *a=0);

// Implementation of an arbitrarily length signed integer: most of the
// math is done by the BigUInt class.
template <class I, class BI>
class BigInt {

  friend real_8 MakeRealFromBigInt<>(const BigInt<I,BI>& int_thing);
  friend void MakeBigIntFromReal<>(real_8 r, 
				   BigInt& int_return);

  friend void MakeBigIntFromBinary<>(const char* in, size_t len,
				     BigInt& int_return);

  friend string MakeBinaryFromBigInt<> (const BigInt& ii);
  friend BigUInt<I,BI> MakeBigUIntFromBigInt<> (const BigInt& s, Allocator *a);

 public:

  // Default: always 0
  BigInt () : impl_() { } // sign embedded and defaults to zero

  BigInt (const char *data, Allocator *a=0) 
  {
    BigInt result = StringToBigIntHelper<BigInt>(data, strlen(data), a);
    this->swap(result);
  }

  BigInt (const string& s, Allocator *a=0) 
  {
    BigInt result = StringToBigIntHelper<BigInt>(s.data(), s.length(), a);
    this->swap(result);
  }


#define BIGINT_CONST(T) BigInt (T i, Allocator* a=0): impl_( i<0 ? T(-i) : T(i), a) { set_sign(i<0 ? 1 : 0); }
#define BIGINT_CONSTU(T) BigInt (T i, Allocator* a=0): impl_(T(i), a) { set_sign(0); }

  BIGINT_CONST(int_1);
  BIGINT_CONSTU(int_u1);
  BIGINT_CONST(int_2);
  BIGINT_CONSTU(int_u2);
  BIGINT_CONST(int_4);
  BIGINT_CONSTU(int_u4);
  BIGINT_CONST(int_8);
  BIGINT_CONSTU(int_u8);
  BIGINT_CONSTU(ALLOW_SIZE_T);
  BIGINT_CONST(ALLOW_LONG);
  BIGINT_CONSTU(ALLOW_UNSIGNED_LONG);
  BIGINT_CONST(ALLOW_LONG_LONG);
  BIGINT_CONSTU(ALLOW_UNSIGNED_LONG_LONG);
  BIGINT_CONST(real_4);
  BIGINT_CONST(real_8);

  // Allow conversion from BigUInt easily.
  BigInt (const BigUInt<I,BI>& from) : impl_(from) { }

  // Take bottom 63 bits, and keep sign
  //operator int_8 () 
  int_8 as ()
  {
    int_8 result = 0;
    int_u8 bits = 0;
    const int len = impl_.length();
    for (int ii=0; ii<int(sizeof(int_u8)/sizeof(I)); ii++) {
      int_u8 xx = (ii<len) ? impl_.data_[ii] : 0;
      xx <<= (sizeof(I)<<3)*ii;
      bits |= xx;
    }
    result = bits;
    if (sign()==1) {
      result = -result;
    }
    return result;
  }


  // Copy constructor
  BigInt (const BigInt& rhs, Allocator*a=0) : 
    // sign_(rhs.sign_), // sign implicitly copied as embdedd in impl
    impl_(rhs.impl_,a) { } 

  // operator=
  BigInt& operator= (const BigInt& rhs) 
  { 
    // sign_ = rhs.sign_; // Not needed: sign in impl
    impl_ = rhs.impl_; return *this; 
  }
  //  BigInt& operator= (int_8 i) 
  //{ BigInt temp(i); this->swap(temp); return *this; }

  BigInt& operator+= (const BigInt& rhs)
  {
    // signs: what we want
    // 0 0 : impl_ +=, sign stays same
    // 1 0 : 
    // 0 1 : impl_ -=
    // 1 1 : impl_ +=, sign stays same
    size_t where = sign() ^ rhs.sign();
    if (where) {
      subtract_(rhs); 
    } else {
      impl_+=rhs.impl_; // signs stays same
    }
    return *this;
  }

  BigInt& operator-= (const BigInt& rhs)
  {
    // signs: what we want (compare to op+= above)
    // 0 0 : as 0 1
    // 1 0 : as 1 1
    // 0 1 : as 0 0
    // 1 1 : as 1 0
    size_t where = sign() ^ rhs.sign();
    if (where) {
      impl_+=rhs.impl_; // signs stays same
    } else {
      return subtract_(rhs); 
    }
    return *this;
  }

  // inplace mult
  BigInt& operator*= (const BigInt& rhs) 
  {
    if (rhs.impl_.zero()) {
      zeroify_();
    } else {
      size_t where = sign() ^ rhs.sign();
      impl_*=rhs.impl_; 
      set_sign(where);
    }
    return *this; 
  }

  // inplace div
  BigInt& operator/= (const BigInt& rhs) 
  {
    BigInt divver, rem;
    DivMod(*this, rhs, 
	   divver, rem);
    this->swap(divver);
    return *this;
  }

  static void DivMod (const BigInt& numerator, const BigInt& denominator, 
		      BigInt& divver, BigInt& remainder)
  {
    if (denominator.impl_.zero()) {
      throw runtime_error("Can't divide by zero");
    }
    size_t sign = numerator.sign() ^ denominator.sign();
    BigUInt<I,BI>::DivMod(numerator.impl_, denominator.impl_,
			  divver.impl_, remainder.impl_);
    // Do exactly what Python does
    // signs of num/den : what to do
    // ---------------------------------------
    // 0 0     : div, rem    101/10  -> 10, 1
    // 1 1     : div, -rem  -101/-10 -> 10, -1
    // 0 1     : -div, -rem  101/-10 -> -11, -9
    // 1 0     : -div, rem  -101/10  -> -11, 9
    if (remainder.impl_.zero()) { // No remainder: keep values, only adjust sign on divver
      divver.set_sign(sign);
    } else if (sign==0) { // both signs the same: keep values, just adjust sign
      remainder.set_sign(denominator.sign());
    } else { // both num and den have different signs, and there is remainder
      ++divver.impl_;
      divver.set_sign(1);  // always negative
      remainder.set_sign(1);
      if (denominator.sign()==1) {
	remainder-=denominator;
	remainder.set_sign(1);
      } else {
	remainder+=denominator;
      }
    }
  }

  // -
  BigInt operator- () const 
  { BigInt res(*this); res.negate(); return res; }


  // Return -1 if this<rhs, 0 if this==rhs, +1 if this>rhs
  int threeWayCompare (const BigInt& other) const
  {
    // signs : what we want
    // 0: 0 0    mag_cmp
    // 1: 1 0    -1
    // 2: 0 1    +1
    // 3: 1 1    -mag_cmp
    size_t where = sign() + (other.sign()<<1); // 0 .. 3
    switch (where) {
    case 0: return impl_.threeWayCompare(other.impl_);
    case 1: return -1;
    case 2: return +1;
    case 3: return -impl_.threeWayCompare(other.impl_);
    default: return 0;
    }
  }

  // Print out
  ostream& print (ostream& os) const
  { if (sign()==1) os << '-'; return impl_.print(os); }

  // Optimized string output function
  string stringize (int default_base = 10) const
  { return impl_.stringize(default_base, sign()==0 ? ' ' : '-'); }

  BigInt& singleDigitAdd (I digit) { return digitThing_(digit, 0); }

  BigInt& singleDigitSub (I digit) { return digitThing_(digit, 1); }

  BigInt& singleDigitMultiply (I digit) 
  { impl_.singleDigitMultiply(digit); return*this;}
  
  BigInt& operator++ ()  // prefix
  { return singleDigitAdd(1); }

  BigInt operator++ (int) // postfix
  { BigInt temp(*this); singleDigitAdd(1); return temp; }

  BigInt& operator-- ()  // prefix
  { return singleDigitSub(1); }

  BigInt operator-- (int) // postfix
  { BigInt temp(*this); singleDigitSub(1); return temp; }

  void swap (BigInt& rhs) 
  { // ::swap(sign_, rhs.sign_);  // impl has sign embedded
    impl_.swap(rhs.impl_); }
  void negate () { set_sign(sign()^1); }

  // Allow comparison to unsigned: 0 if equal, -1 if lt, +1 if gt
  int threeWayCompare (const BigUInt<I, BI>& rhs) const
  { return (sign() == 0) ? impl_.threeWayCompare(rhs) : 1; }

  int length () const { return impl_.length(); }
  int bytes () const { return length() * sizeof(I); }

  // Sign of the number: 0 if positive, 1 if negative
  size_t sign () const           { return impl_.sign(); }
  void   set_sign (size_t sign)  { return impl_.set_sign(sign); }

  Allocator* allocator () const { return impl_.allocator(); }

 protected:
  // Sign-magnitude implementation
  //int_u1         sign_;  // 0==positive, 1==negative 
  // sign embedded in impl (array) so as to keep class in 32 bytes
  BigUInt<I, BI> impl_;  // magnitude of number

  // Turn this into canonical zero
  void zeroify_() 
  {
    impl_.data_.expandTo(1); 
    impl_.data_[0] = 0; 
    set_sign(0);
  }

  // This only gets called when both have different signs, thus we are
  // doing a true subtract (everything else is an add).
  BigInt& subtract_ (const BigInt& rhs)
  {  
    switch (impl_.threeWayCompare(rhs.impl_)) { // compare magnitudes
    case -1: { // lhs is smaller, need to do rhs - lhs
      negate();
      subtractSwitchOrder_(rhs);
      break;
    }
    case  0: { // zero: avoid extra array copies
      zeroify_();
      break;
    }
    case  1: { // rhs is smaller, need to do lhs-rhs 
      impl_-=rhs.impl_;
      break;
    }
    }
    return *this;
  }


  // Do "this = other - this". This can avoid making an extra copy,
  // (which avoids an extra heap malloc, which is typically faster).
  void subtractSwitchOrder_ (const BigInt<I,BI>& other)
  {
    const int this_len = impl_.length();
    const int other_len = other.impl_.length();
    //if (other_len==1) { // optimization
    //  singleDigitSubSwitchOrder(other.impl_.data_[0]); TODO? Special version
    //  return;
    //}
    static const BI mask = BI(I(~I(0))) << (sizeof(I)<<3); // 1s top, 0s bottom
    static const BI unmask = BI(~mask);                 // 0s top, 1s bottom

    // The answer will be the length of the bigger two: since this
    // is an inplace operation, we hope that the expandTo will simply
    // adjust a pointer and not do a realloc (but it may).
    int diff = this_len - other_len;
    int max_len = this_len;
    if (diff<0) { // *this is smaller
      impl_.data_.expandTo(other_len);       
      max_len = other_len;           // lhs_len captures ORIGINAL length
    }

    // Add all elements piecewise (invert to subtract) (with carry)
    // up-to lhs len: Assertion: lhs.length >= rhs.length()
    const I *ldata = other.impl_.data_.data();
    I* rdata = this->impl_.data_.data();
    const I zero = 0;
    const I unzero = ~I(0);
    BI carry = 1;
    for (int ii=0; ii<max_len; ii++) {
      BI lhs_piece = ii>=other_len ?   zero : ldata[ii];
      I  rhs_piece = ii>= this_len ? unzero : ~rdata[ii];
      BI partial_sum = lhs_piece + rhs_piece;
      BI sum = partial_sum + carry;
      carry = (mask & sum)>>(sizeof(I)<<3);
      BI digit = unmask & sum;
      rdata[ii] = digit;
    }
    //if (carry) {
    //  data_.append(carry);
    //}
    impl_.normalize_();
  }

  // Common code for singleDigitAdd and singleDigitSubtract
  BigInt& digitThing_ (I digit, size_t isign)
  {
    if (sign()==isign) {
      impl_.singleDigitAdd(digit);
    } else {
      // Make sure if rolls over, we adjust sign: also happens to be
      // an optimzation
      if (impl_.length()==1) {
	I& other_digit = impl_.data_[0];
	if (other_digit>digit) {
	  other_digit -= digit;
	} else if (other_digit<digit) {
	  other_digit = digit - other_digit;
	  set_sign(isign);
	} else {
	  zeroify_();
	}
      } else {
	impl_.singleDigitSub(digit);
      }
    }
    return *this;
  }

}; // BigInt 

// Comparsions
/*
template <class I, class BI>
inline bool operator> (const BigInt<I,BI>& lhs, const BigInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)==1; }

template <class I, class BI>
inline bool operator>= (const BigInt<I,BI>& lhs, const BigInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)>=0; }

template <class I, class BI>
inline bool operator< (const BigInt<I,BI>& lhs, const BigInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)==-1; }

template <class I, class BI>
inline bool operator<= (const BigInt<I,BI>& lhs, const BigInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)<=0; }

template <class I, class BI>
inline bool operator== (const BigInt<I,BI>& lhs, const BigInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)==0; }

template <class I, class BI>
inline bool operator!= (const BigInt<I,BI>& lhs, const BigInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)!=0; }
*/
#define BIGINT_SPEC_OP(I,BI,OP) inline bool operator OP (const BigInt<I,BI>& lhs, const BigInt<I,BI>& rhs) { return lhs.threeWayCompare(rhs) OP 0; }


#define BIGINT_SPEC(I,BI) \
BIGINT_SPEC_OP(I,BI,==);\
BIGINT_SPEC_OP(I,BI,!=);\
BIGINT_SPEC_OP(I,BI,<);\
BIGINT_SPEC_OP(I,BI,<=);\
BIGINT_SPEC_OP(I,BI,>);\
BIGINT_SPEC_OP(I,BI,>=);

BIGINT_SPEC(int_u1, int_u2);
BIGINT_SPEC(int_u2, int_u4);
BIGINT_SPEC(int_u4, int_u8);

// Basic ops: +, -, /, *
/*
template <class I, class BI>
inline BigInt<I,BI> operator+ (const BigInt<I,BI>& lhs,const BigInt<I,BI>& rhs)
{ 
  BigInt<I,BI> result = lhs;
  return result += rhs;
}

template <class I, class BI>
inline BigInt<I,BI> operator- (const BigInt<I,BI>& lhs,const BigInt<I,BI>& rhs)
{ 
  BigInt<I,BI> result = lhs;
  return result -= rhs;
}

template <class I, class BI>
inline BigInt<I,BI> operator* (const BigInt<I,BI>& lhs,const BigInt<I,BI>& rhs)
{ 
  BigInt<I,BI> result = lhs;
  return result *= rhs;
}

template <class I, class BI>
inline BigInt<I,BI> operator/ (const BigInt<I,BI>& lhs,const BigInt<I,BI>& rhs)
{ 
  BigInt<I,BI> divver; 
  BigInt<I,BI> rem;
  BigInt<I,BI>::DivMod(lhs, rhs, divver, rem);
  return divver;
}

template <class I, class BI>
inline BigInt<I,BI> operator% (const BigInt<I,BI>& lhs,const BigInt<I,BI>& rhs)
{
  BigInt<I,BI> rem; 
  BigInt<I,BI> divver; 
  BigInt<I,BI>::DivMod(lhs, rhs, divver, rem);
  return rem;
}
*/

#define BIGINT_MATHOP(I,BI,OP) \
inline BigInt<I,BI> operator OP (const BigInt<I,BI>& lhs,const BigInt<I,BI>& rhs) \
{ BigInt<I,BI> result=lhs; return result OP##= rhs; }

#define BIGINT_MATHOP_DIV(I,BI,OP,WHICH) \
inline BigInt<I,BI> operator OP (const BigInt<I,BI>& lhs,const BigInt<I,BI>& rhs) \
{ BigInt<I,BI> rem; BigInt<I,BI> divver; BigInt<I,BI>::DivMod(lhs, rhs, divver, rem); return WHICH; }

#define BIGINT_MATHOP_DEF(I,BI) \
  BIGINT_MATHOP(I,BI, +); \
  BIGINT_MATHOP(I,BI, -); \
  BIGINT_MATHOP(I,BI, *); \
  BIGINT_MATHOP_DIV(I,BI, /, divver); \
  BIGINT_MATHOP_DIV(I,BI, %, rem); 


BIGINT_MATHOP_DEF(int_u1, int_u2);
BIGINT_MATHOP_DEF(int_u2, int_u4);
BIGINT_MATHOP_DEF(int_u4, int_u8);


// Stream operations
template <class I, class BI>
inline ostream& operator<< (ostream& os, const BigInt<I,BI>& rhs)
{ return rhs.print(os); }


// Convert from a real_8 to a BigInt: All fractional parts will be
// dropped.
template <class I, class BI>
inline void MakeBigIntFromReal (real_8 r, 
				BigInt<I,BI>& result) 
{
  real_8 plug = (r<0) ? -r : r;
  MakeBigUIntFromReal(plug, result.impl_);
  // Have to plug in sign AFTERWARDS because of copy issues
  result.set_sign(0);
  if (r<0 && !result.impl_.zero()) { // in case rounded down, don't want negative 0
    result.set_sign(1);
  }
}


// Convert a BigInt into a real_8: because of complex template
// interactions, we choose to make this a function for now.  TODO:
// should this be bit-twiddling to make sure we get all the precision?
// (Also may be faster but much less portable)
template<class I, class BI>
inline real_8 MakeRealFromBigInt (const BigInt<I, BI>& int_thing)
{
  size_t int_sign = int_thing.sign(); // may get ruined by op below
  real_8 result = MakeRealFromBigUInt(int_thing.impl_);
  if (int_sign == 1) result = -result;
  return result;
}

// Make a big int from a 2s complement binary stream in little endian
template<class I, class BI>
inline void MakeBigIntFromBinary (const char* in, size_t len,
				  BigInt<I, BI>& int_return)
{
  bool is_negative=MakeBigUIntFromBinary(in, len, int_return.impl_, true);
  if (is_negative) {
    int_return = (~int_return.impl_) + 1;
    int_return.set_sign(1);
  } else {
    int_return.set_sign(0);
  }
}


// Turn a BigInt into a 2s complement little endian stream
template<class I, class BI>
inline string MakeBinaryFromBigInt (const BigInt<I, BI>& ii)
{
  string result;
  size_t sign_bit = 0;
  if (ii.sign()==1) {
    BigUInt<I, BI> impl(ii.impl_);
    impl.negate();
    result = MakeBinaryFromBigUInt(impl, false);
    sign_bit = result[result.length()-1]>>7;
    if (sign_bit==0) result.append("\xff",1);
  } else {
    result = MakeBinaryFromBigUInt(ii.impl_);
    sign_bit = result[result.length()-1]>>7;
    if (sign_bit!=0) result.append("\0",1);
  }
  return result;
}

// Turn a BigInt into a BigUInt (we actually defer the implementation
// TO HERE from the constructor of the BigUInt class so that ocbigint
// and ocbiguint don't have ridiculous dependencies: for the most
// part, BigInt depends on BigUInt: this is the sole place where
// BigUInt depends on BigInt.
template<class I, class BI>
inline BigUInt<I,BI> MakeBigUIntFromBigInt (const BigInt<I,BI>& s,Allocator *a) 
{ BigUInt<I, BI> result(s.impl_, a); result.set_sign(0); return result; }



// What's the best way to use BigInts on a machine?  It seems that
// on 32-bit machines: BigUInt<int_u2, int_u4> is faster
// on 64-bit machines: BigUInt<int_u4, int_u8> is faster
// Use technique from Modern C++ Design to figure out how big pointers
// are: this way we can choose the better implementation for int_un
template <bool flag, typename T, typename U>
struct SelectBigInt {
  typedef T Result;
};
template <typename T, typename U>
struct SelectBigInt<false, T, U> {
  typedef U Result;
};
typedef SelectBigInt<(sizeof(void*)==4), int_u2, int_u4>::Result Smaller_int;
typedef SelectBigInt<(sizeof(void*)==4), int_u4, int_u8>::Result Bigger_int;

typedef BigInt<Smaller_int, Bigger_int> int_n;


// Allow us to compare different types of bigint
template <class I, class BI> 
inline bool operator== (const BigInt<I,BI>& lhs, const BigUInt<I,BI>& rhs) 
{ return lhs.threeWayCompare(rhs)==0; }
template <class I, class BI> 
inline bool operator== (const BigUInt<I,BI>& lhs, const BigInt<I,BI>& rhs) 
{ return rhs==lhs; }

template <class I, class BI> 
inline bool operator!= (const BigInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)!=0; }
template <class I, class BI> 
inline bool operator!= (const BigUInt<I,BI>& lhs, const BigInt<I,BI>& rhs) 
{ return rhs!=lhs; }

template <class I, class BI> 
inline bool operator< (const BigInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)<0; }
template <class I, class BI> 
inline bool operator< (const BigUInt<I,BI>& lhs, const BigInt<I,BI>& rhs) 
{ return rhs>lhs; }

template <class I, class BI> 
inline bool operator<= (const BigInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)<=0; }
template <class I, class BI> 
inline bool operator<= (const BigUInt<I,BI>& lhs, const BigInt<I,BI>& rhs) 
{ return rhs>=lhs; }

template <class I, class BI> 
inline bool operator> (const BigInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)>0; }
template <class I, class BI> 
inline bool operator> (const BigUInt<I,BI>& lhs, const BigInt<I,BI>& rhs) 
{ return rhs<lhs; }

template <class I, class BI> 
inline bool operator>= (const BigInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)>=0; }
template <class I, class BI> 
inline bool operator>= (const BigUInt<I,BI>& lhs, const BigInt<I,BI>& rhs) 
{ return rhs<=lhs; }

// Turn an ASCII string into an int
template <class BIGINT>
inline BIGINT StringToBigIntHelper (const char* data, int len, Allocator *a)
{
  BIGINT result(0,a);  // RVO
  if (len==-1) len=strlen(data);
  char c=' ';
  char sign = '\0';
  int ii;
  // Skip white space
  for (ii=0; ii<len; ii++) {
    c = data[ii];
    if (isspace(c)) continue;
    else if (isdigit(c) || c=='-' || c=='+') break;
    else ii=len; // Done
  }
  // Only accept sign after white space
  if (c=='+' || c=='-') {
    ii++;
    sign=c;
  }  
  for (; ii<len; ii++) {
    c = data[ii];
    if ( !isdigit(c) ) break; // Only keep going if digit
    result.singleDigitMultiply(10);
    result.singleDigitAdd((c-'0'));
  }
  if (sign=='-') {
    result.negate();
  }
  return result;
}

// Turn an ASCII string into an int
inline int_n StringToBigInt (const char* data, int len, Allocator *a=0)
{ return StringToBigIntHelper<int_n>(data, len, a); }

inline int_n StringToBigInt (const char* data, Allocator *a=0)
{ return StringToBigInt(data, strlen(data), a); }

inline int_n StringToBigInt (const string& s, Allocator *a=0)
{ return StringToBigInt(s.data(), s.length(), a); }


OC_END_NAMESPACE

#define BIGINT_H_
#endif // BIGINT_H_
