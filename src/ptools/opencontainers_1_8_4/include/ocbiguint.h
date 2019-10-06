#ifndef BIGUINT_H_
#define BIGUINT_H_

// For arbitrarily sized unsigned integers: Usage is straight-forward:
// just like unsigned ints for the most part, but sometimes you have
// to convert ints explicitly.  We support +, -, *, / and a few bit
// operations.  Examples:
//
//   int_un a;  // initializes to zero
//   int_un b = 17;
//   b += a;  
//   b += int_un(18);    // sometimes have to explicitly cast
//   cout << b << endl;  // just like ints
//   b.print(cout, 16);  // hex print
//   
// The "implementation" (in this case) is the BigUInt class,
// templatized on appropriate integer types: the BigUInt is stored as
// an array of int_u?, but all math (with carrys and the like) has to
// be done in a larger integer so we can detect the carrys and the
// like.  The three major uses are:
//
//   BigUInt<int_u1, int_u2> case1;   // Useful for debugging purposes
//   BigUInt<int_u2, int_u4> case2;   // Faster on 32-bit machines
//   BigUInt<int_u4, int_u8> case3;   // Faster on 64-bit machines
//
// The typedef for int_un detects a 32-bit machine/64-bit machine
// and chooses the faster implementation.
//
// TODO: A few more operations need to be supported so the BigUInt
// behave just like plain ints (&,|^, etc). 

#include "ocarray.h"  // implemented as Array<I>, math in BI
#include <string.h>   // for memcpy
#include <math.h>       // for modf and fmod

OC_BEGIN_NAMESPACE

// Forward decl for converting to real values
template <class I, class BI> class BigInt;  // forward decl;
template <class I, class BI> class BigUInt;  // forward decl;
template <class I, class BI> 
  void MakeBigUIntFromReal (real_8 r, 
			    BigUInt<I,BI>& result);
template<class I, class BI> 
  real_8 MakeRealFromBigUInt (const BigUInt<I, BI>& int_thing);

template<class I, class BI>
  bool MakeBigUIntFromBinary (const char* in_stream, size_t len,
			      BigUInt<I, BI>& int_return,bool sign_ext=false);

template<class I, class BI>
  string MakeBinaryFromBigUInt (const BigUInt<I, BI>& i, bool truncate=true);
				
template<class I, class BI>
  BigUInt<I,BI> MakeBigUIntFromBigInt (const BigInt<I,BI>& s, Allocator *a=0);

template <class BIGINT>
BIGINT StringToBigUIntHelper (const char* data, size_t len=size_t(-1), Allocator *a=0);

// Macros to pull stuff out of memory (binary format) either as
// BIG-ENDIAN or LITTLE-ENDIAN.  Most Intel machines are little-endian by
// default.
#if !defined(MACHINE_BIG_ENDIAN)
#define LOAD_FROM(T, IN, OUT) { T* rhs=((T*)(IN)); (*(OUT))=*rhs; }
#define LOAD_DBL(IN, OUT) { char* rhs=((char*)(IN)); char* lhs=((char*)(OUT)); lhs[0]=rhs[7]; lhs[1]=rhs[6]; lhs[2]=rhs[5]; lhs[3]=rhs[4]; lhs[4]=rhs[3]; lhs[5]=rhs[2]; lhs[6]=rhs[1]; lhs[7]=rhs[0]; }
#else 

#define LOAD_FROM(T, IN, OUT) { char*rhs=((char*)(IN)); char*lhs=((char*)(OUT));  for (int ii=0; ii<sizeof(T); ii++) lhs[ii]=rhs[sizeof(T)-1-ii]; }
//#define LOAD_FROM_4(RHS, LHS) { char*lhs=(char*)LHS; char*rhs=(char*)rhs; lhs[0] = rhs[3]; lhs[1] = rhs[2]; lhs[2]=rhs[1]; lhs[3]=rhs[0]; } 
#define LOAD_DBL(IN, OUT) { real_8* rhs=((real_8*)(IN)); real_8* lhs=((real_8*)(V)); *lrhs=*rhs; }

#endif


// //////////////////////////////////////////////

// A class for using arbitrarily-sized unsigned integers: The values
// are stored in an array of I, all math is done in the larger int BI
// so that we can detect overflows, carrys, etc.
template <class I, class BI>
class BigUInt {


  // Forwards for real-valued conversions
  friend class BigInt<I,BI>;
  friend void MakeBigUIntFromReal<>(real_8 r, BigUInt& result); 
  friend real_8 MakeRealFromBigUInt<>(const BigUInt& int_thing);
  friend bool MakeBigUIntFromBinary<>(const char* in_stream, size_t len,
				      BigUInt& int_return, bool sign_extend);
  friend string MakeBinaryFromBigUInt<>(const BigUInt& i, bool truncate);

  public:

  typedef I Base;
  typedef I MathInt;

  // Default constructor, always zero
  BigUInt () : data_(4) { data_.append(0); } // Too big because frequent expandTos for carrys and the like

 
  // Construct from a string
  BigUInt (const char* str, Allocator *a=0)
  {
    BigUInt result = StringToBigUIntHelper<BigUInt>(str, strlen(str), a);
    this->swap(result);
  }

  // Construct from a string
  BigUInt (const string& s, Allocator *a=0)
  {
    BigUInt result = StringToBigUIntHelper<BigUInt>(s.data(), s.length(), a);
    this->swap(result);
  }

#define BIGUINT_CONST(T) BigUInt (T plain, Allocator* a=0) : data_(sizeof(plain), a) { assignIt_(plain); }

  BIGUINT_CONST(int_1);
  BIGUINT_CONST(int_u1);
  BIGUINT_CONST(int_2);
  BIGUINT_CONST(int_u2);
  BIGUINT_CONST(int_4);
  BIGUINT_CONST(int_u4);
  BIGUINT_CONST(int_8);
  BIGUINT_CONST(int_u8);
  BIGUINT_CONST(ALLOW_SIZE_T);
  BIGUINT_CONST(ALLOW_LONG);
  BIGUINT_CONST(ALLOW_UNSIGNED_LONG);
  BIGUINT_CONST(ALLOW_LONG_LONG);
  BIGUINT_CONST(ALLOW_UNSIGNED_LONG_LONG);
  BIGUINT_CONST(real_4);
  BIGUINT_CONST(real_8);


  // Allow conversion from BigInt to BigUInt
  BigUInt (const BigInt<I, BI>& from, Allocator *a=0) 
  { 
    BigUInt temp = MakeBigUIntFromBigInt(from, a); 
    this->swap(temp);
  }


  // Interface change: until C++11 supports explicit outcasts,
  // we support only "as" to out convert the int_un.
  int_u8 as () 
  {     
    int_u8 result = 0;
    for (size_t ii=0; ii<sizeof(result)/sizeof(I) && ii<size_t(length()); ii++) {
      int_u8 temp = data_[ii];  // int_8 to contain it
      temp <<= (ii*(sizeof(I)<<3));     // shift over ii*sizeof(I)*8 bits
      result |= temp;                   // plop it in
    }
    return result;
  }
   
  operator string () const { return this->stringize(); }

  // Copy constructor
  BigUInt (const BigUInt& rhs, Allocator* a=0) : data_(rhs.data_, a) { }

  // Assignment
  BigUInt& operator= (const BigUInt& rhs) 
  { 
    // If space is there, just copy over and AVOID EXTRA ALLOCATION
    expandTo(rhs.length());
    set_sign(rhs.sign());
    memcpy(data_.data(), rhs.data_.data(), rhs.length()*sizeof(I));
    return *this; 
  }

  // Swap implementations in constant time
  void swap (BigUInt& rhs) { data_.swap(rhs.data_); }

  // Number of "digits": we are implemented using an array of I so
  // length counts the number of "I"s it takes to represent the given
  // number.
  size_t length () const { return data_.length(); }

  // Add to, aka +=
  BigUInt& operator+= (const BigUInt& rhs)
  {
    const size_t lhs_len = length();
    const size_t rhs_len = rhs.length();
    if (rhs_len==1) { // optimization
      return singleDigitAdd(rhs.data_[0]); 
    }
    static const BI mask = BI(I(~I(0))) << (sizeof(I)<<3); // 1s top, 0s bottom
    static const BI unmask = BI(~mask);                   // 0s top, 1s bottom

    // The answer will be the length of the bigger two: since this
    // is an inplace operation, we hope that the expandTo will simply
    // adjust a pointer and not do a realloc (but it may).
    int_ptr diff = lhs_len - rhs_len;
    size_t max_len = lhs_len;
    if (diff<0) { // *this is smaller
      expandTo(rhs_len);
      max_len = rhs_len;           // lhs_len captures ORIGINAL length
    }

    // Add all elements piecewise (with carry) up-to lhs len:
    // Assertion: lhs.length >= rhs.length()
    I* ldata = data_.data();
    const I* rdata = rhs.data_.data();
    const I zero = 0;
    BI carry = 0;
    for (size_t ii=0; ii<max_len; ii++) {
      BI lhs_piece = ii>=lhs_len ? zero : ldata[ii];
      BI rhs_piece = ii>=rhs_len ? zero : rdata[ii];
      BI sum = lhs_piece + rhs_piece + carry;
      carry = (mask & sum)>>(sizeof(I)<<3);
      BI digit = unmask & sum;
      ldata[ii] = digit;
    }
    if (carry) {
      data_.append(carry);
    }
    //normalize_();
    return *this;
    //return this->addOp(other, false); }
  }    
    
  // Bitwise complement
  BigUInt operator~ () const
  {
    BigUInt result(true, length());
    const I* peekdata = data_.data();
    I* pokedata = result.data_.data();
    size_t len = length();
    for (size_t ii=0; ii<len; ii++) {
      pokedata[ii] = ~peekdata[ii];
    }
    return result;
  }

  // Bitwise complement
  BigUInt& negate () 
  {
    I* data = data_.data();
    size_t len = length();
    for (size_t ii=0; ii<len; ii++) {
      data[ii] = ~data[ii];
    }
    return this->singleDigitAdd(1);
  }

  BigUInt& operator++ ()  // prefix
  { return singleDigitAdd(1); }

  BigUInt operator++ (int) // postfix
  { BigUInt temp(*this); singleDigitAdd(1); return temp; }

  BigUInt& operator-- ()  // prefix
  { return singleDigitSub(1); }

  BigUInt operator-- (int) // postfix
  { BigUInt temp(*this); singleDigitSub(1); return temp; }
    
  // Sub from, aka -=
  BigUInt& operator-= (const BigUInt& rhs) 
  {
    const size_t lhs_len = length();
    const size_t rhs_len = rhs.length();
    if (rhs_len==1) { // optimization
      return singleDigitSub(rhs.data_[0]); 
    }
    static const BI mask = BI(I(~I(0))) << (sizeof(I)<<3); // 1s top, 0s bottom
    static const BI unmask = BI(~mask);                    // 0s top, 1s bottom

    // The answer will be the length of the bigger two: since this
    // is an inplace operation, we hope that the expandTo will simply
    // adjust a pointer and not do a realloc (but it may).
    int_ptr diff = lhs_len - rhs_len;
    size_t max_len = lhs_len;
    if (diff<0) { // *this is smaller
      expandTo(rhs_len);
      max_len = rhs_len;           // lhs_len captures ORIGINAL length
    }

    // Add all elements piecewise (invert to subtract) (with carry)
    // up-to lhs len: Assertion: lhs.length >= rhs.length()
    I *ldata = data_.data();
    const I* rdata = rhs.data_.data();
    const I zero = 0;
    const I unzero = ~I(0);
    BI carry = 1;
    for (size_t ii=0; ii<max_len; ii++) {
      BI lhs_piece = ii>=lhs_len ?   zero : ldata[ii];
      I  rhs_piece = ii>=rhs_len ? unzero : ~rdata[ii];
      BI partial_sum = lhs_piece + rhs_piece;
      BI sum = partial_sum + carry;
      carry = (mask & sum)>>(sizeof(I)<<3);
      BI digit = unmask & sum;
      ldata[ii] = digit;
    }
    //if (carry) {
    //  data_.append(carry);
    //}
    normalize_();
    return *this;
  }

  // Mult from aka *=
  BigUInt& operator*= (const BigUInt& other)
  {
    // Optimization
    if (other.length()==1) {
      return singleDigitMultiply(other.data_[0]);
    }

    // General convolution
    static const BI mask = BI(I(~I(0))) << (sizeof(I)<<3);// all 1s top,0s bot
    static const BI unmask = BI(~mask);                 // 0s top, 1s bottom

    BigUInt result(false, length()+other.length()+ 4); // extra space for all the += , the 4 is just some slop
    result.data_.append(0);  // still just 0 ...

    // Normalize so lhs is longer one, rhs is shorter.  
    bool compare = other.length() >= length();
    const BigUInt& lhs = compare ? other : *this;
    const BigUInt& rhs = compare ? *this : other;
    // Assertion, len(lhs) is >= len(rhs)

    // Convolve
    const I* ldata = lhs.data_.data();
    const I* rdata = rhs.data_.data();            // 2 for extra space
    BigUInt intermediate_sum(false, lhs.length()+1+rhs.length()+2); 
    I* idata = intermediate_sum.data_.data();
    for (size_t ii=0; ii<rhs.length(); ii++) {
      // plop zeros in bottom of intermediate sum and reuse as temp each time
      for (size_t kk=0; kk<ii; kk++) {
	idata[kk] = 0;
      }
      // Do a single digit multiply
      BI carry = 0;
      size_t jj = 0;
      BI single_digit = rdata[ii];
      for (; jj<lhs.length(); jj++) {
	BI lhs_piece = ldata[jj];
	BI rhs_piece = single_digit; // rdata[ii];
	BI mult = lhs_piece * rhs_piece + carry;
	carry = (mask & mult)>>(sizeof(I)<<3);
	BI digit = unmask & mult;
	idata[jj+ii] = digit;
      }
      // Left over carry, make sure we keep it, otherwise we are too long
      idata[jj+ii] = carry;
      if (!carry) {
	intermediate_sum.expandTo(lhs.length()+ii);
      } else {
	intermediate_sum.expandTo(lhs.length()+ii+1);
      }
      // Assertion: intermediate_sum has the result of a digit multiply
      result += intermediate_sum;
    }
    this->swap(result);
    return *this;
  }

  // Inplace multiply this by a single digit: faster than general multiply
  BigUInt& singleDigitMultiply (I digit)
  {
    static const BI mask=BI(I(~I(0))) << (sizeof(I)<<3); // 1s top,0s at bottom
    static const BI unmask = BI(~mask); // 0s up top,01 at bottom
    // Do a single digit multiply
    I* idata = data_.data();
    BI carry = 0;
    BI rhs_piece = digit;
    size_t jj = 0;
    size_t len = length();
    for (; jj<len; jj++) {
      BI lhs_piece = idata[jj];
      BI mult = lhs_piece * rhs_piece + carry;
      carry = (mask & mult)>>(sizeof(I)<<3);
      BI digit = unmask & mult;
      idata[jj] = digit;
    }
    // Left over carry, make sure we keep it, otherwise we are too long
    if (carry) data_.append(carry);
    // All done
    return *this;
  }

  // Multiply by "base" and add a single digit: this is NOT shift in
  // the << or >> sense, more like shifting an entire I element.
  void shiftAdd (I digit)
  {
    if (!zero()) {
      expandTo(length()+1);
    }
    I* data = data_.data();
    for (size_t ii=length()-1; ii>=1; ii--) {
      data[ii] = data[ii-1];
    }
    data[0] = digit;
  }

  // Integer divide: give numerator and denominator, return div, remainder:
  // This uses a binary search technique which is slower: we keep it as
  // a way to compare the new "faster" DivMod against the old version.
  static void DivMod2 (const BigUInt& numerator, const BigUInt& denominator, 
		       BigUInt& divver, BigUInt& remainder)
  {
    // Simple case
    if (numerator<denominator) {
      divver = BigUInt(); // zero
      remainder = numerator;
      return;
    }
    // Full divide: 
    const size_t num_len = numerator.length();
    const size_t den_len = denominator.length();
    //divver = BigUInt(true, num_len-den_len+1);
    divver.expandTo(num_len-den_len+1); // Prefer expanding inplace to avoid excess new temporaries
    I* divver_data = divver.data_.data();

    // extract the first den_len "n"-1, digits on of the numerator,
    // starting from MSB: the nth digit will be dropped by loop below
    BigUInt extract(false, den_len); 
    //extract.expandTo(den_len);

    extract.expandTo(den_len-1); // n-1 digits avail, 1 more for grow
    for (int_ptr ii=int_ptr(den_len)-2; ii>=0; ii--) {
      extract.data_[ii] = numerator.data_[ii+(num_len-den_len+1)];
    } 

    // Got through the digits
    BigUInt just_below;
    for (int_ptr ii=int_ptr(num_len-den_len); ii>=0; ii--) {
      // Multiply accum by "base" and drop next digit from numerator
      extract.shiftAdd(numerator.data_[ii]); 
  
      // Choose the next "digit" to multiply, and get the value of 
      // the digit*numerator into just_below
      I div_digit = ChooseClosest(extract, denominator,
				  just_below);
      // Record digit, subtract off
      divver_data[ii] = div_digit;
      extract -= just_below;
    }
    // All through, anything leftover is the remainder
    divver.normalize_(); // *might* be extra zero at front
    remainder = extract;
  }
  
  // Given values n and m, choose the value k such that n>=m*k,
  // but n<m*(k+1).  Returns k as well as m*k (which is closest_mult).
  static BI ChooseClosest (const BigUInt& n, const BigUInt& m,
			   BigUInt& closest_mult)
  {
    BI result;
    BigUInt trial_mk(false, n.length()*2); // Enough space for large mults
    trial_mk.data_.append(0); // ... so is zero,but with some space for appends

    // Simple case: m>n, result is 0
    if (m>n) {
      result = 0;
      closest_mult.swap(trial_mk);
      return result;
    }

    // Use binary search to find the digit: bounds have to be signed
    // integer so they don't roll, int_8 is "big enough" for all 3 I
    // types we expect (int_u1, int_u2, int_u4)
    int_8 upper_bound = I(~(I(0))); // can't excede this as an upper bound
    int_8 lower_bound = 0;
    int_8 mid;
    int compare = -1;
    while (upper_bound>=lower_bound) {
      mid =(upper_bound + lower_bound)>>1; 
      trial_mk = m;
      trial_mk.singleDigitMultiply(mid);  // this avoids extra copies
      compare = trial_mk.threeWayCompare(n);
      switch (compare) {
      case -1: lower_bound = mid+1; break;  
      case  0: closest_mult.swap(trial_mk); return mid; break; // found!
      case +1: upper_bound = mid-1; break;
      }
    }
    // Out of bounds: so we "expected" the answer 
    result=lower_bound-1;
    trial_mk = m;
    trial_mk.singleDigitMultiply(result);  // this avoids extra copies
    closest_mult.swap(trial_mk);
    return result;
  }

  // Single digit division and remainder
  static void singleDigitDivide (const BigUInt& u, I v,
				 BigUInt& q, I& r)
  {
    static BI baseshift = sizeof(I)<<3; // how many bits to shift to *,/ by base

    // Error and simple cases
    if (v==0) 
      throw runtime_error("Division by zero");
    if (u.zero()) {
      q = 0;
      r = 0;
      return;
    }
    // Final answer will have m places (may be some zeroes to contact)
    const int m = u.length();
    q.expandTo(m);

    // Constants
    I* q_ = q.data_.data();
    const I* u_ = u.data_.data();
    int_8 k = 0;
 
    for (int j=m-1; j>=0; j--) {
      q_[j] = (BI(k<<baseshift) + u_[j])/v;
      k = (BI(k<<baseshift) + u_[j]) - BI(q_[j]*v);
    }
    r = k;
    
    // Get rid of leading zeroes
    q.normalize_();
  }


  // DivMod: new, faster implementation.
  static void DivMod (const BigUInt& u, const BigUInt& v,
		      BigUInt& q, BigUInt& r)
  {
    // DivMod based on Knuth, Seminumerical Algorithms and Hacker's 
    // Delight impl.  The Knuth implementation chooses the "proper" qhat 
    // much faster (constant time) rather than logarithmic time of binary
    // search.  The normalization steps make this a touch slower
    // than the bsearch method for smaller numbers, but for larger numbers 
    // it can make a huge difference (2x).
    

    // Constants
    static const BI baseshift = sizeof(I)<<3; 
    static const BI b         = BI(1) << baseshift;
    static const BI basemask  = b - 1;          // all ones for I
    int m = u.length(); const I *u_ = u.data_.data();
    int n = v.length(); const I *v_ = v.data_.data();

    if (n==1) { // special case: 1 digit (also handles 0)
      I small_int_r;
      singleDigitDivide(u, v_[0], 
			q, small_int_r);
      r = small_int_r;
      return;
    } else if ((m < n) ||
	       (m==n && u_[m-1]<v_[n-1])) {  // u < v so no need to do anything
      q = 0; r = u;
      return;
    } 
    // Assertion: u >=v, m and n both >= 1

    // vars
    BI qhat, rhat;
    BI p;          // product of 2 digits
    int s;
    int_8 t, k; // TODO: have template choose int_8, int_4, int_2

    // Normalize (in the Knuth sense, getting the v above v/2
    // so we can apply the qhat check)
    // by shifting v left just enough so that its high-order
    // bit is on, and shift u left the same amount.  We may may have
    // to append a high-order digit on the dividend; we do that
    // unconditionally
    Array<I> uu(2*(m+1)), vv(2*(n));  // on heap, alloca may fail for very
                                    // large numbers: dealloc auto
    I *un = uu.data();
    I *vn = vv.data();
    s = nlz(BI(v_[n-1])) - baseshift;

    // Remainders:
    q.expandTo(m-n+1);
    r.expandTo(n);
    I *q_ = q.data_.data();
    I *r_ = r.data_.data();

    { // Normalize vn and un: Step D1 of Knuth

      if (s==0) { // shift by number of bits > int is undefined for C
	// Shift vn 
	for (int i=n-1; i>0; i--) {
	  vn[i] = v_[i];
	}
	vn[0] = v_[0];
	// Shift un
	un[m] = 0;
	for (int i=m-1; i>0; i--) {
	  un[i] = u_[i];
	}
	un[0] = u_[0];
      } else { // these shifts all defined
	// Shift vn 
	for (int i=n-1; i>0; i--) {
	  vn[i] = (v_[i] << s) | (v_[i-1] >> (baseshift-s));
	}
	vn[0] = v_[0] << s;
	// Shift un
	un[m] = u_[m-1] >> (baseshift-s);
	for (int i=m-1; i>0; i--) {
	  un[i] = (u_[i] << s) | (u_[i-1] >> (baseshift-s));
	}
	un[0] = u_[0] << s;
      }
    }

    // D2
    for (int j=m-n; j>=0; j--) { // main loop 

      // D3: Compute estimate qhat of q[j]
      //if (un[j+n] >= vn[n-1]) {
      //qhat = b-1;
      //} else 
      {
	qhat = ((BI(un[j+n])<<baseshift) + un[j+n-1])/BI(vn[n-1]);
	rhat = ((BI(un[j+n])<<baseshift) + un[j+n-1]) - qhat * vn[n-1]; 

    qhatfix:
	if ((qhat>=b) || 
	    ( qhat*vn[n-2] > (rhat<<baseshift) + un[j+n-2]) ) {
	  qhat -= 1;
	  rhat += vn[n-1];
	  if (rhat < b) goto qhatfix;
	}
      }
      
      // D4: Multiply and subtract
      k = 0;
      for (int i=0; i<n; i++) {
	p = qhat *vn[i];
	t = un[i+j] - k - (p & basemask);  // The p keeps in BI range
	un[i+j] = t;
	k = (p>>baseshift) - (t>>baseshift);
      }
      t = un[j+n] -k;
      un[j+n] = t;
      
      // D5 test remainders
      q_[j] = qhat;   // store quotient digit
      if (t<0) {     // D6: If we subtracted too
	q_[j] -= 1;   // much, add back
	k = 0;
	for (int i=0; i<n; i++) {
	  t = BI(un[i+j]) + BI(vn[i]) + k; // Coersions needed to keep in BI
	  un[i+j] = t;
	  k = t >> baseshift;
	}
	un[j+n] += k;
      }

    } // D7: loop on j

    // D8: Unnormalize remainder
    for (int i=0; i<n; i++) {
      r_[i] = (un[i] >> s) | (un[i+1] << (baseshift-s)); 
    }

    // get rid of excess zeros
    q.normalize_();
    r.normalize_();
  }

  // div =
  BigUInt& operator/= (const BigUInt& other)
  {
    BigUInt divver, rem;
    DivMod(*this, other, divver, rem);
    this->swap(divver);
    return *this;
  }

  // div =
  BigUInt& operator%= (const BigUInt& other)
  {
    BigUInt divver, rem;
    DivMod(*this, other, divver, rem);
    this->swap(rem);
    return *this;
  }

  ostream& dump (ostream& os) 
  {
    for (size_t ii=0; ii<length(); ii++) {
      os << int_u8(data_[ii]) << " ";
    }
    os << endl;
    return os;
  }

  // Detect if zero
  bool zero () const { return length()==1 && data_[0] == 0; }
  
  // Print: this is used by operator<<
  ostream& print (ostream& os, int default_base = 10) const
  {
    static char base_lookup[] = "0123456789ABCDEF";
    if (default_base>16 || default_base<2) {
      throw runtime_error("Illegal base for print: only base 2-16");
    }
    if (zero()) {
      return os << "0";
    } 
    BigUInt res(*this);
    Array<char> pr(length());
    BigUInt divver, rem;
    BigUInt base(default_base);
    while (!res.zero()) {
      DivMod(res, base, divver, rem);
      I lsb = rem.data_[0]; // cout << "lsb = " << lsb << endl;
      char single_digit = base_lookup[lsb];
      pr.append(single_digit);
      res.swap(divver);
    }
    // In reverse order:
    for (int_ptr ii=int_ptr(pr.length())-1; ii>=0; ii--) {
      os << pr[ii];
    }
    return os;
  }

  // Optimized string output function
  string stringize (int default_base = 10, char prefix=' ') const
  {
    static char base_lookup[] = "0123456789ABCDEF";
    if (default_base>16 || default_base<2) {
      throw runtime_error("Illegal base for print: only base 2-16");
    }
    BigUInt res(*this);
    size_t LEN = length() * 10;
    char* a = (char*)alloca(LEN); // Enough space to stringize into based on size of int: don't waste space!
    int ii=LEN-1;
    BigUInt divver, rem;
    BigUInt base(default_base);
    do {
      DivMod(res, base, divver, rem);
      I lsb = rem.data_[0]; // cout << "lsb = " << lsb << endl;
      char single_digit = base_lookup[lsb];
      a[ii--] = single_digit;
      res.swap(divver);
    } while (!res.zero());
    if (prefix!=' ') {
      a[ii--] = prefix;
    }
    return string(&a[ii+1], LEN-ii-1);
  }

  

  // Helper: returns -1 is this<other 0 if this==other, 1 if this>other
  int threeWayCompare (const BigUInt& other) const
  {
    if (length()<other.length()) return -1;
    else if (length()>other.length()) return +1;
    else { // same, compare point wise!  Start at BIG end
      const I* ldata = this->data_.data();
      const I* rdata = other.data_.data();
      for (int ii=int(length())-1; ii>=0; ii--) {
	if (ldata[ii]<rdata[ii]) return -1;
	else if (ldata[ii]>rdata[ii]) return +1;
      }
      // Exactly equal
      return 0;
    }
  }

  // += but only for a single digit: optimization: This can be a significant
  // speed increase to not have to create a BigUInt first
  BigUInt& singleDigitAdd (I single_digit)
  {
    static const BI mask = BI(I(~I(0))) << (sizeof(I)<<3); // 1s top, 0s bottom
    static const BI unmask = BI(~mask);                   // 0s top, 1s bottom

    // Add all elements piecewise (with carry) up-to lhs len:
    // Assertion: lhs.length >= rhs.length()
    const int lhs_len = length();
    I* ldata = data_.data();
    BI carry = single_digit;
    for (int ii=0; ii<lhs_len; ii++) {
      BI lhs_piece = ldata[ii];
      BI sum = lhs_piece + carry;
      carry = (mask & sum)>>(sizeof(I)<<3);
      BI digit = unmask & sum;
      ldata[ii] = digit;
    }
    if (carry) {
      data_.append(carry);
    }
    return *this;
  }

  // -= but only for single digit: This can be significant speedup
  // to NOT have to create a BigUInt first
  BigUInt& singleDigitSub (I single_digit)
  {
    static const BI mask = BI(I(~I(0))) << (sizeof(I)<<3); // 1s top, 0s bottom
    static const BI unmask = BI(~mask);                   // 0s top, 1s bottom

    if (length()==1) { // optimization
      BI other_digit = data_[0];
      other_digit -= single_digit;
      data_[0] = other_digit & unmask;
      return *this;
    }

    // Add all elements piecewise (with carry) up-to lhs len:
    // Assertion: lhs.length >= rhs.length()
    const size_t lhs_len = length();
    I* ldata = data_.data();
    I other = ~single_digit;
    I unzero = ~I(0);
    BI carry = 1;
    for (size_t ii=0; ii<lhs_len; ii++) {
      BI lhs_piece = ldata[ii];
      BI sum = lhs_piece + other + carry;
      carry = (mask & sum)>>(sizeof(I)<<3);
      BI digit = unmask & sum;
      ldata[ii] = digit;
      other = unzero;
    }
    //if (carry) {
    //  data_.append(carry);
    //}
    normalize_();
    return *this;
  }

  // always 0 for an unsigned
  size_t sign () const       { return data_.getBit(); }
  void set_sign (size_t sign) { data_.setBit(sign); }

  const I* data() const { return data_.data(); }
        I* data()       { return data_.data(); }

   Allocator* allocator () const { return data_.allocator(); }

  protected:

  // ///// Data Members

  // Implementation: array of digits: held as I, all math done in BI.

  // Why don't we use vector?  The "problem" with using vector as a
  // holder is that I am not allowed to create an array of n elements,
  // where there are uninitialized ... this can be a slowdown, as
  // initialization "costs" time that we are trying to minimize.
  // Since we know the I is a POD type, we do expandTo all over
  // the place to shrink and grow it.
  Array<I> data_;

  // Protected constructor: allows us to build integers of the right
  // size for intermediate reprs.
  BigUInt (bool set_len, int initial_len) : 
    data_(initial_len) 
  { if (set_len) expandTo(initial_len); }

  // Helper function to assign to data_ from an int_8
  template <class T>
  void assignIt_ (T arg) 
  {
    int_8 plain = static_cast<int_8>(arg);
    const BI mask = ~I(0) ; // all 1s up top,0s at bottom
    for (size_t ii=0; ii<sizeof(plain)/sizeof(I); ii++) {
      data_.append(plain & mask);
      plain >>= (sizeof(I)<<3);
    }
    normalize_();
  }

  // Get rid of excess zeros at the front to keep the impl minimal:
  // zero needs to be length==1 and data_[0]==0
  void normalize_()
  {
    size_t mark = length();
    I* data = data_.data();
    size_t ii=length()-1;      // Can't be length of zero
    for (; ii>0; ii--) {  // .. never make one that's of length 0
      if (data[ii]==0) {
	mark = ii;
	continue;
      } else {
	break;
      }
    }
    expandTo(mark);
  }

 public:
  // Returns: Number of zero bits to left of most significant bit
  // (Time complexity: Logartithmic algorithm, in number of bits of II)
  // for int_u4, nlz(0) = 32  (all zero bits)
  //             nlz(1) = 31  (all zeroes except 1)
  //             nlz(2) = 30  (zeroes in bits 31..2)
  //             nlz(65535) = 16 (zeroes in bits 31..16)
  //             nlz(65536) = 15 (zeroes in bits 31..15)
  // ONLY WORKS FOR int_u8, int_u4, int_u2, int_u1
  template <class II>
  static int nlz (II x) 
  {
    //cerr << "In nlz x=" << x << endl;
    static const unsigned int baseshift = sizeof(II)<<3; // 32, 16, or 8
    if (x==0) return baseshift;

    int n = 0;
    unsigned int power_of_two = baseshift>>1;
    II mask = II(~II(0)) >> (baseshift>>1);  // mask is half of lower half of base, all 1s

    while (power_of_two) {
      if (x <= mask) {
	n += power_of_two; 
	x = x << power_of_two;
      }
      power_of_two = power_of_two >> 1;
      mask = (mask << (power_of_two)) | mask;
    }
    //cerr << n << endl;
    return n;
  }

  // Just like impl expandTo (For future use: may want to intercept)
  void expandTo (size_t new_size) { data_.expandTo(new_size); }

}; // BigUInt

/*
// Comparsions
template <class I, class BI>
inline bool operator> (const BigUInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)==1; }

template <class I, class BI>
inline bool operator>= (const BigUInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)>=0; }

template <class I, class BI>
inline bool operator< (const BigUInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)==-1; }

template <class I, class BI>
inline bool operator<= (const BigUInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)<=0; }

template <class I, class BI>
inline bool operator== (const BigUInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)==0; }

template <class I, class BI>
inline bool operator!= (const BigUInt<I,BI>& lhs, const BigUInt<I,BI>& rhs)
{ return lhs.threeWayCompare(rhs)!=0; }
*/

#define BIGUINT_SPEC_OP(I,BI,OP) inline bool operator OP (const BigUInt<I,BI>& lhs, const BigUInt<I,BI>& rhs) { return lhs.threeWayCompare(rhs) OP 0; }



#define BIGUINT_SPEC(I,BI) \
BIGUINT_SPEC_OP(I,BI,==);\
BIGUINT_SPEC_OP(I,BI,!=);\
BIGUINT_SPEC_OP(I,BI,<);\
BIGUINT_SPEC_OP(I,BI,<=);\
BIGUINT_SPEC_OP(I,BI,>);\
BIGUINT_SPEC_OP(I,BI,>=);

BIGUINT_SPEC(int_u1, int_u2);
BIGUINT_SPEC(int_u2, int_u4);
BIGUINT_SPEC(int_u4, int_u8);


// Basic ops: +, -, /, *
/*
template <class I, class BI>
inline BigUInt<I,BI> operator+ (const BigUInt<I,BI>& lhs,const BigUInt<I,BI>& rhs)
{ 
  BigUInt<I,BI> result = lhs;
  return result += rhs;
}

template <class I, class BI>
inline BigUInt<I,BI> operator- (const BigUInt<I,BI>& lhs,const BigUInt<I,BI>& rhs)
{ 
  BigUInt<I,BI> result = lhs;
  return result -= rhs;
}

template <class I, class BI>
inline BigUInt<I,BI> operator* (const BigUInt<I,BI>& lhs,const BigUInt<I,BI>& rhs)
{ 
  BigUInt<I,BI> result = lhs;
  return result *= rhs;
}

template <class I, class BI>
inline BigUInt<I,BI> operator/ (const BigUInt<I,BI>& lhs,const BigUInt<I,BI>& rhs)
{ 
  BigUInt<I,BI> divver; 
  BigUInt<I,BI> rem;
  BigUInt<I,BI>::DivMod(lhs, rhs, divver, rem);
  return divver;
}

template <class I, class BI>
inline BigUInt<I,BI> operator% (const BigUInt<I,BI>& lhs,const BigUInt<I,BI>& rhs)
{
  BigUInt<I,BI> rem; 
  BigUInt<I,BI> divver; 
  BigUInt<I,BI>::DivMod(lhs, rhs, divver, rem);
  return rem;
}
*/
#define BIGUINT_MATHOP(I,BI,OP) \
inline BigUInt<I,BI> operator OP (const BigUInt<I,BI>& lhs,const BigUInt<I,BI>& rhs) \
{ BigUInt<I,BI> result=lhs; return result OP##= rhs; }

#define BIGUINT_MATHOP_DIV(I,BI,OP,WHICH) \
inline BigUInt<I,BI> operator OP (const BigUInt<I,BI>& lhs,const BigUInt<I,BI>& rhs) \
{ BigUInt<I,BI> rem; BigUInt<I,BI> divver; BigUInt<I,BI>::DivMod(lhs, rhs, divver, rem); return WHICH; }

#define BIGUINT_MATHOP_DEF(I,BI) \
  BIGUINT_MATHOP(I,BI, +); \
  BIGUINT_MATHOP(I,BI, -); \
  BIGUINT_MATHOP(I,BI, *); \
  BIGUINT_MATHOP_DIV(I,BI, /, divver); \
  BIGUINT_MATHOP_DIV(I,BI, %, rem); 


BIGUINT_MATHOP_DEF(int_u1, int_u2);
BIGUINT_MATHOP_DEF(int_u2, int_u4);
BIGUINT_MATHOP_DEF(int_u4, int_u8);

// Stream operations
template <class I, class BI>
inline ostream& operator<< (ostream& os, const BigUInt<I,BI>& rhs)
{ return rhs.print(os); }


// Convert from a real_8 to a BigUInt: All fractional parts will be
// dropped.
template <class I, class BI>
inline void MakeBigUIntFromReal (real_8 r, 
				 BigUInt<I,BI>& result) 
{
  result = BigUInt<I,BI>();
  // Just return 0 if negative: impl-defined anyway

  // We have to divide out the "max I" everytime to convert
  static BI int_max_divver = BI(1)<<(sizeof(I)<<3); // I(~I(0))+1;
  real_8 max_divver = int_max_divver; // Note: This can be a problem if the real_8 can't hold without dropping precision, as long as number of bits of precision of real_8 > number of bits of _I_, should be okay.
  
  if (r>0) { 
    result.expandTo(0);
    real_8 int_part = 0.0, divout = 0.0;
    modf(r, &int_part);
    // Turn int part into BigUInt
    do {
      // Get digit out
      real_8 rem = fmod(int_part, max_divver);
      I digit = I(rem);
      result.data_.append(digit);
      
      // Shift real_8 down
      divout = int_part/max_divver;
      modf(divout, &int_part);
    } while (divout != 0.0);
    result.normalize_();
  } else { // r<0
    int_u8 lt = int_u8(r); // pretend its an int_u8 and just 
    result = lt; 
  }
}


// Convert a BigInt into a real_8: because of complex template
// interactions, we choose to make this a function for now.  TODO:
// should this be bit-twiddling to make sure we get all the precision?
// (Also may be faster but much less portable)
template<class I, class BI>
inline real_8 MakeRealFromBigUInt (const BigUInt<I, BI>& int_thing)
{
  // We have to divide out the "max I" everytime to convert
  static BI int_max_divver = I(~I(0)) + 1;
  real_8 max_divver = int_max_divver; // Note: This can be a problem if the real_8 can't hold without dropping precision, as long as number of bits of precision of real_8 > number of bits of _I_, should be okay.
  
  // Start at lsb to make sure we keep as mch precision as possible:
  // when real_8 gets big, precision will be dropped, so it's better
  // to have the precision in and have it get dropped.
  real_8 result = int_thing.data_[0];
  for (size_t ii=1; ii<int_thing.length(); ii++) {
    result += max_divver*real_8(int_thing.data_[ii]);
    max_divver *= real_8(int_max_divver);
  }
  return result;
}

inline bool IsLittleEndian () 
{
  int_u4 test = 1;
  char* bin_data = reinterpret_cast<char*>(&test);
  return (*bin_data==1);
}

// From a 2s complement representation, create a BigInt: this assumes
// the input stream is little endian.  You can also sign extend if you
// need to.  Returns whether the 2s complement repr would be negative
// (true) or positive (false).
template<class I, class BI>
inline bool MakeBigUIntFromBinary (const char* in, size_t len,
				   BigUInt<I, BI>& int_return,
				   bool sign_extend)
{
  if (len==0) { 
    int_return = BigUInt<I,BI>(); 
    return false;
  }
  char last_byte = in[len-1];
  bool is_negative = ((last_byte>>7)&1)==1;

  // Figure out how much space this will take up
  size_t major = len/sizeof(I);
  size_t minor = len%sizeof(I);  // Will need extra append if minor!=0
  int_return.expandTo(major); 
  
  // Plop the data in
  I* data = int_return.data_.data();
  for (size_t ii=0; ii<major; ii++) {
    I temp;
    LOAD_FROM(I, in, &temp);
    data[ii] = temp;
    in += sizeof(I);
  }
  // There may be a leftover piece we still have to plop in:
  //  Create a quick padded (with 0s) so we can use LOAD_FROM
  if (minor) {
    I temp;
    char padded_in_stream[sizeof(I)];
    char pad = (is_negative && sign_extend)? 0xff : 0;
    for (size_t ii=0; ii<sizeof(I); ii++) {
      padded_in_stream[ii] = (ii<minor) ? in[ii] : pad;
    }
    LOAD_FROM(I, padded_in_stream, &temp);
    int_return.data_.append(temp);
    in += minor;
  }
  return is_negative;
}

// Take data from our class and output a little endian buffer. 
// This is the complement to MakeBigUIntFromBinary
template<class I, class BI>
  string MakeBinaryFromBigUInt (const BigUInt<I, BI>& i, bool truncate)
{
  // This is an overapproximation: the upper I may have zeroes
  // in the MSB positions
  string result(i.length()*sizeof(I), '\0');  // RVO
  char* result_data = const_cast<char*>(result.data());

  const I* data = i.data_.data();
  int_8 kk = 0;
  const size_t additive = IsLittleEndian() ? 0 : sizeof(I)-1;
  for (size_t ii=0; ii<i.length(); ii++) {
    I single = data[ii];
    char* bin_data = reinterpret_cast<char*>(&single);
    for (size_t jj=0; jj<sizeof(I); jj++) {
      result_data[kk++] = bin_data[jj ^ additive];
    }
  }
  
  // Get rid of extraneous zeros
  if (truncate) {
    while (kk>0 && result_data[--kk]==0) ;
    result.resize(kk+1);
  }
  return result;
}

/*
// Take data from our class and output a little endian buffer. 
// This is the complement to MakeBigUIntFromBinary
template<class I, class BI>
  string MakeBinaryFromBigUInt (const BigUInt<I, BI>& i)
{
  // This is an overapproximation: the upper I may have zeroes
  // in the MSB positions
  string result(i.length()*sizeof(I), '\0');  // RVO
  char* result_data = const_cast<char*>(result.data());

  const I* data = i.data_.data();
  int kk = 0;
  const int additive = IsLittleEndian() ? 0 : sizeof(I)-1;
  for (int ii=0; ii<i.length(); ii++) {
    I single = data[ii];
    char* bin_data = reinterpret_cast<char*>(&single);
    for (size_t jj=0; jj<sizeof(I); jj++) {
      result_data[kk++] = bin_data[jj ^ additive];
    }
  }

  // Get rid of extraneous zeros
  while (kk>0 && result_data[--kk]==0) ;
  // sign_bit = (result_data[kk]>>7); // 0 or 1
  result.resize(kk+1);
  return result;
}
*/

// Do integer exponentiation in log time: typically BI will be int_un
// and I will be a plain int (for speed), but for really big
// exponentiations, you'll want I to be int_un too.
template <class I, class BI>
inline BI IntExp (BI base, I power)
{
  BI result = 1;
  BI current_base = base;
  while (power!=0) {
    if (power & 1) result *= current_base;
    current_base *= current_base;
    power >>= 1;
  }
  return result;
}

// Give a decimal approximation of numerator/divisor with precision
// places AFTER the decimal sign.  BIG_INT is probably just int_un.
template <class I, class BI>
inline string DecimalApproximation (const BigUInt<I,BI>& num, 
				    const BigUInt<I,BI>& divisor, 
				    int_u8 precision)
{
  string result; // RVO

  // Figure out if there is an integer part, and print it
  BigUInt<I,BI> d,remain;
  BigUInt<I,BI>::DivMod(num, divisor,
		  d, remain);
  result = Stringize(d);
  if (!remain.zero() && precision!=0) {

    // We want this many places of precision in the decimal
    BigUInt<I,BI> places = remain*IntExp(BigUInt<I,BI>(10), precision);
    BigUInt<I,BI> int_result, r2;
    BigUInt<I,BI>::DivMod(places, divisor,
                   int_result, r2);
    if (r2>(divisor/BigUInt<I,BI>(2))) {  // Round up if needed 
      ++int_result;
    }
    result += "." + Stringize(int_result);
  }
  return result;
}




// What's the best way to use BigUInts on a machine?  It seems that
// on 32-bit machines: BigUInt<int_u2, int_u4> is faster
// on 64-bit machines: BigUInt<int_u4, int_u8> is faster
// Use technique from Modern C++ Design to figure out how big pointers
// are: this way we can choose the better implementation for int_un
template <bool flag, typename T, typename U>
struct SelectBigUInt {
  typedef T Result;
};
template <typename T, typename U>
struct SelectBigUInt<false, T, U> {
  typedef U Result;
};
typedef SelectBigUInt<(sizeof(void*)==4), int_u2, int_u4>::Result Smaller_uint;
typedef SelectBigUInt<(sizeof(void*)==4), int_u4, int_u8>::Result Bigger_uint;

typedef BigUInt<Smaller_uint, Bigger_uint> int_un;


inline string DecimalApprox (const int_un& num, const int_un& den, 
			     int_u8 precision)
{ return DecimalApproximation(num, den, precision); } 


// Turn an ASCII string into an uint
template <class BIGINT>
inline BIGINT StringToBigUIntHelper (const char* data, size_t len, Allocator* a)
{
  BIGINT result(0, a);  // RVO
  if (len==size_t(-1)) len=strlen(data);
  char c=' ';
  char sign = '\0';
  size_t ii;
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

inline int_un StringToBigUInt (const char* data, int /*len*/, Allocator* a=0)
{ return StringToBigUIntHelper<int_un>(data, int(strlen(data)), a); }

inline int_un StringToBigUInt (const char* data, Allocator *a=0)
{ return StringToBigUInt(data, strlen(data), a); }

inline int_un StringToBigUInt (const string& s, Allocator *a=0)
{ return StringToBigUInt(s.data(), s.length(), a); }



OC_END_NAMESPACE

#endif // BIGUINT_H_



