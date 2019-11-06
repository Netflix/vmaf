#ifndef OCRANDOMIZER_H_

#include "ocport.h"
#include <math.h>

// Allow a user to generate pseudo-random numbers exactly one between
// 0 and n-1 wth O(1) space/time. Usage:
//
//  Randomizer r(100);  // all numbers 0..99 exactly once 
//  for (int ii=0; ii<100; ii++) {
//     cerr << r.next() << endl;  // generate the next random number
//  }
// 
// This is accompilished using constant space and constant time
// (amortized) for each number generated.  All numbers between 0 and
// n-1 are generated EXACTLY once.  Note: if you need to generate all
// numbers between 0 and 1 billion randomly exactly once, it requires
// NO EXTRA SPACE!!!  This is implemented using the idea that all
// elements of group G (Abstract Algebra sense) of prime order are
// generators of the group.  So, pick any element a of G (not the
// identity) and a^0, a^1, ... a^(n-1) are all unique as a generates
// the group.

OC_BEGIN_NAMESPACE

// Helper function: Look down the given integer and find the position
// of the highest 1 bit.  Returns -1 if all bits are 0.  The bits are
// numbered from 0 .. n, where bit 0 is the lowest bit.
template <class INT>
inline int FindPositionOfHighest1Bit (INT n)
{
  int mark=-1;
  INT one = 1;  // Make sure we are doing masks with same type of ints
  const int bits = int(sizeof(INT)*8);
  for (int ii=0; ii<bits; ii++) {
    if ((n&one)==one) mark = ii;
    n >>=1;
  }
  return mark;
}

// Helper function: A quick overestimate of the sqrt: can be bad, but
// gives a reasonable bound.
template <class INT>
inline INT IntSqrtOverestimate (INT n)
{
  int pos = FindPositionOfHighest1Bit(n);
  return 1+(n >> (pos)/2);
}

// Helper function: Check if an int is prime or not.
inline bool CheckPrimality (int_u8 n)
{
  if (n<=17) {
    if (n==2 || n==3 || n==5 || n==7 || n==11 || n==13 || n==17) return true;
    else return false;
  } 
  if (n%2==0 || n%3==0 || n%5==0 || n%7==0 || n%11==0 || n%13==0 || n%17==0)
    return false;
  int_u8 sq_root = 0; // IntSqrtOverestimate(n);
  real_8 dn = n;
  if (dn > 1e15) { // double precision loss, can't take chance.
    sq_root = IntSqrtOverestimate(n); // Quick overestimate
  } else {
    sq_root = int_u8(sqrt(dn)+1); // slight overestimate since double can lose precision
  }
  for (int_u8 ii=19; ii<=sq_root; ii+=2) {
    if (n%ii==0) return false;
  }
  return true;
}

// Helper function; Return the first prime above the given number
inline int_u8 PrimeNumberAbove (int_u8 n)
{
  if (n%2==0) n++; // Should be odd!
  for (; !CheckPrimality(n); n+=2)
    ;
  return n;
}


// Produce all numbers from 0..n-1 in psuedo-random order in constant
// space.
class Randomizer {
 public:
  Randomizer (int_u8 n) : n_(n)
  {
    // Pick a number above n that is prime: choose a random place
    // above n to start looking: these keeps the "extra slop" within n
    // so that we will iterate over "2n" overall.
    int_u8 place_to_start = random() % n;
    groupSize_ = PrimeNumberAbove(n+place_to_start);

    // make sure the last element isn't always 0 by having what
    // the user sees "offset" by this amount: so instead
    // of the last element being 0, it be offset_
    offset_ = random() % n;

    // In a group of prime size, all elements are generators: pick
    // some random element from group and start there: obviously 0
    // won't work as a starting point with this method.
    do {
      element_ = random() % groupSize_;
    } while (element_==0);
    next_ = 0;
  }

  int_u8 next ()
  {
    // This is amortized constant generation: over n calls to next we
    // are guaranteed to do about 2n while loops: most of the time
    // it's just one or two "extra iterations" of the while loop.
    int_u8 retval = next_;
    do {
      retval = (element_+retval) % groupSize_;
    } while (retval>=n_);
    next_ = retval;
    // Used to just return next_ directly, but that makes the
    // last element ALWAYS 0, so we have an offset that add to
    // every element (mod n) so we "offset": basically the last
    // element should have whatever value we choose for offset.
    retval = (next_ + offset_) % n_;
    return retval;
  }

 protected:
  int_u8 groupSize_; // The prime order of the group
  int_u8 next_;      // a^next
  int_u8 element_;   // the generator of the group
  int_u8 n_;         // limit the size of the numbers generated (n_<groupSize_)
  int_u8 offset_;    // Last element always 0 unless we add an extra offset
}; // Randomizer

OC_END_NAMESPACE

#define OCRANDOMIZER_H_
#endif // OCRANDOMIZER_H_
