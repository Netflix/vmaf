#ifndef OC_COMBINATIONS_H_
#define OC_COMBINATIONS_H_

#include "ocarray.h"

OC_BEGIN_NAMESPACE

// Compute all combinations of n choose k.  This computes all n choose
// k of combinations of the distinct integers 1..n.  Note that the
// permutations are permuted in sorted order. Other data can be
// "combinated" by using the numbers here as indicies into an array.

// // Typical usage
// #include "occombinations.h"
// {
//   Combinations c(n,k);
//   do {
//     int* cur = c.currentCombination();
//     printCurrentCombination(cur, n, k);
//   }  while (c.nextCombination());
//
class Combinations {

 public:
   
  // Create an engine to compute all n choose k distinct combinations
  // of 1..n
  Combinations (int n, int k) :
    n_(n),
    k_(k),
    data_()
  {
    if (n_<=0 || k_<0 || n<k) {
      throw runtime_error("Combinations only works for n>0,k>=0, n>=k");
    }
    for (int ii=0; ii<k; ii++) {
      data_.append(ii+1);
    }
  }

  // "Seed" the combinations to start at a particular one:  a nextCombination
  // will continue in sorted order from here.  This is for breaking
  // up combinations into parallel computation.  Example:
  // Combinations c2(8,2); // all combinations, in order
  // int seeder[] = { 5, 1 }; // ... start the combinations at 5
  // c2.seed(seeder);
  void seed (int *seed)
  {
    for (int ii=0; ii<k_; ii++) {
      data_[ii] = seed[ii];
    }
  }

  // Compute the nextCombination: if there isn't one (as the current
  // combination is the last in the sorted sequence of
  // combinations), return false.  Otherwise, return true to
  // indicate currentCombination holds the next combination.
  bool nextCombination () 
  { 
    // Some error checking: makes n choose 0 work.
    if (k_<=0) return false;

    // Common case, last element changing
    int* data = data_.data();
    if (data[k_-1] < n_) {
      data[k_-1]+=1;
      return true;
    }

    // Assertion, last element is n.  Find previous element that isn't
    // n_-i ... this represents where we should "restart" the next set
    // of combinations
    int ii=k_-2;
    for (int jj=1; ii>=0; ii--, jj++) {
      if (data[ii] != n_-jj) break;
    }
    
    // Assertion: If ii==-1, then we discovered we are at last combination
    if (ii==-1) return false;

    // Assertion, we found the next element that's goes up 1.
    // Fill in the rest of the data with the next elements
    int updata = data[ii];
    for (int jj=ii; jj<k_; jj++) {
      data[jj] = ++updata;
    }
    return true;
  }

  // Current combination: k elements of array ... can't change!
  int const* currentCombination () { return data_.data(); }

 protected:

  // compute all n choose k combinations: The data array contains the
  // integers 1..n, and based on the current state of the array, we go
  // to the next.
  int n_, k_;
  Array<int> data_;
 
}; // Combinations


OC_END_NAMESPACE

#endif // OC_COMBINATIONS_H_
