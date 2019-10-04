#ifndef OC_PERMUTATIONS_H_

// ///////////////////////////////////////////// Include Files

#include "ocport.h"
#include "ocsort.h"  // For OCSWAP
#include "ocarray.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// The Permutations Class

// Produce all the permutations from 1..n (all n!) of them using only
// O(n) extra space and amortized complexity of O(1) for generating
// each element (a single element generation is worst case O(n), but
// amortized over all elements, generation is constant for each
// element).  If you wish to copy the permuatation out (as opposed to
// just reading it), then the cost increases to O(n) per element.

// Usage:
// Permutations p(n);
// while (p.next()) {
//   const int* cp = p.currentPermutation();
//   // Numbers 1..n in cp[0] .. cp[n-1]
// }

class Permutations {
    
  public:

    // ///// Methods

    // Constructor.  Will generate all permutations from 1..n (all n
    // factorial of them).  Until the "next" method is called, the
    // current permutation is invalid.
    Permutations (int n) :
      n_(n),
      indices_(n>0 ? new int[n] : 0),
      currentPermutation_(n>0 ? new int[n] : 0)
    {
      // Initialize: All maxxed out (ready to go): which will be
      // corrected to 0 at the first call to next and initializes the
      // current Permutation correctly.
      if (n>0) {
	for (int ii=0; ii<n; ii++) {
	  indices_[ii]=n-1-ii;
	  // Initalize: -1 means we are first time, not initialized
	  currentPermutation_[ii] = -1;
	}
      }
      
      // The currentPermutation_ is invalid until next is called.
    }

    // Destructor
    ~Permutations () { delete [] indices_; delete [] currentPermutation_; }

    // Advance to the next permutation: This methods needs to be
    // called before currentPermutation can be called. Hopefully will
    // be inlined for speed.
    bool next ()
    {
      // Increase the index, which may cascade and increase the next  
      int* cp = currentPermutation_;
      for (int stage=0; stage<n_; stage++) {
	int& ii = indices_[stage];	

	// Correct old one
	OCSWAP(int, cp[stage], cp[ii+stage]);

	if (++ii < n_-stage) {

	  // Try a new one
	  OCSWAP(int, cp[stage], cp[ii+stage]);
	  return true;
	}
	ii = 0;
      }

      // Deferred initialization of currentPermutation_
      if (n_>0 && cp[0]==-1) {
	for (int jj=0; jj<n_; jj++)
	  cp[jj] = jj+1;
	return true;
      }

      return false; 
    }


    // Return the current permutation: do NOT modify the return value,
    // as it is dependent on the previous value to generate the next
    // permutation.
    const int* currentPermutation () const { return (const int*)currentPermutation_; }

    // Seed.  Start me off at the nth permutation (0-based, so 0 is
    // the very first).  Once you call thjs method, the
    // currentPermutation is valid. This is a linear (in the length of
    // the permutation) operation.  Once you have seeded the start at
    // a particular point, generation will proceed at O(1) (amortized)
    // generation.  Note that this changes where you are in the
    // sequence: If you set the nth to n!-1, the next call to next()
    // will return false immediately.
    
    // Although you can generate all the permutations using
    // the seed method (see code below):
    // Permutations p(n);
    // for (int ii=0; ii<factorial(ii); ii++)
    //   p.seed(ii);
    //   for (int ii=0; ii<n; ii++)
    //     cerr << p.currentPermutation()[ii] << " ";
    // }
    // It's better to use the next(), because each seed() call
    // is linear, but each next() call is constant (amortized).
    // [The seed only works for permutations less than 21! because
    // 20! < biggest int_u8 < 21!]
    void seed (int_u8 seed_val)
    {
      int* cp = currentPermutation_;
      int* in = indices_;
      
      // Figure out how the indices should be (and initialize the
      // start state of the permutation at the same time).      
      for (int ii=0; ii<n_; ii++) {
	cp[ii] = ii+1; // initialize cp start state
	in[ii] = seed_val % (n_-ii);
	seed_val /= (n_-ii);
      }
      
      // Now, move the values into the proper location
      for (int jj=n_-1; jj>=0; jj--) {
	OCSWAP(int, cp[jj], cp[jj+in[jj]]);
      }
    }


  protected:

    // ///// Data Members 

    // The size of the permutations we are generating.
    int n_;

    // Each index represents a "stage" in the pipeline where the
    // current permutations is generating.  indices_[0] is generating
    // permutations of length n and ranges from 0..n-1, indices_[1] is
    // generating permutations of length n-1 and ranges from 0..n-2,
    // ... indices_[n-1] is generating permutations of length 1 and
    // ranges from 0.
    int* indices_;

    // The current permutation, Length n.
    int* currentPermutation_;

}; // Permutations



// ///////////////////////////////////////////// The PermutationsT Class

// This is a convenience class: Given an array of items, permute the
// items through all possible permutations.  Example:

// Array<string> a; a.append("one"); a.append("two"); a.append("three");
// PermutationsT<string> s(a);
// while (s.next()) {
//   cerr << s.currentPermutation() << endl;
// }

// This assumes that type T supports copy constructor and operator
// (i.e., supports swapping elements).

// Note that this may not be the fastest way to generate all
// permutations from an Array, especially if type T is a type where
// copying is especially expensive.  In that case, it might be better
// to use the Permutations class above, and use it as an
// "indirection", so that you aren't actually moving items around as
// much as using the permutations as indices. Example:

// Array<BigOldDataType> a; // length n
// Permutations p(n);
// while (p.next()) {
//   const int* cp = currentPermutation();
//   
//   // Print current permutation 
//   for (int ii=0; ii<n; ii++) {
//     cout << a[cp[ii]-1] << " "; // -1 because Permutations are from 1..n
//   }
//   cout << endl; 
// }

template <class T>
class PermutationsT {

  public:

    // ///// Methods

    // Constructor: We want to permute the given data through all
    // possibilities. Until the "next" method is called, the current
    // permutation is invalid.
    PermutationsT (const Array<T>& data):
      indices_(data.length()),
      currentPermutation_(data.length()),
      data_(data)
    { }


    // Advance to the next permutation: This methods needs to be
    // called before currentPermutation can be called.
    bool next ()
    {
      bool indices_next = indices_.next();
      // Nothing left to serve
      if (!indices_next)
	return false;

      // Otherwise, get the indices.
      const int* in = indices_.currentPermutation();

      // Rearrange!
      currentPermutation_.clear();
      int dlen = data_.length();
      for (int ii=0; ii<dlen; ii++) {
	currentPermutation_.append(data_[in[ii]-1]);
      }
      return true;
    }

    // Give me access to the current Permutation.
    Array<T>& currentPermutation ()  { return currentPermutation_; }

  protected:

    // ///// Data Members
    
    // The indices of how to shuffle around the data we want to
    // permute through all possibilites.
    Permutations indices_;

    // The current Permutation
    Array<T> currentPermutation_;

    // The data to permute
    const Array<T> data_;
}; 


OC_END_NAMESPACE


#define OC_PERMUTATIONS_H_
#endif // OC_PERMUTATIONS_H_
