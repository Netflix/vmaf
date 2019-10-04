#ifndef OC_PERMUTE_H_
#define OC_PERMUTE_H_

// Generate all permutations of an int array in order,
// much like STL permutation generation.

// Discussion: In tests in 2015, using g++ at -O4, this was
// about 10% faster than the STL.  

// Note, this is also plain C: this can be used directly in
// a C file if you are looking for a good permutation generator

// Note: find_succesor has two implementations: a linear
// search and a binary search.  The linear search is blindingly
// fast as it is such a tight loop.  The binary search may make
// sense if permutating very long lists (lists of > 30 elements?),
// but since 30! is pretty darn large, it's doubtful most people
// would do that, so we use the linear search method.

// Swapping important for this alg: do it fast
#define SWAPME(A,B) int temp=A; A=B; B=temp

#if defined(OC_BEGIN_NAMESPACE) // C++ Namespace: ignored if C code
OC_BEGIN_NAMESPACE
#endif

// reverse in-place
void reverse (int* a, int len)
{
  const int len_2 = len/2;
  for (int ii=0; ii<len_2; ii++) {
    SWAPME(a[ii], a[len-ii-1]);
  }
}


// NOT USED: but can be dropped in for find_successor in this routine.
// Assumptions: a is REVERSE SORTED, and we are looking for the index
// of what would be the next element (successor for point).
// Assume > 3 elements for len.  
int bin_search_successor (int point, int *a, int len)
{
  int left = 0;
  int right = len-1;
  int mid = -1;
  while (left <= right) {
    mid = (right+left)/2;
    if (point > a[mid]) {
      right = mid - 1;
    } else if (point < a[mid]) {
      left = mid + 1;
    } else {
      return mid;
    } 
  }
  return right;
}


// In a reverse sorted array, find the succesor of the given.
// Return the index of the successor, which may be the last element.
// Returns -1 if there is no successor.
int find_successor (int point, int *a, int len)
{
  // Too big? The binary search might be faster than a straight lookup
  //if (len>4) {
  //  return bin_search_successor(point, a, len);
  //}

  // Linear search
  for (int ii=0; ii<len; ii++) {
    if (point > a[ii]) {
      // then previous element is the successor
      return ii-1;
    }
  }
  // Here, then LAST element is successor
  return len-1;
}

// Inplace permute to the next permutation (that would be in 
// sorted order).  This returns true while there are more permutations,
// and false when the passed-in permutation is the last permutation.
int perm (int* a, int len)
{
  // 1 or 0 elements, already exactly 1
  if (len<=1) return 0;

  // Assertion: 2 or more elements.  Optimization:
  // Half of the time, the next permutation is just swapping
  // the last two elements
  if (a[len-1] > a[len-2]) {
    SWAPME(a[len-1], a[len-2]);
    return 1;
  }

  // Okay, search for first element (from end) that is no longer increasing.
  // I.e., Search from behind for first element out of order.
  // (Starts at len-3 because already looked at len-2)
  for (int ii=len-3; ii>=0; ii--) { 
    if (a[ii]<a[ii+1]) { // Found out of order
      const int small_len = len-ii-1;
      int* small_a  = a+ii+1;

      // Find succesor of a[ii] in array to right of a[ii]
      const int relative_succ_index = find_successor(a[ii], small_a, small_len);
      const int succ_index = relative_succ_index + ii + 1;
      SWAPME(a[ii], a[succ_index]);
      reverse(small_a, small_len);
      return 1;
    }
  }

  // Reached here, at end of array: nothing out of order, done.
  return 0;
}

#if defined(OC_END_NAMESPACE) // C++ Namespace: ignored if C code
OC_END_NAMESPACE
#endif


#endif
