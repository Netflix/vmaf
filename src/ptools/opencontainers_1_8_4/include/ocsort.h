#ifndef OCSORT_H_

// Some inplace sorting routines:
// Insertion sort is best case O(n), worst case O(n^2) and is best
//  for small arrays, or things that are almost sorted.
// Quick sort is best case O(n*log(n)), average case O(n*log(n), 
//  worst case O(n^2).  It frequently does better than most sorts
//  (and in fact uses insertion sort as a backup for small containers).

// TODO: In a world that supports proper templates, CONTAINER would be
// CONTAINER<CONTAINED_TYPE> so we wouldn't have to pass both types
// in.

// ///////////////////////////////////////////// Include Files

#include "ocport.h"

OC_BEGIN_NAMESPACE

// ///////////////////////////////////////////// Global Functions

// Swap two elements (macro to absolutely ensure speed)
#define OCSWAP(CT, a, b) { CT t=a; a=b; b=t; }

// Swap two elements: In some cases, we don't have easy access to the
// type (as in the M2SWAP) macro above, so we use this which should be
// just as fast as the macro (but we use the macro to guarantee the
// swap speed).
template <class T>
inline void OCSwap (T& t1, T& t2)
{
  T temp = t1;
  t1 = t2;
  t2 = temp;
}


// NOTE: We don't use these swaps above anymore, as we are hoping that
// the general swap in ocport, and each of the specialized swaps in
// all the oc classes will make this much faster.  We leave them
// around in case user-code was depending on them.  (For examples,
// swap with OCString is much faster now).


// Find the median of three elements and and moves it to the first
// position: Note they the elements are passed by reference so we can
// swap the median element to the front.
template <class T>
inline void OCMedianOfThree (T& _1, T& _2, T& _3)
{
  // 123, 132, 213, 231, 312, 321

  if (_1<_2) {
    // 123, 132, 231
    if (_2<_3) {
      // 123
      swap(_1, _2);
    } else {
      // 132, 231
      if (_1 < _3) {
	// 132
	swap(_1, _3);
	swap(_2, _3);
      } else {
	// 231
	swap(_2, _3);
	return; 
      }
    }

  } else {
    // 213, 312, 321
    if (_2<_3) {
      // 213, 312
      if (_3 > _1) {
	// 213
	return;
      } else {
	// 312
	swap(_1, _3);
      }
    } else {
      // 321
      swap(_1, _2);
      swap(_2, _3);
    }
  }
}


// 2 element hard coded sort
template <class CONTAINED_TYPE>
inline void OCHardCodedSort2 (CONTAINED_TYPE& first, CONTAINED_TYPE& second)
{
  if (second < first)
    swap(first, second);
}


// 3 element hard coded sort (CT is the contained type)
template <class CT>
inline void OCHardCodedSort3 (CT& _1, CT& _2, CT& _3)
{
  // 123, 132, 213, 231, 312, 321

  if (_1<_2) {
    // 123, 132, 231
    if (_2<_3) {
      // 123
    } else {
      // 132, 231
      if (_1 < _3) {
	// 132
	swap(_2, _3);
      } else {
	// 231
	swap(_1, _2);
	swap(_1, _3);
      }
    }

  } else {
    // 213, 312, 321
    if (_2<_3) {
      // 213, 312
      if (_3 > _1) {
	// 213
	swap(_1, _2);
      } else {
	// 312
	swap(_1, _2);
	swap(_2, _3);
      }
    } else {
      // 321
      swap(_1, _3);
    }
  }
}


// 4 element hard coded sort
template <class CT>
inline void OCHardCodedSort4 (CT& _1, CT& _2, CT& _3, CT& _4)
{
  // Assertion: 1 of 4! permutations of 1234
  if (_2<_1) swap(_2, _1);
  if (_4<_3) swap(_4, _3);

  // Assertion: each "half" is sorted: We have
  // 1234, 1324, 1423, 2314, 2413, 3412
  if (_3<_1) swap(_3, _1);
  if (_4<_2) swap(_4, _2);

  // Assertion: We have:
  // 1234, 1324
  if (_3<_2) swap(_3, _2);
}


// Helper: What people think of for insertion sort, except that we
// have to pass a CONTAINED_TYPE so we know the type of the temp: This
// is only because we can't depend on which C++ compiler we have.
// This doesn't do any error checking, and it's better to use the
// wrapped OCInsertSort below (which calls this).
template <class CONTAINER, class CONTAINED_TYPE>
inline void OCInsertionSortHelper_ (CONTAINER& c, CONTAINED_TYPE&, 
				    int start, int length)
{
  const int end = start+length-1;
  int ii;
  for (ii=start+1; ii<=end; ii++) {
    int jj = ii;
    CONTAINED_TYPE temp = c[ii];
    for (; jj>start && temp<c[jj-1]; jj--) {
      c[jj] = c[jj-1];
    }
    c[jj] = temp;
  }
}


// Insertion Sort. An inplace sort sorting a portion of CONTAINER c
// starting from index start, running for a cut of length.  Insertion
// sort is good for arrays that are almost already sorted, or smaller
// arrays.  Note that it's inlineable!

// This assumes that elements of the container support <, == and copy
// construction (so that compares and swaps can occur). The container
// itself must support indexing [] from start to end.
template <class CONTAINER>
inline void OCInsertionSort (CONTAINER& c, int start, int length)
{
  // 0 or 1 elements, all done!
  if (length<2)
    return;

  // For very small arrays, it makes sense to use a real quick
  // hardcoded sort, otherwise do the full insertion sort.
  switch (length) {
  case 2: OCHardCodedSort2(c[start],c[start+1]); break;
  case 3: OCHardCodedSort3(c[start],c[start+1],c[start+2]); break;
  case 4: OCHardCodedSort4(c[start],c[start+1],c[start+2],c[start+3]); break;
  default: OCInsertionSortHelper_(c,c[start],start,length); break;
  }
}


// This is a helper function for QuickSort: Partition the container
// into three pieces: All those items less than the partition value
// those items greater than the partition value and all those items
// equal to the partition_value. Note that the returned value of the
// partition walls is inclusive as are the left_wall and right_wall
// parameters.  Furthermore, we assume that upon entry, c[start] <=
// pivot, c[end] >= pivot so that (1) and (2) don't need the extra
// checks for end of bounds checking.

// This assumes that elements of the container support <, >, == and
// copy construction (so that compares and swaps can occur). The
// container itself suppors indexing [].

// Note: the partition value is passed in by value (because we need to
// know the CONTAINED_TYPE and also, the partition value MAY MOVE).
// Also note that because we are doing computations like start-1 and
// end+1, the indexing pretty much has to be an int (otherwise we get
// roll-over issues with unsigned types).
template <class CONTAINER, class CONTAINED_TYPE>
void OCPartition_ (CONTAINER& c, const CONTAINED_TYPE pivot,
		   int left_wall, int right_wall,
		   int& left_partition_wall, int& right_partition_wall)
{
  // Upon entry, c[start] <= pivot, c[end] >= pivot so that (1) and
  // (2) don't need the extra checks for end of bounds checking.

  // Partition: c[left-wall] & c[right_wall] in proper partition
  for (;;) {
    while(c[++left_wall]  < pivot) ; // (1)
    while(c[--right_wall] > pivot) ; // (2)
    if (left_wall>=right_wall) {
      break;
    } else if (c[left_wall]==c[right_wall]) { 
      // The pivot is at both walls ... have to find some non pivot
      // and move it into one of the walls so we can make progress
      int pivot_wall = left_wall; 
      while (++pivot_wall<right_wall && c[pivot_wall]==pivot) ;
      if (pivot_wall==right_wall) break;     // All pivots, all done!
      swap(c[left_wall], c[pivot_wall]); // Else put non-pivot into left
      left_wall--; // has to recheck
    } else {     
      // Swap walls
      swap(c[left_wall],c[right_wall]);
    }
  }
  
  // All done 
  left_partition_wall = left_wall-1;
  right_partition_wall = right_wall+1;
}

OC_END_NAMESPACE

#include "ocarray.h" // So we can manage stack frames


OC_BEGIN_NAMESPACE

// Sort a portion of an array inplace using QuickSort: If you want to
// sort the entire array, start=0, len=length_of_array. NOTE: start is
// inclusive: It sorts from container[start] for a length of len.

// This assumes that elements of the container support <, >, == and copy
// construction (so that compares and swaps can occur). The container
// needs to support indexing [].

// Allows you to tune when you use insertion sort (instead of
// QuickSort "recursively") inside of OCQuickSort.
#ifndef OC_QSTUNE       
# define OC_QSTUNE 42   
// This number was found through some experimentation on some
// 2000(?)ish machines; the "right number" will vary with your
// machine, but this is probably good enough for most people.
#endif 

template <class CONTAINER>
void OCQuickSort (CONTAINER& c, int start, int length)
{
  // Get rid of recursion for speed purposes (and to stay away from
  // recursive stack management).  Since stack frames are so simple,
  // manage them with a small array with <start, end> for each frame
  Array<int> frames(64); // May grow up to length/OC_QSTUNE in worst case 
  int end = start + length - 1;  // Inclusive end
  frames.append(start); frames.append(end);
  
  while (1) {
    // Pop frame 
    if (frames.length()!=0) {
      end    = frames.removeLast();
      start  = frames.removeLast();
    } else return;
    
  tail:  // Take advantage of tail recursion to avoid a pop of stack
    // Base case:  Either 0 or 1, all done.
    length = end - start + 1;
    if (length<2) { // Allow start to be negative in case of weird arrays
      continue;
    }
    
    // Small case: Do an insertion sort for small containers, as
    // insertion is good for smallish n (which also handles the
    // optimal cases for 2, 3 and 4 elements).
    if (length < OC_QSTUNE) { 
      OCInsertionSort(c, start, length);
      continue;
    }

    // Recursive case: Partition the array into three pieces:
    // Everything less than the partition element, everything equal to
    // the partition elements, and everything greater than the
    // partition elements.

    // Find a "reasonable" partition value: The median of the front,
    // middle and end piece.  By putting the smallest member of the 3
    // up front, and largest of the 3 at end, we keep from using
    // another if test during points (1) and (2) of
    // OCPartition_. [Note that this works even if they are all the
    // same].
    OCHardCodedSort3(c[start], c[(start+end)/2], c[end]);
    int left_wall, right_wall;
    OCPartition_(c, c[(start+end)/2], start, end, left_wall, right_wall);

    // Quicksort "recursively"
    frames.append(start); frames.append(left_wall); 
                                   // OCQuickSort(c, start, left_wall);
    start = right_wall; goto tail; // OCQuickSort(c, right_wall, end); 
  }
}


// Give me the median value (or the item to the left for even sized
// containers) for a cut in the container that starts at index 'start'
// and runs for length 'len' contiguous elements. Note this routine
// has the side-effect of sorting the container inplace.
template <class CONTAINER, class CONTAINED_TYPE>
void OCMedian (CONTAINER& c, int start, int length, 
	       CONTAINED_TYPE& return_value)
{
  OCQuickSort(c, start, length);
  return_value = c[start+length/2];
}

OC_END_NAMESPACE

#define OCSORT_H_
#endif // OCSORT_H_
