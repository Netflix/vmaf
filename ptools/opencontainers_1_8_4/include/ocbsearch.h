#ifndef OCBSEARCH_H_
#define OCBSEARCH_H_

OC_BEGIN_NAMESPACE

// Code to implement binary search.

// Requirements for class CT (container class) are that it has operator[].
// Requirements for class T is that it has operator== and operator<.

// Returns 'true' if the given value was found; 'false' if it was not.
// If 'true', 'location' will contain its index; if 'false'; 'location'
// will point to the next largest element -- the element that 'val' belongs
// just before.

template <class CT, class T>
bool OCBSearch (const CT& array, int length,
		T val,
		int initial_guess,
		int& location)
{
  // In case there ARE no elements
  int point_of_insertion = 0;

  int m = initial_guess;

  // Figure out where we are with binary search
  if (length!=0) {

    // Maybe we're lucky guessers!
    if (array[m] == val) {
      // Make sure we walk back to the leftmost instance of our match
      while (m > 0  &&  array[m-1] == val)
        m--;
      location = m;
      return true;
    }

    int l=0;
    int r=length;

    // Continue until boundaries cross
    while (l <= r  &&  l < length) {

      // Compute midpoint
      m = (l+r)/2;

      // Less than the val we are trying to insert, move left boundary
      if (array[m] < val)
        l = m + 1;

      // Right here!  Found out exactly where we are
      else if (array[m] == val) {
        // Make sure we walk back to the leftmost instance of our match
        while (m > 0  &&  array[m-1] == val)
          m--;
        location = m;
        return true;
      }

      // Else move right boundary
      else
        r = m - 1;
    }

    // At this point, we either found it exactly, or l and r
    // crossed over.  The crossing point is where we want to insert
    point_of_insertion = l;
  }

  location = point_of_insertion;
  return false;
}                                       // BSearchT

OC_END_NAMESPACE

#endif // OCBSEARCH_H_

