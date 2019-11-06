// Test the sorting routines

// ///////////////////////////////////////////// Include Files

#include "ocport.h"
#include "ocarray.h"
#include "ocsort.h"
#include "ocpermutations.h"

#include <stdio.h>

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// ///////////////////////////////////////////// The SortTest Class

class SortTest {
    
  public:

    // ///// Methods

    int tests ();

}; // SortTest



// ///////////////////////////////////////////// Globals

template <class CONTAINER>
bool testIfSorted (const CONTAINER& c, int start, int length)
{
  const int end = start+length-1;
  for (int ii=start; ii<end; ii++) {
    if (c[ii] > c[ii+1])
      return false;
  }
  return true;
}


string StringIt (const int* c, int n)
{
  string s;
  for (int ii=0; ii<n; ii++) {
    s += Stringize(c[ii]) + " ";
  }
  return s;
}


// ///////////////////////////////////////////// SortTest Methods

int SortTest::tests()
{

  int kk;
  int* my_copy = new int[100];

  cout << "** Test if sorted **" << endl;
  for (kk=0; kk<6; kk++) {
    cout << "-> Searching Permutations of length " << kk << " for sorted" << endl;
    Permutations p(kk);
    while (p.next()) {
      const int* r = p.currentPermutation();
      if (testIfSorted(r, 0, kk)) {
	for (int gg=0; gg<kk; gg++) 
	  cout << "[" << r[gg] << "]";
	cout << endl;
      }  
    }       
  }


  
  {
    bool success = true;
    cout << "** Test HardCodedSort2 ** " << endl;

    int c[2];
    Permutations p(2);
    while (p.next()) {
      const int* c1 = p.currentPermutation();
      c[0] = c1[0]; c[1] = c1[1];
      OCHardCodedSort2(c[0], c[1]);

      if (!testIfSorted(c, 0, 1)) {
	cout << "Failure sorting: " << StringIt(c,2) << "->" << StringIt(c1,2) << endl;
	success = false;
      } else {
	cout << "SUCCESS sorting: " << StringIt(c,2) << "->" << StringIt(c1,2) << endl;
      }
    }

    if (success) {
      cout << "SUCCESS for OCHardCodedSort2 "  << endl;
    }
  }




  {
    bool success = true;
    cout << "** Test HardCodedSort3 ** " << endl;

    int c[3];
    Permutations p(3);
    while (p.next()) {
      const int* c1 = p.currentPermutation();
      c[0]=c1[0]; c[1]=c1[1]; c[2]=c1[2];

      OCHardCodedSort3(c[0], c[1], c[2]);

      if (!testIfSorted(c, 0, 2)) {
	cout << "Failure sorting: " << StringIt(c,2) << "->" << StringIt(c1,2) << endl;
	success = false;
      } else {
	cout << "SUCCESS sorting: " << StringIt(c,2) << "->" << StringIt(c1,2) << endl;
      }
    }

    if (success) {
      cout << "SUCCESS for OCHardCodedSort3 "  << endl;
    }
  }


  {
    bool success = true;
    cout << "** Test HardCodedSort4 ** " << endl;

    int c[4];
    Permutations p(4);
    while (p.next()) {
      const int* c1 = p.currentPermutation();
      c[0]=c1[0]; c[1]=c1[1]; c[2]=c1[2]; c[3]=c1[3];

      OCHardCodedSort4(c[0], c[1], c[2], c[3]);

      if (!testIfSorted(c, 0, 3)) {
	cout << "Failure sorting: " << StringIt(c,4) << "->" << StringIt(c1,4) << endl;
	success = false;
      } else {
	cout << "SUCCESS sorting: " << StringIt(c,4) << "->" << StringIt(c1,4) << endl;
      }
    }
    
    if (success) {
      cout << "SUCCESS for OCHardCodedSort4 "  << endl;
    }
  }


  {
    bool success = true;
    cout << "** Test Insertion Sort ** " << endl;

    for (int top=0; top<10; top++) {

      Permutations p(top);
      while (p.next()) {
	const int* a = p.currentPermutation();

	for (int cc=0; cc<top; cc++) my_copy[cc] = a[cc];

	OCInsertionSort(my_copy, 0, top);

	if (!testIfSorted(my_copy, 0, top)) {
	  cout << "Failure sorting from top=: " << top << endl;
	  success = false;
	} else {
	  // cout << "SUCCESS sorting: " << rets[jj] << " -> " << c << endl;
	}
      }
      
      if (success) {
	cout << "SUCCESS for Insertion sort on length:" << top  << endl;
      }
    }
  }


  {
    cout << "** Testing median of 3 **" << endl;

    int r[3];
    Permutations p(3);
    while (p.next()) {
      const int* r1 = p.currentPermutation();
      r[0]=r1[0]; r[1]=r1[1]; r[2]=r1[2];
      OCMedianOfThree(r[0], r[1], r[2]);
      cout << "median at front: " << StringIt(r,3) << "->" << StringIt(r1,3) << endl;
    }
  }


  {
    bool success = true;
    cout << "** Test Quick Sort ** " << endl;

    for (int top=0; top<12; top++) {
      Permutations p(top);
      while (p.next()) {
	const int* a = p.currentPermutation();

	for (int cc=0; cc<top; cc++) { my_copy[cc] = a[cc];} //  cerr << a[cc] << " "; }
	// cerr << endl; 

	OCQuickSort(my_copy, 0, top);

	if (!testIfSorted(my_copy, 0, top)) {
	  cout << "Failure sorting from top=: " << top << endl;
	  success = false;
	} else {
	  // cout << "SUCCESS sorting: " << rets[jj] << " -> " << c << endl;
	}
      }

      if (success) {
	cout << "SUCCESS for Quick sort on length:" << top  << endl;
      }
    }
  }


  {
    cout << "** Lots of randomness testing ** " << endl;
    for (int ii=0; ii<20; ii++) {

      Array<string> a(10000*(ii+1));
      for (int jj=0; jj<int(a.capacity()); jj++) {
	a.append(Stringize(rand()) + " bottles of beer on the wall, take one down, pass it around"); // Want a string that actually goes to the heap!
      }

      cout << "Sorting array of length: " << a.capacity() << endl;
      OCQuickSort(a, 0, a.capacity());
      if (testIfSorted(a, 0, a.capacity())) {
	cout << "SUCCESS sorting!" << endl;
      } else {
	cout << "FAILURE sorting!" << endl;
	exit(1);
      }
    }
  }
  
  delete my_copy;

  return 0;
}



int main ()
{
  SortTest t;
  return t.tests();
  return 0;
}



