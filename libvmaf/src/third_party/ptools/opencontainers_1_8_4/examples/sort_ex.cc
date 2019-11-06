
// Demonstration of Sorting: note that the sorts work for both
// built-in C style arrays and the Array class.

#include "ocport.h"
#include "ocstring.h"
#include "ocarray.h"
#include "ocsort.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  // Quicksort is good if you have a fairly large amount of data in
  // relatively random order
  const char* sts[] = { "hello", "there", "everyone", "how", "are", "you", 0 };
  Array<string> s;
  for (int ii=0; sts[ii]!=0; ii++) 
    s.append(string(sts[ii]));
  cout << "**Before QuickSort:" << s << endl;

  string* sd = s.data();
  OCQuickSort(sd, 0, s.length());

  cout << "**After Quicksort:" << s << endl;

  // Insertion sort is good for data that's almost already sorted.
  int in[8] = { 0, -1, 2, 3, 6, 7, 8, 5 };


  cout << "**Before InsertionSort:";
  for (int jj=0; jj<8; jj++) 
    cout << in[jj] << " ";
  cout << endl;

  int* ina = &in[0]; // you can use in below, but DECUNIX seems to like ina
  OCInsertionSort(ina, 0, 8);

  cout << "**After InsertionSort:";
  for (int kk=0; kk<8; kk++) 
    cout << in[kk] << " ";
  cout << endl;

  // Get the median element
  real_8 med[] = {1.1, 2.2, 3.3, 1.0, 1.5, 3.7, 6.0};

  cout << "**Before Median:";
  for (int ll=0; ll<7; ll++) 
    cout << med[ll] << " ";
  cout << endl;
  
  real_8 med_val;
  real_8* meda = &med[0]; //you can use med below, but DECUNIX seems like meda
  OCMedian(meda, 0, 7, med_val);
  cout << "median value is " << med_val << endl;

  cout << "**After Median:";
  for (int mm=0; mm<7; mm++) 
    cout << med[mm] << " ";
  cout << endl;

}


/* Output:

**Before QuickSort:hello there everyone how are you 
**After Quicksort:are everyone hello how there you 
**Before InsertionSort:0 -1 2 3 6 7 8 5 
**After InsertionSort:-1 0 2 3 5 6 7 8 
**Before Median:1.1 2.2 3.3 1 1.5 3.7 6 
median value is 2.2
**After Median:1 1.1 1.5 2.2 3.3 3.7 6 

*/
