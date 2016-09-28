
//

// ///////////////////////////////////////////// Include Files

#include "ocarray.h"
#include "ocbsearch.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

void tests ()
{
  // "borrowed" this array from the sortedvectort_test

  int_4 ar[] = { 19, 1, 7, 6, 10, 9, 8, 7, 6, 0, -1, -2, -3, -2, -1, 5, 6,
		 7, 8, 12, 13, 14, 15, 16, 70, 71, 72, 73, 74, 73, 73, 74, 
		 19, 19, 19, 1, 1, 100, 999 };
  
  Array<int_4> search_this;
  
  int_u4 ii;

  
  for (ii = 0; ar[ii] != 999; ii++) {
    int insert_here;
    (void)OCBSearch(search_this, search_this.length(), ar[ii], 0, insert_here);
    search_this.insertAt(insert_here, ar[ii]);
  }
  
  cout << "Array to search: " << endl;
  cout << "[" ;
  for (ii = 0; ii < search_this.length(); ii++) {
    cout << search_this[ii] << ",";
  }
  cout << "]" << endl;



  for (int_4 yy = -20; yy <= 100; yy++) {
    
    cout << "Searching for elemet (" + Stringize(yy) + ")" << endl;
    
    int location;

    if (true ==  OCBSearch(search_this, search_this.length(), yy, search_this.length()/2, location)) {
      cout << "\tFound (" + Stringize(yy) + ") at location (" + Stringize(location) + ")" << endl;
    }
  }
  
}



// ///////////////////////////////////////////// Main Program

int main (int, char**)
{
  tests();
}


