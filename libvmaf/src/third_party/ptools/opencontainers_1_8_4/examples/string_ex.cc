
// Demonstrate how the OCString works. Note that implements a subset
// of the STL string for some compatibility.

#include "ocport.h"
#include "ocstring.h"
#include <stdio.h> // printf

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  string empty;  // By default, initialized to "" empty string;
  cout << "empty is:" << empty << endl;
  
  string me = "me"; // Initialize to c-style string
  cout << "me is:" << me << endl;

  string also = "also";
  me = empty + me + also;  // String concatenation
  cout << "concatenation: " << me << endl;

  printf("%s\n", me.c_str()); // C style usage

  // Individual access, read and write!  This reverses the string.
  size_t len = me.length();
  for (int ii=0; ii<len/2; ii++) {
    // Swap opposite sides of string
    char c = me[len-ii-1];
    me[len-ii-1] = me[ii];
    me[ii] = c;
  }
  cout << "reverse = " << me << endl;
}

/* Output:

empty is:
me is:me
concatenation: mealso
mealso
reverse = oslaem

*/
