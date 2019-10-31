
// Demonstrations of the array

#include "ocarray.h"
#include "ocstring.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  string numbers = "0123456789";

  // Array of strings, with a capacity of 10. Nothing actually
  // constructed or in the array just yet.
  cout << "**Arrays of string**" << endl;
  Array<string> as(10);   

# if defined(BAD_CODE)

# error 666
  // Bad code: The array needs be initialized with values using
  // append: You can't set a value via operator[] until something is
  // in there. 
  for (int ii=0; ii<10; ii++) 
    as[ii] = "Hello" + numbers.substr(ii,1);

  // Why?  For efficiency: you don't construct items until have to.
  // Consider what C++ does:
  string* s = new string[10]; // DEFAULT CONSTRUCTS 10 strings in C++
  // Expensive, and you cannot explicitly initialize each member
# endif
  
  // Initialize
  for (int ii=0; ii<10; ii++)
    as.append("Hello" + numbers.substr(ii,1));
  
  // Finally, can look at individual elements
  for (int jj=0; jj<10; jj++)
    cout << as[jj] << " ";
  cout << endl;
  

  // Array of ints, with a capacity to hold 3 ints without expanding.
  cout << "**Arrays of ints**" << endl;
  Array<int> a(3);
  // Because "int" is a type WITHOUT a constructor/destructor, we can cheat
  a.expandTo(10);
  for (int kk=0; kk<10; kk++)
    a[kk] = kk;
  
  // Finally, can look at individual elements
  for (int ll=0; ll<10; ll++)
    cout << a[ll] << " ";
  cout << endl;


  // C style access, if you need plain old pointers, perhaps for speed
  cout << "**C style access**" << endl;
  int* c_ptr = a.data();
  for (int mm=0; mm<10; mm++) 
    cout << c_ptr[mm] << " ";
  cout << endl;

}


/* Output:

**Arrays of string**
Hello0 Hello1 Hello2 Hello3 Hello4 Hello5 Hello6 Hello7 Hello8 Hello9 
**Arrays of ints**
0 1 2 3 4 5 6 7 8 9 
**C style access**
0 1 2 3 4 5 6 7 8 9 

*/
