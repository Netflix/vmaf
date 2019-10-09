
// Demonstration of AVLTrees.  There are two types:
//      AVLTreeT<key, value, expected_size> 
//      AVLTree<value>
// Note that AVLTree<value> == AVLTreeT<string, value, 8>

#include "ocport.h"
#include "ocavltreet.h" // For AVLTreeT

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// Note that we have to provide a HashFunction for out keys: ints.
int_u4 HashFunction (int value)
{ return value; }


int main ()
{
  // Smaller tables than default and different keys.  Lookup
  // phonenumbers to get person's name.  
  AVLTreeT<int, string, 4> reverse_phonebook;
  reverse_phonebook.insertKeyAndValue(2981111, "Fred");
  string& who = reverse_phonebook[2981111];
  cout << "number 2981111 corresponds to " << who << endl;

  // No one by this phone number, but uses default constructor
  // for string to return a valid entry (which is empty).
  string& s = reverse_phonebook[2988855]; 
  cout << "person at 2988855 = " << s << endl;

  // Use insertKeyAndValue for more efficent insertion
  reverse_phonebook[2999999] = "Fred";
  reverse_phonebook[2988855] = "Fred";  // Yes, keys can repeat
  reverse_phonebook[2] = "You!";

  // Iterate.  In key-sorted order
  AVLTreeTIterator<int, string, 4> itt(reverse_phonebook);
  while (itt.next()) {
    cout << itt.key() << " " << itt.value() << endl;
  }
} 

/* Output:
 
number 2981111 corresponds to Fred
person at 2988855 = 
2 You!
2981111 Fred
2988855 Fred
2999999 Fred

*/
