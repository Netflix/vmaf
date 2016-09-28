
// Demonstration of AVLHashs.  There are two types:
//      AVLHashT<key, value, expected_size> 
//      AVLHash<value>
// Note that AVLHash<value> == AVLHashT<string, value, 8>

#include "ocport.h"
#include "ocavlhasht.h" // For AVLHashT

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// Note that we have to provide a HashFunction for out keys: ints.
int_u4 HashFunction (int value)
{ return value % 10; }


int main ()
{
  // Smaller tables than default and different keys.  Lookup
  // phonenumbers to get person's name.  
  AVLHashT<int, string, 4> reverse_phonebook;
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
  reverse_phonebook[3118111] = "Somebody";

  // Iterate.  Again, not in any particular order
  cout << "Iterating in no particular order" << endl;
  AVLHashTIterator<int, string, 4> itt(reverse_phonebook);
  while (itt.next()) {
    cout << itt.key() << " " << itt.value() << endl;
  }

  cout << "Iterating in sorted order" << endl;
  AVLHashT<int, string,4> empty;
  AVLHashTSortedIterator<int, string, 4> ii(empty);
  while (ii.next()) {
    cout << ii.key() << " " << ii.value() << endl;
  }
  ii.reset(reverse_phonebook);
  while (ii.next()) {
    cout << ii.key() << " " << ii.value() << endl;
  }
} 

/* Output:

number 2981111 corresponds to Fred
person at 2988855 = 
Iterating in no particular order
2981111 Fred
3118111 Somebody
2 You!
2988855 Fred
2999999 Fred
Iterating in sorted order
2 You!
2981111 Fred
2988855 Fred
2999999 Fred
3118111 Somebody

*/
