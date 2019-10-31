
// Demonstration of HashTables.  There are two types:
//      HashTableT<key, value, expected_size> 
//      HashTable<value>
// Note that HashTable<value> == HashTableT<string, value, 8>

#include "ocport.h"
#include "ochashtablet.h" // For HashtableT

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
  HashTableT<int, string, 4> reverse_phonebook;
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

  // Iterate.  Again, not in any particular order
  HashTableTIterator<int, string, 4> itt(reverse_phonebook);
  while (itt.next()) {
    cout << itt.key() << " " << itt.value() << endl;
  }
} 

/* Output:
 
number 2981111 corresponds to Fred
person at 2988855 = 
2 You!
2999999 Fred
2988855 Fred
2981111 Fred

*/
