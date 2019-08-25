
// Demonstration of AVLTrees.  There are two types:
//      AVLTreeT<key, value, expected_size> 
//      AVLTree<value>
// Note that AVLTree<value> == AVLTreeT<string, value, 8>

#include "ocavltree.h" // Include for AVLTree

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  AVLTree<int_u8> phonebook;                   // Empty phone book
  phonebook.insertKeyAndValue("Fred", 2988855);  // Add Fred!

  int_u8& lookup = phonebook["Fred"];            // Lookup fred's number
  cout << "Fred's number is " << lookup << endl;

  phonebook["Fred"] = 2981111;                   // Change Fred's number
  cout << "Fred's new number is " << phonebook["Fred"] << endl;

  phonebook.remove("Fred");                      // Fred leaves town
  cout << "Phonebook contains Fred?:" << phonebook.contains("Fred") << endl;

  int_u8& n = phonebook["Fred"];   // Fred gone, so default construction of n 
                                   // (which means n is uninitialized, but
                                   // refers to valid entry)
  // ... but now, can modify n to give Fred a number
  n = 2999999;
  cout << "Fred is now at " << phonebook["Fred"] << endl;

  // Add some people: Note that this can be more expensive than
  // insertKeyAndValue because requires <value> to have default constructor
  phonebook["amy"]    = 1111111;
  phonebook["bill"]   = 2222222;
  phonebook["carson"] = 3333333;
  phonebook["dirk"]   = 4444444;

  // Iterate through all entries:  Note they ARE in sorted order!
  AVLTreeIterator<int_u8> it(phonebook);
  while (it.next()) {
    cout << it.key() << " " << it.value() << endl;
  }

  // Note that AVLTree, AVLHash, and HashTable all have the same interface!
}


/* Output:

Fred's number is 2988855
Fred's new number is 2981111
Phonebook contains Fred?:0
Fred is now at 2999999
Fred 2999999
amy 1111111
bill 2222222
carson 3333333
dirk 4444444

*/
