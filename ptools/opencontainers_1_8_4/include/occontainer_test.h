
// This is a test for the 3 container classes: AVLHash, AVLTree and
// HashTable.  Since they are plug in replacements for each other
// (modulo HashFunction and operator< support for keys), we can use
// the same test for most of them.

// ///////////////////////////////////////////// Includes 

#include "ocport.h"
#include "ocstring.h"

#include <stdio.h>

#if defined(OC_FORCE_NAMESPACE)
 using namespace OC;
#endif

// ///////////////////////////////////////////// The ContainerTest Class

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
class ContainerTest {

  public:

    // ///// Methods
    int tests ();

    // Suite which tests all the basic functionality
    void basicSuite (INTCONTAINER&);
    bool compare (const INTCONTAINER& t, const INTCONTAINER& copy);
    void copyTest ();
    void iteratorTest ();
    void moreIteratorTest ();
    void iterationTest ();
    void newOpTest ();
    void swapTest ();

}; // ContainerTest


// ///////////////////////////////////////////// ContainerTest Methods

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
int ContainerTest<INTCONTAINER, INTIT, STRINGCONTAINER, STRINGIT>::tests ()
{
  INTCONTAINER t;

  basicSuite(t);
  copyTest();
  iteratorTest();
  moreIteratorTest();
  iterationTest();
  newOpTest();
  swapTest();
  return 0;
}


const char* yesdudes[] = {
  "Chris Squire",
  "Jon Anderson",
  "Trevor Horn",
  "Peter",
  "Steve Howe",
  "Trevor Rabin",
  "Tony Kaye",
  "Rick Wakeman",
  "Geoff Downes",
  "Patrick Moraz",
  "Bill Sherwood",
  "Igor",
  "Alan White",
  "Bill Bruford",
  0
};

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
bool ContainerTest<INTCONTAINER, INTIT, STRINGCONTAINER, STRINGIT>::compare (const INTCONTAINER& t, const INTCONTAINER& copy_t)
{
  // COmpare
  int i;
  bool copied_okay = true;
  for (i=0; yesdudes[i] != 0; i++) {
    int_u4 val_t, val_copy_t;
    if (!(copy_t.findValue(yesdudes[i], val_copy_t) &&
	  t.findValue(yesdudes[i], val_t) &&
	  val_t == val_copy_t))
      {
	copied_okay= false;
      }
  }
  
  if (t.entries() == copy_t.entries() && i==int(t.entries()) && copied_okay==true)
    return true;
  else
    return false;
    
}


template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
void ContainerTest<INTCONTAINER, INTIT, STRINGCONTAINER, STRINGIT>::iteratorTest ()
{
  {
    // Build a hash table
    INTCONTAINER t;
    int i;
    for (i=0; yesdudes[i] !=0; i++) {
      t.insertKeyAndValue(yesdudes[i], i);
    }
    
    bool* saw = new bool[t.entries()];
    for (i=0; i<int(t.entries()); i++)
      saw[i] = false;
    
    // Iterate through it
    INTIT ii(t);
    i=0;
    while (++ii) {
      if (t.contains(ii.key())) {
	if (ii.key() == yesdudes[ii.value()])
	  saw[ii.value()] = true;
      }
      i++;
    }
    
    // Make sure all entries were seen
    bool all_seen = true;
    for (i=0; i<int(t.entries()); i++)
      if (saw[i] == false) {
	all_seen = false;
	cerr << "ERROR:  Didn't see " << yesdudes[i] << endl;
      }
    
    if (!all_seen) {
      cerr << "ERROR:  Didn't see all elements when iterated through" << endl;
    } else {
      cout << "Good.  Iterated through all elements" << endl;
    }
    
    delete [] saw;
    
    
    // Try and iterate through an empty table
    t.clear();
    INTIT iii(t);
    if (++iii) {
      cerr << "ERROR: Shouldn't see anything! Iterating through empty list!" << endl;
    } else {
      cout << "Good.  Didn't iterate through anything when going through emoty list " << endl;
    }
    
    // Try and iterate through a list with one element
    t.insertKeyAndValue("Hello", 0);
    INTIT iiii(t);
    while (++iiii) {
      cout << "Should only see this element: " << iiii.key() << "=" 
	   << iiii.value() << endl;
    }
  }
  
  // Iterate through with next()
  {
    // Build a hash table
    INTCONTAINER t;
    int i;
    for (i=0; yesdudes[i] !=0; i++) {
      t.insertKeyAndValue(yesdudes[i], i);
    }
    
    bool* saw = new bool[t.entries()];
    for (i=0; i<int(t.entries()); i++)
      saw[i] = false;
    
    // Iterate through it with next
    INTIT ijk(t);
    i=0;
    while (ijk.next()) {
      if (t.contains(ijk.key())) {
	if (ijk.key() == yesdudes[ijk.value()])
	  saw[ijk.value()] = true;
      }
      i++;
    } 
    
    // Make sure all entries were seen
    bool all_seen = true;
    for (i=0; i<int(t.entries()); i++)
      if (saw[i] == false) {
	all_seen = false;
	cerr << "ERROR:  Didn't see " << yesdudes[i] << endl;
      }
    
    if (!all_seen) {
      cerr << "ERROR:  Didn't see all elements when iterated through" << endl;
    } else {
      cout << "Good.  Iterated through all elements" << endl;
    }

    delete [] saw;
  }
  

  // Make sure op== works
  {
    // Build a hash table
    INTCONTAINER t;
    int i;
    for (i=0; yesdudes[i] !=0; i++) {
      t.insertKeyAndValue(yesdudes[i], i);
    }

    INTCONTAINER tcopy = t;
    if (tcopy!=t || !(tcopy==t)) {
      cerr << "ERROR:  operator== doesn't work!" << endl;
    } else {
      cout << "GOOD: op== works for straight copy!" << endl;
    }

    INTCONTAINER tr; // Build in reverse order
    for (i--; i>=0; i--) {
      tr.insertKeyAndValue(yesdudes[i], i);
    }

    if (tr!=t || !(tr==t)) {
      // For any other container, this wouldn't work, but for
      // OrdAVLHashT, all elements have to be in the same order
      // for this to work, so on THAT test only, the reverse
      // test iteration SHOULD Fail!
      cout << "ERROR:  operator== doesn't work!" << endl;
    } else {      
      cout << "GOOD: op== works for rev copy!" << endl; 
    }
  }
}

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
void ContainerTest<INTCONTAINER, INTIT, STRINGCONTAINER, STRINGIT>::moreIteratorTest ()
{
  // Build a hash table
  INTCONTAINER t;
  int i;
  for (i=0; yesdudes[i] !=0; i++) {
    t.insertKeyAndValue(yesdudes[i], i);
  }

  bool* saw = new bool[t.entries()];
  for (i=0; i<int(t.entries()); i++)
    saw[i] = false;

  // Iterate through it
  INTIT ii(t);
  i=0;
  while (++ii) {
    if (t.contains(ii.key())) {
      if (ii.key() == yesdudes[ii.value()])
	saw[ii.value()] = true;
    }
    i++;
  }
  
  // Make sure all entries were seen
  bool all_seen = true;
  for (i=0; i<int(t.entries()); i++)
    if (saw[i] == false) {
      all_seen = false;
      cerr << "ERROR:  Didn't see " << yesdudes[i] << endl;
    }
  
  if (!all_seen) {
    cerr << "ERROR:  Didn't see all elements when iterated through" << endl;
  } else {
    cout << "Good.  Iterated through all elements" << endl;
  }

  delete [] saw;


  // Try and iterate through an empty table
  t.clear();
  INTIT iii(t);
  if (++iii) {
    cerr << "ERROR: Shouldn't see anything! Iterating through empty list!" << endl;
  } else {
    cout << "Good.  Didn't iterate through anything when going through emoty list " << endl;
  }

  // Try and iterate through a list with one element
  t.insertKeyAndValue("Hello", 0);
  INTIT iiii(t);
  while (++iiii) {
    cout << "Should only see this element: " << iiii.key() << "=" 
	 << iiii.value() << endl;
  }
  
}


template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
void ContainerTest<INTCONTAINER, INTIT, STRINGCONTAINER, STRINGIT>::copyTest ()
{
  // Create a hash table and try to copy it
  INTCONTAINER t;
  int i;
  for (i=0; yesdudes[i] !=0; i++) {
    t.insertKeyAndValue(yesdudes[i], i);
  }


  // Make a copy and compare them
  INTCONTAINER copy_t(t);
  bool copied_okay = compare(t, copy_t);


  if (copied_okay)
    cout << "Good.  It copies correctly." << endl;
  else 
    cerr << "ERROR:  Did no copy construct correctly" << endl;
  
  // Try assignment
  INTCONTAINER a;
  a.insertKeyAndValue("Red Herring",666);
  a = t;

  if (compare(a, t))
    cout << "Good. After assignment, It copies correctly" << endl;
  else
    cerr << "ERROR: Assignment didn't work correctly" << endl;

  if (a.contains("Red Herring"))
    cerr << "ERROR:  Red Herring shouldn't be there!" << endl;

}

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
void ContainerTest<INTCONTAINER,INTIT,STRINGCONTAINER,STRINGIT>::basicSuite (INTCONTAINER& t)
{
  cout << "Table has " << t.entries() << " elements" <<  endl;
  
  t.insertKeyAndValue("Key1", 1);
  int_u4 val = 0;
  if (!t.findValue("Key1", val)) {
    cerr << "ERROR: Didn't find Key1" << endl;
  } else 
    cout << "Good.  Found Key1 to be " << val << endl;


  cout << "Table has " << t.entries() << " elements" <<  endl;


  if (t.remove("Key1")) {
    cout << "Good.  Key1 removed from the table" << endl;
    if (t.findValue("Key1", val))
      cerr << "PROBLEM: Should not have been able to find Key1" << endl;
  } else {
    cout << "ERROR:  Should have been able to remove Key1" << endl;
  }


  // Insert a value
  t.insertKeyAndValue("Superman", 4);
  if (!t.findValue("Superman", val)) {
    cerr << "ERROR: Didn't find Key1" << endl;
  } else 
    cout << "Good.  Found Superman to be " << val << endl;

  cout << "Table has " << t.entries() << " elements" <<  endl;

  // Now try to replace the value
  t.insertKeyAndValue("Superman", 5);
  if (!t.findValue("Superman", val)) {
    cerr << "ERROR: Didn't find Key1" << endl;
  } else 
    cout << "Good.  Found Superman to be " << val << endl;

  cout << "Table has " << t.entries() << " elements" <<  endl;

  // Now remove it, and make sure it's not in there twice
  if (t.remove("Superman")) {
    cout << "Good.  Superman removed from the table" << endl;
    if (t.findValue("Superman", val))
      cerr << "PROBLEM: Should not have been able to find Superman" << endl;
  } else {
    cout << "ERROR:  Should have been able to remove Superman" << endl;
  }


  cout << "Table has " << t.entries() << " elements" <<  endl;


  t.insertKeyAndValue("Superman", 1);
  t.insertKeyAndValue("Batman",   2);
  t.insertKeyAndValue("Wonder Woman", 3);
  t.insertKeyAndValue("Aquaman", 4);
  t.insertKeyAndValue("Green Lantern", 5);
  t.insertKeyAndValue("Flash", 6);



  // test clear
  t.clear();

  cout << "Table has " << t.entries() << " elements" <<  endl;
  if (t.contains("Superman"))
    cerr << "ERROR:  Clear doesn't work." << endl;


  const char* superdudes[] = { 
    "Superman",
    "Batman",  
    "Wonder Woman",
    "Aquaman",
    "Green Lantern",
    "Flash",
    "Apache Chief", 
    "Black Lightning", 
    "Lex Luthor",
    "Joker", 
    "Catwoman",
    "Black manta", 
    "Sinestro", 
    "Reverse-Flash",
    "Giganta",
    0
  };

  for (int i=0; superdudes[i] !=0; i++) {
    // cerr << "Superdude[" << i << "] is " << superdudes[i] << endl;
    t.insertKeyAndValue(superdudes[i], i);
  }
 
  // test contains
  if (t.contains("Superman"))
    cout << "Good.  Superman in the table" << endl;
  else 
    cerr << "ERROR:  Superman not in table!" << endl;

  if (!t.contains("Supernam"))
    cout << "Good.  Supernam NOT in the table" << endl;
  else 
    cerr << "ERROR:  Supernam in table!" << endl;

  if (!t.contains(""))
    cout << "Good.  empty string NOT in the table" << endl;
  else 
    cerr << "ERROR: empty string in table!" << endl;
  
  
  // Test find
  string m;
  if (t.find("Flash", m)) {
    cout << "Good.  Found Flash's key to be " << m << endl;
  } else {
    cerr << "ERROR:  Flash should be in there" << endl;
  }
  
  if (!t.find("hsalF", m)) {
    cout << "Good.  Didn't find hsalF." << endl;
  } else {
    cerr << "ERROR:  FOund hsalF to be " << m << endl;
  }
  

  // Test find value
  if (!t.findValue("Sinestro", val)) {
    cerr << "ERROR: Didn't find Sinestro" << endl;
  } else 
    cout << "Good.  Found Sinestro to be " << val << endl;  

  if (t.findValue("Weaponers of Qward", val)) {
    cerr << "ERROR: Found Weaponers of Qward to be " << val << endl;
  } else 
    cout << "Good.  Didn't find Weaponer's of Qward" << endl;  
  

  // Test find key and value
  string key;
  if (!t.findKeyAndValue("Sinestro", key, val)) {
    cerr << "ERROR: Didn't find Sinestro" << endl;
  } else 
    cout << "Good.  Found " << key << " to be " << val << endl;  

  if (t.findValue("Weaponers of Qward", val)) {
    cerr << "ERROR: Found " << key << " to be " << val << endl;
  } else 
    cout << "Good.  Didn't find Weaponer's of Qward" << endl;  


  // Test isempty

  if (t.isEmpty()) {
    cerr << "ERROR:  Table is empty." << endl;
  } else {
    cout << "Good.  Table is not empty. " << endl;
  }



  // Delete all the elements out, one by one
  for (int j=0; superdudes[j]!=0; j++) {
    if (t.remove(superdudes[j])) {
      cout << "Good.  " << superdudes[j] << " removed." << endl;
      if (t.contains(superdudes[j])) {
	cerr << "ERROR:" << superdudes[j] << " should be gone" << endl;
      }
    }
  }


  // Now test isempty again

  if (!t.isEmpty()) {
    cerr << "ERROR:  Table is not empty." << endl;
  } else {
    cout << "Good.  Table is empty. " << endl;
  }


}					// ContainerTest::tests

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
void ContainerTest<INTCONTAINER, INTIT, STRINGCONTAINER, STRINGIT>::iterationTest ()
{
  STRINGCONTAINER t; 

  const char* first_names[] = { "Homer", "Marge", "Nelson", "Jimbo",
		          "Seymour", "Edna", "Apu", "Helen",
			  "Clancy", "Ralph", "Ned", "Abraham",
	                  "Montgomery", "Waylon", "Julius", "Hirschel", 
                          "Moe", "Nick", "Barney", "Patti",
			  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
			  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
			  "U", "V", "W", "X", "Y", "Z",
			  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
			 "a","b","c","d","e","f","g","h","i","j","k","l","m",
			 "n","o","p","q","r","s","t","u","v","w","x","y","z",

			 "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			 "10", "11", "12", "13", "14", "15", "16", "17", "18","19", 
			 "20", "21", "22", "23", "24", "25",
			  0 };
  const char* last_names[] = { "Simpson", "Bouvier", "Muntz", "Jones",
                         "Skinner", "Krabapple", "Nahasapeemapetelon", "Lovejoy",
			 "Wiggum", "Wiggum", "Flanders", "Simpson", 
                         "Burns", "Smithers", "Hibbert", "Kristofski", 
                         "Syzlak", "Riviera", "Gumble", "Bouvier",
			 "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			 "10", "11", "12", "13", "14", "15", "16", "17", "18","19", 
			 "20", "21", "22", "23", "24", "25",
			  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
			 "a","b","c","d","e","f","g","h","i","j","k","l","m",
			 "n","o","p","q","r","s","t","u","v","w","x","y","z",
			  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
			  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
			  "U", "V", "W", "X", "Y", "Z",
			 0 };

  int ii=0; int jj;
  for (ii=0; first_names[ii] != 0; ii++)
    t.insertKeyAndValue(first_names[ii], last_names[ii]);

  for (jj=ii-1; jj!=-1; jj--)
    t.remove(first_names[jj]);

  STRINGIT next(t);
  while (next())
    cerr << "ERROR:  Should not be anything else in this table" << endl;


  // Try a whole bunch of randomization.  We tend to favor inserts
  // during the beginning and removals from the beginning
  const int threshold = 100;
  for (int kk=1; kk<threshold; kk++) {


    // Choose a random thing in the table
    int where = rand() % jj;
    
    // If beats threshold, then we swap operations
    int operat = (rand() % threshold < kk);


    if (operat) {
      t.remove(first_names[where]);

    } else {
      t.insertKeyAndValue(first_names[where], last_names[where]);

    }
  }

  // Now that it has been pretty randomized, go through and
  // delete everything
  for (ii=0; ii<jj; ii++)
    t.remove(first_names[ii]);

  STRINGIT nn(t);
  while (nn()) {
    cerr << nn.key() << " " << nn.value() << endl;
  }
  
}

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
void ContainerTest<INTCONTAINER,INTIT,STRINGCONTAINER,STRINGIT>::newOpTest ()
{
  STRINGCONTAINER t; 
  t.insertKeyAndValue("test1", "test2");
  STRINGCONTAINER tcopy = t;
  try {
    string s = t("test1"); // okay
    string s2 = t("BADKEY"); // error
  } catch (const out_of_range& re) {
    cout << "As expected: " << re.what() << endl;
  }
  if (tcopy != t) cerr << "ERROR! Expected unchanged container" << endl;
}

template <class INTCONTAINER, class INTIT, class STRINGCONTAINER, class STRINGIT>
void ContainerTest<INTCONTAINER,INTIT,STRINGCONTAINER,STRINGIT>::swapTest ()
{
  STRINGCONTAINER t1; 
  t1.insertKeyAndValue("test1", "test2");
  t1.insertKeyAndValue("test10", "test20");
  t1.insertKeyAndValue("test100", "test200");
  STRINGCONTAINER t2;
  t2["hello"] = "there";
  t2["howdy"] = "ho";
  {
    cout << "t1 entries=" << t1.entries() << endl;
    STRINGIT nn(t1);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }
  {
    cout << "t2 entries=" << t2.entries() << endl;
    STRINGIT nn(t2);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }


  swap(t1, t2);
  {
    cout << "t1 entries=" << t1.entries() << endl;
    STRINGIT nn(t1);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }
  {
    cout << "t2 entries=" << t2.entries() << endl;
    STRINGIT nn(t2);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }



  t1["1"] = "2";
  t1["3"] = "3";
  t2.remove("test1");

  {
    cout << "t1 entries=" << t1.entries() << endl;
    STRINGIT nn(t1);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }
  {
    cout << "t2 entries=" << t2.entries() << endl;
    STRINGIT nn(t2);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }

  swap(t1, t2);

  {
    cout << "t1 entries=" << t1.entries() << endl;
    STRINGIT nn(t1);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }
  {
    cout << "t2 entries=" << t2.entries() << endl;
    STRINGIT nn(t2);
    while (nn()) {
      cout << nn.key() << " " << nn.value() << endl;
    }
  }

}

