
// Test the Array

// ///////////////////////////////////////////// Include Files

#include "ocarray.h"
#include "ocstring.h"
#include <stdio.h>

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif



// ///////////////////////////////////////////// The ArrayTest Class

class ArrayTest {
    
  public:

    // ///// Methods

    int tests ();

}; // ArrayTest


void intArrayTest()
{
  int ii;
  Array<int> a(16);

  // Append 10 things...no resize necessary
  for (ii=0; ii<10; ii++)
    a.append(ii);

  // Print it out ... this tests the ()
  cout << a;

  // Append another 10 things.  This should cause a resize
  for (ii=10; ii<20; ii++)
    a.append(ii);

  // Print it out ... this tests the []
  cout << "### testing []" << endl;
  for (ii=0; ii<int(a.length()); ii++)
    cout << a[ii] << " ";
  cout << endl;


  // Print it out again 
  cout << "### testing ()" << endl;
  for (ii=0; ii<int(a.length()); ii++)
    cout << a(ii) << " ";
  cout << endl;

  // Print it out
  cout << "### testing at" << endl;
  for (ii=0; ii<int(a.length()); ii++)
    cout << a.at(ii) << " ";
  cout << endl;

  
  // Out of range
  cout << "### testing out of range for []" << endl;
  try {
    cout << a[int(a.length())] << endl;
  } catch (const out_of_range& b) {
    cout << "GOOD: Range error as expected" << endl;
  }


  // Out of range
  cout << "### testing out of range for at" << endl;
  try {
    cout << a.at(int(a.length())) << endl;
  } catch (out_of_range& b) {
    cout << "GOOD: Range error as expected" << endl;
  }



  // Try another one, and let it resize all the way up
  cout << "### Starting with a size of zero and letting it resize up" << endl;
  Array<int> b(0);
  for (ii=0; ii<32; ii++) {
    b.append(ii);
    cout << "## Just inserted " << ii << endl << b << endl; 
  }

  // Try the copy constructor
  cout << "### Testing the copy constructor" << endl;
  Array<int> bc(b);
  cout << bc << endl;

  // Testing the operator =
  cout << "### Testing the op=" << endl;
  Array<int> beq(2);
  cout << "beq (before) = " << beq << endl;
  cout << "bc (before) = " << bc << endl;
  beq = bc;
  cout << "beq (after) = " << beq << endl;
  cout << "bc (after) = " << bc << endl;


  // Test the clear
  cout << "### testing the clear" << endl;
  cout << "bc (before) = " << bc << endl;
  bc.clear();
  cout << "bc (after) = " << bc << endl;

  // Test the contains
  cout << "### testing the contains" << endl;
  cout << "beq (before) = " << beq << endl;

  if (beq.contains(5))
    cout << "GOOD: contains works ... (has 5)" << endl;
  else
    cout << "ERROR:  contains DOES NOT work" << endl;
  if (!beq.contains(100))
    cout << "GOOD: contains works ... (DOES NOT have 100)" << endl;
  else
    cout << "ERROR:  contains DOES NOT work" << endl;
  cout << "beq (after) = " << beq << endl << endl;  


  // Test find
  cout << "### testing the find" << endl;
  int find_int = 999;
  cout << "beq (before) = " << beq << endl;

  if (beq.find(5, find_int))
    cout << "GOOD: find works ... (has " << find_int << endl;
  else
    cout << "ERROR:  find DOES NOT work: " << find_int << endl;
  if (!beq.find(100, find_int))
    cout << "GOOD: find works ... (DOES NOT have 100 and didn't set " << find_int << ")" << endl;
  else
    cout << "ERROR:  find DOES NOT work" << endl;
  cout << "beq (after) = " << beq << endl << endl;

  // Test index
  cout << "### testing the index" << endl;
  cout << "beq (before) = " << beq << endl;

  int ind = beq.index(5);
  if (ind != Array<int>::ARRAY_NPOS)
    cout << "GOOD: index works ... (found 5 at index " << ind << ")" << endl;
  else
    cout << "ERROR:  index DOES NOT work: " << find_int << endl;
  ind = beq.index(100);
  if (ind == Array<int>::ARRAY_NPOS)
    cout << "GOOD: index works ... (DOES NOT have 100)" << endl;
  else
    cout << "ERROR:  index DOES NOT work" << endl;
  cout << "beq (after) = " << beq << endl << endl;

  // take a look at the data
  cout << "### testing data" << endl;
  const int* d = beq.data();
  for (ii=0; ii<int(beq.length()); ii++)
    cout << d[ii] << " ";
  cout << endl;


  // Check if insert works
  cout << "### testing insert" << endl;
  Array<int> arg(4);
  for (ii=0; ii<3; ii++)
    arg.insert(ii);
  cout << arg << endl;

  // Check if insertAt works
  cout << "### testing insertAt" << endl;
  cout << endl;
  cout << " # testing if can insertAt in empty" << endl;
  Array<int> emp(0);
  
  try {
    emp.insertAt(1, 666);
  } catch (out_of_range& be) {
    cout << "GOOD:  Can't insertAt early into empty" << endl;
  }

  cout << " # insert right at front of empty" << endl;
  emp.insertAt(0, 777);
  cout << emp << endl;


  cout << " # try multiple inserts ats" << endl;
  emp.insertAt(0, 666);
  cout << emp << endl;
  emp.insertAt(0,555);
  cout << emp << endl;
  emp.insertAt(2, 600);
  cout << emp << endl;


  // Test is empty
  cout << endl << "### testing isEmpty" << endl;
  Array<int> empt;
  if (emp.isEmpty())
    cout << "GOOD: is empty" << endl;
  else
    cout << "ERROR: empty" << endl;
  empt.append(999);
  if (!empt.isEmpty())
    cout << "GOOD: not empty" << endl;
  else
    cout << "ERROR: empty" << endl;

  // test prepend
  Array<int> tr(2);
  cout << "### testing prepend" << endl;
  for (ii=0; ii<4; ii++)
    tr.prepend(ii);
  cout << tr << endl;

  // Test removeAll
  cout << "### testing removeAll" << endl;
  Array<int> g;
  for (ii=0; ii<20; ii++)
    g.append(0);
  g.append(3);
  cout << "Removing [4] from " << endl
       << g << endl;
  size_t items = g.removeAll(4);
  cout << " removed a total of " << items << " items,  giving:" << endl 
       << g << endl;
  
  cout << "Removing [0] from " << endl
       << g << endl;
  items = g.removeAll(0);
  cout << " removed a total of " << items << " items,  giving:" << endl 
       << g << endl;

  cout << "Removing [3] from " << endl
       << g << endl;
  items = g.removeAll(3);
  cout << " removed a total of " << items << " items,  giving:" << endl 
       << g << endl;

  
  // Tests remove
  cout << "### testing remove" << endl;
  Array<int> r(1);
  for (ii=0; ii<8; ii++)
    r.prepend(ii);
  for (ii=0; ii<8; ii++) {
    cout << " r before = " << r << endl;
    if (r.remove(ii)) 
      cout << " removed " << ii << " okay ... left: " << r << endl;
    else
      cout << "ERROR removing " << ii << endl;
  }
  if (r.remove(8))
    cout << "ERROR:  Can't remove anymore" << endl;


  // Tests removeAt
  cout << "### testing removeAt" << endl;
  Array<int> f(0);
  for (ii=0; ii<8; ii++)
    f.append(ii);
  for (ii=0; ii<8; ii++) {
    Array<int> fcopy(f);
    cout << " removing position " << ii << " from " << fcopy << endl;
    int t = fcopy.removeAt(ii);
    cout << " at that pos was " << t << " and left over array is " << fcopy << endl;
  }

  try {
    f.removeAt(8);
  } catch (out_of_range& e) {
    cout << "GOOD:  caught bounds error for removeAt" << endl;
  }


  // Test removeRange
  cout << "### testing removeRange" << endl;
  Array<int> rang(0), fullRang(0);
  for (ii=0; ii<8; ii++) 
    fullRang.append(ii); // Fill

  for (ii=-1; ii<9; ii++) {
    for (int kk=-1; kk<9; kk++) {
      rang = fullRang;   // Full copy every time
      // cerr << "ii = " << ii << " kk = " << kk << endl;
      bool no_except = true;
      try {
	rang.removeRange(ii, kk);
      } catch (const out_of_range& e) {
	no_except = false;
	cout << "*BadIndex for start = " << ii << " run_length = " << kk << endl;
      }
      
      if (no_except) {
	cout << "Effects of removeRange("<<ii << ", " << kk << "):" << rang << endl;
      }


    }
  }

}

void stringArrayTest ()
{
  // It's important that we try this out with non primitive datatypes.

  int ii;
  Array<string> a(16);

  // Append 10 things...no resize necessary
  for (ii=0; ii<10; ii++)
    a.append(Stringize(ii));

  // Print it out ... this tests the ()
  cout << a << endl;

  // Append another 10 things.  This should cause a resize
  for (ii=10; ii<20; ii++)
    a.append(Stringize(ii));

  // Print it out ... this tests the []
  cout << "### testing []" << endl;
  for (ii=0; ii<int(a.length()); ii++)
    cout << a[ii] << " ";
  cout << endl;


  // Print it out again 
  cout << "### testing ()" << endl;
  for (ii=0; ii<int(a.length()); ii++)
    cout << a(ii) << " ";
  cout << endl;

  // Print it out
  cout << "### testing at" << endl;
  for (ii=0; ii<int(a.length()); ii++)
    cout << a.at(ii) << " ";
  cout << endl;

  // Out of range
  cout << "### testing out of range for []" << endl;
  try {
    cout << a[int(a.length())] << endl;
  } catch (out_of_range& b) {
    cout << "GOOD: Range error as expected" << endl;
  }


  // Out of range
  cout << "### testing out of range for at" << endl;
  try {
    cout << a.at(int(a.length())) << endl;
  } catch (out_of_range& b) {
    cout << "GOOD: Range error as expected" << endl;
  }



  // Try another one, and let it resize all the way up
  cout << "### Starting with a size of zero and letting it resize up" << endl;
  Array<string> b(0);
  for (ii=0; ii<32; ii++) {
    b.append(Stringize(ii));
    cout << "## Just inserted " << ii << endl << b << endl; 
  }

  // Try the copy constructor
  cout << "### Testing the copy constructor" << endl;
  Array<string> bc(b);
  cout << bc << endl;

  // Testing the operator =
  cout << "### Testing the op=" << endl;
  Array<string> beq(2);
  cout << "beq (before) = " << beq << endl;
  cout << "bc (before) = " << bc << endl;
  beq = bc;
  cout << "beq (after) = " << beq << endl;
  cout << "bc (after) = " << bc << endl;


  // Test the clear
  cout << "### testing the clear" << endl;
  cout << "bc (before) = " << bc << endl;
  bc.clear();
  cout << "bc (after) = " << bc << endl;

  // Test the contains
  cout << "### testing the contains" << endl;
  cout << "beq (before) = " << beq << endl;

  if (beq.contains("5"))
    cout << "GOOD: contains works ... (has 5)" << endl;
  else
    cout << "ERROR:  contains DOES NOT work" << endl;
  if (!beq.contains("100"))
    cout << "GOOD: contains works ... (DOES NOT have 100)" << endl;
  else
    cout << "ERROR:  contains DOES NOT work" << endl;
  cout << "beq (after) = " << beq << endl << endl;  


  // Test find
  cout << "### testing the find" << endl;
  string find_string = "999";
  cout << "beq (before) = " << beq << endl;

  if (beq.find("5", find_string))
    cout << "GOOD: find works ... (has " << find_string << endl;
  else
    cout << "ERROR:  find DOES NOT work: " << find_string << endl;
  if (!beq.find("100", find_string))
    cout << "GOOD: find works ... (DOES NOT have 100 and didn't set " << find_string << ")" << endl;
  else
    cout << "ERROR:  find DOES NOT work" << endl;
  cout << "beq (after) = " << beq << endl << endl;

  // Test index
  cout << "### testing the index" << endl;
  cout << "beq (before) = " << beq << endl;

  int ind = beq.index("5");
  if (ind != Array<string>::ARRAY_NPOS)
    cout << "GOOD: index works ... (found 5 at index " << ind << ")" << endl;
  else
    cout << "ERROR:  index DOES NOT work: " << endl;
  ind = beq.index("100");
  if (ind == Array<string>::ARRAY_NPOS)
    cout << "GOOD: index works ... (DOES NOT have 100)" << endl;
  else
    cout << "ERROR:  index DOES NOT work" << endl;
  cout << "beq (after) = " << beq << endl << endl;

  // take a look at the data
  cout << "### testing data" << endl;
  const string* d = beq.data();
  for (ii=0; ii<int(beq.length()); ii++)
    cout << d[ii] << " ";
  cout << endl;


  // Check if insert works
  cout << "### testing insert" << endl;
  Array<string> arg(4);
  for (ii=0; ii<3; ii++)
    arg.insert(Stringize(ii));
  cout << arg << endl;

  // Check if insertAt works
  cout << "### testing insertAt" << endl;
  cout << endl;
  cout << " # testing if can insertAt in empty" << endl;
  Array<string> emp(0);
  
  try {
    emp.insertAt(1, "666");
  } catch (out_of_range& be) {
    cout << "GOOD:  Can't insertAt early into empty" << endl;
  }

  cout << " # insert right at front of empty" << endl;
  emp.insertAt(0, "777");
  cout << emp << endl;


  cout << " # try multiple inserts ats" << endl;
  emp.insertAt(0, "666");
  cout << emp << endl;
  emp.insertAt(0,"555");
  cout << emp << endl;
  emp.insertAt(2, "600");
  cout << emp << endl;


  // Test is empty
  cout << endl << "### testing isEmpty" << endl;
  Array<string> empt;
  if (empt.isEmpty())
    cout << "GOOD: is empty" << endl;
  else
    cout << "ERROR: empty" << endl;
  empt.append("999");
  if (!empt.isEmpty())
    cout << "GOOD: not empty" << endl;
  else
    cout << "ERROR: empty" << endl;

  // test prepend
  Array<string> tr(2);
  cout << "### testing prepend" << endl;
  for (ii=0; ii<4; ii++)
    tr.prepend(Stringize(ii));
  cout << tr << endl;

  // Test removeAll
  cout << "### testing removeAll" << endl;
  Array<string> g;
  for (ii=0; ii<20; ii++)
    g.append("0");
  g.append("3");
  cout << "Removing [4] from " << endl
       << g << endl;
  size_t items = g.removeAll("4");
  cout << " removed a total of " << items << " items,  giving:" << endl 
       << g << endl;
  
  cout << "Removing [0] from " << endl
       << g << endl;
  items = g.removeAll("0");
  cout << " removed a total of " << items << " items,  giving:" << endl 
       << g << endl;

  cout << "Removing [3] from " << endl
       << g << endl;
  items = g.removeAll("3");
  cout << " removed a total of " << items << " items,  giving:" << endl 
       << g << endl;

  
  // Tests remove
  cout << "### testing remove" << endl;
  Array<string> r(1);
  for (ii=0; ii<8; ii++)
    r.prepend(Stringize(ii));
  for (ii=0; ii<8; ii++) {
    cout << " r before = " << r << endl;
    if (r.remove(Stringize(ii))) 
      cout << " removed " << ii << " okay ... left: " << r << endl;
    else
      cout << "ERROR removing " << ii << endl;
  }
  if (r.remove("8"))
    cout << "ERROR:  Can't remove anymore" << endl;


  // Tests removeAt
  cout << "### testing removeAt" << endl;
  Array<string> f(0);
  for (ii=0; ii<8; ii++)
    f.append(Stringize(ii));
  for (ii=0; ii<8; ii++) {
    Array<string> fcopy(f);
    cout << " removing position " << ii << " from " << fcopy << endl;
    string t = fcopy.removeAt(ii);
    cout << " at that pos was " << t << " and left over array is " << fcopy << endl;
  }

  try {
    f.removeAt(8);
  } catch (out_of_range& e) {
    cout << "GOOD:  caught bounds error for removeAt" << endl;
  }


  // Test removeRange
  cout << "### testing removeRange" << endl;
   // create a long string that HAS to get heap collected so that
   // we know the destructors are working
  string SomeLongString = "The quick brown fox jumped over the lazy dogs";
  SomeLongString += SomeLongString;
  SomeLongString += SomeLongString;
  Array<string> rang(0), fullRang(0);
  for (ii=0; ii<8; ii++) 
    fullRang.append(Stringize(ii)+SomeLongString); // Fill

  for (ii=-1; ii<9; ii++) {
    for (int kk=-1; kk<9; kk++) {
      rang = fullRang;   // Full copy every time
      // cerr << "ii = " << ii << " kk = " << kk << endl;
      bool no_except = true;
      try {
	rang.removeRange(ii, kk);
      } catch (const out_of_range& e) {
	no_except = false;
	cout << "*BadIndex for start = " << ii << " run_length = " << kk << endl;
	for (int ll=0; ll<8; ll++)
	  if (fullRang[ll] != rang[ll]) 
	    cout << "PROBLEM!!! fullRang and rang should be same arrays!" << endl;
      }

      // Output only if no exception:  
      if (no_except) {
	cout << "Effects of removeRange("<<ii << ", " << kk << "):";
	for (int ll=0; ll<int(rang.length()); ll++) {
	  cout << rang[ll].substr(0,1) << " ";
	}
	cout << endl;
      }

    }
  }
}


Array<char> StringToArray (const string& a)
{
  Array<char> ret(int(a.length()));
  for (int ii=0; ii<int(a.length()); ++ii) {
    ret.append(a[ii]);
  }
  return ret;
}

void compareTest ()
{
  const char* a[] = { "a", "aa", "aab", 0 };
  
  for (int ii=0; a[ii] !=0; ++ii) {
    string aa = "aa";
    cout << aa << " <  " << a[ii] << " :" 
	 << int(StringToArray(aa) < StringToArray(a[ii])) << endl;
    cout << aa << " <= " << a[ii] << " :" 
	 << int(StringToArray(aa) <= StringToArray(a[ii])) << endl;
    cout << aa << " >  " << a[ii] << " :" 
	 << int(StringToArray(aa) > StringToArray(a[ii])) << endl;
    cout << aa << " >= " << a[ii] << " :" 
	 << int(StringToArray(aa) >= StringToArray(a[ii])) << endl;
    cout << aa << " != " << a[ii] << " :" 
	 << int(StringToArray(aa) != StringToArray(a[ii])) << endl;
    cout << aa << " == " << a[ii] << " :" 
	 << int(StringToArray(aa) == StringToArray(a[ii])) << endl;
  }
}

void swapTest ()
{
  Array<string> s1(2), s2(8);
  for (int ii=0; ii<3; ii++) {
    s1.append(Stringize(ii));
    s2.append("abcdefghijklmnopqrstuvwxyz0123456789"+Stringize(ii));
  }

  cout << s1 << " capacity:" << s1.capacity() << " length:" << s1.length()<< endl;
  cout << s2 << " capacity:" << s2.capacity() << " length:" << s2.length()<< endl;

  swap(s1, s2);

  cout << s1 << " capacity:" << s1.capacity() << " length:" << s1.length()<< endl;
  cout << s2 << " capacity:" << s2.capacity() << " length:" << s2.length()<< endl;
  for (int ii=0; ii<2; ii++) {
    s1.append("YYZ"+Stringize(ii));
    s2.append("ZZZ"+Stringize(ii));
  }
  
  cout << s1 << " capacity:" << s1.capacity() << " length:" << s1.length()<< endl;
  cout << s2 << " capacity:" << s2.capacity() << " length:" << s2.length()<< endl;
  swap(s1, s2);
  cout << s1 << " capacity:" << s1.capacity() << " length:" << s1.length()<< endl;
  cout << s2 << " capacity:" << s2.capacity() << " length:" << s2.length()<< endl;
}


void fillTest ()
{
  cout << "Fill Tests:" << endl;
  Array<string> a(10);
  cout << Stringize(a) << endl;
  a.fill("one", 1);
  cout << Stringize(a) << endl;
  a.fill("two", 2);
  cout << Stringize(a) << endl;
  a.fill("three", 3);
  cout << Stringize(a) << endl;
  a.fill("threemore", 3);
  cout << Stringize(a) << endl;
  a.fill("to_capacity");
  cout << Stringize(a) << endl;
  cout << "a.capacity() == " << a.capacity() << endl;
  a.append("force_doubles");
  cout << "a.capacity() == " << a.capacity() << endl;
  cout << Stringize(a) << endl;
  a.fill("AFTERDOUBLE");
  cout << Stringize(a) << endl;
  cout << "a.capacity() == " << a.capacity() << endl;

  // Fill makes little sense with pointers
  Array<int> b(6);
  cout << Stringize(b) << endl;
  b.fill(666);
  cout << Stringize(b) << endl;
}

// This tests if the new MoveArray primitives work
#define ITERS 1
#include "ocval.h"
void valTest ()
{
  for (int jj=0; jj<ITERS; jj++) {
    Arr a;
    Val v1 = Tab("{ 'a':1, 'b':2, 'c':3, 'd':{ 'e': 5 } }");
    Val v2 = Arr("[1,2,3,'abc', {'a':1}, (1,2.2,'three',[])]");
    Val v3 = Tup(1,2,3,"abc", Tab("{'a':1}"), Tup(1,2.2,"three"), Arr());
    Val v4 = OTab("o{ 'a':1, 'b':2, 'c':3, 'd':{ 'e': 5 } }");
    Val v5 = 1;
    for (int ii=0; ii<5000; ii++) {
      a.append(v1);
      //cerr << a << endl;
      a.append(v2);
      //cerr << a << endl;
      a.append(v3);
      //cerr << a << endl;
      a.append(v4);
      //cerr << a << endl;
      a.append(v5);
      //cerr << a << endl;
    }
    if (jj%ITERS==1) cout << a[99] << endl;
  }
}

// ///////////////////////////////////////////// ArrayTest Methods

int ArrayTest::tests()
{
  intArrayTest();
  stringArrayTest();
  compareTest();
  swapTest();
  fillTest();
  valTest();

  Array<int> a(3);
  cout << a.capacity() << " " << a.length() << " " << a.getBit() << endl;
  a.setBit(1);
  cout << a.capacity() << " " << a.length() << " " << a.getBit() << endl;
  a.setBit(0);
  cout << a.capacity() << " " << a.length() << " " << a.getBit() << endl;

  return 0;
}


int main ()
{
  ArrayTest t;
  return t.tests();
}



