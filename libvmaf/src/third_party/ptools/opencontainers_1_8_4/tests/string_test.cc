
// Some simple tests for the string class.

// ///////////////////////////////////////////// Include Files

// Use OC's string
#include "ocport.h"
#include "ocstring.h" 
#include "ocstreamingpool.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

// ///////////////////////////////////////////// The StringTest Class

class StringTest {

  public:

    // ///// Methods

    int tests ();

}; // StringTest


// ///////////////////////////////////////////// StringTest Methods

int StringTest::tests ()
{
  OCString s1;
  cout << "Empty String:\"" << s1 << "\"" << endl;

  OCString as1;
  cout << "Another Empty string:\"" << as1 << "\"" << endl;

  if (s1!=as1)
    cout << "PROBLEM:  Empty strings aren't same!" << endl;

  OCString s2("Hello There Everyone");
  cout << "Constructing a string from a C-style string:\"" << s2 
       << "\"" << endl;

  // Now reconstruct the same string
  OCString s3("Hello There Everyone");
  cout << "Constructing same string from a C-style string:\"" << s3
       << "\"" << endl;

  // They should compare to the same thing
  if (s3==s2) {
    cout << "Okay!  s3 and s2 compared correctly" << endl;
  } else {
    cout << "PROBLEM:  The strings s2 and s3 should be the same" << endl;
  }

  // Compare against a plain C style const string
  if (s3=="Hello There Everyone") {
    cout << "Okay!  s3 and c-style string compared correctly" << endl;
  } else {
    cout << "PROBLEM:  The s3 and c-style string should be same" << endl;
  }

  // Compare against a plain C style const string
  if ("Hello There Everyone"==s3) {
    cout << "Okay!  c-style string and s3 compared correctly" << endl;
  } else {
    cout << "PROBLEM:  The c-style string and s3 should be same" << endl;
  }


  // Now test inequality
  OCString s4("There Hello Everyone");
  

  // They should compare to the same thing
  if (s3!=s4) {
    cout << "Okay!  s3 and s4 compared correctly" << endl;
  } else {
    cout << "PROBLEM:  The strings s2 and s4 should not be the same" << endl;
  }

  // Compare against a plain C style const string
  if (s4!="Hello There Everyone") {
    cout << "Okay!  s4 and c-style string compared correctly" << endl;
  } else {
    cout << "PROBLEM:  The s4 and c-style string should not be same" << endl;
  }

  // Compare against a plain C style const string
  if ("Hello There Everyone"!=s4) {
    cout << "Okay!  c-style string and 43 compared correctly" << endl;
  } else {
    cout << "PROBLEM:  The c-style string and s3 should not be same" << endl;
  }


  // test copy construction
  OCString sc(s3);
  if (s3!=sc) {
    cout << "PROBLEM:  sc and s3 should be same from copy construction" << endl;

  } else {

    cout << "Okay!  sc and s3 should be same from copy construction" << endl;
  }


  // Test assignment;
  OCString ss;
  cout << "Before assignment: ss=" << ss << " and sc=" << sc << endl;
  ss = sc;
  cout << "After assignment:  ss=" << ss << " and sc=" << sc << endl;
  if (ss==sc) {
    cout << "Assignment okay" << endl;
  } else {
    cout << "Assignment broken" << endl;
  }

  // test +=
  OCString a1("Blue");
  OCString a2("Snowcone");
  cout << "Before +=:  a1 is \"" << a1 << "\"" << endl;
  cout << "Before +=:  a2 is \"" << a2 << "\"" << endl;
  a1 += a2;
  cout << "After  +=:  a1 is \"" << a1 << "\"" << endl;
  cout << "After  +=:  a2 is \"" << a2 << "\"" << endl;

  // Test concatenate
  OCString c1("Stone");
  OCString c2("Wall");
  cout << "Before concatenate: c1=\"" << c1 << "\"" << endl;
  cout << "Before concatenate: c2=\"" << c2 << "\"" << endl;
  OCString c12 = c1 + c2;
  cout << "After concatenate: c1+c2=\"" << c12 << "\"" << endl;
  cout << "After concatenate: c1=\"" << c1 << "\"" << endl;
  cout << "After  concatenate: c2=\"" << c2 << "\"" << endl;


  // Make sure that if I do the same concatenate that we find it.
  OCString cc1("Stone");
  OCString cc2("Wall");
  OCString cc12(c1+c2);
  if (cc12 == c12 && c1 == cc1 && c2 == c2) {
    cout << "Correct:  Same string still after concatenate" << endl;
  } else {
    cout << "PROBLEM: Different strings" << endl;
  }

  // Create the string directly and make sure it goes in there the
  // same way
  OCString same("StoneWall");
  if (cc12==same) {
    cout << "Correct:  Straight string same as concatenated string" << endl;
  } else {
    cout << "PROBLEM:  Straight string NOT same as concated string" << endl;
  }



  // Test the at method
  OCString at_test = "something";
  for (int ijk=0; ijk<= int(at_test.length())+1; ijk++) {
    try {
      cout << "ijk = " << ijk << ":";
      cout << at_test.at(ijk) << endl;
    } catch (...) {  
      cout << "ijk=" << ijk << " is correctly out of range" << endl;
    }
  }

  return 0;
}					// StringTest::tests




// ///////////////////////////////////////////// Main Program

int main ()
{
  StringTest t;
  return t.tests();
  return 0;
}

