
// Make sure we have all the decisions taken care of for this
// platform.

// If this fails, take a look at the ocport.h file for details.

// ///////////////////////////////////////////// Includes

#include "ocport.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// ///////////////////////////////////////////// Main

#define OUTPUTME(T, S)  {  \
  if (sizeof(T) != S)  \
    cerr << " **PROBLEM:  The sizeof your "#T" is incorrect." << endl; \
  else  \
    cout << " ok: The size of your "#T" is correct." << endl; \
}

int main ()
{
  // If this doesn't even compile, you need 
  // #define OC_NEEDS_BOOL
  cout << "DECISION:  Do we have a bool on this platform?" << endl;
  bool b = false; b = true;
  if (b) {
    cout << " ok.  This compiled.  OC_NEEDS_BOOL is correct" << endl;
  } else {
    cout << "Problems setting bools?" << endl;
  }

  // If this doesn't even compile, you need 
  // #define OC_NEW_STYLE_INCLUDES
  cout << "DECISION: Type of includes?" << endl;
  cout << " ok.  This compiled.  OC_NEW_STYLE_INCLUDES is correct" << endl;

  // Check sizes are right ... you may need
  cout << "DECISION: make sure we have ints right size" << endl;
  OUTPUTME(int_1,  1);
  OUTPUTME(int_u1, 1);
  OUTPUTME(int_2,  2);
  OUTPUTME(int_u2, 2);
  OUTPUTME(int_4,  4);
  OUTPUTME(int_u4, 4);
  OUTPUTME(int_8,  8);
  OUTPUTME(int_u8, 8);

  // #define OC_NO_STD_CHAR
  cout << "DECISION: Make sure an int_1 can hold negative values properly" << endl;
  int_1 a = -127;
  int ii = a;  
  cout << "--> ii had better be -127 and a should be < 0: ii=" << ii << " a<0 = " << (a<0) << endl;

  // If you see this, check your defines for ... 
  // #define OC_LONG_INT_IS_64BIT
  cout << "DECISION: size of your int_8 " << endl;
  if (sizeof(int_8) != 8) 
    cerr << " **PROBLEM:  The sizeof your int_8 is incorrect. Check your OC_LONG_INT_IS_64BIT setting." << endl;
  else 
    cout << " ok. 8 bytes correct, so your OC_LONG_INT_IS_64BIT is probably correct" << endl;

  // Check sizeof pointers
  cout << "DECISION: sizeof pointers " << endl;
#if OC_BYTES_IN_POINTER==8
  if (sizeof(int*) != 8) 
    cerr << " **PROBLEM: your OC_BYTES_IN_POINTER is set incorrectly" << endl;
  else
    cout << " ok. your OC_BYTES_IN_POINTER is set correctly" << endl;

#elif OC_BYTES_IN_POINTER==4
  if (sizeof(int*) != 4) 
    cerr << " **PROBLEM: your OC_BYTES_IN_POINTER is set incorrectly" << endl;
  else
    cout << " ok. your OC_BYTES_IN_POINTER is set correctly" << endl;

#else 

  //cerr << " ** PROBLEM!  You haven't set OC_BYTES_IN_POINTER set correctly at all" << endl;
  // User lets system discover automatically
  if (sizeof(int*) != sizeof(AVLP)) 
    cerr << " **PROBLEM: your OC_BYTES_IN_POINTER is set incorrectly" << endl;
  else
    cout << " ok. your OC_BYTES_IN_POINTER is set correctly" << endl;

#endif



}
