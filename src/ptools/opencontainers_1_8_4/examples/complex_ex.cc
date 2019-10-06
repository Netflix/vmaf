
// Demonstrate how Permutations work

#include "occomplex.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  complex_8 a0;
  complex_8 a1(1);
  complex_8 a2(2,3);
  complex_8 a3(4.0,5.0f);

  cout << "**Build some complex_8s:";
  cout << a0 << " " << a1 << " " << a2 << " " << a3 << endl;

  complex_16 b0;
  complex_16 b1(1); 
  complex_16 b2(2,3);
  complex_16 b3(4.0, 5.0f);

  cout << "**Build some complex_16s";
  cout << b0 << " " << b1 << " " << b2 << " " << b3 << endl;

  cout << "**Test complex_8 x= COMPLEX_8 ops" << endl;
  cout << a2 << "+=" << a3 << " -> ";
  a2 += a3; cout << a2 << endl;
  cout << a2 << "-=" << a3 << " -> ";
  a2 -= a3; cout << a2 << endl;
  cout << a2 << "*=" << a3 << " -> ";
  a2 *= a3; cout << a2 << endl;
  cout << a2 << "/=" << a3 << " -> ";
  a2 /= a3; cout << a2 << endl;

  cout << "**Test complex_8 x= SCALAR ops" << endl;
  real_4 scalar1 = 6.0;
  cout << a2 << "+=" << scalar1 << " -> ";
  a2 += scalar1; cout << a2 << endl;
  cout << a2 << "-=" << scalar1 << " -> ";
  a2 -= scalar1; cout << a2 << endl;
  cout << a2 << "*=" << scalar1 << " -> ";
  a2 *= scalar1; cout << a2 << endl;
  cout << a2 << "/=" << scalar1 << " -> ";
  a2 /= scalar1; cout << a2 << endl;

  cout << "**Test complex_16 x= COMPLEX_16 ops" << endl;
  cout << b2 << "+=" << b3 << " -> ";
  b2 += b3; cout << b2 << endl;
  cout << b2 << "-=" << b3 << " -> ";
  b2 -= b3; cout << b2 << endl;
  cout << b2 << "*=" << b3 << " -> ";
  b2 *= b3; cout << b2 << endl;
  cout << b2 << "/=" << b3 << " -> ";
  b2 /= b3; cout << b2 << endl;

  cout << "**Test complex_16 x= SCALAR ops" << endl;
  real_8 scalar2 = 6.0;
  cout << b2 << "+=" << scalar2 << " -> ";
  b2 += scalar2; cout << b2 << endl;
  cout << b2 << "-=" << scalar2 << " -> ";
  b2 -= scalar2; cout << b2 << endl;
  cout << b2 << "*=" << scalar2 << " -> ";
  b2 *= scalar2; cout << b2 << endl;
  cout << b2 << "/=" << scalar2 << " -> ";
  b2 /= scalar2; cout << b2 << endl;
  

  cout << "**Test complex_8 x COMPLEX_8 ops" << endl;
  complex_8 a5;
  cout << a2 << "+" << a3 << " -> ";
  a5 = a2 + a3; cout << a5 << endl;
  cout << a2 << "-" << a3 << " -> ";
  a5 = a2 - a3; cout << a5 << endl;
  cout << a2 << "*" << a3 << " -> ";
  a5 = a2 * a3; cout << a5 << endl;
  cout << a2 << "/" << a3 << " -> ";
  a5 = a2 / a3; cout << a5 << endl;

  cout << "**Test complex_8 x= SCALAR ops" << endl;
  real_4 scalar3 = 6.0;
  cout << a2 << "+" << scalar3 << " -> ";
  a5 = a2 + scalar3; cout << a5 << endl;
  cout << scalar3 << "+" << a2 << " -> ";
  a5 = scalar3 + a2; cout << a5 << endl;

  cout << a2 << "-" << scalar3 << " -> ";
  a5 = a2 - scalar3; cout << a5 << endl;
  cout << scalar3 << "-" << a2 << " -> ";
  a5 = scalar3 - a2; cout << a5 << endl;

  cout << a2 << "*" << scalar3 << " -> ";
  a5 = a2 * scalar3; cout << a5 << endl;
  cout << scalar3 << "*" << a2 << " -> ";
  a5 = scalar3 * a2; cout << a5 << endl;

  cout << a2 << "/" << scalar3 << " -> ";
  a5 = a2 / scalar3; cout << a5 << endl;
  cout << scalar3 << "/" << a2 << " -> ";
  a5 = scalar3 / a2; cout << a5 << endl;

  cout << "**Test complex_16 x= SCALAR ops" << endl;
  complex_16 b5;
  real_8 scalar4 = 6.0;
  cout << b2 << "+" << scalar4 << " -> ";
  b5 = b2 + scalar4; cout << b5 << endl;
  cout << scalar4 << "+" << b2 << " -> ";
  b5 = scalar4 + b2; cout << b5 << endl;

  cout << b2 << "-" << scalar4 << " -> ";
  b5 = b2 - scalar4; cout << b5 << endl;
  cout << scalar4 << "-" << b2 << " -> ";
  b5 = scalar4 - b2; cout << b5 << endl;

  cout << b2 << "*" << scalar4 << " -> ";
  b5 = b2 * scalar4; cout << b5 << endl;
  cout << scalar4 << "*" << b2 << " -> ";
  b5 = scalar4 * b2; cout << b5 << endl;

  cout << b2 << "/" << scalar4 << " -> ";
  b5 = b2 / scalar4; cout << b5 << endl;
  cout << scalar4 << "/" << b2 << " -> ";
  b5 = scalar4 / b2; cout << b5 << endl;


  cout << "**Test: Mix it up, just make sure these compile" << endl;
  a5 = a2 + 7;  a5 = 7 +a2; a5 = a2 + 7.0;  a5 = 7.0 + a2;
  a5 += 7;  a5 += 7.0; a5 +=0; cout << a5 << endl;

  b5 = b2 + 7;  b5 = 7 +b2; b5 = b2 + 7.0;  b5 = 7.0 + b2;
  b5 += 7;  b5 += 7.0; b5 +=0; cout << b5 << endl;

  complex_8 a6(b5);  cout << "a6 = " << a6 << endl;  
  complex_16 b6(a6); cout << "b6 = " << b6 << endl;
  a6 = b5; cout << "a6 = " << a6 << endl;  
  b6 = a6; cout << "b6 = " << b6 <<endl;

  complex_8 aa;
  aa = 0; { istrstream is(" (3+4j)"); is >> aa; cout << aa << endl; }
  aa = 0; { istrstream is(" ( 3 + 4j)");  is >> aa; cout << aa << endl; }
  aa = 0; { istrstream is("   (   3.1415  -  4.333 j )");  is >> aa; cout << aa << endl; }
}

