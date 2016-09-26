
// Make sure int_un and int_n fit well into the Val structure

#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  int_n me = 1789234790273409057ULL;
  real_8 me8 = me.as();
  cout << me << " " << me8 << endl;
  string mes = me.stringize();
  cout << me << " " << mes << endl;

  Val you = "234958927634589234759263459234523475927345";
  real_8 you8 = you;
  cout << you << " " << you8 << endl;
  int_n yos = you;
  cout << you << " " << yos << endl;
  int_8 i8 = you;
  // cout << you << " " << i8 << " .. this is zero because the stream fails because the value is too big" << endl; // To impl defined to be in a test

  you = yos;
  cout << you << " " << Stringize(you) << endl;
  string outter = you;
  cout << you << " " << outter << endl;
  int_8 ooo = you;
  cout << you << " " << ooo << endl;

  you = "123456789";
  you8 = you;
  cout << you << " " << you8 << endl;
  yos = you;
  cout << you << " " << yos << endl;
  i8 = you;
  cout << you << " " << i8 << endl;
  

  //cerr << sizeof(int_n) << " " << sizeof(int_un) << endl;
  if (sizeof(int_n)>VALBIGINT) { 
    cout << "sizeof(int_n) is " << sizeof(int_n) << endl
	 << "which is NOT VALBIGINT" << VALBIGINT << endl;
  }
  if (sizeof(int_un)>VALBIGUINT) { 
    cout << "sizeof(int_un) is " << sizeof(int_un) << endl
	 << "which is NOT VALBIGINT" << VALBIGUINT << endl;
  }

  int_un u = 256;
  Val v = u;
  cout << v << endl;
  v = u;
  cout << v << endl;

  int_n n = -2112;
  Val vv = n;
  cout << vv << endl;
  vv = n;
  cout << vv << endl;

  v = 100;
  int_un nn = v;
  cout << nn << endl;
  nn = v;
  cout << nn << endl;
  //nn = int_un(v);   // Ambiguous: causes syntax errors: see below
  cout << nn << endl;
  //cout << int_un(v) << endl; // Ambiguous: see next to see how to rectify 
  cout << v.operator int_un() << endl;
  
  v = -100;
  int_n uu = v;
  cout << uu << endl;
  uu = v;
  cout << uu << endl;
  //uu = int_n(v); // Ambiguous: causes syntax errors: see below
  cout << uu << endl;
  //cout << int_n(v) << endl; // Ambiguous
  cout << v.operator int_n() << endl; // Not ambigious, but cumbersome


  cout << string(v) << endl;
  cout << int_8(v) << endl;
  //cout << Tab(v) << endl;
  //  Tab t = v;

  // Problem area: make sure real_8 negative converts out as int_u8
  //real_8 rrr = -1.1;
  //int_u8 gg = rrr;
  //cout << gg << endl;
  //int_un bb = rrr;
  //cout << bb << endl;
  //Val kk= -1.1;
  //gg = kk;
  //bb = kk;
  //cout << "gg=" << gg << " bb=" << bb << endl;
  
  // problem area:negative bigint stays negative?
  Val lhs = -0.66;
  Val rhs = None;
  int_n l1 = lhs;
  int_n r1 = rhs;
  int_8 ll1 = lhs;
  int_8 rr1 = rhs;
  cout << l1 << " " << r1 << " " << ll1 << " " << rr1 << endl;

  Arr a = "[ None, 'abc', (1+2j), (-1-100j), {'a': 1, 'b':2 }, [], {}, o{'a':1, 'b':2 }, (1,2,3), -2000000000000, -256, -1.1, -1, -1.0, -.66, 0, .66, 1, 1.0, 1.1, 256, 200000000000]";
  bool no_compare = false;
  for (size_t ii=0; ii<a.length(); ii++) {
    Val& v1 = a[ii];
    int_un u_intun_1;
    int_u8 u_int8_1 = 666;
    int_n i_intn_1;
    int_8 i_int8_1 = 666;
    try {
      u_intun_1 = v1;
    } catch (const exception& e) {
      cerr << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
      no_compare = true;
    }
    try {
      i_intn_1 = v1;
    } catch (const exception& e) {
      cerr << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
      no_compare = true;
    }
    try {
      u_int8_1 = v1;
    } catch (const exception& e) {
      cerr << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
      no_compare = true;
    }
    try {
      i_int8_1 = v1;
    } catch (const exception& e) {
      cerr << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
      no_compare = true;
    }
    cout << "v1=" << v1 << endl; 
    for (size_t jj=0; jj<a.length(); jj++) {
      Val& v2 = a[jj];

      try {
	cout << "..[" << bool(v1<v2) << " " << bool(v1==v2) << " " << bool(v1>v2) << "]";
      } catch (const exception& e) {
	cout << "Can't compare types for some reason:" << e.what() << endl;
      }
      cout << "v2=" << v2;
      int_un u_intun_2;
      int_n  i_intn_2;
      int_u8 u_intu8_2 = 666;
      int_8  i_int8_2 = 666;
      try {
	u_intun_2 = v2;
      } catch (const exception& e) {
	cout << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
	continue;
      }
      try { 
	i_intn_2 = v2;
      } catch (const exception& e) {
	cout << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
	continue;
      }
      try {
	u_intu8_2 = v2;
      } catch (const exception& e) {
	cout << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
	continue;
      }
      try { 
	i_int8_2 = v2;
      } catch (const exception& e) {
	cout << "Can't convert " << v1 << " into an int_un:" << e.what() << endl;
	continue;
      }
      cout << endl;
      //cout << "[" << u_intun_1 << " " << u_int8_1 << " " << u_intun_2 << " " << i_intn_1 << " " << " " << i_intn_2 << "]" << endl;
      if (!no_compare) {
	try {
	  if (bool (u_intun_1<u_intun_2) != bool(u_int8_1<u_intu8_2)) {
	    cout << "big uint didn't compare same as uint" << u_intun_1 << " " << u_intun_2 << " " << u_int8_1 << " " << u_intu8_2 << endl;
	    exit(1);
	  }
	  if (bool (i_intn_1<i_intn_2) != bool(i_int8_1<i_int8_2)) {
	    cout << "big int didn't compare same as int" << u_intun_1 << " " << u_intun_2 << " " << u_int8_1 << " " << u_intu8_2 << endl;
	    exit(1);
	  }
	}
	catch (const exception& e) {
	  cout << "Can't compare types for some reason:" << e.what() << endl;
	}
      }
      no_compare = false;
    }
    cout << endl;
  }

  for (int ii=0; ii<15; ii++) {
    cout << "10**" << ii << " is ... " << endl;
    cout << IntExp(int_u8(10), int_u8(ii)) << endl;
  }
  cout << "0**0 is" << IntExp(0,0) << endl;

  cout << "[" << DecimalApprox(100, 1, 10) << "]" << endl;
  cout << "[" << DecimalApprox(100, 2, 10) << "]" << endl;
  cout << "[" << DecimalApprox(1, 1, 10) << "]" << endl;
  cout << "[" << DecimalApprox(0, 1, 10) << "]" << endl;
  cout << "[" << DecimalApprox(1, 2, 1) << "]" << endl;
  cout << "[" << DecimalApprox(1, 3, 4) << "]" << endl;
  cout << "[" << DecimalApprox(2, 3, 4) << "]" << endl;
  cout << "[" << DecimalApprox(14502345, 78978, 16) << "]" << endl;

  cout.precision(16); 
  cout << real_8(14502345.0/78978.0) << endl;
  for (int ii=0; ii<25; ii++) {
    cout << "i" << DecimalApprox(14502345, 78978, ii) << endl;
    if (ii+3 <17) cout.precision(ii+3); 
    cout << "r" << real_8(14502345.0/78978.0) << endl;
  }
  cout << "[" << DecimalApprox(2,3,100) << "]" << endl;
  cout << "[" << DecimalApprox(22,7,7) << "]" << endl;
  cout << "[" << DecimalApprox(500,100,7) << "]" << endl;
}
