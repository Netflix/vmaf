
#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE) 
using namespace OC;
#endif

template <class INT>
void Tester (INT n)
{
  string fs = StringizeInt(n);
  string s  = GenericStringize(n);
  if (s != fs) { 
    cerr << "YUK!" << TagFor((INT*)0) << " " << s << " " << fs << endl; 
    exit(1); 
  }
  cout << fs << " " << s << endl;
}

template <class INT>
void TesterU (INT n)
{
  string fs = StringizeUInt(n);
  string s  = GenericStringize(n);
  if (s != fs) { 
    cerr << "YUK!" << TagFor((INT*)0) << " " << s << " " << fs << endl; 
    exit(1); 
  }
  cout << fs << " " << s << endl;
}

int main (int argc, char* argv[])
{
  if (argc==1) {
    cout << "**Testing the stringize routines" << endl;
    //cout << Stringize(int_1(-128)) << endl;
    //cout << StringizeInt(int_1(-128)) << endl;
    cout << StringizeInt(int_2(-32768)) << endl; 
    cout << StringizeInt(int_2(-1)) << endl; 
    cout << StringizeInt(int_2(0)) << endl; 
    cout << StringizeInt(int_2(32767)) << endl; 
    cout << StringizeInt(int_2(-32768)) << endl; 
    
    
    //Tester(int_1(-128));
    //Tester(int_1(+127));
    Tester(int_2(-32768));
    Tester(int_2(+32767));
    Tester(int_4(-2147483648LL));
    Tester(int_4(+2147483647L));
    //Tester(int_8(-9223372036854775808LL)); // compiler complains about this ... do equivalent below
    Tester(int_8(int_8(1)<<63));
    Tester(int_8(+9223372036854775807LL));
    
    //Tester(int_u1(0));
    //Tester(int_u1(255));
    TesterU(int_u2(0));
    TesterU(int_u2(+65535));
    TesterU(int_u4(0));
    TesterU(int_u4(+4294967295UL));
    TesterU(int_u8(0ULL));
    TesterU(int_u8(18446744073709551615ULL));
    
    for (int ii=-32768; ii<=32767; ii+=1) {
      Tester(int_2(ii));
    }
    
    for (int ii=0; ii<=65535; ii+=1) {
      TesterU(int_u2(ii));
    }
  }

  // Test how fast we can insert into a Table.  Compare this to a
  // similar Python program for diffs ...  We seem to be 20% faster for
  // inserting ints, deep copies we are significantly faster (2x?),
  // and about 25% slower for ints.  If we use OCString, then we are
  // almost the same speed!  ARGH! I wish STL string used the small
  // string optimization!
  else {
    Val big = new Tab();
    string key = "The quick brown fox jumped over the lazy dogs so many time to make a really big string";
    for (int ii=0; ii<1000; ii++) {
      big[Stringize(ii*100)] = key; 
    }
    
    for (int jj=0; jj<400; jj++) {
      Tab oo;
      
      for (int ii=0; ii<10000; ii++) {
	Val value = (big); // key
	//Val key= ii; 
	Val key=Stringize(ii); 
	oo.swapInto(key, value);
	//oo[key] = value;
      }
    }
  }


}

#if defined(PYTHON_PROGRAM)

from copy import *

key =  "The quick brown fox jumped over the lazy dogs so many time to make a really big string"
big = { }
for ii in xrange(0,1000) :
   big[str(ii*100)] = key

for x in xrange(0, 400) :
  oo = { }
  for ii in xrange(0,10000) :
     # value = deepcopy(big)
     value = big
     key = ii
     # key = str(ii)
     oo[key] = value

#endif
