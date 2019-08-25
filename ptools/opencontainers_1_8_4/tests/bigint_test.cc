
// Test for making sure BigInt works
#include "ocbigint.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


template <typename I>
I nchoosek (int n, int k)
{
  int sym = n-k;
  if (sym>0 && sym<k) k = sym;

  I num = 1;
  I den = 1;
  for (int ii=0; ii<k; ii++) {
    num *= I(n-ii);
    den *= I(ii+1);
  }
  return num/den;
}

int_u2 AsInt (char c) 
{
  char ss[2] = { 0, 0 };
  ss[ (IsLittleEndian()) ? 0 : 1 ] = c;
  int_u2* u2p = reinterpret_cast<int_u2*>(&ss[0]);
  return *u2p;
}

template <class I, class BI>
void testBin (BigInt<I,BI> ii, const char* data=0, int len=0)
{
  cout << "BigInt: " << ii << endl;
  string s = MakeBinaryFromBigInt(ii);
  cout << "   .. bin len:" << s.length() << endl;
  for (size_t ii=0; ii<s.length(); ii++) {
      cout << AsInt(s[ii]) << " ";
  }
  cout << endl;

  // Check data?
  if (data!=0) {
    for (int ii=0; ii<len; ii++) {
      if (data[ii] != s[ii]) {
	cerr << "***ERROR" << ii << " Expected:" << string(data,len) << " but saw:" << s << endl;
	exit(1);
      }
    }
  }
}


void testingall (int low, int high, int inc)
{
  typedef BigInt<int_u1, int_u2> BI;

  cout << "Testing all:" << low << " " << high << " " << inc << endl;
  int kk = 0;
  for (int_8 ii=low; ii<high; ii+=inc) {
    if (++kk%100 == 0) cout << ii << "!" << endl;
    for (int_8 jj=-low; jj<high; jj+=inc) {
      //cout << jj << "!" << endl;

      BI lhs = ii;
      BI rhs = jj;
      BI rrr;
      int_8 result;
  
      rrr = lhs * rhs;
      result = ii * jj;
      if (rrr.stringize() != StringizeInt(result)) {
	cerr << "*" << endl;
	cerr << ii << " " << jj << endl;
	cerr << rrr << " " << result << endl;
	cerr << lhs << " " << rhs << endl;
	exit(1);
      }
      if (jj >0 && ii>=0) {
	// no floating point exception
	rrr = lhs / rhs;
	result = ii / jj;
	if (rrr.stringize() != StringizeInt(result)) {
	  cerr << "/" << endl;
	  cerr << ii << " " << jj << endl;
	  cerr << rrr << " " << result << endl;
	  exit(1);
	}

	rrr = lhs % rhs;
	result = ii % jj;
	if (rrr.stringize() != StringizeInt(result)) {
	  cerr << "%" << endl;
	  cerr << ii << " " << jj << endl;
	  cerr << rrr << " " << result << endl;
	  exit(1);
	}
      }
      
      rrr = lhs + rhs;
      result = ii + jj;
      if (rrr.stringize() != StringizeInt(result)) {
	cerr << "+" << endl;
	cerr << ii << " " << jj << endl;
	cerr << rrr << " " << result << endl;
	exit(1);
      }

      rrr = lhs - rhs;
      result = ii - jj;
      if (rrr.stringize() != StringizeInt(result)) {
	cerr << "-" << endl;
	cerr << ii << " " << jj << endl;
	cerr << rrr << " " << result << endl;
	exit(1);
      }
    }
  }
}

int main (int argc, char** argv)
{

  if (argc==2) {
    int_n sum;
    for (int n=1000; n<1099; n++) {
      //cout << "n" << n << endl;
      for (int x=0; x<n; x++) {
	//cout << "x" << x << endl;
	sum += nchoosek<int_n>(n,x);
	//cout << sum << endl;
      }
    }
    //sum.dump(cout) << endl;
    cout << sum << endl;
    exit(1);
  }

  if (argc==3) {
    int_n oi=0;
    for (int jj=0; jj<8000; jj++) {
      //int_8 sum = 0;
      oi = jj; // ~int_u8(0); // jj;
      for (int ii=0; ii<256000; ii+=1) {
	//sum += oi;
	//sum.singleDigitSub(oi);
	--oi;
	//oi.singleDigitSub(1);
	//       oi -= 1;
      }
    }
    cout << oi << endl;
    exit(1);
  }


  // make life SO much easier.
  typedef BigInt<int_u1, int_u2> BI;

  BI negation_test(10);
  cout << negation_test << endl;
  negation_test.negate();
  cout << negation_test << endl;
  negation_test.negate();
  cout << negation_test << endl;


  cout << (sizeof(BI)<= 32) << endl;  // make sure fits in Val


  BI a(100);
  cout << a << endl;;

  BI b(0);
  cout << b << endl;

  a = 127;
  cout << a << endl;;

  a = 128;
  cout << a << endl;;

  a = 32768;
  cout << a << endl;;

  a = -1;
  cout << a << endl;;

  a = -2;
  cout << a << endl;;

  BI c;
  cout << c << endl;

  BI i1 = 1;
  BI i2 = 1;
  i1+= i2;
  cout << i1 << endl;

  BI i3 = 127;
  BI i4 = 128;
  i3 += i4;
  cout << i3 << endl;

  i3 = 255;
  i4 = 1;
  i4+=i3;
  cout << i4 << endl;


  // Biggest int_8
  i3 = ~(int_8(1)<<63);
  cout << i3 << endl;
  i4 = 1;
  i3 += 1;
  cout << i3 << endl;

  cout << "bigger ints" << endl;
  i3 = 65535;
  cout << i3 << endl;
  i4 = 1;
  cout << i4 << endl;
  i3 += i4;
  cout << i3 << endl;

  cout << "bigger ints" << endl;
  i3 = 8128;
  cout << i3 << endl;
  i4 = 128;
  cout << i4 << endl;
  i3 += i4;
  cout << i3 << endl;


  // Try negataive
  cout << "Try subtract" << endl;
  i3 = 1;
  i4 = 1;
  i3 -= i4;
  cout << i3 << endl;

  i3=17;
  i4=16;
  i3-=i4;
  cout << i3 << endl;

  i3=127;
  i4=0;
  i3-=i4;
  cout << i3 << endl;

  i3=127;
  i4=128;
  i3-=i4;
  cout << i3 << endl;

  i3=65535;
  i4=1;
  i3-=i4;
  cout << i3 << endl;

  cout << "bigger ints" << endl;
  i3 = 65535;
  cout << i3 << endl;
  i4 = 65534;
  cout << i4 << endl;
  i3 -= i4;
  cout << i3 << endl;

  cout << "bigger ints" << endl;
  i3 = 65534;
  cout << i3 << endl;
  i4 = 65534;
  cout << i4 << endl;
  i3 -= i4;
  cout << i3 << endl;


  cout << "bigger ints" << endl;
  i3 = 65534;
  cout << i3 << endl;
  i4 = 65535;
  cout << i4 << endl;
  i3 -= i4;
  cout << i3 << endl;

  cout << "256-250" << endl;
  i3 = 256;
  cout << i3 << endl;
  i4 = 250;
  cout << i4 << endl;
  i3 -= i4;
  cout << i3 << endl;

  cout << "Try multiply" << endl;
  i3=10;
  i4= 10;
  i3 *= i4;
  cout << i3 << endl;

  i3=100;
  i4= 100;
  i3 *= i4;
  cout << i3 << endl;

  i3=65535;
  i4= 100;
  i3 *= i4;
  cout << i3 << endl;

  i3=65535*100;
  i4= 100;
  i3 *= i4;
  cout << i3 << endl;

  cout << "here" << endl;
  i3=65535*100;
  cout << i3 << endl;
  i4= 256;
  cout << i4 << endl;
  i3 *= i4;
  cout << i3 << endl;

  cout << "here2 " << endl;
  i3=65535*100;
  cout << i3 << endl;
  i4= 65535;
  cout << i4 << endl;
  i3 *= i4;
  cout << i3 << endl;

  cout << "here3 " << endl;
  i3=65535*100;
  cout << i3 << endl;
  i4= 65535*100;
  cout << i4 << endl;
  i3 *= i4;
  cout << i3 << endl;

  cout << "Try divide" << endl;
  i3 = 4;
  i4 = 16;
  BI rem, divver;
  BI::DivMod(i3, i4, rem, divver);
  cout << rem << endl;
  cout << divver << endl;

  i3 = 16;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  cout << rem << endl;
  cout << divver << endl;

  i3 = 17;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  cout << rem << endl;
  cout << divver << endl;

  i3 = 18;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  cout << rem << endl;
  cout << divver << endl;

  i3 = 19;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  cout << rem << endl;
  cout << divver << endl;


  i3 = 20;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  cout << rem << endl;
  cout << divver << endl;

  cout << "Testing stringize" << endl;
  for (int ii=-66538; ii<65538; ii++) {
    i3 = ii;
    string s = i3.stringize();
    //cout << s << endl;
    string ss = StringizeInt(ii);
    if (ss!=s) {
      cerr << s << " " << ss << endl;
      cerr << ii << " " << i3 << endl;
      exit(1);
    }
  }

  i3 = -65538;
  i4 = -65538;
  BI me = i3 * i4;
  cout << me << endl;

  i3 = 101;
  i4 = 10;
  me = i3/i4;
  BI rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = -101;
  i4 = 10;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = 101;
  i4 = -10;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = -101;
  i4 = -10;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = 100;
  i4 = 10;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = -100;
  i4 = 10;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = 100;
  i4 = -10;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = -100;
  i4 = -10;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = 10;
  i4 = 100;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = -10;
  i4 = 100;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = 10;
  i4 = -100;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  i3 = -10;
  i4 = -100;
  me = i3/i4;
  rr = i3 % i4;
  cout << me << endl;
  cout << rr << endl;

  cout << "testing inc" << endl;
  i3=-10;
  i4=10;
  for (int ii=0; ii<20; ii++) {
    cout << ++i3 << endl;
    cout << --i4 << endl;
  }

  i3=-10;
  i4=10;
  for (int ii=0; ii<20; ii++) {
    cout << i3++ << endl;
    cout << i4-- << endl;
  }

  i3=1;
  cout << --i3 << endl;

  i3=-13;
  i4 = -13;
  i3-=i4;
  cout << i3 << endl;

  i3=-65538;
  i4 = -65538;
  i3-=i4;
  cout << i3 << endl;

  i3=-13;
  i4 = 13;
  i3+=i4;
  cout << i3 << endl;

  i3=-65538;
  i4 = 65538;
  i3+=i4;
  cout << i3 << endl;

  i3=65538;
  i4 =-65538;
  i3+=i4;
  cout << i3 << endl;

  cout << "testing export" << endl;
  for (int_8 ii=-65538; ii<65538; ii++) {
    i3 = ii;
    int_8 out = i3.as();
    if (out != ii) {
      cerr << "no export" << ii << " " << out << endl;
      exit(1);
    }
  }

  i3 = -138;
  i4 = -238;
  cout << i3 - i4 << endl;
  cout << i3 << endl;
  cout << i4 << endl;
  i3 -= i4;
  cout << i3 << endl;

  
  cout << "Testing subtract" << endl;
  for (int_8 ii=-65538; ii<65538; ii+=100) {
    for (int_8 jj=-65538; jj<65538; jj+=100) {
      i3 = ii;
      i4 = jj;
      int_8 sub = (i3-i4).as();
      i3 -= i4;

      if (ii-jj != sub || ii-jj != i3) {
	cerr << "bad sub" << ii-jj << " " << sub << endl;
	cerr << ii << " " << jj << endl;
	exit(1);
      }
    }
  }
  
  testingall(-65538, 65538, 100);
  testingall(-258, 258, 1);
  

  BI sum = 0;
#define BIG 1
  if (BIG) {
    for (int ii=0; ii<65538; ii++) {
      if (ii%10000==0) cout << "..." << ii << endl;
      BI b = ii;
      string aaa = Stringize(ii);
      string bbb;
      {
	ostringstream os;
	b.print(os);
	bbb = os.str();
      }
      if (aaa!=bbb) {
	cerr << "Uh oh" << endl;
	cerr << aaa << endl;
	cerr << bbb << endl;
	cerr << ii << endl;
	//b.dump(cerr);
	exit(1);
      }
      sum += b;
      BI nl = nchoosek<BI>(ii+1,2);
      if (sum != nl) {
	cerr << "eh?" << endl;
	cout << ii << " sum = " << sum << " " << nl << endl;
	exit(1);
      }
    }
    cout << sum << endl;
    cout << nchoosek<BI>(65538, 2) << endl;
  }
  
  for (int ii=0; ii<30; ii++) {
    cout << ii << " choose ?" << endl;
    for (int jj=0; jj<ii; jj++) {
      
      int_u8 nk = nchoosek<int_u8>(ii,jj);
      BI nkb    = nchoosek<BI>(ii,jj);
      
      ostringstream os;
      nkb.print(os);
      string sb = os.str();

      string s = Stringize(nk);
      if (sb != s) { 
	cerr << "uh ok!" << endl; 
	cerr << ii << " choose " << jj << endl;
	cerr << nk << endl;
	cerr << nkb << endl;
	//nkb.dump(cerr) << endl;
	exit(1); 
      }

    }
  }

  cout << "Converting between real_8 and BigUint" << endl;
  real_8 orig = 1;
  BI rrr = 1;
  real_8 bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = 255.0;
  rrr = 255;
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = -255.0;
  rrr = -255;
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = 256.0;
  rrr = 256;
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = -256.0;
  rrr = -256;
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = 274365;
  rrr = 274365;
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = 27436538568045ULL;
  rrr = 27436538568045ULL;
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  cout.precision(16);
  orig = 281474976710656ULL; // 2**48
  rrr = 281474976710656ULL; // 2**48
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  // On SOME systems, if you compile with -O3 or -O4, the tests below
  // might not work!
  //{
  //  orig = 281474976710656666ULL; // (2**48) * 1000 + 666 
  //  BigUInt<int_u1,int_u2> mr  = 281474976710656666ULL; // 2**48 * 1000 + 666
  //  bb = MakeRealFromBigUInt(mr);
  //  cout << "real_8:" << bb << " BI:" << mr << " ==?" << (orig==bb) << endl;
  //}

  orig = 281474976710656666ULL; // (2**48) * 1000 + 666 
  //rrr  = 281474976710656666ULL; // 2**48 * 1000 + 666
  rrr  = 281474976710656666LL; // 2**48 * 1000 + 666
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = 281474976710656e+200; // (2**48) e+300
  rrr = 281474976710656ULL; // 2**48 
  for (int ii=0; ii<200; ii++) {
    rrr *= 10;
  }
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = 2.81474976710656e+200; // (2**48) e+300
  rrr = 281474976710656ULL; // 2**48 
  for (int ii=0; ii<(200-14); ii++) {
    rrr *= 10;
  }
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;

  orig = 281474976710656e+300; // (2**48) e+300
  rrr = 281474976710656ULL; // 2**48 
  for (int ii=0; ii<300; ii++) {
    rrr *= 10;
  }
  bb = MakeRealFromBigInt(rrr);
  cout << "real_8:" << bb << " BI:" << rrr << " ==?" << (orig==bb) << endl;


  cout << "Converting from real_8 to BigInt" << endl;

  BI ggg;
  BI orig_int = 1;
  bb = 1.0;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;


  orig_int = 0;
  bb = 0.0;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  // Impl defined: but usually just 0
  //real_8 what_happens = -1;
  //int_u8 this_happens = int_u8(what_happens);
  //cout << this_happens << endl;

  // negatives become 1
  orig_int = -1;
  bb = -1.0;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = -1000000;
  bb = -1000000;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = 255;
  bb = 255.0;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = 256;
  bb = 256.0;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = 65536;
  bb = 65536.0;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;


  orig_int = 65535;
  bb = 65535.0;
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = 281474976710656ULL; // 2**48
  bb = 281474976710656ULL; // 2**48
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  cout << ".. expect failures now ... real_8 can only hold about 48 bits of precision" << endl;

  orig_int = 281474976710656666ULL; // 2**48 * 1000 + 666
  bb = 281474976710656666ULL; // 2**48
  MakeBigIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  {
    int_n i1; int_un i2;
    Array<int_4> a;
    a.append(-256); 
    a.append(-255); 
    a.append(-1);
    a.append(0); 
    a.append(1);
    a.append(255); 
    a.append(256);
    Array<int_u4> b;
    b.append(0); 
    b.append(1);
    b.append(255); 
    b.append(256);
    b.append(65536);

    for (size_t ii=0; ii<a.length(); ii++) {
      for (size_t jj=0; jj<b.length(); jj++) {
	int_4 o1 = a[ii];
	int_u4 o2 = b[jj];

	i1 = a[ii];
	i2 = b[jj];

	cout << o1 << "&" << o2 << "  " << i1 << "&" << i2 << endl;
	// cout     << "<:" <<  bool(o1<o2) << bool(i1<i2)  << endl
	//     << " <=:"  << bool(o1<=o2) << bool(i1<=i2) << endl
	//     << " >:" << bool(o1>o2)  << bool(i1>i2)  << endl
	//     << " >=:" << bool(o1>=o2) <<  bool(i1>=i2)  << endl
	//     << " ==:" << bool(o1==o2) << bool(i1==i2)  << endl
	//     << " !=:" << bool(o1!=o2) << bool(i1!=i2) << endl
	//     << endl;

	
	if ( (int_u4(o1)<o2)  != (i1<i2) )   { cerr << "<" << endl; exit(1); }
	if ( (int_u4(o1)<=o2) != (i1<=i2))   { cerr << "<=" << endl; exit(1); }
	if ( (int_u4(o1)>o2)  != (i1>i2))   { cerr << ">" << endl; exit(1); }
	if ( (int_u4(o1)>=o2) != (i1>=i2))  { cerr << ">=" << endl; exit(1); }
	if ( (int_u4(o1)!=o2) != (i1!=i2))   { cerr << "!=" << endl; exit(1); }
	if ( (int_u4(o1)==o2) != (i1==i2))  { cerr << "==" << endl; exit(1); }


      }
    }


    for (size_t ii=0; ii<a.length(); ii++) {
      for (size_t jj=0; jj<b.length(); jj++) {

	int_4 o1 = a[ii];
	int_u4 o2 = b[jj];

	i1 = a[ii];
	i2 = b[jj];
	cout << o2 << "&" << o1 << "  " << i2 << "&" << i1 << endl;
	//cout << "<:" << bool(o2<o1)  << bool(i2<i1) << endl  
	//     <<" <=:"  << bool(o2<=o1) << bool(i2<=i1) << endl 
	//     << " >:" << bool(o2>o1) << bool(i2>i1) << endl 
	//     << " >=:" << bool(o2>=o1) << bool(i2>=i1) << endl 
	//     << " ==:" << bool(o2==o1) << bool(i2==i1) << endl 
	//     << " !=:" << bool(o2!=o1)  << bool(i2!=i1) << endl 
	//     << endl;

	if ( (o2<int_u4(o1))  != (i2<i1) )   { cerr << "<" << endl; exit(1); }
	if ( (o2<=int_u4(o1)) != (i2<=i1))   { cerr << "<=" << endl; exit(1); }
	if ( (o2>int_u4(o1))  != (i2>i1))   { cerr << ">" << endl; exit(1); }
	if ( (o2>=int_u4(o1)) != (i2>=i1))  { cerr << ">=" << endl; exit(1); }
	if ( (o2!=int_u4(o1)) != (i2!=i1))   { cerr << "!=" << endl; exit(1); }
	if ( (o2==int_u4(o1)) != (i2==i1))  { cerr << "==" << endl; exit(1); }


      }
    }
  }


  cout << "Trying to make BigInt from binary stream" << endl;
  {
    int_n i;
    MakeBigIntFromBinary("\x00\x01", 2, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x00", 1, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x01", 1, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x15_\xd0\xacK\x9b\xb6\x01", 8, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x15_\x04|\x9f\xb1\xe3\xf2\xfd\x1e\x66", 11, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x01\x01", 2, i);
    cout << i << endl;

    MakeBigIntFromBinary("\xff", 1, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x00\x80", 2, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x00\x80\x00", 3, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x00\x00\x01", 3, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x7f", 1, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x80\x00", 2, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x80", 1, i);
    cout << i << endl;

    MakeBigIntFromBinary("\x81", 1, i);
    cout << i << endl;
  }


  cout << "** Trying to make a binary stream from BigINT" << endl;
  {
    testBin(BI(0), "", 0);
    testBin(BI(1), "\x01", 1);
    testBin(BI(127), "\x7f", 1);
    testBin(BI(128), "\x80\x00", 2);
    testBin(BI(255), "\xff\x00", 2);
    testBin(BI(256), "\x00\x01", 2);
    testBin(BI(32767), "\xff\x7f", 2);
    testBin(BI(32768), "\x00\x80\x00", 3);
    testBin(BI(65535), "\xff\xff\x00", 3);
    testBin(BI(65536), "\x00\x00\x01", 3);
    testBin(BI(1000000000), "\x00\xca\x9a;", 4);

    testBin(BigInt<int_u2, int_u4>(0), "", 0);
    testBin(BigInt<int_u2, int_u4>(1), "\x01", 1);
    testBin(BigInt<int_u2, int_u4>(127), "\x7f", 1);
    testBin(BigInt<int_u2, int_u4>(128), "\x80\x00", 2);
    testBin(BigInt<int_u2, int_u4>(255), "\xff\x00", 2);
    testBin(BigInt<int_u2, int_u4>(256), "\x00\x01", 2);
    testBin(BigInt<int_u2, int_u4>(32767), "\xff\x7f", 2);
    testBin(BigInt<int_u2, int_u4>(32768), "\x00\x80\x00", 3);
    testBin(BigInt<int_u2, int_u4>(65535), "\xff\xff\x00", 3);
    testBin(BigInt<int_u2, int_u4>(65536), "\x00\x00\x01", 3);
    testBin(BigInt<int_u2, int_u4>(1000000000), "\x00\xca\x9a;", 4);

    testBin(BI(-1),   "\xff", 1);
    testBin(BI(-127), "\x81", 1);
    testBin(BI(-128), "\x80", 1);
    testBin(BI(-129), "\x7f\xff", 2);
    testBin(BI(-255), "\x01\xff", 2);
    testBin(BI(-256), "\x00\xff", 2);
    testBin(BI(-257), "\xff\xfe", 2);
    testBin(BI(-32767), "\x01\x80", 2);
    testBin(BI(-32768), "\x00\x80", 2);
    testBin(BI(-32769), "\xff\x7f\xff", 3);
    testBin(BI(-65534), "\x02\x00\xff", 3);
    testBin(BI(-65535), "\x01\x00\xff", 3);
    testBin(BI(-65536), "\x00\x00\xff", 3);
    testBin(BI(-65537), "\xff\xff\xfe", 3);
    testBin(BI(-1000000000), "\0006e\xc4", 4);

    testBin(BigInt<int_u2, int_u4>(-1),   "\xff", 1);
    testBin(BigInt<int_u2, int_u4>(-127), "\x81", 1);
    testBin(BigInt<int_u2, int_u4>(-128), "\x80", 1);
    testBin(BigInt<int_u2, int_u4>(-129), "\x7f\xff", 2);
    testBin(BigInt<int_u2, int_u4>(-255), "\x01\xff", 2);
    testBin(BigInt<int_u2, int_u4>(-256), "\x00\xff", 2);
    testBin(BigInt<int_u2, int_u4>(-257), "\xff\xfe", 2);
    testBin(BigInt<int_u2, int_u4>(-32767), "\x01\x80", 2);
    testBin(BigInt<int_u2, int_u4>(-32768), "\x00\x80", 2);
    testBin(BigInt<int_u2, int_u4>(-32769), "\xff\x7f\xff", 3);
    testBin(BigInt<int_u2, int_u4>(-65534), "\x02\x00\xff", 3);
    testBin(BigInt<int_u2, int_u4>(-65535), "\x01\x00\xff", 3);
    testBin(BigInt<int_u2, int_u4>(-65536), "\x00\x00\xff", 3);
    testBin(BigInt<int_u2, int_u4>(-65537), "\xff\xff\xfe", 3);
    testBin(BigInt<int_u2, int_u4>(-1000000000), "\0006e\xc4", 4);
  }

  cout << "Convert from BigUInt to BigInt" << endl;
  {
    
    int_un a1;
    int_un a2;
    a1 = 100;
    a2 = 666;
    int_n bbb = a1;
    cout << bbb << endl;
    bbb = a2;
    cout << bbb << endl;
    
    a1 = StringToBigInt("123456789123456789123456789");
    a2 = StringToBigInt("666666666666666666666666666");
    int_n bbbbb = a1;
    cout << bbbbb << endl;
    bbbbb = a2;
    cout << bbbbb << endl;
  }

  cout << "Convert from BigInt to BigUInt" << endl;
  {
    
    int_n a1;
    int_n a2;
    a1 = 100;
    a2 = -666;
    int_un bbb = a1;
    cout << bbb << endl;
    bbb = a2;
    cout << bbb << endl;
    
    a1 = StringToBigInt("123456789123456789123456789");
    a2 = StringToBigInt("-666666666666666666666666666");
    int_un bbbbb = a1;
    cout << bbbbb << endl;
    bbbbb = a2;
    cout << bbbbb << endl;
  }

  for (int ooo=-256; ooo<256; ooo+=127) 
  {
    cout << "----------------------ooo" << ooo << "Different types" << endl;
    int_n x0 = "1234567890123456789012345678901234567890"; cout << x0 << endl;
    int_n x1 = int_1(ooo); cout << x1 << endl;
    int_n x2 = int_u1(ooo); cout << x2 << endl;
    int_n x3 = int_2(ooo); cout << x3 << endl;
    int_n x4 = int_u2(ooo); cout << x4 << endl;
    int_n x5 = int_4(ooo); cout << x5 << endl;
    int_n x6 = int_u4(ooo); cout << x6 << endl;
    int_n x7 = int_8(ooo); cout << x7 << endl;
    int_n x8 = int_u8(ooo); cout << x8 << endl;
    int_n x9 = real_8(ooo); cout << x9 << endl;
    int_n xx = real_4(ooo); cout << xx << endl;
    int_n xy = int(ooo); cout << xy << endl;
    int_n xz = long(ooo); cout << xz << endl;
    //int_n xa = size_t(ooo); cout << xa << endl; // size_t is platform dependent

    x0 = "1234567890123456789012345678901234567890"; cout << x0 << endl;
    x1 = int_1(ooo); cout << x1 << endl;
    x2 = int_u1(ooo); cout << x2 << endl;
    x3 = int_2(ooo); cout << x3 << endl;
    x4 = int_u2(ooo); cout << x4 << endl;
    x5 = int_4(ooo); cout << x5 << endl;
    x6 = int_u4(ooo); cout << x6 << endl;
    x7 = int_8(ooo); cout << x7 << endl;
    x8 = int_u8(ooo); cout << x8 << endl;
    x9 = real_8(ooo); cout << x9 << endl;
    xx = real_4(ooo); cout << xx << endl;
    xy = int(ooo); cout << xy << endl;
    xz = long(ooo); cout << xz << endl;
    //xa = size_t(ooo); cout << xa << endl; // size_t is platform dependent

    int_n ii = int_n("100000000000000000000000000000000000000000000") + int_n(ooo);
    cout << ii << endl;

    {
      int_n ii = int_n("100000000000000000000000000000000000000000000") + ooo;
      cout << ii << endl;
      int_n jj = ooo + int_n("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_n ii = int_n("100000000000000000000000000000000000000000000") * ooo;
      cout << ii << endl;
      int_n jj = ooo * int_n("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_n ii = int_n("100000000000000000000000000000000000000000000") - ooo;
      cout << ii << endl;
      int_n jj = ooo - int_n("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_n ii = int_n("100000000000000000000000000000000000000000000") / ooo;
      cout << ii << endl;
      int_n jj = ooo / int_n("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_n ii = int_n("100000000000000000000000000000000000000000000") % int_4(ooo);
      cout << ii << endl;
      int_n jj = ooo % int_n("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }

  }
  {
    for (int i=-1; i<2; i++) {
      for (int j=-1; j<2; j++) {
	int_n ii=i;
	int_n jj=j;
	bool lt, ltn;
	
	lt = i<j;
	ltn = ii<jj;
	if (lt != ltn) { 
	  cerr << "< not working!" << endl;
	  cout << "ii is " << ii << ", jj is " << jj << lt << ltn << endl;
	  exit(1);
	}

	lt = i<=j;
	ltn = ii<=jj;
	if (lt != ltn) { 
	  cerr << "<= not working!" << endl;
	  cout << "ii is " << ii << ", jj is " << jj << lt << ltn << endl;
	  exit(1);
	}

	lt = i>j;
	ltn = ii>jj;
	if (lt != ltn) { 
	  cerr << "> not working!" << endl;
	  cout << "ii is " << ii << ", jj is " << jj << lt << ltn << endl;
	  exit(1);
	}

	lt = i>=j;
	ltn = ii>=jj;
	if (lt != ltn) { 
	  cerr << ">= not working!" << endl;
	  cout << "ii is " << ii << ", jj is " << jj << lt << ltn << endl;
	  exit(1);
	}

	lt = i==j;
	ltn = ii==jj;
	if (lt != ltn) { 
	  cerr << "== not working!" << endl;
	  cout << "ii is " << ii << ", jj is " << jj << lt << ltn << endl;
	  exit(1);
	}

	lt = i!=j;
	ltn = ii!=jj;
	if (lt != ltn) { 
	  cerr << "!= not working!" << endl;
	  cout << "ii is " << ii << ", jj is " << jj << lt << ltn << endl;
	  exit(1);
	}



      }
    }
  }

  cout << "Okay!" << endl;
}
