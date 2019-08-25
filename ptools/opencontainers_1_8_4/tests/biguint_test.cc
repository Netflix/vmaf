
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
  //cerr << "WORKING: num=" << num << ";den=" << den << endl;
  //typedef Array<int_u4> AA;
  //cerr << "WORKING HERE" << endl;
  //AA* np = (Array<int_u4>*) &num;
  //cerr << "WORKING num as array of " << np->length() << " " << *np << endl;
  //AA* dp = (Array<int_u4>*) &den;
  //cerr << "WORKING den as array of " << dp->length() << " " << *dp << endl;
  //I res = num/den;
  //AA* rp = (Array<int_u4>*) &res;
  //cerr << "WORKING res as array of " << rp->length() << " " << *rp << endl;

  return num/den;
}

template <typename I>
I nchoosek2 (int n, int k)
{
  int sym = n-k;
  if (sym>0 && sym<k) k = sym;

  I num = 1;
  I den = 1;
  for (int ii=0; ii<k; ii++) {
    num *= I(n-ii);
    den *= I(ii+1);
  }
  //cerr << "num=" << num << ";den=" << den << endl;
  //typedef Array<int_u4> AA;
  //cerr << "HERE" << endl;
  //AA* np = (Array<int_u4>*) &num;
  //cerr << "num as array of " << np->length() << " " << *np << endl;
  //AA* dp = (Array<int_u4>*) &den;
  //cerr << "den as array of " << dp->length() << " " << *dp << endl;
  
  I q,r; 
  I::DivMod2(num, den, q, r);
  //cerr << "q is " << q << " r is " << r << endl;

  //AA* rp = (Array<int_u4>*) &q;
  //cerr << "res as array of " << rp->length() << " " << *rp << endl;

  return q;
}

int_u2 AsInt (char c) 
{
  char ss[2] = { 0, 0 };
  ss[ (IsLittleEndian()) ? 0 : 1 ] = c;
  int_u2* u2p = reinterpret_cast<int_u2*>(&ss[0]);
  return *u2p;
}

template <class I, class BI>
void testBin (BigUInt<I,BI> ii)
{
  cout << "BigInt: " << ii << endl;
  string s = MakeBinaryFromBigUInt(ii);
  cout << "   .. bin len:" << s.length() << endl;
  for (size_t ii=0; ii<s.length(); ii++) {
      cout << AsInt(s[ii]) << " ";
  }
  cout << endl;
}


/*
template <class I, class BI>
void normalize_test (I seed, int s)
{
  static const I baseshift = sizeof(I)<<3; 
  cerr << s << endl;
  int n = 2;
  I* v_ = new I[n];
  cerr << "v_ = ";
  for (int ii=0; ii<n; ii++) {
    v_[ii] = seed+327*ii;
    cerr << v_[ii] << " ";
  }
  cerr << endl;
  cerr << I(seed) << endl;
  I* vn = new I[n];
  {
    // THIS IS WHERE THE stuff go down
    // Shift vn 
    for (int i=n-1; i>0; i--) {
      // vn[i] = (v_[i] << s) | (v_[i-1] >> (baseshift-s));
      I temp1 = (v_[i] << s);
      I temp2 = ((v_[i-1]) >> (baseshift-s));
      I temp3 = temp1 | temp2;
      cerr << "TEMPS:" << temp1 << " " << temp2 << " " << temp3 << endl;
      // temp2 is WRONG!!!!! it should be 0
      vn[i] = temp3;
    }
    vn[0] = v_[0] << s;
  }
  // output
  cerr << "vn = ";
  for (int ii=0; ii<n; ii++) {
    cerr << vn[ii] << " ";
  }
  cerr << endl;
}
*/

int main (int argc, char** argv)
{
  typedef  int_un BM;
  // typedef BigUInt<int_u2, int_u4> BM;
  if (argc==2) {
    
    BM sum, sum2;

    for (int n=1000; n<1099; n++) {
      //cout << "n" << n << endl;
      for (int x=0; x<n; x++) {
	//cout << "x" << x << endl;
	sum += nchoosek<BM>(n,x);
	//sum2 += nchoosek2<BM>(n,x);
	//if (sum != sum2) { cerr << "!!! Not equal!! n and x" << n << " " << x << endl; cerr << sum << endl << sum2 << endl; exit(1);}
	//cout << sum << endl;
      }
    }
    //sum.dump(cout) << endl;
    cout << sum << endl;
    cout << sum2 << endl;
    exit(1);
  }

  if (argc==3) {
    int_un oi=0;
    for (int jj=0; jj<8000; jj++) {
      //int_un sum = 0;
      oi = jj; // ~int_u8(0);
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
  typedef BigUInt<int_u1, int_u2> BI;
  //  typedef BigUInt<int_u4, int_u8> BBBI;

  //for (int_u8 ii=0; ii<10; ii++) {
  // cerr << BBBI::nlz(ii) << endl;
  //}
  //cerr << BBBI::nlz(0x3ffffffffffffffEULL) << endl;
  //cerr << BBBI::nlz(0x4ffffffffffffffEULL) << endl;
  //cerr << BBBI::nlz(0x7ffffffffffffffEULL) << endl;
  //cerr << BBBI::nlz(0x8ffffffffffffffEULL) << endl;
  //cerr << BBBI::nlz(0xfffffffffffffffEULL) << endl;
  //cerr << BBBI::nlz(0xffffffffffffffffULL) << endl;
  //exit(1);

  // RTS
  {
    /*
    for (int aa = 0; aa<2550000; aa++) {
      if ((aa & 0xff)==0) cout << aa << " ";
      for (int bb = 1; bb<255; bb++) {
	BI a=aa;
	BI b=bb;
	BI q,r;
	a.singleDigitDivide(a,b, q,r);
	//cout << a << "/" << b << "= q(" << q << ") r(" << r << ")" << endl;
	if (aa / bb != q) {
	  cerr << "Wrong quotient:" << aa << " " << bb << " " << q << endl;
	  exit(1);
	}
	if (aa % bb != r) {
	  cerr << "Wrong remainder:" << aa << " " << bb << " " << q << endl;
	  exit(1);
	}
	
      }
    }
    */
    //exit(1);

    /*
    for (int ii=0; ii<65536; ii++) {
      int_u4 x = ii;
      cout << "nlz(" << x << ") =" << BI::nlz(x) << endl;
    }

    for (int_u4 ii=0x80000000; ii<0x80000004; ii++) {
      int_u4 x = ii;
      cout << "nlz(" << x << ") =" << BI::nlz(x) << endl;
    }
    for (int_u4 ii=0x7fffff00; ii<0x80000004; ii++) {
      int_u4 x = ii;
      cout << "nlz(" << x << ") =" << BI::nlz(x) << endl;
    }
    for (int_u4 ii=0x3fffff00; ii<0x40000004; ii++) {
      int_u4 x = ii;
      cout << "nlz(" << x << ") =" << BI::nlz(x) << endl;
    }
   
    for (int ii=0; ii<255; ii++) {
      int_u1 x = ii;
      cout << "nlz(" << x << ") =" << int(BI::nlz(x)) << endl;
    }
   
    exit(1);
    */


    for (int ii=0; ii<95537000; ii+=22200) {
      //if (ii%10000==0) cerr << ii << endl;

      for (int jj=1; jj<9553700; jj+=1220) {

	int_u8 q1 = ii/jj;
	int_u8 r1 = ii%jj;

	BI q,r;
	BI::DivMod(ii, jj, q, r);
	if (q1 != q) { 
	  cerr << "DIV" << q1 << " " << q << " " << r << endl; 
	  char *n = 0;
	  cout << *n << endl;
	}
	if (r1 != r) { 
	  cerr << "REM" << r1 << " " << r << " " << q << endl; 
	  char *n = 0;
	  cout << *n << endl;
	}
      }
    }
  }
  // RTS



  BI a(100);
  a.dump(cout);

  BI b(0);
  b.dump(cout);

  a = 127;
  a.dump(cout);

  a = 128;
  a.dump(cout);

  a = 32768;
  a.dump(cout);

  // Keep warning! Test to make sure we can see the warning
  // a = -1; // This is the line that would give the warning ... to make compiles cleaner, we do the line below
  a = BI(int_u8(-1));
  a.dump(cout);
  cout << "Expecting a 'cannot convert warning': JUST LIKE C! on line " << __LINE__ << endl;

  a = int_u8(-2); // no warning
  a.dump(cout);

  BI c;
  c.dump(cout);

  BI i1 = 1;
  BI i2 = 1;
  i1+= i2;
  i1.dump(cout);

  BI i3 = 127;
  BI i4 = 128;
  i3 += i4;
  i3.dump(cout);

  i3 = 255;
  i4 = 1;
  i4+=i3;
  i4.dump(cout);

  // Biggest int_8
  i3 = ~(int_8(1)<<63);
  i3.dump(cout);
  i4 = 1;
  i3 += 1;
  i3.dump(cout);

  cout << "bigger ints" << endl;
  i3 = 65535;
  i3.dump(cout);
  i4 = 1;
  i4.dump(cout);
  i3 += i4;
  i3.dump(cout);

  cout << "bigger ints" << endl;
  i3 = 8128;
  i3.dump(cout);
  i4 = 128;
  i4.dump(cout);
  i3 += i4;
  i3.dump(cout);

  // Try negataive
  cout << "Try subtract" << endl;
  i3 = 1;
  i4 = 1;
  i3 -= i4;
  i3.dump(cout);

  i3=17;
  i4=16;
  i3-=i4;
  i3.dump(cout);

  i3=127;
  i4=0;
  i3-=i4;
  i3.dump(cout);

  i3=127;
  i4=128;
  i3-=i4;
  i3.dump(cout);

  i3=65535;
  i4=1;
  i3-=i4;
  i3.dump(cout);

  cout << "bigger ints" << endl;
  i3 = 65535;
  i3.dump(cout);
  i4 = 65534;
  i4.dump(cout);
  i3 -= i4;
  i3.dump(cout);

  cout << "bigger ints" << endl;
  i3 = 65534;
  i3.dump(cout);
  i4 = 65534;
  i4.dump(cout);
  i3 -= i4;
  i3.dump(cout);


  cout << "bigger ints" << endl;
  i3 = 65534;
  i3.dump(cout);
  i4 = 65535;
  i4.dump(cout);
  i3 -= i4;
  i3.dump(cout);

  cout << "256-250" << endl;
  i3 = 256;
  i3.dump(cout);
  i4 = 250;
  i4.dump(cout);
  i3 -= i4;
  i3.dump(cout);


  cout << "Try multiply" << endl;
  i3=10;
  i4= 10;
  i3 *= i4;
  i3.dump(cout);

  i3=100;
  i4= 100;
  i3 *= i4;
  i3.dump(cout);

  i3=65535;
  i4= 100;
  i3 *= i4;
  i3.dump(cout);

  i3=65535*100;
  i4= 100;
  i3 *= i4;
  i3.dump(cout);

  cout << "here" << endl;
  i3=65535*100;
  i3.dump(cout);
  i4= 256;
  i4.dump(cout);
  i3 *= i4;
  i3.dump(cout);

  cout << "here2 " << endl;
  i3=65535*100;
  i3.dump(cout);
  i4= 65535;
  i4.dump(cout);
  i3 *= i4;
  i3.dump(cout);

  cout << "here3 " << endl;
  i3=65535*100;
  i3.dump(cout);
  i4= 65535*100;
  i4.dump(cout);
  i3 *= i4;
  i3.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 15;
  i4 = 15;
  BI res;
  int digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 256;
  i4 = 256;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 5;
  i4 = 2;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 52;
  i4 = 2;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 53;
  i4 = 2;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 54;
  i4 = 2;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 54;
  i4 = 53;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 128;
  i4 = 127;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "testing choose closest" << endl;
  i3 = 65538;
  i4 = 261;
  digit = BI::ChooseClosest(i3, i4, res);
  cout << digit << endl;
  res.dump(cout);

  cout << "Try divide" << endl;
  i3 = 4;
  i4 = 16;
  BI rem, divver;
  BI::DivMod(i3, i4, rem, divver);
  rem.dump(cout);
  divver.dump(cout);

  i3 = 16;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  rem.dump(cout);
  divver.dump(cout);

  i3 = 17;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  rem.dump(cout);
  divver.dump(cout);

  i3 = 18;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  rem.dump(cout);
  divver.dump(cout);

  i3 = 19;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  rem.dump(cout);
  divver.dump(cout);


  i3 = 20;
  i4 = 4;
  rem = 0; divver = 0;
  BI::DivMod(i3, i4, rem, divver);
  rem.dump(cout);
  divver.dump(cout);

  i3 = 6652800;
  i4 = 40320;
  rem = 0; divver = 0;
  digit = BI::ChooseClosest(i3, i4, rem);
  cout << digit << endl;
  rem.dump(cout);

  i3 = 1;
  i3.singleDigitMultiply(int_u1(256));
  cout << i3 << endl;

  int_u8 trial = 2;
  for (int_u8 ii=0; ii<65535; ii+=1) {
    i3 = trial;
    int_u8 res = i3.as(); // get back out
    if (res!=i3) {
      cerr << "can't get out?" << ii << " " << i3 << " " << res << endl;
    }
    trial <<= 1;
    trial += 253;
  }
  i3 = ~int_u8(0);
  i3.dump(cout);
  cout << i3 << endl;
  i3 *= BI(255);
  i3.dump(cout);
  cout << i3 << endl;
  int_u8 outt = i3.as();
  cout << outt << endl;
  i4 = outt;
  i4.dump(cout);
  cout << i4 << endl;

  //for (int_u8 jj = 0; jj<4294967296ULL; jj+=256) {
  for (int_u8 jj = 0; jj<6555360; jj+=256) {
    i3 = jj;
    i4 = jj;
    for (int ii=0; ii<255; ii++) {
      BI n(ii);
      i3 += n;
      i4.singleDigitAdd(ii);
      //cout << i3 << " " << i4 << endl;
      if (i3!=i4) {
	cerr << "Uh oh! Add" << endl;
	exit(1);
      }
    }
  }

  //for (int_u8 jj = 0; jj<4294967296ULL; jj+=256) {
  for (int_8 jj = 0; jj<65538; jj+=1) {
  //  for (int_8 jj = 0; jj<1; jj+=1) {
    i3 = jj;
    i4 = jj;
    for (int ii=0; ii<255; ii++) {
      BI n(ii);
      i3 -= n;
      i4.singleDigitSub(ii);
      //cout << i3 << " " << i4 << endl;
      int_u8 rr = jj-ii;
      //if (!((i3==rr) && (i3==i4))) {
      if (i3!=i4) {
	cerr << i3.length() << " " << i4.length() << endl;
	i3.dump(cerr); i4.dump(cerr);
	cout << rr << " " << ii << " " << jj << " " << i3 << " " << i4 << endl;
	cerr << "Uh oh! Sub" << endl;
	exit(1);
      }
    }
  }

  for (int ii=0; ii<256; ii++) {
    //    cerr << "ii=" << ii << endl;
    for (int jj=0; jj<256; jj++) {
      //cerr << jj << endl;
      i3 = ii;
      i3.singleDigitMultiply(jj);
      i4 = BI(ii) * BI(jj);
      if (i3!=i4) { 
	cerr << "Uh oh" << endl;
	cerr << ii << " " << jj << endl;
	cerr << i3 << " " << i4 << endl;
	exit(1);
      }
    }
  }
  
  cout << "trying print" << endl;
  i3 = 2550;
  i3.print(cout) << endl;

  cout << "trying print" << endl;
  i3 = 2551;
  i3.print(cout) << endl;

  cout << "trying print" << endl;
  i3 = 65536;
  i3.print(cout) << endl;

  cout << "trying /" << endl;
  i3 = 7* 6 * 5 * 4 * 3 * 2 * 1;
  i3.print(cout) << endl;
  i4 = 6 * 5 * 4 * 3 * 2 * 1;
  i4.print(cout) << endl;
  BI jj = i3/i4;
  jj.dump(cout) << endl;
  jj.print(cout) << endl;
  cout << "..." << endl;

  cout << "trying /" << endl;
  i3 = 11*10*9*8*7* 6 * 5 * 4;
  i3.print(cout) << endl;
  i3.dump(cout) << endl;
  i4 = 8*7*6 * 5 * 4 * 3 * 2 * 1;
  i4.print(cout) << endl;
  i4.dump(cout) << endl;
  jj = i3/i4;
  jj.dump(cout) << endl;
  jj.print(cout) << endl;
  cout << "..." << endl;

  cout << "Testing stringize" << endl;
  for (int ii=0; ii<65538; ii++) {
    i3 = ii;
    string s = i3.stringize();
    cout << s << endl;
  }


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
	b.dump(cerr);
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
    
#define XOMETHING 0
    if (XOMETHING) {
      cout << "Massive Choose closest testing" << endl;
      BI tff(255);
      for (int ii=65538; ii>=1; ii--) {
	cout << ii << endl;
	for (int jj=1; jj<ii; jj++) {
	  //cout << ii << " " << jj << endl;
	  BI n = ii;
	  BI m = jj;
	  BI res;
	  if (tff*m < n) continue;
	  try {
	    int_u1 digit = BI::ChooseClosest(n, m, res);
	    
	    BI dig = digit;
	    if (!((dig*m <= n) && (n < ((dig+BI(1))*m)))) {
	      cerr << "Uh oh!" << endl;
	      cout << "n=" << n << " m=" << m << " digit=" << int(digit) << endl;
	      cout << "lower=" << dig*m << " upper=" << (dig+BI(1))*m << endl;
	      cout << bool((dig*m)<=n) << " " << bool( (n<((dig+BI(1))*m)) ) << endl;
	      cout << bool(res<=n) << endl;
	      n.dump(cerr);
	      m.dump(cerr);
	      cerr << int(digit) << endl;
	      res.dump(cerr);
	      exit(1);
	    }
	  } catch (const exception& e) {
	    // skipping those that don't fit
	  }
	}
      }
    }
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
	nkb.dump(cerr) << endl;
	exit(1); 
      }

    }
  }

  cout << "Converting between real_8 and BigUint" << endl;
  real_8 orig = 1;
  BI rr = 1;
  real_8 bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;

  orig = 255.0;
  rr = 255;
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;

  orig = 256.0;
  rr = 256;
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;

  orig = 274365;
  rr = 274365;
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;

  orig = 27436538568045ULL;
  rr = 27436538568045ULL;
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;

  orig = 281474976710656ULL; // 2**48
  rr = 281474976710656ULL; // 2**48
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;

  cout.precision(16);
  orig = 281474976710656666ULL; // (2**48) * 1000 + 666 
  rr =   281474976710656666ULL; // 2**48 * 1000 + 666
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;


  orig = 281474976710656e+200; // (2**48) e+300
  rr = 281474976710656ULL; // 2**48 
  for (int ii=0; ii<200; ii++) {
    rr *= 10;
  }
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;

  orig = 281474976710656e+300; // (2**48) e+300
  rr = 281474976710656ULL; // 2**48 
  for (int ii=0; ii<300; ii++) {
    rr *= 10;
  }
  bb = MakeRealFromBigUInt(rr);
  cout << "real_8:" << bb << " BI:" << rr << " ==?" << (orig==bb) << endl;


  cout << "Converting from real_8 to BigUInt" << endl;

  BI ggg;
  BI orig_int = 1;
  bb = 1.0;
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;


  orig_int = 0;
  bb = 0.0;
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  // Impl defined: but usually just 0, or all 1s:?
  //real_8 what_happens = -1;
  //int_u8 this_happens = int_u8(what_happens);
  //cout << "This happens when converting negative real_8 (" << what_happens << ") to int_u8:" << int_4(this_happens) << endl;

  // negatives become 
  //orig_int = -1;
  //bb = -1.0;
  //MakeBigUIntFromReal(bb, ggg);
  //cout << " BI:" << int_4(ggg) << " real_8:" << bb << " ==?" << (orig_int==int_4(ggg)) << endl;

  orig_int = 255;
  bb = 255.0;
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = 256;
  bb = 256.0;
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = 65536;
  bb = 65536.0;
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;


  orig_int = 65535;
  bb = 65535.0;
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  orig_int = 281474976710656ULL; // 2**48
  bb = 281474976710656ULL; // 2**48
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;

  cout << ".. expect failures now ... real_8 can only hold about 48 bits of precision" << endl;

  orig_int = 281474976710656666ULL; // 2**48 * 1000 + 666
  bb = 281474976710656666ULL; // 2**48
  MakeBigUIntFromReal(bb, ggg);
  cout << " BI:" << ggg << " real_8:" << bb << " ==?" << (orig_int==ggg) << endl;


  cout << "Trying to make BigUInt from binary stream" << endl;
  {
    int_un i;
    MakeBigUIntFromBinary("\x00\x01", 2, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x00", 1, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x01", 1, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x15_\xd0\xacK\x9b\xb6\x01", 8, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x15_\x04|\x9f\xb1\xe3\xf2\xfd\x1e\x66", 11, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x01\x01", 2, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\xff", 1, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x00\x80", 2, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x00\x80\x00", 3, i);
    cout << i << endl;

    MakeBigUIntFromBinary("\x00\x00\x01", 3, i);
    cout << i << endl;
  }

  cout << "** Trying to make a binary stream from BigUINT" << endl;
  {
    testBin(BI(0));
    testBin(BI(1));
    testBin(BI(127));
    testBin(BI(128));
    testBin(BI(255));
    testBin(BI(256));
    testBin(BI(32767));
    testBin(BI(32768));
    testBin(BI(65535));
    testBin(BI(65536));
    testBin(BI(1000000000));

    testBin(BigUInt<int_u2, int_u4>(0));
    testBin(BigUInt<int_u2, int_u4>(1));
    testBin(BigUInt<int_u2, int_u4>(127));
    testBin(BigUInt<int_u2, int_u4>(128));
    testBin(BigUInt<int_u2, int_u4>(255));
    testBin(BigUInt<int_u2, int_u4>(256));
    testBin(BigUInt<int_u2, int_u4>(32767));
    testBin(BigUInt<int_u2, int_u4>(32768));
    testBin(BigUInt<int_u2, int_u4>(65535));
    testBin(BigUInt<int_u2, int_u4>(65536));
    testBin(BigUInt<int_u2, int_u4>(1000000000));
  }

  int_un mxv = StringToBigUInt("1000000000000000000000000000000000000000");

  // Test and make sure that D5 step of the Knuth algorithm 
  // gives ok correction  
  {
    typedef BigUInt<int_u4, int_u8> BI8;

    BI8 u = StringToBigIntHelper<BI8>("190797007524439073807468042969529173669356994749940177394741882673528979787005053706368049835514900244303495954950709725762186311224148828811920216904542206960744666169364221195289538436845390250168663932838805192055137154390912666527533007309292687539092257043362517857366624699975402375462954490293259233303137330643531556539739921926201438606439020075174723029056838272505051571967594608350063404495977660656269020823960825567012344189908927956646011998057988548630107637380993519826582389781888135705408653045219655801758081251164080554609057468028203308718724654081055323215860189611391296030471108443146745671967766308925858547271507311563765171008318248647110097614890313562856541784154881743146033909602737947385055355960331855614540900081456378659068370317267696980001187750995491090350108417050917991562167972281070161305972518044872048331306383715094854938415738549894606070722584737978176686422134354526989443028353644037187375385397838259511833166416134323695660367676897722287918773420968982326089026150031515424165462111337527431154890666327374921446276833564519776797633875503548665093914556482031482248883127023777039667707976559857333357013727342079099064400455741830654320379350833236245819348824064783585692924881021978332974949906122664421376034687815350484991");
    BI8 v = StringToBigIntHelper<BI8>("18446744092529142403");
    //BI8 v = StringToBigInt("18446744092529142401");
    
    BI8 q1,r1,q2,r2;
    BI8::DivMod(u,v,q1,r1); 
    BI8::DivMod2(u,v,q2,r2); 
    if (q1!=q2 ||(r1!=r2)) {
      // Knuth gives different answer!
      if (q1!=q2) { cerr << "DivMods q don't match:" << endl << q1 << endl << q2 << endl; }
      if (r1!=r2) { cerr << "DivMods r don't match:" << endl << r1 << endl << r2 << endl; }
      cerr << (q2 * v + r2 == u) << endl;
      cerr << (q1 * v + r1 == u) << endl;

      u.dump(cout); cout << endl;
      q1.dump(cout); cout << endl;
      q2.dump(cout); cout << endl;
      q1.dump(cout); cout << endl;
      q2.dump(cout); cout << endl;
      exit(1);
    }
    
  }


  for (int ooo=-256; ooo<256; ooo+=127) 
  {
    cout << "----------------------ooo" << ooo << "Different types" << endl;
    int_un x0 = "1234567890123456789012345678901234567890"; cout << x0 << endl;
    int_un x1 = int_1(ooo); cout << x1 << endl;
    int_un x2 = int_u1(ooo); cout << x2 << endl;
    int_un x3 = int_2(ooo); cout << x3 << endl;
    int_un x4 = int_u2(ooo); cout << x4 << endl;
    int_un x5 = int_4(ooo); cout << x5 << endl;
    int_un x6 = int_u4(ooo); cout << x6 << endl;
    int_un x7 = int_8(ooo); cout << x7 << endl;
    int_un x8 = int_u8(ooo); cout << x8 << endl;
    int_un x9 = real_8(ooo); cout << x9 << endl;
    int_un xx = real_4(ooo); cout << xx << endl;
    int_un xy = int(ooo); cout << xy << endl;
    int_un xz = long(ooo); cout << xz << endl;
    // int_un xa = size_t(ooo); cout << xa << endl; // 32 vs. 64, 

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
    // xa = size_t(ooo); cout << xa << endl;

    {
      int_un ii = int_un("100000000000000000000000000000000000000000000") + ooo;
      cout << ii << endl;
      int_un jj = ooo + int_un("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_un ii = int_un("100000000000000000000000000000000000000000000") * ooo;
      cout << ii << endl;
      int_un jj = ooo * int_un("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_un ii = int_un("100000000000000000000000000000000000000000000") - ooo;
      cout << ii << endl;
      int_un jj = ooo - int_un("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_un ii = int_un("100000000000000000000000000000000000000000000") / ooo;
      cout << ii << endl;
      int_un jj = ooo / int_un("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
    {
      int_un ii = int_un("100000000000000000000000000000000000000000000") % int_u4(ooo);
      cout << ii << endl;
      int_un jj = ooo % int_un("100000000000000000000000000000000000000000000");
      cout << jj << endl;
    }
  }

  {
    for (int_2 i=0; i<3; i++) {
      for (int_2 j=0; j<3; j++) {
	int_un ii=i;
	int_un jj=j;
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
#if defined(OC_BIGINT_OUTCONVERT_AS)
#  define AS(x) ((x).as()) 
#else 
#  define AS(x) ((x).operator int_8())
#endif

  int_n ii = "10000";
  int_8 n = AS(ii);
  cout << n << endl;

  cout << "Okay!" << endl;
}
