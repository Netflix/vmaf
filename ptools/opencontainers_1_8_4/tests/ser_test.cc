
#include "ocser.h"
#include <ctype.h>
#include "ocserialize.h"
#include "occonvert.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

string hexer (int_u1 a)
{
  char s[4] = { 0,0,0,0 };
  s[0] = 'x';

  int g = (a >> 4);
  s[1] = g["0123456789ABCDEF"];

  s[2] = (a &0x0F)["0123456789ABCDEF"];

  s[3] = 0;
  return s;
}

void printer (char* mem, size_t len)
{
  cerr << "*** Length of memory = " << len << endl;
  for (size_t ii=0; ii<len; ii++) {
    char c = mem[ii];
    if (isprint(c)) {
      cerr << c;
    }
    else {
      cerr << hexer(c);
    }
    if (ii%20 == 19) cerr << endl; else cerr << " ";
  }
  cerr << endl;
}

void putme (const Val& v, bool compare=true)
{
  // cerr << "*****v = " << v << endl;
  OCSerializer o;
  o.put(v);
  
  size_t len;
  char* mem = o.peek(len);
  //printer(mem, len);

  // Compare against old way
  if (compare) {
    size_t b = BytesToSerialize(v);
    char* nm = new char[b];
    char* end = Serialize(v, nm);
    size_t new_len = end - nm;
    
    if (new_len != len) {
      cerr << "Lengths of serialization are different: new_len=" << new_len
	   << " and old len= " << len << endl;
      exit(1);
    }
    if (memcmp(mem, nm, len)!=0) {
      cerr << "old mem and new mem different" << endl;

      cerr << v << endl;
      cerr << "OCSerialized" << endl;
      printer(mem, len);
      cerr << "old way" << endl;
      printer(nm, len);
      exit(1);
    }

    // Now see if both deserialize the same way
    Val result;
    char* ender = Deserialize(result, mem);
    size_t old_len = ender-mem;
    if (old_len != new_len) {
      cerr << "When deserializing, different lengths" << endl;
      exit(1);
    }
    OCDeserializer od(nm, true);
    Val result2;
    od.load(result2);
    //cerr << "TESTING: " << result2 << endl;
    //cerr << "   more .. " << result << endl;
    if (result2 != result) {
      cerr << "When comparing deserialized results, they are different" << endl;
      exit(1);
    }
    
    if (result2 != v) {
      cerr << "Result2 not the same thing passed in!" << endl;
      exit(1);
    }
  }

  // Even if not comparing against old way, still make sure can
  // serialize and deserialize and get back same result
  else {
    OCDeserializer ocd(mem);
    Val res;
    ocd.load(res);
    if (v!=res) {
      cerr << "Res not the same thing passed in!" << endl;
      exit(1);
    }
  }


  cout << "...okay" << endl;
}

#define ARRAYSIZES 16384
#define ARRCXCOPY(T,N)  Val vv##N = Array<cx_t<T> >(); Array<cx_t<T> >& aa##N = vv##N; \
    for (size_t ii=0; ii<ARRAYSIZES; ii++) aa##N.append(cx_t<T>(ii, ii+1)); operate(vv##N, true)

void createBigTable (Tab& tt, void (*operate)(const Val&, bool))
{
  cout << "Comparing old serializing with new to make sure make results:" << endl;
  
  Val v0 = Tab();
  operate(v0, true);

  Val v1 = int_u1(128);
  operate(v1, true);

  Val v2 = int_1(-1);
  operate(v2, true);

  Val v3 = int_u2(16384);
  operate(v3, true);

  Val v4 = int_2(-32768);
  operate(v4, true);

  Val v5 = int_4(1000000);
  operate(v5, true);

  Val v6 = int_u4(100000);
  operate(v6, true);

  Val v7 = int_8(-1);
  operate(v7, true);

  Val v8 = int_u8(-1);
  operate(v8, true);

  Val v9 = real_8(123.456);
  operate(v9, true);

  Val v10 = real_4(123.456);
  operate(v10, true);

  Val v11 = complex_8(1,2);
  operate(v11, true);

  Val v12 = complex_16(1,2);
  operate(v12, true);

  Val v13 = Arr("[1,2.0, 'three']");
  operate(v13, true);

  Val vv1 = cx_t<int_1>(10,11);
  operate(vv1, true);

  Val vv2 = cx_t<int_u1>(10,11);
  operate(vv2, true);

  Val vv3 = cx_t<int_2>(10,11);
  operate(vv3, true);

  Val vv4 = cx_t<int_u2>(10,11);
  operate(vv4, true);

  Val vv5 = cx_t<int_4>(10,11);
  operate(vv5, true);

  Val vv6 = cx_t<int_u4>(10,11);
  operate(vv6, true);

  Val vv7 = cx_t<int_8>(10,11);
  operate(vv7, true);

  Val vv8 = cx_t<int_u8>(10,11);
  operate(vv8, true);

  Val v14 = Array<real_8>();
  Array<real_8>& a = v14;
  for (int ii=0; ii<ARRAYSIZES; ii++) a.append(ii);
  operate(v14, true);

  Val v15 = Array<int_u2>();
  Array<int_u2>& aa = v15;
  for (int ii=0; ii<ARRAYSIZES; ii++) aa.append(ii);
  operate(v15, true);

  Val v16 = Array<complex_16>();
  Array<complex_16>& aaa = v16;
  for (int ii=0; ii<ARRAYSIZES; ii++) aaa.append(ii);
  operate(v16, true);

  ARRCXCOPY(int_1, 20);
  ARRCXCOPY(int_u1, 21);
  ARRCXCOPY(int_2, 22);
  ARRCXCOPY(int_u2, 23);
  ARRCXCOPY(int_4, 24);
  ARRCXCOPY(int_u4, 25);
  ARRCXCOPY(int_8, 26);
  ARRCXCOPY(int_u8, 27);

  tt = Tab();
  tt.append(v0);
  tt.append(v1);
  tt.append(v2);
  tt.append(v3);
  tt.append(v4);
  tt.append(v5);
  tt.append(v6);
  tt.append(v7);
  tt.append(v8);
  tt.append(v9);
  tt.append(v10);
  tt.append(v11);
  tt.append(v12);
  tt.append(v13);
  tt.append(v14);
  tt.append(v15);
  tt.append(v16);
  
  tt.append(vv20);
  tt.append(vv21);
  tt.append(vv22);
  tt.append(vv23);
  tt.append(vv24);
  tt.append(vv25);
  tt.append(vv26);
  tt.append(vv27);

  operate(tt, true);
}

void empty (const Val&, bool)
{ }


void timingTest (int way)
{
  cout << "Timing serializing big tables with method:" << way << endl;
    
  const int iters=2000;

  // create a very big table by replicating it a few times
  Val res = Tab();
  Tab copy;
  createBigTable(copy, empty);
  for (int ii=0; ii<10; ii++) {
    res.append(copy);
  }

  // res.prettyPrint(cout);
  // cout << endl;

  bool first_time = true;
  if (way==0) {
    for (int ii=0; ii<iters; ii++) {
      size_t bytes = BytesToSerialize(res);
      char *me = new char[bytes];
      Serialize(res,me);

      Val done;
      Deserialize(done, me);

      delete [] me;

      if (first_time) {
	first_time = false;
	if (done != res) { 
	  cerr << "Should be same!" << endl; 
	  exit(1); 
	}
      }
    }
  } else {
    for (int ii=0; ii<iters; ii++) {
      // OCSerializer ocs(17041625); // Make sure allocate enough ..
      OCSerializer ocs;
      ocs.put(res);

      OCDeserializer ocd(ocs.abandon(), true);
      Val done;
      ocd.load(done);

      if (first_time) {
	first_time = false;
	if (done != res) { 
	  cerr << "Should be same!" << endl; 
	  exit(1); 
	}
      }
    }
  }
}

void serproxy() 
{
  cout << "Trying PROXY " << endl;

  Proxy p;
  Val vp = p;
  putme(vp, true);
  
  Proxy p1 = new Arr("[1,2]");
  Val tt = Tab();
  tt.append(p1);
  tt.append(p1);
  putme(tt, true);
  
  Proxy p2 = Locked(new Tab("{'a':1}"));
  Val ttt = Tab();
  ttt.append(p2);
  ttt.append(p2);
  putme(ttt, true);
  
  Proxy p3 = Locked(new Array<real_8>());
  Array<real_8>& a8 = p3;
  for (int ii=0; ii<2; ii++) a8.append(ii);
  Val t4 = Tab();
  t4.append(p3);
  t4.append(p3);
  putme(t4, true);
  
  Proxy p4 = Locked(new Array<int_u2>());
  Array<int_u2>& a2 = p4;
  for (int ii=0; ii<5; ii++) a2.append(ii);
  Val t5 = Tab();
  t5.append(p4);
  t5.append(p4);
  t5.append(p4);
  putme(t5, true);
  
  Tab con;
  con.append(p1);
  con.append(p2);
  con.append(p3);
  con.append(p4);
  con.append(p1);
  con.append(p2);
  con.append(p3);
  con.append(p4);
  putme(con, true);
  cout << "...okay" << endl;
  
  const int bytes = 5000;
  char * mem = new char[bytes];
  {
    StreamingPool* sp = StreamingPool::CreateStreamingPool(mem, bytes);
    Proxy p5 = Shared(sp, Tab("{'shared': True}"));
    Val yy = Tab();
    yy.append(p5);
    yy.append(p5);
    putme(yy, true);
    
    sp->scheduleForDeletion();
  }
  delete [] mem;
}


// When you convert a full structure, you change the Proxies too,
// so to do "true compares", we need two separate copies.
Val MakeProxyThing ()
{
  Proxy p1 = new Tup(1,2,3);
  Proxy p2 = new OTab("o{'a':1 }");
  Proxy p3 = new Arr("[1,2.2,'three']");
  Proxy p4 = new Array<int_u4>(10);
  Array<int_u4>& a = p4; 
  a.fill(666);
  Proxy p5 = new Tab("{'dddddddddd':100.1}");
  Proxy p6 = new Tab("{'dddddddddd':o{ 'a':1, 'v':2 }, 'gg':(1,2,3) }");
  Tup t2(p1, p1, p2, p2, p3, p3, p4, p4, p5, p5, p6, p6);
  Tup t(t2, t2, new OTab("o{'a':1, 'b':2}"));
  t[2]["c"] = p2;
  return t;
}

void serOTabTup (bool serialize_compat, bool deserialize_compat)
{
  const char* lookup[] = { 
			   "()",
			   "(1)",
			   "(1,2)",
			   "(1,2,3)",
			   "(1,2,3,(1,2))",
			   "(1,2,3,(1,2))",
			   
			   "o{ }",
			   "o{ 'a':1 }",
			   "o{ 'a':1, 'b':2 }",

			   "o{ 'a':1, 'b':(1,2,3), 'c':{'nn':3, 3:(None, None)} }",
			   "o{'a':2893475892734095720934759287345972345}",
			   "o{'a':2893475892734095720934759287345972345, 'g':(435782635692873465896239569284635, 1), 'k':[235024305897239085234, 2, (1)] }",
			   "some proxy",
			   NULL
  };

  for (int ii=0; lookup[ii]!=NULL; ii++) {
    Val v;
    Val conv_v;
    if (string(lookup[ii])=="some proxy") {
      v = MakeProxyThing(); 
      conv_v = MakeProxyThing();

    } else {
      v = Eval(lookup[ii]);
      conv_v = v;
      
    }
    ConvertAllOTabTupBigIntToTabArrStr(conv_v);  
    cout << "Before: (conversion=" << serialize_compat << ")" << endl;
    v.prettyPrint(cout);

    // Serialize
    size_t bytes = BytesToSerialize(v, 
				    serialize_compat);
    cout << " bytes to serialize:" << bytes << endl;
    char* buff = new char[bytes];
    char* buffend = Serialize(v, buff, 
			      serialize_compat);  // force to deal with OTab and Tup!
    size_t bytes_gone = buffend - buff;
    if (bytes_gone!=bytes) {
      cerr << "..something wrong in serialization: got " << bytes_gone << endl;
      exit(1);
    }

    // Unserialize to make sure same
    Val ret;
    char* deser_ptr = Deserialize(ret, buff, 
		 deserialize_compat);  // force to deal with OTab and Tup!
    size_t diff = deser_ptr - buff;
    //cout << ".. bytes to deserialize" << diff << endl;
    if (diff != bytes) { 
      cout << "different bytes?" << endl; exit(1); 
    }
    cout << "After: (conversion=" << deserialize_compat << ")" << endl;
    ret.prettyPrint(cout);


    // v was original, ret was what was deserialized, conv_ret was
    // converted ret, v_conv was converted Val.
    Val conv_ret = ret;
    ConvertAllOTabTupBigIntToTabArrStr(conv_ret);  
    cout << " .. converted DeSerialized"; conv_ret.prettyPrint(cout);
    cout << " .. converted Serialized"; conv_v.prettyPrint(cout);

    if (serialize_compat && deserialize_compat) {
      // Orig converted, so that's all ret will see
      if (conv_v != ret) {
	cout << "1???? conv_v != ret .. not same" << endl;
	exit(1);
      }
    } else if (!serialize_compat && deserialize_compat) {
      // Orig NO conversion, deserialized will convert
      if (conv_v != ret) {
	cout << conv_v << endl;
	cout << ret << endl;
	cout << "2????? conv_v!=ret ... not same" << endl;
	exit(1);
      }
    } else if (serialize_compat && !deserialize_compat) {
      // Conversion on front-end, no conversion on back
      if (conv_v != ret) {
	cout << "3????? conv_v!=ret ... not same" << endl;
	exit(1);
      }
    } else { // !serialize_compat && !deserialize_compat) {
      // Neither in compat mode, compare as-is.
      if (v!=ret) {

	v.prettyPrint(cout); cout << endl;
	ret.prettyPrint(cout); cout << endl;

	cout << "4????? v!=ret ... not same" << endl;
	exit(1);
      }
    }
    
    delete  [] buff;
  }
  
}


bool compareEm (Val& v)
{
  cout << "Testing Array Type:" << v("array").subtype << endl;
  size_t bytes = BytesToSerialize(v);
  cout << "Bytes: " << bytes << endl;
  char* buff = new char[bytes];
  Serialize(v, buff);

  Val result;
  Deserialize(result, buff);
  if (result != v) {
    cerr << "  ERROR! Not same!" << endl;
    return false;
  } else {
    cout << "Same." << endl;
    return true;
  }
}


template <class T>
void TestBigArrays (T*)
{
  Val t = Tab();
  Val& v = t["array"] = Array<T>();
  Array<T>& a = v;
  size_t big = size_t(1) << 30;
  a.expandTo(big);
  T* adata = a.data();
  cout << "big=" <<  big << " a.length()=" << a.length() << " a.capacity()=" << a.capacity() << endl;
  size_t mask = 0x00FFFFFF;
  for (size_t ii=0; ii<big; ii++) {
    if ( (ii & mask)==0 ) {
      cout << ".";
      cout.flush();
    }
    adata[ii] = T(ii);
  }
  cout << "Finished initializing" << endl;
  compareEm(t);
}

#define TESTBIG(T) do { T x=0; TestBigArrays(&x); }  while (0)
void bigNews (bool full)
{
  // Here's a test that's hard to run unless you have PLENTY of memory
  /*
  {
    Val v = Tab();
    Tab& t = v;
    size_t big  = size_t(1) << 32;

    for (size_t ii=0; ii<big; ii++) {
      if ( (ii & mask)==0 ) {
	cout << ".";
	cout.flush();
      }
      t[ii] = ii;
    }
    cout << "Finished initializing" << endl;
    compareEm(v);
  }
  */

  if (full) {
    TESTBIG(int_1);
    TESTBIG(int_u1);
    TESTBIG(int_2);
    TESTBIG(int_u2);
    TESTBIG(int_4);
    TESTBIG(int_u4);
    TESTBIG(int_8);
    TESTBIG(int_u8);
    TESTBIG(cx_t<int_1>);
    TESTBIG(cx_t<int_u1>);
    TESTBIG(cx_t<int_2>);
    TESTBIG(cx_t<int_u2>);
    TESTBIG(cx_t<int_4>);
    TESTBIG(cx_t<int_u4>);
    TESTBIG(cx_t<int_8>);
    TESTBIG(cx_t<int_u8>);
    TESTBIG(real_4);
    TESTBIG(real_8);
    TESTBIG(complex_8);
    TESTBIG(complex_16);
  } else {
    TESTBIG(real_4);
  }
}

int main (int argc, char**argv)
{
  int way = -10;
  if (argc==2) {
    way = atoi(argv[1]);
  }
  
  if (way==0 || way==1) {
    timingTest(way);
  } else if (way==-1) {
    Tab t;
    createBigTable(t, putme);
  } else if (way==-2) {
    serproxy();
  } else if (way==-3) {
    serOTabTup(true, true);  // Both convert
    serOTabTup(true, false); // serialize converts, deserialize doesn't (doesn't matter though)
    serOTabTup(false, true); // serialize DOES NOT convert, deserialize does
    serOTabTup(false, false); // NO conversion
  } else if (way==-4) {
    bigNews(false); 
  } else if (way==-5) {
    bigNews(true);
  }

  // Do everything
  else {
    cout << "Test everything" << endl;
    timingTest(0);
    timingTest(1);
    
    Tab t;
    createBigTable(t, putme);

    serproxy();

    serOTabTup(true, true);  // Both convert
    serOTabTup(true, false); // serialize converts, deserialize doesn't (doesn't matter though)
    serOTabTup(false, true); // serialize DOES NOT convert, deserialize does
    serOTabTup(false, false); // NO conversion

    bigNews(false);
  }
}
