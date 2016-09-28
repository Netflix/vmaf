
#include "ocproxy.h"
#include "ocval.h"
#include "ocstreamingpool.h"
#include "ocpermutations.h"

#include "octhread.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// Tests for Proxies in three forms:  Plain reference counted,
// Locked (for threads) reference counted, Locked (for threads)
// but forced into "a shared memory pool"


// We want the interface to look like: Two kinds of ways.  Ways when
// you need the lock, and if you are smart enough, so you don't.
//
// This will give you a borrowed reference to the table.
// This allows you to change the table:  THIS ASSUMES YOU KNOW
// WHAT YOU ARE DOING AND DON'T HAVE TO WORRY ABOUT SYNC.
//
//   Tab& borrowed_reference = proxy;
//
//   or
//   Array<real>& data = proxy;
//
//
// A read-only copy.  This assumes you know the table won't change
// while you have a reference to it
//   const Tab& borrowed_ref = proxy;
//
//
// Inside the transation lock, you can do anything
// to the table, and it will be atomic
//  {
//    TransactionLock tl(proxy);
//
//    // You can change to you heart's content, and not mess up
//    // any others inside of a TransactionLock
//    Tab& borrowed_reference = proxy;
//
//    // You can look at and know that no one else inside a
//    // TransationLock will change it
//    const Tab& borrwed = proxy;
//  }
//
//    or
//  LockedTab tl = proxy;
//
// creation of a Val inside of shared memory
//  Val v = Shared(shm, Tab("{'a':1, 'b':2}"));
//  v["c"] = 17;   // FORCES "c" and 17 into same shared memory section



// Helper function to make sure a particular value is in shared memory
bool InSharedMemory (Allocator* a, const Val& v)
{
  char* start = (char*)a;
  char* end   = start + a->bytes();

  if (v.a != a) {
    cout << "Not using the same allocator" << endl;
    return false;
  }
  if (v.tag == 'n') {
    Arr* ar = (Arr*)&v.u.n;
    if (ar->allocator() != a) {
      cout << "Array not using right allocator" << endl;
      return false;
    }
  } else if (v.tag=='t') {
    Tab* ar = (Tab*)&v.u.n;
    if (ar->allocator() != a) {
      cout << "Tab not using right allocator" << endl;
      return false;
    }
  } else if (v.tag=='a') {
    OCString* os = (OCString*)&v.u.a;
    if (os->length()>32) {
      if (os->allocator() != a) {
	cout << "Tab not using right allocator" << endl;
	return false;
      } 
    }
    if ( ((char*)(&v)) < start || ((char*)(&v)) >=end) {
      cout << "for string data, which fits in buffer, not in the right range of mem" << endl;
      return false;
    }
  } 

  // All other checks have passed, just make sure Val is question in shared
  if ( ((char*)(&v)) < start || ((char*)(&v)) >=end) {
    cout << "for any 'POD' types, not in the right range of mem" << endl;
    return false;
  }
  return true;
}


// Thread routine
void* t1(void* data)
{
  try {
    Val* vp = (Val*)data;
    Tab& t = *vp;
    
    // Figure out which thread we are: 0 or 1
    int thread_number = 0;
    if (t.contains(1)) {
      thread_number = 1;
    }
    //cout << "Thread number " << thread_number << endl;
    // 0 thread writes evens, 1 thread writes odds
    for (int ii=thread_number; ii<1000000; ii+=2) {
      {
	Val o = t(thread_number);
	TransactionLock tl(o);
	o[ii] = thread_number;
      }
    }
  } catch (exception& e) {
    cerr << "ERROR! " << e.what() << endl;
  }
  return 0;
}

// Create two threads and have them bang into the same
// Tab:  no segfaults!!
void ThreadsTest ()
{
  // All threads will be appending to this Table
  Val shared_table = Locked(new Tab("{ }"));
  
  Val thread0 = Tab("{0:None}");
  thread0[0] = shared_table;
  Val thread1 = Tab("{1:None}");
  thread1[1] = shared_table;
  {
    cout << "Thread number " << 0 << endl;
    cout << "Thread number " << 1 << endl;
    OCThread a("thread a", false);
    OCThread b("thread b", false);
    
    a.start(t1, &thread0);
    b.start(t1, &thread1);
    
    // When threads "destruct" they have to join
  }

  // Check to make sure both tables inserted okay
  bool okay = true;
  for (int ii=0; ii<100000; ii++) {
    Val check = shared_table[ii];
    if (int_4(check) != (ii%2)) {
      cerr << "Table was invalid at " << check << endl;
      okay = false;
    } 
  }
  if (okay) cout << "Okay:  Threads test passed" << endl;
}

// Change by reference
void Change (Val& v)
{
  // Can explicitly get out the tab
  Tab& borrowed_reference = v;
  borrowed_reference["yup"] = 17;

  // Or can change directly KNOWING it's a table
  v["again"] = 666;
}


// This test should be run with valgrind for more
// confidence

int main ()
{
  // Try plain proxies with no locks:  this assumes only no concurrency
  // on the table as only one thread will ever look at it
  Proxy p = new Tab("{'a':1, 'b':2}");
  cout << "Proxy looks like: " << p << endl;

  // Test copy construction and destruction: Make sure basically works
  {
    Proxy copy_p(p); // copy
    cout << "Proxy copy looks like: " << copy_p << endl;
  } // destruct

  // A Borrowed reference is a "raw" reference that isn't reference counted
  // and isn't locked... it's only valid as long as the proxy in question
  // is valid
  Tab& borrow = p;  // okay
  cout << "borrowed reference of p looks like: " << borrow << endl;
  
  // Can't get out an Arr when looking for Tab!
  try {
    Arr& a = p;
    if (a.length()==888888) exit(1);  // Dumb test
  } catch (const exception& e) {
    cout << "Expected Error: " << e.what() << endl;
  }

  // Test adopting val constructor
  Val vv(new Tab("{'should' : 'allow carefully' }"));   // Should adopt
  Val vvv = new Tab("{'should' : 'allow carefully' }"); // Should adopt
  cout << "Adopting Val (i.e., Proxy) looks like: " << vv 
       << " " << vvv << endl;
  
  // Test operator=
  vv = new Tab("{'eh?': True}");   // Should adopt
  vv[0] = 100;   // Now can use
  cout << "After operator= and op[] test:" << vv << endl;

  // Make sure can change
  Tab& borrowed_reference = p;  
  borrowed_reference["hello"] = 100;
  Val v = p;
  Change(v);
  
  // borrowed_reference["myself"] = p;  // CIRCULAR!  prints forever: like early versions of Python
  cout << p << endl;

  Tab new_table("{'stuff':1}");
  new_table["proxy"] = p;
  cout << "Proxy: make sure you can insert into table:" << new_table << endl;
  
  // A TransactionLock doesn't do anything if the item is just "plain
  // ref counted": You need a locked reference count for that
  {
    Val oo = new Tab();
    TransactionLock ool(oo);  // NO OP because no Lock specified
    oo["set something"] = None;
    cout << "Transaction Lock with plain ref count:" <<oo << endl;
  }
  
  // Can only TransactionLock a Proxy
  try { // SHOULD FAIL!!
    Val oo = Tab();
    TransactionLock ool(oo);  // Doesn't do anything because no Lock!!
  } catch (const logic_error& e) {
    cout << "Expected Failure--Can only lock on a Proxy: " << e.what() << endl;
  }

  // Adopt a table, with a Lock to protect against multiple threads
  Val dataobj = Locked(new Tab("{'HEADER': { }, 'DATA': array([1,2,3]) }"));
  cout << "BEFORE MANIPULATION:" << dataobj << endl;
  {
    TransactionLock tl(dataobj);

    // Manipulate Header
    Tab& hdr = dataobj("HEADER");
    hdr["AXIS"] = "Time";
    hdr["NAME"] = "Blah";

    // Manipulate Data
    Array<int_4>& a = dataobj("DATA");
    a.expandTo(10);
    int_4* data = a.data();
    for (int ii=0; ii<10; ii++) {
      data[ii] = ii;
    }

    // If you are going to print out the table, do it inside
    // of a lock so you know it's not changing!
    cout << "AFTER MANIPULATION:" << dataobj << endl;
  }

  // Test a lot of threads banging against the same table
  ThreadsTest();

  // Can we put Arrs in Proxies
  cout << "Test creating Arrays with proxies" << endl;
  Proxy pa = new Arr("[1,2,3]");
  cout << pa << endl;
  //  pa[0] = 17;  // ... NOT ALLOWED ...  Too slow for array access
    
  Arr& aaa = pa;
  aaa[0] = 17;
  cout << "Proxies with Arrs: can we update:" << pa << endl;

  // Sharing Arrs
  Val g = new Array<real_8>();
  Tab t; t["data"] = g;
  Array<real_8>& ga = g;
  ga.expandTo(10);
  for (int ii=0; ii<10; ii++)
    ga[ii] = ii;
  cout << t << endl;
  cout << g << endl;
  
  // Nope, can only get out Array<real_8> for this example
  try {
    Arr& a = t("data");
    if (a.length()==777777) exit(1); // Dumb test
  } catch (const logic_error& er) {
    cout << "Expecting Exception because wrong type:" << er.what() << endl;
  }

  // Make sure we can lock down an Arr
  Val vl = Locked(new Arr("[1,2,3]"));
  {
    TransactionLock tl(vl);
    Arr& a = vl;
    a.append("hello");
    cout << a << endl;
    vl[0] = 17;  // This is okay for an Arr because it's an integer
    cout << vl(0) << endl;  // okay
    try { // out of bounds 
      cout << vl(100) << endl;  
    } catch (const out_of_range& re) {
      cout << "Expected:" << re.what() << endl;
    }
    try {
      vl["hello"] = 19;
    } catch (const logic_error& re) {
      cout << "Expected:" << re.what() << endl;
    }
    cout << a << endl;
  }

  // make sure subscripts work for proxies
  Val oo = new Tab("{'a':1}");
  oo["hello"] = 17.6;
  try {
    oo("not gonna happen") = 666;
  } catch (const out_of_range& re) {
    cout << "Expected subscript error:" << re.what() << endl;
  }
  cout << oo << endl;

  // Subscripts should NOT work for Array<real_8>
  Val rr = new Array<real_8>(10);
  Array<real_8>& myarr = rr;
  myarr.append(1.1);
  try {
    rr[0] = 9;
  } catch (const logic_error& re) {
    cout << "Expected subscript error:" << re.what() << endl;
  }

  // create a shared pool
  const size_t bytes = 10000;
  char* memory = new char[bytes];
  StreamingPool* shm = StreamingPool::CreateStreamingPool(memory, bytes);

  {
    Proxy p = Shared(shm, Tab("{0:100.1}"));
    Val sv = p;
    sv[1] = Tab("{10:10}"); // should FORCE copy into Shared memory
    cout << "Tab in shared memory:" << sv << endl;
  }


  // Can we put Arrs in shared memory?
  {
    // Put an Arr in shared memory
    Proxy sj = Shared(shm, Arr("[8,9]"));
    Arr& ao = sj;
    cout << ao << endl;
    ao.append(100);
    cout << ao << endl;

    // because the Arr is in shared memory, we should be 
    // able to fill one entry with ANOTHER shared memory thingee
    ao[0] = Array<real_8>(10);   // COPY SHOULD BE FORCED INTO SHARED MEMORY
    // should it be locked?

    ao[1] = Shared(shm, Arr("[666,777]"));  // Proxy to pool in same memory

    // We WANT some kind of error for this!! It's probably a real bad idea
    // to have the shared memory struct itself contain a local..
    // Or do we want to allow this for "smart users?"
    ao[2] = new Tab("{888:555.5}");

    cout << ao << endl;
  }


  // Can we have proxies for ints?     No.  TODO: ?
  int_1* ipp = new int_1(5);
  try {
    Val ps = ipp;
  } catch (const logic_error& le) {
    cout << "NO PROXIES FOR PLAIN DATA! Expected:" << le.what() << endl;
  }
  delete ipp;

  // Can we have proxies for Strings?  Yes: BUT ONLY OCSTRINGS so they
  // can go into shared memory!!!
  string* sp = new string("hello");
  try {
    Val ps = sp;  
  } catch (const logic_error& le) {
    cout << "NO PROXIES FOR STRINGS!: Strings are immutable (like in Python)" << le.what() << endl;
  }
  delete sp;
  
  // normal strings (like STL strings) can't be used easily with the
  // Allocator and Shared Memory
  OCString* ocp = new OCString("hello");
  try {
    Val pps = ocp;
  } catch (const exception& e) {
    cout << "NO PROXIES FOR OCSTRINGS either!: Strings are immutable (like in Python)" << e.what() << endl;
  }
  delete ocp;

  //try {
  //  // Won't compile!  Good!
  //  // Proxy ps = new int_1(5);  
  //} catch (const logic_error& le) {
  //  cout << "Expected:" << le.what();
  //}

  // Can we detect NON shared?
  {
    Tab non = "{ 'a': 123, 'b': 12 }";
    if (InSharedMemory(shm, non)) throw logic_error("?? non Not in shared memory"+Stringize(non));

    It jj(non);
    while (jj()) {
      const Val& key = jj.key();
      Val& value = jj.value();
      if (InSharedMemory(shm, key)) throw logic_error("?? Key Not in shared memory"+string(key));
      if (InSharedMemory(shm, value)) throw logic_error("?? Value Not in shared memory"+string(value));
    }
    
  }

  // Make sure we can put strings in shm
  {
    Val gg = Shared(shm, Tab());
    gg["hello"] = "in shared memory";   // Forced into shared memory
    gg["array"] = Arr("[1,2,3]");       // Forced into shared memory
    gg["long"] = "123456789 123456789 123456789 123456789"; // more than 40
    string ss = gg("long");
    gg[ss] = ss+ ss;

    // Check to make sure everything in shared memory
    It ii(gg);
    while (ii()) {
      const Val& key = ii.key();
      if (!InSharedMemory(shm, key)) throw logic_error("key expected in shm");
      Val& value = ii.value();
      if (!InSharedMemory(shm, value)) throw logic_error("value expected in shm");
    }
  }

  StreamingPool::Clean(shm);
  delete [] memory;

  // make sure length works!
  {
    const char* tr[] = { "None", "1", "1.0", // Exceptions, not containers
		   "'hello'", "''", "'a'", // strings
		   "{ }", "{'a':1}", "{ 'a':2, 'b':3 }", // Tab
		   "[]", "[None]", "[None, None]", // Arr
		   "array([])", "array([1 2])", "array([1.0])", // Array
		   "p{}", "p{'a':1}", "p{'a':'b', 'g':2}", // Proxy Tab
		   "A[]", "A[None]", "A[None, None]", // Arr
		   "*array([])", "*array([1 2])", "*array([1])", // Array
		   // There are no proxies for string
		   0 };
    for (const char**tt = tr; *tt!=0; tt++) {
      Val v;
      string s;
      if (**tt=='p') { // Tab proxy
	s  =string(*tt+1);
	v = new Tab(s);
      } else if (**tt=='A') {
	s = string(*tt+1);
      	v = new Arr(s);
      } else if (**tt=='*') {
	s=*tt+1;
	v = Eval(s);
	Array<int_4>& a = v;
	v = new Array<int_4>(a);
      } else {
	s=*tt;
	v = Eval(s);
      }
      //cout << "v.tag" << v.tag << " v.subtype" << v.subtype << endl;

      int length = -1;
      try {
	length = v.length();
      } catch (const logic_error& re) {
	cout << "Can't take length of " << s << endl;
      }
      if (length!=-1) cout << "Length is " << length << " of " << s << endl;
    }
  }

}
