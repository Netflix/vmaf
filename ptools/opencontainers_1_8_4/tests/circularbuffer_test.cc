
// Test the CircularBuffer

// ///////////////////////////////////////////// Include Files

#include "occircularbuffer.h"
#include "ocstring.h"
#include "ocport.h"
#include "ocpermutations.h"

#include <stdio.h>

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// ///////////////////////////////////////////// The CircularBufferTest Class

class CircularBufferTest {
    
  public:

    // ///// Methods

    int tests ();

}; // CircularBufferTest

template <class T>
void statCB (CircularBuffer<T>& a)
{
  cout << "empty:" << a.empty() << " full:" << a.full() << " length:" << a.length() << " capacity:" << a.capacity() << endl;
  cout << a << endl;
}


template <class T>
void testT (const T& a1, const T& a2, const T& /*a3*/, 
	    Array<T>& l, const T& mid, bool infinite)
{
  CircularBuffer<T> a(1, infinite);
  statCB(a);
  
  a.put(a1);
  statCB(a);
  
  bool saw = false;
  try {
    a.put(a2);
  } catch (const runtime_error& e) {
    saw = true;
    cout << "Expected: " << e.what() << endl;
  }
  if (!infinite && !saw) cout << "ERROR! Expected an error" << endl;
  statCB(a);

  cout << a.get();
  statCB(a);

  saw = false;
  try {
    //T c = a.get();
    (void)a.get();
  } catch (const runtime_error& e) {
    saw = true;
    cout << "Expected: " << e.what() << endl;
  }
  if (!infinite && !saw) cout << "ERROR! Expected an error" << endl;
  statCB(a);

  PermutationsT<T> p(l);
  while (p.next()) {

    CircularBuffer<T> b(3);    
    Array<T>& current = p.currentPermutation();
    for (int ii=0; ii<int(l.length()); ii++) {
      //cout << current[ii];
    } 
    for (int ii=0; ii<int(l.length()); ii++) {
      try {
	if (current[ii]<=mid) {
	  b.put(current[ii]);
	  //cout << "p" << current[ii];
	} else {
	  //T temp = b.get();
	  (void)b.get();
	  //cout << "g" << temp;
	}
      } catch (const runtime_error& e) {
	//cout << "e";
      }
    }
    //cout << endl;
  }
 

}


void finiteTest ()
{
  Array<int> a; 
  for (int ii=0; ii<8; ii++) a.append(ii);
  testT(100,200,300, 
	a, 4, false);

  Array<string> aa; 
  for (int ii=0; ii<8; ii++) aa.append(Stringize(ii));
  testT<string>(string("the"),string("quick"),string("brown"), 
		aa, Stringize(4), false);
}


void infiniteTest ()
{
  CircularBuffer<int> cb(2, true);
  statCB(cb);
  cb.put(100);
  statCB(cb);
  cb.put(200);
  statCB(cb);
  cb.put(300); // should cause expand
  statCB(cb);

  cout << "g" << cb.get() << endl;
  statCB(cb);
  cb.put(400);
  statCB(cb);
  cb.put(500);
  statCB(cb);
  cb.put(600);
  statCB(cb);

  CircularBuffer<string> cbs(2, true);
  statCB(cbs);
  cbs.put("a100");
  statCB(cbs);
  cbs.put("a200");
  statCB(cbs);
  cbs.put("a300"); // should cause expand
  statCB(cbs);

  cout << "g" << cbs.get() << endl;
  statCB(cbs);
  cbs.put("a400");
  statCB(cbs);
  cbs.put("a500");
  statCB(cbs);
  cbs.put("a600");
  statCB(cbs);


  Array<int> a; 
  for (int ii=0; ii<8; ii++) a.append(ii);
  testT(100,200,300, 
	a, 4, true);
  
  Array<string> aa; 
  for (int ii=0; ii<8; ii++) aa.append(Stringize(ii));
  testT<string>(string("the"),string("quick"),string("brown"), 
		aa, Stringize(4), true);
}


void peekConsumeTest ()
{
  {
    CircularBuffer<string> c(5);
    c.put("one");
    c.put("two");
    c.put("three");
    c.put("four");
    for (int ii=0; ii<5; ii++) {
      try {
	string xx = c.peek(ii);
	cout << ii << ":" << xx << endl;
      } catch (runtime_error& e) {
	cout << e.what() << endl;
      }
    }
  }

  {
    for (int ii=0; ii<6; ii++) {
      CircularBuffer<string> c(5);
      c.put("one");
      c.put("two");
      c.put("three");
      c.put("four");
      
      cout << "Consuming:" << ii << endl;
      try {
	c.consume(ii);
	
	for (int ii=0; ii<c.length(); ii++) {
	  string xx = c.peek(ii);
	  cout << xx << " ";
	}
	cout << endl;
      } catch (runtime_error& e) {
	cout << e.what() << endl;
      }
      cout << "... done consuming" << ii << endl;
    }
  }
}

void pushbackTest ()
{
  cout << "**** Putback tests" << endl;
  {
    CircularBuffer<int> c(5);
    c.pushback(100);
    cout << c.peek(0) << endl;
    try {
      c.peek(1);
    } catch (const exception& e) {
      cout << e.what() << endl;  // expected
    }
  }
  {
    CircularBuffer<int> c(2);
    c.pushback(100);
    cout << c.peek(0) << endl;
    try {
      c.peek(1);
    } catch (const exception& e) {
      cout << e.what() << endl;  // expected
    }
  }

  {
    CircularBuffer<int> c(2);
    c.pushback(100);
    c.pushback(200);
    cout << c.peek(0) << endl;
    cout << c.peek(1) << endl;

    try {
      c.peek(2);
    } catch (const exception& e) {
      cout << e.what() << endl;  // expected
    }
  }

  {
    CircularBuffer<int> c(2);
    c.pushback(100);
    c.pushback(200);
    try {
      c.pushback(300);
    } catch (const exception& e) {
      cout << e.what() << endl;  // expected
    }
  }

  {
    CircularBuffer<int> c(2);
    c.pushback(100);
    c.pushback(200);
    try {
      c.put(300);
    } catch (const exception& e) {
      cout << e.what() << endl;  // expected
    }
  }
 
  {
    CircularBuffer<string> c(5);
    c.pushback("zero");
    c.put("one");
    c.put("two");
    c.put("three");
    c.put("four");
    for (int ii=0; ii<6; ii++) {
      try {
	string xx = c.peek(ii);
	cout << ii << ":" << xx << endl;
      } catch (runtime_error& e) {
	cout << e.what() << endl;
      }
    }
  }
  {
    CircularBuffer<string> c(5);
    c.put("one");
    c.put("two");
    c.put("three");
    c.pushback("zero");
    c.put("four");
    for (int ii=0; ii<6; ii++) {
      try {
	string xx = c.peek(ii);
	cout << ii << ":" << xx << endl;
      } catch (runtime_error& e) {
	cout << e.what() << endl;
      }
    }
  }

  {
    CircularBuffer<string> c(6);
    c.put("one");
    c.put("two");
    c.put("three");
    c.pushback("zero");
    c.put("four");
    for (int ii=0; ii<6; ii++) {
      try {
	string xx = c.peek(ii);
	cout << ii << ":" << xx << endl;
      } catch (runtime_error& e) {
	cout << e.what() << endl;
      }
    }
  }

  {
    CircularBuffer<int> c(2, true);
    c.pushback(100);
    c.pushback(200);
    c.pushback(300);
    for (int ii=0; ii<3; ii++) {
      cout << c.peek(ii) << endl;
    }
    c.pushback(400);
    for (int ii=0; ii<4; ii++) {
      cout << c.peek(ii) << endl;
    }
    c.pushback(500);
    for (int ii=0; ii<5; ii++) {
      cout << c.peek(ii) << endl;
    }
  }

  {
    CircularBuffer<int> c(2, true);
    try {
      c.drop();
    } catch (const runtime_error& e) {
      cout << "Expected empty exception:" << e.what() << endl;
    }
    c.put(100);
    cout << "len:" << c.length() << " with " << c.peek() << endl;
    cout << "... dropped " << c.drop() << endl;
    cout << "len:" << c.length() << endl;
    c.put(200);
    c.put(300);
    cout << "len:" << c.length() << " with " << c.peek() << endl;
    cout << "... dropped " << c.drop() << endl;
    cout << "len:" << c.length() << " with " << c.peek() << endl;
    cout << "... dropped " << c.drop() << endl;
    try {
      c.drop();
    } catch (const runtime_error& e) {
      cout << "Expected empty exception:" << e.what() << endl;
    }
  }

  {
    CircularBuffer<int> c(3, true);
    try {
      c.drop();
    } catch (const runtime_error& e) {
      cout << "Expected empty exception:" << e.what() << endl;
    }
    c.put(100);
    cout << "len:" << c.length() << " with " << c.peek() << endl;
    cout << "... dropped " << c.drop() << endl;
    cout << "len:" << c.length() << endl;
    c.put(200);
    c.put(300);
    cout << "len:" << c.length() << " with " << c.peek() << endl;
    cout << "... dropped " << c.drop() << endl;
    cout << "len:" << c.length() << " with " << c.peek() << endl;
    cout << "... dropped " << c.drop() << endl;
    try {
      c.drop();
    } catch (const runtime_error& e) {
      cout << "Expected empty exception:" << e.what() << endl;
    }
  }

}



// ///////////////////////////////////////////// CircularBufferTest Methods

int CircularBufferTest::tests()
{
  finiteTest();
  infiniteTest();
  peekConsumeTest();
  pushbackTest();

  return 0;
}


int main ()
{
  CircularBufferTest t;
  return t.tests();
}



