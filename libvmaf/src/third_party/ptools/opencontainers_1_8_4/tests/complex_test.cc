
#include "ocval.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

template <class T>
void testT (T*, bool signd)
{
  // Check tags
  cout << TagFor((T*)0) << endl;
  cout << TagFor((cx_t<T>*)0) << endl;
  cout << TagFor((Array<T>*)0) << endl;
  cout << TagFor((Array<cx_t<T> >*)0) << endl;

  // Can we construct and do stuff?
  cx_t<T> a (77,78);
  cout << a << endl;
  if (sizeof(T)==1) {
    cout << int_4(mag2(a)) << endl;
  } else {
    cout << mag2(a) << endl;
  }

  if (signd) {
    cx_t<T> ss(-77,-78);
    cout << ss << endl;
    if (sizeof(T)==1) {
      cout << int_4(mag2(ss)) << endl;
    } else {
      cout << mag2(ss) << endl;
    }
  }

  // Get it into a Val
  Val v = a;
  cout << v << endl;

  // get it out of a Val
  cx_t<T> res = v;
  cout << res << endl;

  // Look at an array of them
  Array<cx_t<T> > arr(3);
  for (size_t ii=0; ii<arr.capacity(); ii++) {
    arr.append(cx_t<T>(ii*10, ii*20));
  }
  cout << arr << endl;

  // Get into a Val
  Val va = arr;
  cout << va << endl;

  // And get back out
  Array<cx_t<T> >& out = va;
  cout << out << endl;

  // Make sure Proxies work
  Val p = new Array<cx_t<T> >(1);
  Array<cx_t<T> >& ax = p;
  ax.append(cx_t<T>(100,150));
  cout << ax << endl;
  cout << p << endl;
  cout << "Tag=" << p.tag << " subtype=" << p.subtype << " proxy?" << IsProxy(p) << endl;
}

int main ()
{
  testT((int_1*)0, true);
  testT((int_u1*)0,false );
  testT((int_2*)0, true);
  testT((int_u2*)0,false);
  testT((int_4*)0, true);
  testT((int_u4*)0,false);
  testT((int_8*)0, true);
  testT((int_u8*)0,false);
}
