
// Testing to make sure the insertion order is preserved
// under lots of insertins and deletions.

#include "ocordavlhash.h"
#include "ocavlhash.h"
#include "ocrandomizer.h"
#include <math.h>

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

template <class K, class V>
inline void IterPrint (OrdAVLHashT<K, V,8> o1)
{
  for (OrdAVLHashTIterator<K,V,8> ii(o1); ii(); ) {
    cout << ii.key() << ":" << ii.value() << endl;
  }
}

// Go through and try to randomly insert and remove and make sure we
// preserve the insertion and removals correctly.

void CheckInsertionOrder (const int LEN)
{
  // Keep track of the elements you remove
  Randomizer r(LEN);
  Array<int> a(LEN);
  for (int ii=0; ii<LEN; ii++) {
    a.append(r.next());
  }

  //cerr << "Try JUST removes out of a table and make sure a large table deletes well." << endl;
  {  
    // All in order
    OrdAVLHashT<int, int, 8> o;
    for (int ii=0; ii<LEN; ii++) {
      o[ii] = ii;
    }
    
    // Now, remove a random one and check
    OrdAVLHashT<int, int, 8> other;
    for (int ii=0; ii<LEN; ii++) {
      other[a[ii]] = a[ii];
      if (o.remove(a[ii])==false) { cerr << "Eh???" << endl; exit(1); }
      Array<int> f(LEN-ii);
      int uu = 0;
      for (OrdAVLHashTIterator<int,int,8> ll(o); ll(); ) {
	f.append(ll.key());
	uu++;
      }
      // First, check that in order
      for (int jj=1; jj<int(f.length()); jj++) {
	if (f[jj-1]>=f[jj]) { 
	  cerr << ".. out of order.." << endl;
	  exit(1);
	}
      }
    }
    
    int yy = 0;
    for (OrdAVLHashTIterator<int,int,8> kk(other); kk(); ) {
      if (kk.key() != a[yy]) {
	cerr << "not same" << endl;
      }
      yy++;
    }
  }

  //cerr << "Now, when deletes, insert back at end: should be same order at end" << endl;
  {
    // All in order
    OrdAVLHashT<int, int, 8> o;
    for (int ii=0; ii<LEN; ii++) {
      o[ii] = ii;
    }

    // Now, remove a random one and check
    for (int ii=0; ii<LEN; ii++) {
      //IterPrint(o);
      if (o.remove(a[ii])==false) { cerr << "Eh???" << endl; exit(1); }
      //IterPrint(o);

      Array<int> f(LEN-ii);
      int uu = 0;
      for (OrdAVLHashTIterator<int,int,8> ll(o); ll(); ) {
	f.append(ll.key());
	uu++;
      }
      // First, check that in order
      for (int jj=1; jj<int(LEN-ii-1); jj++) {
	if (f[jj-1]>=f[jj]) { 
	  cerr << ".. out of order.." << endl;
	  exit(1);
	}
      }
      // Insert back after deleted it
      //cerr << "Inserting back" << a[ii] << endl;
      o[a[ii]] = a[ii];
      //IterPrint(o);

      /*
      // Check and make sure same order as random
      int yy = 0;
      for (OrdAVLHashTIterator<int,int,8> kk(o); kk(); ) {
	cerr << "iter:" << yy << " " << kk.key() << " " << a[yy] << endl;
	if (yy<=ii) {
	  // compare in order keys
	  if (a[yy] != yy) {
	    cerr << "in order keys not same" << endl;
	  }
	} else {
	  // randomly placed keys
	  if (kk.key() != a[yy]) {
	    cerr << "end keys not same" << endl;
	  }
	}
	yy++;	
      } 
      */
    }
    
    // Check and make sure same order as random
    int yy = 0;
    for (OrdAVLHashTIterator<int,int,8> kk(o); kk(); ) {
      //cerr << "iter:" << yy << " " << kk.key() << " " << a[yy] << endl;
      if (kk.key() != a[yy]) {
	cerr << "not same" << endl;
	exit(1);
      }
      yy++;
    }
  }
  
}


int main ()
{
  OrdAVLHashT<string, string,8> o;
  IterPrint(o);

  OrdAVLHashT<string, string,8> o0;
  o0["aaa1"] = "bbb1";
  IterPrint(o0);

  OrdAVLHashT<string, string,8> o1;
  o1["a1"] = "b1";
  o1["a2"] = "b2";
  o1["a3"] = "b3";
  o1["a4"] = "b4";
  o1["a5"] = "b5";
  o1["a6"] = "b6";
  IterPrint(o1);

  o1.remove("a3");
  IterPrint(o1);
  o1["a3"] = "b3 again"; 
  IterPrint(o1); // should be at end of list

  o1["a2"] = "something else, but shouldn't change insertion order";
  IterPrint(o1); 

  for (int ii=1; ii<2000; ii++) {
    cout << ii << endl;
    CheckInsertionOrder(ii);
  };
}

