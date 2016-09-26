
#include "ocfixedsizeallocator.h"
#include "ocpermute.h"
#include <iostream>
using namespace std;

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif


int main ()
{
  {
    FixedSizeAllocator<1, 2> fsa;
    cout << "Is Pristine?" << fsa.isPristine() << endl;
    fsa.print(cout);
    char* mem = fsa.allocate();
    fsa.print(cout);
    cout << "Is Pristine?" << fsa.isPristine() << endl;
    if (mem==NULL) {
      cerr << "ERROR: should be able to allocate 1 thing!" << endl;
    }
    fsa.deallocate(mem);
    fsa.print(cout);
    cout << "Is Pristine?" << fsa.isPristine() << endl;

    mem = fsa.allocate();
    //cout << int_ptr(mem) << " " << int_ptr(&fsa) << endl;
    fsa.print(cout);
    cout << "Is Pristine?" << fsa.isPristine() << endl;

    cout << "Is this piece of memory from here? .. should be yes ..."
	 <<  fsa.isMemoryFromHere(mem) << endl;

    mem = fsa.allocate();
    //cout << int_ptr(mem) << " " << int_ptr(&fsa) << endl;
    fsa.print(cout);
    cout << "Is Pristine?" << fsa.isPristine() << endl;

    cout << "Is NULL from here?  ... should be no ..."
	 << fsa.isMemoryFromHere(NULL) << endl;

  }

  FixedSizeAllocator<1, 12> fsa;
  int p[12];
  char *pointers[12];
  for (int ppp = 3; ppp<12; ppp++) {
    // Init perm
    for (int ii=0; ii<ppp; ii++) {
      p[ii] = ii;
    }

    // Do something with this perm
    do {
      //for (int ii=0; ii<ppp; ii++) cout << p[ii] << " ";
      //cout << endl;
      for (int ii=0; ii<ppp; ii++) {
	pointers[ii] = fsa.allocate();
	pointers[ii][0] = 0;
      }

      for (int ii=0; ii<ppp; ii++) {
	if (!fsa.isMemoryFromHere(pointers[p[ii]])) {
	  cerr << "ERROR!  memory nopt there!";
	  exit(1);
	}
	fsa.deallocate(pointers[p[ii]]);
      }
      if (!fsa.isPristine()) {
	cerr << "ERROR: Should be pristine" << endl;
      }
    } while (perm(p, ppp));

    fsa.print(cout);
  }

}
