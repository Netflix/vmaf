
#include "occombinations.h"
#include <iostream>

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

void printCurrentCombination(int const* cur, int n, int k)
{
  cout << "(";
  for (int ii=0; ii<k; ii++) {
    cout << cur[ii];
    if (ii!=k-1) cout << " ";
  }
  cout << ")" << endl;
}

void showAll (int n, int k, bool prnt)
{
  if (prnt) cout << "n = " << n << " k = " << k << endl; 
  Combinations c(n,k);
  do {
    int const* cur = c.currentCombination();
    if (prnt) printCurrentCombination(cur, n, k);
  } while (c.nextCombination());
}

int main (int argc, char** argv)
{
  // 3 args, just compute what the user wants
  if (argc==3 || argc==4) {
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    bool prnt = (argc==3);
    showAll(n,k, prnt);
  } 

  // Otherwise, try to compute a bunch and make sure we get the right answer
  else {
    for (int n=-1; n<10; n++) {
      for (int k=-1; k<12; k++) {
	try {
	  showAll(n,k, true);
	} catch (const exception& e) {
	  cout << e.what() << endl;
	}
      }
    }
  }

  return 0;
}
