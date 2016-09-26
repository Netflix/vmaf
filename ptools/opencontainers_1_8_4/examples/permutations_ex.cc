
// Demonstrate how Permutations work

#include "ocpermutations.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

int main ()
{
  // Generate permutations of 123
  Permutations p(3);

  // Generate all 4! permutations
  cout << "**Start from beginning***" << endl;
  while (p.next()) {
    const int* perm = p.currentPermutation();
    for (int ii=0; ii<3; ii++)
      cout << perm[ii];
    cout << endl;
  }

  // Seed to start at a particular point in the permutations: Useful
  // for restarting permutation testing.
  p.seed(2);
  cout << "**Start from middle***" << endl;
  while (p.next()) {
    const int* perm = p.currentPermutation();
    for (int ii=0; ii<3; ii++)
      cout << perm[ii];
    cout << endl;
  }
}


/* Output:

**Start from beginning***
123
213
321
132
312
231
**Start from middle***
132
312
231

*/
