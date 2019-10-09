

// ///////////////////////////////////////////// Include Files

#include "occontainer_test.h"
#include "ochashtable.h"      

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

// ///////////////////////////////////////////// Main Program

int main ()
{
  ContainerTest<HashTable<int_u4>, HashTableIterator<int_u4>,
    HashTable<string>, HashTableIterator<string> > t;
  return t.tests();
}



