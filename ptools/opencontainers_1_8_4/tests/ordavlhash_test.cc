

// ///////////////////////////////////////////// Include Files

#include "occontainer_test.h"
#include "ocordavlhash.h"      

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif 

// ///////////////////////////////////////////// Main Program

int main ()
{
  ContainerTest<OrdAVLHash<int_u4>, OrdAVLHashIterator<int_u4>,
    OrdAVLHash<string>, OrdAVLHashIterator<string> > t;
  return t.tests();
}



