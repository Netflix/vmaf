
// test to make sure the Randomizer does what we say

#include "ocrandomizer.h"
#include "ocarray.h"


#if defined(OC_FORCE_NAMESPACE) 
using namespace OC;
#endif

void RandomizerCheckInsertionOrder (int_u8 LEN)
{
  Randomizer r(LEN);
  Array<int_u8> a(LEN);
  for (int_u8 ii=0; ii<LEN; ii++) {
    a.append(r.next());
  }
  //cerr << a << endl;

  for (int_u8 ii=0; ii<LEN; ii++) {
    if (!a.contains(ii)) {
      cerr << "Doesn't have" << ii << endl;
      // cerr << r.groupSize_ << " " << r.n_ << " " << r.next_ << " " << r.element_ << endl;
      exit(1);
    }
  }
}

int main ()
{
  for (int i=0; i<65536; i++) {
    //cerr << ii << " " << FindPositionOfHighest1Bit(ii) << endl;
    int est = IntSqrtOverestimate(i);
    double true_root = sqrt(double(i));
    //cerr << "i=" << i << " true_root=" << true_root << " overest=" << est << " overest>=true_root?" << bool(est>=(true_root)) << endl;
    if (est<true_root) { 
      cerr << "sqrt overstimator not working" << endl;
      exit(1);
    }
  }

  for (int ii=1; ii<2000; ii++) {
    //if (ii%100==0) 
    {
      cout << "Checking Randomizer with 0.." << ii-1 << endl;
    }
    RandomizerCheckInsertionOrder(ii);
  }
  
}
