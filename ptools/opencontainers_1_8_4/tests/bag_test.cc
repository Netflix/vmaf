// Test program for bags and cupboards and keyedcupboards

#include "ocbag.h"
#include "ocval.h"
#include "octhread.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

// Data for each thread
struct Data {
  void* b;
  int worker_number;
  int bag;
  Data (void*bb, int n, int is_bag) : b(bb), worker_number(n), bag(is_bag) { }
}; 


#define ADD_RESULT result += item
//#define ADD_RESULT result.singleDigitAdd(item)
#define SUB_RESULT result -= item
// #define SUB_RESULT result.singleDigitSub(item)

int bound = 1;
void* Entry (Data *dp)
{
  int_un result = 0;
  // real_8 result = 0;

  
  // iCupboard
  if (dp->bag==1) {
    //cerr << "cupboard" << endl;
    iCupboard *bp = (iCupboard*)dp->b;
    int starting_drawer = dp->worker_number;
    int_u4 item = -1;
    while (bp->get(starting_drawer, item, starting_drawer)) {
      for (volatile int ii=0; ii<bound; ii++) {
	if (ii %2==0) {
	  ADD_RESULT;
	} else {
	  SUB_RESULT;
	}
      }
      ADD_RESULT;
      //result = result + item * 3;
    }
  }

  // iDrawer
  else if (dp->bag==3) {
    //cerr << "iDrawer" << endl;
    iDrawer *bp = (iDrawer*)dp->b;
    int_u4 item = -1;
    while (bp->get(item)) {
      for (volatile int ii=0; ii<bound; ii++) {
	if (ii %2==0) {
	  ADD_RESULT;
	} else {
	  SUB_RESULT;
	}
      }
      ADD_RESULT;
      //result = result + item * 3;
    }
  }


  else {
    cerr << "?????????" << endl;
  }
  int_un *rp = new int_un(result);
  return (void*)rp;
}


void* ThreadEntry (void* data)
{
  Data *dp = (Data*)data;
  //  alloca(dp->worker_number*500); // Avoid 64-kilobytes Aliasing Problem
  return Entry(dp);
}




void mainloop (string s, int_u4 n, int_u4 drawers)
{
  void* b = 0;
  int kind = 0;

  if (s=="cupboard") {
    // Create a Cupboard where each thread stays in its drawer
    iCupboard *bp = new iCupboard(n, drawers);
    kind = 1;
    b = bp;
    cout << "CUP";
  } else if (s=="drawer") {
    iDrawer *bp = new iDrawer(0, n);
    kind = 3;
    b = bp;
    cout << "DRAWER";
  }
  cout << endl;

  int threads = drawers;
  Array<OCThread*> thr(threads);
  for (int ii=0; ii<threads; ii++) {
    thr.append(new OCThread("thread"+Stringize(ii), false));
  }
  
  cout << "Start up a bunch" << endl;
  for (int ii=0; ii<threads; ii++) {
    thr[ii]->start(&ThreadEntry, new Data(b, ii, kind));
  }
  
  cout << "Wait for em to finish" << endl;
  int_un result = 0;
  for (int ii=0; ii<threads; ii++) {
    int_un *partial = (int_un*)thr[ii]->join();
    result += *partial;
  }

  cout << "Done:" << result << endl;
}


void usage(char **argv)
{
  cerr << "usage:" << argv[0] << " drawer|cupboard n workers [optional bound]" << endl;
  cerr << "    (note: the number of workers is how many threads to spawn: \n     it's also the same number of drawers)" << endl;
  exit(1);
}

int main (int argc, char **argv)
{
  if (argc==1) {
    // run from autotest
    Arr a = "['drawer', 'cupboard']";
    for (size_t ii=0; ii<a.length(); ii++) {
      string bag_type = a[ii];
      mainloop(bag_type, 1000000, 4);
    }
    exit(1);
  }

  if (argc!=4 && argc!=5) usage(argv);
  string s = argv[1];
  if (s!="cupboard" && s!="drawer") usage(argv);
  int_u4 n = atoi(argv[2]);
  int_u4 threads = atoi(argv[3]);
  bound = (argc==4) ? 1 : atoi(argv[4]);  // A bound of 1 means doing almost do no work between pulls: we are just testing how fast we can pull from the bag.  Any larger number will represent "more work" before the next pull
  cout << "BAGTYPE:" << s << " n=" << n << " threads=" << threads << " bound=" << bound << endl;

  mainloop(s, n, threads);
}
