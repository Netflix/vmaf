
// A Test that exercises the memory pool

#include "ocproxy.h"
#include "ocval.h"
#include "ocstreamingpool.h"
#include "ocpermutations.h"

#if defined(OC_FORCE_NAMESPACE)
using namespace OC;
#endif

void testPool (const int length, const int order)
{
  const int bytes = 5000;
  char * mem = new char[bytes];
  StreamingPool* sp = StreamingPool::CreateStreamingPool(mem, bytes);
  
  // char* h[length]; // all zeroes!
  char** h = new char*[length];
  for (int ii=0; ii<length; ii++) {
    try {
      h[ii] = sp->allocate(8);
      char* c = h[ii];
      //  Fill mem
      for (int jj=0; jj<8; jj++) { 
	c[jj] = jj;
      }
      //cout << h[ii] << endl;
    } catch (const runtime_error& e) {
      cout << "Ran out of memory in that pool ... seeing if we clean up okay" << endl;
      h[ii] = 0;
    }
  }
  
  if (order==1) {
    for (int ii=0; ii<length; ii++) {
      if (h[ii])
	sp->deallocate(h[ii]);
    }
  } else if (order==-1) {
    for (int ii=length-1; ii>=0; ii--) {
      if (h[ii])
	sp->deallocate(h[ii]);
    }
  }
  string m = 
    "Simple "+Stringize(length)+" allocation then "+
    Stringize(length)+" deallocation at order:"+Stringize(order);
  if (sp->isPristine()) { 
    cout << m << ":GOOD" << endl;
  } else {
    cerr << m << ":FAILED" << endl;
    throw runtime_error(m);
  }
  sp->scheduleForDeletion();
  delete [] mem;
  delete [] h;
}

void permuteDeallocations (const int length, const int* cp, bool random_sizes=false)
{
  const int bytes = 5000;
  char * mem = new char[bytes];
  StreamingPool* sp = StreamingPool::CreateStreamingPool(mem, bytes);
  
  // char* h[length];
  char** h = new char*[length];
  for (int ii=0; ii<length; ii++) {
    int r = 8;
    if (random_sizes) r = (random()+1) % 100;
    try {
      h[ii] = sp->allocate(r);
      // Fill mem
      char* c = h[ii];
      for (int jj=0; jj<r; jj++) { 
	c[jj] = jj;
      }
      //cout << h[ii] << endl;
    } catch (const runtime_error& e) {
      cout << "Ran out of memory in that pool ... seeing if we clean up okay" << endl;
      h[ii] = 0;
    }
  }
  //if (length!=9 && sp->isFull()) { 
  //  cout << "Expected still space?" << sp->isFull() << endl; 
  //  throw runtime_error("still");
  //}
  //if (length==9 && !sp->isFull()) { 
  //  cout << "Expected full?" << sp->isFull() << endl; 
  //  throw runtime_error("full");
  // }
  
  for (int ii=0; ii<length; ii++) {
    char* current = h[cp[ii]-1];
    if (current) 
      sp->deallocate(current);
  }
  
  string m = 
    "PermuteDeallocations "+Stringize(length)+" allocation then "+
    Stringize(length)+" deallocations for permutation:";
  for (int ii=0; ii<length; ii++) {
    m+= Stringize(cp[ii])+ " ";
  }
  if (sp->isPristine()) { 
    // cout << m << ":GOOD" << endl;
  } else {
    cerr << m << ":FAILED" << endl;
    throw runtime_error(m);
  }
  sp->scheduleForDeletion();
  delete [] mem;
  delete [] h;
}

void permuteAD (const int length, const int* cp, bool random_sizes = false,
		bool use_small_allocator = false)
{
  if (length%2!=0) throw runtime_error("For this test, length has to be even");

  int bytes = 5000;
  if (use_small_allocator) {
    int some_junk = 1000;  // extra bytes for extra management
    bytes += 2*sizeof(FixedSizeAllocator<64,256>) + some_junk;
  }
  char * mem = new char[bytes];
  StreamingPool* sp = StreamingPool::CreateStreamingPool(mem, bytes, 4, use_small_allocator);
  
  //char* h[length];
  char** h = new char*[length];
  for (int ii=0; ii<length; ii++) h[ii] = 0;

  for (int ii=0; ii<length; ii++) {
    int r = 8;
    if (random_sizes) r = (random()+1)%200; 
    int idx = cp[ii]-1;
    int left,right;
    // This defines a range [left, right] of two elements.
    // The first one that appears of (0,1), (2,3), ...
    // becomes the allocation.  When the second one appears,
    // it becomes the deallocation.  Obviously length has to be
    // even or some pair will fail.
    if (idx%2==0) {
      left = idx;
      right = idx+1;
    } else {
      left = idx-1;
      right = idx;
    }
    // Nothing there yet, so must be allocation!
    if (h[left]==0) { 
    
      try {
	h[left] = sp->allocate(r);
	// Fill mem
	char* c = h[left];
	for (int jj=0; jj<r; jj++) { 
	  c[jj] = jj;
	}
	//cout << h[ii] << endl;
      } catch (const runtime_error& e) {
	cout << "Ran out of memory in that pool ... seeing if we clean up okay" << endl;
	h[left] = 0;
      }
    }
    // Something there, so deallocate
    else {
      sp->deallocate(h[left]);
      h[left]=0;
    }
  }
  
  string m = 
    "PermuteAllocationsDeallocations "+Stringize(length)+" allocation then "+
    Stringize(length)+" deallocations for permutation:";
  for (int ii=0; ii<length; ii++) {
    m+= Stringize(cp[ii])+ " ";
  }
  if (sp->isPristine()) { 
    // cout << m << ":GOOD" << endl;
  } else {
    cerr << m << ":FAILED" << endl;
    throw runtime_error(m);
  }
  sp->scheduleForDeletion();
  delete [] mem;
  delete [] h;
}


// create a pool of the given size, try to allocate a piece of memory
// out of it of that size
void testMax (int bytes) 
{
  char * mem = 0;
  try {
    mem = new char[bytes];
    StreamingPool* sp = StreamingPool::CreateStreamingPool(mem, bytes);
    
    int big = sp->biggestFree();
    char* h = sp->allocate(big);
    for (int ii=0; ii<big; ii++) {
      h[ii] = ii;
    }
    bool should_fail = false;
    try {
      char* fail = sp->allocate(1);
      if (fail+1 == fail) exit(1); // Dumb test
    } catch (const exception& e) {
      should_fail = true;
    }
    if (!should_fail) {
      throw runtime_error("Try to allocate after a full pool SHOULD have failed!");
    }
    
    if (!sp->isFull()) {
      throw runtime_error("Pool should be completely full:"+Stringize(bytes));
    }
    sp->deallocate(h);
    if (!sp->isPristine()) {
      throw runtime_error("Pool should be completely empty:"+Stringize(bytes));
    }
    sp->scheduleForDeletion();
  } catch (const exception& e) {
    static int saw_error = 1;
    if (saw_error) { 
      cout << "Can detect if memory too small" << endl;
      saw_error = 0;
    }
  }
  delete [] mem;
}


int main ()
{  
  //cerr << sizeof(Val) << " " << sizeof(Proxy) << " " << sizeof(Tab) << " " << sizeof(OTab) << endl;

  if (sizeof(int_ptr)==4 && MEMORY_BOUNDARY != INT_MIN) {
    cerr << "??? MEMORY_BOUNDARY should be same as INT_MIN for 32-bit" << endl;
    exit(1);
  } else if (sizeof(int_ptr)==8 && MEMORY_BOUNDARY != LONG_MIN) {
    cerr << "??? MEMORY_BOUNDARY should be same as LONG_MIN for 64-bit" << endl;
    exit(1);
  }
   
  for (int pairs=0; pairs<50; pairs++) {
    testPool(pairs, 1);
    testPool(pairs, -1);
  }
  
  for (int length=1; length<10; length++) {
    cout << "Permuting Deallocations Only:" << length << endl;
    Permutations pp(length);
    while (pp.next()) {
      const int* cp = pp.currentPermutation();
      //for (int ii=0; ii<length; ii++) cerr << cp[ii] << endl;
      permuteDeallocations(length, cp);
    }
  }


  for (int length=2; length<12; length+=2) {
    cout << "Permuting Allocations and Deallocations:" << length << endl;
    Permutations pp(length);
    while (pp.next()) {
      const int* cp = pp.currentPermutation();
      //for (int ii=0; ii<length; ii++) cerr << cp[ii] << endl;
      permuteAD(length, cp);
    }
  }

  
  cout << "Testing single big allocations" << endl;
  for (int ii=0; ii<1000; ii++) {
    try {
      testMax(ii);
    } catch (const exception& e) {
      cout << ii << ":" << e.what() << endl;
    }

  }

  cout << "Testing all sorts of random" << endl;  

  for (int length=1; length<10; length++) {
    cout << "Permuting Deallocations Only with random lengths:" << length << endl;
    Permutations pp(length);
    while (pp.next()) {
      const int* cp = pp.currentPermutation();
      //for (int ii=0; ii<length; ii++) cerr << cp[ii] << endl;
      permuteDeallocations(length, cp, true);
    }
  }

  for (int length=2; length<12; length+=2) {
    cout << "Permuting Allocations and Deallocations with random allocation sizes:" << length << endl;
    Permutations pp(length);
    while (pp.next()) {
      const int* cp = pp.currentPermutation();
      //for (int ii=0; ii<length; ii++) cerr << cp[ii] << endl;
      permuteAD(length, cp, true, false);
    }
  }

  for (int length=2; length<12; length+=2) {
    cout << "Permuting Allocations and Deallocations with random allocation sizes WITH SMALL ALLOCATOR:" << length << endl;
    Permutations pp(length);
    while (pp.next()) {
      const int* cp = pp.currentPermutation();
      //for (int ii=0; ii<length; ii++) cerr << cp[ii] << endl;
      permuteAD(length, cp, true, true);
    }
  }

  
}
