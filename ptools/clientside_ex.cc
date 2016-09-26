
#include "shmboot.h"

// Example demonstrating how to use the ClientSide (along with SHMMain
// and ServerSide in the serverside.cc file) to have a point to point
// communication from process to process with shared memory.  Vals are
// allocated in shared memory and "sent" from one process to the
// other: because the "pointers" are copied and not a full table, this
// is a very fast way to do interprocess communication.

int main (int argc, char** argv)
{
  cout << "******** FOR REPRODUCIBLE RESULTS: May have to start this as\n";
  if (sizeof(void*)==4) {
    cout << "  % setarch i386 -L -R " << argv[0] << " ...\n";
  } else if (sizeof(void*)==8) {
    cout << "  % setarch x86_64 -L -R " << argv[0] << " ...\n";
  } else {
    cout << " ???  Unknown arch ??? use uname -i to find your arch " 
	 << " and use " << endl
	 << "  % setarch xxx -L -R " << endl
	 << "     where xxx is the result of 'uname -i'" << endl;
  }
  cout << "***********************************************" << endl;

  if (argc!=3) {
    cerr << "Usage:" << argv[0] << " shared_memory_name pipename" << endl
	 << "       -shared_memory_name is string name of shared memory pool\n"
	 << "       -pipename is name of pipe where serverside puts data"
	 << endl;
    exit(1);
  }
  string shared_mem_name = argv[1];
  string pipename = argv[2];
  bool debug = true;
  
  // (0) ... somebody else has created a SHMMain somewhere ...

  // (1) A ClientSide talks to a ServerSide
  ClientSide client(shared_mem_name, pipename, debug);
  client.start();
  CQ& pipe = client.pipe(); // Once started, can get communication point

  // (2) ... And start talking!
  for (int ii=0; ii<100; ii++) {
    
    // Use timeout: that way you won't get stuck in a condvar that
    // never returns.
    Val data;
    bool dequeued = false;
    while (!dequeued) {
      dequeued = pipe.dequeue(.2, data);
    }
    
    // Dequeued
    data.prettyPrint(cout);
  }
  
}
